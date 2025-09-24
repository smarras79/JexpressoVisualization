#------------------------------------------------------------------------------------
# INSTRUCTIONS TO RUN THIS SCRIPT:
#
# /Applications/ParaView-5.11.2.app/Contents/bin/pvpython myviewer.py
#
#------------------------------------------------------------------------------------

# Import necessary libraries
from paraview.simple import *
from vtkmodules.numpy_interface import dataset_adapter as dsa
import numpy as np
import os
import re

# --- Script Configuration ---
pvtu_file_path = './iter_406.pvtu'

# --- Part 1: Data Extraction Settings ---
DO_DATA_EXTRACTION = False # Set to True to save the CSV file
point_array_name_to_extract = 'u' 
output_point_data_csv = 'extracted_u_data.csv'

# --- Part 2: Visualization Settings ---
DO_SLICE_AND_VISUALIZE = True

# --- NEW: CHOOSE YOUR SLICE AXIS HERE ---
# Set this to 'X', 'Y', or 'Z' to define the slice plane.
#slice_axis = 'X'
#slice_coordinate = 5120

slice_axis = 'Y'
slice_coordinate = 5120

#slice_axis = 'Z'
#slice_coordinate = 0.1
# --- END OF NEW FEATURE ---

color_by_array_name = 'u' 

# --- Dynamic output filename generation ---
try:
    base_name = os.path.basename(pvtu_file_path)
    match = re.search(r'iter_(\d+)', base_name)
    iteration_number = match.group(1) if match else "unknown"
    output_image_filename = f"PV_{color_by_array_name}_iter_{iteration_number}_slice_{slice_axis}.png"
    print(f"INFO: Set output filename to '{output_image_filename}'")
except Exception:
    output_image_filename = 'slice_visualization.png'
    print(f"WARNING: Could not parse iteration number. Using default filename.")

# --- Main Script Logic ---
paraview.simple._DisableFirstRenderCameraReset()

print(f"INFO: Loading data from '{pvtu_file_path}'...")
reader = XMLPartitionedUnstructuredGridReader(FileName=[pvtu_file_path])
reader.UpdatePipeline()

# Fetch data from ParaView server so we can access its arrays in Python
data_object = servermanager.Fetch(reader)
wrapped_data = dsa.WrapDataObject(data_object)

# --- PART 1: DATA EXTRACTION (Optional) ---
# ... (this part is unchanged) ...

# --- PART 2: SLICING AND VISUALIZATION ---
if DO_SLICE_AND_VISUALIZE:
    print("\n--- Starting Part 2: Slicing and Visualization ---")

    bounds = reader.GetDataInformation().GetBounds()
    slice_axis = slice_axis.upper() # Make the choice case-insensitive

    # --- CHANGED: Dynamic Slice Configuration ---
    if slice_axis == 'X':
        slice_normal = [1.0, 0.0, 0.0]
        if slice_coordinate is None:
            slice_coordinate = (bounds[0] + bounds[1]) / 2.0 # Midpoint of X-axis
        slice_origin = [slice_coordinate, 0.0, 0.0]
    elif slice_axis == 'Y':
        slice_normal = [0.0, 1.0, 0.0]
        if slice_coordinate is None:
            slice_coordinate = (bounds[2] + bounds[3]) / 2.0 # Midpoint of Y-axis
        slice_origin = [0.0, slice_coordinate, 0.0]
    elif slice_axis == 'Z':
        slice_normal = [0.0, 0.0, 1.0]
        if slice_coordinate is None:
            slice_coordinate = (bounds[4] + bounds[5]) / 2.0 # Midpoint of Z-axis
        slice_origin = [0.0, 0.0, slice_coordinate]
    else:
        print(f"ERROR: Invalid slice_axis '{slice_axis}'. Please choose 'X', 'Y', or 'Z'.")
        exit()
    
    print(f"INFO: Auto-detecting slice position at {slice_axis} = {slice_coordinate:.3f}")
    
    # Create and configure the slice
    slice1 = Slice(Input=reader)
    slice1.SliceType = 'Plane'
    slice1.SliceType.Origin = slice_origin
    slice1.SliceType.Normal = slice_normal
    print(f"INFO: Created a slice perpendicular to the {slice_axis}-axis.")
    # --- END OF CHANGE ---
    
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = [1200, 800]
    slice_display = Show(slice1, view, 'GeometryRepresentation')

    slice1.UpdatePipeline()
    slice_info = slice1.GetPointDataInformation()
    available_arrays = [slice_info.GetArray(i).Name for i in range(slice_info.GetNumberOfArrays())]
    
    if color_by_array_name in available_arrays:
        print(f"INFO: Coloring slice by point data array: '{color_by_array_name}'")
        ColorBy(slice_display, ('POINTS', color_by_array_name))
        slice_display.RescaleTransferFunctionToDataRange(True, False)
        slice_display.SetScalarBarVisibility(view, True)
        
        lut = GetColorTransferFunction(color_by_array_name)
        lut.ApplyPreset('Blues', True)
        print(f"INFO: Applied 'Blues' color map.")
    else:
        print(f"WARNING: Array '{color_by_array_name}' not found for coloring.")

    view.ResetCamera()
    view.StillRender() 

    print(f"INFO: Saving screenshot to '{output_image_filename}'...")
    SaveScreenshot(output_image_filename, view, ImageResolution=[1200, 800])
    print("  > Screenshot saved successfully.")

print("\n--- Script finished ---")
