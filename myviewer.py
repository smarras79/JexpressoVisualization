#------------------------------------------------------------------------------------
# INSTRUCTIONS TO RUN THIS SCRIPT:
#
# /Applications/ParaView-5.11.2.app/Contents/bin/pvpython myviewer.py
#
#------------------------------------------------------------------------------------

# Import necessary libraries
from paraview.simple import *
import os
import re

# --- Script Configuration ---
pvtu_file_path = './iter_406.pvtu'
point_array_name = 'u'
color_by_array_name = 'u'
output_point_data_csv = 'extracted_u_data.csv'
DO_SLICE_AND_VISUALIZE = True

# --- NEW: Slicing and Orientation Configuration ---
# Choose the axis normal for the slice: 'X', 'Y', or 'Z'.
# 'X' creates a YZ plane, 'Y' an XZ plane, 'Z' an XY plane.
slice_axis = 'X'

# Specify the coordinate on the chosen axis for the slice.
# If set to None, the script will automatically slice through the middle of the domain.
slice_coordinate = 0.1
# --- END OF NEW CONFIGURATION ---


# --- Dynamic output filename generation ---
try:
    # Determine plane name based on slice axis for filename
    axis_map = {'X': 'YZ_plane', 'Y': 'XZ_plane', 'Z': 'XY_plane'}
    plane_name = axis_map.get(slice_axis.upper(), 'unknown_plane')

    base_name = os.path.basename(pvtu_file_path)
    match = re.search(r'iter_(\d+)', base_name)
    iteration_number = match.group(1) if match else "unknown"

    output_image_filename = f"PV_{color_by_array_name}_iter_{iteration_number}_{plane_name}.png"
    print(f"INFO: Set output filename to '{output_image_filename}'")
except Exception:
    iteration_number = "unknown"
    output_image_filename = 'slice_visualization.png'
    print(f"WARNING: Could not parse iteration number. Using default filename.")


# --- Main Script Logic ---
paraview.simple._DisableFirstRenderCameraReset()

print(f"INFO: Loading data from '{pvtu_file_path}'...")
reader = XMLPartitionedUnstructuredGridReader(FileName=[pvtu_file_path])
reader.UpdatePipeline()


# --- PART 2: SLICING AND VISUALIZATION ---
if DO_SLICE_AND_VISUALIZE:
    print("\n--- Starting Part 2: Slicing and Visualization ---")

    bounds = reader.GetDataInformation().GetBounds()
    
    # --- MODIFIED: Auto-detection logic based on slice_axis ---
    current_slice_coordinate = slice_coordinate
    if current_slice_coordinate is None:
        if slice_axis.upper() == 'X':
            # Midpoint of the X-axis (bounds[0] and bounds[1])
            current_slice_coordinate = (bounds[0] + bounds[1]) / 2.0
        elif slice_axis.upper() == 'Y':
            # Midpoint of the Y-axis (bounds[2] and bounds[3])
            current_slice_coordinate = (bounds[2] + bounds[3]) / 2.0
        else: # 'Z'
            # Midpoint of the Z-axis (bounds[4] and bounds[5])
            current_slice_coordinate = (bounds[4] + bounds[5]) / 2.0
        print(f"INFO: Auto-detecting slice position at {slice_axis} = {current_slice_coordinate:.3f}")

    # --- MODIFIED: Set slice and camera parameters based on user choice ---
    slice_params = {
        'X': {'origin': [current_slice_coordinate, 0, 0], 'normal': [1, 0, 0], 'cam_pos': [1, 0, 0], 'cam_up': [0, 0, 1]},
        'Y': {'origin': [0, current_slice_coordinate, 0], 'normal': [0, 1, 0], 'cam_pos': [0, 1, 0], 'cam_up': [0, 0, 1]},
        'Z': {'origin': [0, 0, current_slice_coordinate], 'normal': [0, 0, 1], 'cam_pos': [0, 0, 1], 'cam_up': [0, 1, 0]}
    }
    
    selected_axis = slice_axis.upper()
    if selected_axis not in slice_params:
        raise ValueError(f"Invalid slice_axis: '{slice_axis}'. Must be 'X', 'Y', or 'Z'.")
    
    params = slice_params[selected_axis]

    # Create and configure the slice
    slice1 = Slice(Input=reader)
    slice1.SliceType = 'Plane'
    slice1.SliceType.Origin = params['origin']
    slice1.SliceType.Normal = params['normal']
    print(f"INFO: Created a slice in the {plane_name} at {selected_axis} = {current_slice_coordinate:.3f}")

    # Create view and display the slice
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = [1200, 800]
    slice_display = Show(slice1, view, 'GeometryRepresentation')

    # Configure color mapping
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

    # --- MODIFIED: Set camera orientation for the chosen slice view ---
    print(f"INFO: Setting camera for {plane_name} view.")
    view.CameraPosition = params['cam_pos']
    view.CameraViewUp = params['cam_up']
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraParallelProjection = 1

    # Frame the slice in the view and render
    view.ResetCamera()
    view.StillRender()

    # Save the final image
    print(f"INFO: Saving screenshot to '{output_image_filename}'...")
    SaveScreenshot(output_image_filename, view, ImageResolution=[1200, 800])
    print("  > Screenshot saved successfully.")

print("\n--- Script finished ---")
