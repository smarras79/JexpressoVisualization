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
import numpy as np

# --- Script Configuration ---
pvtu_file_path = './iter_406.pvtu'
point_array_name = 'u'
color_by_array_name = 'u'
output_point_data_csv = 'extracted_u_data.csv'

# --- NEW: Averaging and Slicing Control ---
# Set to True to perform Y-averaging. Set to False to use the slicing method.
DO_Y_AVERAGING = False

# If True, this sets the resolution of the grid used for averaging.
# Higher values are more accurate but use more memory and time.
AVERAGING_RESOLUTION = [150, 150, 150] # Resolution in X, Y, Z

# Slicing configuration (only used if DO_Y_AVERAGING is False)
DO_SLICE_AND_VISUALIZE = True
slice_axis = 'X' # 'X', 'Y', or 'Z'
slice_coordinate = 0.1 # Use None for auto-midpoint
# --- END OF NEW CONFIGURATION ---


# --- Dynamic output filename generation ---
try:
    base_name = os.path.basename(pvtu_file_path)
    match = re.search(r'iter_(\d+)', base_name)
    iteration_number = match.group(1) if match else "unknown"

    if DO_Y_AVERAGING:
        plane_name = 'XZ_plane_Y_averaged'
    else:
        axis_map = {'X': 'YZ_plane', 'Y': 'XZ_plane', 'Z': 'XY_plane'}
        plane_name = axis_map.get(slice_axis.upper(), 'unknown_plane')

    output_image_filename = f"PV_{color_by_array_name}_iter_{iteration_number}_{plane_name}.png"
    print(f"INFO: Set output filename to '{output_image_filename}'")
except Exception:
    output_image_filename = 'visualization.png'
    print(f"WARNING: Could not parse filename. Using default.")


# --- Main Script Logic ---
paraview.simple._DisableFirstRenderCameraReset()

print(f"INFO: Loading data from '{pvtu_file_path}'...")
reader = XMLPartitionedUnstructuredGridReader(FileName=[pvtu_file_path])
reader.UpdatePipeline()


# --- PART 2: DATA PROCESSING AND VISUALIZATION ---

# --- BRANCH 1: PERFORM Y-AXIS AVERAGING ---
if DO_Y_AVERAGING:
    print("\n--- Starting Y-Axis Averaging Workflow ---")
    
    # Required VTK modules for data manipulation
    from vtkmodules.numpy_interface import dataset_adapter as dsa
    from vtk import vtkImageData
    from vtk.util.numpy_support import numpy_to_vtk

    # 1. Resample data to a uniform grid
    print(f"INFO: Resampling data to a uniform grid of {AVERAGING_RESOLUTION}...")
    resampled = ResampleToImage(Input=reader)
    resampled.SamplingDimensions = AVERAGING_RESOLUTION
    resampled.UpdatePipeline()

    # 2. Fetch data from ParaView server and convert to NumPy array
    print("INFO: Fetching data and converting to NumPy array...")
    vtk_data = servermanager.Fetch(resampled)
    wrapped_data = dsa.WrapDataObject(vtk_data)
    point_data_3d_flat = wrapped_data.PointData[color_by_array_name]
    dims = resampled.SamplingDimensions
    point_data_3d = point_data_3d_flat.reshape(dims[2], dims[1], dims[0]) # Reshape to (Nz, Ny, Nx)

    # 3. Calculate the mean along the Y-axis
    print("INFO: Calculating mean along the Y-axis...")
    averaged_data_2d = np.mean(point_data_3d, axis=1) # Average along Ny -> shape becomes (Nz, Nx)
    averaged_array_name = f"{color_by_array_name}_Y_avg"

    # 4. Create a 2D plane using vtkStructuredGrid - More robust approach
    print("INFO: Creating 2D structured grid for averaged data...")
    from vtk import vtkStructuredGrid, vtkPoints
    
    # Debug: Print array shape
    print(f"DEBUG: Averaged data shape: {averaged_data_2d.shape}")
    print(f"DEBUG: Original dimensions: {dims}")
    
    # Ensure we have the correct shape: (Nz, Nx) for Z-X plane
    nz, nx = averaged_data_2d.shape
    print(f"DEBUG: Creating grid with nx={nx}, nz={nz}")
    
    # Create structured grid
    structured_grid = vtkStructuredGrid()
    structured_grid.SetDimensions(nx, nz, 1)  # (Nx, Nz, 1) for a 2D plane
    
    # Create points for the grid
    points = vtkPoints()
    bounds = reader.GetDataInformation().GetBounds()
    
    # Create coordinate arrays
    x_coords = np.linspace(bounds[0], bounds[1], nx)
    z_coords = np.linspace(bounds[4], bounds[5], nz)
    y_coord = (bounds[2] + bounds[3]) / 2.0  # Average Y position
    
    print(f"DEBUG: X range: [{bounds[0]:.3f}, {bounds[1]:.3f}], Z range: [{bounds[4]:.3f}, {bounds[5]:.3f}]")
    
    # Add points in the correct order for structured grid (i varies fastest)
    for k in range(nz):  # Z direction (outer loop)
        for i in range(nx):  # X direction (inner loop)
            points.InsertNextPoint(x_coords[i], y_coord, z_coords[k])
    
    structured_grid.SetPoints(points)
    
    # Add the scalar data - flatten in row-major order to match point ordering
    vtk_array = numpy_to_vtk(averaged_data_2d.flatten('C'), deep=True)
    vtk_array.SetName(averaged_array_name)
    structured_grid.GetPointData().SetScalars(vtk_array)
    
    print(f"DEBUG: Created structured grid with {structured_grid.GetNumberOfPoints()} points and {structured_grid.GetNumberOfCells()} cells")

    # 5. Push the new data object into the ParaView pipeline
    producer = TrivialProducer(registrationName='Y_Averaged_Data')
    producer.GetClientSideObject().SetOutput(structured_grid)
    producer.UpdatePipeline()
    
    # Debug: Check what we actually created
    producer_info = producer.GetDataInformation()
    print(f"DEBUG: Producer bounds: {producer_info.GetBounds()}")
    print(f"DEBUG: Producer number of points: {producer_info.GetNumberOfPoints()}")
    print(f"DEBUG: Producer number of cells: {producer_info.GetNumberOfCells()}")

    # 6. Visualize the resulting 2D averaged data
    print("INFO: Visualizing the averaged data...")
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = [1200, 800]
    
    # Let ParaView automatically choose the best representation
    display = Show(producer, view)
    
    # Set the correct representation type using the Representation property
    display.Representation = 'Surface'
    
    # Alternative representations to try if Surface doesn't work:
    # display.Representation = 'Surface With Edges'
    # display.Representation = 'Wireframe'

    ColorBy(display, ('POINTS', averaged_array_name))
    display.RescaleTransferFunctionToDataRange(True, False)
    display.SetScalarBarVisibility(view, True)
    lut = GetColorTransferFunction(averaged_array_name)
    lut.ApplyPreset('Blues', True)

    # Camera setup for X-Z plane
    print("INFO: Setting camera for X-Z plane view.")
    view.CameraPosition = [0, 1, 0]
    view.CameraViewUp = [0, 0, 1]
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraParallelProjection = 1
    view.ResetCamera()
    view.StillRender()
    
    source_to_visualize = producer # For consistency if needed later

# --- BRANCH 2: PERFORM SLICING ---
elif DO_SLICE_AND_VISUALIZE:
    print("\n--- Starting Slicing Workflow ---")

    bounds = reader.GetDataInformation().GetBounds()
    
    current_slice_coordinate = slice_coordinate
    if current_slice_coordinate is None:
        if slice_axis.upper() == 'X':
            current_slice_coordinate = (bounds[0] + bounds[1]) / 2.0
        elif slice_axis.upper() == 'Y':
            current_slice_coordinate = (bounds[2] + bounds[3]) / 2.0
        else: # 'Z'
            current_slice_coordinate = (bounds[4] + bounds[5]) / 2.0
        print(f"INFO: Auto-detecting slice position at {slice_axis} = {current_slice_coordinate:.3f}")

    slice_params = {
        'X': {'origin': [current_slice_coordinate, 0, 0], 'normal': [1, 0, 0], 'cam_pos': [1, 0, 0], 'cam_up': [0, 0, 1]},
        'Y': {'origin': [0, current_slice_coordinate, 0], 'normal': [0, 1, 0], 'cam_pos': [0, 1, 0], 'cam_up': [0, 0, 1]},
        'Z': {'origin': [0, 0, current_slice_coordinate], 'normal': [0, 0, 1], 'cam_pos': [0, 0, 1], 'cam_up': [0, 1, 0]}
    }
    
    selected_axis = slice_axis.upper()
    if selected_axis not in slice_params:
        raise ValueError(f"Invalid slice_axis: '{slice_axis}'. Must be 'X', 'Y', or 'Z'.")
    params = slice_params[selected_axis]

    slice1 = Slice(Input=reader)
    slice1.SliceType = 'Plane'
    slice1.SliceType.Origin = params['origin']
    slice1.SliceType.Normal = params['normal']
    print(f"INFO: Created a slice in the {plane_name} at {selected_axis} = {current_slice_coordinate:.3f}")

    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = [1200, 800]
    slice_display = Show(slice1, view, 'GeometryRepresentation')

    slice1.UpdatePipeline()
    available_arrays = [slice1.GetPointDataInformation().GetArray(i).Name for i in range(slice1.GetPointDataInformation().GetNumberOfArrays())]

    if color_by_array_name in available_arrays:
        print(f"INFO: Coloring slice by point data array: '{color_by_array_name}'")
        ColorBy(slice_display, ('POINTS', color_by_array_name))
        slice_display.RescaleTransferFunctionToDataRange(True, False)
        slice_display.SetScalarBarVisibility(view, True)
        lut = GetColorTransferFunction(color_by_array_name)
        lut.ApplyPreset('Blues', True)
    else:
        print(f"WARNING: Array '{color_by_array_name}' not found for coloring.")

    print(f"INFO: Setting camera for {plane_name} view.")
    view.CameraPosition = params['cam_pos']
    view.CameraViewUp = params['cam_up']
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraParallelProjection = 1
    view.ResetCamera()
    view.StillRender()
    
    source_to_visualize = slice1 # For consistency if needed later

# --- Final Step: Save the output ---
if DO_Y_AVERAGING or DO_SLICE_AND_VISUALIZE:
    print(f"\nINFO: Saving screenshot to '{output_image_filename}'...")
    SaveScreenshot(output_image_filename, GetActiveView(), ImageResolution=[1200, 800])
    print("  > Screenshot saved successfully.")

print("\n--- Script finished ---")
