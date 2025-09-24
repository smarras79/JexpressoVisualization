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
# Set to True to perform averaging. Set to False to use the slicing method.
DO_AVERAGING = True

# Averaging configuration
AVERAGING_AXIS = 'Z'  # 'X', 'Y', or 'Z' - the axis along which to average
AVERAGING_LIMITS = {
    'X': [None, None],  # [Xmin, Xmax] - set to [None, None] for full X range
    'Z': [0.1, 10.0], # [Ymin, Ymax] - only average data between Y=0.0 and Y=5000.0
    'Y': [None, None]   # [Zmin, Zmax] - set to [None, None] for full Z range
}
AVERAGING_RESOLUTION = [150, 150, 150]  # Resolution in X, Y, Z for resampling

# Slicing configuration (only used if DO_AVERAGING is False)
DO_SLICE_AND_VISUALIZE = True
slice_axis = 'X' # 'X', 'Y', or 'Z'
slice_coordinate = 0.1 # Use None for auto-midpoint
# --- END OF NEW CONFIGURATION ---


# --- Dynamic output filename generation ---
try:
    base_name = os.path.basename(pvtu_file_path)
    match = re.search(r'iter_(\d+)', base_name)
    iteration_number = match.group(1) if match else "unknown"

    if DO_AVERAGING:
        plane_name = f'{AVERAGING_AXIS}_averaged'
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

# --- BRANCH 1: PERFORM AVERAGING WITH GEOMETRICAL LIMITS ---
if DO_AVERAGING:
    print(f"\n--- Starting {AVERAGING_AXIS}-Axis Averaging Workflow ---")
    
    # Validate averaging axis
    if AVERAGING_AXIS.upper() not in ['X', 'Y', 'Z']:
        raise ValueError(f"Invalid AVERAGING_AXIS: '{AVERAGING_AXIS}'. Must be 'X', 'Y', or 'Z'.")
    
    axis_index = {'X': 0, 'Y': 1, 'Z': 2}[AVERAGING_AXIS.upper()]
    
    # Required VTK modules for data manipulation
    from vtkmodules.numpy_interface import dataset_adapter as dsa
    from vtk import vtkImageData, vtkStructuredGrid, vtkPoints
    from vtk.util.numpy_support import numpy_to_vtk

    # 1. Get original bounds and calculate effective bounds with limits
    original_bounds = reader.GetDataInformation().GetBounds()
    print(f"INFO: Original data bounds: X=[{original_bounds[0]:.2f}, {original_bounds[1]:.2f}], "
          f"Y=[{original_bounds[2]:.2f}, {original_bounds[3]:.2f}], "
          f"Z=[{original_bounds[4]:.2f}, {original_bounds[5]:.2f}]")
    
    # Apply geometrical limits
    effective_bounds = list(original_bounds)
    axis_names = ['X', 'Y', 'Z']
    
    for i, axis_name in enumerate(axis_names):
        limits = AVERAGING_LIMITS.get(axis_name, [None, None])
        if limits[0] is not None:
            effective_bounds[i*2] = max(limits[0], original_bounds[i*2])
        if limits[1] is not None:
            effective_bounds[i*2+1] = min(limits[1], original_bounds[i*2+1])
    
    print(f"INFO: Effective bounds after applying limits: X=[{effective_bounds[0]:.2f}, {effective_bounds[1]:.2f}], "
          f"Y=[{effective_bounds[2]:.2f}, {effective_bounds[3]:.2f}], "
          f"Z=[{effective_bounds[4]:.2f}, {effective_bounds[5]:.2f}]")

    # 2. Create a clip filter to extract only the region within geometrical limits
    print("INFO: Applying geometrical limits using clip filters...")
    
    clipped_data = reader
    
    # Apply clipping for each axis if limits are specified
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        limits = AVERAGING_LIMITS.get(axis_name, [None, None])
        
        if limits[0] is not None or limits[1] is not None:
            # Create clip planes for this axis
            if limits[0] is not None:  # Min limit
                clip_min = Clip(Input=clipped_data)
                clip_min.ClipType = 'Plane'
                normal = [0, 0, 0]
                normal[i] = 1
                clip_min.ClipType.Normal = normal
                clip_min.ClipType.Origin = [0, 0, 0]
                clip_min.ClipType.Origin[i] = limits[0]
                clip_min.Invert = 0  # Keep data above the plane (replaced InsideOut)
                clipped_data = clip_min
                print(f"  Applied {axis_name}_min = {limits[0]}")
            
            if limits[1] is not None:  # Max limit
                clip_max = Clip(Input=clipped_data)
                clip_max.ClipType = 'Plane'
                normal = [0, 0, 0]
                normal[i] = -1  # Flip normal for max limit
                clip_max.ClipType.Normal = normal
                clip_max.ClipType.Origin = [0, 0, 0]
                clip_max.ClipType.Origin[i] = limits[1]
                clip_max.Invert = 0  # Keep data below the plane (replaced InsideOut)
                clipped_data = clip_max
                print(f"  Applied {axis_name}_max = {limits[1]}")

    # 3. Resample the clipped data to a uniform grid
    print(f"INFO: Resampling clipped data to uniform grid of {AVERAGING_RESOLUTION}...")
    resampled = ResampleToImage(Input=clipped_data)
    resampled.SamplingDimensions = AVERAGING_RESOLUTION
    
    # Set sampling bounds to the effective bounds
    resampled.SamplingBounds = effective_bounds
    resampled.UpdatePipeline()

    # 4. Fetch data and convert to NumPy array
    print("INFO: Fetching data and converting to NumPy array...")
    vtk_data = servermanager.Fetch(resampled)
    wrapped_data = dsa.WrapDataObject(vtk_data)
    point_data_3d_flat = wrapped_data.PointData[color_by_array_name]
    dims = resampled.SamplingDimensions
    point_data_3d = point_data_3d_flat.reshape(dims[2], dims[1], dims[0])  # (Nz, Ny, Nx)

    # 5. Calculate the average along the specified axis
    print(f"INFO: Calculating mean along the {AVERAGING_AXIS}-axis...")
    
    if AVERAGING_AXIS.upper() == 'X':
        averaged_data_2d = np.mean(point_data_3d, axis=2)  # Average along X -> (Nz, Ny)
        result_axes = ['Y', 'Z']
        grid_dims = [dims[1], dims[2]]  # (Ny, Nz)
        bounds_indices = [2, 3, 4, 5]  # Y and Z bounds
    elif AVERAGING_AXIS.upper() == 'Y':
        averaged_data_2d = np.mean(point_data_3d, axis=1)  # Average along Y -> (Nz, Nx)
        result_axes = ['X', 'Z']
        grid_dims = [dims[0], dims[2]]  # (Nx, Nz)
        bounds_indices = [0, 1, 4, 5]  # X and Z bounds
    else:  # Z
        averaged_data_2d = np.mean(point_data_3d, axis=0)  # Average along Z -> (Ny, Nx)
        result_axes = ['X', 'Y']
        grid_dims = [dims[0], dims[1]]  # (Nx, Ny)
        bounds_indices = [0, 1, 2, 3]  # X and Y bounds
    
    averaged_array_name = f"{color_by_array_name}_{AVERAGING_AXIS}_avg"
    print(f"INFO: Created {result_axes[0]}-{result_axes[1]} plane with shape {averaged_data_2d.shape}")

    # 6. Create a 2D plane using vtkStructuredGrid
    print("INFO: Creating 2D structured grid for averaged data...")
    
    # Debug: Print array shape
    print(f"DEBUG: Averaged data shape: {averaged_data_2d.shape}")
    print(f"DEBUG: Grid dimensions: {grid_dims}")
    
    n_axis1, n_axis2 = grid_dims
    
    # Create structured grid
    structured_grid = vtkStructuredGrid()
    structured_grid.SetDimensions(n_axis1, n_axis2, 1)  # 2D plane
    
    # Create points for the grid
    points = vtkPoints()
    
    # Create coordinate arrays for the two remaining axes
    axis1_coords = np.linspace(effective_bounds[bounds_indices[0]], effective_bounds[bounds_indices[1]], n_axis1)
    axis2_coords = np.linspace(effective_bounds[bounds_indices[2]], effective_bounds[bounds_indices[3]], n_axis2)
    
    # Calculate the average coordinate for the averaged-out axis
    avg_coord = (effective_bounds[axis_index*2] + effective_bounds[axis_index*2+1]) / 2.0
    
    print(f"INFO: {result_axes[0]} range: [{effective_bounds[bounds_indices[0]]:.3f}, {effective_bounds[bounds_indices[1]]:.3f}]")
    print(f"INFO: {result_axes[1]} range: [{effective_bounds[bounds_indices[2]]:.3f}, {effective_bounds[bounds_indices[3]]:.3f}]")
    print(f"INFO: Averaged {AVERAGING_AXIS} coordinate: {avg_coord:.3f}")
    
    # Add points in the correct order for structured grid
    for j in range(n_axis2):  # Second axis (outer loop)
        for i in range(n_axis1):  # First axis (inner loop)
            if AVERAGING_AXIS.upper() == 'X':
                # Y-Z plane, X averaged out
                points.InsertNextPoint(avg_coord, axis1_coords[i], axis2_coords[j])
            elif AVERAGING_AXIS.upper() == 'Y':
                # X-Z plane, Y averaged out  
                points.InsertNextPoint(axis1_coords[i], avg_coord, axis2_coords[j])
            else:  # Z
                # X-Y plane, Z averaged out
                points.InsertNextPoint(axis1_coords[i], axis2_coords[j], avg_coord)
    
    structured_grid.SetPoints(points)
    
    # Add the scalar data - flatten in row-major order to match point ordering
    vtk_array = numpy_to_vtk(averaged_data_2d.flatten('C'), deep=True)
    vtk_array.SetName(averaged_array_name)
    structured_grid.GetPointData().SetScalars(vtk_array)
    
    print(f"DEBUG: Created structured grid with {structured_grid.GetNumberOfPoints()} points and {structured_grid.GetNumberOfCells()} cells")

    # 7. Push the new data object into the ParaView pipeline
    producer = TrivialProducer(registrationName=f'{AVERAGING_AXIS}_Averaged_Data')
    producer.GetClientSideObject().SetOutput(structured_grid)
    producer.UpdatePipeline()
    
    # Debug: Check what we actually created
    producer_info = producer.GetDataInformation()
    print(f"DEBUG: Producer bounds: {producer_info.GetBounds()}")
    print(f"DEBUG: Producer number of points: {producer_info.GetNumberOfPoints()}")
    print(f"DEBUG: Producer number of cells: {producer_info.GetNumberOfCells()}")

    # 8. Visualize the resulting 2D averaged data
    print("INFO: Visualizing the averaged data...")
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = [1200, 800]
    
    # Let ParaView automatically choose the best representation
    display = Show(producer, view)
    
    # Set the correct representation type using the Representation property
    display.Representation = 'Surface'

    ColorBy(display, ('POINTS', averaged_array_name))
    display.RescaleTransferFunctionToDataRange(True, False)
    display.SetScalarBarVisibility(view, True)
    lut = GetColorTransferFunction(averaged_array_name)
    lut.ApplyPreset('Blues', True)

    # Camera setup based on which plane we're viewing
    print(f"INFO: Setting camera for {result_axes[0]}-{result_axes[1]} plane view.")
    
    if AVERAGING_AXIS.upper() == 'X':
        # Viewing Y-Z plane
        view.CameraPosition = [1, 0, 0]
        view.CameraViewUp = [0, 0, 1]
    elif AVERAGING_AXIS.upper() == 'Y':
        # Viewing X-Z plane  
        view.CameraPosition = [0, 1, 0]
        view.CameraViewUp = [0, 0, 1]
    else:  # Z
        # Viewing X-Y plane
        view.CameraPosition = [0, 0, 1]
        view.CameraViewUp = [0, 1, 0]
    
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraParallelProjection = 1
    view.ResetCamera()
    view.StillRender()
    
    source_to_visualize = producer

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
if DO_AVERAGING or DO_SLICE_AND_VISUALIZE:
    print(f"\nINFO: Saving screenshot to '{output_image_filename}'...")
    SaveScreenshot(output_image_filename, GetActiveView(), ImageResolution=[1200, 800])
    print("  > Screenshot saved successfully.")

print("\n--- Script finished ---")
