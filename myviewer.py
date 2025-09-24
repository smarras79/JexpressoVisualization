#!/usr/bin/env python3
"""
ParaView Data Averaging and Slicing Tool

INSTRUCTIONS TO RUN:
/Applications/ParaView-5.11.2.app/Contents/bin/pvpython this_script.py

This script provides two main functionalities:
1. Averaging: Calculate spatial averages along X, Y, or Z axis with optional geometric limits
2. Slicing: Create 2D slices at specific coordinates

Author: Enhanced ParaView Analysis Script
"""

#==============================================================================
# USER CONFIGURATION SECTION
# Modify these parameters according to your needs
#==============================================================================

# --- Input Data Configuration ---
INPUT_FILE = './iter_406.pvtu'
DATA_ARRAY = 'w'  # Use 'VELOMAG' for automatic velocity magnitude calculation from u,v,w components
#
# ρ
# p
# θ
# u,v,w
#

# --- Special Data Arrays ---
# If DATA_ARRAY is set to one of these special values, the script will automatically
# calculate the derived quantity from component fields:
SPECIAL_ARRAYS = {
    'VELOMAG': ['u', 'v', 'w']  # Velocity magnitude from u,v,w components
    # Add more special arrays here as needed, e.g.:
    # 'VORTMAG': ['omega_x', 'omega_y', 'omega_z']  # Vorticity magnitude
}

# --- Analysis Mode Selection ---
USE_AVERAGING = False # Set to False to use slicing instead

# --- Averaging Configuration (only used when USE_AVERAGING = True) ---
AVERAGING_CONFIG = {
    'axis': 'Y',  # Choose 'X', 'Y', or 'Z' - the axis to average along
    'resolution': [150, 150, 150],  # [X, Y, Z] resolution for resampling
    'geometric_limits': {
        'X': [None, None],  # [min, max] or [None, None] for full range
        'Y': [0.1, 10.0],   # [min, max] or [None, None] for full range  
        'Z': [None, None]   # [min, max] or [None, None] for full range
    }
}

# --- Slicing Configuration (only used when USE_AVERAGING = False) ---
SLICING_CONFIG = {
    'axis': 'Z',        # Choose 'X', 'Y', or 'Z' - the axis normal to the slice
    'coordinate': 100   # Slice position, or None for auto-center
}

# --- Visualization Configuration ---
VISUALIZATION_CONFIG = {
    'image_size': [1200, 800],
    'color_map': 'Blues',  # ParaView color map name
    'auto_filename': True  # Generate filename automatically from input
}

#==============================================================================
# CORE IMPLEMENTATION - DO NOT MODIFY BELOW THIS LINE
#==============================================================================

import os
import re
import numpy as np
from paraview.simple import *

class ParaViewAnalyzer:
    """Core analysis engine for ParaView data processing."""
    
    def __init__(self, input_file, data_array):
        self.input_file = input_file
        self.data_array = data_array
        self.reader = None
        self.output_filename = None
        
    def _setup_paraview(self):
        """Initialize ParaView environment."""
        paraview.simple._DisableFirstRenderCameraReset()
        
    def _load_data(self):
        """Load the input data file and handle special array calculations."""
        print(f"INFO: Loading data from '{self.input_file}'...")
        self.reader = XMLPartitionedUnstructuredGridReader(FileName=[self.input_file])
        self.reader.UpdatePipeline()
        
        # Check if we need to calculate a derived quantity
        if self.data_array in SPECIAL_ARRAYS:
            self._calculate_derived_array()
    
    def _calculate_derived_array(self):
        """Calculate derived arrays like velocity magnitude from component fields."""
        if self.data_array == 'VELOMAG':
            self._calculate_velocity_magnitude()
        # Add other derived array calculations here as needed
    
    def _calculate_velocity_magnitude(self):
        """Calculate velocity magnitude from u, v, w components using ParaView Calculator."""
        print("INFO: Calculating velocity magnitude from u, v, w components...")
        
        # Verify that u, v, w arrays exist
        point_arrays = [self.reader.GetPointDataInformation().GetArray(i).Name 
                       for i in range(self.reader.GetPointDataInformation().GetNumberOfArrays())]
        
        required_components = SPECIAL_ARRAYS['VELOMAG']
        missing_components = [comp for comp in required_components if comp not in point_arrays]
        
        if missing_components:
            available = ", ".join(point_arrays)
            missing = ", ".join(missing_components)
            raise ValueError(f"Missing velocity components: {missing}. "
                           f"Available arrays: {available}")
        
        print(f"INFO: Found velocity components: {', '.join(required_components)}")
        
        # Create calculator to compute velocity magnitude
        calculator = Calculator(Input=self.reader)
        calculator.ResultArrayName = 'VELOMAG'
        calculator.Function = 'sqrt(u*u + v*v + w*w)'
        calculator.UpdatePipeline()
        
        # Replace reader with calculator output
        self.reader = calculator
        
        print("INFO: Velocity magnitude calculation completed successfully")
        
    def _generate_filename(self, config):
        """Generate output filename based on configuration."""
        if not config.get('auto_filename', True):
            return 'visualization.png'
            
        try:
            base_name = os.path.basename(self.input_file)
            match = re.search(r'iter_(\d+)', base_name)
            iteration = match.group(1) if match else "unknown"
            
            if USE_AVERAGING:
                plane_name = f"{AVERAGING_CONFIG['axis']}_averaged"
            else:
                axis_map = {'X': 'YZ_plane', 'Y': 'XZ_plane', 'Z': 'XY_plane'}
                plane_name = axis_map.get(SLICING_CONFIG['axis'].upper(), 'unknown_plane')
                
            return f"PV_{self.data_array}_iter_{iteration}_{plane_name}.png"
        except Exception:
            return 'visualization.png'
    
    def _apply_geometric_limits(self, limits):
        """Apply geometric clipping based on user-defined limits."""
        if not any(any(limit is not None for limit in axis_limits) 
                  for axis_limits in limits.values()):
            return self.reader  # No clipping needed
            
        print("INFO: Applying geometric limits using clip filters...")
        clipped_data = self.reader
        
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            axis_limits = limits.get(axis_name, [None, None])
            
            if axis_limits[0] is not None:  # Min limit
                clip_min = Clip(Input=clipped_data)
                clip_min.ClipType = 'Plane'
                normal = [0, 0, 0]
                normal[i] = 1
                clip_min.ClipType.Normal = normal
                clip_min.ClipType.Origin = [0, 0, 0]
                clip_min.ClipType.Origin[i] = axis_limits[0]
                clip_min.Invert = 0
                clipped_data = clip_min
                print(f"  Applied {axis_name}_min = {axis_limits[0]}")
                
            if axis_limits[1] is not None:  # Max limit
                clip_max = Clip(Input=clipped_data)
                clip_max.ClipType = 'Plane'
                normal = [0, 0, 0]
                normal[i] = -1
                clip_max.ClipType.Normal = normal
                clip_max.ClipType.Origin = [0, 0, 0]
                clip_max.ClipType.Origin[i] = axis_limits[1]
                clip_max.Invert = 0
                clipped_data = clip_max
                print(f"  Applied {axis_name}_max = {axis_limits[1]}")
                
        return clipped_data
    
    def _calculate_effective_bounds(self, original_bounds, limits):
        """Calculate the effective bounds after applying geometric limits."""
        effective_bounds = list(original_bounds)
        
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            axis_limits = limits.get(axis_name, [None, None])
            if axis_limits[0] is not None:
                effective_bounds[i*2] = max(axis_limits[0], original_bounds[i*2])
            if axis_limits[1] is not None:
                effective_bounds[i*2+1] = min(axis_limits[1], original_bounds[i*2+1])
                
        return effective_bounds
    
    def perform_averaging(self, config):
        """Perform spatial averaging analysis."""
        from vtkmodules.numpy_interface import dataset_adapter as dsa
        from vtk import vtkStructuredGrid, vtkPoints
        from vtk.util.numpy_support import numpy_to_vtk
        
        axis = config['axis'].upper()
        resolution = config['resolution']
        limits = config['geometric_limits']
        
        print(f"\n--- Starting {axis}-Axis Averaging Workflow ---")
        
        if axis not in ['X', 'Y', 'Z']:
            raise ValueError(f"Invalid averaging axis: '{axis}'. Must be 'X', 'Y', or 'Z'.")
            
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        
        # Get bounds and apply limits
        original_bounds = self.reader.GetDataInformation().GetBounds()
        print(f"INFO: Original bounds: X=[{original_bounds[0]:.2f}, {original_bounds[1]:.2f}], "
              f"Y=[{original_bounds[2]:.2f}, {original_bounds[3]:.2f}], "
              f"Z=[{original_bounds[4]:.2f}, {original_bounds[5]:.2f}]")
        
        effective_bounds = self._calculate_effective_bounds(original_bounds, limits)
        print(f"INFO: Effective bounds: X=[{effective_bounds[0]:.2f}, {effective_bounds[1]:.2f}], "
              f"Y=[{effective_bounds[2]:.2f}, {effective_bounds[3]:.2f}], "
              f"Z=[{effective_bounds[4]:.2f}, {effective_bounds[5]:.2f}]")
        
        # Apply geometric clipping
        clipped_data = self._apply_geometric_limits(limits)
        
        # Resample to uniform grid
        print(f"INFO: Resampling to uniform grid {resolution}...")
        resampled = ResampleToImage(Input=clipped_data)
        resampled.SamplingDimensions = resolution
        resampled.SamplingBounds = effective_bounds
        resampled.UpdatePipeline()
        
        # Extract and process data
        print("INFO: Processing data...")
        vtk_data = servermanager.Fetch(resampled)
        wrapped_data = dsa.WrapDataObject(vtk_data)
        
        # Handle both regular and derived arrays
        actual_array_name = self.data_array
        if actual_array_name not in [wrapped_data.PointData.GetArrayName(i) 
                                   for i in range(wrapped_data.PointData.GetNumberOfArrays())]:
            print(f"WARNING: Array '{actual_array_name}' not found, using first available array")
            actual_array_name = wrapped_data.PointData.GetArrayName(0)
        
        data_3d_flat = wrapped_data.PointData[actual_array_name]
        dims = resampled.SamplingDimensions
        data_3d = data_3d_flat.reshape(dims[2], dims[1], dims[0])  # (Nz, Ny, Nx)
        
        # Perform averaging based on axis
        averaging_map = {
            'X': {'numpy_axis': 2, 'result_axes': ['Y', 'Z'], 'grid_dims': [dims[1], dims[2]], 'bounds_idx': [2, 3, 4, 5]},
            'Y': {'numpy_axis': 1, 'result_axes': ['X', 'Z'], 'grid_dims': [dims[0], dims[2]], 'bounds_idx': [0, 1, 4, 5]},
            'Z': {'numpy_axis': 0, 'result_axes': ['X', 'Y'], 'grid_dims': [dims[0], dims[1]], 'bounds_idx': [0, 1, 2, 3]}
        }
        
        avg_info = averaging_map[axis]
        averaged_data_2d = np.mean(data_3d, axis=avg_info['numpy_axis'])
        array_name = f"{self.data_array}_{axis}_avg"
        
        print(f"INFO: Created {avg_info['result_axes'][0]}-{avg_info['result_axes'][1]} plane "
              f"with shape {averaged_data_2d.shape}")
        
        # Create structured grid for visualization
        return self._create_structured_grid(averaged_data_2d, array_name, effective_bounds, 
                                          avg_info, axis_index, axis)
    
    def _create_structured_grid(self, data_2d, array_name, bounds, avg_info, axis_index, axis):
        """Create a VTK structured grid from 2D averaged data."""
        from vtk import vtkStructuredGrid, vtkPoints
        from vtk.util.numpy_support import numpy_to_vtk
        
        n_axis1, n_axis2 = avg_info['grid_dims']
        
        structured_grid = vtkStructuredGrid()
        structured_grid.SetDimensions(n_axis1, n_axis2, 1)
        
        points = vtkPoints()
        bounds_idx = avg_info['bounds_idx']
        
        axis1_coords = np.linspace(bounds[bounds_idx[0]], bounds[bounds_idx[1]], n_axis1)
        axis2_coords = np.linspace(bounds[bounds_idx[2]], bounds[bounds_idx[3]], n_axis2)
        avg_coord = (bounds[axis_index*2] + bounds[axis_index*2+1]) / 2.0
        
        # Add points based on averaging axis
        for j in range(n_axis2):
            for i in range(n_axis1):
                if axis == 'X':
                    points.InsertNextPoint(avg_coord, axis1_coords[i], axis2_coords[j])
                elif axis == 'Y':
                    points.InsertNextPoint(axis1_coords[i], avg_coord, axis2_coords[j])
                else:  # Z
                    points.InsertNextPoint(axis1_coords[i], axis2_coords[j], avg_coord)
        
        structured_grid.SetPoints(points)
        
        vtk_array = numpy_to_vtk(data_2d.flatten('C'), deep=True)
        vtk_array.SetName(array_name)
        structured_grid.GetPointData().SetScalars(vtk_array)
        
        # Create producer
        producer = TrivialProducer(registrationName=f'{axis}_Averaged_Data')
        producer.GetClientSideObject().SetOutput(structured_grid)
        producer.UpdatePipeline()
        
        return producer, avg_info['result_axes']
    
    def perform_slicing(self, config):
        """Perform 2D slicing analysis."""
        axis = config['axis'].upper()
        coordinate = config['coordinate']
        
        print(f"\n--- Starting {axis}-Axis Slicing Workflow ---")
        
        if axis not in ['X', 'Y', 'Z']:
            raise ValueError(f"Invalid slice axis: '{axis}'. Must be 'X', 'Y', or 'Z'.")
        
        bounds = self.reader.GetDataInformation().GetBounds()
        
        if coordinate is None:
            bounds_idx = {'X': [0, 1], 'Y': [2, 3], 'Z': [4, 5]}[axis]
            coordinate = (bounds[bounds_idx[0]] + bounds[bounds_idx[1]]) / 2.0
            print(f"INFO: Auto-center slice at {axis} = {coordinate:.3f}")
        
        slice_params = {
            'X': {'origin': [coordinate, 0, 0], 'normal': [1, 0, 0], 
                  'cam_pos': [1, 0, 0], 'cam_up': [0, 0, 1], 'result_axes': ['Y', 'Z']},
            'Y': {'origin': [0, coordinate, 0], 'normal': [0, 1, 0], 
                  'cam_pos': [0, 1, 0], 'cam_up': [0, 0, 1], 'result_axes': ['X', 'Z']},
            'Z': {'origin': [0, 0, coordinate], 'normal': [0, 0, 1], 
                  'cam_pos': [0, 0, 1], 'cam_up': [0, 1, 0], 'result_axes': ['X', 'Y']}
        }
        
        params = slice_params[axis]
        
        slice_filter = Slice(Input=self.reader)
        slice_filter.SliceType = 'Plane'
        slice_filter.SliceType.Origin = params['origin']
        slice_filter.SliceType.Normal = params['normal']
        slice_filter.UpdatePipeline()
        
        print(f"INFO: Created slice at {axis} = {coordinate:.3f}")
        
        # Verify the data array exists in the slice
        slice_arrays = [slice_filter.GetPointDataInformation().GetArray(i).Name 
                       for i in range(slice_filter.GetPointDataInformation().GetNumberOfArrays())]
        
        if self.data_array not in slice_arrays:
            print(f"WARNING: Array '{self.data_array}' not found in slice. Available: {', '.join(slice_arrays)}")
            if slice_arrays:
                print(f"INFO: Using first available array: {slice_arrays[0]}")
        
        return slice_filter, params['result_axes']
    
    def visualize_and_save(self, source, result_axes, vis_config):
        """Create visualization and save output."""
        print("INFO: Creating visualization...")
        
        view = GetActiveViewOrCreate('RenderView')
        view.ViewSize = vis_config['image_size']
        
        display = Show(source, view)
        display.Representation = 'Surface'
        
        # Determine the correct array name for coloring
        array_name = self._get_coloring_array_name(source)
        
        print(f"INFO: Coloring by array: {array_name}")
        ColorBy(display, ('POINTS', array_name))
        display.RescaleTransferFunctionToDataRange(True, False)
        display.SetScalarBarVisibility(view, True)
        
        lut = GetColorTransferFunction(array_name)
        lut.ApplyPreset(vis_config.get('color_map', 'Blues'), True)
        
        # Set camera based on result plane
        self._setup_camera(view, result_axes)
        
        # Save output
        filename = self._generate_filename(vis_config)
        print(f"INFO: Saving to '{filename}'...")
        SaveScreenshot(filename, view, ImageResolution=vis_config['image_size'])
        print("INFO: Visualization saved successfully.")
    
    def _get_coloring_array_name(self, source):
        """Determine the appropriate array name for coloring the visualization."""
        if hasattr(source, 'GetPointDataInformation'):
            # For slicing - get arrays from the slice
            arrays = [source.GetPointDataInformation().GetArray(i).Name 
                     for i in range(source.GetPointDataInformation().GetNumberOfArrays())]
            
            # First try to find the exact array name
            if self.data_array in arrays:
                return self.data_array
            
            # For derived arrays, look for the calculated array in the source
            for array in arrays:
                if self.data_array.lower() in array.lower():
                    return array
            
            # Fallback to first available array
            if arrays:
                print(f"WARNING: '{self.data_array}' not found, using '{arrays[0]}'")
                return arrays[0]
                
        else:
            # For averaging - return the averaged array name
            if USE_AVERAGING:
                axis = AVERAGING_CONFIG['axis'].upper()
                return f"{self.data_array}_{axis}_avg"
        
        # Final fallback
        return self.data_array
    
    def _setup_camera(self, view, result_axes):
        """Configure camera based on the viewing plane."""
        camera_configs = {
            ('Y', 'Z'): {'pos': [1, 0, 0], 'up': [0, 0, 1]},  # X averaged/sliced
            ('X', 'Z'): {'pos': [0, 1, 0], 'up': [0, 0, 1]},  # Y averaged/sliced
            ('X', 'Y'): {'pos': [0, 0, 1], 'up': [0, 1, 0]}   # Z averaged/sliced
        }
        
        config = camera_configs.get(tuple(result_axes), {'pos': [1, 1, 1], 'up': [0, 0, 1]})
        
        view.CameraPosition = config['pos']
        view.CameraViewUp = config['up']
        view.CameraFocalPoint = [0, 0, 0]
        view.CameraParallelProjection = 1
        view.ResetCamera()
        view.StillRender()
    
    def run_analysis(self):
        """Execute the complete analysis workflow."""
        self._setup_paraview()
        self._load_data()
        
        if USE_AVERAGING:
            source, result_axes = self.perform_averaging(AVERAGING_CONFIG)
        else:
            source, result_axes = self.perform_slicing(SLICING_CONFIG)
        
        self.visualize_and_save(source, result_axes, VISUALIZATION_CONFIG)
        print("\n--- Analysis completed successfully ---")

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================

if __name__ == "__main__":
    analyzer = ParaViewAnalyzer(INPUT_FILE, DATA_ARRAY)
    analyzer.run_analysis()
