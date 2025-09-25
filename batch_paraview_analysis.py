#!/usr/bin/env python3
"""
Parallel-Enabled ParaView Batch Processor

TO RUN THIS:
==============================================================================
Single process (auto-detect all files):
python3 batch_paraview_analysis.py

Parallel processing with specific ranges:
python3 batch_paraview_analysis.py --range 100 200 2  # files 100-200, step 2
python3 batch_paraview_analysis.py --range 300 400 2  # files 300-400, step 2
python3 batch_paraview_analysis.py --range 500 600 2  # files 500-600, step 2

Or with file lists:
python3 batch_paraview_analysis.py --files iter_100.pvtu iter_150.pvtu iter_200.pvtu

Wulver example:
srun python3 batch_paraview_analysis.py --range 100 300 5 --process-id 1 &
srun python3 batch_paraview_analysis.py --range 305 500 5 --process-id 2 &
==============================================================================
"""

import subprocess
import os
import sys
import glob
import re
import tempfile
import logging
import argparse
from pathlib import Path
from datetime import datetime

#==============================================================================
# BATCH PROCESSING CONFIGURATION
#==============================================================================

# --- File Pattern Configuration ---
FILE_PATTERNS = {
    'pattern_type': 'iteration',
    'base_directory': './',
    'file_template': 'iter_{}.pvtu',
    'number_range': None,  # Will be set by command line args or auto-detect
}

# --- Analysis Configuration ---
BATCH_CONFIG = {
    #==============================================================================
    # SELECT VARIABLE
    #==============================================================================
    #'data_array': 'VELOMAG',
    'data_array': 'w',
    #'data_array': 'θ',
    #'data_array': 'ρ',
    
    #==============================================================================
    # SLICING OR AVERAGING?
    #==============================================================================
    #'analysis_mode': 'averaging',  # 'averaging' or 'slicing'
    'analysis_mode': 'slicing',  # 'averaging' or 'slicing'

    #==============================================================================
    # AVERAGING
    #==============================================================================
    'averaging': {
        'axis': 'Y',
        'resolution': [150, 150, 150],
        'geometric_limits': {
            'X': [None, None],
            'Y': [4500.0, 5500.0],
            'Z': [None, None]
        }
    },

    #==============================================================================
    # SLICING
    #==============================================================================
    'slicing': {
        'axis': 'Z',
        'coordinate': 100  # Use None for auto-center
    },
    
    'visualization': {
        'image_size': [1200, 800],
        'color_map': 'Blues',
    }
}

# --- Processing Options ---
PROCESSING_OPTIONS = {
    'output_directory': './batch_output/',
    'continue_on_error': True,
    'paraview_executable': 'pvpython',
    #'paraview_executable': '/Applications/ParaView-5.11.2.app/Contents/bin/pvpython',
    'paraview_args': ['--force-offscreen-rendering'],
    'timeout_seconds': 300,  # 5 minutes per file
    'log_file_prefix': 'batch_processing',  # Will add process ID
}

#==============================================================================
# COMMAND LINE ARGUMENT PARSING
#==============================================================================

def parse_arguments():
    """Parse command line arguments for parallel processing."""
    parser = argparse.ArgumentParser(
        description='ParaView Batch Processor with Parallel Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all files
  python3 batch_paraview_analysis.py
  
  # Process specific range
  python3 batch_paraview_analysis.py --range 100 200 2
  
  # Process specific files
  python3 batch_paraview_analysis.py --files iter_100.pvtu iter_200.pvtu
  
  # Show what files would be processed
  python3 batch_paraview_analysis.py --range 100 200 5 --dry-run
  
  # Suggest parallel processing ranges
  python3 batch_paraview_analysis.py --suggest-parallel
  
  # Parallel processing on cluster
  python3 batch_paraview_analysis.py --range 100 299 5 --process-id 1 &
  python3 batch_paraview_analysis.py --range 300 499 5 --process-id 2 &
  python3 batch_paraview_analysis.py --range 500 699 5 --process-id 3 &
  wait  # Wait for all background processes to complete
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--range', nargs=3, type=int, metavar=('START', 'END', 'STEP'),
                       help='Process files in range: start end step')
    group.add_argument('--files', nargs='+', metavar='FILE',
                       help='Process specific files')
    group.add_argument('--suggest-parallel', action='store_true',
                       help='Suggest how to split files across multiple processes')
    
    parser.add_argument('--process-id', type=int, default=None,
                       help='Process ID for parallel runs (affects log file naming)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what files would be processed without actually processing')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip files that already have output (default: True)')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of parallel processes to suggest (default: 4)')
    
    return parser.parse_args()

#==============================================================================
# IMPLEMENTATION
#==============================================================================

def setup_logging(process_id=None, log_level='INFO'):
    """Setup logging for the batch processor."""
    # Create unique log file name for parallel processes
    if process_id is not None:
        log_file = f"{PROCESSING_OPTIONS['log_file_prefix']}_proc_{process_id:03d}.log"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{PROCESSING_OPTIONS['log_file_prefix']}_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger

def extract_number_from_filename(filepath):
    """Extract number from filename based on template pattern."""
    filename = Path(filepath).name
    template = FILE_PATTERNS['file_template']
    
    # Create regex pattern from template
    regex_template = re.escape(template).replace(r'\{\}', r'([+-]?(?:\d+\.?\d*|\.\d+))')
    match = re.match(regex_template, filename)
    
    if match:
        try:
            number_str = match.group(1)
            try:
                return int(number_str)
            except ValueError:
                return float(number_str)
        except (ValueError, IndexError):
            pass
    
    # Fallback: extract any number from filename
    numbers = re.findall(r'([+-]?(?:\d+\.?\d*|\.\d+))', filename)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            try:
                return float(numbers[0])
            except ValueError:
                pass
    
    return 0

def find_files(args):
    """Find and sort files based on command line arguments."""
    base_dir = Path(FILE_PATTERNS['base_directory'])
    template = FILE_PATTERNS['file_template']
    
    if args.files:
        # Use specific files provided
        files = []
        for file_path in args.files:
            abs_path = os.path.abspath(file_path)
            if os.path.exists(abs_path):
                files.append(abs_path)
            else:
                print(f"WARNING: File not found: {file_path}")
        return files
    
    elif args.range:
        # Use specified range
        start, end, step = args.range
        files = []
        current = start
        while current <= end:
            filename = base_dir / template.format(current)
            if filename.exists():
                files.append(str(filename.absolute()))
            current += step
        return sorted(files)
    
    else:
        # Auto-detect all files (original behavior)
        glob_pattern = template.replace('{}', '*')
        search_path = base_dir / glob_pattern
        matching_files = glob.glob(str(search_path))
        
        if not matching_files:
            return []
        
        # Sort by extracted number
        files_with_numbers = []
        regex_template = re.escape(template).replace(r'\{\}', r'([+-]?(?:\d+\.?\d*|\.\d+))')
        
        for file_path in matching_files:
            abs_path = os.path.abspath(file_path)
            filename = Path(file_path).name
            match = re.match(regex_template, filename)
            
            if match:
                try:
                    number_str = match.group(1)
                    try:
                        number = int(number_str)
                    except ValueError:
                        number = float(number_str)
                    files_with_numbers.append((number, abs_path))
                except (ValueError, IndexError):
                    continue
        
        files_with_numbers.sort(key=lambda x: x[0])
        return [file_path for _, file_path in files_with_numbers]

def generate_output_filename(input_file, output_dir):
    """Generate output filename for the processed image."""
    file_number = extract_number_from_filename(input_file)
    
    # Format number for consistent sorting
    if isinstance(file_number, int):
        number_str = f"{file_number:06d}"
    else:
        number_str = f"{file_number:08.3f}".replace('.', '_')
    
    mode = 'avg' if BATCH_CONFIG['analysis_mode'] == 'averaging' else 'slice'
    axis = (BATCH_CONFIG['averaging']['axis'] if BATCH_CONFIG['analysis_mode'] == 'averaging' 
           else BATCH_CONFIG['slicing']['axis'])
    
    filename = f"{BATCH_CONFIG['data_array']}_{mode}_{axis}_{number_str}.png"
    return str(output_dir / filename)

def generate_processing_script(input_file, output_file):
    """Generate a standalone script to process a single file."""
    
    # Convert paths to absolute to avoid issues
    input_abs = os.path.abspath(input_file)
    output_abs = os.path.abspath(output_file)
    
    script_content = f"""#!/usr/bin/env python3
import os
import sys
from paraview.simple import *

def main():
    try:
        print("Starting ParaView processing...")
        
        # Setup ParaView
        paraview.simple._DisableFirstRenderCameraReset()
        
        # Load data
        print("Loading data...")
        reader = XMLPartitionedUnstructuredGridReader(FileName=[r"{input_abs}"])
        reader.UpdatePipeline()
        
        # Check if data loaded successfully
        data_info = reader.GetDataInformation()
        if data_info.GetNumberOfPoints() == 0:
            print("ERROR: No data points found")
            return False
        
        print(f"Loaded {{data_info.GetNumberOfPoints()}} points")
        
        # Get available arrays
        point_arrays = [reader.GetPointDataInformation().GetArray(i).Name 
                       for i in range(reader.GetPointDataInformation().GetNumberOfArrays())]
        print(f"Available arrays: {{', '.join(point_arrays)}}")
        
        # Handle special arrays
        source = reader
        data_array = "{BATCH_CONFIG['data_array']}"
        
        if data_array == 'VELOMAG':
            if all(comp in point_arrays for comp in ['u', 'v', 'w']):
                print("Calculating velocity magnitude...")
                calculator = Calculator(Input=reader)
                calculator.ResultArrayName = 'VELOMAG'
                calculator.Function = 'sqrt(u*u + v*v + w*w)'
                calculator.UpdatePipeline()
                source = calculator
            else:
                print("ERROR: Missing velocity components for VELOMAG")
                return False
        
        # Perform analysis
        if "{BATCH_CONFIG['analysis_mode']}" == 'averaging':
            # Averaging analysis
            axis = "{BATCH_CONFIG['averaging']['axis']}".upper()
            resolution = {BATCH_CONFIG['averaging']['resolution']}
            limits = {BATCH_CONFIG['averaging']['geometric_limits']}
            
            print(f"Starting {{axis}}-axis averaging...")
            
            from vtkmodules.numpy_interface import dataset_adapter as dsa
            from vtk import vtkStructuredGrid, vtkPoints
            from vtk.util.numpy_support import numpy_to_vtk
            import numpy as np
            
            original_bounds = source.GetDataInformation().GetBounds()
            print(f"Original bounds: X=[{{original_bounds[0]:.2f}}, {{original_bounds[1]:.2f}}], Y=[{{original_bounds[2]:.2f}}, {{original_bounds[3]:.2f}}], Z=[{{original_bounds[4]:.2f}}, {{original_bounds[5]:.2f}}]")
            
            effective_bounds = list(original_bounds)
            for i, axis_name in enumerate(['X', 'Y', 'Z']):
                axis_limits = limits.get(axis_name, [None, None])
                if axis_limits[0] is not None:
                    effective_bounds[i*2] = max(axis_limits[0], original_bounds[i*2])
                if axis_limits[1] is not None:
                    effective_bounds[i*2+1] = min(axis_limits[1], original_bounds[i*2+1])
            
            print(f"Effective bounds: X=[{{effective_bounds[0]:.2f}}, {{effective_bounds[1]:.2f}}], Y=[{{effective_bounds[2]:.2f}}, {{effective_bounds[3]:.2f}}], Z=[{{effective_bounds[4]:.2f}}, {{effective_bounds[5]:.2f}}]")
            
            clipped_data = source
            for i, axis_name in enumerate(['X', 'Y', 'Z']):
                axis_limits = limits.get(axis_name, [None, None])
                
                if axis_limits[0] is not None:
                    clip_min = Clip(Input=clipped_data)
                    clip_min.ClipType = 'Plane'
                    normal = [0, 0, 0]
                    normal[i] = 1
                    clip_min.ClipType.Normal = normal
                    clip_min.ClipType.Origin = [0, 0, 0]
                    clip_min.ClipType.Origin[i] = axis_limits[0]
                    clip_min.Invert = 0
                    clipped_data = clip_min
                    print(f"Applied {{axis_name}}_min = {{axis_limits[0]}}")
                    
                if axis_limits[1] is not None:
                    clip_max = Clip(Input=clipped_data)
                    clip_max.ClipType = 'Plane'
                    normal = [0, 0, 0]
                    normal[i] = -1
                    clip_max.ClipType.Normal = normal
                    clip_max.ClipType.Origin = [0, 0, 0]
                    clip_max.ClipType.Origin[i] = axis_limits[1]
                    clip_max.Invert = 0
                    clipped_data = clip_max
                    print(f"Applied {{axis_name}}_max = {{axis_limits[1]}}")
            
            print(f"Resampling to uniform grid {{resolution}}...")
            resampled = ResampleToImage(Input=clipped_data)
            resampled.SamplingDimensions = resolution
            resampled.SamplingBounds = effective_bounds
            resampled.UpdatePipeline()
            
            print("Processing 3D data for averaging...")
            from paraview.simple import servermanager
            vtk_data = servermanager.Fetch(resampled)
            wrapped_data = dsa.WrapDataObject(vtk_data)
            
            available_arrays = [wrapped_data.PointData.GetArrayName(i) 
                               for i in range(wrapped_data.PointData.GetNumberOfArrays())]
            if data_array not in available_arrays:
                print(f"WARNING: Array '{{data_array}}' not found, using first available array")
                actual_array_name = available_arrays[0]
            else:
                actual_array_name = data_array
            
            data_3d_flat = wrapped_data.PointData[actual_array_name]
            dims = resampled.SamplingDimensions
            data_3d = data_3d_flat.reshape(dims[2], dims[1], dims[0])
            
            print(f"Data shape: {{data_3d.shape}}")
            
            axis_index = {{'X': 0, 'Y': 1, 'Z': 2}}[axis]
            
            if axis == 'X':
                averaged_data_2d = np.mean(data_3d, axis=2)
                result_axes = ['Y', 'Z']
                grid_dims = [dims[1], dims[2]]
                bounds_indices = [2, 3, 4, 5]
            elif axis == 'Y':
                averaged_data_2d = np.mean(data_3d, axis=1)
                result_axes = ['X', 'Z']
                grid_dims = [dims[0], dims[2]]
                bounds_indices = [0, 1, 4, 5]
            else:
                averaged_data_2d = np.mean(data_3d, axis=0)
                result_axes = ['X', 'Y']
                grid_dims = [dims[0], dims[1]]
                bounds_indices = [0, 1, 2, 3]
            
            print(f"Averaged data shape: {{averaged_data_2d.shape}}")
            print(f"Created {{result_axes[0]}}-{{result_axes[1]}} plane")
            
            n_axis1, n_axis2 = grid_dims
            structured_grid = vtkStructuredGrid()
            structured_grid.SetDimensions(n_axis1, n_axis2, 1)
            
            points = vtkPoints()
            axis1_coords = np.linspace(effective_bounds[bounds_indices[0]], effective_bounds[bounds_indices[1]], n_axis1)
            axis2_coords = np.linspace(effective_bounds[bounds_indices[2]], effective_bounds[bounds_indices[3]], n_axis2)
            avg_coord = (effective_bounds[axis_index*2] + effective_bounds[axis_index*2+1]) / 2.0
            
            for j in range(n_axis2):
                for i in range(n_axis1):
                    if axis == 'X':
                        points.InsertNextPoint(avg_coord, axis1_coords[i], axis2_coords[j])
                    elif axis == 'Y':
                        points.InsertNextPoint(axis1_coords[i], avg_coord, axis2_coords[j])
                    else:
                        points.InsertNextPoint(axis1_coords[i], axis2_coords[j], avg_coord)
            
            structured_grid.SetPoints(points)
            
            vtk_array = numpy_to_vtk(averaged_data_2d.flatten('C'), deep=True)
            vtk_array.SetName(f"{{data_array}}_{{axis}}_avg")
            structured_grid.GetPointData().SetScalars(vtk_array)
            
            from paraview.simple import TrivialProducer
            producer = TrivialProducer(registrationName=f'{{axis}}_Averaged_Data')
            producer.GetClientSideObject().SetOutput(structured_grid)
            producer.UpdatePipeline()
            
            source = producer
            data_array = f"{{data_array}}_{{axis}}_avg"
            
        else:
            # Slicing analysis
            axis = "{BATCH_CONFIG['slicing']['axis']}".upper()
            coordinate = {BATCH_CONFIG['slicing']['coordinate'] if BATCH_CONFIG['slicing']['coordinate'] is not None else 'None'}
            
            bounds = source.GetDataInformation().GetBounds()
            print(f"Data bounds: X=[{{bounds[0]:.2f}}, {{bounds[1]:.2f}}], Y=[{{bounds[2]:.2f}}, {{bounds[3]:.2f}}], Z=[{{bounds[4]:.2f}}, {{bounds[5]:.2f}}]")
            
            if coordinate is None:
                bounds_idx = {{'X': [0, 1], 'Y': [2, 3], 'Z': [4, 5]}}[axis]
                coordinate = (bounds[bounds_idx[0]] + bounds[bounds_idx[1]]) / 2.0
                print(f"Auto-selected slice coordinate: {{coordinate}}")
            
            slice_params = {{
                'X': {{'origin': [coordinate, 0, 0], 'normal': [1, 0, 0]}},
                'Y': {{'origin': [0, coordinate, 0], 'normal': [0, 1, 0]}},
                'Z': {{'origin': [0, 0, coordinate], 'normal': [0, 0, 1]}}
            }}
            
            params = slice_params[axis]
            print(f"Creating slice at {{axis}} = {{coordinate}}")
            
            slice_filter = Slice(Input=source)
            slice_filter.SliceType = 'Plane'
            slice_filter.SliceType.Origin = params['origin']
            slice_filter.SliceType.Normal = params['normal']
            slice_filter.UpdatePipeline()
            
            source = slice_filter
        
        # Get arrays from processed source
        final_arrays = [source.GetPointDataInformation().GetArray(i).Name 
                       for i in range(source.GetPointDataInformation().GetNumberOfArrays())]
        print(f"Final arrays available: {{', '.join(final_arrays)}}")
        
        # Visualize
        print("Creating visualization...")
        view = GetActiveViewOrCreate('RenderView')
        view.ViewSize = {BATCH_CONFIG['visualization']['image_size']}
        
        display = Show(source, view)
        display.Representation = 'Surface'
        
        # Color by data array
        if data_array in final_arrays:
            array_name = data_array
        elif final_arrays:
            array_name = final_arrays[0]
            print(f"Using first available array: {{array_name}}")
        else:
            print("ERROR: No arrays found for coloring")
            return False
        
        print(f"Coloring by: {{array_name}}")
        ColorBy(display, ('POINTS', array_name))
        display.RescaleTransferFunctionToDataRange(True, False)
        display.SetScalarBarVisibility(view, True)
        
        lut = GetColorTransferFunction(array_name)
        lut.ApplyPreset("{BATCH_CONFIG['visualization']['color_map']}", True)
        
        # Set camera based on analysis mode and axis
        if "{BATCH_CONFIG['analysis_mode']}" == 'averaging':
            # Camera setup for averaging
            avg_axis = "{BATCH_CONFIG['averaging']['axis']}".upper()
            if avg_axis == 'X':
                # Viewing Y-Z plane
                view.CameraPosition = [1, 0, 0]
                view.CameraViewUp = [0, 0, 1]
            elif avg_axis == 'Y':
                # Viewing X-Z plane  
                view.CameraPosition = [0, 1, 0]
                view.CameraViewUp = [0, 0, 1]
            else:  # Z
                # Viewing X-Y plane
                view.CameraPosition = [0, 0, 1]
                view.CameraViewUp = [0, 1, 0]
        else:
            # Camera setup for slicing
            slice_axis = "{BATCH_CONFIG['slicing']['axis']}".upper()
            if slice_axis == 'X':
                view.CameraPosition = [1, 0, 0]
                view.CameraViewUp = [0, 0, 1]
            elif slice_axis == 'Y':
                view.CameraPosition = [0, 1, 0]
                view.CameraViewUp = [0, 0, 1]
            else:  # Z
                view.CameraPosition = [0, 0, 1]
                view.CameraViewUp = [0, 1, 0]
        
        view.CameraFocalPoint = [0, 0, 0]
        view.CameraParallelProjection = 1
        view.ResetCamera()
        view.StillRender()
        
        # Save screenshot
        print(f"Saving to: {{r'{output_abs}'}}")
        SaveScreenshot(r"{output_abs}", view, ImageResolution={BATCH_CONFIG['visualization']['image_size']})
        
        # Verify output file
        if os.path.exists(r"{output_abs}"):
            size = os.path.getsize(r"{output_abs}")
            print(f"Output file size: {{size}} bytes")
            if size < 1000:
                print("WARNING: Output file is very small")
                return False
        else:
            print("ERROR: Output file was not created")
            return False
        
        print("Processing completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: {{str(e)}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
    
    return script_content

def suggest_parallel_ranges(files, num_processes=4):
    """Suggest how to split files across multiple processes."""
    if not files:
        print("No files found to suggest ranges for.")
        return
    
    print(f"\nSuggested parallel processing with {num_processes} processes:")
    print("=" * 60)
    
    # Extract numbers and sort them
    file_numbers = []
    for f in files:
        num = extract_number_from_filename(f)
        file_numbers.append(num)
    
    if not file_numbers:
        return
    
    file_numbers.sort()
    min_num, max_num = file_numbers[0], file_numbers[-1]
    total_files = len(file_numbers)
    files_per_process = total_files // num_processes
    
    print(f"Total files: {total_files}")
    print(f"File numbers range: {min_num} to {max_num}")
    print(f"Files per process: ~{files_per_process}")
    print()
    
    # Split files into chunks
    chunk_size = len(file_numbers) // num_processes
    remainder = len(file_numbers) % num_processes
    
    ranges = []
    start_idx = 0
    
    for i in range(num_processes):
        # Distribute remainder across first processes
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size - 1
        
        if start_idx >= len(file_numbers):
            break
            
        start_num = file_numbers[start_idx]
        end_num = file_numbers[min(end_idx, len(file_numbers) - 1)]
        
        # Calculate appropriate step size based on actual file numbers
        if start_idx == end_idx:
            step = 1
        else:
            # Find the most common step size in this range
            steps = []
            for j in range(start_idx, min(end_idx, len(file_numbers) - 1)):
                steps.append(file_numbers[j+1] - file_numbers[j])
            
            if steps:
                # Use the minimum step as a safe choice
                step = min(steps)
            else:
                step = 1
        
        # Ensure step is at least 1
        step = max(1, step)
        
        ranges.append((start_num, end_num, step))
        
        file_count = len([n for n in file_numbers[start_idx:end_idx+1]])
        print(f"Process {i+1}: python3 batch_paraview_analysis.py --range {start_num} {end_num} {step} --process-id {i+1} &")
        print(f"           # Will process ~{file_count} files")
        
        start_idx = end_idx + 1
    
    print()
    print("To run all processes:")
    print("#!/bin/bash")
    for i, (start_num, end_num, step) in enumerate(ranges):
        print(f"python3 batch_paraview_analysis.py --range {start_num} {end_num} {step} --process-id {i+1} &")
    print("wait  # Wait for all processes to complete")
    print()
    print("Or use the dry-run flag to test first:")
    print("python3 batch_paraview_analysis.py --range START END STEP --dry-run")
    
    return ranges

def process_files():
    """Process all files using subprocess isolation."""
    
    args = parse_arguments()
    logger = setup_logging(args.process_id, args.log_level)
    
    # Override output directory if specified
    if args.output_dir:
        PROCESSING_OPTIONS['output_directory'] = args.output_dir
    
    print("ParaView Parallel Batch Processor")
    print("=" * 50)
    print(f"Data array: {BATCH_CONFIG['data_array']}")
    print(f"Analysis mode: {BATCH_CONFIG['analysis_mode']}")
    print(f"Process ID: {args.process_id or 'main'}")
    
    if args.range:
        print(f"Range: {args.range[0]} to {args.range[1]} (step {args.range[2]})")
    elif args.files:
        print(f"Specific files: {len(args.files)} files")
    else:
        print("Mode: Auto-detect all files")
    
    print("=" * 50)
    
    # Find files based on arguments
    files = find_files(args)
    if not files:
        logger.error("No files found to process")
        return
    
    logger.info(f"Found {len(files)} files to process")
    
    # Show files in dry-run mode
    if args.dry_run:
        print("\nDRY RUN - Files that would be processed:")
        for f in files:
            print(f"  {Path(f).name}")
        return
    
    for f in files:
        logger.info(f"  {Path(f).name}")
    
    # Setup output directory
    output_dir = Path(PROCESSING_OPTIONS['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    processed_count = 0
    failed_count = 0
    start_time = datetime.now()
    
    for i, input_file in enumerate(files, 1):
        logger.info(f"Processing file {i}/{len(files)}: {Path(input_file).name}")
        
        try:
            # Generate output filename
            output_file = generate_output_filename(input_file, output_dir)
            logger.info(f"  Output: {Path(output_file).name}")
            
            # Skip if output already exists (useful for resuming parallel jobs)
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                if size > 1000:  # Reasonable size threshold
                    logger.info(f"  ⏭  Skipping (output exists): {Path(output_file).name}")
                    processed_count += 1
                    continue
            
            # Generate processing script
            script_content = generate_processing_script(input_file, output_file)
            
            # Write temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                temp_script.write(script_content)
                temp_script_path = temp_script.name
            
            try:
                # Run subprocess
                logger.info("  Starting ParaView subprocess...")
                if PROCESSING_OPTIONS['paraview_args']:
                    result = subprocess.run([PROCESSING_OPTIONS['paraview_executable']] +  
                    PROCESSING_OPTIONS['paraview_args'] + [temp_script_path], 
                    capture_output=True, text=True, timeout=PROCESSING_OPTIONS['timeout_seconds'])
                else: 
                    result = subprocess.run([
                    PROCESSING_OPTIONS['paraview_executable'],
                    temp_script_path], 
                    capture_output=True, text=True, timeout=PROCESSING_OPTIONS['timeout_seconds'])
                
                if result.returncode == 0:
                    logger.info(f"  ✓ Success: {Path(output_file).name}")
                    processed_count += 1
                    
                    # Verify output file
                    if os.path.exists(output_file):
                        size = os.path.getsize(output_file)
                        logger.info(f"    File size: {size:,} bytes")
                        if size < 1000:
                            logger.warning("    Small file size - may indicate visualization issue")
                    else:
                        logger.warning("    Output file not found")
                        
                    # Log subprocess output for debugging
                    if result.stdout and logger.level <= logging.DEBUG:
                        logger.debug(f"    Subprocess output: {result.stdout}")
                else:
                    logger.error(f"  ✗ Failed: Return code {result.returncode}")
                    if result.stderr:
                        logger.error(f"    Error: {result.stderr}")
                    if result.stdout:
                        logger.error(f"    Output: {result.stdout}")
                    failed_count += 1
                    
                    if not PROCESSING_OPTIONS['continue_on_error']:
                        break
                        
            finally:
                # Clean up temporary script
                try:
                    os.unlink(temp_script_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ Timeout processing {input_file}")
            failed_count += 1
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            failed_count += 1
            
            if not PROCESSING_OPTIONS['continue_on_error']:
                break
    
    # Summary
    total_time = datetime.now() - start_time
    logger.info("\n" + "=" * 50)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"Process ID: {args.process_id or 'main'}")
    logger.info(f"Total time: {total_time}")
    logger.info(f"Successfully processed: {processed_count} files")
    logger.info(f"Failed: {failed_count} files")
    
    if processed_count > 0:
        logger.info(f"\nOutput files are in: {output_dir.absolute()}")
        logger.info("Files are numbered for easy GIF creation:")
        output_files = sorted(output_dir.glob("*.png"))
        for f in output_files[:5]:  # Show first 5
            logger.info(f"  {f.name}")
        if len(output_files) > 5:
            logger.info(f"  ... and {len(output_files) - 5} more")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Special mode: suggest parallel ranges
    if args.suggest_parallel:
        files = find_files(args)
        suggest_parallel_ranges(files, args.num_processes)
    else:
        process_files()
