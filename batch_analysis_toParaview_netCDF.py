#!/usr/bin/env python3
"""
Parallel-Enabled ParaView Batch Processor for PNG and NetCDF Output

This script can generate 2D images (PNG) and/or 2D data files (NetCDF)
from 3D ParaView data files (.pvtu).

REQUIREMENTS:
==============================================================================
- For PNG output: A local installation of ParaView.
- For NetCDF output: The following Python libraries must be installed:
  pip install numpy scipy xarray netcdf4

TO RUN THIS:
==============================================================================
# Process files 100-200 (step 2) and create both PNG and NetCDF files
python3 batch_paraview_analysis.py --range 100 200 2 --output-format both

# Create only NetCDF files
python3 batch_paraview_analysis.py --range 100 200 2 --output-format netcdf

# Suggest parallel processing ranges
python3 batch_paraview_analysis.py --suggest-parallel --num-processes 4
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

# Import libraries required for NetCDF creation
try:
    import numpy as np
    import xarray as xr
    from scipy.interpolate import griddata
except ImportError as e:
    print(f"Error: A required library is missing. Please install it.")
    print("For NetCDF functionality, you need: pip install numpy scipy xarray netcdf4")
    print(f"Original error: {e}")
    sys.exit(1)


#==============================================================================
# BATCH PROCESSING CONFIGURATION
#==============================================================================

# --- File Pattern Configuration ---
FILE_PATTERNS = {
    'pattern_type': 'iteration',
    'base_directory': '/Users/simone/Work-local/Codes/Jexpresso/output/CompEuler/LESsmago/output-10240x10240x3000/',
    'file_template': 'iter_{}.pvtu',
    'number_range': None,  # Will be set by command line args or auto-detect
}

# --- Analysis Configuration ---
BATCH_CONFIG = {
    #==============================================================================
    # SELECT VARIABLE
    #==============================================================================
    'data_array': 'w',
    
    #==============================================================================
    # SLICING OR AVERAGING?
    #==============================================================================
    'analysis_mode': 'slicing',  # 'averaging' or 'slicing'

    #==============================================================================
    # OUTPUT FORMAT
    #==============================================================================
    'output_format': ['png', 'netcdf'], # Options: 'png', 'netcdf', or both in a list
    'netcdf_grid_resolution': 512, # Base resolution for the longest axis of the NetCDF grid

    #==============================================================================
    # AVERAGING
    #==============================================================================
    'averaging': {
        'axis': 'Y',
        'resolution': [150, 150, 150],
        'geometric_limits': {
            'X': [None, None], 'Y': [4500.0, 5500.0], 'Z': [None, None]
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
    'temp_directory': './batch_temp/', # For intermediate files for NetCDF creation
    'continue_on_error': True,
    'paraview_executable': '/Applications/ParaView-5.11.2.app/Contents/bin/pvpython',
    'paraview_args': ['--force-offscreen-rendering'],
    'timeout_seconds': 300,  # 5 minutes per file
    'log_file_prefix': 'batch_processing',
}

#==============================================================================
# COMMAND LINE ARGUMENT PARSING
#==============================================================================

def parse_arguments():
    """Parse command line arguments for parallel processing."""
    parser = argparse.ArgumentParser(
        description='ParaView Batch Processor with Parallel Support for PNG and NetCDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--range', nargs=3, type=int, metavar=('START', 'END', 'STEP'),
                        help='Specify a range of file numbers to process.')
    parser.add_argument('--files', nargs='+', type=str,
                        help='A specific list of files to process.')
    parser.add_argument('--process-id', type=str, default=None,
                        help='An identifier for this process, used for logging.')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set the logging level.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip processing if the output file already exists.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override the output directory specified in the config.')
    parser.add_argument('--suggest-parallel', action='store_true',
                        help='Suggest parallel command ranges instead of running.')
    parser.add_argument('--num-processes', type=int, default=4,
                        help='Number of processes to suggest for parallel execution.')
    
    parser.add_argument('--output-format', type=str, choices=['png', 'netcdf', 'both'],
                       default=None, help='Specify output format (overrides config)')

    return parser.parse_args()

#==============================================================================
# IMPLEMENTATION
#==============================================================================

def setup_logging(process_id=None, log_level='INFO'):
    """Sets up a logger that prints to console and saves to a file."""
    log_dir = Path(PROCESSING_OPTIONS['output_directory']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid_str = f"_pid_{process_id}" if process_id else ""
    log_filename = log_dir / f"{PROCESSING_OPTIONS['log_file_prefix']}{pid_str}_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler(sys.stdout)

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def extract_number_from_filename(filepath):
    """Extracts the numerical part of the filename for sorting."""
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+\.?\d*)', filename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return float(match.group(1))
    return filename

def find_files(args):
    """Finds and sorts files based on configuration and command-line arguments."""
    logger = logging.getLogger(__name__)
    
    if args.files:
        logger.info(f"Processing a specific list of {len(args.files)} files.")
        return [str(Path(f).absolute()) for f in args.files]

    base_dir = Path(FILE_PATTERNS['base_directory'])
    all_files = sorted(glob.glob(str(base_dir / FILE_PATTERNS['file_template'].format('*'))))
    
    if not all_files:
        logger.error(f"No files found with pattern '{FILE_PATTERNS['file_template']}' in '{base_dir.absolute()}'")
        return []

    start_num, end_num, step = (args.range[0], args.range[1], args.range[2]) if args.range else (None, None, 1)

    if start_num is None:
        logger.info("No range specified. Processing all found files.")
        return all_files

    filtered_files = []
    for f in all_files:
        num = extract_number_from_filename(f)
        if isinstance(num, (int, float)) and start_num <= num <= end_num and (num - start_num) % step == 0:
            filtered_files.append(f)
    
    logger.info(f"Found {len(filtered_files)} files to process in range {start_num}-{end_num} (step {step}).")
    return filtered_files

def generate_output_basename(input_file, output_dir):
    """Generate a base output filename without extension."""
    file_number = extract_number_from_filename(input_file)
    
    if isinstance(file_number, int):
        number_str = f"{file_number:06d}"
    else:
        number_str = f"{file_number:08.3f}".replace('.', '_')
    
    mode = 'avg' if BATCH_CONFIG['analysis_mode'] == 'averaging' else 'slice'
    axis = (BATCH_CONFIG['averaging']['axis'] if BATCH_CONFIG['analysis_mode'] == 'averaging' 
           else BATCH_CONFIG['slicing']['axis'])
    
    filename = f"{BATCH_CONFIG['data_array']}_{mode}_{axis}_{number_str}"
    return str(output_dir / filename)

def generate_processing_script(input_file, png_output_file, temp_dir):
    """Generate a standalone script to process a single file."""
    
    input_abs = os.path.abspath(input_file)
    png_output_abs = os.path.abspath(png_output_file) if png_output_file else "None"
    
    temp_npz_file = os.path.join(
        os.path.abspath(temp_dir),
        f"temp_data_{Path(input_file).stem}.npz"
    )

    save_png = "'png' in BATCH_CONFIG['output_format']"
    save_npz = "'netcdf' in BATCH_CONFIG['output_format']"

    ### MODIFICATION HERE ###
    script_content = f"""#!/usr/bin/env python3
import sys
import os
from paraview.simple import *
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa
from paraview.simple import servermanager

BATCH_CONFIG = {BATCH_CONFIG}

def main():
    try:
        input_file = r'{input_abs}'
        data_array = BATCH_CONFIG['data_array']

        print(f"Processing file: {{input_file}}")

        # --- DATA LOADING (Robust Method) ---
        source = OpenDataFile(input_file)
        if not source:
            print(f"ERROR: Failed to open file {{input_file}} with OpenDataFile.")
            return False
        
        # --- ANALYSIS ---
        if BATCH_CONFIG['analysis_mode'] == 'averaging':
            print("Mode: Averaging")
            resample = ResampleToImage(Input=source)
            resample.SamplingDimensions = BATCH_CONFIG['averaging']['resolution']
            
            bounds = source.GetDataInformation().GetBounds()
            geo_limits = BATCH_CONFIG['averaging']['geometric_limits']
            
            x_min = geo_limits['X'][0] if geo_limits['X'][0] is not None else bounds[0]
            x_max = geo_limits['X'][1] if geo_limits['X'][1] is not None else bounds[1]
            y_min = geo_limits['Y'][0] if geo_limits['Y'][0] is not None else bounds[2]
            y_max = geo_limits['Y'][1] if geo_limits['Y'][1] is not None else bounds[3]
            z_min = geo_limits['Z'][0] if geo_limits['Z'][0] is not None else bounds[4]
            z_max = geo_limits['Z'][1] if geo_limits['Z'][1] is not None else bounds[5]
            
            resample.SetSamplingBounds(x_min, x_max, y_min, y_max, z_min, z_max)
            
            average = PythonCalculator(Input=resample)
            average.Expression = f"average(inputs[0].PointData['{{data_array}}'], '{BATCH_CONFIG['averaging']['axis']}')"
            source = average

        elif BATCH_CONFIG['analysis_mode'] == 'slicing':
            print("Mode: Slicing")
            slice_op = Slice(Input=source)
            slice_op.SliceType = 'Plane'
            
            axis = BATCH_CONFIG['slicing']['axis'].upper()
            if axis == 'X': slice_op.SliceType.Normal = [1.0, 0.0, 0.0]
            elif axis == 'Y': slice_op.SliceType.Normal = [0.0, 1.0, 0.0]
            else: slice_op.SliceType.Normal = [0.0, 0.0, 1.0]

            coordinate = BATCH_CONFIG['slicing']['coordinate']
            if coordinate is None:
                bounds = source.GetDataInformation().GetBounds()
                center = [(bounds[i] + bounds[i+1]) / 2.0 for i in [0, 2, 4]]
                slice_op.SliceType.Origin = center
            else:
                slice_op.SliceType.Origin = [0,0,0]
                if axis == 'X': slice_op.SliceType.Origin[0] = coordinate
                elif axis == 'Y': slice_op.SliceType.Origin[1] = coordinate
                else: slice_op.SliceType.Origin[2] = coordinate
            source = slice_op
        
        final_arrays = [source.GetPointDataInformation().GetArray(i).Name 
                       for i in range(source.GetPointDataInformation().GetNumberOfArrays())]
        
        if data_array in final_arrays: array_name_to_use = data_array
        elif final_arrays:
            array_name_to_use = final_arrays[0]
            print(f"Using first available array: {{array_name_to_use}}")
        else:
            print("ERROR: No arrays found for processing")
            return False

        # --- OUTPUT GENERATION ---
        if {save_png}:
            print("Creating visualization for PNG...")
            view = GetActiveViewOrCreate('RenderView')
            display = Show(source, view)
            ColorBy(display, ('POINTS', array_name_to_use))
            
            color_map = GetColorTransferFunction(array_name_to_use)
            color_map.ApplyPreset(BATCH_CONFIG['visualization']['color_map'], True)
            
            scalar_bar = GetScalarBar(color_map, view)
            scalar_bar.Visibility = 1
            
            view.ResetCamera()
            Render()
            
            SaveScreenshot(r"{png_output_abs}", view, ImageResolution=BATCH_CONFIG['visualization']['image_size'])
            
            if os.path.exists(r"{png_output_abs}") and os.path.getsize(r"{png_output_abs}") > 1000:
                print(f"PNG file created successfully.")
            else:
                print(f"ERROR: PNG file was not created or is empty.")
                return False

        if {save_npz}:
            print("Extracting data for NetCDF file...")
            try:
                vtk_data = servermanager.Fetch(source)
                wrapped_data = dsa.WrapDataObject(vtk_data)
                
                points = wrapped_data.Points
                values = wrapped_data.PointData[array_name_to_use]
                
                print(f"  Extracted {{len(points)}} points.")
                np.savez_compressed(
                    r"{temp_npz_file}",
                    points=points,
                    values=values,
                    array_name=array_name_to_use
                )
                print(f"INTERMEDIATE_FILE:{{r'{temp_npz_file}'}}")
                
            except Exception as e:
                print(f"ERROR extracting data for NetCDF: {{e}}")
                return False

        print("Processing completed successfully")
        return True
        
    except Exception as e:
        import traceback
        print("An unexpected error occurred during ParaView processing:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
    return script_content

def suggest_parallel_ranges(files, num_processes=4):
    """Suggests command line ranges for parallel processing."""
    if not files:
        print("No files found to process.")
        return

    total_files = len(files)
    chunk_size = (total_files + num_processes - 1) // num_processes

    print("\n" + "="*60)
    print(f"Suggested parallel command ranges for {num_processes} processes:")
    print("="*60)

    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size - 1, total_files - 1)

        if start_index >= total_files:
            continue

        start_num = extract_number_from_filename(files[start_index])
        end_num = extract_number_from_filename(files[end_index])
        
        step = 1
        if len(files) > 1:
            step = extract_number_from_filename(files[1]) - extract_number_from_filename(files[0])
            if step <= 0: step = 1

        print(f"\n# Process {i+1}:")
        print(f"python3 {Path(__file__).name} --range {start_num} {end_num} {step} --process-id {i+1} &")
    
    print("\n" + "="*60)

def create_netcdf_from_npz(npz_path, nc_path, analysis_mode, axis, data_array, resolution):
    """
    Reads an NPZ file, interpolates data onto a regular grid, and saves as NetCDF.
    """
    try:
        data = np.load(npz_path)
        points_3d = data['points']
        values = data['values']
        array_name = str(data['array_name'])
        
        if analysis_mode == 'slicing':
            if axis.upper() == 'X': points_2d, dims, coord_names = points_3d[:, [1, 2]], ['y', 'z'], ('y', 'z')
            elif axis.upper() == 'Y': points_2d, dims, coord_names = points_3d[:, [0, 2]], ['x', 'z'], ('x', 'z')
            else: points_2d, dims, coord_names = points_3d[:, [0, 1]], ['x', 'y'], ('y', 'x')
        elif analysis_mode == 'averaging':
             if axis.upper() == 'X': points_2d, dims, coord_names = points_3d[:, [1, 2]], ['y', 'z'], ('z', 'y')
             elif axis.upper() == 'Y': points_2d, dims, coord_names = points_3d[:, [0, 2]], ['x', 'z'], ('z', 'x')
             else: points_2d, dims, coord_names = points_3d[:, [0, 1]], ['x', 'y'], ('y', 'x')

        min1, max1 = points_2d[:, 0].min(), points_2d[:, 0].max()
        min2, max2 = points_2d[:, 1].min(), points_2d[:, 1].max()
        range1, range2 = max1 - min1, max2 - min2
        
        if range1 >= range2:
            n1, n2 = resolution, int(resolution * (range2 / range1)) if range1 > 0 else 1
        else:
            n2, n1 = resolution, int(resolution * (range1 / range2)) if range2 > 0 else 1
        
        coords1, coords2 = np.linspace(min1, max1, n1), np.linspace(min2, max2, n2)
        grid1, grid2 = np.meshgrid(coords1, coords2)

        interpolated_array = griddata(points_2d, values, (grid1, grid2), method='linear', fill_value=np.nan)

        ds = xr.Dataset(
            {array_name: (coord_names, interpolated_array)},
            coords={dims[0]: (dims[0], coords1), dims[1]: (dims[1], coords2)}
        )
        ds[dims[0]].attrs.update(units='m', long_name=f'{dims[0].upper()} Coordinate')
        ds[dims[1]].attrs.update(units='m', long_name=f'{dims[1].upper()} Coordinate')
        ds.attrs.update(
            title=f'{analysis_mode.capitalize()} of {data_array} on {axis.upper()}-plane',
            creation_date=str(datetime.now())
        )
        ds.to_netcdf(nc_path)
        return True

    except Exception as e:
        logging.getLogger(__name__).error(f"  Failed to create NetCDF file: {e}")
        return False
    finally:
        if os.path.exists(npz_path):
            os.remove(npz_path)

def process_files():
    """Process all files using subprocess isolation."""
    
    args = parse_arguments()
    logger = setup_logging(args.process_id, args.log_level)

    if args.output_dir:
        PROCESSING_OPTIONS['output_directory'] = args.output_dir

    if args.output_format:
        BATCH_CONFIG['output_format'] = ['png', 'netcdf'] if args.output_format == 'both' else [args.output_format]
    
    logger.info("Starting batch processing...")
    logger.info(f"Process ID: {args.process_id or 'main'}")
    logger.info(f"Output directory: {Path(PROCESSING_OPTIONS['output_directory']).absolute()}")
    
    files = find_files(args)
    if not files:
        logger.warning("No files to process. Exiting.")
        return
    
    output_dir = Path(PROCESSING_OPTIONS['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(PROCESSING_OPTIONS['temp_directory'])
    if 'netcdf' in BATCH_CONFIG['output_format']:
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory for NetCDF data: {temp_dir.absolute()}")
    
    start_time = datetime.now()
    processed_count, failed_count = 0, 0

    for i, input_file in enumerate(files, 1):
        logger.info(f"Processing file {i}/{len(files)}: {Path(input_file).name}")
        
        try:
            output_basename = generate_output_basename(input_file, output_dir)
            png_file, nc_file = f"{output_basename}.png", f"{output_basename}.nc"

            if args.skip_existing:
                png_exists = not ('png' in BATCH_CONFIG['output_format']) or (os.path.exists(png_file) and os.path.getsize(png_file) > 1000)
                nc_exists = not ('netcdf' in BATCH_CONFIG['output_format']) or (os.path.exists(nc_file) and os.path.getsize(nc_file) > 100)
                if png_exists and nc_exists:
                    logger.info(f"  ⏭  Skipping (all requested outputs exist)")
                    processed_count += 1
                    continue
            
            script_content = generate_processing_script(input_file, png_file, temp_dir)
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as temp_script:
                temp_script.write(script_content)
                temp_script_path = temp_script.name

            try:
                command = [PROCESSING_OPTIONS['paraview_executable'], *PROCESSING_OPTIONS['paraview_args'], temp_script_path]
                logger.debug(f"  Executing: {' '.join(command)}")
                
                result = subprocess.run(command, capture_output=True, text=True, timeout=PROCESSING_OPTIONS['timeout_seconds'])

                if result.returncode == 0:
                    logger.info(f"  ✓ Subprocess finished successfully.")
                    processed_count += 1
                    
                    intermediate_file = next((line.split(":", 1)[1] for line in result.stdout.splitlines() if line.startswith("INTERMEDIATE_FILE:")), None)
                    
                    if intermediate_file and os.path.exists(intermediate_file):
                        logger.info(f"  Creating NetCDF file: {Path(nc_file).name}")
                        success = create_netcdf_from_npz(
                            intermediate_file, nc_file, BATCH_CONFIG['analysis_mode'],
                            (BATCH_CONFIG['averaging']['axis'] if BATCH_CONFIG['analysis_mode'] == 'averaging' else BATCH_CONFIG['slicing']['axis']),
                            BATCH_CONFIG['data_array'], BATCH_CONFIG['netcdf_grid_resolution']
                        )
                        if success: logger.info(f"    ✓ NetCDF created: {Path(nc_file).name}")
                        else:
                            logger.error(f"    ✗ Failed to create NetCDF for {Path(input_file).name}")
                            failed_count, processed_count = failed_count + 1, processed_count - 1
                else:
                    logger.error(f"  ✗ ParaView script failed for {Path(input_file).name}")
                    logger.error(f"    Return Code: {result.returncode}")
                    logger.error("    --- STDOUT ---\n    " + "\n    ".join(result.stdout.splitlines()))
                    logger.error("    --- STDERR ---\n    " + "\n    ".join(result.stderr.splitlines()))
                    failed_count += 1

            finally:
                if os.path.exists(temp_script_path):
                    os.unlink(temp_script_path)
                    
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ Timeout processing {input_file}"); failed_count += 1
        except Exception as e:
            logger.error(f"  ✗ Error: {e}"); failed_count += 1
            if not PROCESSING_OPTIONS['continue_on_error']: break
    
    total_time = datetime.now() - start_time
    logger.info("\n" + "=" * 50 + f"\nBATCH PROCESSING COMPLETE\nProcess ID: {args.process_id or 'main'}\nTotal time: {total_time}"
                f"\nSuccessfully processed: {processed_count} files\nFailed: {failed_count} files")
    
    if processed_count > 0:
        logger.info(f"\nOutput files are in: {output_dir.absolute()}")

if __name__ == "__main__":
    args = parse_arguments()
    if args.suggest_parallel:
        suggest_parallel_ranges(find_files(args), args.num_processes)
    else:
        process_files()
