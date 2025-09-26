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
python3 batch_analysis_toParaview_netCDF.py --range 100 200 2 --output-format both

# Create only NetCDF files
python3 batch_analysis_toParaview_netCDF.py --range 100 200 2 --output-format netcdf --analysis-mode averaging

# Suggest parallel processing ranges
python3 batch_analysis_toParaview_netCDF.py --suggest-parallel --num-processes 4
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
import traceback

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
    'data_array': 'w',
    'analysis_mode': 'slicing',  # 'averaging' or 'slicing'
    'output_format': ['png', 'netcdf'],
    'netcdf_grid_resolution': 512,

    'averaging': {
        'axis': 'Y',
        'resolution': [150, 150, 150],
        'geometric_limits': {'X': [None, None], 'Y': [4500.0, 5500.0], 'Z': [None, None]}
    },
    'slicing': {
        'axis': 'Z',
        'coordinate': 100
    },
    'visualization': {
        'image_size': [1200, 800],
        'color_map': 'Blues',
    }
}

# --- Processing Options ---
PROCESSING_OPTIONS = {
    'output_directory': './batch_output/',
    'temp_directory': './batch_temp/',
    'continue_on_error': True,
    'paraview_executable': '/Applications/ParaView-5.11.2.app/Contents/bin/pvpython',
    'paraview_args': ['--force-offscreen-rendering'],
    'timeout_seconds': 300,
    'log_file_prefix': 'batch_processing',
}

#==============================================================================
# (Argument Parsing and other helper functions remain unchanged)
#==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description='ParaView Batch Processor', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--range', nargs=3, type=int, metavar=('START', 'END', 'STEP'), help='Range of file numbers to process.')
    parser.add_argument('--files', nargs='+', type=str, help='A specific list of files to process.')
    parser.add_argument('--process-id', type=str, default=None, help='Identifier for logging.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if output file already exists.')
    parser.add_argument('--output-dir', type=str, default=None, help='Override the output directory.')
    parser.add_argument('--suggest-parallel', action='store_true', help='Suggest parallel command ranges.')
    parser.add_argument('--num-processes', type=int, default=4, help='Number of processes for parallel suggestion.')
    parser.add_argument('--output-format', type=str, choices=['png', 'netcdf', 'both'], default=None, help='Override output format.')
    parser.add_argument('--analysis-mode', type=str, choices=['slicing', 'averaging'], default=None, help='Override analysis mode.')
    return parser.parse_args()

def setup_logging(process_id=None, log_level='INFO'):
    log_dir = Path(PROCESSING_OPTIONS['output_directory']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{PROCESSING_OPTIONS['log_file_prefix']}_{process_id or 'main'}_{timestamp}.log"
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def extract_number_from_filename(filepath):
    match = re.search(r'(\d+\.?\d*)', os.path.basename(filepath))
    try: return int(match.group(1)) if match else 0
    except ValueError: return float(match.group(1))

def find_files(args):
    logger = logging.getLogger(__name__)
    base_dir = Path(FILE_PATTERNS['base_directory'])
    all_files = sorted(glob.glob(str(base_dir / FILE_PATTERNS['file_template'].format('*'))), key=extract_number_from_filename)
    if not all_files:
        logger.error(f"No files found with pattern '{FILE_PATTERNS['file_template']}' in '{base_dir.absolute()}'"); return []
    if args.range:
        start, end, step = args.range
        files = [f for f in all_files if start <= extract_number_from_filename(f) <= end and (extract_number_from_filename(f) - start) % step == 0]
        logger.info(f"Found {len(files)} files in range {start}-{end} (step {step}).")
        return files
    logger.info(f"Processing all {len(all_files)} found files.")
    return all_files

def generate_output_basename(input_file, output_dir):
    num = extract_number_from_filename(input_file)
    num_str = f"{num:06d}" if isinstance(num, int) else f"{num:08.3f}".replace('.', '_')
    mode = BATCH_CONFIG['analysis_mode'][:3]
    axis = BATCH_CONFIG[BATCH_CONFIG['analysis_mode']]['axis']
    return str(output_dir / f"{BATCH_CONFIG['data_array']}_{mode}_{axis}_{num_str}")

#==============================================================================

def generate_processing_script(input_file, png_output_file, temp_dir):
    """Generate a standalone script to process a single file."""
    input_abs = os.path.abspath(input_file)
    png_output_abs = os.path.abspath(png_output_file) if png_output_file else "None"
    temp_npz_file = os.path.join(os.path.abspath(temp_dir), f"temp_data_{Path(input_file).stem}.npz")

    save_png = "'png' in BATCH_CONFIG['output_format']"
    save_npz = "'netcdf' in BATCH_CONFIG['output_format']"

    script_content = f"""#!/usr/bin/env python3
import sys, os, traceback
from paraview.simple import *
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa
import vtk

BATCH_CONFIG = {BATCH_CONFIG}

def main():
    try:
        source = OpenDataFile(r'{input_abs}')
        if not source:
            print(f"ERROR: Failed to open file r'{input_abs}'"); return False
        
        # --- ANALYSIS ---
        if BATCH_CONFIG['analysis_mode'] == 'averaging':
            print("Mode: Averaging (resampling only)")
            resample = ResampleToImage(Input=source)
            resample.SamplingDimensions = BATCH_CONFIG['averaging']['resolution']
            bounds = source.GetDataInformation().GetBounds()
            geo = BATCH_CONFIG['averaging']['geometric_limits']
            resample.SamplingBounds = [
                geo['X'][0] or bounds[0], geo['X'][1] or bounds[1],
                geo['Y'][0] or bounds[2], geo['Y'][1] or bounds[3],
                geo['Z'][0] or bounds[4], geo['Z'][1] or bounds[5]
            ]
            source = resample
        
        elif BATCH_CONFIG['analysis_mode'] == 'slicing':
            print("Mode: Slicing")
            s = BATCH_CONFIG['slicing']
            slice_op = Slice(Input=source, SliceType='Plane')
            axis, coord = s['axis'].upper(), s['coordinate']
            normals = {{'X': [1,0,0], 'Y': [0,1,0], 'Z': [0,0,1]}}
            slice_op.SliceType.Normal = normals[axis]
            if coord is None:
                slice_op.SliceType.Origin = source.GetDataInformation().GetCenter()
            else:
                origin = [0,0,0]; origin[('X','Y','Z').index(axis)] = coord
                slice_op.SliceType.Origin = origin
            source = slice_op

        # --- Get array name ---
        all_arrays = source.GetPointDataInformation()
        array_name_to_use = BATCH_CONFIG['data_array']
        if not all_arrays.GetArray(array_name_to_use):
             found_array = all_arrays.GetArray(0).Name if all_arrays.GetNumberOfArrays() > 0 else None
             if not found_array: print("ERROR: No data arrays found."); return False
             print(f"WARN: '{{array_name_to_use}}' not found. Using '{{found_array}}'.")
             array_name_to_use = found_array
        
        # --- OUTPUT GENERATION ---
        if {save_png}:
            render_source = Slice(Input=source, SliceType='Plane', SliceOffsetValues=[0.0]) if BATCH_CONFIG['analysis_mode'] == 'averaging' else source
            view = GetActiveViewOrCreate('RenderView')
            display = Show(render_source, view)
            ColorBy(display, ('POINTS', array_name_to_use))
            GetColorTransferFunction(array_name_to_use).ApplyPreset(BATCH_CONFIG['visualization']['color_map'], True)
            GetScalarBar(GetColorTransferFunction(array_name_to_use), view).Visibility = 1
            view.ResetCamera(); Render()
            SaveScreenshot(r"{png_output_abs}", view, ImageResolution=BATCH_CONFIG['visualization']['image_size'])
            print("PNG file created successfully.")

        if {save_npz}:
            print("Extracting data for intermediate file...")
            vtk_data = servermanager.Fetch(source)
            wrapped_data = dsa.WrapDataObject(vtk_data)
            values = wrapped_data.PointData[array_name_to_use]
            
            # --- MODIFIED PART ---
            save_args = {{'values': values, 'array_name': array_name_to_use}}
            if isinstance(vtk_data, vtk.vtkImageData):
                print("  Saving structured data (ImageData)")
                save_args.update({{'dims': vtk_data.GetDimensions(), 'origin': vtk_data.GetOrigin(), 
                                   'spacing': vtk_data.GetSpacing(), 'is_image_data': True}})
            else:
                print("  Saving unstructured data")
                save_args.update({{'points': wrapped_data.Points, 'is_image_data': False}})
            
            np.savez_compressed(r"{temp_npz_file}", **save_args)
            print(f"INTERMEDIATE_FILE:{{r'{temp_npz_file}'}}")
        
        return True
    except Exception as e:
        print(f"ERROR in ParaView script: {{e}}\\n{{traceback.format_exc()}}"); return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
"""
    return script_content

def create_netcdf_from_npz(npz_path, nc_path):
    """Reads NPZ, processes data, and saves as NetCDF."""
    try:
        data = np.load(npz_path)
        # --- CORRECTED PART ---
        array_name = str(data['array_name'])
        
        if BATCH_CONFIG['analysis_mode'] == 'averaging' and data['is_image_data']:
            dims, origin, spacing = data['dims'], data['origin'], data['spacing']
            values_3d = data['values'].reshape(dims[2], dims[1], dims[0]) # VTK order is Z, Y, X
            
            axis_map = {'X': 2, 'Y': 1, 'Z': 0}
            avg_axis_idx = axis_map[BATCH_CONFIG['averaging']['axis'].upper()]
            averaged_array = np.mean(values_3d, axis=avg_axis_idx)
            
            if avg_axis_idx == 2: # Avg over X -> ZY plane
                coords1, coords2 = origin[2] + np.arange(dims[2]) * spacing[2], origin[1] + np.arange(dims[1]) * spacing[1]
                ds = xr.Dataset({array_name: (('z', 'y'), averaged_array)}, coords={'z': coords1, 'y': coords2})
            elif avg_axis_idx == 1: # Avg over Y -> ZX plane
                coords1, coords2 = origin[2] + np.arange(dims[2]) * spacing[2], origin[0] + np.arange(dims[0]) * spacing[0]
                ds = xr.Dataset({array_name: (('z', 'x'), averaged_array)}, coords={'z': coords1, 'x': coords2})
            else: # Avg over Z -> YX plane
                coords1, coords2 = origin[1] + np.arange(dims[1]) * spacing[1], origin[0] + np.arange(dims[0]) * spacing[0]
                ds = xr.Dataset({array_name: (('y', 'x'), averaged_array)}, coords={'y': coords1, 'x': coords2})

        else: # Slicing mode
            points_3d, values = data['points'], data['values']
            axis_map = {'X': (1,2), 'Y': (0,2), 'Z': (0,1)}
            dims_map = {'X': ('y','z'), 'Y': ('x','z'), 'Z': ('x','y')}
            slice_axis = BATCH_CONFIG['slicing']['axis'].upper()
            
            points_2d = points_3d[:, list(axis_map[slice_axis])]
            dim_names = dims_map[slice_axis]

            min1, max1 = points_2d[:, 0].min(), points_2d[:, 0].max()
            min2, max2 = points_2d[:, 1].min(), points_2d[:, 1].max()
            res = BATCH_CONFIG['netcdf_grid_resolution']
            n1 = res if (max1 - min1) >= (max2 - min2) else int(res * (max1 - min1) / (max2 - min2))
            n2 = res if (max2 - min2) > (max1 - min1) else int(res * (max2 - min2) / (max1 - min1))

            grid_x, grid_y = np.meshgrid(np.linspace(min1, max1, n1), np.linspace(min2, max2, n2))
            interpolated_values = griddata(points_2d, values, (grid_x, grid_y), method='linear')
            
            ds = xr.Dataset({array_name: (dim_names, interpolated_values)},
                            coords={dim_names[0]: grid_x[0,:], dim_names[1]: grid_y[:,0]})

        ds.to_netcdf(nc_path)
        return True

    except Exception as e:
        logging.getLogger(__name__).error(f"  Failed to create NetCDF: {e}\n{traceback.format_exc()}")
        return False
    finally:
        if os.path.exists(npz_path): os.remove(npz_path)

def process_files():
    args = parse_arguments()
    logger = setup_logging(args.process_id, args.log_level)
    
    if args.output_dir: PROCESSING_OPTIONS['output_directory'] = args.output_dir
    if args.output_format: BATCH_CONFIG['output_format'] = ['png', 'netcdf'] if args.output_format == 'both' else [args.output_format]
    if args.analysis_mode: BATCH_CONFIG['analysis_mode'] = args.analysis_mode

    logger.info(f"Starting batch processing... Mode: {BATCH_CONFIG['analysis_mode'].upper()}")
    
    files = find_files(args)
    if not files: logger.warning("No files to process. Exiting."); return
    
    output_dir = Path(PROCESSING_OPTIONS['output_directory']); output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(PROCESSING_OPTIONS['temp_directory']); temp_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now() # Corrected initialization
    processed_count, failed_count = 0, 0
    for i, input_file in enumerate(files, 1):
        logger.info(f"Processing file {i}/{len(files)}: {Path(input_file).name}")
        try:
            output_basename = generate_output_basename(input_file, output_dir)
            png_file, nc_file = f"{output_basename}.png", f"{output_basename}.nc"
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as f:
                f.write(generate_processing_script(input_file, png_file, temp_dir)); temp_script_path = f.name
            
            try:
                cmd = [PROCESSING_OPTIONS['paraview_executable'], *PROCESSING_OPTIONS['paraview_args'], temp_script_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=PROCESSING_OPTIONS['timeout_seconds'])
                
                if result.returncode == 0 and "ERROR" not in result.stdout:
                    logger.info("  ✓ ParaView script finished successfully.")
                    intermediate_file = next((line.split(":", 1)[1] for line in result.stdout.splitlines() if line.startswith("INTERMEDIATE_FILE:")), None)
                    if 'netcdf' in BATCH_CONFIG['output_format']:
                        if intermediate_file and os.path.exists(intermediate_file):
                            if create_netcdf_from_npz(intermediate_file, nc_file):
                                logger.info(f"    ✓ NetCDF created: {Path(nc_file).name}")
                            else:
                                logger.error(f"    ✗ Failed to create NetCDF for {Path(input_file).name}"); failed_count+=1
                        else:
                            logger.error("    ✗ Intermediate file for NetCDF not found."); failed_count+=1
                    processed_count += 1
                else:
                    logger.error(f"  ✗ ParaView script failed for {Path(input_file).name}\n"
                                 f"    --- STDOUT ---\n    " + "\n    ".join(result.stdout.splitlines()) + "\n"
                                 f"    --- STDERR ---\n    " + "\n    ".join(result.stderr.splitlines())); failed_count += 1
            finally:
                if os.path.exists(temp_script_path): os.unlink(temp_script_path)
        except Exception as e:
            logger.error(f"  ✗ Main script error: {e}\n{traceback.format_exc()}"); failed_count += 1
            if not PROCESSING_OPTIONS['continue_on_error']: break

    total_time = datetime.now() - start_time
    logger.info(f"\nBATCH PROCESSING COMPLETE\nTotal time: {total_time}\nSuccess: {processed_count}, Failed: {failed_count}")


if __name__ == "__main__":
    process_files()
