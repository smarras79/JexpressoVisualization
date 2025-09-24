#!/usr/bin/env python3
"""

TO RUN IT:

python3.10 batch_paraview_analysis.py

Working ParaView Batch Processor with subprocess isolation

This approach runs each file in a separate ParaView process to completely
avoid state conflicts between files. Each file gets a fresh ParaView session.
"""

import subprocess
import os
import sys
import glob
import re
import tempfile
import logging
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
    'number_range': None,  # None for auto-detection, or [start, end, step]
}

# --- Analysis Configuration ---
# 
# List of possible symbols that you may have in your VTU so that you can paste them here:
# 
# ρ
# θ
#
BATCH_CONFIG = {
    #'data_array': 'VELOMAG',
    'data_array': 'θ',
    'analysis_mode': 'slicing',  # 'averaging' or 'slicing'
    
    'averaging': {
        'axis': 'Y',
        'resolution': [150, 150, 150],
        'geometric_limits': {
            'X': [None, None],
            'Y': [0.1, 10.0],
            'Z': [None, None]
        }
    },
    
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
    'paraview_executable': '/Applications/ParaView-5.11.2.app/Contents/bin/pvpython',
    'timeout_seconds': 300,  # 5 minutes per file
    'log_file': 'batch_processing.log'
}

#==============================================================================
# IMPLEMENTATION
#==============================================================================

def setup_logging():
    """Setup logging for the batch processor."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROCESSING_OPTIONS['log_file']),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

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

def find_files():
    """Find and sort files matching the pattern."""
    base_dir = Path(FILE_PATTERNS['base_directory'])
    template = FILE_PATTERNS['file_template']
    number_range = FILE_PATTERNS.get('number_range', None)
    
    if number_range is not None:
        # Use specified range
        files = []
        start, end, step = number_range
        current = start
        while current <= end:
            filename = base_dir / template.format(current)
            if filename.exists():
                files.append(str(filename.absolute()))
            current += step
        return sorted(files)
    
    else:
        # Auto-detect files
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

def generate_output_filename(input_file):
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
    output_dir = Path(PROCESSING_OPTIONS['output_directory'])
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
            print("Averaging analysis not implemented in subprocess version")
            return False
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
        
        # Set camera based on slice axis
        axis = "{BATCH_CONFIG['slicing']['axis']}".upper()
        if axis == 'X':
            view.CameraPosition = [1, 0, 0]
            view.CameraViewUp = [0, 0, 1]
        elif axis == 'Y':
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

def process_files():
    """Process all files using subprocess isolation."""
    
    logger = setup_logging()
    
    print("ParaView Subprocess Batch Processor")
    print("=" * 50)
    print(f"Data array: {BATCH_CONFIG['data_array']}")
    print(f"Analysis mode: {BATCH_CONFIG['analysis_mode']}")
    print(f"File pattern: {FILE_PATTERNS['file_template']}")
    print("=" * 50)
    
    # Find files
    files = find_files()
    if not files:
        logger.error("No files found to process")
        return
    
    logger.info(f"Found {len(files)} files to process")
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
            output_file = generate_output_filename(input_file)
            logger.info(f"  Output: {Path(output_file).name}")
            
            # Generate processing script
            script_content = generate_processing_script(input_file, output_file)
            
            # Write temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                temp_script.write(script_content)
                temp_script_path = temp_script.name
            
            try:
                # Run subprocess
                logger.info("  Starting ParaView subprocess...")
                result = subprocess.run([
                    PROCESSING_OPTIONS['paraview_executable'],
                    temp_script_path
                ], capture_output=True, text=True, timeout=PROCESSING_OPTIONS['timeout_seconds'])
                
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
                    if result.stdout:
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
    process_files()
