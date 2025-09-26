#!/bin/bash -l
#SBATCH --job-name=Julia_PVTU_Batch
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=smarras
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10  # Adjust based on your needs
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=4000M

module load bright shared mpich/ge/gcc/64 foss/2024a ParaView Julia

# Navigate to your data directory
cd /scratch/smarras/smarras/output/64x64x24/CompEuler/LESsmago/output/
# Alternative path (uncomment if needed):
#cd /scratch/smarras/smarras/output/64x64x36_5kmX5kmX3km/CompEuler/LESsmago/output/

echo "Starting Julia-based PVTU batch processing..."
echo "Job ID: $SLURM_JOB_ID, CPUs: $SLURM_NTASKS_PER_NODE"
echo "Working directory: $(pwd)"

#=============================================
# CONFIGURATION PARAMETERS
#=============================================
SCRIPT_NAME="batch_analysis_toNetCDF.jl"     # Your Julia script name
FILE_PREFIX="iter_"                          # PVTU file prefix
RESOLUTION=200                               # Grid resolution (100, 200, 400, 800)
SLICE_COORD=0.1                              # Z-coordinate for velocity slices
OUTPUT_DIR="batch_output"                    # Output directory
VARIABLES="w,u,v"                            # Velocity components to process
REYNOLDS_STRESS="uv,uw,vw"                   # Reynolds stress components
EXPORT_FORMAT="netcdf"                       # Export format: netcdf, csv, or both

#=============================================
# ENSURE JULIA PACKAGES ARE INSTALLED
#=============================================
echo "Ensuring Julia packages are installed..."
julia -e 'using Pkg; Pkg.add(["NearestNeighbors", "NCDatasets", "ArgParse", "Dates"])'

# Test NCDatasets installation
echo "Testing NCDatasets installation..."
julia -e 'using NCDatasets; println("NCDatasets loaded successfully - ready for NetCDF export")'

#=============================================
# DRY RUN - Check what files will be processed
#=============================================
echo "Running dry-run to check available files..."
julia $SCRIPT_NAME --dry-run --file-prefix $FILE_PREFIX

#=============================================
# BATCH PROCESSING OPTIONS - Choose one of the following
#=============================================

# OPTION 1: PARALLEL PROCESSING BY FILE RANGES
# Split work across multiple processes for different file ranges
echo "Starting parallel processing with file ranges..."

julia $SCRIPT_NAME \
    --range 100 199 1 \
    --process-id 1 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS \
    --export-format $EXPORT_FORMAT &

julia $SCRIPT_NAME \
    --range 200 299 1 \
    --process-id 2 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS \
    --export-format $EXPORT_FORMAT &

julia $SCRIPT_NAME \
    --range 300 399 1 \
    --process-id 3 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS \
    --export-format $EXPORT_FORMAT &

julia $SCRIPT_NAME \
    --range 400 499 1 \
    --process-id 4 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS \
    --export-format $EXPORT_FORMAT &

julia $SCRIPT_NAME \
    --range 500 600 1 \
    --process-id 5 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS \
    --export-format $EXPORT_FORMAT &
    --range 200 299 1 \
    --process-id 2 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS &

julia $SCRIPT_NAME \
    --range 300 399 1 \
    --process-id 3 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS &

julia $SCRIPT_NAME \
    --range 400 499 1 \
    --process-id 4 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS &

julia $SCRIPT_NAME \
    --range 500 600 1 \
    --process-id 5 \
    --resolution $RESOLUTION \
    --slice-coord $SLICE_COORD \
    --file-prefix $FILE_PREFIX \
    --output-dir $OUTPUT_DIR \
    --variables $VARIABLES \
    --reynolds-stress $REYNOLDS_STRESS &

# OPTION 2: HIGH-RESOLUTION PROCESSING (Uncomment to use)
# Use higher resolution for selected files or variables
# julia $SCRIPT_NAME \
#     --range 100 150 1 \
#     --process-id 1 \
#     --resolution 400 \
#     --slice-coord $SLICE_COORD \
#     --file-prefix $FILE_PREFIX \
#     --output-dir "high_res_output" \
#     --variables "w" \
#     --reynolds-stress "uv" &

# julia $SCRIPT_NAME \
#     --range 151 200 1 \
#     --process-id 2 \
#     --resolution 400 \
#     --slice-coord $SLICE_COORD \
#     --file-prefix $FILE_PREFIX \
#     --output-dir "high_res_output" \
#     --variables "w" \
#     --reynolds-stress "uv" &

# OPTION 3: VARIABLE-SPECIFIC PROCESSING (Uncomment to use)
# Process different variables in parallel for the same file range
# julia $SCRIPT_NAME \
#     --range 100 200 1 \
#     --process-id 1 \
#     --resolution $RESOLUTION \
#     --file-prefix $FILE_PREFIX \
#     --output-dir "var_specific_output" \
#     --variables "u" \
#     --reynolds-stress "" &

# julia $SCRIPT_NAME \
#     --range 100 200 1 \
#     --process-id 2 \
#     --resolution $RESOLUTION \
#     --file-prefix $FILE_PREFIX \
#     --output-dir "var_specific_output" \
#     --variables "v" \
#     --reynolds-stress "" &

# julia $SCRIPT_NAME \
#     --range 100 200 1 \
#     --process-id 3 \
#     --resolution $RESOLUTION \
#     --file-prefix $FILE_PREFIX \
#     --output-dir "var_specific_output" \
#     --variables "w" \
#     --reynolds-stress "uv,uw,vw" &

# OPTION 4: PROCESS ALL FILES WITHOUT RANGES (Uncomment to use)
# Let each process handle all files (useful for small datasets)
# julia $SCRIPT_NAME \
#     --process-id 1 \
#     --resolution $RESOLUTION \
#     --slice-coord $SLICE_COORD \
#     --file-prefix $FILE_PREFIX \
#     --output-dir $OUTPUT_DIR \
#     --variables $VARIABLES \
#     --reynolds-stress $REYNOLDS_STRESS

#=============================================
# WAIT FOR ALL PROCESSES TO COMPLETE
#=============================================
echo "Waiting for all processes to complete..."
wait

#=============================================
# POST-PROCESSING SUMMARY
#=============================================
echo ""
echo "All processes completed! Generating summary..."

# Count output files
if [ -d "$OUTPUT_DIR" ]; then
    total_files=$(find $OUTPUT_DIR -name "*.nc" -o -name "*.csv" | wc -l)
    nc_files=$(find $OUTPUT_DIR -name "*.nc" | wc -l)
    csv_files=$(find $OUTPUT_DIR -name "*.csv" | wc -l)
    
    echo "Total output files generated: $total_files"
    echo "  NetCDF files: $nc_files"
    echo "  CSV files: $csv_files"
    echo ""
    echo "Output directory structure:"
    ls -la $OUTPUT_DIR/
    
    # Show file count per process
    for process_dir in $OUTPUT_DIR/process_*; do
        if [ -d "$process_dir" ]; then
            process_nc_files=$(find "$process_dir" -name "*.nc" | wc -l)
            process_csv_files=$(find "$process_dir" -name "*.csv" | wc -l)
            process_id=$(basename "$process_dir")
            echo "  $process_id: $process_nc_files NetCDF, $process_csv_files CSV files"
            
            # Show sample files from this process
            sample_files=$(find "$process_dir" -name "*.nc" | head -3)
            for sample in $sample_files; do
                echo "    NetCDF: $(basename "$sample")"
            done
        fi
    done
    
    # Show file types generated
    echo ""
    echo "File breakdown:"
    velocity_files=$(find $OUTPUT_DIR -name "*_slice_*.nc" | wc -l)
    reynolds_files=$(find $OUTPUT_DIR -name "*_reynolds_*.nc" | wc -l)
    echo "  Velocity slice NetCDF files: $velocity_files"
    echo "  Reynolds stress NetCDF files: $reynolds_files"
    
else
    echo "Warning: Output directory $OUTPUT_DIR not found!"
fi

# Check for errors in SLURM output
echo ""
echo "Checking SLURM job output for errors..."
if [ -f "${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err" ]; then
    error_size=$(stat -f%z "${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err" 2>/dev/null || stat -c%s "${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err" 2>/dev/null || echo "0")
    if [ "$error_size" -gt 0 ]; then
        echo "SLURM errors detected (check ${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err)"
        echo "Last 10 lines of error file:"
        tail -10 "${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"
    else
        echo "No SLURM errors detected"
    fi
fi

echo ""
echo "================================================"
echo "BATCH PROCESSING SUMMARY"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Completion time: $(date)"
echo "Working directory: $(pwd)"
echo "Configuration used:"
echo "  - Script: $SCRIPT_NAME"
echo "  - File prefix: $FILE_PREFIX"
echo "  - Resolution: $RESOLUTION x $RESOLUTION"
echo "  - Slice coordinate: $SLICE_COORD"
echo "  - Variables: $VARIABLES"
echo "  - Reynolds stress: $REYNOLDS_STRESS"
echo "  - Export format: $EXPORT_FORMAT"
echo "  - Output directory: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    echo "  - Total NetCDF files generated: $nc_files"
    echo "  - Total CSV files generated: $csv_files"
    echo "  - Velocity slice files: $velocity_files"
    echo "  - Reynolds stress files: $reynolds_files"
fi
echo "================================================"

#=============================================
# PYTHON VISUALIZATION (OPTIONAL)
#=============================================
# Uncomment these lines to automatically generate plots from NetCDF files

# echo ""
# echo "Generating Python visualizations from NetCDF files..."
# 
# # Check if Python script exists
# PYTHON_SCRIPT="plot_netcdf.py"
# if [ -f "$PYTHON_SCRIPT" ]; then
#     # Create plots directory
#     PLOTS_DIR="${OUTPUT_DIR}/plots"
#     mkdir -p "$PLOTS_DIR"
#     
#     # Generate individual plots for each NetCDF file
#     echo "Creating individual plots..."
#     python3 "$PYTHON_SCRIPT" "${OUTPUT_DIR}"/process_*/*.nc --output-dir "$PLOTS_DIR"
#     
#     # Generate combined plots if multiple files exist
#     nc_count=$(find "$OUTPUT_DIR" -name "*.nc" | wc -l)
#     if [ "$nc_count" -gt 1 ]; then
#         echo "Creating combined plots..."
#         python3 "$PYTHON_SCRIPT" "${OUTPUT_DIR}"/process_*/iter_*_w_*.nc --output-dir "$PLOTS_DIR" --combined
#         python3 "$PYTHON_SCRIPT" "${OUTPUT_DIR}"/process_*/iter_*_reynolds_*.nc --output-dir "$PLOTS_DIR" --combined
#     fi
#     
#     plot_count=$(find "$PLOTS_DIR" -name "*.png" | wc -l)
#     echo "Generated $plot_count PNG plots in $PLOTS_DIR"
# else
#     echo "Python plotting script not found. NetCDF files can be plotted manually using:"
#     echo "  python3 plot_netcdf.py ${OUTPUT_DIR}/process_*/*.nc --output-dir plots"
# fi

#=============================================
# DATA ANALYSIS SUMMARY (OPTIONAL)
#=============================================
# Uncomment to generate analysis summary

# echo ""
# echo "Generating data analysis summary..."
# SUMMARY_FILE="${OUTPUT_DIR}/analysis_summary_${SLURM_JOB_ID}.txt"
# 
# {
#     echo "PVTU NetCDF Analysis Report - Job $SLURM_JOB_ID"
#     echo "Generated: $(date)"
#     echo "============================================="
#     echo ""
#     echo "Configuration:"
#     echo "  Resolution: $RESOLUTION x $RESOLUTION"
#     echo "  Variables: $VARIABLES"
#     echo "  Reynolds Stress: $REYNOLDS_STRESS"
#     echo "  Export Format: $EXPORT_FORMAT"
#     echo ""
#     echo "Files processed: $total_files total files"
#     echo "  NetCDF files: $nc_files"
#     echo "  CSV files: $csv_files"
#     echo ""
#     echo "File listing:"
#     find $OUTPUT_DIR -type f -name "*.nc" | head -20
#     if [ $(find $OUTPUT_DIR -type f -name "*.nc" | wc -l) -gt 20 ]; then
#         echo "  ... and $(( $(find $OUTPUT_DIR -type f -name "*.nc" | wc -l) - 20 )) more files"
#     fi
# } > "$SUMMARY_FILE"
# 
# echo "Analysis summary saved: $SUMMARY_FILE"

#=============================================
# OPTIONAL: CLEANUP AND ARCHIVAL
#=============================================
# Uncomment these lines if you want to compress outputs or clean up

# echo ""
# echo "Compressing NetCDF files for archival..."
# tar -czf "pvtu_netcdf_results_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M).tar.gz" $OUTPUT_DIR/

# echo "Cleaning up temporary files (uncomment to enable)..."
# # rm -f *.tmp julia_*.log

echo ""
echo "NetCDF files ready for Python analysis!"
echo "Example usage:"
echo "  python3 plot_netcdf.py ${OUTPUT_DIR}/process_*/*.nc --output-dir plots"
echo "  python3 plot_netcdf.py ${OUTPUT_DIR}/process_*/*.nc --analyze"

# Count output files
if [ -d "$OUTPUT_DIR" ]; then
    total_files=$(find $OUTPUT_DIR -name "*.png" | wc -l)
    echo "Total output files generated: $total_files"
    echo "Output directory structure:"
    ls -la $OUTPUT_DIR/
    
    # Show file count per process
    for process_dir in $OUTPUT_DIR/process_*; do
        if [ -d "$process_dir" ]; then
            process_files=$(find "$process_dir" -name "*.png" | wc -l)
            process_id=$(basename "$process_dir")
            echo "  $process_id: $process_files files"
        fi
    done
else
    echo "Warning: Output directory $OUTPUT_DIR not found!"
fi

# Check for any error files
echo ""
echo "Checking for errors in process outputs..."
for i in {1..5}; do
    if [ -f "process_${i}.err" ]; then
        error_size=$(stat -f%z "process_${i}.err" 2>/dev/null || stat -c%s "process_${i}.err")
        if [ $error_size -gt 0 ]; then
            echo "Process $i had errors (check process_${i}.err)"
        else
            echo "Process $i completed successfully"
        fi
    fi
done

echo ""
echo "================================================"
echo "BATCH PROCESSING SUMMARY"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Completion time: $(date)"
echo "Working directory: $(pwd)"
echo "Configuration used:"
echo "  - File prefix: $FILE_PREFIX"
echo "  - Resolution: $RESOLUTION"
echo "  - Slice coordinate: $SLICE_COORD"
echo "  - Output directory: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    echo "  - Total PNG files generated: $total_files"
fi
echo "================================================"

#=============================================
# OPTIONAL: CLEANUP OR ARCHIVAL
#=============================================
# Uncomment these lines if you want to compress outputs or clean up
# echo "Compressing output files..."
# tar -czf "pvtu_analysis_results_${SLURM_JOB_ID}.tar.gz" $OUTPUT_DIR/

# echo "Cleaning up temporary files..."
# rm -f process_*.err process_*.out
