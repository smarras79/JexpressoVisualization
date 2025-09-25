#!/bin/bash -l
#SBATCH --job-name=ParaView_Parallel
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=high_smarras
#SBATCH --account=smarras
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8  # Number of parallel processes
#SBATCH --time=23:59:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M

# Load required modules
module load bright shared mpich/ge/gcc/64 foss/2024a ParaView

# Change to your data directory
cd /scratch/smarras/smarras/output/64x64x24/CompEuler/LESsmago/output/

# Create nodefile for reference
nodelist=$(scontrol show hostname $SLURM_NODELIST)
printf "%s\n" "${nodelist[@]}" > nodefile

echo "Starting ParaView Parallel Batch Processing"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs requested: $SLURM_NTASKS_PER_NODE"
echo "Start time: $(date)"

# CRITICAL: Configure your actual file ranges here
# First run: python3 batch_paraview_analysis.py --suggest-parallel --num-processes $SLURM_NTASKS_PER_NODE
# to see what ranges to use, then customize below:

# USER CONFIGURATION: Modify these ranges based on your actual files
FILE_RANGES=(
    "100 199 5"    # Process 1: files 100-199, step 5
    "200 299 5"    # Process 2: files 200-299, step 5  
    "300 399 5"    # Process 3: files 300-399, step 5
    "400 499 5"    # Process 4: files 400-499, step 5
    "500 582 2"    # Process 5: files 500-599, step 5
)

# Limit to available CPUs
NUM_PROCESSES=${#FILE_RANGES[@]}
if [ $NUM_PROCESSES -gt $SLURM_NTASKS_PER_NODE ]; then
    NUM_PROCESSES=$SLURM_NTASKS_PER_NODE
fi

echo "Launching $NUM_PROCESSES parallel processes..."
echo "============================================="

# Launch processes in parallel using srun for proper SLURM integration
pids=()
for ((i=0; i<NUM_PROCESSES; i++)); do
    range=(${FILE_RANGES[i]})
    start_file=${range[0]}
    end_file=${range[1]}
    step=${range[2]}
    
    echo "Process $((i+1)): Processing files $start_file to $end_file (step $step)"
    
    # Launch using srun for proper SLURM job control and resource allocation
    srun --ntasks=1 --exclusive --job-name="paraview_proc_$((i+1))" \
         python3 batch_paraview_analysis.py \
         --range $start_file $end_file $step \
         --process-id $((i+1)) \
         --log-level INFO &
    
    pids+=($!)
    
    # Brief delay to avoid filesystem conflicts during startup
    sleep 1
done

echo ""
echo "All $NUM_PROCESSES processes launched with PIDs: ${pids[*]}"
echo "Each process is running on a separate CPU core via srun --exclusive"
echo "Waiting for all processes to complete..."
echo "============================================="

# Monitor progress
start_time=$(date +%s)
completed=0
total_processes=$NUM_PROCESSES

while [ $completed -lt $total_processes ]; do
    sleep 30  # Check every 30 seconds
    
    completed=0
    for pid in "${pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            ((completed++))
        fi
    done
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    echo "Progress: $completed/$total_processes processes completed (${elapsed}s elapsed)"
done

# Final wait to collect exit codes
echo "All processes finished. Collecting exit codes..."
failed_count=0
for i in "${!pids[@]}"; do
    pid=${pids[i]}
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Process $((i+1)) (PID $pid): SUCCESS"
    else
        echo "Process $((i+1)) (PID $pid): FAILED (exit code $exit_code)"
        ((failed_count++))
    fi
done

echo ""
echo "============================================="
echo "PARALLEL PROCESSING COMPLETED!"
echo "End time: $(date)"
echo "Successful processes: $((total_processes - failed_count))/$total_processes"
if [ $failed_count -gt 0 ]; then
    echo "Failed processes: $failed_count"
fi

# Summary
echo ""
echo "Processing Summary:"
echo "==================="

if [ -d "batch_output" ]; then
    total_outputs=$(find batch_output -name "*.png" | wc -l)
    echo "Total PNG files created: $total_outputs"
    echo "Output directory size: $(du -sh batch_output 2>/dev/null | cut -f1)"
    
    # Show sample files
    echo ""
    echo "Sample output files:"
    find batch_output -name "*.png" | sort | head -5
    
    if [ $total_outputs -gt 5 ]; then
        echo "... and $((total_outputs - 5)) more files"
    fi
    
    echo ""
    echo "Files are ready for GIF creation:"
    echo "cd batch_output && convert *.png animation.gif"
else
    echo "No batch_output directory found - check for errors"
fi

# Log file summary
echo ""
echo "Individual process logs:"
for log_file in batch_processing_proc_*.log; do
    if [ -f "$log_file" ]; then
        echo "  $log_file: $(wc -l < "$log_file") lines"
    fi
done

echo ""
echo "============================================="
echo "Job completed! Check batch_output/ for PNG files."
