# PVTU Time Averaging Tool

This tool performs time-averaging of parallel VTU (PVTU) files using ParaView's Python interface. It can process large datasets in parallel using MPI.

## Overview

Given a series of PVTU files representing different timesteps:
```
iter_1/          iter_1.pvtu
iter_2/          iter_2.pvtu
iter_3/          iter_3.pvtu
...
iter_N/          iter_N.pvtu
```

The script will:
1. Read all PVTU files and their parallel VTU components
2. Compute the time average of all data fields
3. Output a new PVTU file with the averaged data
4. Use MPI parallelization for efficient processing

## Requirements

- ParaView with Python support (pvpython)
- mpi4py (for parallel execution)
- MPI implementation (e.g., OpenMPI, MPICH)

## Installation

The tool uses ParaView's built-in Python environment (pvpython). Make sure ParaView is installed and `pvpython` is in your PATH.

To check if pvpython is available:
```bash
which pvpython
pvpython --version
```

## Configuration

Edit the "USER SETTINGS" section in `pvtu_time_average_parallel.py`:

```python
# Base directory containing PVTU files
base_directory = "./"

# File pattern for PVTU files (use wildcards)
file_pattern = "iter_*.pvtu"

# Output file name
output_file = "time_averaged.pvtu"

# Optional: Specify iteration range to process
start_index = None  # e.g., 1 or None
end_index = None    # e.g., 100 or None

# Arrays to exclude from averaging
exclude_arrays = ['vtkGhostType', 'GlobalNodeId', 'GlobalElementId']
```

### Configuration Parameters

- **base_directory**: Directory where your PVTU files are located
- **file_pattern**: Glob pattern to match PVTU files (e.g., `"iter_*.pvtu"`, `"output_*.pvtu"`)
- **output_file**: Name of the output PVTU file with time-averaged data
- **start_index**: Optional starting iteration index (set to `None` to start from first file)
- **end_index**: Optional ending iteration index (set to `None` to process until last file)
- **exclude_arrays**: List of array names to exclude from averaging (typically IDs and ghost cell markers)

## Usage

### Serial Execution (Single Core)

For small datasets or testing:

```bash
pvpython pvtu_time_average_parallel.py
```

### Parallel Execution (Recommended)

For large datasets, use MPI parallelization:

```bash
# Using mpiexec
mpiexec -n 8 pvpython pvtu_time_average_parallel.py

# Using mpirun
mpirun -n 8 pvpython pvtu_time_average_parallel.py

# Using srun (on SLURM systems)
srun -n 8 pvpython pvtu_time_average_parallel.py
```

Replace `8` with the number of MPI processes you want to use.

### HPC/SLURM Submission

For running on HPC clusters with SLURM, use the provided submission script:

```bash
sbatch submit_time_average.sh
```

Or create your own submission script (see example below).

## Example Workflows

### Example 1: Average all iterations

```python
base_directory = "/path/to/simulation/output"
file_pattern = "iter_*.pvtu"
output_file = "time_averaged_all.pvtu"
start_index = None
end_index = None
```

```bash
mpiexec -n 16 pvpython pvtu_time_average_parallel.py
```

### Example 2: Average specific range (iterations 10-50)

```python
base_directory = "/path/to/simulation/output"
file_pattern = "iter_*.pvtu"
output_file = "time_averaged_10_50.pvtu"
start_index = 10
end_index = 50
```

```bash
mpiexec -n 8 pvpython pvtu_time_average_parallel.py
```

### Example 3: Running on HPC with SLURM

Create a submission script `submit_time_average.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=time_avg
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --account=your_account

# Load required modules
module purge
module load ParaView/5.11.0
module load mpich/ge/gcc/64

# Run the time averaging script
srun pvpython pvtu_time_average_parallel.py
```

Submit with:
```bash
sbatch submit_time_average.sh
```

## Output

The script generates:
- A PVTU file (e.g., `time_averaged.pvtu`) containing time-averaged data
- Associated VTU files in a subdirectory with the parallel data

All averaged fields will have `_avg` appended to their names:
- `U` → `U_avg`
- `p` → `p_avg`
- `T` → `T_avg`
- etc.

## How It Works

1. **File Discovery**: Scans the directory for PVTU files matching the pattern
2. **Initialization**: Reads the first file to determine data structure and array types
3. **Accumulation**: Iterates through all timesteps, accumulating data:
   - Each MPI rank processes its local portion of the parallel data
   - Data is accumulated in memory (sum of values at each point/cell)
4. **Averaging**: Divides accumulated sums by the number of timesteps
5. **Output**: Writes the averaged data to a new PVTU file

## Performance Tips

1. **Use parallel execution**: The script is designed to run in parallel. Use as many MPI ranks as you have cores available.

2. **Memory considerations**: The script keeps accumulated data in memory. For very large datasets, you may need:
   - Sufficient RAM (roughly 2x the size of one timestep)
   - To process a subset of iterations at a time

3. **I/O optimization**:
   - Run on the same filesystem where the data is stored
   - Use fast scratch storage if available
   - Avoid running over network filesystems when possible

4. **Number of MPI ranks**:
   - Use the same number of ranks as your original simulation (or a divisor)
   - More ranks = less memory per rank but more communication overhead
   - Typical: 4-32 ranks depending on data size

## Troubleshooting

### "No files found matching pattern"
- Check that `base_directory` is correct
- Verify that `file_pattern` matches your file naming convention
- Ensure PVTU files have numeric indices in their names

### "mpi4py not available"
- The script will run in serial mode
- To enable parallel mode, install mpi4py in ParaView's Python:
  ```bash
  /path/to/paraview/bin/python -m pip install mpi4py
  ```

### Memory errors
- Reduce the number of timesteps being averaged
- Use more MPI ranks to distribute the data
- Process in batches (average 1-50, then 51-100, etc.)

### "vtkGhostType" or ID arrays have wrong values
- Add these array names to `exclude_arrays` list
- These arrays should not be averaged

## Technical Details

### Array Handling
- **Point Data**: Arrays defined at mesh vertices
- **Cell Data**: Arrays defined at cell centers
- Both types are averaged independently

### Data Types
- All arrays are converted to `float64` for accumulation
- This prevents overflow and maintains precision
- Output maintains original data structure

### Parallel Strategy
- Uses ParaView's native parallel I/O
- Each MPI rank reads its portion of the PVTU data
- Accumulation happens locally on each rank
- No inter-rank communication needed (embarrassingly parallel)

## Advanced Usage

### Excluding Specific Arrays

Some arrays should not be averaged (IDs, flags, etc.). Add them to the exclusion list:

```python
exclude_arrays = [
    'vtkGhostType',           # Ghost cell markers
    'GlobalNodeId',           # Node IDs
    'GlobalElementId',        # Element IDs
    'GlobalCellId',           # Cell IDs
    'GlobalPointId',          # Point IDs
    'partition_id',           # Partition information
    'processor_id',           # Processor information
]
```

### Processing Multiple Ranges

To compute averages for different time windows:

```bash
# Early time average (iterations 1-20)
# Edit script: start_index=1, end_index=20, output_file="early_avg.pvtu"
mpiexec -n 8 pvpython pvtu_time_average_parallel.py

# Middle time average (iterations 21-40)
# Edit script: start_index=21, end_index=40, output_file="middle_avg.pvtu"
mpiexec -n 8 pvpython pvtu_time_average_parallel.py

# Late time average (iterations 41-60)
# Edit script: start_index=41, end_index=60, output_file="late_avg.pvtu"
mpiexec -n 8 pvpython pvtu_time_average_parallel.py
```

## References

- [ParaView Python Documentation](https://docs.paraview.org/en/latest/ReferenceManual/pythonProgrammableFilter.html)
- [VTK File Formats](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
- [ParaView Parallel Processing](https://docs.paraview.org/en/latest/UsersGuide/parallelDataVisualization.html)

## Support

For issues or questions:
- Check the ParaView documentation
- Review the script's log output for error messages
- Ensure your ParaView version is 5.9 or later

## License

This script is provided as-is for research and educational purposes.
