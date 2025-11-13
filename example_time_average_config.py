#!/usr/bin/env python3
"""
Example configuration file for PVTU time averaging

Copy this file and modify the parameters for your specific case.
Then update the USER SETTINGS section in pvtu_time_average_parallel.py
with these values.
"""

# ==============================================================================
# EXAMPLE 1: Average all iterations in current directory
# ==============================================================================
example1 = {
    'base_directory': './',
    'file_pattern': 'iter_*.pvtu',
    'output_file': 'time_averaged_all.pvtu',
    'start_index': None,
    'end_index': None,
    'exclude_arrays': [
        'vtkGhostType',
        'GlobalNodeId',
        'GlobalElementId',
        'GlobalCellId',
        'GlobalPointId',
    ]
}

# ==============================================================================
# EXAMPLE 2: Average iterations 10-100 in a specific directory
# ==============================================================================
example2 = {
    'base_directory': '/path/to/simulation/output',
    'file_pattern': 'iter_*.pvtu',
    'output_file': 'time_averaged_10_100.pvtu',
    'start_index': 10,
    'end_index': 100,
    'exclude_arrays': [
        'vtkGhostType',
        'GlobalNodeId',
        'GlobalElementId',
    ]
}

# ==============================================================================
# EXAMPLE 3: Average OpenFOAM output files
# ==============================================================================
example3 = {
    'base_directory': './postProcessing/surfaces',
    'file_pattern': 'output_*.pvtu',
    'output_file': 'openfoam_time_averaged.pvtu',
    'start_index': None,
    'end_index': None,
    'exclude_arrays': [
        'vtkGhostType',
        'ProcessId',
    ]
}

# ==============================================================================
# EXAMPLE 4: Average late-time statistics (steady state)
# ==============================================================================
example4 = {
    'base_directory': './',
    'file_pattern': 'timestep_*.pvtu',
    'output_file': 'steady_state_average.pvtu',
    'start_index': 1000,  # Start after transients
    'end_index': 2000,
    'exclude_arrays': [
        'vtkGhostType',
        'GlobalNodeId',
        'GlobalElementId',
        'GlobalCellId',
        'GlobalPointId',
        'partition_id',
    ]
}

# ==============================================================================
# EXAMPLE 5: Multiple averaging windows for turbulence analysis
# ==============================================================================

# Early time window
early_time = {
    'base_directory': './turbulence_data',
    'file_pattern': 'flow_*.pvtu',
    'output_file': 'early_time_average.pvtu',
    'start_index': 1,
    'end_index': 50,
    'exclude_arrays': ['vtkGhostType', 'GlobalNodeId', 'GlobalElementId']
}

# Middle time window
middle_time = {
    'base_directory': './turbulence_data',
    'file_pattern': 'flow_*.pvtu',
    'output_file': 'middle_time_average.pvtu',
    'start_index': 51,
    'end_index': 100,
    'exclude_arrays': ['vtkGhostType', 'GlobalNodeId', 'GlobalElementId']
}

# Late time window
late_time = {
    'base_directory': './turbulence_data',
    'file_pattern': 'flow_*.pvtu',
    'output_file': 'late_time_average.pvtu',
    'start_index': 101,
    'end_index': 150,
    'exclude_arrays': ['vtkGhostType', 'GlobalNodeId', 'GlobalElementId']
}

# ==============================================================================
# USAGE NOTES
# ==============================================================================

"""
To use one of these configurations:

1. Choose an example configuration above (e.g., example1)

2. Open pvtu_time_average_parallel.py

3. Update the USER SETTINGS section with these values:

   base_directory = example1['base_directory']
   file_pattern = example1['file_pattern']
   output_file = example1['output_file']
   start_index = example1['start_index']
   end_index = example1['end_index']
   exclude_arrays = example1['exclude_arrays']

4. Run the script:

   Serial:
   $ pvpython pvtu_time_average_parallel.py

   Parallel (recommended):
   $ mpiexec -n 16 pvpython pvtu_time_average_parallel.py

   HPC/SLURM:
   $ sbatch submit_time_average.sh

COMMON FILE PATTERNS:
- 'iter_*.pvtu'       : matches iter_1.pvtu, iter_2.pvtu, iter_100.pvtu, etc.
- 'output_*.pvtu'     : matches output_1.pvtu, output_2.pvtu, etc.
- 'timestep_*.pvtu'   : matches timestep_1.pvtu, timestep_2.pvtu, etc.
- 'flow_*.pvtu'       : matches flow_1.pvtu, flow_2.pvtu, etc.
- 'result*.pvtu'      : matches result1.pvtu, result100.pvtu, etc.

ARRAY EXCLUSION:
Common arrays to exclude from averaging:
- vtkGhostType       : Ghost cell markers (ParaView internal)
- GlobalNodeId       : Global node IDs
- GlobalElementId    : Global element IDs
- GlobalCellId       : Global cell IDs
- GlobalPointId      : Global point IDs
- partition_id       : MPI partition information
- processor_id       : Processor/rank information
- ProcessId          : Process ID markers

PERFORMANCE TIPS:
- Use as many MPI ranks as you have available cores
- For very large datasets, use more ranks to distribute memory
- Process subsets of time if memory is limited
- Store data on fast local storage (not network FS) if possible
"""

if __name__ == "__main__":
    print("This is a configuration reference file.")
    print("Copy these settings to pvtu_time_average_parallel.py")
    print("\nAvailable examples:")
    print("  - example1: Average all iterations")
    print("  - example2: Average specific range")
    print("  - example3: OpenFOAM output")
    print("  - example4: Steady state averaging")
    print("  - example5: Multiple time windows")
