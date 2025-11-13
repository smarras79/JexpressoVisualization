#!/usr/bin/env pvpython
"""
Time-average PVTU files in parallel using ParaView/pvpython

This script reads a series of PVTU files (e.g., iter_1.pvtu, iter_2.pvtu, ..., iter_N.pvtu)
and computes the time average of all data fields across all timesteps.

Usage:
    # Serial execution:
    pvpython pvtu_time_average_parallel.py

    # Parallel execution (recommended):
    mpiexec -n <num_procs> pvpython pvtu_time_average_parallel.py

Configuration:
    Edit the USER SETTINGS section below to specify:
    - base_directory: Directory containing PVTU files and subdirectories
    - file_pattern: Pattern to match PVTU files (e.g., "iter_*.pvtu")
    - output_file: Name of output file for time-averaged data
    - start_index: Starting iteration index (optional)
    - end_index: Ending iteration index (optional)
"""

import os
import sys
import glob
import re
import numpy as np
from paraview.simple import *
from paraview import servermanager as sm
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util import numpy_support as ns

# Try to import MPI for parallel processing
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    HAS_MPI = True
except ImportError:
    rank = 0
    size = 1
    HAS_MPI = False
    print("[WARNING] mpi4py not available. Running in serial mode.", flush=True)

# ============================================================================
# USER SETTINGS - Modify these parameters for your case
# ============================================================================

# Base directory containing PVTU files
base_directory = "./"

# File pattern for PVTU files (use wildcards)
# Examples: "iter_*.pvtu", "output_*.pvtu", "timestep_*.pvtu"
file_pattern = "iter_*.pvtu"

# Output file name (will be saved in base_directory)
output_file = "time_averaged.pvtu"

# Optional: Specify iteration range to process
# Set to None to process all files matching the pattern
start_index = None  # e.g., 1 or None
end_index = None    # e.g., 100 or None

# Arrays to exclude from averaging (typically index/id fields)
# Add any array names you don't want to average
exclude_arrays = ['vtkGhostType', 'GlobalNodeId', 'GlobalElementId',
                  'GlobalCellId', 'GlobalPointId']

# ============================================================================
# END USER SETTINGS
# ============================================================================


def print_log(msg, root_only=True):
    """Print log message (optionally only from root rank)"""
    if not root_only or rank == 0:
        print(f"[Rank {rank}] {msg}", flush=True)


def find_pvtu_files(base_dir, pattern, start_idx=None, end_idx=None):
    """
    Find and sort PVTU files matching the pattern

    Returns:
        List of (index, filepath) tuples sorted by index
    """
    search_path = os.path.join(base_dir, pattern)
    files = glob.glob(search_path)

    if not files:
        raise RuntimeError(f"No files found matching pattern: {search_path}")

    # Extract numeric indices from filenames
    indexed_files = []
    for fpath in files:
        fname = os.path.basename(fpath)
        # Try to extract number from filename
        match = re.search(r'(\d+)', fname)
        if match:
            idx = int(match.group(1))
            if start_idx is not None and idx < start_idx:
                continue
            if end_idx is not None and idx > end_idx:
                continue
            indexed_files.append((idx, fpath))

    if not indexed_files:
        raise RuntimeError("No files found with numeric indices in specified range")

    # Sort by index
    indexed_files.sort(key=lambda x: x[0])

    return indexed_files


def read_pvtu_file(filepath):
    """Read a PVTU file using XMLPartitionedUnstructuredGridReader"""
    print_log(f"Reading: {filepath}", root_only=False)

    reader = XMLPartitionedUnstructuredGridReader(FileName=filepath)
    reader.UpdatePipeline()

    return reader


def get_array_info(source):
    """Get information about available point and cell arrays"""
    data_info = source.GetDataInformation()

    point_arrays = {}
    cell_arrays = {}

    # Point data
    pd_info = data_info.GetPointDataInformation()
    for i in range(pd_info.GetNumberOfArrays()):
        arr_info = pd_info.GetArrayInformation(i)
        name = arr_info.GetName()
        n_components = arr_info.GetNumberOfComponents()
        point_arrays[name] = n_components

    # Cell data
    cd_info = data_info.GetCellDataInformation()
    for i in range(cd_info.GetNumberOfArrays()):
        arr_info = cd_info.GetArrayInformation(i)
        name = arr_info.GetName()
        n_components = arr_info.GetNumberOfComponents()
        cell_arrays[name] = n_components

    return point_arrays, cell_arrays


def initialize_accumulators(source, exclude_list):
    """
    Initialize accumulator dictionaries for time averaging

    Returns:
        point_accum, cell_accum: Dictionaries mapping array names to None
    """
    point_arrays, cell_arrays = get_array_info(source)

    # Filter out excluded arrays
    point_accum = {name: None for name in point_arrays.keys()
                   if name not in exclude_list}
    cell_accum = {name: None for name in cell_arrays.keys()
                  if name not in exclude_list}

    return point_accum, cell_accum


def accumulate_data(source, point_accum, cell_accum):
    """
    Add current timestep data to accumulators

    This function runs on each MPI rank with its local portion of the data
    """
    # Fetch the actual data on this rank
    data_obj = sm.Fetch(source)

    if data_obj is None:
        print_log("Warning: Fetch returned None", root_only=False)
        return

    # Handle multiblock datasets (common with parallel PVTU)
    from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet, vtkCompositeDataSet

    blocks = []
    if isinstance(data_obj, vtkCompositeDataSet):
        # Extract all blocks
        iterator = data_obj.NewIterator()
        iterator.UnRegister(None)
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            block = data_obj.GetDataSet(iterator)
            if block:
                blocks.append(block)
            iterator.GoToNextItem()
    else:
        blocks = [data_obj]

    # Process each block
    for block in blocks:
        wrapped = dsa.WrapDataObject(block)

        # Accumulate point data
        for name in point_accum.keys():
            if name in wrapped.PointData.keys():
                arr = np.asarray(wrapped.PointData[name], dtype=np.float64)

                if point_accum[name] is None:
                    # First timestep: initialize
                    point_accum[name] = arr.copy()
                else:
                    # Subsequent timesteps: add
                    point_accum[name] += arr

        # Accumulate cell data
        for name in cell_accum.keys():
            if name in wrapped.CellData.keys():
                arr = np.asarray(wrapped.CellData[name], dtype=np.float64)

                if cell_accum[name] is None:
                    # First timestep: initialize
                    cell_accum[name] = arr.copy()
                else:
                    # Subsequent timesteps: add
                    cell_accum[name] += arr


def compute_time_average(point_accum, cell_accum, n_timesteps):
    """Divide accumulated sums by number of timesteps"""
    for name in point_accum.keys():
        if point_accum[name] is not None:
            point_accum[name] /= float(n_timesteps)

    for name in cell_accum.keys():
        if cell_accum[name] is not None:
            cell_accum[name] /= float(n_timesteps)


def create_averaged_output(template_source, point_accum, cell_accum):
    """
    Create output with time-averaged data by directly modifying the VTK data object
    """
    # Fetch the actual data from the template source
    # This gets the data local to this MPI rank
    data_obj = sm.Fetch(template_source)

    if data_obj is None:
        print_log("Warning: Could not fetch data for output creation", root_only=False)
        return template_source

    # Handle multiblock datasets
    from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet, vtkCompositeDataSet

    if isinstance(data_obj, vtkCompositeDataSet):
        # For composite datasets, add averaged arrays to each block
        iterator = data_obj.NewIterator()
        iterator.UnRegister(None)
        iterator.InitTraversal()

        while not iterator.IsDoneWithTraversal():
            block = data_obj.GetDataSet(iterator)
            if block:
                # Add point arrays
                for name, data in point_accum.items():
                    if data is not None:
                        vtk_arr = ns.numpy_to_vtk(data, deep=1)
                        vtk_arr.SetName(name + "_avg")
                        block.GetPointData().AddArray(vtk_arr)

                # Add cell arrays
                for name, data in cell_accum.items():
                    if data is not None:
                        vtk_arr = ns.numpy_to_vtk(data, deep=1)
                        vtk_arr.SetName(name + "_avg")
                        block.GetCellData().AddArray(vtk_arr)

            iterator.GoToNextItem()
    else:
        # Single block dataset
        for name, data in point_accum.items():
            if data is not None:
                vtk_arr = ns.numpy_to_vtk(data, deep=1)
                vtk_arr.SetName(name + "_avg")
                data_obj.GetPointData().AddArray(vtk_arr)

        for name, data in cell_accum.items():
            if data is not None:
                vtk_arr = ns.numpy_to_vtk(data, deep=1)
                vtk_arr.SetName(name + "_avg")
                data_obj.GetCellData().AddArray(vtk_arr)

    # Now we need to create a TrivialProducer to wrap this modified data
    # so we can use it with SaveData
    from paraview.simple import TrivialProducer
    producer = TrivialProducer()
    producer.GetClientSideObject().SetOutput(data_obj)
    producer.UpdatePipeline()

    return producer


def main():
    print_log("=" * 80)
    print_log("PVTU Time Averaging Tool (Parallel Version)")
    print_log("=" * 80)
    print_log(f"Running on {size} MPI rank(s)")
    print_log("")

    # Find input files
    print_log("Finding PVTU files...")
    indexed_files = find_pvtu_files(base_directory, file_pattern,
                                     start_index, end_index)

    n_files = len(indexed_files)
    print_log(f"Found {n_files} PVTU files to process:")
    if rank == 0:
        for idx, fpath in indexed_files[:5]:  # Show first 5
            print_log(f"  [{idx}] {os.path.basename(fpath)}")
        if n_files > 5:
            print_log(f"  ... and {n_files - 5} more files")
    print_log("")

    if n_files == 0:
        print_log("ERROR: No files found to process!")
        return 1

    # Read first file to get array structure
    print_log("Initializing from first file...")
    first_reader = read_pvtu_file(indexed_files[0][1])

    # Initialize accumulators
    point_accum, cell_accum = initialize_accumulators(first_reader, exclude_arrays)

    n_point_arrays = len(point_accum)
    n_cell_arrays = len(cell_accum)
    print_log(f"Will average {n_point_arrays} point array(s) and {n_cell_arrays} cell array(s)")
    if rank == 0:
        print_log(f"Point arrays: {list(point_accum.keys())}")
        print_log(f"Cell arrays: {list(cell_accum.keys())}")
    print_log("")

    # Time average loop
    print_log("Computing time average...")
    for i, (idx, filepath) in enumerate(indexed_files):
        print_log(f"Processing timestep {i+1}/{n_files} (index={idx})", root_only=False)

        if i == 0:
            # Already read first file
            reader = first_reader
        else:
            reader = read_pvtu_file(filepath)

        # Accumulate data from this timestep
        accumulate_data(reader, point_accum, cell_accum)

        # Clean up reader (except the first one, we'll use it as template)
        if i > 0:
            Delete(reader)

    print_log("")
    print_log("Computing averages...")
    compute_time_average(point_accum, cell_accum, n_files)

    # Create output with averaged data
    print_log("Creating output dataset...")
    averaged_source = create_averaged_output(first_reader, point_accum, cell_accum)

    # Write output
    output_path = os.path.join(base_directory, output_file)
    print_log(f"Writing output to: {output_path}")

    # Use SaveData which handles parallel writing automatically
    SaveData(output_path, proxy=averaged_source)

    print_log("")
    print_log("=" * 80)
    print_log("Time averaging complete!")
    print_log(f"Output written to: {output_path}")

    # List the created files
    if rank == 0:
        base_name = os.path.splitext(output_file)[0]
        vtu_pattern = os.path.join(base_directory, f"{base_name}*.vtu")
        vtu_files = glob.glob(vtu_pattern)
        if vtu_files:
            print_log(f"Created {len(vtu_files)} VTU piece file(s)")

    print_log("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print_log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
