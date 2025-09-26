import os
import glob
import re
import numpy as np
import pyvista as pv
import xarray as xr
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import griddata
from itertools import combinations_with_replacement

def calculate_and_save_netcdf(
    data_directory: str,
    file_pattern: str,
    output_filename: str = "reynolds_stresses.nc",
    variables: list = ['u', 'v', 'w'],
    coord_precision: int = 6,
    base_resolution: int = 256,
    start_step: int = None,
    end_step: int = None
):
    """
    Calculates Reynolds stresses, skipping any unreadable files, and saves the
    results to a NetCDF file after robust interpolation.
    """
    # --- 1. FILE DISCOVERY AND FILTERING (No changes) ---
    print("🔍 Finding and filtering data files...")
    all_pvtu_files = sorted(glob.glob(os.path.join(data_directory, file_pattern)))
    if not all_pvtu_files:
        raise FileNotFoundError(f"No files found for pattern '{file_pattern}' in '{data_directory}'")

    if start_step is not None or end_step is not None:
        filtered_files = []
        pattern = re.compile(r'(\d+)\.pvtu$')
        for fpath in all_pvtu_files:
            match = pattern.search(os.path.basename(fpath))
            if match:
                step = int(match.group(1))
                in_range = True
                if start_step is not None and step < start_step: in_range = False
                if end_step is not None and step > end_step: in_range = False
                if in_range: filtered_files.append(fpath)
        pvtu_files = filtered_files
    else:
        pvtu_files = all_pvtu_files
    
    if not pvtu_files:
        raise FileNotFoundError(f"No files found in the specified range [{start_step}-{end_step}]")
    
    print(f"Found {len(pvtu_files)} time steps to process.")

    first_mesh = pv.read(pvtu_files[0])
    points_3d = first_mesh.points
    xz_coords = np.round(points_3d[:, [0, 2]], decimals=coord_precision)
    unique_xz, inverse_indices = np.unique(xz_coords, axis=0, return_inverse=True)
    num_xz_points = len(unique_xz)
    print(f"Identified {num_xz_points} unique (x, z) points for averaging.")

    # --- 2. PERFORM TWO-PASS CALCULATION (MODIFIED FOR ROBUSTNESS) ---
    print("\n--- Pass 1 of 2: Calculating sums for mean velocities ---")
    sums = {var: np.zeros(num_xz_points, dtype=np.float64) for var in variables}
    per_location_counts = np.zeros(num_xz_points, dtype=np.int64)
    
    # ### --- NEW: ADD A COUNTER FOR SUCCESFULLY PROCESSED FILES --- ###
    successful_file_count = 0

    for fpath in tqdm(pvtu_files, desc="Processing files for means"):
        # ### --- NEW: TRY/EXCEPT BLOCK TO SKIP BAD FILES --- ###
        try:
            mesh = pv.read(fpath)
            # This check ensures the mesh has points and the expected data
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in variables):
                raise ValueError("Mesh is empty or missing required variables.")
            
            for var in variables:
                np.add.at(sums[var], inverse_indices, mesh.point_data[var])
            
            # Count y-points only on the first successful read
            if successful_file_count == 0:
                np.add.at(per_location_counts, inverse_indices, 1)
            
            successful_file_count += 1
            
        except Exception as e:
            # Print a warning and continue to the next file
            tqdm.write(f"\n⚠️ WARNING: Skipping file '{os.path.basename(fpath)}' due to error: {e}")
            continue

    if successful_file_count == 0:
        raise RuntimeError("Could not successfully read any files. Aborting.")
        
    # ### --- MODIFIED: CALCULATE TOTAL COUNT BASED ON SUCCESSFUL READS --- ###
    total_counts = per_location_counts * successful_file_count
    means = {var: np.divide(sums[var], total_counts, where=total_counts > 0) for var in variables}

    print(f"\n--- Pass 2 of 2: Calculating Reynolds stress components (based on {successful_file_count} files) ---")
    stress_keys = [f"{v1}'{v2}'" for v1, v2 in combinations_with_replacement(variables, 2)]
    stress_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in stress_keys}
    
    for fpath in tqdm(pvtu_files, desc="Processing files for fluctuations"):
        # ### --- NEW: TRY/EXCEPT BLOCK TO SKIP BAD FILES CONSISTENTLY --- ###
        try:
            mesh = pv.read(fpath)
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in variables):
                raise ValueError("Mesh is empty or missing required variables.")
                
            fluctuations = {var: mesh.point_data[var] - means[var][inverse_indices] for var in variables}
            for v1, v2 in combinations_with_replacement(variables, 2):
                product = fluctuations[v1] * fluctuations[v2]
                np.add.at(stress_sums[f"{v1}'{v2}'"], inverse_indices, product)
                
        except Exception:
            # We already warned in Pass 1, so we can just skip silently here
            continue
            
    reynolds_stresses = {key: np.divide(stress_sums[key], total_counts, where=total_counts > 0) for key in stress_keys}

    # --- 3. ROBUST DELAUNAY-BASED INTERPOLATION (No changes) ---
    print(f"\n📈 Performing robust Delaunay interpolation...")
    all_point_data = {f"mean_{var}": data for var, data in means.items()}
    for key, data in reynolds_stresses.items():
        clean_key = key.replace("'", "")
        all_point_data[f"reynolds_stress_{clean_key}"] = data
        
    x_min, x_max, z_min, z_max = unique_xz[:, 0].min(), unique_xz[:, 0].max(), unique_xz[:, 1].min(), unique_xz[:, 1].max()
    x_range, z_range = x_max - x_min, z_max - z_min
    if x_range >= z_range:
        nx = base_resolution
        nz = int(base_resolution * (z_range / x_range)) if z_range > 0 else 1
    else:
        nz = base_resolution
        nx = int(base_resolution * (x_range / z_range)) if x_range > 0 else 1
    
    print(f"Data aspect ratio preserved. New grid resolution: ({nx}, {nz})")
    x_coords = np.linspace(x_min, x_max, nx)
    z_coords = np.linspace(z_min, z_max, nz)
    grid_x, grid_z = np.meshgrid(x_coords, z_coords)

    data_vars = {}
    for field_name, point_values in all_point_data.items():
        print(f"  Interpolating {field_name}...")
        interpolated_array = griddata(
            points=unique_xz, values=point_values, xi=(grid_x, grid_z),
            method='linear', fill_value=np.nan
        )
        data_vars[field_name] = (('z', 'x'), interpolated_array)

    # --- 4. CONSTRUCT XARRAY DATASET AND SAVE TO NETCDF (No changes) ---
    print(f"💾 Constructing dataset and saving to NetCDF file '{output_filename}'...")
    ds = xr.Dataset(data_vars, coords={'x': ('x', x_coords), 'z': ('z', z_coords)})
    ds.x.attrs.update(units='m', long_name='Streamwise Coordinate')
    ds.z.attrs.update(units='m', long_name='Wall-Normal Coordinate')
    ds.attrs.update(
        title='Time-Spanwise Averaged Reynolds Stresses and Mean Velocities',
        source_directory=data_directory,
        creation_date=str(datetime.now()),
        processed_files=successful_file_count
    )
    ds.to_netcdf(output_filename)
    print("\n✅ All calculations are complete. Data saved to NetCDF.")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    DATA_DIR = "/Users/simone/Work-local/Codes/Jexpresso/output/CompEuler/LESsmago/output-10240x10240x3000"
    #DATA_DIR = "/scratch/smarras/smarras/output/64x64x36_5kmX5kmX3km/CompEuler/LESsmago/output"
    FILE_PATTERN = "iter_*.pvtu"
    OUTPUT_NC_FILE = "reynolds_stresses.nc"
    BASE_GRID_RESOLUTION = 512
    START_STEP = 150
    END_STEP = 1000

    # --- EXECUTION ---
    if not os.path.isdir(DATA_DIR):
        print(f"\n❌ Error: The specified data directory does not exist: {DATA_DIR}")
    else:
        calculate_and_save_netcdf(
            data_directory=DATA_DIR,
            file_pattern=FILE_PATTERN,
            output_filename=OUTPUT_NC_FILE,
            variables=['u', 'v', 'w'],
            base_resolution=BASE_GRID_RESOLUTION,
            start_step=START_STEP,
            end_step=END_STEP
        )
