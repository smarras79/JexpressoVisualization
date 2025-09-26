import os
import glob
import re
import numpy as np
import pyvista as pv
import xarray as xr
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import griddata
from itertools import combinations_with_replacement, product as cartesian_product

def calculate_and_save_netcdf(
    data_directory: str,
    file_pattern: str,
    output_filename: str = "turbulence_statistics.nc",
    vel_variables: list = ['u', 'v', 'w'],
    scalar_variables: list = [],
    coord_precision: int = 6,
    base_resolutionx: int = 512,
    base_resolutionz: int = 256,
    start_step: int = None,
    end_step: int = None
):
    """
    Calculates Reynolds stresses, heat fluxes, triple moments, and other turbulence
    statistics, skipping any unreadable files. Saves results to a NetCDF file.
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

    all_variables = vel_variables + scalar_variables

    # --- 2. PERFORM TWO-PASS CALCULATION (No changes) ---
    print("\n--- Pass 1 of 2: Calculating sums for mean quantities ---")
    sums = {var: np.zeros(num_xz_points, dtype=np.float64) for var in all_variables}
    per_location_counts = np.zeros(num_xz_points, dtype=np.int64)
    successful_file_count = 0

    for fpath in tqdm(pvtu_files, desc="Processing files for means"):
        try:
            mesh = pv.read(fpath)
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in all_variables):
                missing_vars = [v for v in all_variables if v not in mesh.point_data]
                raise ValueError(f"Mesh is empty or missing required variables: {missing_vars}")
            
            for var in all_variables:
                np.add.at(sums[var], inverse_indices, mesh.point_data[var])
            
            if successful_file_count == 0:
                np.add.at(per_location_counts, inverse_indices, 1)
            
            successful_file_count += 1
            
        except Exception as e:
            tqdm.write(f"\n⚠️ WARNING: Skipping file '{os.path.basename(fpath)}' due to error: {e}")
            continue

    if successful_file_count == 0:
        raise RuntimeError("Could not successfully read any files. Aborting.")
        
    total_counts = per_location_counts * successful_file_count
    means = {var: np.divide(sums[var], total_counts, where=total_counts > 0) for var in all_variables}

    print(f"\n--- Pass 2 of 2: Calculating fluctuations and moments (based on {successful_file_count} files) ---")
    
    stress_keys = [f"{v1}'{v2}'" for v1, v2 in combinations_with_replacement(vel_variables, 2)]
    stress_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in stress_keys}
    
    vel_triple_keys = [f"{v1}'{v2}'{v3}'" for v1, v2, v3 in combinations_with_replacement(vel_variables, 3)]
    mixed_triple_keys = [f"{v1}'{v2}'{s}'" for s in scalar_variables for v1, v2 in combinations_with_replacement(vel_variables, 2)]
    triple_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in vel_triple_keys + mixed_triple_keys}
    
    scalar_variance_keys = [f"{s}'{s}'" for s in scalar_variables]
    heat_flux_keys = [f"{v}'{s}'" for s in scalar_variables for v in vel_variables]
    scalar_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in scalar_variance_keys + heat_flux_keys}
    
    for fpath in tqdm(pvtu_files, desc="Processing files for fluctuations"):
        try:
            mesh = pv.read(fpath)
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in all_variables):
                raise ValueError("Mesh is empty or missing required variables.")
                
            fluctuations = {var: mesh.point_data[var] - means[var][inverse_indices] for var in all_variables}

            for v1, v2 in combinations_with_replacement(vel_variables, 2):
                product = fluctuations[v1] * fluctuations[v2]
                np.add.at(stress_sums[f"{v1}'{v2}'"], inverse_indices, product)
            
            for s in scalar_variables:
                product_var = fluctuations[s] * fluctuations[s]
                np.add.at(scalar_sums[f"{s}'{s}'"], inverse_indices, product_var)
                for v in vel_variables:
                    product_flux = fluctuations[v] * fluctuations[s]
                    np.add.at(scalar_sums[f"{v}'{s}'"], inverse_indices, product_flux)

            for v1, v2, v3 in combinations_with_replacement(vel_variables, 3):
                product = fluctuations[v1] * fluctuations[v2] * fluctuations[v3]
                np.add.at(triple_sums[f"{v1}'{v2}'{v3}'"], inverse_indices, product)
            
            for s in scalar_variables:
                for v1, v2 in combinations_with_replacement(vel_variables, 2):
                    product = fluctuations[v1] * fluctuations[v2] * fluctuations[s]
                    np.add.at(triple_sums[f"{v1}'{v2}'{s}'"], inverse_indices, product)
                    
        except Exception:
            continue
            
    reynolds_stresses = {key: np.divide(stress_sums[key], total_counts, where=total_counts > 0) for key in stress_keys}
    triple_moments = {key: np.divide(triple_sums[key], total_counts, where=total_counts > 0) for key in triple_sums}
    scalar_stats = {key: np.divide(scalar_sums[key], total_counts, where=total_counts > 0) for key in scalar_sums}

    # --- 3. ROBUST DELAUNAY-BASED INTERPOLATION ---
    print(f"\n📈 Performing robust Delaunay interpolation...")
    all_point_data = {}
    for var, data in means.items():
        all_point_data[f"mean_{var}"] = data
    for key, data in reynolds_stresses.items():
        clean_key = key.replace("'", "")
        all_point_data[f"reynolds_stress_{clean_key}"] = data
        
    ### --- BUG FIX: Correctly separate and add scalar statistics --- ###
    # Add scalar variances to the data dictionary
    for key_format in scalar_variance_keys:
        clean_key = key_format.replace("'", "")
        all_point_data[f"scalar_variance_{clean_key}"] = scalar_stats[key_format]

    # Add heat fluxes to the data dictionary
    for key_format in heat_flux_keys:
        clean_key = key_format.replace("'", "")
        all_point_data[f"heat_flux_{clean_key}"] = scalar_stats[key_format]
    ### --- END FIX --- ###

    for key, data in triple_moments.items():
        clean_key = key.replace("'", "")
        all_point_data[f"triple_moment_{clean_key}"] = data

    x_min, x_max, z_min, z_max = unique_xz[:, 0].min(), unique_xz[:, 0].max(), unique_xz[:, 1].min(), unique_xz[:, 1].max()
    x_range, z_range = x_max - x_min, z_max - z_min
    #if x_range >= z_range:
    #    nx = base_resolution
    #    nz = int(base_resolution * (z_range / x_range)) if z_range > 0 else 1
    #else:
    #    nz = base_resolution
    #    nx = int(base_resolution * (x_range / z_range)) if x_range > 0 else 1
    nz = base_resolutionz
    nx = base_resolutionx
    
    print(f"Data aspect ratio preserved. New grid resolution: ({nx}, {nz})")
    x_coords = np.linspace(x_min, x_max, nx)
    z_coords = np.linspace(z_min, z_max, nz)
    grid_x, grid_z = np.meshgrid(x_coords, z_coords)

    data_vars = {}
    for field_name, point_values in tqdm(all_point_data.items(), desc="Interpolating fields"):
        interpolated_array = griddata(
            points=unique_xz, values=point_values, xi=(grid_x, grid_z),
            method='linear', fill_value=np.nan
        )
        data_vars[field_name] = (('z', 'x'), interpolated_array)

    # --- 4. CONSTRUCT XARRAY DATASET AND SAVE TO NETCDF ---
    print(f"💾 Constructing dataset and saving to NetCDF file '{output_filename}'...")
    ds = xr.Dataset(data_vars, coords={'x': ('x', x_coords), 'z': ('z', z_coords)})
    ds.x.attrs.update(units='m', long_name='Streamwise Coordinate')
    ds.z.attrs.update(units='m', long_name='Wall-Normal Coordinate')
    ds.attrs.update(
        title='Time-Spanwise Averaged Turbulence Statistics',
        source_directory=data_directory,
        creation_date=str(datetime.now()),
        processed_files=successful_file_count
    )
    ds.to_netcdf(output_filename)
    print("\n✅ All calculations are complete. Data saved to NetCDF.")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    # IMPORTANT: Update this to your actual data directory
    DATA_DIR = "/Users/simone/Work-local/Codes/Jexpresso/output/CompEuler/LESsmago/output-10240x10240x3000"
    FILE_PATTERN = "iter_*.pvtu"
    OUTPUT_NC_FILE = "turbulence_statistics.nc"
    BASE_GRID_RESOLUTIONX = 512 #512
    BASE_GRID_RESOLUTIONZ = 300 #512
    START_STEP = 150
    END_STEP = 1000
    
    # --- VARIABLE DEFINITION ---
    VELOCITY_VARS = ['u', 'v', 'w']
    # If your scalar is named 'T', change 'theta' to 'T'. Use [] if you have no scalars.
    SCALAR_VARS = ['θ', 'p']

    # --- EXECUTION ---
    if not os.path.isdir(DATA_DIR):
        print(f"\n❌ Error: The specified data directory does not exist: {DATA_DIR}")
        print("Please update the 'DATA_DIR' variable in the script.")
    else:
        calculate_and_save_netcdf(
            data_directory=DATA_DIR,
            file_pattern=FILE_PATTERN,
            output_filename=OUTPUT_NC_FILE,
            vel_variables=VELOCITY_VARS,
            scalar_variables=SCALAR_VARS,
            base_resolutionx=BASE_GRID_RESOLUTIONX,
            base_resolutionz=BASE_GRID_RESOLUTIONZ,
            start_step=START_STEP,
            end_step=END_STEP
        )
