import os
import glob
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
    base_resolution: int = 256
):
    """
    Calculates Reynolds stresses and uses Delaunay triangulation (via Scipy) to
    interpolate them onto a regular grid before saving to a NetCDF file.
    """
    # --- 1. FILE DISCOVERY AND GRID SETUP (No changes) ---
    print("🔍 Finding data files and setting up the 2D averaging plane...")
    pvtu_files = sorted(glob.glob(os.path.join(data_directory, file_pattern)))
    if not pvtu_files:
        raise FileNotFoundError(f"No files found for pattern '{file_pattern}' in '{data_directory}'")
    print(f"Found {len(pvtu_files)} time steps.")

    first_mesh = pv.read(pvtu_files[0])
    points_3d = first_mesh.points
    xz_coords = np.round(points_3d[:, [0, 2]], decimals=coord_precision)
    unique_xz, inverse_indices = np.unique(xz_coords, axis=0, return_inverse=True)
    num_xz_points = len(unique_xz)
    print(f"Identified {num_xz_points} unique (x, z) points for averaging.")

    # --- 2. PERFORM TWO-PASS CALCULATION (No changes) ---
    print("\n--- Pass 1 of 2: Calculating sums for mean velocities ---")
    sums = {var: np.zeros(num_xz_points, dtype=np.float64) for var in variables}
    total_counts = np.zeros(num_xz_points, dtype=np.int64)
    for fpath in tqdm(pvtu_files, desc="Processing files for means"):
        mesh = pv.read(fpath)
        for var in variables: np.add.at(sums[var], inverse_indices, mesh.point_data[var])
        if fpath == pvtu_files[0]: np.add.at(total_counts, inverse_indices, 1)
    total_counts *= len(pvtu_files)
    means = {var: np.divide(sums[var], total_counts, where=total_counts > 0) for var in variables}

    print("\n--- Pass 2 of 2: Calculating Reynolds stress components ---")
    stress_keys = [f"{v1}'{v2}'" for v1, v2 in combinations_with_replacement(variables, 2)]
    stress_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in stress_keys}
    for fpath in tqdm(pvtu_files, desc="Processing files for fluctuations"):
        mesh = pv.read(fpath)
        fluctuations = {var: mesh.point_data[var] - means[var][inverse_indices] for var in variables}
        for v1, v2 in combinations_with_replacement(variables, 2):
            product = fluctuations[v1] * fluctuations[v2]
            np.add.at(stress_sums[f"{v1}'{v2}'"], inverse_indices, product)
    reynolds_stresses = {key: np.divide(stress_sums[key], total_counts, where=total_counts > 0) for key in stress_keys}

    # --- 3. ROBUST DELAUNAY-BASED INTERPOLATION (NEW IMPLEMENTATION) ---
    print(f"\n📈 Performing robust Delaunay interpolation...")
    
    # Create a dictionary of all pointwise data fields to be interpolated
    all_point_data = {f"mean_{var}": data for var, data in means.items()}
    for key, data in reynolds_stresses.items():
        clean_key = key.replace("'", "")
        all_point_data[f"reynolds_stress_{clean_key}"] = data
        
    # Define the target regular grid using dynamic resolution
    x_min, x_max = unique_xz[:, 0].min(), unique_xz[:, 0].max()
    z_min, z_max = unique_xz[:, 1].min(), unique_xz[:, 1].max()
    x_range = x_max - x_min
    z_range = z_max - z_min
    
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

    # Interpolate each data field one-by-one using scipy.griddata
    data_vars = {}
    for field_name, point_values in all_point_data.items():
        print(f"  Interpolating {field_name}...")
        interpolated_array = griddata(
            points=unique_xz,           # The source (x,z) points
            values=point_values,        # The data values at those points
            xi=(grid_x, grid_z),        # The target grid
            method='linear',            # Delaunay triangulation based
            fill_value=np.nan           # Use NaN for points outside the data domain
        )
        data_vars[field_name] = (('z', 'x'), interpolated_array)

    # --- 4. CONSTRUCT XARRAY DATASET AND SAVE TO NETCDF (No changes) ---
    print(f"💾 Constructing dataset and saving to NetCDF file '{output_filename}'...")

    ds = xr.Dataset(
        data_vars,
        coords={'x': ('x', x_coords), 'z': ('z', z_coords)}
    )

    ds.x.attrs['units'] = 'm'
    ds.x.attrs['long_name'] = 'Streamwise Coordinate'
    ds.z.attrs['units'] = 'm'
    ds.z.attrs['long_name'] = 'Wall-Normal Coordinate'
    
    ds.attrs['title'] = 'Time-Spanwise Averaged Reynolds Stresses and Mean Velocities'
    ds.attrs['source_directory'] = data_directory
    ds.attrs['created_by'] = 'Python script'
    ds.attrs['creation_date'] = str(datetime.now())

    ds.to_netcdf(output_filename)
        
    print("\n✅ All calculations are complete. Data saved to NetCDF.")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    DATA_DIR = "/Users/simone/Work-local/Codes/Jexpresso/output/CompEuler/LESsmago/output-10240x10240x3000"
    
    FILE_PATTERN = "iter_*.pvtu"
    
    OUTPUT_NC_FILE = "reynolds_stresses.nc"
    
    BASE_GRID_RESOLUTION = 512

    # --- EXECUTION ---
    if not os.path.isdir(DATA_DIR):
        print(f"\n❌ Error: The specified data directory does not exist: {DATA_DIR}")
    else:
        calculate_and_save_netcdf(
            data_directory=DATA_DIR,
            file_pattern=FILE_PATTERN,
            output_filename=OUTPUT_NC_FILE,
            variables=['u', 'v', 'w'],
            base_resolution=BASE_GRID_RESOLUTION
        )
