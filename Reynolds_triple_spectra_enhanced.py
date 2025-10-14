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
import matplotlib.pyplot as plt

# (Helper functions sanitize_var_name, get_primed_name, and plotting functions remain unchanged)
def sanitize_var_name(name):
    """Replace Unicode characters that cause NetCDF encoding issues"""
    return name.replace('θ', 'theta')

def get_primed_name(key, all_vars):
    clean_key = key.replace("'", "")
    prime_map = {var: f"{sanitize_var_name(var)}p" for var in all_vars}
    components = []
    i = 0
    while i < len(clean_key):
        found_var = None
        for var in sorted(all_vars, key=len, reverse=True):
            if clean_key[i:].startswith(var):
                found_var = var
                break
        if found_var:
            components.append(found_var)
            i += len(found_var)
        else:
            i += 1
    return "".join([prime_map[c] for c in components])

def plot_second_moment_profiles(dataset, output_dir):
    print(f"📊 Generating second-moment profile plots in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    z_coords = dataset['z'].values
    mean_vars = [v for v in dataset.data_vars if v.startswith('mean_')]
    moment_vars = [v for v in dataset.data_vars if v not in mean_vars and not v.startswith('grad_')]
    for var_name in tqdm(moment_vars, desc="Generating profile plots"):
        try:
            profile_data = dataset[var_name].mean(dim='x').values
            title = f"Vertical Profile of {var_name}"; xlabel = f"$\\langle {var_name} \\rangle$"
            plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(profile_data, z_coords, color='k', linewidth=2.5)
            ax.set_xlabel(xlabel, fontsize=14); ax.set_ylabel('Height, z [m]', fontsize=12)
            ax.set_title(title, fontsize=14, weight='bold'); ax.set_ylim(bottom=0)
            ax.tick_params(axis='both', which='major', labelsize=10)
            plot_filename = os.path.join(output_dir, f"{var_name}_profile.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
        except Exception as e:
            print(f"\n⚠️ WARNING: Could not generate plot for {var_name}. Reason: {e}")
    print(f"✅ Second-moment profile plots saved successfully.")

# (Other utility functions like calculate_and_plot_spanwise_spectra, etc. remain unchanged)

def calculate_and_save_averaged_stats(
    data_directory: str,
    file_pattern: str,
    output_filename: str,
    vel_variables: list,
    scalar_variables: list,
    coord_precision: int,
    base_resolutionx: int,
    base_resolutionz: int,
    start_step: int,
    end_step: int,
    profile_plot_filename: str,
    second_moment_plot_dir: str,
    kinematic_viscosity: float,
    thermal_diffusivity: float
):
    """
    Main function to calculate time-averaged statistics, including dissipation rates, and save them.
    """
    all_variables = vel_variables + scalar_variables
    print("🔍 Finding and filtering data files for averaging...")
    # ... (File finding logic remains the same)
    all_pvtu_files = sorted(glob.glob(os.path.join(data_directory, file_pattern)))
    if not all_pvtu_files: raise FileNotFoundError(f"No files found for pattern '{file_pattern}' in '{data_directory}'")
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
    else: pvtu_files = all_pvtu_files
    if not pvtu_files: raise FileNotFoundError(f"No files found in the specified range [{start_step}-{end_step}]")
    print(f"Found {len(pvtu_files)} time steps to process.")

    first_mesh = pv.read(pvtu_files[0])
    points_3d = first_mesh.points
    xz_coords = np.round(points_3d[:, [0, 2]], decimals=coord_precision)
    unique_xz, inverse_indices = np.unique(xz_coords, axis=0, return_inverse=True)
    num_xz_points = len(unique_xz)
    print(f"Identified {num_xz_points} unique (x, z) points for averaging.")
    
    print("\n--- Pass 1 of 2: Calculating sums for mean quantities ---")
    # ... (Pass 1 logic remains the same)
    sums = {var: np.zeros(num_xz_points, dtype=np.float64) for var in all_variables}
    per_location_counts = np.zeros(num_xz_points, dtype=np.int64)
    successful_file_count = 0
    for fpath in tqdm(pvtu_files, desc="Processing files for means"):
        try:
            mesh = pv.read(fpath)
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in all_variables): continue
            for var in all_variables: np.add.at(sums[var], inverse_indices, mesh.point_data[var])
            if successful_file_count == 0: np.add.at(per_location_counts, inverse_indices, 1)
            successful_file_count += 1
        except Exception as e:
            tqdm.write(f"\n⚠️ WARNING: Skipping file '{os.path.basename(fpath)}' due to error: {e}")
            continue
    if successful_file_count == 0: raise RuntimeError("Could not successfully read any files. Aborting.")
    total_counts = per_location_counts * successful_file_count
    means = {var: np.divide(sums[var], total_counts, where=total_counts > 0) for var in all_variables}

    ### MODIFIED: Define the target interpolation grid *before* Pass 2
    print("\n📈 Defining target Cartesian grid for interpolation and gradient calculations...")
    x_min, x_max = unique_xz[:, 0].min(), unique_xz[:, 0].max()
    z_min, z_max = unique_xz[:, 1].min(), unique_xz[:, 1].max()
    nx, nz = base_resolutionx, base_resolutionz
    x_coords = np.linspace(x_min, x_max, nx)
    z_coords = np.linspace(z_min, z_max, nz)
    grid_x, grid_z = np.meshgrid(x_coords, z_coords)
    print(f"   Target grid resolution: {nx}x{nz}")

    print(f"\n--- Pass 2 of 2: Calculating fluctuations, moments, and instantaneous gradients (based on {successful_file_count} files) ---")
    stress_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in [f"{v1}'{v2}'" for v1, v2 in combinations_with_replacement(vel_variables, 2)]}
    # ... (other sum initializations for moments are the same)
    triple_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in [f"{v1}'{v2}'{v3}'" for v1, v2, v3 in combinations_with_replacement(vel_variables, 3)] + [f"{v1}'{v2}'{s}'" for s in scalar_variables for v1, v2 in combinations_with_replacement(vel_variables, 2)]}
    scalar_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in [f"{s}'{s}'" for s in scalar_variables] + [f"{v}'{s}'" for s in scalar_variables for v in vel_variables]}

    ### NEW: Initialize sums for dissipation terms on the STRUCTURED grid
    # This dictionary will hold the sum of the squares of instantaneous gradients: Σ(dui/dxj)^2
    dissipation_sums_structured = {
        'sum_sq_grad_vel': np.zeros((nz, nx), dtype=np.float64)
    }
    if 'θ' in scalar_variables:
        dissipation_sums_structured['sum_sq_grad_theta'] = np.zeros((nz, nx), dtype=np.float64)

    for fpath in tqdm(pvtu_files, desc="Pass 2/2: Fluctuations & Gradients"):
        try:
            mesh = pv.read(fpath)
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in all_variables): continue
            
            # --- Part A: Calculate fluctuations on the native grid (efficient) ---
            fluctuations = {var: mesh.point_data[var] - means[var][inverse_indices] for var in all_variables}
            for v1, v2 in combinations_with_replacement(vel_variables, 2): np.add.at(stress_sums[f"{v1}'{v2}'"], inverse_indices, fluctuations[v1] * fluctuations[v2])
            # ... (Calculation of other moments on native grid is the same)
            for s in scalar_variables:
                np.add.at(scalar_sums[f"{s}'{s}'"], inverse_indices, fluctuations[s] * fluctuations[s])
                for v in vel_variables: np.add.at(scalar_sums[f"{v}'{s}'"], inverse_indices, fluctuations[v] * fluctuations[s])
            for v1, v2, v3 in combinations_with_replacement(vel_variables, 3): np.add.at(triple_sums[f"{v1}'{v2}'{v3}'"], inverse_indices, fluctuations[v1] * fluctuations[v2] * fluctuations[v3])
            for s in scalar_variables:
                for v1, v2 in combinations_with_replacement(vel_variables, 2): np.add.at(triple_sums[f"{v1}'{v2}'{s}'"], inverse_indices, fluctuations[v1] * fluctuations[v2] * fluctuations[s])

            ### NEW/MODIFIED: Part B: Calculate instantaneous gradients on the structured grid ###
            # This is the robust method you requested.
            
            # 1. Interpolate instantaneous fields to the structured grid
            points_2d = mesh.points[:, [0, 2]] # Use only (x, z) for interpolation
            
            # Velocity fields
            u_inst = griddata(points_2d, mesh.point_data['u'], (grid_x, grid_z), method='linear')
            v_inst = griddata(points_2d, mesh.point_data['v'], (grid_x, grid_z), method='linear')
            w_inst = griddata(points_2d, mesh.point_data['w'], (grid_x, grid_z), method='linear')
            
            # 2. Calculate gradients using numpy.gradient for structured data
            # np.gradient returns derivatives in (axis 0, axis 1) order -> (z, x)
            du_dz, du_dx = np.gradient(u_inst, z_coords, x_coords)
            dv_dz, dv_dx = np.gradient(v_inst, z_coords, x_coords)
            dw_dz, dw_dx = np.gradient(w_inst, z_coords, x_coords)
            
            # 3. Sum the squares and accumulate
            # Note: We assume d/dy derivatives are zero for the 2D plane data
            sum_sq_grad_inst_vel = (du_dx**2 + du_dz**2) + (dv_dx**2 + dv_dz**2) + (dw_dx**2 + dw_dz**2)
            dissipation_sums_structured['sum_sq_grad_vel'] += np.nan_to_num(sum_sq_grad_inst_vel)

            # Scalar field
            if 'θ' in scalar_variables:
                theta_inst = griddata(points_2d, mesh.point_data['θ'], (grid_x, grid_z), method='linear')
                dtheta_dz, dtheta_dx = np.gradient(theta_inst, z_coords, x_coords)
                sum_sq_grad_inst_theta = dtheta_dx**2 + dtheta_dz**2
                dissipation_sums_structured['sum_sq_grad_theta'] += np.nan_to_num(sum_sq_grad_inst_theta)

        except Exception as e:
            tqdm.write(f"\n⚠️ WARNING: Could not process file {os.path.basename(fpath)}. Reason: {e}")
            continue

    # --- Final Averaging and Dataset Construction ---
    print("\nFinalizing averages and interpolating remaining fields...")
    reynolds_stresses = {key: np.divide(stress_sums[key], total_counts, where=total_counts > 0) for key in stress_sums}
    # ... (averaging for other moments is the same)
    triple_moments = {key: np.divide(triple_sums[key], total_counts, where=total_counts > 0) for key in triple_sums}
    scalar_stats = {key: np.divide(scalar_sums[key], total_counts, where=total_counts > 0) for key in scalar_sums}
    
    ### MODIFIED: Calculate the time-average of the squared gradients
    mean_sq_grads = {key: val / successful_file_count for key, val in dissipation_sums_structured.items()}

    print("💾 Constructing xarray dataset and saving to NetCDF...")
    data_vars = {}
    
    # Interpolate all stats calculated on the native grid
    all_native_stats = {**means, **reynolds_stresses, **triple_moments, **scalar_stats}
    for field_name, point_values in tqdm(all_native_stats.items(), desc="Interpolating fields"):
        # Sanitize names for NetCDF compatibility
        if "'" in field_name:
            sanitized_name = get_primed_name(field_name, all_variables)
        else:
            sanitized_name = f"mean_{field_name}"
        sanitized_name = sanitize_var_name(sanitized_name)

        interpolated_array = griddata(unique_xz, point_values, (grid_x, grid_z), method='linear')
        data_vars[sanitized_name] = (('z', 'x'), interpolated_array)

    ds = xr.Dataset(data_vars, coords={'x': ('x', x_coords), 'z': ('z', z_coords)})

    ### MODIFIED: Add the pre-calculated mean squared gradient terms to the dataset
    # These are already on the structured grid, no interpolation needed.
    ds['sum_mean_sq_grad_inst_vel'] = (('z', 'x'), mean_sq_grads['sum_sq_grad_vel'])
    if 'θ' in scalar_variables:
        ds['sum_mean_sq_grad_inst_theta'] = (('z', 'x'), mean_sq_grads['sum_sq_grad_theta'])

    ### FINAL DISSIPATION CALCULATION ###
    print("🔬 Calculating TKE and scalar variance dissipation rates...")
    try:
        # Calculate squared gradients of the MEAN velocity fields (d<ui>/dxj)^2
        d_mean_u_dx = ds['mean_u'].differentiate("x"); d_mean_u_dz = ds['mean_u'].differentiate("z")
        d_mean_v_dx = ds['mean_v'].differentiate("x"); d_mean_v_dz = ds['mean_v'].differentiate("z")
        d_mean_w_dx = ds['mean_w'].differentiate("x"); d_mean_w_dz = ds['mean_w'].differentiate("z")
        sum_sq_grad_mean_vel = (d_mean_u_dx**2 + d_mean_u_dz**2) + (d_mean_v_dx**2 + d_mean_v_dz**2) + (d_mean_w_dx**2 + d_mean_w_dz**2)
        
        # Final TKE dissipation calculation: eps = nu * [ <(dui/dxj)^2> - (d<ui>/dxj)^2 ]
        eps = kinematic_viscosity * (ds['sum_mean_sq_grad_inst_vel'] - sum_sq_grad_mean_vel)
        ds['eps'] = eps.where(eps > 0) # Enforce positivity
        ds['eps'].attrs = {'units': 'm^2/s^3', 'long_name': 'Turbulent Kinetic Energy Dissipation Rate'}

        if 'θ' in scalar_variables and 'sum_mean_sq_grad_inst_theta' in ds:
            d_mean_theta_dx = ds['mean_theta'].differentiate("x")
            d_mean_theta_dz = ds['mean_theta'].differentiate("z")
            sum_sq_grad_mean_theta = d_mean_theta_dx**2 + d_mean_theta_dz**2
            
            eps_t = thermal_diffusivity * (ds['sum_mean_sq_grad_inst_theta'] - sum_sq_grad_mean_theta)
            ds['eps_t'] = eps_t.where(eps_t > 0)
            ds['eps_t'].attrs = {'units': 'K^2/s', 'long_name': 'Half Temperature Variance Dissipation Rate'}
        
        # Drop the intermediate gradient terms from the final file
        ds = ds.drop_vars(['sum_mean_sq_grad_inst_vel', 'sum_mean_sq_grad_inst_theta'], errors='ignore')
        print("✅ Dissipation rates calculated and added to the dataset.")
    except Exception as e:
        print(f"\n❌ ERROR: Could not calculate dissipation rates. Reason: {e}")

    ds.to_netcdf(output_filename)
    print(f"\n✅ Time-averaged calculations complete. Data saved to '{output_filename}'.")
    
    # ... (Plotting logic remains the same)
    if profile_plot_filename:
        # ...
        pass
    if second_moment_plot_dir:
        plot_second_moment_profiles(ds, second_moment_plot_dir)

    return pvtu_files

if __name__ == '__main__':
    # --- CONFIGURATION ---
    DATA_DIR = "/Users/simone/Work-local/Codes/Jexpresso/output/CompEuler/LESsmago/64x64x36_5kmX5kmX3km" # Use current directory or specify path
    FILE_PATTERN = "iter_*.pvtu"
    OUTPUT_NC_AVERAGED_FILE = "turbulence_statistics_averaged.nc"
    PROFILE_PLOT_FILE = "vertical_wind_profile.png"
    SECOND_MOMENT_PLOT_DIR = "second_moment_profiles"
    BASE_GRID_RESOLUTIONX = 512
    BASE_GRID_RESOLUTIONZ = 300
    START_STEP = 150
    END_STEP = 1000

    ### FLUID PROPERTIES ###
    KINEMATIC_VISCOSITY = 1.0e-5  # [m^2/s]
    THERMAL_DIFFUSIVITY = 1.4e-5  # [m^2/s]

    # --- VARIABLE DEFINITION ---
    VELOCITY_VARS = ['u', 'v', 'w']
    SCALAR_VARS = ['θ'] # Removed 'p' as it's not typically used in dissipation

    # --- EXECUTION ---
    if not os.path.isdir(DATA_DIR):
        print(f"\n❌ Error: The specified data directory does not exist: {DATA_DIR}")
    else:
        calculate_and_save_averaged_stats(
            data_directory=DATA_DIR, file_pattern=FILE_PATTERN, output_filename=OUTPUT_NC_AVERAGED_FILE,
            vel_variables=VELOCITY_VARS, scalar_variables=SCALAR_VARS, coord_precision=6,
            base_resolutionx=BASE_GRID_RESOLUTIONX, base_resolutionz=BASE_GRID_RESOLUTIONZ,
            start_step=START_STEP, end_step=END_STEP,
            profile_plot_filename=PROFILE_PLOT_FILE, second_moment_plot_dir=SECOND_MOMENT_PLOT_DIR,
            kinematic_viscosity=KINEMATIC_VISCOSITY, thermal_diffusivity=THERMAL_DIFFUSIVITY
        )
