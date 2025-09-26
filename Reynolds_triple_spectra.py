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

def get_primed_name(key, all_vars):
    """
    Converts a moment key like "u'v'" or "u'u'θ'" into the new
    naming convention like "upvp" or "upupthetap".
    """
    clean_key = key.replace("'", "")
    prime_map = {var: f"{var.replace('θ', 'theta')}p" for var in all_vars}
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
    """
    Plots the vertical profiles of all second-order moments found in the dataset.
    """
    print(f"📊 Generating second-moment profile plots in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    z_coords = dataset['z'].values
    mean_vars = [v for v in dataset.data_vars if v.startswith('mean_')]
    moment_vars = [v for v in dataset.data_vars if v not in mean_vars]
    
    for var_name in tqdm(moment_vars, desc="Generating profile plots"):
        try:
            profile_data = dataset[var_name].mean(dim='x').values
            title = f"Vertical Profile of {var_name}"
            xlabel = f"$\\langle {var_name} \\rangle$"
            
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(profile_data, z_coords, color='k', linewidth=2.5)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel('Height, z [m]', fontsize=12)
            ax.set_title(title, fontsize=14, weight='bold')
            ax.set_ylim(bottom=0)
            ax.tick_params(axis='both', which='major', labelsize=10)
            plot_filename = os.path.join(output_dir, f"{var_name}_profile.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"\n⚠️ WARNING: Could not generate plot for {var_name}. Reason: {e}")
    print(f"✅ Second-moment profile plots saved successfully.")


def calculate_and_plot_spanwise_spectra(
    pvtu_files: list,
    variables: list,
    output_filename: str
):
    """
    Calculates the time-averaged 1D spanwise turbulent spectra from all snapshots.
    """
    print(f"\n⚡ Calculating and plotting time-averaged spanwise spectra...")
    if not pvtu_files:
        print("⚠️ No files available for spectra calculation. Skipping.")
        return

    # --- 1. Initialize from a representative snapshot ---
    try:
        first_mesh_container = pv.read(pvtu_files[0])
        mesh = first_mesh_container.combine(merge_points=False) if isinstance(first_mesh_container, pv.MultiBlock) else first_mesh_container
    except Exception as e:
        print(f"❌ ERROR: Could not read first file for spectra setup. Reason: {e}")
        return

    points = mesh.points
    bounds = mesh.bounds
    center_x = (bounds[0] + bounds[1]) / 2
    center_z = (bounds[4] + bounds[5]) / 2
    closest_point_idx = np.argmin(np.sqrt((points[:, 0] - center_x)**2 + (points[:, 2] - center_z)**2))
    actual_x, _, actual_z = points[closest_point_idx]
    indices = np.where((np.isclose(points[:, 0], actual_x)) & (np.isclose(points[:, 2], actual_z)))[0]
    
    if len(indices) < 2:
        print("⚠️ Could not find a line of data in the y-direction. Skipping spectra.")
        return
        
    y_coords = points[indices, 1]
    sort_order = np.argsort(y_coords)
    indices = indices[sort_order]
    y_coords = y_coords[sort_order]

    Ly = y_coords.max() - y_coords.min()
    Ny = len(y_coords)

    # Wavenumbers (only positive frequencies)
    k_y = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)[:Ny//2]

    # --- 2. Loop through all files to accumulate spectra ---
    psd_sums = {var: np.zeros(Ny // 2) for var in variables}
    spectra_success_count = 0
    
    for fpath in tqdm(pvtu_files, desc="Calculating spectra for each timestep"):
        try:
            mesh_container = pv.read(fpath)
            mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
            if not all(v in mesh.point_data for v in variables): continue
            
            for var in variables:
                data_line = mesh.point_data[var][indices]
                fluctuations = data_line - np.mean(data_line)
                fft_coeffs = np.fft.fft(fluctuations)
                psd = (np.abs(fft_coeffs)**2) / (Ny**2)
                psd_sums[var] += psd[:Ny//2]
            
            spectra_success_count += 1

        except Exception as e:
            tqdm.write(f"\n⚠️ WARNING: Could not process {os.path.basename(fpath)} for spectra. Reason: {e}")
            continue

    if spectra_success_count == 0:
        print("❌ ERROR: No files could be processed for spectra. Aborting plot.")
        return

    # --- 3. Average the spectra ---
    avg_psd = {var: psd_sums[var] / spectra_success_count for var in variables}

    # --- 4. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'u': 'r', 'v': 'g', 'w': 'b', 'θ': 'orange'}

    for var in variables:
        var_label = var.replace('θ', r'\theta')
        label_str = f"$E_{{{var_label}}}(k_y)$"
        ax.loglog(k_y, avg_psd[var], color=colors.get(var, 'k'), label=label_str)

    k_ref_start_index = max(5, int(len(k_y) * 0.1))
    k_ref_end_index = min(len(k_y) - 10, int(len(k_y) * 0.4))
    if k_ref_start_index < k_ref_end_index:
        k_ref = np.array([k_y[k_ref_start_index], k_y[k_ref_end_index]])
        # Find a good vertical position for the reference slope
        ref_energy = avg_psd['u'][k_ref_start_index]
        ax.plot(k_ref, ref_energy * (k_ref/k_ref[0])**(-5/3), 'k--', label="$k_y^{-5/3}$ Slope")

    ax.set_xlabel('Spanwise Wavenumber, $k_y$ [m$^{-1}$]', fontsize=14)
    ax.set_ylabel('Time-Averaged Power Spectral Density, $E(k_y)$', fontsize=14)
    ax.set_title(f'1D Spanwise Turbulent Spectra at z={actual_z:.2f}m (Avg. of {spectra_success_count} files)', fontsize=16, weight='bold')
    ax.set_ylim(bottom=1e-9)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Time-averaged spectra plot saved successfully to {output_filename}")


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
    end_step: int = None,
    profile_plot_filename: str = None,
    second_moment_plot_dir: str = None
):
    """
    Main function to calculate statistics and optionally generate plots.
    """
    all_variables = vel_variables + scalar_variables
    print("🔍 Finding and filtering data files...")
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

    print(f"\n--- Pass 2 of 2: Calculating fluctuations and moments (based on {successful_file_count} files) ---")
    stress_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in [f"{v1}'{v2}'" for v1, v2 in combinations_with_replacement(vel_variables, 2)]}
    triple_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in [f"{v1}'{v2}'{v3}'" for v1, v2, v3 in combinations_with_replacement(vel_variables, 3)] + [f"{v1}'{v2}'{s}'" for s in scalar_variables for v1, v2 in combinations_with_replacement(vel_variables, 2)]}
    scalar_sums = {key: np.zeros(num_xz_points, dtype=np.float64) for key in [f"{s}'{s}'" for s in scalar_variables] + [f"{v}'{s}'" for s in scalar_variables for v in vel_variables]}
    for fpath in tqdm(pvtu_files, desc="Processing files for fluctuations"):
        try:
            mesh = pv.read(fpath)
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in all_variables): continue
            fluctuations = {var: mesh.point_data[var] - means[var][inverse_indices] for var in all_variables}
            for v1, v2 in combinations_with_replacement(vel_variables, 2): np.add.at(stress_sums[f"{v1}'{v2}'"], inverse_indices, fluctuations[v1] * fluctuations[v2])
            for s in scalar_variables:
                np.add.at(scalar_sums[f"{s}'{s}'"], inverse_indices, fluctuations[s] * fluctuations[s])
                for v in vel_variables: np.add.at(scalar_sums[f"{v}'{s}'"], inverse_indices, fluctuations[v] * fluctuations[s])
            for v1, v2, v3 in combinations_with_replacement(vel_variables, 3): np.add.at(triple_sums[f"{v1}'{v2}'{v3}'"], inverse_indices, fluctuations[v1] * fluctuations[v2] * fluctuations[v3])
            for s in scalar_variables:
                for v1, v2 in combinations_with_replacement(vel_variables, 2): np.add.at(triple_sums[f"{v1}'{v2}'{s}'"], inverse_indices, fluctuations[v1] * fluctuations[v2] * fluctuations[s])
        except Exception: continue
    reynolds_stresses = {key: np.divide(stress_sums[key], total_counts, where=total_counts > 0) for key in stress_sums}
    triple_moments = {key: np.divide(triple_sums[key], total_counts, where=total_counts > 0) for key in triple_sums}
    scalar_stats = {key: np.divide(scalar_sums[key], total_counts, where=total_counts > 0) for key in scalar_sums}
    
    print(f"\n📈 Performing robust Delaunay interpolation...")
    all_point_data = {}
    for var, data in means.items(): all_point_data[f"mean_{var}"] = data
    for key, data in reynolds_stresses.items(): all_point_data[get_primed_name(key, all_variables)] = data
    scalar_variance_keys = [f"{s}'{s}'" for s in scalar_variables]
    scalar_flux_keys = [f"{v}'{s}'" for s in scalar_variables for v in vel_variables]
    for key_format in scalar_variance_keys: all_point_data[get_primed_name(key_format, all_variables)] = scalar_stats[key_format]
    for key_format in scalar_flux_keys: all_point_data[get_primed_name(key_format, all_variables)] = scalar_stats[key_format]
    for key, data in triple_moments.items(): all_point_data[get_primed_name(key, all_variables)] = data

    x_min, x_max, z_min, z_max = unique_xz[:, 0].min(), unique_xz[:, 0].max(), unique_xz[:, 1].min(), unique_xz[:, 1].max()
    nx, nz = base_resolutionx, base_resolutionz
    print(f"Data aspect ratio preserved. New grid resolution: ({nx}, {nz})")
    x_coords, z_coords = np.linspace(x_min, x_max, nx), np.linspace(z_min, z_max, nz)
    grid_x, grid_z = np.meshgrid(x_coords, z_coords)
    data_vars = {}
    for field_name, point_values in tqdm(all_point_data.items(), desc="Interpolating fields"):
        interpolated_array = griddata(points=unique_xz, values=point_values, xi=(grid_x, grid_z), method='linear', fill_value=np.nan)
        data_vars[field_name] = (('z', 'x'), interpolated_array)

    print(f"💾 Constructing dataset and saving to NetCDF file '{output_filename}'...")
    ds = xr.Dataset(data_vars, coords={'x': ('x', x_coords), 'z': ('z', z_coords)})
    ds.x.attrs.update(units='m', long_name='Streamwise Coordinate')
    ds.z.attrs.update(units='m', long_name='Wall-Normal Coordinate')
    ds.attrs.update(title='Time-Spanwise Averaged Turbulence Statistics', source_directory=data_directory, creation_date=str(datetime.now()), processed_files=successful_file_count)
    ds.to_netcdf(output_filename)
    print("\n✅ All calculations are complete. Data saved to NetCDF.")

    if profile_plot_filename:
        print(f"📊 Generating and saving vertical wind profile to '{profile_plot_filename}'...")
        try:
            if 'mean_u' in ds:
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(7, 7))
                ds['mean_u'].mean(dim='x').plot(y='z', ax=ax, color='k', linewidth=2.5)
                ax.set_xlabel('Mean Streamwise Velocity, U [m s$^{-1}$]', fontsize=12)
                ax.set_ylabel('Height, z [m]', fontsize=12)
                ax.set_title('Time and Horizontally Averaged Wind Profile', fontsize=14, weight='bold')
                ax.set_ylim(bottom=0)
                plt.savefig(profile_plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Profile plot saved successfully to {profile_plot_filename}")
        except Exception as e: print(f"\n❌ ERROR: Could not generate profile plot. Reason: {e}")

    if second_moment_plot_dir:
        plot_second_moment_profiles(ds, second_moment_plot_dir)
        
    calculate_and_plot_spanwise_spectra(
        pvtu_files=pvtu_files,
        variables=['u', 'v', 'w', 'θ'],
        output_filename="turbulent_spanwise_spectra_time_averaged.png"
    )

if __name__ == '__main__':
    DATA_DIR = "/Users/simone/Work-local/Codes/Jexpresso/output/CompEuler/LESsmago/output-10240x10240x3000"
    FILE_PATTERN = "iter_*.pvtu"
    OUTPUT_NC_FILE = "turbulence_statistics_primed.nc"
    PROFILE_PLOT_FILE = "vertical_wind_profile.png"
    SECOND_MOMENT_PLOT_DIR = "second_moment_profiles"
    BASE_GRID_RESOLUTIONX = 512
    BASE_GRID_RESOLUTIONZ = 300
    START_STEP = 150
    END_STEP = 1000
    
    VELOCITY_VARS = ['u', 'v', 'w']
    SCALAR_VARS = ['θ', 'p']

    if not os.path.isdir(DATA_DIR):
        print(f"\n❌ Error: The specified data directory does not exist: {DATA_DIR}")
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
            end_step=END_STEP,
            profile_plot_filename=PROFILE_PLOT_FILE,
            second_moment_plot_dir=SECOND_MOMENT_PLOT_DIR
        )
