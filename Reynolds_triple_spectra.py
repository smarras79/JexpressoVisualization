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
import shutil

# --- NEW: Moved sanitize_var_name to the top level for global use ---
def sanitize_var_name(name):
    """Replace Unicode characters that cause NetCDF encoding issues"""
    return name.replace('θ', 'theta')

def get_primed_name(key, all_vars):
    # (This helper function remains the same)
    clean_key = key.replace("'", "")
    # --- MODIFIED: Uses the global sanitize_var_name function ---
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
    # (This plotting function remains the same)
    print(f"📊 Generating second-moment profile plots in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    z_coords = dataset['z'].values
    mean_vars = [v for v in dataset.data_vars if v.startswith('mean_')]
    moment_vars = [v for v in dataset.data_vars if v not in mean_vars]
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


def calculate_and_plot_spanwise_spectra(pvtu_files, variables, output_filename):
    # (This spectra function remains the same)
    print(f"\n⚡ Calculating and plotting time-averaged spanwise spectra...")
    if not pvtu_files: return
    try:
        mesh_container = pv.read(pvtu_files[0])
        mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
    except Exception as e:
        print(f"❌ ERROR: Could not read first file for spectra setup. Reason: {e}"); return
    points = mesh.points; bounds = mesh.bounds
    center_x = (bounds[0] + bounds[1]) / 2; center_z = (bounds[4] + bounds[5]) / 2
    closest_point_idx = np.argmin(np.sqrt((points[:, 0] - center_x)**2 + (points[:, 2] - center_z)**2))
    actual_x, _, actual_z = points[closest_point_idx]
    indices = np.where((np.isclose(points[:, 0], actual_x)) & (np.isclose(points[:, 2], actual_z)))[0]
    if len(indices) < 2: print("⚠️ Could not find a line of data in the y-direction. Skipping spectra."); return
    y_coords = points[indices, 1]; sort_order = np.argsort(y_coords)
    indices = indices[sort_order]; y_coords = y_coords[sort_order]
    Ly = y_coords.max() - y_coords.min(); Ny = len(y_coords)
    k_y = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)[:Ny//2]
    psd_sums = {var: np.zeros(Ny // 2) for var in variables}; spectra_success_count = 0
    for fpath in tqdm(pvtu_files, desc="Calculating spectra for each timestep"):
        try:
            mesh_container = pv.read(fpath)
            mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
            if not all(v in mesh.point_data for v in variables): continue
            for var in variables:
                fluctuations = mesh.point_data[var][indices] - np.mean(mesh.point_data[var][indices])
                fft_coeffs = np.fft.fft(fluctuations)
                psd = (np.abs(fft_coeffs)**2) / (Ny**2)
                psd_sums[var] += psd[:Ny//2]
            spectra_success_count += 1
        except Exception as e:
            tqdm.write(f"\n⚠️ WARNING: Could not process {os.path.basename(fpath)} for spectra. Reason: {e}")
            continue
    if spectra_success_count == 0: print("❌ ERROR: No files could be processed for spectra. Aborting plot."); return
    avg_psd = {var: psd_sums[var] / spectra_success_count for var in variables}
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'u': 'r', 'v': 'g', 'w': 'b', 'θ': 'orange'}
    for var in variables:
        var_label = var.replace('θ', r'theta'); label_str = f"$E_{{{var_label}}}(k_y)$"
        ax.loglog(k_y, avg_psd[var], color=colors.get(var, 'k'), label=label_str)
    if len(k_y) > 20:
        k_ref_start_index = max(5, int(len(k_y) * 0.1)); k_ref_end_index = min(len(k_y) - 10, int(len(k_y) * 0.4))
        if k_ref_start_index < k_ref_end_index:
            k_ref = np.array([k_y[k_ref_start_index], k_y[k_ref_end_index]])
            ref_energy = avg_psd['u'][k_ref_start_index]
            ax.plot(k_ref, ref_energy * (k_ref/k_ref[0])**(-5/3), 'k--', label="$k_y^{-5/3}$ Slope")
    ax.set_xlabel('Spanwise Wavenumber, $k_y$ [m$^{-1}$]', fontsize=14); ax.set_ylabel('Time-Averaged Power Spectral Density, $E(k_y)$', fontsize=14)
    ax.set_title(f'1D Spanwise Turbulent Spectra at z={actual_z:.2f}m (Avg. of {spectra_success_count} files)', fontsize=16, weight='bold')
    ax.set_ylim(bottom=1e-9); ax.legend(fontsize=12); ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"✅ Time-averaged spectra plot saved successfully to {output_filename}")
    
    # --- NEW: Save spectrum data to NetCDF ---
    print(f"💾 Saving spectrum data to NetCDF...")
    nc_filename = output_filename.replace('.png', '.nc')
    
    # Create dataset with wavenumber as coordinate
    data_vars = {}
    for var in variables:
        sanitized_var = sanitize_var_name(var)
        data_vars[f'E_{sanitized_var}'] = (('k_y',), avg_psd[var])
    
    ds_spectrum = xr.Dataset(
        data_vars,
        coords={'k_y': ('k_y', k_y)}
    )
    
    # Add metadata attributes
    ds_spectrum.k_y.attrs.update({
        'units': 'm^-1',
        'long_name': 'Spanwise wavenumber',
        'description': 'Wavenumber in the spanwise (y) direction'
    })
    
    for var in variables:
        sanitized_var = sanitize_var_name(var)
        ds_spectrum[f'E_{sanitized_var}'].attrs.update({
            'units': 'm^3 s^-2',
            'long_name': f'Power spectral density of {var}',
            'description': f'Time-averaged 1D energy spectrum for {var} velocity component'
        })
    
    ds_spectrum.attrs.update({
        'title': 'Time-Averaged 1D Spanwise Turbulent Energy Spectra',
        'description': f'1D spanwise spectra extracted at x={actual_x:.2f}m, z={actual_z:.2f}m',
        'extraction_location_x': actual_x,
        'extraction_location_z': actual_z,
        'spanwise_extent_Ly': Ly,
        'number_of_points': Ny,
        'number_of_timesteps_averaged': spectra_success_count,
        'creation_date': str(datetime.now())
    })
    
    ds_spectrum.to_netcdf(nc_filename)
    print(f"✅ Spectrum data saved to {nc_filename}")
    
def calculate_and_plot_spanwise_spectra_old(pvtu_files, variables, output_filename):
    # (This spectra function remains the same)
    print(f"\n⚡ Calculating and plotting time-averaged spanwise spectra...")
    if not pvtu_files: return
    try:
        mesh_container = pv.read(pvtu_files[0])
        mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
    except Exception as e:
        print(f"❌ ERROR: Could not read first file for spectra setup. Reason: {e}"); return
    points = mesh.points; bounds = mesh.bounds
    center_x = (bounds[0] + bounds[1]) / 2; center_z = (bounds[4] + bounds[5]) / 2
    closest_point_idx = np.argmin(np.sqrt((points[:, 0] - center_x)**2 + (points[:, 2] - center_z)**2))
    actual_x, _, actual_z = points[closest_point_idx]
    indices = np.where((np.isclose(points[:, 0], actual_x)) & (np.isclose(points[:, 2], actual_z)))[0]
    if len(indices) < 2: print("⚠️ Could not find a line of data in the y-direction. Skipping spectra."); return
    y_coords = points[indices, 1]; sort_order = np.argsort(y_coords)
    indices = indices[sort_order]; y_coords = y_coords[sort_order]
    Ly = y_coords.max() - y_coords.min(); Ny = len(y_coords)
    k_y = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)[:Ny//2]
    psd_sums = {var: np.zeros(Ny // 2) for var in variables}; spectra_success_count = 0
    for fpath in tqdm(pvtu_files, desc="Calculating spectra for each timestep"):
        try:
            mesh_container = pv.read(fpath)
            mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
            if not all(v in mesh.point_data for v in variables): continue
            for var in variables:
                fluctuations = mesh.point_data[var][indices] - np.mean(mesh.point_data[var][indices])
                fft_coeffs = np.fft.fft(fluctuations)
                psd = (np.abs(fft_coeffs)**2) / (Ny**2)
                psd_sums[var] += psd[:Ny//2]
            spectra_success_count += 1
        except Exception as e:
            tqdm.write(f"\n⚠️ WARNING: Could not process {os.path.basename(fpath)} for spectra. Reason: {e}")
            continue
    if spectra_success_count == 0: print("❌ ERROR: No files could be processed for spectra. Aborting plot."); return
    avg_psd = {var: psd_sums[var] / spectra_success_count for var in variables}
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'u': 'r', 'v': 'g', 'w': 'b', 'θ': 'orange'}
    for var in variables:
        var_label = var.replace('θ', r'theta'); label_str = f"$E_{{{var_label}}}(k_y)$"
        ax.loglog(k_y, avg_psd[var], color=colors.get(var, 'k'), label=label_str)
    if len(k_y) > 20:
        k_ref_start_index = max(5, int(len(k_y) * 0.1)); k_ref_end_index = min(len(k_y) - 10, int(len(k_y) * 0.4))
        if k_ref_start_index < k_ref_end_index:
            k_ref = np.array([k_y[k_ref_start_index], k_y[k_ref_end_index]])
            ref_energy = avg_psd['u'][k_ref_start_index]
            ax.plot(k_ref, ref_energy * (k_ref/k_ref[0])**(-5/3), 'k--', label="$k_y^{-5/3}$ Slope")
    ax.set_xlabel('Spanwise Wavenumber, $k_y$ [m$^{-1}$]', fontsize=14); ax.set_ylabel('Time-Averaged Power Spectral Density, $E(k_y)$', fontsize=14)
    ax.set_title(f'1D Spanwise Turbulent Spectra at z={actual_z:.2f}m (Avg. of {spectra_success_count} files)', fontsize=16, weight='bold')
    ax.set_ylim(bottom=1e-9); ax.legend(fontsize=12); ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"✅ Time-averaged spectra plot saved successfully to {output_filename}")


def interpolate_and_save_3d_snapshot(pvtu_files, variables, output_filename):
    # (This 3D snapshot function remains the same)
    print(f"\n🧊 Preparing 3D snapshot for '{output_filename}'...")
    if not pvtu_files: print("⚠️ No files available for 3D snapshot. Skipping."); return
    snapshot_path = pvtu_files[len(pvtu_files) // 2]
    print(f"   Using snapshot: {os.path.basename(snapshot_path)}")
    try:
        mesh_container = pv.read(snapshot_path)
        mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
    except Exception as e:
        print(f"❌ ERROR: Could not read snapshot file. Reason: {e}"); return
    bounds = mesh.bounds; n_points_orig = mesh.n_points
    dims = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    aspect = dims / dims.min()
    nx, ny, nz = (aspect * (n_points_orig / aspect.prod())**(1/3)).round().astype(int)
    print(f"   Interpolating onto a {nx}x{ny}x{nz} structured grid...")
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dims[0]/nx, dims[1]/ny, dims[2]/nz), origin=(bounds[0], bounds[2], bounds[4]))
    interpolated_grid = grid.interpolate(mesh, radius=dims.min()*0.5)
    x = interpolated_grid.x; y = interpolated_grid.y; z = interpolated_grid.z
    ds_3d = xr.Dataset(coords={'x': ('x', x), 'y': ('y', y), 'z': ('z', z)})
    for var in variables:
        if var in interpolated_grid.point_data:
            # --- MODIFIED: Uses the global sanitize_var_name function for consistency ---
            sanitized_name = sanitize_var_name(var)
            data_3d = interpolated_grid.point_data[var].reshape(nz, ny, nx)
            ds_3d[sanitized_name] = (('z', 'y', 'x'), data_3d)
    print(f"   Saving 3D snapshot to '{output_filename}'...")
    ds_3d.to_netcdf(output_filename)
    print(f"✅ 3D snapshot saved successfully.")

def process_and_save_instantaneous_slices(
    pvtu_files,
    output_dir,
    all_variables,
    base_resolutionx,
    base_resolutionz,
    x_slice_loc,
    y_slice_loc,
    z_slice_loc
):
    """
    NEW: Loops through every file, extracts slices, and saves each to a separate NC file.
    """
    print(f"\n🔪 Extracting instantaneous 2D slices for each timestep...")
    os.makedirs(output_dir, exist_ok=True)

    # Use a regex to extract the step number for unique filenames
    pattern = re.compile(r'(\d+)\.pvtu$')

    for fpath in tqdm(pvtu_files, desc="Processing instantaneous slices"):
        match = pattern.search(os.path.basename(fpath))
        if not match: continue
        step = int(match.group(1))

        try:
            mesh_container = pv.read(fpath)
            mesh = mesh_container.combine() if isinstance(mesh_container, pv.MultiBlock) else mesh_container
            bounds = mesh.bounds
            center = mesh.center

            data_vars = {}
            coords = {}

            slice_locations = {'x': x_slice_loc, 'y': y_slice_loc, 'z': z_slice_loc}

            for dim, loc in slice_locations.items():
                if loc is not None:
                    if dim == 'x': normal, origin = ([1,0,0], [loc, center[1], center[2]])
                    elif dim == 'y': normal, origin = ([0,1,0], [center[0], loc, center[2]])
                    else: normal, origin = ([0,0,1], [center[0], center[1], loc])

                    plane = mesh.slice(normal=normal, origin=origin)
                    if plane.n_points > 0:
                        for var in all_variables:
                            if var in plane.point_data:
                                # --- MODIFIED: Sanitize variable name before using it ---
                                sanitized_var = sanitize_var_name(var)
                                var_name = f"{sanitized_var}_slice_{dim}{int(loc)}"
                                if dim == 'x': # yz plane
                                    points, c1_name, c2_name = plane.points[:, 1:], 'y', 'z'
                                    c1_coords = np.linspace(bounds[2], bounds[3], base_resolutionz); c2_coords = np.linspace(bounds[4], bounds[5], base_resolutionz)
                                elif dim == 'y': # xz plane
                                    points, c1_name, c2_name = plane.points[:, [0,2]], 'x', 'z'
                                    c1_coords = np.linspace(bounds[0], bounds[1], base_resolutionx); c2_coords = np.linspace(bounds[4], bounds[5], base_resolutionz)
                                else: # xy plane
                                    points, c1_name, c2_name = plane.points[:, :2], 'x', 'y'
                                    c1_coords = np.linspace(bounds[0], bounds[1], base_resolutionx); c2_coords = np.linspace(bounds[2], bounds[3], base_resolutionx)

                                grid_c1, grid_c2 = np.meshgrid(c1_coords, c2_coords)
                                data_slice = griddata(points, plane[var], (grid_c1, grid_c2), method='linear')

                                c1_coord_name = f"{c1_name}_{dim}slice"; c2_coord_name = f"{c2_name}_{dim}slice"
                                data_vars[var_name] = ((c2_coord_name, c1_coord_name), data_slice)
                                coords[c1_coord_name] = (c1_coord_name, c1_coords)
                                coords[c2_coord_name] = (c2_coord_name, c2_coords)

            if data_vars:
                ds_slice = xr.Dataset(data_vars, coords=coords)
                slice_filename = os.path.join(output_dir, f"slices_step_{step}.nc")
                ds_slice.to_netcdf(slice_filename)

        except Exception as e:
            tqdm.write(f"\n⚠️ WARNING: Could not process slices for {os.path.basename(fpath)}. Reason: {e}")

    print("✅ Instantaneous slice processing complete.")

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
    second_moment_plot_dir: str
):
    """
    Main function to calculate ONLY time-averaged statistics and save them.
    """
    all_variables = vel_variables + scalar_variables
    print("🔍 Finding and filtering data files for averaging...")
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

    # --- MODIFIED: Removed the local sanitize_var_name function as it's now global ---

    for var, data in means.items():
        sanitized_name = sanitize_var_name(f"mean_{var}")
        all_point_data[sanitized_name] = data

    for key, data in reynolds_stresses.items():
        sanitized_name = sanitize_var_name(get_primed_name(key, all_variables))
        all_point_data[sanitized_name] = data

    scalar_variance_keys = [f"{s}'{s}'" for s in scalar_variables]
    scalar_flux_keys = [f"{v}'{s}'" for s in scalar_variables for v in vel_variables]

    for key_format in scalar_variance_keys:
        sanitized_name = sanitize_var_name(get_primed_name(key_format, all_variables))
        all_point_data[sanitized_name] = scalar_stats[key_format]

    for key_format in scalar_flux_keys:
        sanitized_name = sanitize_var_name(get_primed_name(key_format, all_variables))
        all_point_data[sanitized_name] = scalar_stats[key_format]

    for key, data in triple_moments.items():
        sanitized_name = sanitize_var_name(get_primed_name(key, all_variables))
        all_point_data[sanitized_name] = data

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
    ds.x.attrs.update(units='m', long_name='Streamwise Coordinate'); ds.z.attrs.update(units='m', long_name='Wall-Normal Coordinate')
    ds.attrs.update(title='Time-Averaged Turbulence Statistics', source_directory=data_directory, creation_date=str(datetime.now()), processed_files=successful_file_count)
    ds.to_netcdf(output_filename)
    print("\n✅ Time-averaged calculations are complete. Data saved to NetCDF.")

    # --- Plotting from the averaged data ---
    if profile_plot_filename:
        print(f"📊 Generating and saving vertical wind profile to '{profile_plot_filename}'...")
        try:
            if 'mean_u' in ds:
                plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(7, 7))
                ds['mean_u'].mean(dim='x').plot(y='z', ax=ax, color='k', linewidth=2.5)
                ax.set_xlabel('Mean Streamwise Velocity, U [m s$^{-1}$]', fontsize=12); ax.set_ylabel('Height, z [m]', fontsize=12)
                ax.set_title('Time and Horizontally Averaged Wind Profile', fontsize=14, weight='bold')
                ax.set_ylim(bottom=0); plt.savefig(profile_plot_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
                print(f"✅ Profile plot saved successfully to {profile_plot_filename}")
        except Exception as e: print(f"\n❌ ERROR: Could not generate profile plot. Reason: {e}")

    if second_moment_plot_dir:
        plot_second_moment_profiles(ds, second_moment_plot_dir)

    return pvtu_files

if __name__ == '__main__':
    # --- CONFIGURATION ---
    #DATA_DIR = "/Users/simone/Work-local/Codes/Jexpresso/output/CompEuler/LESsmago/64x64x36_5kmX5kmX3km"
    #DATA_DIR = "/scratch/smarras/smarras/output/64x64x48_5kmX5kmX3km_128cores/CompEuler/LESsmago/output/"
    #DATA_DIR = "/scratch/smarras/smarras/output/32x32x24_5kmX5kmX3km/CompEuler/LESsmago32x32x24/output/"
    #DATA_DIR = "/scratch/smarras/smarras/output/64x64x36_5kmX5kmX3km/CompEuler-CFL-at2200s/LESsmago/output/"
    DATA_DIR = "/scratch/smarras/hw59/output/LESICP2_80x40x45_nop6_10kmX5kmX2dot8km_2/CompEuler/LESICP2/output"
    FILE_PATTERN = "iter_*.pvtu"
    BASE_GRID_RESOLUTIONX = 512
    BASE_GRID_RESOLUTIONZ = 300
    START_STEP = 400
    END_STEP = 659

    OUTPUT_NC_AVERAGED_FILE = DATA_DIR + "/400to659turbulence_statistics_averaged.nc" # File for all time-averaged data
    PROFILE_PLOT_FILE = DATA_DIR + "/400to659vertical_wind_profile.png"
    SECOND_MOMENT_PLOT_DIR = DATA_DIR + "/400to659second_moment_profiles"
    SPECTRA_PLOT_FILE = DATA_DIR + "/400to659turbulent_spanwise_spectra_time_averaged.png"
    SNAPSHOT_3D_NC_FILE = DATA_DIR + "/400to659snapshot_3d.nc"
    INSTANTANEOUS_SLICE_DIR = DATA_DIR + "/400to659instantaneous_slice/" # NEW: Directory for per-timestep slice files
    

    ## --- 1. Create a single timestamped directory for all run outputs ---
    ## This line creates a new directory like 'DATA_DIR/20251008_103118_run_output'
    #output_dir = os.path.join(DATA_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run_output")
    #os.makedirs(output_dir, exist_ok=True)
    #
    ## --- 2. Define all file and directory paths inside the new output directory ---
    ## All generated data will now be neatly saved in the folder created above.
    #OUTPUT_NC_AVERAGED_FILE = os.path.join(output_dir, "turbulence_statistics_averaged.nc")
    #PROFILE_PLOT_FILE = os.path.join(output_dir, "vertical_wind_profile.png")
    #SECOND_MOMENT_PLOT_DIR = os.path.join(output_dir, "second_moment_profiles")
    #SPECTRA_PLOT_FILE = os.path.join(output_dir, "turbulent_spanwise_spectra_time_averaged.png")
    #SNAPSHOT_3D_NC_FILE = os.path.join(output_dir, "snapshot_3d.nc")
    #INSTANTANEOUS_SLICE_DIR = os.path.join(output_dir, "instantaneous_slice/")
    
    # --- SLICE & 3D SNAPSHOT CONFIGURATION ---
    WRITE_3D_SNAPSHOT = False
    X_SLICE_LOC = 2560.0
    Y_SLICE_LOC = 2560.0
    Z_SLICE_LOC = 100.0

    # --- VARIABLE DEFINITION ---
    VELOCITY_VARS = ['u', 'v', 'w']
    SCALAR_VARS = ['θ', 'p']

    # --- EXECUTION ---
    if not os.path.isdir(DATA_DIR):
        print(f"\n❌ Error: The specified data directory does not exist: {DATA_DIR}")
    else:
        # 1. Calculate and save all time-averaged statistics
        processed_files = calculate_and_save_averaged_stats(
            data_directory=DATA_DIR, file_pattern=FILE_PATTERN, output_filename=OUTPUT_NC_AVERAGED_FILE,
            vel_variables=VELOCITY_VARS, scalar_variables=SCALAR_VARS, coord_precision=6,
            base_resolutionx=BASE_GRID_RESOLUTIONX, base_resolutionz=BASE_GRID_RESOLUTIONZ,
            start_step=START_STEP, end_step=END_STEP,
            profile_plot_filename=PROFILE_PLOT_FILE, second_moment_plot_dir=SECOND_MOMENT_PLOT_DIR
        )

        # 2. Process and save instantaneous 2D slices for each timestep
        if INSTANTANEOUS_SLICE_DIR:
            process_and_save_instantaneous_slices(
                pvtu_files=processed_files, output_dir=INSTANTANEOUS_SLICE_DIR,
                all_variables=VELOCITY_VARS + SCALAR_VARS,
                base_resolutionx=BASE_GRID_RESOLUTIONX, base_resolutionz=BASE_GRID_RESOLUTIONZ,
                x_slice_loc=X_SLICE_LOC, y_slice_loc=Y_SLICE_LOC, z_slice_loc=Z_SLICE_LOC
            )

        # 3. Calculate and plot time-averaged spectra
        if SPECTRA_PLOT_FILE:
            calculate_and_plot_spanwise_spectra(
                pvtu_files=processed_files, variables=['u', 'v', 'w', 'θ'], output_filename=SPECTRA_PLOT_FILE
            )

        # 4. Optionally, save a full 3D snapshot
        if WRITE_3D_SNAPSHOT and SNAPSHOT_3D_NC_FILE:
            interpolate_and_save_3d_snapshot(
                pvtu_files=processed_files, variables=VELOCITY_VARS + SCALAR_VARS, output_filename=SNAPSHOT_3D_NC_FILE
            )
            
