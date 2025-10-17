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


def calculate_and_plot_spanwise_spectra(pvtu_files, variables, output_filename, max_z=None):
    # --- ENHANCED: Added max_z parameter to limit vertical extent ---
    print(f"\n⚡ Calculating and plotting time-averaged spanwise spectra...")
    if max_z is not None:
        print(f"   📏 Limiting analysis to z ≤ {max_z:.1f} m")
    if not pvtu_files: return
    try:
        mesh_container = pv.read(pvtu_files[0])
        mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
    except Exception as e:
        print(f"❌ ERROR: Could not read first file for spectra setup. Reason: {e}"); return
    
    points = mesh.points; bounds = mesh.bounds
    
    # --- NEW: Apply max_z filter if specified ---
    if max_z is not None:
        valid_z_mask = points[:, 2] <= max_z
        if not np.any(valid_z_mask):
            print(f"❌ ERROR: No points found with z ≤ {max_z}. Skipping spectra.")
            return
        points = points[valid_z_mask]
        # Update bounds based on filtered points
        bounds = [points[:, 0].min(), points[:, 0].max(),
                  points[:, 1].min(), points[:, 1].max(),
                  points[:, 2].min(), points[:, 2].max()]
    
    center_x = (bounds[0] + bounds[1]) / 2
    center_z = (bounds[4] + bounds[5]) / 2  # This will now respect max_z if set
    
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
            
            # --- NEW: Apply max_z filter to each timestep ---
            if max_z is not None:
                mesh_points = mesh.points
                valid_z_mask = mesh_points[:, 2] <= max_z
                if not np.any(valid_z_mask): continue
                # Need to find indices in the filtered mesh
                filtered_points = mesh_points[valid_z_mask]
                filtered_indices = np.where((np.isclose(filtered_points[:, 0], actual_x)) & 
                                           (np.isclose(filtered_points[:, 2], actual_z)))[0]
                if len(filtered_indices) < 2: continue
                
                for var in variables:
                    var_data = mesh.point_data[var][valid_z_mask]
                    fluctuations = var_data[filtered_indices] - np.mean(var_data[filtered_indices])
                    fft_coeffs = np.fft.fft(fluctuations)
                    psd = (np.abs(fft_coeffs)**2) / (Ny**2)
                    psd_sums[var] += psd[:Ny//2]
            else:
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
    
    title_suffix = f" (z ≤ {max_z:.1f}m)" if max_z is not None else ""
    ax.set_xlabel('Spanwise Wavenumber, $k_y$ [m$^{-1}$]', fontsize=14)
    ax.set_ylabel('Time-Averaged Power Spectral Density, $E(k_y)$', fontsize=14)
    ax.set_title(f'1D Spanwise Turbulent Spectra at z={actual_z:.2f}m{title_suffix} (Avg. of {spectra_success_count} files)', 
                 fontsize=16, weight='bold')
    ax.set_ylim(bottom=1e-9); ax.legend(fontsize=12); ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"✅ Time-averaged spectra plot saved successfully to {output_filename}")
    
    # --- Save spectrum data to NetCDF ---
    print(f"💾 Saving spectrum data to NetCDF...")
    nc_filename = output_filename.replace('.png', '.nc')
    
    data_vars = {}
    for var in variables:
        sanitized_var = sanitize_var_name(var)
        data_vars[f'E_{sanitized_var}'] = (('k_y',), avg_psd[var])
    
    ds_spectrum = xr.Dataset(data_vars, coords={'k_y': ('k_y', k_y)})
    
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
        'max_z_filter': max_z if max_z is not None else 'None',
        'creation_date': str(datetime.now())
    })
    
    ds_spectrum.to_netcdf(nc_filename)
    print(f"✅ Spectrum data saved to {nc_filename}")


def interpolate_and_save_3d_snapshot(pvtu_files, variables, output_filename, max_z=None):
    # --- ENHANCED: Added max_z parameter ---
    print(f"\n🧊 Preparing 3D snapshot for '{output_filename}'...")
    if max_z is not None:
        print(f"   📏 Limiting snapshot to z ≤ {max_z:.1f} m")
    if not pvtu_files: print("⚠️ No files available for 3D snapshot. Skipping."); return
    snapshot_path = pvtu_files[len(pvtu_files) // 2]
    print(f"   Using snapshot: {os.path.basename(snapshot_path)}")
    try:
        mesh_container = pv.read(snapshot_path)
        mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
    except Exception as e:
        print(f"❌ ERROR: Could not read snapshot file. Reason: {e}"); return
    
    # --- NEW: Apply max_z filter ---
    if max_z is not None:
        points = mesh.points
        valid_mask = points[:, 2] <= max_z
        if not np.any(valid_mask):
            print(f"❌ ERROR: No points with z ≤ {max_z}. Aborting 3D snapshot.")
            return
        mesh = mesh.extract_points(valid_mask)
        print(f"   Filtered to {mesh.n_points} points (z ≤ {max_z:.1f}m)")
    
    bounds = mesh.bounds; n_points_orig = mesh.n_points
    dims = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    aspect = dims / dims.min()
    nx, ny, nz = (aspect * (n_points_orig / aspect.prod())**(1/3)).round().astype(int)
    print(f"   Interpolating onto a {nx}x{ny}x{nz} structured grid...")
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dims[0]/nx, dims[1]/ny, dims[2]/nz), 
                        origin=(bounds[0], bounds[2], bounds[4]))
    interpolated_grid = grid.interpolate(mesh, radius=dims.min()*0.5)
    x = interpolated_grid.x; y = interpolated_grid.y; z = interpolated_grid.z
    ds_3d = xr.Dataset(coords={'x': ('x', x), 'y': ('y', y), 'z': ('z', z)})
    for var in variables:
        if var in interpolated_grid.point_data:
            sanitized_name = sanitize_var_name(var)
            data_3d = interpolated_grid.point_data[var].reshape(nz, ny, nx)
            ds_3d[sanitized_name] = (('z', 'y', 'x'), data_3d)
    
    # Add max_z to metadata
    ds_3d.attrs['max_z_filter'] = max_z if max_z is not None else 'None'
    
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
    z_slice_loc,
    max_z=None
):
    """
    NEW: Loops through every file, extracts slices, and saves each to a separate NC file.
    ENHANCED: Added max_z parameter to limit vertical extent
    """
    print(f"\n🔪 Extracting instantaneous 2D slices for each timestep...")
    if max_z is not None:
        print(f"   📏 Limiting data to z ≤ {max_z:.1f} m")
    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r'(\d+)\.pvtu$')

    for fpath in tqdm(pvtu_files, desc="Processing instantaneous slices"):
        match = pattern.search(os.path.basename(fpath))
        if not match: continue
        step = int(match.group(1))

        try:
            mesh_container = pv.read(fpath)
            mesh = mesh_container.combine() if isinstance(mesh_container, pv.MultiBlock) else mesh_container
            
            # --- NEW: Apply max_z filter ---
            if max_z is not None:
                points = mesh.points
                valid_mask = points[:, 2] <= max_z
                if not np.any(valid_mask): continue
                mesh = mesh.extract_points(valid_mask)
            
            bounds = mesh.bounds
            center = mesh.center

            data_vars = {}
            coords = {}

            slice_locations = {'x': x_slice_loc, 'y': y_slice_loc, 'z': z_slice_loc}

            for dim, loc in slice_locations.items():
                if loc is not None:
                    # Skip z-slices that exceed max_z
                    if dim == 'z' and max_z is not None and loc > max_z:
                        continue
                    
                    if dim == 'x': normal, origin = ([1,0,0], [loc, center[1], center[2]])
                    elif dim == 'y': normal, origin = ([0,1,0], [center[0], loc, center[2]])
                    else: normal, origin = ([0,0,1], [center[0], center[1], loc])

                    plane = mesh.slice(normal=normal, origin=origin)
                    if plane.n_points > 0:
                        for var in all_variables:
                            if var in plane.point_data:
                                sanitized_var = sanitize_var_name(var)
                                var_name = f"{sanitized_var}_slice_{dim}{int(loc)}"
                                if dim == 'x': # yz plane
                                    points, c1_name, c2_name = plane.points[:, 1:], 'y', 'z'
                                    c1_coords = np.linspace(bounds[2], bounds[3], base_resolutionz)
                                    c2_coords = np.linspace(bounds[4], bounds[5], base_resolutionz)
                                elif dim == 'y': # xz plane
                                    points, c1_name, c2_name = plane.points[:, [0,2]], 'x', 'z'
                                    c1_coords = np.linspace(bounds[0], bounds[1], base_resolutionx)
                                    c2_coords = np.linspace(bounds[4], bounds[5], base_resolutionz)
                                else: # xy plane
                                    points, c1_name, c2_name = plane.points[:, :2], 'x', 'y'
                                    c1_coords = np.linspace(bounds[0], bounds[1], base_resolutionx)
                                    c2_coords = np.linspace(bounds[2], bounds[3], base_resolutionx)

                                grid_c1, grid_c2 = np.meshgrid(c1_coords, c2_coords)
                                data_slice = griddata(points, plane[var], (grid_c1, grid_c2), method='linear')

                                c1_coord_name = f"{c1_name}_{dim}slice"
                                c2_coord_name = f"{c2_name}_{dim}slice"
                                data_vars[var_name] = ((c2_coord_name, c1_coord_name), data_slice)
                                coords[c1_coord_name] = (c1_coord_name, c1_coords)
                                coords[c2_coord_name] = (c2_coord_name, c2_coords)

            if data_vars:
                ds_slice = xr.Dataset(data_vars, coords=coords)
                ds_slice.attrs['max_z_filter'] = max_z if max_z is not None else 'None'
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
    second_moment_plot_dir: str,
    max_z: float = None
):
    """
    Main function to calculate ONLY time-averaged statistics and save them.
    ENHANCED: Added max_z parameter to limit vertical extent of analysis
    
    Parameters:
    -----------
    max_z : float, optional
        Maximum z-coordinate (height) to include in analysis. 
        Points with z > max_z will be excluded. 
        If None, all points are included.
    """
    all_variables = vel_variables + scalar_variables
    print("🔍 Finding and filtering data files for averaging...")
    if max_z is not None:
        print(f"   📏 Will limit analysis to z ≤ {max_z:.1f} m")
    
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
    
    # Read first mesh and apply z-filter
    first_mesh = pv.read(pvtu_files[0])
    points_3d = first_mesh.points
    
    # --- NEW: Apply max_z filter ---
    if max_z is not None:
        z_mask = points_3d[:, 2] <= max_z
        n_excluded = np.sum(~z_mask)
        n_total = len(z_mask)
        print(f"   Excluding {n_excluded}/{n_total} points with z > {max_z:.1f} m ({100*n_excluded/n_total:.1f}%)")
        points_3d = points_3d[z_mask]
        if len(points_3d) == 0:
            raise ValueError(f"No points remain after filtering with max_z={max_z}. Check your domain bounds.")
    
    xz_coords = np.round(points_3d[:, [0, 2]], decimals=coord_precision)
    unique_xz, inverse_indices_ref = np.unique(xz_coords, axis=0, return_inverse=True)
    num_xz_points = len(unique_xz)
    print(f"Identified {num_xz_points} unique (x, z) points for averaging (after z-filtering).")
    
    print("\n--- Pass 1 of 2: Calculating sums for mean quantities ---")
    sums = {var: np.zeros(num_xz_points, dtype=np.float64) for var in all_variables}
    per_location_counts = np.zeros(num_xz_points, dtype=np.int64)
    successful_file_count = 0
    
    for fpath in tqdm(pvtu_files, desc="Processing files for means"):
        try:
            mesh = pv.read(fpath)
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in all_variables): 
                continue
            
            # --- NEW: Apply max_z filter to each file ---
            mesh_points = mesh.points
            if max_z is not None:
                z_mask = mesh_points[:, 2] <= max_z
                if not np.any(z_mask): continue
                mesh_points = mesh_points[z_mask]
                filtered_data = {var: mesh.point_data[var][z_mask] for var in all_variables}
            else:
                filtered_data = {var: mesh.point_data[var] for var in all_variables}
            
            # Recompute inverse indices for this file's filtered points
            xz_coords_file = np.round(mesh_points[:, [0, 2]], decimals=coord_precision)
            # Map to the reference unique_xz
            inverse_indices = np.zeros(len(xz_coords_file), dtype=np.int64)
            for i, xz in enumerate(xz_coords_file):
                # Find matching index in unique_xz
                matches = np.where(np.all(np.isclose(unique_xz, xz), axis=1))[0]
                if len(matches) > 0:
                    inverse_indices[i] = matches[0]
            
            for var in all_variables:
                np.add.at(sums[var], inverse_indices, filtered_data[var])
            
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
    stress_sums = {key: np.zeros(num_xz_points, dtype=np.float64) 
                   for key in [f"{v1}'{v2}'" for v1, v2 in combinations_with_replacement(vel_variables, 2)]}
    triple_sums = {key: np.zeros(num_xz_points, dtype=np.float64) 
                   for key in [f"{v1}'{v2}'{v3}'" for v1, v2, v3 in combinations_with_replacement(vel_variables, 3)] + 
                              [f"{v1}'{v2}'{s}'" for s in scalar_variables for v1, v2 in combinations_with_replacement(vel_variables, 2)]}
    scalar_sums = {key: np.zeros(num_xz_points, dtype=np.float64) 
                   for key in [f"{s}'{s}'" for s in scalar_variables] + 
                              [f"{v}'{s}'" for s in scalar_variables for v in vel_variables]}
    
    for fpath in tqdm(pvtu_files, desc="Processing files for fluctuations"):
        try:
            mesh = pv.read(fpath)
            if mesh.n_points == 0 or not all(v in mesh.point_data for v in all_variables): 
                continue
            
            # --- NEW: Apply max_z filter ---
            mesh_points = mesh.points
            if max_z is not None:
                z_mask = mesh_points[:, 2] <= max_z
                if not np.any(z_mask): continue
                mesh_points = mesh_points[z_mask]
                filtered_data = {var: mesh.point_data[var][z_mask] for var in all_variables}
            else:
                filtered_data = {var: mesh.point_data[var] for var in all_variables}
            
            # Recompute inverse indices
            xz_coords_file = np.round(mesh_points[:, [0, 2]], decimals=coord_precision)
            inverse_indices = np.zeros(len(xz_coords_file), dtype=np.int64)
            for i, xz in enumerate(xz_coords_file):
                matches = np.where(np.all(np.isclose(unique_xz, xz), axis=1))[0]
                if len(matches) > 0:
                    inverse_indices[i] = matches[0]
            
            fluctuations = {var: filtered_data[var] - means[var][inverse_indices] for var in all_variables}
            
            for v1, v2 in combinations_with_replacement(vel_variables, 2):
                np.add.at(stress_sums[f"{v1}'{v2}'"], inverse_indices, 
                         fluctuations[v1] * fluctuations[v2])
            
            for s in scalar_variables:
                np.add.at(scalar_sums[f"{s}'{s}'"], inverse_indices, 
                         fluctuations[s] * fluctuations[s])
                for v in vel_variables:
                    np.add.at(scalar_sums[f"{v}'{s}'"], inverse_indices, 
                             fluctuations[v] * fluctuations[s])
            
            for v1, v2, v3 in combinations_with_replacement(vel_variables, 3):
                np.add.at(triple_sums[f"{v1}'{v2}'{v3}'"], inverse_indices, 
                         fluctuations[v1] * fluctuations[v2] * fluctuations[v3])
            
            for s in scalar_variables:
                for v1, v2 in combinations_with_replacement(vel_variables, 2):
                    np.add.at(triple_sums[f"{v1}'{v2}'{s}'"], inverse_indices, 
                             fluctuations[v1] * fluctuations[v2] * fluctuations[s])
        except Exception:
            continue
    
    reynolds_stresses = {key: np.divide(stress_sums[key], total_counts, where=total_counts > 0) 
                        for key in stress_sums}
    triple_moments = {key: np.divide(triple_sums[key], total_counts, where=total_counts > 0) 
                     for key in triple_sums}
    scalar_stats = {key: np.divide(scalar_sums[key], total_counts, where=total_counts > 0) 
                   for key in scalar_sums}

    print(f"\n📈 Performing robust Delaunay interpolation...")
    all_point_data = {}

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

    x_min, x_max = unique_xz[:, 0].min(), unique_xz[:, 0].max()
    z_min, z_max = unique_xz[:, 1].min(), unique_xz[:, 1].max()
    nx, nz = base_resolutionx, base_resolutionz
    print(f"Data aspect ratio preserved. New grid resolution: ({nx}, {nz})")
    x_coords = np.linspace(x_min, x_max, nx)
    z_coords = np.linspace(z_min, z_max, nz)
    grid_x, grid_z = np.meshgrid(x_coords, z_coords)
    
    data_vars = {}
    for field_name, point_values in tqdm(all_point_data.items(), desc="Interpolating fields"):
        interpolated_array = griddata(points=unique_xz, values=point_values, 
                                     xi=(grid_x, grid_z), method='linear', fill_value=np.nan)
        data_vars[field_name] = (('z', 'x'), interpolated_array)

    print(f"💾 Constructing dataset and saving to NetCDF file '{output_filename}'...")
    ds = xr.Dataset(data_vars, coords={'x': ('x', x_coords), 'z': ('z', z_coords)})
    ds.x.attrs.update(units='m', long_name='Streamwise Coordinate')
    ds.z.attrs.update(units='m', long_name='Wall-Normal Coordinate')
    ds.attrs.update(
        title='Time-Averaged Turbulence Statistics',
        source_directory=data_directory,
        creation_date=str(datetime.now()),
        processed_files=successful_file_count,
        max_z_filter=max_z if max_z is not None else 'None',
        z_range_actual=f"{z_min:.2f} to {z_max:.2f} m"
    )
    ds.to_netcdf(output_filename)
    print("\n✅ Time-averaged calculations are complete. Data saved to NetCDF.")

    # --- Plotting from the averaged data ---
    if profile_plot_filename:
        print(f"📊 Generating and saving vertical wind profile to '{profile_plot_filename}'...")
        try:
            if 'mean_u' in ds:
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(7, 7))
                ds['mean_u'].mean(dim='x').plot(y='z', ax=ax, color='k', linewidth=2.5)
                ax.set_xlabel('Mean Streamwise Velocity, U [m s$^{-1}$]', fontsize=12)
                ax.set_ylabel('Height, z [m]', fontsize=12)
                title = 'Time and Horizontally Averaged Wind Profile'
                if max_z is not None:
                    title += f' (z ≤ {max_z:.1f}m)'
                ax.set_title(title, fontsize=14, weight='bold')
                ax.set_ylim(bottom=0)
                plt.savefig(profile_plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Profile plot saved successfully to {profile_plot_filename}")
        except Exception as e:
            print(f"\n❌ ERROR: Could not generate profile plot. Reason: {e}")

    if second_moment_plot_dir:
        plot_second_moment_profiles(ds, second_moment_plot_dir)

    return pvtu_files

if __name__ == '__main__':
    # --- CONFIGURATION ---
    DATA_DIR = "/scratch/smarras/hw59/output/LESICP2_80x40x45_nop6_10kmX5kmX2dot8km_2/CompEuler/LESICP2/output"
    FILE_PATTERN = "iter_*.pvtu"
    BASE_GRID_RESOLUTIONX = 512
    BASE_GRID_RESOLUTIONZ = 300
    START_STEP = 80
    END_STEP = 140

    # --- NEW: VERTICAL DOMAIN FILTERING ---
    # Set MAX_Z to limit analysis to a specific height (e.g., exclude upper damping layer)
    # Examples:
    #   MAX_Z = 1000.0  # Analyze only atmospheric boundary layer (0-1000m)
    #   MAX_Z = 2000.0  # Include more of the domain
    #   MAX_Z = None    # Use entire vertical domain (default behavior)
    MAX_Z = None  # Change this to limit vertical extent


    # --- NEW: AUTOMATIC FILENAME PREFIX GENERATION ---
    # Create step range string for filenames based on START_STEP and END_STEP
    if START_STEP is not None and END_STEP is not None:
        step_range = f"{START_STEP}to{END_STEP}"
    elif START_STEP is not None:
        step_range = f"from{START_STEP}"
    elif END_STEP is not None:
        step_range = f"upto{END_STEP}"
    else:
        step_range = "allsteps"
    
    # If MAX_Z is set, add it to output filenames for clarity
    z_suffix = f"_maxz{int(MAX_Z)}" if MAX_Z is not None else ""
    
    # Combine step range and z-filter into filename prefix
    file_prefix = f"{step_range}{z_suffix}"
    
    OUTPUT_NC_AVERAGED_FILE = DATA_DIR + f"/{file_prefix}turbulence_statistics_averaged.nc"
    PROFILE_PLOT_FILE = DATA_DIR + f"/{file_prefix}vertical_wind_profile.png"
    SECOND_MOMENT_PLOT_DIR = DATA_DIR + f"/{file_prefix}second_moment_profiles"
    SPECTRA_PLOT_FILE = DATA_DIR + f"/{file_prefix}turbulent_spanwise_spectra_time_averaged.png"
    SNAPSHOT_3D_NC_FILE = DATA_DIR + f"/{file_prefix}snapshot_3d.nc"
    INSTANTANEOUS_SLICE_DIR = DATA_DIR + f"/{file_prefix}instantaneous_slice/"
    
    # --- SLICE & 3D SNAPSHOT CONFIGURATION ---
    WRITE_3D_SNAPSHOT = False
    X_SLICE_LOC = 2560.0
    Y_SLICE_LOC = 2560.0
    Z_SLICE_LOC = 100.0
    z1 = 20.0
    z0 = 0.1
    kappa = 0.4

    # --- VARIABLE DEFINITION ---
    VELOCITY_VARS = ['u', 'v', 'w']
    SCALAR_VARS = ['θ', 'p']

    # --- EXECUTION ---
    if not os.path.isdir(DATA_DIR):
        print(f"\n❌ Error: The specified data directory does not exist: {DATA_DIR}")
    else:
        print("\n" + "="*70)
        print("TURBULENCE STATISTICS POST-PROCESSING")
        print("="*70)
        if MAX_Z is not None:
            print(f"⚠️  VERTICAL FILTERING ACTIVE: Processing data with z ≤ {MAX_Z:.1f} m")
        else:
            print("ℹ️  No vertical filtering: Processing entire domain")
        print("="*70 + "\n")
        
        # 1. Calculate and save all time-averaged statistics
        processed_files = calculate_and_save_averaged_stats(
            data_directory=DATA_DIR,
            file_pattern=FILE_PATTERN,
            output_filename=OUTPUT_NC_AVERAGED_FILE,
            vel_variables=VELOCITY_VARS,
            scalar_variables=SCALAR_VARS,
            coord_precision=6,
            base_resolutionx=BASE_GRID_RESOLUTIONX,
            base_resolutionz=BASE_GRID_RESOLUTIONZ,
            start_step=START_STEP,
            end_step=END_STEP,
            profile_plot_filename=PROFILE_PLOT_FILE,
            second_moment_plot_dir=SECOND_MOMENT_PLOT_DIR,
            max_z=MAX_Z  # NEW PARAMETER
        )

        # 2. Process and save instantaneous 2D slices for each timestep
        if INSTANTANEOUS_SLICE_DIR:
            process_and_save_instantaneous_slices(
                pvtu_files=processed_files,
                output_dir=INSTANTANEOUS_SLICE_DIR,
                all_variables=VELOCITY_VARS + SCALAR_VARS,
                base_resolutionx=BASE_GRID_RESOLUTIONX,
                base_resolutionz=BASE_GRID_RESOLUTIONZ,
                x_slice_loc=X_SLICE_LOC,
                y_slice_loc=Y_SLICE_LOC,
                z_slice_loc=Z_SLICE_LOC,
                max_z=MAX_Z  # NEW PARAMETER
            )

        # 3. Calculate and plot time-averaged spectra
        if SPECTRA_PLOT_FILE:
            calculate_and_plot_spanwise_spectra(
                pvtu_files=processed_files,
                variables=['u', 'v', 'w', 'θ'],
                output_filename=SPECTRA_PLOT_FILE,
                max_z=MAX_Z  # NEW PARAMETER
            )

        # 4. Optionally, save a full 3D snapshot
        if WRITE_3D_SNAPSHOT and SNAPSHOT_3D_NC_FILE:
            interpolate_and_save_3d_snapshot(
                pvtu_files=processed_files,
                variables=VELOCITY_VARS + SCALAR_VARS,
                output_filename=SNAPSHOT_3D_NC_FILE,
                max_z=MAX_Z  # NEW PARAMETER
            )
        
        print("\n" + "="*70)
        print("✅ ALL PROCESSING COMPLETE")
        print("="*70)
