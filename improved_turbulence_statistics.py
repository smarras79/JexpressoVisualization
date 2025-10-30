import os
import glob
import re
import numpy as np
import pyvista as pv
import xarray as xr
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import griddata, LinearNDInterpolator
from itertools import combinations_with_replacement, product as cartesian_product
import matplotlib.pyplot as plt
import shutil

# =============================================================================
# CARTESIAN GRID CONFIGURATION (USER-DEFINED)
# =============================================================================
class CartesianGridConfig:
    """Configuration for the Cartesian interpolation grid"""
    def __init__(self, 
                 x_min=None, x_max=None, nx=128,
                 y_min=None, y_max=None, ny=128,
                 z_min=None, z_max=None, nz=64):
        """
        Define Cartesian grid parameters.
        
        Parameters:
        -----------
        x_min, x_max : float or None
            Domain extent in x-direction. If None, use mesh bounds.
        nx : int
            Number of grid points in x-direction
        y_min, y_max : float or None
            Domain extent in y-direction. If None, use mesh bounds.
        ny : int
            Number of grid points in y-direction
        z_min, z_max : float or None
            Domain extent in z-direction. If None, use mesh bounds.
        nz : int
            Number of grid points in z-direction
        """
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        
        self.y_min = y_min
        self.y_max = y_max
        self.ny = ny
        
        self.z_min = z_min
        self.z_max = z_max
        self.nz = nz
        
    def create_grid(self, mesh_bounds=None):
        """
        Create the Cartesian grid coordinates.
        
        Parameters:
        -----------
        mesh_bounds : tuple or None
            Mesh bounds as (xmin, xmax, ymin, ymax, zmin, zmax)
            Used if domain limits are not specified
            
        Returns:
        --------
        X, Y, Z : ndarray
            3D coordinate arrays (shape: nz x ny x nx)
        x, y, z : ndarray
            1D coordinate vectors
        """
        # Determine grid bounds
        if mesh_bounds is not None:
            x_min = self.x_min if self.x_min is not None else mesh_bounds[0]
            x_max = self.x_max if self.x_max is not None else mesh_bounds[1]
            y_min = self.y_min if self.y_min is not None else mesh_bounds[2]
            y_max = self.y_max if self.y_max is not None else mesh_bounds[3]
            z_min = self.z_min if self.z_min is not None else mesh_bounds[4]
            z_max = self.z_max if self.z_max is not None else mesh_bounds[5]
        else:
            if any(v is None for v in [self.x_min, self.x_max, self.y_min, 
                                       self.y_max, self.z_min, self.z_max]):
                raise ValueError("Grid bounds must be specified or mesh_bounds provided")
            x_min, x_max = self.x_min, self.x_max
            y_min, y_max = self.y_min, self.y_max
            z_min, z_max = self.z_min, self.z_max
        
        # Create 1D coordinate vectors
        x = np.linspace(x_min, x_max, self.nx)
        y = np.linspace(y_min, y_max, self.ny)
        z = np.linspace(z_min, z_max, self.nz)
        
        # Create 3D meshgrid (using 'ij' indexing for consistency with xarray)
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        
        return X, Y, Z, x, y, z

# =============================================================================
# INTERPOLATION FUNCTIONS
# =============================================================================

def interpolate_pvtu_to_cartesian(pvtu_file, variables, grid_config, method='linear'):
    """
    Interpolate PVTU unstructured mesh data onto a Cartesian grid.
    
    Parameters:
    -----------
    pvtu_file : str
        Path to PVTU file
    variables : list of str
        List of variable names to interpolate
    grid_config : CartesianGridConfig
        Configuration for the Cartesian grid
    method : str
        Interpolation method ('linear' or 'nearest')
        
    Returns:
    --------
    dict : Dictionary with interpolated data
        Keys: variable names, values: 3D numpy arrays (nz x ny x nx)
    coords : dict
        Dictionary with coordinate arrays: 'X', 'Y', 'Z', 'x', 'y', 'z'
    """
    # Read mesh
    try:
        mesh_container = pv.read(pvtu_file)
        mesh = mesh_container.combine(merge_points=False) if isinstance(mesh_container, pv.MultiBlock) else mesh_container
    except Exception as e:
        raise RuntimeError(f"Could not read PVTU file: {e}")
    
    # Get mesh points
    points = mesh.points  # Shape: (n_points, 3)
    
    # Create Cartesian grid
    X, Y, Z, x, y, z = grid_config.create_grid(mesh_bounds=mesh.bounds)
    
    # Flatten grid for interpolation
    cart_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    print(f"   Interpolating from {len(points)} unstructured points to {len(cart_points)} Cartesian points...")
    print(f"   Grid resolution: {grid_config.nx} x {grid_config.ny} x {grid_config.nz}")
    
    # Interpolate each variable
    interpolated_data = {}
    
    for var in variables:
        if var not in mesh.point_data:
            print(f"   ⚠️ Warning: Variable '{var}' not found in mesh. Skipping.")
            continue
        
        var_data = mesh.point_data[var]
        
        # Use scipy's LinearNDInterpolator for better performance with 3D data
        if method == 'linear':
            interpolator = LinearNDInterpolator(points, var_data, fill_value=np.nan)
            interpolated = interpolator(cart_points)
        else:  # nearest neighbor
            interpolated = griddata(points, var_data, cart_points, method='nearest')
        
        # Reshape to 3D grid (nz x ny x nx)
        interpolated_data[var] = interpolated.reshape(grid_config.nz, grid_config.ny, grid_config.nx)
    
    coords = {
        'X': X, 'Y': Y, 'Z': Z,
        'x': x, 'y': y, 'z': z
    }
    
    return interpolated_data, coords


def interpolate_multiple_timesteps(pvtu_files, variables, grid_config, method='linear'):
    """
    Interpolate multiple PVTU files onto the same Cartesian grid.
    
    Parameters:
    -----------
    pvtu_files : list of str
        List of PVTU file paths
    variables : list of str
        Variables to interpolate
    grid_config : CartesianGridConfig
        Grid configuration
    method : str
        Interpolation method
        
    Returns:
    --------
    data_4d : dict
        Dictionary with keys as variable names and values as 4D arrays (nt x nz x ny x nx)
    coords : dict
        Coordinate arrays
    """
    # Get grid from first file
    first_data, coords = interpolate_pvtu_to_cartesian(pvtu_files[0], variables, grid_config, method)
    
    # Initialize 4D arrays
    nt = len(pvtu_files)
    nz, ny, nx = grid_config.nz, grid_config.ny, grid_config.nx
    
    data_4d = {var: np.zeros((nt, nz, ny, nx)) for var in first_data.keys()}
    
    # Store first timestep
    for var in first_data.keys():
        data_4d[var][0] = first_data[var]
    
    # Interpolate remaining timesteps
    for t, fpath in enumerate(tqdm(pvtu_files[1:], desc="Interpolating timesteps", initial=1, total=nt)):
        try:
            interp_data, _ = interpolate_pvtu_to_cartesian(fpath, variables, grid_config, method)
            for var in interp_data.keys():
                data_4d[var][t+1] = interp_data[var]
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to interpolate {os.path.basename(fpath)}: {e}")
            # Keep zeros for this timestep
    
    return data_4d, coords

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_var_name(name):
    """Replace Unicode characters that cause NetCDF encoding issues"""
    return name.replace('θ', 'theta')

def get_primed_name(key, all_vars):
    """Get the primed variable name for fluctuations"""
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

# =============================================================================
# STATISTICS CALCULATION ON CARTESIAN GRID
# =============================================================================

def calculate_turbulence_statistics_cartesian(data_4d, vel_variables, scalar_variables=None):
    """
    Calculate turbulence statistics on Cartesian grid data.
    
    Parameters:
    -----------
    data_4d : dict
        Dictionary with 4D arrays (nt x nz x ny x nx) for each variable
    vel_variables : list of str
        Velocity variable names (e.g., ['u', 'v', 'w'])
    scalar_variables : list of str or None
        Scalar variable names (e.g., ['θ', 'p'])
        
    Returns:
    --------
    stats : dict
        Dictionary containing:
        - Time-averaged means (shape: nz x ny x nx)
        - Second moments (Reynolds stresses, fluxes)
        - All averaged over time dimension
    """
    if scalar_variables is None:
        scalar_variables = []
    
    all_variables = vel_variables + scalar_variables
    
    print("\n📊 Calculating turbulence statistics on Cartesian grid...")
    
    stats = {}
    
    # Calculate time-averaged means
    print("   Computing time-averaged means...")
    means = {}
    for var in all_variables:
        if var in data_4d:
            # Average over time (axis 0)
            means[var] = np.mean(data_4d[var], axis=0)  # Shape: (nz, ny, nx)
            stats[f'mean_{sanitize_var_name(var)}'] = means[var]
    
    # Calculate fluctuations and second moments
    print("   Computing fluctuations and second moments...")
    
    # Get all unique second-moment combinations
    all_vars_for_moments = vel_variables + scalar_variables
    moment_pairs = list(combinations_with_replacement(all_vars_for_moments, 2))
    
    for var1, var2 in tqdm(moment_pairs, desc="   Processing second moments"):
        if var1 not in data_4d or var2 not in data_4d:
            continue
        
        # Calculate fluctuations for each timestep
        # data_4d[var] has shape (nt, nz, ny, nx)
        # means[var] has shape (nz, ny, nx)
        # Need to broadcast means to (nt, nz, ny, nx)
        fluct1 = data_4d[var1] - means[var1][np.newaxis, :, :, :]
        fluct2 = data_4d[var2] - means[var2][np.newaxis, :, :, :]
        
        # Calculate second moment (time average of fluctuation product)
        second_moment = np.mean(fluct1 * fluct2, axis=0)  # Shape: (nz, ny, nx)
        
        # Create variable name
        moment_name = get_primed_name(f"{var1}'{var2}'", all_vars_for_moments)
        stats[moment_name] = second_moment
    
    print(f"✅ Calculated {len(stats)} statistical quantities")
    
    return stats


def save_cartesian_statistics_to_netcdf(stats, coords, output_filename, vel_variables, scalar_variables):
    """
    Save Cartesian grid statistics to NetCDF file.
    
    Parameters:
    -----------
    stats : dict
        Dictionary with statistical variables (each has shape nz x ny x nx)
    coords : dict
        Coordinate arrays ('x', 'y', 'z')
    output_filename : str
        Output NetCDF filename
    vel_variables : list of str
        Velocity variable names
    scalar_variables : list of str
        Scalar variable names
    """
    print(f"\n💾 Saving Cartesian statistics to NetCDF: {output_filename}")
    
    x = coords['x']
    y = coords['y']
    z = coords['z']
    
    # Create xarray Dataset
    data_vars = {}
    for var_name, var_data in stats.items():
        data_vars[var_name] = (('z', 'y', 'x'), var_data)
    
    ds = xr.Dataset(
        data_vars,
        coords={
            'x': ('x', x),
            'y': ('y', y),
            'z': ('z', z)
        }
    )
    
    # Add attributes
    ds.x.attrs.update({
        'units': 'm',
        'long_name': 'Streamwise coordinate',
        'axis': 'X'
    })
    
    ds.y.attrs.update({
        'units': 'm',
        'long_name': 'Spanwise coordinate',
        'axis': 'Y'
    })
    
    ds.z.attrs.update({
        'units': 'm',
        'long_name': 'Vertical coordinate',
        'axis': 'Z'
    })
    
    # Add variable attributes
    for var in vel_variables + scalar_variables:
        sanitized = sanitize_var_name(var)
        if f'mean_{sanitized}' in ds:
            ds[f'mean_{sanitized}'].attrs.update({
                'long_name': f'Time-averaged {var}',
                'description': f'Reynolds-averaged {var} on Cartesian grid'
            })
    
    # Global attributes
    ds.attrs.update({
        'title': 'Turbulence Statistics on Cartesian Grid',
        'description': 'Time-averaged statistics interpolated onto regular Cartesian grid',
        'creation_date': datetime.now().isoformat(),
        'grid_type': 'Cartesian',
        'grid_resolution': f'{len(x)} x {len(y)} x {len(z)}',
        'interpolation_method': 'linear'
    })
    
    # Save to NetCDF
    ds.to_netcdf(output_filename, engine='netcdf4')
    print(f"✅ Statistics saved successfully")


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_turbulence_statistics_on_cartesian_grid(
    data_directory,
    file_pattern,
    output_filename,
    grid_config,
    vel_variables=['u', 'v', 'w'],
    scalar_variables=['θ', 'p'],
    start_step=None,
    end_step=None,
    interpolation_method='linear'
):
    """
    Main function to process turbulence statistics on a Cartesian grid.
    
    Parameters:
    -----------
    data_directory : str
        Directory containing PVTU files
    file_pattern : str
        File pattern (e.g., 'iter_*.pvtu')
    output_filename : str
        Output NetCDF filename for statistics
    grid_config : CartesianGridConfig
        Configuration for Cartesian grid
    vel_variables : list of str
        Velocity variable names
    scalar_variables : list of str
        Scalar variable names
    start_step : int or None
        Starting iteration number
    end_step : int or None
        Ending iteration number
    interpolation_method : str
        Interpolation method ('linear' or 'nearest')
        
    Returns:
    --------
    processed_files : list
        List of processed PVTU files
    stats : dict
        Computed statistics
    coords : dict
        Coordinate arrays
    """
    print("\n" + "="*70)
    print("TURBULENCE STATISTICS ON CARTESIAN GRID")
    print("="*70)
    
    # Find PVTU files
    search_pattern = os.path.join(data_directory, file_pattern)
    all_files = sorted(glob.glob(search_pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern: {search_pattern}")
    
    print(f"📁 Found {len(all_files)} PVTU files")
    
    # Filter by iteration range
    def extract_iteration(filename):
        match = re.search(r'iter_(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else None
    
    filtered_files = []
    for f in all_files:
        iter_num = extract_iteration(f)
        if iter_num is None:
            continue
        if start_step is not None and iter_num < start_step:
            continue
        if end_step is not None and iter_num > end_step:
            continue
        filtered_files.append(f)
    
    if not filtered_files:
        raise ValueError("No files found in specified iteration range")
    
    print(f"📊 Processing {len(filtered_files)} files in iteration range")
    if start_step is not None:
        print(f"   Start iteration: {start_step}")
    if end_step is not None:
        print(f"   End iteration: {end_step}")
    
    # Interpolate all timesteps to Cartesian grid
    print(f"\n🔄 Interpolating to Cartesian grid...")
    all_variables = vel_variables + scalar_variables
    
    data_4d, coords = interpolate_multiple_timesteps(
        filtered_files, 
        all_variables, 
        grid_config,
        method=interpolation_method
    )
    
    # Calculate turbulence statistics
    stats = calculate_turbulence_statistics_cartesian(
        data_4d,
        vel_variables,
        scalar_variables
    )
    
    # Save to NetCDF
    save_cartesian_statistics_to_netcdf(
        stats,
        coords,
        output_filename,
        vel_variables,
        scalar_variables
    )
    
    return filtered_files, stats, coords


# =============================================================================
# VISUALIZATION FUNCTIONS (Modified for Cartesian grid)
# =============================================================================

def plot_vertical_profiles_cartesian(stats, coords, output_filename, variables=None):
    """
    Plot vertical profiles of statistics from Cartesian grid data.
    
    Parameters:
    -----------
    stats : dict
        Dictionary with statistical variables
    coords : dict
        Coordinate arrays
    output_filename : str
        Output plot filename
    variables : list of str or None
        Variables to plot. If None, plot velocity components
    """
    print(f"\n📊 Plotting vertical profiles...")
    
    z = coords['z']
    
    if variables is None:
        # Default: plot mean velocity components
        variables = ['mean_u', 'mean_v', 'mean_w']
    
    fig, axes = plt.subplots(1, len(variables), figsize=(5*len(variables), 6))
    if len(variables) == 1:
        axes = [axes]
    
    for ax, var in zip(axes, variables):
        if var not in stats:
            print(f"   ⚠️ Warning: {var} not found in statistics")
            continue
        
        # Average over x and y to get vertical profile
        profile = np.mean(stats[var], axis=(1, 2))  # Average over y and x
        
        ax.plot(profile, z, 'k-', linewidth=2)
        ax.set_xlabel(f'{var}', fontsize=12)
        ax.set_ylabel('Height z [m]', fontsize=12)
        ax.set_title(f'Vertical Profile of {var}', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Vertical profiles saved to {output_filename}")


def plot_horizontal_slice_cartesian(stats, coords, var_name, z_index, output_filename):
    """
    Plot horizontal slice of a variable at specified z-level.
    
    Parameters:
    -----------
    stats : dict
        Dictionary with statistical variables
    coords : dict
        Coordinate arrays
    var_name : str
        Variable name to plot
    z_index : int
        Index along z-axis for the slice
    output_filename : str
        Output filename
    """
    if var_name not in stats:
        print(f"⚠️ Variable {var_name} not found")
        return
    
    x = coords['x']
    y = coords['y']
    z = coords['z']
    
    data_slice = stats[var_name][z_index, :, :]
    
    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, data_slice, levels=50, cmap='RdBu_r')
    plt.colorbar(label=var_name)
    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)
    plt.title(f'{var_name} at z = {z[z_index]:.1f} m', fontsize=14, weight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Horizontal slice saved to {output_filename}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    
    # =============================================================================
    # USER CONFIGURATION
    # =============================================================================
    
    # Input/Output Configuration
    #DATA_DIR = "/scratch/smarras/smarras/output/LESICP4_scaling-8nodes-64x32x36_10kmX10kmX3km/CompEuler/LESICP4/output"
    DATA_DIR = "./tmp/"
    FILE_PATTERN = "iter_*.pvtu"
    OUTPUT_NC_FILE = DATA_DIR + "/turbulence_statistics_cartesian.nc"
    PROFILE_PLOT_FILE = DATA_DIR + "/vertical_profiles_cartesian.png"
    SLICE_PLOT_FILE = DATA_DIR + "/horizontal_slice_cartesian.png"
    
    # Iteration range
    START_STEP = 600 #1800
    END_STEP = 2161
    
    # Variables to process
    VELOCITY_VARS = ['u', 'v', 'w']
    SCALAR_VARS = ['θ', 'p']
    
    # =============================================================================
    # CARTESIAN GRID CONFIGURATION (USER-DEFINED)
    # =============================================================================
    
    # Option 1: Let the code automatically determine bounds from the mesh
    # and only specify resolution
    #grid_config = CartesianGridConfig(
    #    nx=256,   # Number of points in x-direction
    #    ny=128,   # Number of points in y-direction
    #    nz=36     # Number of points in z-direction
    #)
    
    # Option 2: Explicitly define the domain extent and resolution
    # grid_config = CartesianGridConfig(
    #     x_min=0.0,    x_max=5120.0,  nx=256,
    #     y_min=0.0,    y_max=5120.0,  ny=256,
    #     z_min=0.0,    z_max=1000.0,  nz=100
    # )
    
    # Option 3: Define only certain dimensions explicitly
    grid_config = CartesianGridConfig(
        z_min=0.0,    z_max=1500.0,  nz=72,  # Limit vertical extent
        nx=256,       ny=128                 # Use auto bounds for x, y
    )
    
    # Interpolation method: 'linear' or 'nearest'
    INTERPOLATION_METHOD = 'linear'
    
    # =============================================================================
    # EXECUTION
    # =============================================================================
    
    if not os.path.isdir(DATA_DIR):
        print(f"\n❌ Error: Data directory does not exist: {DATA_DIR}")
    else:
        # Process turbulence statistics on Cartesian grid
        processed_files, stats, coords = process_turbulence_statistics_on_cartesian_grid(
            data_directory=DATA_DIR,
            file_pattern=FILE_PATTERN,
            output_filename=OUTPUT_NC_FILE,
            grid_config=grid_config,
            vel_variables=VELOCITY_VARS,
            scalar_variables=SCALAR_VARS,
            start_step=START_STEP,
            end_step=END_STEP,
            interpolation_method=INTERPOLATION_METHOD
        )
        
        # Plot vertical profiles
        plot_vertical_profiles_cartesian(
            stats=stats,
            coords=coords,
            output_filename=PROFILE_PLOT_FILE,
            variables=['mean_u', 'mean_v', 'mean_w']
        )
        
        # Plot horizontal slice (at mid-height)
        z_mid_index = len(coords['z']) // 2
        plot_horizontal_slice_cartesian(
            stats=stats,
            coords=coords,
            var_name='mean_u',
            z_index=z_mid_index,
            output_filename=SLICE_PLOT_FILE
        )
        
        print("\n" + "="*70)
        print("✅ ALL PROCESSING COMPLETE")
        print("="*70)
        print(f"\nOutput files:")
        print(f"  • Statistics: {OUTPUT_NC_FILE}")
        print(f"  • Profiles:   {PROFILE_PLOT_FILE}")
        print(f"  • Slice:      {SLICE_PLOT_FILE}")
        
