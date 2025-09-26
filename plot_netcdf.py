#!/usr/bin/env python3
"""
Simple NetCDF plotter using only scipy and matplotlib
Avoids netCDF4 and xarray dependency issues
"""

import sys
import subprocess
import os
import glob
from pathlib import Path

def install_if_missing(package):
    """Install package if not available"""
    try:
        __import__(package)
        return True
    except ImportError:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except:
            return False

# Install required packages (these are more reliable than netCDF4)
required = ["numpy", "matplotlib", "scipy"]
for pkg in required:
    if not install_if_missing(pkg):
        print(f"Failed to install {pkg}")
        sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf_file
import argparse

def read_netcdf_with_scipy(filename):
    """Read NetCDF file using multiple methods for compatibility"""
    # First try scipy.io (NetCDF3 only)
    try:
        with netcdf_file(filename, 'r', mmap=False) as nc:
            variables = {}
            
            # Read coordinate variables
            if 'x' in nc.variables:
                variables['x'] = nc.variables['x'].data.copy()
            if 'y' in nc.variables:
                variables['y'] = nc.variables['y'].data.copy()
            
            # Find data variables
            coord_vars = ['x', 'y']
            data_vars = [v for v in nc.variables.keys() if v not in coord_vars]
            
            # Read data variables
            for var_name in data_vars:
                variables[var_name] = {
                    'data': nc.variables[var_name].data.copy(),
                    'name': var_name
                }
                
                # Try to get attributes
                var_obj = nc.variables[var_name]
                if hasattr(var_obj, 'long_name'):
                    variables[var_name]['long_name'] = var_obj.long_name
                if hasattr(var_obj, 'units'):
                    variables[var_name]['units'] = var_obj.units
            
            # Get global attributes
            global_attrs = {}
            if hasattr(nc, 'title'):
                global_attrs['title'] = nc.title
            if hasattr(nc, 'created'):
                global_attrs['created'] = nc.created
                
            return variables, global_attrs, True
            
    except Exception as e:
        print(f"scipy.io failed for {filename}: {e}")
        
        # Fallback: Try netCDF4 if available
        try:
            # Try to import and use netCDF4
            print("Trying netCDF4 library...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "netcdf4"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            from netCDF4 import Dataset
            
            with Dataset(filename, 'r') as nc:
                variables = {}
                
                # Read coordinates
                if 'x' in nc.variables:
                    variables['x'] = nc.variables['x'][:].copy()
                if 'y' in nc.variables:
                    variables['y'] = nc.variables['y'][:].copy()
                
                # Find data variables
                coord_vars = ['x', 'y']
                data_vars = [v for v in nc.variables.keys() if v not in coord_vars]
                
                # Read data variables
                for var_name in data_vars:
                    var_obj = nc.variables[var_name]
                    variables[var_name] = {
                        'data': var_obj[:].copy(),
                        'name': var_name
                    }
                    
                    # Get attributes
                    if hasattr(var_obj, 'long_name'):
                        variables[var_name]['long_name'] = var_obj.long_name
                    if hasattr(var_obj, 'units'):
                        variables[var_name]['units'] = var_obj.units
                
                # Global attributes
                global_attrs = {}
                if hasattr(nc, 'title'):
                    global_attrs['title'] = nc.title
                if hasattr(nc, 'getncattr'):
                    try:
                        global_attrs['title'] = nc.getncattr('title')
                    except:
                        pass
                
                return variables, global_attrs, True
                
        except Exception as e2:
            print(f"netCDF4 fallback also failed: {e2}")
            
            # Final fallback: Try to read as HDF5 (since NetCDF4 is HDF5-based)
            try:
                print("Trying h5py as final fallback...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                import h5py
                
                with h5py.File(filename, 'r') as hf:
                    variables = {}
                    
                    # Read coordinates
                    if 'x' in hf:
                        variables['x'] = hf['x'][:]
                    if 'y' in hf:
                        variables['y'] = hf['y'][:]
                    
                    # Find data variables
                    coord_vars = ['x', 'y']
                    data_vars = [v for v in hf.keys() if v not in coord_vars]
                    
                    # Read data variables
                    for var_name in data_vars:
                        variables[var_name] = {
                            'data': hf[var_name][:],
                            'name': var_name
                        }
                    
                    global_attrs = {}
                    if 'title' in hf.attrs:
                        global_attrs['title'] = hf.attrs['title'].decode() if isinstance(hf.attrs['title'], bytes) else hf.attrs['title']
                    
                    return variables, global_attrs, True
                    
            except Exception as e3:
                print(f"All reading methods failed for {filename}")
                print(f"scipy.io: {e}")
                print(f"netCDF4: {e2}")  
                print(f"h5py: {e3}")
                return None, None, False

def plot_netcdf_simple(filename, output_dir="plots", levels=30, colormap="viridis"):
    """Create simple contour plot from NetCDF file"""
    print(f"Processing: {filename}")
    
    variables, attrs, success = read_netcdf_with_scipy(filename)
    if not success:
        return False
    
    # Get coordinates
    if 'x' not in variables or 'y' not in variables:
        print(f"Missing x or y coordinates in {filename}")
        return False
    
    x = variables['x']
    y = variables['y']
    
    # Find data variables
    data_vars = [k for k in variables.keys() if k not in ['x', 'y']]
    
    if not data_vars:
        print(f"No data variables found in {filename}")
        return False
    
    # Plot each data variable
    for var_name in data_vars:
        try:
            var_info = variables[var_name]
            data = var_info['data']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create filled contour plot
            cs = ax.contourf(x, y, data.T, levels=levels, cmap=colormap)
            
            # Add colorbar
            cbar = plt.colorbar(cs, ax=ax)
            cbar.set_label(var_name, fontsize=12)
            
            # Add contour lines
            ax.contour(x, y, data.T, levels=levels//2, colors='black', alpha=0.3, linewidths=0.5)
            
            # Labels and title
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            
            title = attrs.get('title', f'{var_name} distribution')
            ax.set_title(title, fontsize=14)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Save plot
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(filename).stem
            output_file = os.path.join(output_dir, f"{base_name}_{var_name}.png")
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved: {output_file}")
            
        except Exception as e:
            print(f"✗ Failed to plot {var_name} from {filename}: {e}")
    
    return True

def analyze_netcdf_simple(filename):
    """Analyze NetCDF file and print statistics"""
    print(f"\n{'='*60}")
    print(f"Analysis: {filename}")
    print(f"{'='*60}")
    
    variables, attrs, success = read_netcdf_with_scipy(filename)
    if not success:
        return
    
    # Print global attributes
    if attrs:
        print("Global attributes:")
        for key, value in attrs.items():
            print(f"  {key}: {value}")
        print()
    
    # Print coordinate info
    if 'x' in variables and 'y' in variables:
        x = variables['x']
        y = variables['y']
        print(f"Coordinates:")
        print(f"  X range: [{x.min():.3f}, {x.max():.3f}] ({len(x)} points)")
        print(f"  Y range: [{y.min():.3f}, {y.max():.3f}] ({len(y)} points)")
        print()
    
    # Analyze data variables
    data_vars = [k for k in variables.keys() if k not in ['x', 'y']]
    
    for var_name in data_vars:
        var_info = variables[var_name]
        data = var_info['data']
        
        print(f"Variable: {var_name}")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Min: {data.min():.6e}")
        print(f"  Max: {data.max():.6e}")
        print(f"  Mean: {data.mean():.6e}")
        print(f"  Std: {data.std():.6e}")
        
        # Check for problematic values
        n_nan = np.isnan(data).sum()
        n_inf = np.isinf(data).sum()
        if n_nan > 0:
            print(f"  Warning: {n_nan} NaN values")
        if n_inf > 0:
            print(f"  Warning: {n_inf} infinite values")
        print()

def plot_combined_simple(netcdf_files, output_dir="plots"):
    """Create combined subplot for multiple NetCDF files"""
    if len(netcdf_files) == 0:
        return False
    
    # Determine layout
    n_files = len(netcdf_files)
    cols = min(3, n_files)
    rows = (n_files + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_files == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    for i, filename in enumerate(netcdf_files):
        if i >= len(axes):
            break
            
        variables, attrs, success = read_netcdf_with_scipy(filename)
        if not success:
            continue
        
        try:
            x = variables['x']
            y = variables['y']
            
            # Get first data variable
            data_vars = [k for k in variables.keys() if k not in ['x', 'y']]
            if not data_vars:
                continue
                
            var_name = data_vars[0]
            data = variables[var_name]['data']
            
            # Create contour plot
            cs = axes[i].contourf(x, y, data.T, levels=20, cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(cs, ax=axes[i], shrink=0.8)
            cbar.set_label(var_name, fontsize=10)
            
            # Labels and title
            title = Path(filename).stem
            axes[i].set_title(title, fontsize=10)
            axes[i].set_xlabel('X', fontsize=9)
            axes[i].set_ylabel('Y', fontsize=9)
            axes[i].set_aspect('equal')
            
        except Exception as e:
            print(f"✗ Failed to plot {filename}: {e}")
    
    # Hide unused subplots
    for i in range(n_files, len(axes)):
        axes[i].set_visible(False)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "combined_plots.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Combined plot saved: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Simple NetCDF plotter using scipy")
    parser.add_argument("files", nargs="+", help="NetCDF files or patterns")
    parser.add_argument("--output-dir", default="plots", help="Output directory")
    parser.add_argument("--levels", type=int, default=30, help="Contour levels")
    parser.add_argument("--colormap", default="viridis", help="Colormap")
    parser.add_argument("--analyze", action="store_true", help="Analysis only")
    parser.add_argument("--combined", action="store_true", help="Combined plot")
    
    args = parser.parse_args()
    
    # Expand file patterns
    all_files = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            all_files.extend(matches)
        elif os.path.exists(pattern):
            all_files.append(pattern)
    
    # Filter NetCDF files
    nc_files = [f for f in all_files if f.endswith('.nc')]
    
    if not nc_files:
        print("No .nc files found!")
        return
    
    print(f"Found {len(nc_files)} NetCDF files:")
    for f in sorted(nc_files):
        print(f"  {f}")
    
    if args.analyze:
        for filename in nc_files:
            analyze_netcdf_simple(filename)
    else:
        if args.combined and len(nc_files) > 1:
            plot_combined_simple(nc_files, args.output_dir)
        else:
            for filename in nc_files:
                plot_netcdf_simple(filename, args.output_dir, args.levels, args.colormap)

if __name__ == "__main__":
    main()
