#!/usr/bin/env python3
"""
Convert PVTU (Parallel VTK Unstructured Grid) files to NetCDF format.

This script reads a .pvtu file and its corresponding parallel .vtu files
and exports the gridded data to NetCDF format.
"""

import numpy as np
import pyvista as pv
from netCDF4 import Dataset
import os
from pathlib import Path


def read_pvtu_file(pvtu_filepath, vtu_directory):
    """
    Read a PVTU file and its corresponding VTU files.
    
    Parameters:
    -----------
    pvtu_filepath : str
        Path to the .pvtu file
    vtu_directory : str
        Directory containing the parallel .vtu files
        
    Returns:
    --------
    mesh : pyvista.UnstructuredGrid
        Combined mesh from all parallel files
    """
    print(f"Reading PVTU file: {pvtu_filepath}")
    
    # Read the PVTU file - PyVista automatically handles parallel files
    mesh = pv.read(pvtu_filepath)
    
    print(f"Mesh loaded successfully:")
    print(f"  Number of points: {mesh.n_points}")
    print(f"  Number of cells: {mesh.n_cells}")
    print(f"  Available arrays: {mesh.array_names}")
    
    return mesh


def create_structured_grid_from_points(mesh):
    """
    Attempt to create a structured representation from unstructured mesh.
    If the mesh is already structured, extract dimensions.
    Otherwise, interpolate onto a regular grid.
    
    Parameters:
    -----------
    mesh : pyvista.UnstructuredGrid or pyvista.StructuredGrid
        Input mesh
        
    Returns:
    --------
    x_coords, y_coords, z_coords : np.ndarray
        1D coordinate arrays
    structured_data : dict
        Dictionary with variable names as keys and 3D arrays as values
    """
    # Get the bounds of the mesh
    bounds = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
    
    # Try to determine if data is already on a structured grid
    points = mesh.points
    
    # Extract unique coordinates
    unique_x = np.unique(np.round(points[:, 0], decimals=10))
    unique_y = np.unique(np.round(points[:, 1], decimals=10))
    unique_z = np.unique(np.round(points[:, 2], decimals=10))
    
    nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)
    
    print(f"\nDetected grid dimensions:")
    print(f"  X: {nx} points, range [{unique_x.min():.3f}, {unique_x.max():.3f}]")
    print(f"  Y: {ny} points, range [{unique_y.min():.3f}, {unique_y.max():.3f}]")
    print(f"  Z: {nz} points, range [{unique_z.min():.3f}, {unique_z.max():.3f}]")
    
    # Check if this is a structured grid
    if nx * ny * nz == mesh.n_points:
        print("Data appears to be on a structured grid!")
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(unique_x, unique_y, unique_z, indexing='ij')
        
        # Initialize structured data dictionary
        structured_data = {}
        
        # Extract point data
        for array_name in mesh.point_data.keys():
            data = mesh.point_data[array_name]
            
            # Create a mapping from coordinates to data values
            coord_to_value = {}
            for i, point in enumerate(points):
                coord = (round(point[0], 10), round(point[1], 10), round(point[2], 10))
                if data.ndim == 1:
                    coord_to_value[coord] = data[i]
                else:
                    coord_to_value[coord] = data[i, :]
            
            # Reshape data to structured grid
            if data.ndim == 1:
                structured_array = np.zeros((nx, ny, nz))
                for i, x in enumerate(unique_x):
                    for j, y in enumerate(unique_y):
                        for k, z in enumerate(unique_z):
                            coord = (round(x, 10), round(y, 10), round(z, 10))
                            if coord in coord_to_value:
                                structured_array[i, j, k] = coord_to_value[coord]
                            else:
                                structured_array[i, j, k] = np.nan
            else:
                # Multi-component data (vectors)
                n_components = data.shape[1]
                structured_array = np.zeros((nx, ny, nz, n_components))
                for i, x in enumerate(unique_x):
                    for j, y in enumerate(unique_y):
                        for k, z in enumerate(unique_z):
                            coord = (round(x, 10), round(y, 10), round(z, 10))
                            if coord in coord_to_value:
                                structured_array[i, j, k, :] = coord_to_value[coord]
                            else:
                                structured_array[i, j, k, :] = np.nan
            
            structured_data[array_name] = structured_array
            print(f"  Structured array '{array_name}' shape: {structured_array.shape}")
        
        return unique_x, unique_y, unique_z, structured_data
    
    else:
        print("Data is not on a structured grid. Interpolating to regular grid...")
        
        # Create a regular grid for interpolation
        # Use reasonable resolution based on the number of points
        nx = min(100, int(np.cbrt(mesh.n_points)))
        ny = min(100, int(np.cbrt(mesh.n_points)))
        nz = min(100, int(np.cbrt(mesh.n_points)))
        
        x_coords = np.linspace(bounds[0], bounds[1], nx)
        y_coords = np.linspace(bounds[2], bounds[3], ny)
        z_coords = np.linspace(bounds[4], bounds[5], nz)
        
        # Create structured grid for interpolation
        grid = pv.StructuredGrid()
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid.points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        grid.dimensions = [nx, ny, nz]
        
        # Interpolate data onto structured grid
        interpolated = grid.sample(mesh)
        
        structured_data = {}
        for array_name in interpolated.point_data.keys():
            data = interpolated.point_data[array_name]
            if data.ndim == 1:
                structured_array = data.reshape((nx, ny, nz))
            else:
                n_components = data.shape[1]
                structured_array = data.reshape((nx, ny, nz, n_components))
            
            structured_data[array_name] = structured_array
            print(f"  Interpolated array '{array_name}' shape: {structured_array.shape}")
        
        return x_coords, y_coords, z_coords, structured_data


def write_netcdf(output_filepath, x_coords, y_coords, z_coords, data_dict):
    """
    Write structured data to NetCDF file.
    
    Parameters:
    -----------
    output_filepath : str
        Output NetCDF file path
    x_coords, y_coords, z_coords : np.ndarray
        1D coordinate arrays
    data_dict : dict
        Dictionary with variable names as keys and arrays as values
    """
    print(f"\nWriting NetCDF file: {output_filepath}")
    
    # Create NetCDF file
    nc = Dataset(output_filepath, 'w', format='NETCDF4')
    
    # Create dimensions
    nc.createDimension('x', len(x_coords))
    nc.createDimension('y', len(y_coords))
    nc.createDimension('z', len(z_coords))
    
    # Create coordinate variables
    x_var = nc.createVariable('x', 'f8', ('x',))
    y_var = nc.createVariable('y', 'f8', ('y',))
    z_var = nc.createVariable('z', 'f8', ('z',))
    
    x_var[:] = x_coords
    y_var[:] = y_coords
    z_var[:] = z_coords
    
    # Add coordinate attributes
    x_var.units = 'meters'
    x_var.long_name = 'x-coordinate'
    y_var.units = 'meters'
    y_var.long_name = 'y-coordinate'
    z_var.units = 'meters'
    z_var.long_name = 'z-coordinate'
    
    # Add data variables
    for var_name, data in data_dict.items():
        print(f"  Adding variable: {var_name}")
        
        if data.ndim == 3:
            # Scalar field
            var = nc.createVariable(var_name, 'f8', ('x', 'y', 'z'), 
                                   fill_value=np.nan, zlib=True, complevel=4)
            var[:] = data
        elif data.ndim == 4:
            # Vector field
            n_components = data.shape[3]
            if n_components == 3:
                # Create dimension for vector components if not exists
                if 'component' not in nc.dimensions:
                    nc.createDimension('component', 3)
                
                var = nc.createVariable(var_name, 'f8', ('x', 'y', 'z', 'component'),
                                       fill_value=np.nan, zlib=True, complevel=4)
                var[:] = data
                var.long_name = f"{var_name} vector field"
            else:
                # Generic multi-component data
                comp_dim_name = f'component_{var_name}'
                if comp_dim_name not in nc.dimensions:
                    nc.createDimension(comp_dim_name, n_components)
                
                var = nc.createVariable(var_name, 'f8', ('x', 'y', 'z', comp_dim_name),
                                       fill_value=np.nan, zlib=True, complevel=4)
                var[:] = data
    
    # Add global attributes
    nc.title = "Converted from PVTU file"
    nc.source = "pvtu_to_netcdf.py"
    nc.history = f"Created on {np.datetime64('now')}"
    
    # Close the file
    nc.close()
    print(f"NetCDF file written successfully!")


def main():
    """Main function to convert PVTU to NetCDF."""
    
    # File paths
    pvtu_file = "./LESICP2_pvinterpolated.pvtu"
    vtu_directory = "./LESICP2_pvinterpolated"
    output_file = "./LESICP2_pvinterpolated.nc"
    
    # Check if files exist
    if not os.path.exists(pvtu_file):
        print(f"Error: PVTU file '{pvtu_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return
    
    if not os.path.exists(vtu_directory):
        print(f"Warning: VTU directory '{vtu_directory}' not found!")
        print(f"Attempting to read PVTU file directly...")
    
    # Read PVTU file
    mesh = read_pvtu_file(pvtu_file, vtu_directory)
    
    # Convert to structured grid
    x_coords, y_coords, z_coords, structured_data = create_structured_grid_from_points(mesh)
    
    # Write to NetCDF
    write_netcdf(output_file, x_coords, y_coords, z_coords, structured_data)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
