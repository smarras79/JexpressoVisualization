import xarray as xr
import glob
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import numpy as np

# --- Configuration (Adjust these settings) ---
input_file_pattern = '/Users/simone/Downloads/400to659instantaneous_slice/*.nc' 
#variable_name = 'w_slice_z100'
variable_name = 'theta_slice_z100'

output_gif_name = f'{variable_name}_timeseries.gif'
concat_dim_name = 'time'
gif_duration_s = 0.1
v_min_limit = None 
v_max_limit = None
# ----------------------------------------------------------------

def create_gif_from_netcdf(pattern, var_name, output_file, concat_dim, duration):
    """
    Combines NetCDF files, plots each time step of a selected variable, and 
    saves the sequence of plots as a GIF.
    """
    
    # 💡 THE FIX: Explicitly sort the filenames right after using glob.
    # This ensures that files named '001.nc', '002.nc', '010.nc' are read in order.
    file_list = sorted(glob.glob(pattern)) 
    
    if not file_list:
        print(f"Error: No files found matching pattern: {pattern}")
        return

    print(f"Found {len(file_list)} files to combine and process. First file: {os.path.basename(file_list[0])}, Last file: {os.path.basename(file_list[-1])}")
    
    filenames = [] # Initialize list for temporary image files
    
    try:
        # Step 1: Combine the NetCDF files into one Dataset
        print("Combining files using xarray...")
        combined_ds = xr.open_mfdataset(
            file_list,
            combine='nested',
            concat_dim=concat_dim,
            join='outer'  
        )
        
        # ALSO SORT BY TIME: This is a secondary safeguard in case the file naming 
        # is slightly out of sync with the time stamps themselves.
        combined_ds = combined_ds.sortby(concat_dim)

        if var_name not in combined_ds.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in the combined dataset.")

        data_array = combined_ds[var_name]
        
        # Get consistent plotting limits if not set
        global v_min_limit, v_max_limit
        if v_min_limit is None or v_max_limit is None:
            v_min_limit = data_array.min().compute().item()
            v_max_limit = data_array.max().compute().item()
            print(f"Auto-setting plot limits: min={v_min_limit:.2f}, max={v_max_limit:.2f}")

        # Step 2: Plotting and saving frames
        print(f"Generating {data_array[concat_dim].size} frames...")
        
        for i, time_step in enumerate(data_array[concat_dim]):
            frame_data = data_array.sel({concat_dim: time_step}).compute()
            
            fig, ax = plt.subplots()
            
            frame_data.squeeze().plot(
                ax=ax, 
                cmap='viridis', 
                vmin=v_min_limit, 
                vmax=v_max_limit, 
                cbar_kwargs={'label': f'{var_name} ({frame_data.units if hasattr(frame_data, "units") else "units"})'}
            )
            
            time_label = str(time_step.values)
            ax.set_title(f'{var_name} at Time: {time_label}')
            
            filename = f'temp_frame_{i:04d}.png'
            plt.savefig(filename, dpi=100)
            plt.close(fig) 
            filenames.append(filename)

        # Step 3: Create the GIF
        print("Creating GIF from frames...")
        with imageio.get_writer(output_file, duration=duration, loop=0) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        
        print(f"✅ GIF successfully created and saved as: {output_file}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        
    finally:
        # Clean up temporary files regardless of success or failure
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)
        print("Clean up complete.")

if __name__ == "__main__":
    create_gif_from_netcdf(
        input_file_pattern, 
        variable_name, 
        output_gif_name, 
        concat_dim_name, 
        gif_duration_s
    )
