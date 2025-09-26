#!/usr/bin/env julia
"""
    BATCH PROCESSING: PVTU Analysis with NetCDF output
    Extracts slices and turbulence statistics, exports to NetCDF format for Python
    Uses only core Julia packages - no graphics dependencies
    """

using Statistics
using NearestNeighbors
using ArgParse
using Dates
using NCDatasets

struct CompletePVTUData
    points::Matrix{Float64}
    u::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
    n_points::Int
    bounds::NamedTuple
    n_pieces::Int
end

function parse_commandline()
    """Parse command line arguments for batch processing"""
    s = ArgParseSettings(description = "Batch PVTU Analysis with Data Export")
    
    @add_arg_table! s begin
        "--range"
        help = "File range to process: start end step"
        nargs = 3
        arg_type = Int
        required = false
        "--process-id"
        help = "Process ID for output organization"
        arg_type = Int
        default = 1
        "--resolution"
        help = "Grid resolution for interpolation"
        arg_type = Int
        default = 200
        "--output-dir"
        help = "Output directory"
        arg_type = String
        default = "batch_output"
        "--file-prefix"
        help = "PVTU file prefix (e.g., 'iter_' for iter_XXX.pvtu)"
        arg_type = String
        default = "iter_"
        "--slice-coord"
        help = "Z-coordinate for velocity slice"
        arg_type = Float64
        default = 0.1
        "--dry-run"
        help = "Show files that would be processed without processing"
        action = :store_true
        "--variables"
        help = "Variables to process (comma-separated: w,u,v)"
        arg_type = String
        default = "w"
        "--reynolds-stress"
        help = "Reynolds stress components to calculate (comma-separated: uv,uw,vw)"
        arg_type = String
        default = "uv"
        "--export-format"
        help = "Export format: netcdf, csv, or both"
        arg_type = String
        default = "netcdf"
    end
    
    return parse_args(s)
end

function find_pvtu_files(file_prefix::String, start_idx::Int, end_idx::Int, step::Int)
    """Find available PVTU files in the specified range"""
    available_files = String[]
    missing_files = Int[]
    
    for i in start_idx:step:end_idx
        filename = "$(file_prefix)$(i).pvtu"
        if isfile(filename)
            push!(available_files, filename)
        else
            push!(missing_files, i)
        end
    end
    
    return available_files, missing_files
end

function read_single_piece_safe(filename::String)
    """Read a single VTU piece with robust, dynamic header parsing."""
    try
        file_data = read(filename)
        header_str = String(file_data[1:min(end, 4096)])

        piece_match = match(r"<Piece NumberOfPoints=\"(\d+)\"", header_str)
        if piece_match === nothing
            error("Could not find '<Piece NumberOfPoints=...>' tag")
        end
        num_points = parse(Int, piece_match.captures[1])

        appended_match = findfirst("<AppendedData encoding=\"raw\">", header_str)
        if appended_match === nothing
            error("Could not find AppendedData section")
        end
        after_tag = header_str[last(appended_match):end]
        underscore_match = findfirst("_", after_tag)
        if underscore_match === nothing
            error("Could not find binary data start marker '_'")
        end
        binary_start = last(appended_match) + first(underscore_match)

        function get_offset(name_pattern)
            regex_offset = Regex("<DataArray[^>]*Name=\"$(name_pattern)\"[^>]*offset=\"(\\d+)\"")
            match_offset = match(regex_offset, header_str)
            if match_offset === nothing
                error("Could not find DataArray tag for '$name_pattern'")
            end
            return parse(Int, match_offset.captures[1])
        end

        function read_block(offset::Int)
            header_pos = binary_start + offset
            block_size_header = reinterpret(UInt64, file_data[header_pos:header_pos+7])[1]
            data_start = header_pos + 8
            data_end = data_start + Int(block_size_header) - 1
            return reinterpret(Float64, file_data[data_start:data_end])
        end

        points_raw = read_block(get_offset("Points"))
        points_matrix = reshape(points_raw, 3, num_points)

        u_data = read_block(get_offset("u"))
        v_data = read_block(get_offset("v"))
        w_data = read_block(get_offset("w"))

        return points_matrix, u_data, v_data, w_data, true
    catch e
        println("ERROR reading piece $filename: $e")
        return nothing, nothing, nothing, nothing, false
    end
end

function read_all_parallel_pieces(pvtu_file::String)
    """Read ALL parallel VTU pieces and combine them"""
    println("=== READING: $pvtu_file ===")
    
    pvtu_content = read(pvtu_file, String)
    piece_matches = collect(eachmatch(r"<Piece\s+Source=\"([^\"]+)\"", pvtu_content))
    piece_files = [m.captures[1] for m in piece_matches]
    
    println("Found $(length(piece_files)) parallel pieces to read")
    
    pvtu_dir = dirname(pvtu_file)
    if isempty(pvtu_dir); pvtu_dir = "."; end
    
    all_points = Matrix{Float64}[]
    all_u, all_v, all_w = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]
    successful, total_points = 0, 0
    
    for (i, piece_file) in enumerate(piece_files)
        piece_path = joinpath(pvtu_dir, piece_file)
        points, u, v, w, success = read_single_piece_safe(piece_path)
        
        if success
            push!.((all_points, all_u, all_v, all_w), (points, u, v, w))
            successful += 1
            total_points += size(points, 2)
            if i % 50 == 0 || i <= 5
                println("  Piece $i/$(length(piece_files)): $(size(points, 2)) points read successfully")
            end
        else
            println("  Failed to read piece: $piece_file")
        end
    end
    
    println("Successfully read $successful/$(length(piece_files)) pieces")
    println("Total points across all pieces: $total_points")
    
    if isempty(all_points); error("No data pieces were read successfully."); end
    
    combined_points = hcat(all_points...)
    combined_u, combined_v, combined_w = vcat(all_u...), vcat(all_v...), vcat(all_w...)
    
    bounds = (
        xmin=minimum(view(combined_points, 1, :)), xmax=maximum(view(combined_points, 1, :)),
        ymin=minimum(view(combined_points, 2, :)), ymax=maximum(view(combined_points, 2, :)),
        zmin=minimum(view(combined_points, 3, :)), zmax=maximum(view(combined_points, 3, :))
    )
    
    return CompletePVTUData(combined_points, combined_u, combined_v, combined_w, 
                            total_points, bounds, successful)
end

function apply_smoothing_filter(grid::Matrix{Float64}, kernel_size::Int=3)
    """Apply a simple box filter to smooth the grid data"""
    rows, cols = size(grid)
    smoothed = copy(grid)
    half_kernel = kernel_size ÷ 2
    
    for i in (1+half_kernel):(rows-half_kernel)
        for j in (1+half_kernel):(cols-half_kernel)
            window_sum = 0.0
            count = 0
            for di in -half_kernel:half_kernel, dj in -half_kernel:half_kernel
                window_sum += grid[i + di, j + dj]
                count += 1
            end
            smoothed[i, j] = window_sum / count
        end
    end
    return smoothed
end

function create_filled_contour_slice(data::CompletePVTUData, var::String, axis::String, coord::Float64, resolution::Int)
    """Create filled contour slice with specified resolution - export data only"""
    field_data = getproperty(data, Symbol(var))
    axis_idx = Dict("X" => 1, "Y" => 2, "Z" => 3)[uppercase(axis)]
    
    tolerance = 0.05 * (getproperty(data.bounds, Symbol(lowercase(axis) * "max")) - getproperty(data.bounds, Symbol(lowercase(axis) * "min")))
    slice_mask = abs.(view(data.points, axis_idx, :) .- coord) .<= tolerance
    
    println("Found $(sum(slice_mask)) points for $var slice with tolerance $tolerance")
    if sum(slice_mask) < 3; error("Not enough points in slice."); end

    slice_points = data.points[:, slice_mask]
    slice_field = field_data[slice_mask]
    
    other_axes = filter(x -> x != axis_idx, [1, 2, 3])
    c1_data = view(slice_points, other_axes[1], :)
    c2_data = view(slice_points, other_axes[2], :)
    
    c1_range = range(extrema(c1_data)..., length=resolution)
    c2_range = range(extrema(c2_data)..., length=resolution)
    
    c1_array = collect(c1_data)
    c2_array = collect(c2_data)
    
    grid_data = zeros(length(c1_range), length(c2_range))
    
    total_area = (maximum(c1_array) - minimum(c1_array)) * (maximum(c2_array) - minimum(c2_array))
    avg_point_spacing = sqrt(total_area / length(c1_array))
    search_radius = 2.0 * avg_point_spacing
    
    tree_data = [c1_array c2_array]'
    kdtree = KDTree(tree_data)
    
    total_points = length(c1_range) * length(c2_range)
    progress_interval = max(1, total_points ÷ 20)
    processed = 0
    
    for (i, c1) in enumerate(c1_range), (j, c2) in enumerate(c2_range)
        neighbors = inrange(kdtree, [c1, c2], search_radius)
        
        if !isempty(neighbors)
            weights = Float64[]
            values  = Float64[]
            
            for neighbor_idx in neighbors
                dist = sqrt((c1 - c1_array[neighbor_idx])^2 + (c2 - c2_array[neighbor_idx])^2)
                weight = 1.0 / (dist + 1e-12)
                push!(weights, weight)
                push!(values, slice_field[neighbor_idx])
            end
            
            grid_data[i, j] = sum(weights .* values) / sum(weights)
        else
            nearest_idx, _ = nn(kdtree, [c1, c2])
            grid_data[i, j] = slice_field[nearest_idx]
        end
        
        processed += 1
        if processed % progress_interval == 0
            progress = round(100 * processed / total_points, digits=1)
            println("  Interpolation progress: $(progress)%")
        end
    end
    
    grid_data = apply_smoothing_filter(grid_data)
    
    axis_names = ["X", "Y", "Z"]
    return c1_range, c2_range, grid_data, axis_names[other_axes[1]], axis_names[other_axes[2]]
end

function calculate_reynolds_stress_contours(data::CompletePVTUData, comp::String, avg_axis::String, resolution::Int)
    """Calculate Reynolds stress contours with specified resolution - export data only"""
    println("\n=== REYNOLDS STRESS: <$comp'> averaged over $avg_axis ===")
    println("Using resolution: $(resolution)×$(resolution)")
    
    u_p = data.u .- mean(data.u)
    v_p = data.v .- mean(data.v)
    w_p = data.w .- mean(data.w)
    
    if comp == "uv"
        rs_data = u_p .* v_p
    elseif comp == "uw"
        rs_data = u_p .* w_p
    elseif comp == "vw"
        rs_data = v_p .* w_p
    elseif comp == "uu"
        rs_data = u_p .* u_p
    elseif comp == "vv"
        rs_data = v_p .* v_p
    elseif comp == "ww"
        rs_data = w_p .* w_p
    else
        error("Unknown Reynolds stress component: $comp")
    end
    
    axis_idx = Dict("X" => 1, "Y" => 2, "Z" => 3)[uppercase(avg_axis)]
    other_axes = filter(x -> x != axis_idx, [1, 2, 3])
    
    if avg_axis == "Y"
        c1_data = view(data.points, 1, :)
        c2_data = view(data.points, 3, :)
        axis1_name, axis2_name = "X", "Z"
    elseif avg_axis == "X"
        c1_data = view(data.points, 2, :)
        c2_data = view(data.points, 3, :)
        axis1_name, axis2_name = "Y", "Z"
    elseif avg_axis == "Z"
        c1_data = view(data.points, 1, :)
        c2_data = view(data.points, 2, :)
        axis1_name, axis2_name = "X", "Y"
    else
        error("Unknown averaging axis: $avg_axis")
    end
    
    c1_range = range(extrema(c1_data)..., length=resolution)
    c2_range = range(extrema(c2_data)..., length=resolution)
    
    println("Projecting onto $(axis1_name)-$(axis2_name) plane (averaging over $avg_axis)")
    
    c1_array = collect(c1_data)
    c2_array = collect(c2_data)
    
    total_area = (maximum(c1_array) - minimum(c1_array)) * (maximum(c2_array) - minimum(c2_array))
    avg_point_spacing = sqrt(total_area / length(c1_array))
    search_radius = 2.0 * avg_point_spacing
    
    tree_data = [c1_array c2_array]'
    kdtree = KDTree(tree_data)
    
    rs_grid = zeros(length(c1_range), length(c2_range))
    
    total_points = length(c1_range) * length(c2_range)
    progress_interval = max(1, total_points ÷ 20)
    processed = 0
    
    for (i, c1) in enumerate(c1_range), (j, c2) in enumerate(c2_range)
        neighbors = inrange(kdtree, [c1, c2], search_radius)
        
        if !isempty(neighbors)
            weights = Float64[]
            values = Float64[]
            
            for neighbor_idx in neighbors
                dist = sqrt((c1 - c1_array[neighbor_idx])^2 + (c2 - c2_array[neighbor_idx])^2)
                weight = 1.0 / (dist + 1e-12)
                push!(weights, weight)
                push!(values, rs_data[neighbor_idx])
            end
            
            rs_grid[i, j] = sum(weights .* values) / sum(weights)
        else
            nearest_idx, _ = nn(kdtree, [c1, c2])
            rs_grid[i, j] = rs_data[nearest_idx]
        end
        
        processed += 1
        if processed % progress_interval == 0
            progress = round(100 * processed / total_points, digits=1)
            println("  Reynolds stress interpolation progress: $(progress)%")
        end
    end
    
    rs_grid = apply_smoothing_filter(rs_grid)
    
    return c1_range, c2_range, rs_grid, axis1_name, axis2_name
end

function export_to_netcdf(c1_range, c2_range, grid, xl, yl, var, axis, coord, filename)
    """Export grid data to NetCDF format for Python analysis"""
    println("Exporting to NetCDF: $filename")
    
    try
        # Convert ranges to arrays
        c1_array = collect(c1_range)
        c2_array = collect(c2_range)
        
        # Create NetCDF file
        NCDataset(filename, "c") do ds
            # Define dimensions
            defDim(ds, "x", length(c1_array))
            defDim(ds, "y", length(c2_array))
            
            # Define coordinate variables
            x_var = defVar(ds, "x", Float64, ("x",))
            y_var = defVar(ds, "y", Float64, ("y",))
            
            # Define data variable
            data_var = defVar(ds, var, Float64, ("x", "y"))
            
            # Write coordinate data
            x_var[:] = c1_array
            y_var[:] = c2_array
            
            # Write grid data
            data_var[:, :] = grid
            
            # Add attributes
            x_var.attrib["long_name"] = xl
            x_var.attrib["units"] = "m"  # Assuming meters - adjust as needed
            
            y_var.attrib["long_name"] = yl
            y_var.attrib["units"] = "m"  # Assuming meters - adjust as needed
            
            data_var.attrib["long_name"] = var
            data_var.attrib["slice_axis"] = axis
            data_var.attrib["slice_coordinate"] = coord
            
            # Global attributes
            ds.attrib["title"] = "$var slice at $axis = $coord"
            ds.attrib["created"] = string(Dates.now())
            ds.attrib["source"] = "Julia PVTU Analysis"
            ds.attrib["resolution"] = "$(length(c1_array))x$(length(c2_array))"
            ds.attrib["data_min"] = minimum(grid)
            ds.attrib["data_max"] = maximum(grid)
            ds.attrib["x_min"] = minimum(c1_array)
            ds.attrib["x_max"] = maximum(c1_array)
            ds.attrib["y_min"] = minimum(c2_array)
            ds.attrib["y_max"] = maximum(c2_array)
        end
        
        println("✓ Successfully exported NetCDF: $filename")
        return true
        
    catch e
        println("✗ Failed to export NetCDF: $filename - $e")
        return false
    end
end

function export_data(c1_range, c2_range, grid, xl, yl, var, axis, coord, base_filename, format::String)
    """Export grid data in specified format"""
    
    if format == "netcdf" || format == "both"
        # Export as NetCDF
        netcdf_file = base_filename * ".nc"
        export_to_netcdf(c1_range, c2_range, grid, xl, yl, var, axis, coord, netcdf_file)
    end
    
    if format == "csv" || format == "both"
        # Export as CSV with coordinates
        csv_file = base_filename * ".csv"
        
        println("Exporting CSV: $csv_file")
        
        # Create metadata
        metadata = Dict(
            "variable" => var,
            "axis" => axis,
            "coordinate" => coord,
            "x_label" => xl,
            "y_label" => yl,
            "x_min" => minimum(c1_range),
            "x_max" => maximum(c1_range),
            "y_min" => minimum(c2_range),
            "y_max" => maximum(c2_range),
            "resolution" => "$(length(c1_range))x$(length(c2_range))",
            "data_min" => minimum(grid),
            "data_max" => maximum(grid),
            "timestamp" => string(Dates.now())
        )
        
        # Create coordinate arrays
        c1_array = collect(c1_range)
        c2_array = collect(c2_range)
        
        # Prepare data for export: x, y, value
        export_data_array = Float64[]
        x_coords = Float64[]
        y_coords = Float64[]
        
        for i in 1:length(c1_range), j in 1:length(c2_range)
            push!(x_coords, c1_array[i])
            push!(y_coords, c2_array[j])
            push!(export_data_array, grid[i, j])
        end
        
        # Write CSV file
        try
            open(csv_file, "w") do io
                # Write header
                println(io, "X,Y,$var")
                
                # Write data
                for k in 1:length(x_coords)
                    println(io, "$(x_coords[k]),$(y_coords[k]),$(export_data_array[k])")
                end
            end
            
            # Write metadata file
            meta_file = base_filename * "_metadata.txt"
            open(meta_file, "w") do io
                for (key, value) in metadata
                    println(io, "$key: $value")
                end
            end
            
            println("✓ Exported CSV: $csv_file")
            println("✓ Exported metadata: $meta_file")
            
        catch e
            println("✗ Failed to export CSV: $csv_file - $e")
        end
    end
end

function process_single_file(pvtu_file::String, args::Dict, process_id::Int)
    """Process a single PVTU file according to specified parameters"""
    println("\n" * "="^80)
    println("PROCESSING: $pvtu_file (Process $process_id)")
    println("="^80)
    
    file_base = replace(basename(pvtu_file), ".pvtu" => "")
    
    try
        data = read_all_parallel_pieces(pvtu_file)
        
        variables = [String(strip(var)) for var in split(args["variables"], ",")]
        reynolds_components = [String(strip(comp)) for comp in split(args["reynolds-stress"], ",")]
        
        output_dir = joinpath(args["output-dir"], "process_$(process_id)")
        mkpath(output_dir)
        
        # Process velocity slices
        for var in variables
            if var in ["u", "v", "w"]
                println("\n--- Processing velocity slice: $var ---")
                c1, c2, grid, xl, yl = create_filled_contour_slice(data, var, "Z", args["slice-coord"], args["resolution"])
                base_filename = joinpath(output_dir, "$(file_base)_$(var)_slice_$(args["resolution"])")
                export_data(c1, c2, grid, xl, yl, var, "Z", args["slice-coord"], base_filename, args["export-format"])
            else
                println("Skipping unknown variable: $var (valid: u, v, w)")
            end
        end
        
        # Process Reynolds stress components
        for comp in reynolds_components
            if comp in ["uv", "uw", "vw", "uu", "vv", "ww"] && !isempty(comp)
                println("\n--- Processing Reynolds stress: $comp ---")
                c1, c2, rs_grid, xl, yl = calculate_reynolds_stress_contours(data, comp, "Y", args["resolution"])
                base_filename = joinpath(output_dir, "$(file_base)_reynolds_$(comp)_$(args["resolution"])")
                export_data(c1, c2, rs_grid, xl, yl, "Reynolds <$(comp)'>", "Y-avg", 0.0, base_filename, args["export-format"])
            elseif !isempty(comp)
                println("Skipping unknown Reynolds stress component: $comp")
            end
        end
        
        println("✓ Successfully processed: $pvtu_file")
        return true
        
    catch e
        println("✗ FAILED processing $pvtu_file: $e")
        showerror(stdout, e, catch_backtrace())
        return false
    end
end

function main()
    """Main function for batch processing"""
    args = parse_commandline()
    
    println("="^80)
    println("MINIMAL PVTU ANALYSIS - Process $(args["process-id"])")
    println("Started: $(Dates.now())")
    println("="^80)
    
    println("Configuration:")
    for (key, value) in args
        println("  $key: $value")
    end
    println()
    
    local available_files::Vector{String}
    local missing_files::Vector{Int} = Int[]
    
    if args["range"] !== nothing && length(args["range"]) == 3
        start_idx, end_idx, step = args["range"]
        available_files, missing_files = find_pvtu_files(args["file-prefix"], start_idx, end_idx, step)
        
        println("File range: $(start_idx):$(step):$(end_idx)")
        println("Available files: $(length(available_files))")
        if !isempty(missing_files)
            println("Missing files (indices): $(missing_files)")
        end
    else
        available_files = filter(f -> endswith(f, ".pvtu"), readdir("."))
        println("Processing all PVTU files in current directory: $(length(available_files)) files")
        
        if isempty(available_files)
            println("No files to process!")
            println("\nTroubleshooting:")
            println("1. Check if you're in the correct directory")
            println("2. Verify PVTU files exist with prefix '$(args["file-prefix"])'")
            println("3. Use --range START END STEP to specify file range")
            println("4. Use --dry-run to see what files would be processed")
            return
        end
        
        # Process files
        successful = 0
        failed = 0
        start_time = time()
        
        for (i, file) in enumerate(available_files)
            println("\n[$(i)/$(length(available_files))] Processing: $file")
            
            if process_single_file(file, args, args["process-id"])
                successful += 1
            else
                failed += 1
            end
            
            # Progress update
            elapsed = time() - start_time
            avg_time_per_file = elapsed / i
            estimated_remaining = avg_time_per_file * (length(available_files) - i)
            
            println("Progress: $(i)/$(length(available_files)) files completed")
            println("Elapsed: $(round(elapsed/60, digits=2)) min, Estimated remaining: $(round(estimated_remaining/60, digits=2)) min")
        end
        
        # Final summary
        total_time = time() - start_time
        println("\n" * "="^80)
        println("BATCH PROCESSING COMPLETE - Process $(args["process-id"])")
        println("="^80)
        println("Total files processed: $(length(available_files))")
        println("Successful: $successful")
        println("Failed: $failed")
        println("Total time: $(round(total_time/60, digits=2)) minutes")
        println("Average time per file: $(round(total_time/length(available_files), digits=2)) seconds")
        println("Finished: $(Dates.now())")
        println("="^80)
        
        # Show output summary
        if successful > 0
            output_dir = joinpath(args["output-dir"], "process_$(args["process-id"])")
            if isdir(output_dir)
                nc_files = length(filter(f -> endswith(f, ".nc"), readdir(output_dir)))
                csv_files = length(filter(f -> endswith(f, ".csv"), readdir(output_dir)))
                println("\nOutput files generated:")
                println("  NetCDF files: $nc_files")
                println("  CSV files: $csv_files")
                println("  Output directory: $output_dir")
            end
        end
    end
end 

# Run main function
main()

