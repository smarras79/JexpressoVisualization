#!/usr/bin/env julia
"""
BATCH PROCESSING implementation: All parallel pieces + Filled contours + SLURM support
Supports command line arguments for batch processing multiple PVTU files
"""

using Statistics
using GLMakie
using ColorSchemes
using NearestNeighbors
using ArgParse
using Dates

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
    s = ArgParseSettings(description = "Batch PVTU Analysis for SLURM")
    
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
        # Read only the first few KB for the header to improve performance
        header_str = String(file_data[1:min(end, 4096)])

        # Find the number of points from the main <Piece> tag. This is the most reliable source.
        piece_match = match(r"<Piece NumberOfPoints=\"(\d+)\"", header_str)
        if piece_match === nothing
            error("Could not find '<Piece NumberOfPoints=...>' tag")
        end
        num_points = parse(Int, piece_match.captures[1])

        # Find the start of the appended binary data section
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

        # Helper function to find the offset of a DataArray
        function get_offset(name_pattern)
            regex_offset = Regex("<DataArray[^>]*Name=\"$(name_pattern)\"[^>]*offset=\"(\\d+)\"")
            match_offset = match(regex_offset, header_str)
            if match_offset === nothing
                error("Could not find DataArray tag for '$name_pattern'")
            end
            return parse(Int, match_offset.captures[1])
        end

        # Helper function to read a block of binary data
        function read_block(offset::Int)
            header_pos = binary_start + offset
            block_size_header = reinterpret(UInt64, file_data[header_pos:header_pos+7])[1]
            data_start = header_pos + 8
            data_end = data_start + Int(block_size_header) - 1
            return reinterpret(Float64, file_data[data_start:data_end])
        end

        # Read all data blocks
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
    
    # Parse PVTU file
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

function create_filled_contour_slice(data::CompletePVTUData, var::String, axis::String, coord::Float64, resolution::Int)
    """Create filled contour slice with specified resolution"""
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
   
    # Convert views to regular arrays
    c1_array = collect(c1_data)
    c2_array = collect(c2_data)
    
    # IMPROVED INTERPOLATION: Use weighted average instead of nearest neighbor
    grid_data = zeros(length(c1_range), length(c2_range))
    
    # Calculate reasonable search radius based on point density
    total_area = (maximum(c1_array) - minimum(c1_array)) * (maximum(c2_array) - minimum(c2_array))
    avg_point_spacing = sqrt(total_area / length(c1_array))
    search_radius = 2.0 * avg_point_spacing
    
    # Build KDTree for efficient neighbor search
    tree_data = [c1_array c2_array]'
    kdtree = KDTree(tree_data)
    
    for (i, c1) in enumerate(c1_range), (j, c2) in enumerate(c2_range)
        # Find all points within search radius
        neighbors = inrange(kdtree, [c1, c2], search_radius)
        
        if !isempty(neighbors)
            # Weighted average based on inverse distance
            weights = Float64[]
            values  = Float64[]
            
            for neighbor_idx in neighbors
                dist = sqrt((c1 - c1_array[neighbor_idx])^2 + (c2 - c2_array[neighbor_idx])^2)
                weight = 1.0 / (dist + 1e-12)  # Add small epsilon to avoid division by zero
                push!(weights, weight)
                push!(values, slice_field[neighbor_idx])
            end
            
            # Weighted average
            grid_data[i, j] = sum(weights .* values) / sum(weights)
        else
            # Fallback to nearest neighbor if no points in radius
            nearest_idx, _ = nn(kdtree, [c1, c2])
            grid_data[i, j] = slice_field[nearest_idx]
        end
    end
    
    # Apply smoothing filter to reduce remaining noise
    grid_data = apply_smoothing_filter(grid_data)
    
    axis_names = ["X", "Y", "Z"]
    return c1_range, c2_range, grid_data, axis_names[other_axes[1]], axis_names[other_axes[2]]
end

function plot_filled_contours(c1_range, c2_range, grid, xl, yl, var, axis, coord, file)
    """Plot filled contours with improved visualization"""
    # Use Figure() constructor without deprecated resolution parameter
    fig = Figure(size = (1400, 1000), fontsize = 18)
    ax = Axis(fig[1, 1], xlabel=xl, ylabel=yl, title="$var slice at $axis = $coord", aspect=DataAspect())
    
    # IMPROVED CONTOUR PLOTTING: Use more levels and better colormap
    # Calculate sensible contour levels
    data_min, data_max = extrema(grid)
    data_range = data_max - data_min
    
    # Use adaptive number of levels based on data range
    if abs(data_range) < 1e-10
        levels = 10  # Fallback for nearly constant data
    else
        levels = 30  # More levels for smoother appearance
    end
    
    # Create contour plot with improved settings
    co = contourf!(ax, c1_range, c2_range, grid', 
                   levels=levels, 
                   colormap=:viridis,
                   extendlow=:auto,
                   extendhigh=:auto)
    
    Colorbar(fig[1, 2], co, label=var)
    
    # Add contour lines for better visualization
    contour!(ax, c1_range, c2_range, grid', 
             levels=levels÷2, 
             color=:black, 
             alpha=0.3, 
             linewidth=0.5)
    
    # Add error handling for file saving
    try
        save(file, fig, px_per_unit = 2)
        println("✓ Saved: $file")
    catch e
        println("Error saving plot: $e")
        # Try alternative save without px_per_unit
        try
            save(file, fig)
            println("✓ Saved: $file (without px_per_unit)")
        catch e2
            println("✗ Failed to save: $file - $e2")
        end
    end
end

# Smoothing filter to reduce noise
function apply_smoothing_filter(grid::Matrix{Float64}, kernel_size::Int=3)
    """Apply a simple box filter to smooth the grid data"""
    rows, cols = size(grid)
    smoothed = copy(grid)
    half_kernel = kernel_size ÷ 2
    
    for i in (1+half_kernel):(rows-half_kernel)
        for j in (1+half_kernel):(cols-half_kernel)
            # Average over kernel window
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

function calculate_reynolds_stress_contours(data::CompletePVTUData, comp::String, avg_axis::String, resolution::Int)
    """Calculate Reynolds stress contours with specified resolution"""
    # Calculate fluctuating components
    u_p = data.u .- mean(data.u)
    v_p = data.v .- mean(data.v)
    w_p = data.w .- mean(data.w)
    
    # Calculate Reynolds stress properly
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
    
    c1_data = view(data.points, other_axes[1], :)
    c2_data = view(data.points, other_axes[2], :)
    
    c1_range = range(extrema(c1_data)..., length=resolution)
    c2_range = range(extrema(c2_data)..., length=resolution)
    
    # IMPROVED REYNOLDS STRESS INTERPOLATION
    c1_array = collect(c1_data)
    c2_array = collect(c2_data)
    
    # Calculate reasonable search radius
    total_area = (maximum(c1_array) - minimum(c1_array)) * (maximum(c2_array) - minimum(c2_array))
    avg_point_spacing = sqrt(total_area / length(c1_array))
    search_radius = 2.0 * avg_point_spacing
    
    tree_data = [c1_array c2_array]'
    kdtree = KDTree(tree_data)
    
    # Pre-allocate the grid for better performance
    rs_grid = zeros(length(c1_range), length(c2_range))
    
    for (i, c1) in enumerate(c1_range), (j, c2) in enumerate(c2_range)
        # Use weighted interpolation for Reynolds stress too
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
    end
    
    # Apply smoothing
    rs_grid = apply_smoothing_filter(rs_grid)
    
    axis_names = ["X", "Y", "Z"]
    return c1_range, c2_range, rs_grid, axis_names[other_axes[1]], axis_names[other_axes[2]]
end

function process_single_file(pvtu_file::String, args::Dict, process_id::Int)
    """Process a single PVTU file according to specified parameters"""
    println("\n" * "="^80)
    println("PROCESSING: $pvtu_file (Process $process_id)")
    println("="^80)
    
    # Extract file identifier for output naming
    file_base = replace(basename(pvtu_file), ".pvtu" => "")
    
    try
        # Read data
        data = read_all_parallel_pieces(pvtu_file)
        
        # Parse variables and Reynolds stress components - convert to String
        variables = [String(strip(var)) for var in split(args["variables"], ",")]
        reynolds_components = [String(strip(comp)) for comp in split(args["reynolds-stress"], ",")]
        
        # Create output directory
        output_dir = joinpath(args["output-dir"], "process_$(process_id)")
        mkpath(output_dir)
        
        # Process velocity slices
        for var in variables
            if var in ["u", "v", "w"]
                println("\n--- Processing velocity slice: $var ---")
                c1, c2, grid, xl, yl = create_filled_contour_slice(data, var, "Z", args["slice-coord"], args["resolution"])
                output_file = joinpath(output_dir, "$(file_base)_$(var)_slice_$(args["resolution"]).png")
                plot_filled_contours(c1, c2, grid, xl, yl, var, "Z", args["slice-coord"], output_file)
            else
                println("Skipping unknown variable: $var (valid: u, v, w)")
            end
        end
        
        # Process Reynolds stress components
        for comp in reynolds_components
            if comp in ["uv", "uw", "vw", "uu", "vv", "ww"] && !isempty(comp)
                println("\n--- Processing Reynolds stress: $comp ---")
                c1, c2, rs_grid, xl, yl = calculate_reynolds_stress_contours(data, comp, "Y", args["resolution"])
                output_file = joinpath(output_dir, "$(file_base)_reynolds_$(comp)_$(args["resolution"]).png")
                plot_filled_contours(c1, c2, rs_grid, xl, yl, "Reynolds <$(comp)'>", "Y-avg", 0.0, output_file)
            elseif !isempty(comp)
                println("Skipping unknown Reynolds stress component: $comp (valid: uv, uw, vw, uu, vv, ww)")
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
    println("BATCH PVTU ANALYSIS - Process $(args["process-id"])")
    println("Started: $(Dates.now())")
    println("="^80)
    
    # Print configuration
    println("Configuration:")
    for (key, value) in args
        println("  $key: $value")
    end
    println()
    
    # Determine files to process
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
        # Process all PVTU files in current directory
        available_files = filter(f -> endswith(f, ".pvtu"), readdir("."))
        println("Processing all PVTU files in current directory: $(length(available_files)) files")
        
        if isempty(available_files)
            println("No PVTU files found in current directory!")
            println("Current directory contents:")
            for file in readdir(".")
                if contains(file, args["file-prefix"]) || endswith(file, ".pvtu")
                    println("  $file")
                end
            end
        end
    end
    
    # Handle dry run
    if args["dry-run"]
        println("\nDRY RUN - Files that would be processed:")
        for file in available_files
            println("  $file")
        end
        println("\nTotal: $(length(available_files)) files")
        return
    end
    
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
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
    try
        using NearestNeighbors, GLMakie, ColorSchemes, ArgParse, Dates
    catch
        using Pkg
        Pkg.add.(["NearestNeighbors", "GLMakie", "ColorSchemes", "ArgParse", "Dates"])
        using NearestNeighbors, GLMakie, ColorSchemes, ArgParse, Dates
    end
    
    main()
end
