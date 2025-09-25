#!/usr/bin/env julia
"""
INTERACTIVE implementation: All parallel pieces + Filled contours + User-selectable resolution
Allows user to choose interpolation resolution for grid generation
"""

using Statistics
using GLMakie
using ColorSchemes
using NearestNeighbors

struct CompletePVTUData
    points::Matrix{Float64}
    u::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
    n_points::Int
    bounds::NamedTuple
    n_pieces::Int
end

#
#  MODIFIED FUNCTION
#  This version reads the NumberOfPoints from the main <Piece> tag first,
#  making it robust to variations in the rest of the XML header.
#
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
    println("=== READING ALL PARALLEL PIECES ===")
    
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
    
    println("\nSuccessfully read $successful/$(length(piece_files)) pieces")
    println("Total points across all pieces: $total_points")
    
    if isempty(all_points); error("No data pieces were read successfully."); end
    
    combined_points = hcat(all_points...)
    combined_u, combined_v, combined_w = vcat(all_u...), vcat(all_v...), vcat(all_w...)
    
    bounds = (
        xmin=minimum(view(combined_points, 1, :)), xmax=maximum(view(combined_points, 1, :)),
        ymin=minimum(view(combined_points, 2, :)), ymax=maximum(view(combined_points, 2, :)),
        zmin=minimum(view(combined_points, 3, :)), zmax=maximum(view(combined_points, 3, :))
    )
    
    println("\nCOMPLETE DOMAIN from ALL pieces:")
    println("  Points: $(size(combined_points, 2))")
    println("  X: [$(bounds.xmin), $(bounds.xmax)]")
    println("  Y: [$(bounds.ymin), $(bounds.ymax)]") 
    println("  Z: [$(bounds.zmin), $(bounds.zmax)]")
    
    return CompletePVTUData(combined_points, combined_u, combined_v, combined_w, 
                            total_points, bounds, successful)
end

function get_user_resolution()
    """Interactive function to get resolution from user with suggestions"""
    println("\n" * "="^60)
    println("GRID RESOLUTION SELECTION")
    println("="^60)
    println("Choose the interpolation grid resolution for slice generation:")
    println()
    println("Resolution options and their characteristics:")
    println("  • Low (100x100):     Fast, good for quick preview")
    println("  • Medium (200x200):  Balanced speed/quality")
    println("  • High (400x400):    Detailed, slower processing")
    println("  • Ultra (800x800):   Maximum detail, slow")
    println("  • Custom (N×N):      Your own choice")
    println()
    println("Recommendations:")
    println("  - Start with Medium (200) for exploration")
    println("  - Use High (400) for publication-quality plots")
    println("  - Use Ultra (800) only for final high-resolution images")
    println()
    
    local resolution::Int  # Declare variable explicitly
    
    while true
        print("Enter your choice [Low/Medium/High/Ultra/Custom] or number: ")
        user_input = strip(readline())
        
        # Handle different input formats
        choice = lowercase(user_input)
        
        if choice in ["low", "l", "100"]
            resolution = 100
            println("Selected: Low resolution ($(resolution)×$(resolution))")
            return resolution
        elseif choice in ["medium", "m", "med", "200"]
            resolution = 200
            println("Selected: Medium resolution ($(resolution)×$(resolution))")
            return resolution
        elseif choice in ["high", "h", "400"]
            resolution = 400
            println("Selected: High resolution ($(resolution)×$(resolution))")
            return resolution
        elseif choice in ["ultra", "u", "800"]
            resolution = 800
            println("Selected: Ultra resolution ($(resolution)×$(resolution))")
            return resolution
        elseif choice in ["custom", "c"]
            while true
                print("Enter custom resolution (e.g., 300): ")
                custom_input = strip(readline())
                try
                    resolution = parse(Int, custom_input)
                    if resolution < 50
                        println("Warning: Resolution too low (< 50), using minimum of 50")
                        resolution = 50
                    elseif resolution > 2000
                        println("Warning: Very high resolution (> 2000), this may take a long time!")
                        print("Continue with $(resolution)? [y/n]: ")
                        confirm = lowercase(strip(readline()))
                        if confirm != "y" && confirm != "yes"
                            continue
                        end
                    end
                    println("Selected: Custom resolution ($(resolution)×$(resolution))")
                    return resolution
                catch
                    println("Invalid input. Please enter a number.")
                end
            end
        else
            # Try to parse as direct number
            try
                resolution = parse(Int, user_input)
                if resolution < 50
                    println("Warning: Resolution too low (< 50), using minimum of 50")
                    resolution = 50
                elseif resolution > 2000
                    println("Warning: Very high resolution (> 2000), this may take a long time!")
                    print("Continue with $(resolution)? [y/n]: ")
                    confirm = lowercase(strip(readline()))
                    if confirm != "y" && confirm != "yes"
                        continue
                    end
                end
                println("Selected: $(resolution)×$(resolution) resolution")
                return resolution
            catch
                println("Invalid input. Please try again.")
                println("Valid options: Low, Medium, High, Ultra, Custom, or a number")
            end
        end
    end
end

function create_filled_contour_slice(data::CompletePVTUData, var::String, axis::String, coord::Float64, resolution::Int)
    println("\n=== CREATING FILLED CONTOUR SLICE: $var at $axis=$coord ===")
    println("Using resolution: $(resolution)×$(resolution)")
    
    field_data = getproperty(data, Symbol(var))
    axis_idx = Dict("X" => 1, "Y" => 2, "Z" => 3)[uppercase(axis)]
    
    tolerance = 0.05 * (getproperty(data.bounds, Symbol(lowercase(axis) * "max")) - getproperty(data.bounds, Symbol(lowercase(axis) * "min")))
    slice_mask = abs.(view(data.points, axis_idx, :) .- coord) .<= tolerance
    
    println("Found $(sum(slice_mask)) points for slice with tolerance $tolerance")
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
    
    println("Using search radius: $search_radius for interpolation")
    
    # Build KDTree for efficient neighbor search
    tree_data = [c1_array c2_array]'
    kdtree = KDTree(tree_data)
    
    # Progress tracking for large grids
    total_points = length(c1_range) * length(c2_range)
    progress_interval = max(1, total_points ÷ 20)  # Update every 5%
    processed = 0
    
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
        
        # Progress update for large grids
        processed += 1
        if processed % progress_interval == 0
            progress = round(100 * processed / total_points, digits=1)
            println("  Interpolation progress: $(progress)%")
        end
    end
    
    # Apply smoothing filter to reduce remaining noise
    grid_data = apply_smoothing_filter(grid_data)
    
    println("Grid interpolation completed: $(size(grid_data))")
    axis_names = ["X", "Y", "Z"]
    return c1_range, c2_range, grid_data, axis_names[other_axes[1]], axis_names[other_axes[2]]
end

function plot_filled_contours(c1_range, c2_range, grid, xl, yl, var, axis, coord, file)
    println("\n=== PLOTTING: $file ===")
    
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
        println("Saved plot to: $file")
    catch e
        println("Error saving plot: $e")
        # Try alternative save without px_per_unit
        try
            save(file, fig)
            println("Saved plot to: $file (without px_per_unit)")
        catch e2
            println("Failed to save plot: $e2")
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
    println("\n=== REYNOLDS STRESS: <$comp'> averaged over $avg_axis ===")
    println("Using resolution: $(resolution)×$(resolution)")
    
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
    
    # Progress tracking
    total_points = length(c1_range) * length(c2_range)
    progress_interval = max(1, total_points ÷ 20)
    processed = 0
    
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
        
        # Progress update
        processed += 1
        if processed % progress_interval == 0
            progress = round(100 * processed / total_points, digits=1)
            println("  Reynolds stress interpolation progress: $(progress)%")
        end
    end
    
    # Apply smoothing
    rs_grid = apply_smoothing_filter(rs_grid)
    
    axis_names = ["X", "Y", "Z"]
    return c1_range, c2_range, rs_grid, axis_names[other_axes[1]], axis_names[other_axes[2]]
end

function main()
    try
        println("Starting PVTU Analysis with Interactive Resolution Selection")
        println("="^70)
        
        # Check for command line resolution argument first
        cmd_resolution = parse_command_line_args()
        
        # Get resolution (either from command line or interactive)
        if cmd_resolution !== nothing
            println("Using command line resolution: $(cmd_resolution)")
            resolution = cmd_resolution
        else
            resolution = get_user_resolution()
        end
        
        println("\nLoading data...")
        data = read_all_parallel_pieces("iter_406.pvtu")
        mkpath("correct_output")

        # Test 1: Velocity slice
        println("\n" * "="^60)
        println("GENERATING VELOCITY SLICE")
        println("="^60)
        z_mid = 0.1
        c1, c2, grid, xl, yl = create_filled_contour_slice(data, "w", "Z", z_mid, resolution)
        plot_filled_contours(c1, c2, grid, xl, yl, "w", "Z", z_mid, "correct_output/w_slice_$(resolution).png")

        # Test 2: Reynolds stress
        println("\n" * "="^60)
        println("GENERATING REYNOLDS STRESS PLOT")
        println("="^60)
        c1, c2, rs_grid, xl, yl = calculate_reynolds_stress_contours(data, "uv", "Y", resolution)
        plot_filled_contours(c1, c2, rs_grid, xl, yl, "Reynolds Stress <u'v'>", "Y-avg", 0.0, "correct_output/reynolds_uv_$(resolution).png")
        
        println("\n" * "="^70)
        println("✓ Analysis successful!")
        println("Files saved with resolution $(resolution)×$(resolution):")
        println("  • correct_output/w_slice_$(resolution).png")
        println("  • correct_output/reynolds_uv_$(resolution).png")
        println("="^70)
        
    catch e
        println("\n✗ FAILED: $e"); 
        showerror(stdout, e, catch_backtrace())
    end
end

# Command line argument processing
function parse_command_line_args()
    """Parse command line arguments for batch processing"""
    if length(ARGS) > 0
        try
            resolution = parse(Int, ARGS[1])
            if resolution < 50 || resolution > 2000
                println("Warning: Resolution $(resolution) outside recommended range (50-2000)")
            end
            return resolution
        catch
            println("Invalid resolution argument: $(ARGS[1])")
            println("Usage: julia script.jl [resolution]")
            return nothing
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        using NearestNeighbors, GLMakie, ColorSchemes
    catch
        using Pkg; 
        Pkg.add.(["NearestNeighbors", "GLMakie", "ColorSchemes"]); 
        using NearestNeighbors, GLMakie, ColorSchemes
    end
    
    main()
end
