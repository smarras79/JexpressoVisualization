#!/usr/bin/env julia
"""
        Parallel-Enabled ParaView Batch Processor (Julia Version)

        Computes slices, spatial averages, or Reynolds stresses from simulation data.

        TO RUN THIS:
        ==============================================================================
        julia batch_paraview.jl
        julia batch_paraview.jl --range 100 200 2 --skip-existing
        ==============================================================================
        """

using ArgParse
using Dates
using Logging
using Printf
using Base.Filesystem

#-------------------------------
# BATCH PROCESSING CONFIGURATION
#-------------------------------

# --- File Pattern Configuration ---#
const FILE_PATTERNS = Dict{String, Any}(
    "pattern_type" => "iteration",
    "base_directory" => "./",
    "file_template" => "iter_{}.pvtu",
)

# --- Analysis Configuration ---#
const BATCH_CONFIG = Dict{String, Any}(
    #
    # Set the desired analysis mode here
    #
    # "averaging", "slicing", or "reynolds_stress"
    #"analysis_mode" => "slicing",   
    #"analysis_mode" => "averaging",
    "analysis_mode" => "reynolds_stress",
    
    #-------------------------------
    # AVERAGING
    #-------------------------------
    "averaging" => Dict{String, Any}(
        "data_array" => "VELOMAG", # Which variable to average
        "axis" => "Y",
        "resolution" => [150, 150, 150],
        "geometric_limits" => Dict{String, Any}(
            "X" => [nothing, nothing],
            "Y" => [nothing, nothing],
            "Z" => [nothing, nothing]
        )
    ),

    #-------------------------------
    # SLICING
    #-------------------------------
    "slicing" => Dict{String, Any}(
        "data_array" => "w", # Which variable to slice
        "axis" => "Z",
        "coordinate" => 100  # Use nothing for auto-center
    ),

    #-------------------------------
    # REYNOLDS STRESS
    #-------------------------------
    "reynolds_stress" => Dict{String, Any}(
        # This is the variable you want to visualize
        "component" => "uv", # Options: "uu", "vv", "ww", "uv", "uw", "vw"
        
        # This is the axis along which the spatial average is calculated
        "averaging_axis" => "Y",
        "resolution" => [150, 150, 150], # Resolution for intermediate calculations
        "geometric_limits" => Dict{String, Any}(
            "X" => [nothing, nothing],
            "Y" => [nothing, nothing],
            "Z" => [nothing, nothing]
        )
    ),

    #-------------------------------
    # VISUALIZATION
    #-------------------------------
    "visualization" => Dict{String, Any}(
        "image_size" => [1200, 800],
        "color_map" => "Viridis (matplotlib)"
    )
)

# --- Processing Options ---
const PROCESSING_OPTIONS = Dict{String, Any}(
    "output_directory" => "./batch_output/",
    "continue_on_error" => true,
    "paraview_executable" => "/Applications/ParaView-5.11.2.app/Contents/bin/pvpython",
    "paraview_args" => ["--force-offscreen-rendering"],
    "timeout_seconds" => 400,
    "log_file_prefix" => "batch_processing"
)

#-------------------------------
# COMMAND LINE ARGUMENT PARSING
#-------------------------------

function parse_arguments()
    s = ArgParseSettings(description = "ParaView Batch Processor with Parallel Support")

    @add_arg_table! s begin
        "--range"
            help = "Process files in range: start end step"
            nargs = 3
            arg_type = Int
        "--files"
            help = "Process specific files"
            nargs = '*'
        "--suggest-parallel"
            help = "Suggest how to split files"
            action = :store_true
        "--process-id"
            help = "Process ID for parallel runs"
            arg_type = Int
        "--output-dir"
            help = "Override output directory"
            arg_type = String
        "--log-level"
            help = "Set logging level"
            default = "INFO"
            range_tester = x -> x in ["DEBUG", "INFO", "WARNING", "ERROR"]
        "--dry-run"
            help = "Show what files would be processed"
            action = :store_true
        "--skip-existing"
            help = "Skip processing if the output file already exists"
            action = :store_true
        "--num-processes"
            help = "Number of parallel processes to suggest"
            arg_type = Int
            default = 4
    end
    return parse_args(ARGS, s)
end

#-------------------------------
# IMPLEMENTATION
#-------------------------------
function setup_logging(process_id = nothing, log_level = "INFO")
    if process_id !== nothing
        log_file = "$(PROCESSING_OPTIONS["log_file_prefix"])_proc_$(lpad(process_id, 3, '0')).log"
    else
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        log_file = "$(PROCESSING_OPTIONS["log_file_prefix"])_$(timestamp).log"
    end

    level = if log_level == "DEBUG"; Logging.Debug
    elseif log_level == "INFO"; Logging.Info
    elseif log_level == "WARNING"; Logging.Warn
    else; Logging.Error
    end

    io = open(log_file, "w")
    file_logger = SimpleLogger(io, level)
    
    global_logger(TeeLogger(ConsoleLogger(stdout, level), file_logger))
    @info "Logging to: $log_file"
    return io
end

struct TeeLogger <: AbstractLogger
    loggers::Vector{AbstractLogger}
end
TeeLogger(loggers...) = TeeLogger(collect(loggers))
Logging.handle_message(logger::TeeLogger, args...; kwargs...) = for l in logger.loggers; Logging.handle_message(l, args...; kwargs...); end
Logging.shouldlog(logger::TeeLogger, args...) = any(Logging.shouldlog(l, args...) for l in logger.loggers)
Logging.min_enabled_level(logger::TeeLogger) = minimum(Logging.min_enabled_level(l) for l in logger.loggers)

# *** THIS IS THE CORRECTED FUNCTION ***
function extract_number_from_filename(filepath::String)
    """Extract number from filename, first by template, then by general search."""
    filename = basename(filepath)
    template = FILE_PATTERNS["file_template"]

    # 1. Try to match the specific template
    regex_template = replace(replace(template, r"[.^$*+?{}[\]\\|()\-]" => s"\\&"), "{}" => raw"([+-]?(?:\d+\.?\d*|\.\d+))")
    m = match(Regex(regex_template), filename)
    if m !== nothing
        try
            num_str = m.captures[1]
            try
                return parse(Int, num_str)
            catch
                return parse(Float64, num_str)
            end
        catch
             # Fall through to the general search
        end
    end

    # 2. Fallback: find the first number in the filename
    m_fallback = match(r"([+-]?(?:\d+\.?\d*|\.\d+))", filename)
    if m_fallback !== nothing
        try
            num_str = m_fallback.captures[1]
            try
                return parse(Int, num_str)
            catch
                return parse(Float64, num_str)
            end
        catch
            # Fall through to returning 0
        end
    end

    @warn "Could not extract a number from filename: $filename. Defaulting to 0."
    return 0
end


function find_files(args::Dict)
    base_dir = FILE_PATTERNS["base_directory"]
    template = FILE_PATTERNS["file_template"]
    if get(args, "files", nothing) !== nothing && !isempty(args["files"])
        return filter(isfile, abspath.(args["files"]))
    elseif get(args, "range", nothing) !== nothing
        start_val, end_val, step = args["range"]
        files = String[]; for i in start_val:step:end_val; f = joinpath(base_dir, replace(template, "{}"=>string(i))); isfile(f) && push!(files, abspath(f)); end; return files
    else
        glob_pattern = replace(template, "{}" => "*")
        all_files = isdir(base_dir) ? readdir(base_dir) : []
        pattern_regex = Regex("^" * replace(replace(glob_pattern, r"[.^$+?{}[\]\\|()\-]" => s"\\&"), "*" => ".*") * raw"$")
        matching_files = filter(f -> match(pattern_regex, f) !== nothing, all_files)
        files_with_numbers = [(extract_number_from_filename(f), abspath(joinpath(base_dir, f))) for f in matching_files]
        sort!(files_with_numbers, by = x -> x[1])
        return [f[2] for f in files_with_numbers]
    end
end

function generate_output_filename(input_file::String, output_dir::String)
    file_number = extract_number_from_filename(input_file)
    number_str = isa(file_number, Int) ? @sprintf("%06d", file_number) : replace(@sprintf("%08.3f", file_number), "." => "_")
    mode = BATCH_CONFIG["analysis_mode"]
    local filename
    if mode == "averaging"; config = BATCH_CONFIG["averaging"]; filename = "$(config["data_array"])_avg_$(config["axis"])_$(number_str).png"
    elseif mode == "slicing"; config = BATCH_CONFIG["slicing"]; filename = "$(config["data_array"])_slice_$(config["axis"])_$(number_str).png"
    elseif mode == "reynolds_stress"; config = BATCH_CONFIG["reynolds_stress"]; filename = "RS_$(config["component"])_avg_$(config["averaging_axis"])_$(number_str).png"
    else; filename = "output_$(number_str).png"
    end
    return joinpath(output_dir, filename)
end

function generate_processing_script(input_file::String, output_file::String)
    input_abs = abspath(input_file)
    output_abs = abspath(output_file)
    mode = BATCH_CONFIG["analysis_mode"]
    vis_config = BATCH_CONFIG["visualization"]
    image_size = vis_config["image_size"]
    color_map = vis_config["color_map"]

    analysis_code = ""
    camera_code = ""
    data_array = ""
    velomag_calc = ""

    # --- Slicing Mode ---
    if mode == "slicing"
        config = BATCH_CONFIG["slicing"]
        data_array = config["data_array"]
        axis = config["axis"]
        coordinate = config["coordinate"]

        if data_array == "VELOMAG"
            velomag_calc = "        source = Calculator(Input=reader, ResultArrayName='VELOMAG', Function='sqrt(u*u+v*v+w*w)')"
        end

        analysis_code = """
        # Slicing analysis
        axis = '$axis'.upper()
        coordinate = $(coordinate === nothing ? "None" : coordinate)

        bounds = source.GetDataInformation().GetBounds()
        if coordinate is None:
            bounds_idx = {'X': [0, 1], 'Y': [2, 3], 'Z': [4, 5]}[axis]
            coordinate = (bounds[bounds_idx[0]] + bounds[bounds_idx[1]]) / 2.0
        
        print(f"Creating slice at {axis} = {coordinate}")
        slice_filter = Slice(Input=source, SliceType='Plane')
        if axis == 'X':
            slice_filter.SliceType.Origin = [coordinate, 0, 0]
            slice_filter.SliceType.Normal = [1, 0, 0]
        elif axis == 'Y':
            slice_filter.SliceType.Origin = [0, coordinate, 0]
            slice_filter.SliceType.Normal = [0, 1, 0]
        else: # Z
            slice_filter.SliceType.Origin = [0, 0, coordinate]
            slice_filter.SliceType.Normal = [0, 0, 1]
        
        source = slice_filter
"""
        camera_code = """
        slice_axis = '$axis'.upper()
        if slice_axis == 'X':
            view.CameraPosition = [10, 0, 0]; view.CameraViewUp = [0, 0, 1]
        elif slice_axis == 'Y':
            view.CameraPosition = [0, 10, 0]; view.CameraViewUp = [0, 0, 1]
        else: # Z
            view.CameraPosition = [0, 0, 10]; view.CameraViewUp = [0, 1, 0]
"""

    # --- Reynolds Stress Mode ---
    elseif mode == "reynolds_stress"
        config = BATCH_CONFIG["reynolds_stress"]
        comp = config["component"]
        axis = config["averaging_axis"]
        resolution = config["resolution"]
        data_array = "RS_$(comp)_avg"
        axis_map_py = "{'X': 2, 'Y': 1, 'Z': 0}"
        
        analysis_code = """
        # --- Reynolds Stress Calculation ---
        print('Starting Reynolds Stress Calculation...')
        from vtkmodules.numpy_interface import dataset_adapter as dsa
        from vtk.util.numpy_support import numpy_to_vtk
        import numpy as np

        resampled_3d = ResampleToImage(Input=source, SamplingDimensions=$resolution)
        resampled_3d.UpdatePipeline()
        
        vtk_data_3d = servermanager.Fetch(resampled_3d)
        wrapped_data_3d = dsa.WrapDataObject(vtk_data_3d)
        dims = resampled_3d.SamplingDimensions
        
        u_inst = wrapped_data_3d.PointData['u'].reshape(dims[2], dims[1], dims[0])
        v_inst = wrapped_data_3d.PointData['v'].reshape(dims[2], dims[1], dims[0])
        w_inst = wrapped_data_3d.PointData['w'].reshape(dims[2], dims[1], dims[0])

        print(f"Calculating mean velocity field by averaging along the '$axis'-axis...")
        axis_map = $axis_map_py
        avg_axis_idx = axis_map.get('$axis'.upper())
        
        u_mean_2d = np.mean(u_inst, axis=avg_axis_idx)
        v_mean_2d = np.mean(v_inst, axis=avg_axis_idx)
        w_mean_2d = np.mean(w_inst, axis=avg_axis_idx)

        if avg_axis_idx == 2: # X-axis average
            u_mean_3d = np.tile(u_mean_2d[:, :, np.newaxis], (1, 1, dims[0]))
            v_mean_3d = np.tile(v_mean_2d[:, :, np.newaxis], (1, 1, dims[0]))
            w_mean_3d = np.tile(w_mean_2d[:, :, np.newaxis], (1, 1, dims[0]))
        elif avg_axis_idx == 1: # Y-axis average
            u_mean_3d = np.tile(u_mean_2d[:, np.newaxis, :], (1, dims[1], 1))
            v_mean_3d = np.tile(v_mean_2d[:, np.newaxis, :], (1, dims[1], 1))
            w_mean_3d = np.tile(w_mean_2d[:, np.newaxis, :], (1, dims[1], 1))
        else: # Z-axis average
            u_mean_3d = np.tile(u_mean_2d[np.newaxis, :, :], (dims[2], 1, 1))
            v_mean_3d = np.tile(v_mean_2d[np.newaxis, :, :], (dims[2], 1, 1))
            w_mean_3d = np.tile(w_mean_2d[np.newaxis, :, :], (dims[2], 1, 1))

        u_prime = u_inst - u_mean_3d
        v_prime = v_inst - v_mean_3d
        w_prime = w_inst - w_mean_3d

        rs_3d = {"uu": u_prime**2, "vv": v_prime**2, "ww": w_prime**2, "uv": u_prime*v_prime, "uw": u_prime*w_prime, "vw": v_prime*w_prime}['$comp']
        final_component_2d = np.mean(rs_3d, axis=avg_axis_idx)

        from vtk import vtkStructuredGrid, vtkPoints
        original_bounds = source.GetDataInformation().GetBounds()
        
        if avg_axis_idx == 2: grid_dims, bounds_indices = ([dims[1], dims[2]], [2, 3, 4, 5])
        elif avg_axis_idx == 1: grid_dims, bounds_indices = ([dims[0], dims[2]], [0, 1, 4, 5])
        else: grid_dims, bounds_indices = ([dims[0], dims[1]], [0, 1, 2, 3])

        n1, n2 = grid_dims
        structured_grid = vtkStructuredGrid(); structured_grid.SetDimensions(n1, n2, 1)
        points = vtkPoints()
        c1 = np.linspace(original_bounds[bounds_indices[0]], original_bounds[bounds_indices[1]], n1)
        c2 = np.linspace(original_bounds[bounds_indices[2]], original_bounds[bounds_indices[3]], n2)
        avg_c = (original_bounds[avg_axis_idx*2] + original_bounds[avg_axis_idx*2+1]) / 2.0
        for j in range(n2):
            for i in range(n1):
                if avg_axis_idx == 2: points.InsertNextPoint(avg_c, c1[i], c2[j])
                elif avg_axis_idx == 1: points.InsertNextPoint(c1[i], avg_c, c2[j])
                else: points.InsertNextPoint(c1[i], c2[j], avg_c)
        
        structured_grid.SetPoints(points)
        vtk_array = numpy_to_vtk(final_component_2d.flatten('C'), deep=True); vtk_array.SetName('$data_array')
        structured_grid.GetPointData().SetScalars(vtk_array)
        
        producer = TrivialProducer(registrationName='FinalData'); producer.GetClientSideObject().SetOutput(structured_grid)
        source = producer
"""
        camera_code = """
        avg_axis = '$axis'.upper()
        if avg_axis == 'X':
            view.CameraPosition = [10, 0, 0]; view.CameraViewUp = [0, 0, 1]
        elif avg_axis == 'Y':
            view.CameraPosition = [0, 10, 0]; view.CameraViewUp = [0, 0, 1]
        else: # Z
            view.CameraPosition = [0, 0, 10]; view.CameraViewUp = [0, 1, 0]
"""
    # (Add averaging mode here if needed)
    end

    # --- FINAL SCRIPT ASSEMBLY ---
    return """#!/usr/bin/env python3
import os, sys
from paraview.simple import *
def main():
    try:
        print("Starting ParaView processing...")
        paraview.simple._DisableFirstRenderCameraReset()
        reader = XMLPartitionedUnstructuredGridReader(FileName=[r'$input_abs'])
        reader.UpdatePipeline()
        if reader.GetDataInformation().GetNumberOfPoints() == 0:
            print("ERROR: No data points found"); return False
        
        source = reader
        
        point_data_info = reader.GetPointDataInformation()
        available_arrays = [point_data_info.GetArray(i).Name for i in range(point_data_info.GetNumberOfArrays())]
        required_vel = ['u', 'v', 'w']
        if not all(x in available_arrays for x in required_vel):
            print(f"ERROR: Missing one or more velocity components {required_vel}. Available: {available_arrays}")
            return False
        
$velomag_calc

$analysis_code
        print("Creating visualization...")
        view = GetActiveViewOrCreate('RenderView')
        view.ViewSize = $image_size
        display = Show(source, view)
        display.Representation = 'Surface'
        
        ColorBy(display, ('POINTS', '$data_array'))
        lut = GetColorTransferFunction('$data_array')
        lut.ApplyPreset('$color_map', True)
        display.SetScalarBarVisibility(view, True)
        
$camera_code
        view.CameraFocalPoint = [0, 0, 0]
        view.CameraParallelProjection = 1
        view.ResetCamera()
        view.StillRender()
        
        print(f"Saving to: {r'$output_abs'}")
        SaveScreenshot(r"$output_abs", view, ImageResolution=$image_size)
        if not os.path.exists(r"$output_abs") or os.path.getsize(r"$output_abs") < 1000:
            print("ERROR: Output file was not created or is too small.")
            return False
        
        print("Processing completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}"); import traceback; traceback.print_exc(); return False
if __name__ == "__main__":
    sys.exit(0 if main() else 1)
"""
end

function process_file(input_file::String, output_dir::String)
    output_file = generate_output_filename(input_file, output_dir)
    @info "Processing: $(basename(input_file)) -> $(basename(output_file))"
    script_content = generate_processing_script(input_file, output_file)
    temp_script_path = "temp_paraview_script.py"
    try
        open(temp_script_path, "w") do f; write(f, script_content); end
        pvpython_exe = PROCESSING_OPTIONS["paraview_executable"]
        pvpython_args = PROCESSING_OPTIONS["paraview_args"]
        cmd = `$pvpython_exe $pvpython_args $temp_script_path`
        output = Pipe()
        proc = run(pipeline(cmd, stdout=output, stderr=output), wait=false)
        close(output.in)
        output_task = @async read(output, String)
        start_time = time()
        while process_running(proc)
            if time() - start_time > PROCESSING_OPTIONS["timeout_seconds"]; @error "Processing timed out for $input_file."; kill(proc); return false; end
            sleep(1)
        end
        proc_output = fetch(output_task)
        if proc.exitcode != 0
            @error "ParaView script failed for $input_file with exit code $(proc.exitcode)."; @error "See full pvpython output below:\n$proc_output"; return false
        end
        @info "Successfully processed $input_file."
        return true
    catch e; @error "An error occurred while processing $input_file: $e"; return false
    finally; isfile(temp_script_path) && rm(temp_script_path); end
end

function main()
    args = parse_arguments()
    log_io = setup_logging(args["process-id"], args["log-level"])
    try
        output_dir = get(args, "output-dir", nothing) !== nothing ? args["output-dir"] : PROCESSING_OPTIONS["output_directory"]
        @info "Starting ParaView Batch Processor"; @info "Output directory: $output_dir"
        files_to_process = find_files(args)
        if isempty(files_to_process); @warn "No files found to process. Exiting."; return; end
        if args["dry-run"]
            println("\n--- DRY RUN ---"); for f in files_to_process; println("- $(basename(f)) -> $(basename(generate_output_filename(f, output_dir)))"); end; println("-----------------"); return
        end
        isdir(output_dir) || mkpath(output_dir)
        @info "Found $(length(files_to_process)) files to process."
        success_count = 0; error_count = 0
        for (i, file_path) in enumerate(files_to_process)
            @info "--- [File $i/$(length(files_to_process))] ---"
            if args["skip-existing"] && isfile(generate_output_filename(file_path, output_dir))
                @info "Output file exists, skipping."; continue
            end
            if process_file(file_path, output_dir); success_count += 1
            else; error_count += 1; if !PROCESSING_OPTIONS["continue_on_error"]; @error "Stopping due to error."; break; end; end
        end
        @info "=========================================="; @info "Batch processing finished."; @info "Successfully processed: $success_count"; @info "Errors: $error_count"; @info "=========================================="
    finally; close(log_io); end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
