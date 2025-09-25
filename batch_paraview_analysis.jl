#!/usr/bin/env julia
"""
        Parallel-Enabled ParaView Batch Processor (Julia Version)

        Computes slices, spatial averages, or Reynolds stresses from simulation data.
        """

using ArgParse
using Dates
using Logging
using Printf
using Base.Filesystem

#-------------------------------
# BATCH PROCESSING CONFIGURATION
#-------------------------------

const FILE_PATTERNS = Dict{String, Any}(
    "pattern_type" => "iteration",
    "base_directory" => "./",
    "file_template" => "iter_{}.pvtu",
)

const BATCH_CONFIG = Dict{String, Any}(
    "analysis_mode" => "reynolds_stress",
    "slicing" => Dict{String, Any}(
        "data_array" => "w",
        "axis" => "Z",
        "coordinate" => 100
    ),
    "reynolds_stress" => Dict{String, Any}(
        "component" => "uv",
        "averaging_axis" => "Y",
        "resolution" => [150, 150, 150],
    ),
    "visualization" => Dict{String, Any}(
        "image_size" => [1200, 800],
        "color_map" => "Viridis (matplotlib)"
    )
)

const PROCESSING_OPTIONS = Dict{String, Any}(
    "output_directory" => "./batch_output/",
    "continue_on_error" => true,
    "paraview_executable" => "pvpython",
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
        "--process-id"
            help = "Process ID for parallel runs"
            arg_type = Int
        "--dry-run"
            help = "Show what files would be processed"
            action = :store_true
        "--skip-existing"
            help = "Skip if output file exists"
            action = :store_true
    end
    return parse_args(ARGS, s)
end

#-------------------------------
# IMPLEMENTATION
#-------------------------------
function setup_logging(process_id = nothing, log_level = "INFO")
    pid_str = process_id !== nothing ? "_proc_$(lpad(process_id, 3, '0'))" : ""
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    log_file = "$(PROCESSING_OPTIONS["log_file_prefix"])$(pid_str)_$(timestamp).log"
    level = get(Dict("DEBUG"=>Logging.Debug, "INFO"=>Logging.Info), log_level, Logging.Info)
    io = open(log_file, "w")
    logger = SimpleLogger(io, level)
    global_logger(TeeLogger(ConsoleLogger(stdout, level), logger))
    @info "Logging to: $log_file"
    return io
end

struct TeeLogger <: AbstractLogger; loggers::Vector{AbstractLogger}; end
TeeLogger(loggers...) = TeeLogger(collect(loggers))
Logging.handle_message(logger::TeeLogger, args...; kwargs...) = for l in logger.loggers; Logging.handle_message(l, args...; kwargs...); end
Logging.shouldlog(logger::TeeLogger, args...) = any(Logging.shouldlog(l, args...) for l in logger.loggers)
Logging.min_enabled_level(logger::TeeLogger) = minimum(Logging.min_enabled_level(l) for l in logger.loggers)

function extract_number_from_filename(filepath::String)
    filename = basename(filepath)
    template = FILE_PATTERNS["file_template"]
    
    # 1. Try template match
    regex_template = replace(replace(template, r"[.^$*+?{}[\]\\|()\-]" => s"\\&"), "{}" => raw"([+-]?(?:\d+\.?\d*|\.\d+))")
    m = match(Regex(regex_template), filename)
    if m !== nothing
        try
            num_str = m.captures[1]
            try return parse(Int, num_str) catch; return parse(Float64, num_str) end
        catch
            # Fall through
        end
    end

    # 2. Fallback to any number
    m_fallback = match(r"([+-]?(?:\d+\.?\d*|\.\d+))", filename)
    if m_fallback !== nothing
        try
            num_str = m_fallback.captures[1]
            try return parse(Int, num_str) catch; return parse(Float64, num_str) end
        catch
            # Fall through
        end
    end

    @warn "Could not extract number from: $filename. Defaulting to 0."
    return 0
end

function find_files(args::Dict)
    base_dir = FILE_PATTERNS["base_directory"]
    template = FILE_PATTERNS["file_template"]
    if get(args, "files", nothing) !== nothing && !isempty(args["files"])
        return filter(isfile, abspath.(args["files"]))
    elseif get(args, "range", nothing) !== nothing
        start_val, end_val, step = args["range"]; files = String[]
        for i in start_val:step:end_val; f = joinpath(base_dir, replace(template, "{}"=>string(i))); isfile(f) && push!(files, abspath(f)); end; return files
    else
        glob_pattern = replace(template, "{}" => "*"); all_files = isdir(base_dir) ? readdir(base_dir) : []
        pattern_regex = Regex("^" * replace(replace(glob_pattern, r"[.^$*+?{}[\]\\|()\-]" => s"\\&"), "*" => ".*") * raw"$")
        matching_files = filter(f -> match(pattern_regex, f) !== nothing, all_files)
        files_with_numbers = [(extract_number_from_filename(f), abspath(joinpath(base_dir, f))) for f in matching_files]
        sort!(files_with_numbers, by = x -> x[1]); return [f[2] for f in files_with_numbers]
    end
end

function generate_output_filename(input_file::String, output_dir::String)
    file_number = extract_number_from_filename(input_file)
    number_str = isa(file_number, Int) ? @sprintf("%06d", file_number) : replace(@sprintf("%08.3f", file_number), "." => "_")
    mode = BATCH_CONFIG["analysis_mode"]; local filename
    if mode == "slicing"; config = BATCH_CONFIG["slicing"]; filename = "$(config["data_array"])_slice_$(config["axis"])_$(number_str).png"
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

    if mode == "slicing"
        config = BATCH_CONFIG["slicing"]
        data_array = config["data_array"]
        axis = config["axis"]
        coordinate = config["coordinate"]
        if data_array == "VELOMAG"; velomag_calc = "        source = Calculator(Input=reader, ResultArrayName='VELOMAG', Function='sqrt(u*u+v*v+w*w)')"; end
        analysis_code = """
        axis = '$axis'.upper()
        coordinate = $(coordinate === nothing ? "None" : coordinate)
        if coordinate is None:
            bounds = source.GetDataInformation().GetBounds()
            bounds_idx = {'X': [0, 1], 'Y': [2, 3], 'Z': [4, 5]}[axis]
            coordinate = (bounds[bounds_idx[0]] + bounds[bounds_idx[1]]) / 2.0
        slice_filter = Slice(Input=source, SliceType='Plane')
        if axis == 'X': slice_filter.SliceType.Origin = [coordinate, 0, 0]; slice_filter.SliceType.Normal = [1, 0, 0]
        elif axis == 'Y': slice_filter.SliceType.Origin = [0, coordinate, 0]; slice_filter.SliceType.Normal = [0, 1, 0]
        else: slice_filter.SliceType.Origin = [0, 0, coordinate]; slice_filter.SliceType.Normal = [0, 0, 1]
        source = slice_filter
        """
        camera_code = "if '$axis'.upper() == 'X': view.CameraPosition = [10,0,0]; view.CameraViewUp = [0,0,1]\nelif '$axis'.upper() == 'Y': view.CameraPosition = [0,10,0]; view.CameraViewUp = [0,0,1]\nelse: view.CameraPosition = [0,0,10]; view.CameraViewUp = [0,1,0]"

    elseif mode == "reynolds_stress"
        config = BATCH_CONFIG["reynolds_stress"]
        data_array = "RS_$(config["component"])_avg"
        axis = config["averaging_axis"]
        resolution = config["resolution"]
        analysis_code = """
        from vtkmodules.numpy_interface import dataset_adapter as dsa
        from vtk.util.numpy_support import numpy_to_vtk
        import numpy as np
        resampled_3d = ResampleToImage(Input=source, SamplingDimensions=$resolution)
        vtk_data_3d = servermanager.Fetch(resampled_3d)
        wrapped_data_3d = dsa.WrapDataObject(vtk_data_3d)
        dims = resampled_3d.SamplingDimensions
        u, v, w = (wrapped_data_3d.PointData[c].reshape(dims[2],dims[1],dims[0]) for c in ['u','v','w'])
        axis_map = {'X': 2, 'Y': 1, 'Z': 0}
        avg_axis_idx = axis_map.get('$axis'.upper())
        u_mean, v_mean, w_mean = (np.mean(c, axis=avg_axis_idx) for c in [u,v,w])
        if avg_axis_idx == 2: u_mean_3d,v_mean_3d,w_mean_3d = (np.tile(c[:,:,np.newaxis],(1,1,dims[0])) for c in [u_mean,v_mean,w_mean])
        elif avg_axis_idx == 1: u_mean_3d,v_mean_3d,w_mean_3d = (np.tile(c[:,np.newaxis,:],(1,dims[1],1)) for c in [u_mean,v_mean,w_mean])
        else: u_mean_3d,v_mean_3d,w_mean_3d = (np.tile(c[np.newaxis,:,:],(dims[2],1,1)) for c in [u_mean,v_mean,w_mean])
        u_p, v_p, w_p = (u-u_mean_3d, v-v_mean_3d, w-w_mean_3d)
        rs_3d = {"uu":u_p**2, "vv":v_p**2, "ww":w_p**2, "uv":u_p*v_p, "uw":u_p*w_p, "vw":v_p*w_p}['$(config["component"])']
        final_2d = np.mean(rs_3d, axis=avg_axis_idx)
        from vtk import vtkStructuredGrid, vtkPoints
        bounds = source.GetDataInformation().GetBounds()
        if avg_axis_idx == 2: grid_dims, b_idx = ([dims[1],dims[2]], [2,3,4,5])
        elif avg_axis_idx == 1: grid_dims, b_idx = ([dims[0],dims[2]], [0,1,4,5])
        else: grid_dims, b_idx = ([dims[0],dims[1]], [0,1,2,3])
        n1, n2 = grid_dims
        sgrid = vtkStructuredGrid(); sgrid.SetDimensions(n1,n2,1)
        pts = vtkPoints()
        c1 = np.linspace(bounds[b_idx[0]], bounds[b_idx[1]], n1)
        c2 = np.linspace(bounds[b_idx[2]], bounds[b_idx[3]], n2)
        avg_c = (bounds[avg_axis_idx*2] + bounds[avg_axis_idx*2+1])/2.0
        for j in range(n2):
            for i in range(n1):
                if avg_axis_idx==2: pts.InsertNextPoint(avg_c, c1[i], c2[j])
                elif avg_axis_idx==1: pts.InsertNextPoint(c1[i], avg_c, c2[j])
                else: pts.InsertNextPoint(c1[i], c2[j], avg_c)
        sgrid.SetPoints(pts)
        vtk_arr = numpy_to_vtk(final_2d.flatten('C'), deep=True); vtk_arr.SetName('$data_array')
        sgrid.GetPointData().SetScalars(vtk_arr)
        producer = TrivialProducer(registrationName='FinalData'); producer.GetClientSideObject().SetOutput(sgrid)
        source=producer
        """
        camera_code = "if '$axis'.upper() == 'X': view.CameraPosition = [10,0,0]; view.CameraViewUp = [0,0,1]\nelif '$axis'.upper() == 'Y': view.CameraPosition = [0,10,0]; view.CameraViewUp = [0,0,1]\nelse: view.CameraPosition = [0,0,10]; view.CameraViewUp = [0,1,0]"
    end

    return """#!/usr/bin/env python3
import os, sys
from paraview.simple import *
def main():
    try:
        reader = XMLPartitionedUnstructuredGridReader(FileName=[r'$input_abs'])
        source = reader
        point_data_info = reader.GetPointDataInformation()
        available_arrays = [point_data_info.GetArray(i).Name for i in range(point_data_info.GetNumberOfArrays())]
        if not all(x in available_arrays for x in ['u', 'v', 'w']):
            print(f"ERROR: Missing velocity components. Available: {available_arrays}"); return False
$velomag_calc
$analysis_code
        view = GetActiveViewOrCreate('RenderView'); view.ViewSize = $image_size
        display = Show(source, view)
        ColorBy(display, ('POINTS', '$data_array'))
        lut = GetColorTransferFunction('$data_array'); lut.ApplyPreset('$color_map', True)
        display.SetScalarBarVisibility(view, True)
$camera_code
        view.CameraFocalPoint = [0,0,0]; view.CameraParallelProjection = 1; view.ResetCamera()
        SaveScreenshot(r'$output_abs', view, ImageResolution=$image_size)
        if not os.path.exists(r'$output_abs') or os.path.getsize(r'$output_abs') < 1000:
            print("ERROR: Output file not created or too small."); return False
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}"); import traceback; traceback.print_exc(); return False
if __name__ == "__main__":
    sys.exit(0 if main() else 1)
"""
end

function process_file(input_file::String, output_dir::String, process_id::Union{Int, Nothing})
    output_file = generate_output_filename(input_file, output_dir)
    @info "Processing: $(basename(input_file)) -> $(basename(output_file))"
    pid_for_file = process_id !== nothing ? process_id : getpid()
    temp_script_path = "temp_paraview_script_$(pid_for_file).py"
    try
        script_content = generate_processing_script(input_file, output_file)
        open(temp_script_path, "w") do f; write(f, script_content); end
        cmd = `$(PROCESSING_OPTIONS["paraview_executable"]) $(PROCESSING_OPTIONS["paraview_args"]) $temp_script_path`
        output = Pipe()
        proc = run(pipeline(cmd, stdout=output, stderr=output), wait=false)
        close(output.in); output_task = @async read(output, String)
        start_time = time()
        while process_running(proc)
            if time() - start_time > PROCESSING_OPTIONS["timeout_seconds"]; @error "Timeout on $input_file."; kill(proc); return false; end
            sleep(1)
        end
        proc_output = fetch(output_task)
        if proc.exitcode != 0; @error "pvpython failed for $input_file.\nOutput:\n$proc_output"; return false; end
        @info "Successfully processed $input_file."
        return true
    catch e; @error "Error processing $input_file: $e"; return false
    finally; isfile(temp_script_path) && rm(temp_script_path); end
end

function main()
    args = parse_arguments()
    log_io = setup_logging(args["process-id"], "INFO")
    try
        output_dir = PROCESSING_OPTIONS["output_directory"]
        @info "Starting ParaView Batch Processor"; @info "Output directory: $output_dir"
        files_to_process = find_files(args)
        if isempty(files_to_process); @warn "No files found."; return; end
        if args["dry-run"]; println("\n--- DRY RUN ---"); for f in files_to_process; println("- $(basename(f)) -> $(basename(generate_output_filename(f, output_dir)))"); end; return; end
        isdir(output_dir) || mkpath(output_dir)
        @info "Found $(length(files_to_process)) files."
        success, errors = 0, 0
        for (i, file_path) in enumerate(files_to_process)
            @info "--- [File $i/$(length(files_to_process))] ---"
            if args["skip-existing"] && isfile(generate_output_filename(file_path, output_dir)); @info "Output exists, skipping."; continue; end
            if process_file(file_path, output_dir, args["process-id"]); success += 1; else; errors += 1; if !PROCESSING_OPTIONS["continue_on_error"]; @error "Stopping."; break; end; end
        end
        @info "================="; @info "Finished."; @info "Success: $success"; @info "Errors: $errors"
    finally; close(log_io); end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
