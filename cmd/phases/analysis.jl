function analyze(arenapath, inferencepath, tmpdir, analysesdir)
    rocfile = joinpath(analysesdir, replace(basename(inferencepath), INFERENCE_EXT => ROC_EXT))

    mkpath(analysesdir)

    arena, _, _ = readarena(arenapath)
    inferences = readinferences(inferencepath)
    rocs = Dict{String, Any}[]
    for inference in inferences
        edges = inference["edges"]
        r = try
            roc(arena, edges)
        catch
            @show inference
            rethrow
        end
        push!(rocs, Dict("method" => inference["method"],
                         "rescorer" => inference["rescorer"],
                         "roc" => r))
    end

    tmp, io = mktempgz(tmpdir)
    try
        JSON.print(io, rocs, 2)
    finally
        close(io)
    end
    mv(tmp, rocfile)
end

function analyze(datadir; force=false)
    simsdir = joinpath(datadir, "sims")
    infdir = joinpath(datadir, "inference")
    datadir = joinpath(datadir, "analysis")
    if force
        rm(datadir; force=true, recursive=true)
    end
    if !ispath(datadir)
        tmpdir = joinpath(datadir, "tmp")
        mkpath(tmpdir)

        for (root, _, files) in walkdir(infdir)
            analysesdir = joinpath(datadir, relpath(root, infdir))
            futures = Future[]
            for file in files
                if isinference(file)
                    arenapath = joinpath(simsdir, relpath(root, infdir),
                                         replace(file, INFERENCE_EXT => ARENA_EXT))
                    inferencepath = joinpath(root, file)
                    push!(futures, @spawn analyze(arenapath, inferencepath, tmpdir, analysesdir))
                end
            end
            foreach(wait, futures)
        end
        rm(tmpdir; recursive=true)
    else
        @warn "The analysis directory \"$datadir\" already exists; skipping analysis..."
    end

    datadir
end
