function infernetworks(methods, rescorers, arenapath, arena, series, tmpdir, inferencedir)
    simname = replace(basename(arenapath), ARENA_EXT => "")
    inferencefile = joinpath(inferencedir, simname * INFERENCE_EXT)

    mkpath(inferencedir)

    inferences = Any[]
    for (method_name, method) in methods
        push!(inferences, Dict("method" => method_name,
                               "rescorer" => nothing,
                               "edges" => @spawn infer(method, series)))
        for (rescorer_name, rescorer) in rescorers
            push!(inferences, Dict("method" => method_name,
                                   "rescorer" => rescorer_name,
                                   "edges" => @spawn infer(method, series, rescorer)))
        end
    end

    for inference in inferences
        inference["edges"] = fetch(inference["edges"])
    end

    inftmp, iio = mktempgz(tmpdir)
    try
        JSON.print(iio, inferences, 2)
    finally
        close(iio)
    end
    mv(inftmp, inferencefile)
end

function infernetworks(methods, rescorers, datadir; force=false)
    simsdir = joinpath(datadir, "sims")
    datadir = joinpath(datadir, "inference")
    if force
        rm(datadir; force=true, recursive=true)
    end
    if !ispath(datadir)
        tmpdir = joinpath(datadir, "tmp")
        mkpath(tmpdir)

        for (root, _, files) in walkdir(simsdir)
            infdir = joinpath(datadir, relpath(root, simsdir))
            futures = Future[]
            for file in files
                if isarena(file)
                    arenapath = joinpath(root, file)
                    arena, series = readsim(arenapath)
                    push!(futures, @spawn infernetworks(methods, rescorers, arenapath, arena,
                                                        series, tmpdir, infdir))
                end
            end
            foreach(wait, futures)
        end
        rm(tmpdir; recursive=true)
    else
        @warn "The inference directory \"$datadir\" already exists; skipping inference..."
    end

    datadir
end
