using Gomen, GZip, JSON, DelimitedFiles

isarena(path) = endswith(path, ARENA_EXT)

isseries(path) = endswith(path, SERIES_EXT)

isinference(path) = endswith(path, INFERENCE_EXT)

isroc(path) = endswith(path, ROC_EXT)

function mktempgz(parent=tempdir())
    filename, io = mktemp(parent)
    close(io)
    filename, GZip.open(filename, "w")
end

function readarena(arenapath)
    try
        io = GZip.open(arenapath, "r")
        d = JSON.parse(io)
        close(io)

        restore(AbstractArena, d["arena"]), d["rounds"], d["replicates"]
    catch
        @error "An error occurred while reading \"$arenapath\""
        rethrow
    end
end

function readseries(seriespath, rounds, replicates)
    try
        io = GZip.open(seriespath, "r")
        series = readdlm(io, Int)
        close(io)

        n = length(series) ÷ (rounds * replicates)

        reshape(series, replicates, rounds, n)
    catch
        @error "An error occurred while reading \"$seriespath\""
        rethrow
    end
end

function readsim(arenapath)
    simname = replace(arenapath, ARENA_EXT => "")
    seriespath = simname * SERIES_EXT

    arena, rounds, replicates = readarena(arenapath)
    series = readseries(seriespath, rounds, replicates)

    arena, series
end

function readinferences(inferencepath)
    inferences = try
        io = GZip.open(inferencepath, "r")
        inferences = JSON.parse(io)
        close(io)

        inferences
    catch
        @error "An error occurred while reading \"$inferencepath\""
        rethrow
    end
    foreach(inferences) do inference
        inference["edges"] = map(e -> restore(EdgeEvidence, e), inference["edges"])
    end
    inferences
end
