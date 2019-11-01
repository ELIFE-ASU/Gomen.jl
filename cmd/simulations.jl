function simulate(graph::SimpleGraph, scheme::AbstractScheme, game, rounds, replicates,
                  tmpdir, outdir)
    s, t = sparam(game), tparam(game)

    arenafile = joinpath(outdir, "$(s)_$(t).arena.gz")
    seriesfile = joinpath(outdir, "$(s)_$(t).series.gz")

    arena = Arena(game, graph, scheme)
    series = play(arena, rounds, replicates)

    atmp, aio = mktempgz(tmpdir)
    stmp, sio = mktempgz(tmpdir)
    try
        JSON.print(aio, Dict("arena" => arena, "rounds" => rounds, "replicates" => replicates), 2)
        writedlm(sio, series)
    finally
        close(sio)
        close(aio)
    end
    mv(atmp, arenafile)
    mv(stmp, seriesfile)
end

function simulate(graph::SimpleGraph, schemes, games, rounds, replicates, tmpdir, outdir)
    graphhash = string(hash(graph))
    for scheme in schemes
        rule, param = Meta.parse(string(scheme.rule)).args
        schemedir = joinpath(outdir, lowercase(string(rule)), string(param), graphhash)
        mkpath(schemedir)
        for game in games
            simulate(graph, scheme, game, rounds, replicates, tmpdir, schemedir)
        end
    end
end

function simulate(gen::GraphGenerator, schemes, games, rounds, replicates, tmpdir, outdir)
    futures = Future[]
    for graph in gen
        gendir = joinpath(outdir, string(param(gen)))
        mkpath(gendir)
        f = @spawn simulate(graph, schemes, games, rounds, replicates, tmpdir, gendir)
        push!(futures, f)
    end
    foreach(wait, futures)
end

function simulate(graphs, schemes, games, rounds, replicates, datadir; force=false)
    datadir = joinpath(datadir, "sims")
    if force
        rm(datadir; force=true, recursive=true)
    end
    if !ispath(datadir)
        tmpdir = joinpath(datadir, "tmp")
        mkpath(tmpdir)

        futures = Future[]
        for (name, gen) in graphs
            for graph in gen
                graphdir = joinpath(datadir, name, string(nv(graph)))
                mkpath(graphdir)
                f = @spawn simulate(graph, schemes, games, rounds, replicates, tmpdir, graphdir)
                push!(futures, f)
            end
        end

        foreach(wait, futures)
        rm(tmpdir; recursive=true)
    else
        @warn "The simulation directory \"$datadir\" already exists; skipping simulations..."
    end
    datadir
end
