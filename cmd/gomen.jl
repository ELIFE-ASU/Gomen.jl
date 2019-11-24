using Distributed

@everywhere const ARENA_EXT = ".arena.gz"
@everywhere const SERIES_EXT = ".series.gz"
@everywhere const INFERENCE_EXT = ".inference.gz"
@everywhere const ROC_EXT = ".roc.gz"

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere include("util.jl")
@everywhere include("simulations.jl")
@everywhere include("inference.jl")
@everywhere include("analysis.jl")

@everywhere harmonicmean(p::Float64, q::Float64) = if p != zero(p) && q != zero(q)
    2*p*q / (p + q)
else
    zero(p)*zero(q)
end

function gomen(games, graphs, schemes, rounds, replicates, methods, rescorers, datadir;
               forcesim=false, forceinf=false)
    @info "Saving data in \"$datadir\""

    @info "Running simulations..."
    @time simulate(graphs, schemes, games, rounds, replicates, datadir; force=forcesim)

    @info "Inferring networks..."
    @time infernetworks(methods, rescorers, datadir; force=forceinf)
end
