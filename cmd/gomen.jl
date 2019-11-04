using Distributed, Dates

@everywhere const ARENA_EXT = ".arena.gz"
@everywhere const SERIES_EXT = ".series.gz"
@everywhere const INFERENCE_EXT = ".inference.gz"

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere include("util.jl")
@everywhere include("simulations.jl")
@everywhere include("inference.jl")

function getdatadir(outdir)
    version = 1
    datadir = joinpath(outdir, Dates.format(now(), "Y-m-d"))

    while ispath(datadir * "v$version")
        version += 1
    end
    datadir * "v$version"
end

@everywhere harmonicmean(p::Float64, q::Float64) = if p != zero(p) && q != zero(q)
    2*p*q / (p + q)
else
    zero(p)*zero(q)
end

function gomen(; datadir=getdatadir("data"), forcesim=false, forceinf=false)
    N = 1
    nodes = [10]
    ks = [1]
    ps = [0.5]

    rounds = 10
    replicates = 10

    nperm = 100

    graphs = Dict(
        "cycle" => (cycle_graph(n) for n in nodes),
        "wheel" => (wheel_graph(n) for n in nodes),
        "star" => (star_graph(n) for n in nodes),
        "barabasi-albert" => (BarabasiAlbertGenerator(N, n, k) for n in nodes for k in ks),
        "erdos-renyi" => (ErdosRenyiGenerator(N, n, p) for n in nodes for p in ps),
    )
    schemes = CounterFactual.([Sigmoid(), Sigmoid(0.1), Sigmoid(10.), Heaviside()])
    games = Games(0.5, 0.5)

    methods = Dict(
        "mutual info" => MIMethod(),
        "lagged mutual info" => LaggedMIMethod(),
        "significant mutual info" => SigMIMethod(nperm),
        "significant lagged mutual info" => SigLaggedMIMethod(nperm)
    )

    rescorers = Dict(
        "clr rescorer" => CLRRescorer(),
        "Γ rescorer" => GammaRescorer(),
        "harmonic Γ rescorer" => GammaRescorer(harmonicmean)
    )

    @info "Saving data in \"$datadir\""

    @info "Running simulations..."
    @time simulate(graphs, schemes, games, rounds, replicates, datadir; force=forcesim)

    @info "Inferring networks..."
    @time infernetworks(methods, rescorers, datadir; force=forceinf)
end
