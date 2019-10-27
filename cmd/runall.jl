using Distributed

addprocs(4)

@everywhere pool = default_worker_pool()

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere include("simulations.jl")

function getdatadir(outdir)
    version = 1
    datadir = joinpath(outdir, Dates.format(now(), "Y-m-d"))

    while ispath(datadir * "v$version")
        version += 1
    end
    datadir * "v$version"
end

function main()
    N = 10
    nodes = [10]
    ks = [1]
    ps = [0.5]

    graphs = Dict(
        "cycle" => (cycle_graph(n) for n in nodes),
        "wheel" => (wheel_graph(n) for n in nodes),
        "star" => (star_graph(n) for n in nodes),
        "barabasi-albert" => (BarabasiAlbertGenerator(N, n, k) for n in nodes for k in ks),
        "erdos-renyi" => (ErdosRenyiGenerator(N, n, p) for n in nodes for p in ps),
    )
    schemes = CounterFactual.([Sigmoid(), Sigmoid(0.1), Sigmoid(10.), Heaviside()])
    games = Games(0.1, 0.1)

    datadir = getdatadir("data")

    @info "Saving data in \"$datadir\""

    @info "Running simulations..."
    @time simulate(graphs, schemes, games, 10, 10, datadir)
end

main()
