function getdatadir(outdir)
    version = 1
    datadir = joinpath(outdir, Dates.format(now(), "Y-m-d"))

    while ispath(datadir * "v$version")
        version += 1
    end
    datadir * "v$version"
end

include("gomen.jl")

const N = 1
const nodes = [10]
const ks = [1]
const ps = [0.5]
const nperm = 100
const rounds = 10
const replicates = 10

const games = Games(0.5, 0.5)
const graphs = Dict(
    "cycle" => (cycle_graph(n) for n in nodes),
    "wheel" => (wheel_graph(n) for n in nodes),
    "star" => (star_graph(n) for n in nodes),
    "barabasi-albert" => (BarabasiAlbertGenerator(N, n, k) for n in nodes for k in ks),
    "erdos-renyi" => (ErdosRenyiGenerator(N, n, p) for n in nodes for p in ps),
)
const schemes = CounterFactual.([Sigmoid(), Sigmoid(0.1), Sigmoid(10.), Heaviside()])

const methods = Dict(
    "mutual info" => MIMethod(),
    "lagged mutual info" => LaggedMIMethod(),
    "significant mutual info" => SigMIMethod(nperm),
    "significant lagged mutual info" => SigLaggedMIMethod(nperm)
)

const rescorers = Dict(
    "clr rescorer" => CLRRescorer(),
    "Γ rescorer" => GammaRescorer(),
    "harmonic Γ rescorer" => GammaRescorer(harmonicmean)
)

const datadir = getdatadir("data")

gomen(games, graphs, schemes, rounds, replicates, methods, rescorers, datadir)
