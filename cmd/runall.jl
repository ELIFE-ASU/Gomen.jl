using Distributed, JSON

args = include("args.jl")

struct Config
    gds::Float64
    gdt::Float64
    nodes::Vector{Int}
    nrand::Int
    ps::Vector{Float64}
    ks::Vector{Int}
    betas::Vector{Float64}
    replicates::Int
    rounds::Int
    permutations::Int

    function Config(args::Dict{String, Any})
        if isempty(args["nodes"])
            throw(ArgumentError("nodes array may not be empty"))
        elseif isempty(args["ps"])
            throw(ArgumentError("Erdős-Rényi bias array may not be empty"))
        elseif isempty(args["ks"])
            throw(ArgumentError("Barabási–Albert parameter array may not be empty"))
        elseif isempty(args["betas"])
            throw(ArgumentError("Sigmoid rule parameter array may not be empty"))
        end

        new(args["gds"], args["gdt"],
            args["nodes"], args["nrand"], args["ps"], args["ks"],
            args["betas"],
            args["replicates"], args["rounds"],
            args["permutations"]
           )
    end
end

config = Config(args)

if args["procs"] > 0
    if args["slurm"]
        error("SLURM is not yet supported")
    end
    addprocs(args["procs"])
end

include("gomen.jl")

const N = args["nrand"]
const nodes = args["nodes"]
const ks = args["ks"]
const ps = args["ps"]
const nperm = args["permutations"]
const rounds = args["rounds"]
const replicates = args["replicates"]

const games = Games(args["gds"], args["gdt"])
const graphs = Dict(
    "cycle" => (cycle_graph(n) for n in nodes),
    "wheel" => (wheel_graph(n) for n in nodes),
    "star" => (star_graph(n) for n in nodes),
    "barabasi-albert" => (BarabasiAlbertGenerator(N, n, k) for n in nodes for k in ks),
    "erdos-renyi" => (ErdosRenyiGenerator(N, n, p) for n in nodes for p in ps),
)

const rules = push!(AbstractRule[Sigmoid(β) for β in args["betas"]], Heaviside())

const schemes = CounterFactual.(rules)

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

const datadir = args["datadir"]

mkpath(datadir)
open(joinpath(datadir, "config.json"), "w") do io
    JSON.print(io, config, 2)
end

gomen(games, graphs, schemes, rounds, replicates, methods, rescorers, datadir;
      forcesim = args["force-simulation"], forceinf = args["force-inference"])
