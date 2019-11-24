using Distributed, JSON

include("config.jl")
include("args.jl")

args = parse_args(s)
config = Config(args)

mkpath(args["datadir"])
open(print(config), tmpconfigfile(args["datadir"]), "w")

if args["procs"] > 0
    if args["slurm"]
        error("SLURM is not yet supported")
    end
    addprocs(args["procs"])
end

include("gomen.jl")

const games = Games(config.gds, config.gdt)
const graphs = Dict(
    "cycle" => (cycle_graph(n) for n in config.nodes),
    "wheel" => (wheel_graph(n) for n in config.nodes),
    "star" => (star_graph(n) for n in config.nodes),
    "barabasi-albert" => (BarabasiAlbertGenerator(config.nrand, n, k)
                          for n in config.nodes
                          for k in cnfig.ks),
    "erdos-renyi" => (ErdosRenyiGenerator(config.nrand, n, p)
                      for n in config.nodes
                      for p in config.ps),
)

const rules = push!(AbstractRule[Sigmoid(β) for β in config.betas], Heaviside())

const schemes = CounterFactual.(rules)

const methods = Dict(
    "mutual info" => MIMethod(),
    "lagged mutual info" => LaggedMIMethod(),
    "significant mutual info" => SigMIMethod(config.permutations),
    "significant lagged mutual info" => SigLaggedMIMethod(config.permutations)
)

const rescorers = Dict(
    "clr rescorer" => CLRRescorer(),
    "Γ rescorer" => GammaRescorer(),
    "harmonic Γ rescorer" => GammaRescorer(harmonicmean)
)

mv(tmpconfigfile(args["datadir"]), configfile(args["datadir"]); force=true)

gomen(games, graphs, schemes, config.rounds, config.replicates, methods, rescorers, args["datadir"];
      forcesim = args["force-simulation"], forceinf = args["force-inference"])
