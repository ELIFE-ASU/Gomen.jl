using Distributed, JSON, ClusterManagers

include("config.jl")
include("args.jl")

args = parse_args(s)
config = if args["config"] != nothing
    @info "Using simulation and inference parameters from configuration \"$(args["config"])\""
    parseconfig(args["config"])
else
    Config(args)
end

mkpath(args["datadir"])

if isfile(configfile(args["datadir"]))
    oldconfig = parseconfig(configfile(args["datadir"]))
    if config != oldconfig
        if forcesimulation(config, oldconfig) && !args["force-simulation"]
            @warn "Simulation parameters have changed; forcing simulation and inference phases..."
            args["force-simulation"] = true
            args["force-inference"] = true
        elseif forceinference(config, oldconfig) && !args["force-inference"]
            @warn "Inference parameters have changed; forcing inference phase..."
            args["force-inference"] = true
        end
    end
end
open(print(config), tmpconfigfile(args["datadir"]), "w")

if args["procs"] > 0
    if args["slurm"]
        addprocs(SlurmManager(args["procs"]))
    else
        addprocs(args["procs"])
    end
end

include("gomen.jl")

const games = Games(config.ds, config.dt)
const graphs = Dict(
    "cycle" => (cycle_graph(n) for n in config.nodes),
    "wheel" => (wheel_graph(n) for n in config.nodes),
    "star" => (star_graph(n) for n in config.nodes),
    "barabasi-albert" => (BarabasiAlbertGenerator(config.nrand, n, k)
                          for n in config.nodes
                          for k in config.ks),
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

gomen(games, graphs, schemes, config.rounds, config.replicates, methods, rescorers, args["datadir"];
      forcesim = args["force-simulation"], forceinf = args["force-inference"])

mv(tmpconfigfile(args["datadir"]), configfile(args["datadir"]); force=true)
