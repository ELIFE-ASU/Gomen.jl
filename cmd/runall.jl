using ArgParse, Dates, Distributed, JSON

getdatadir(outdir) = joinpath(outdir, Dates.format(now(), "Y-m-d"))

const s = ArgParseSettings(version = "1.0", add_version = true)

add_arg_group(s, "Input and Output")
@add_arg_table s begin
    "--datadir"
        help = "output directory"
        arg_type = String
        default = getdatadir("data")
end

add_arg_group(s, "Game Parameters")
@add_arg_table s begin
    "--gds"
        help = "S-parameter step size for Games generator"
        arg_type = Float64
        default = 0.5
    "--gdt"
        help = "T-parameter step size for Games generator"
        arg_type = Float64
        default = 0.5
end

add_arg_group(s, "Graph Parameters")
@add_arg_table s begin
    "--nodes"
        help = "number of nodes of in each graph"
        arg_type = Int
        nargs = '+'
        default = [10]
    "--nrand"
        help = "number of random networks to generate"
        arg_type = Int
        default = 1
    "--ps"
        help = "biases for Erdős-Rényi models"
        arg_type = Float64
        nargs = '+'
        default = [0.5]
    "--ks"
        help = "connectivity parameter for Barabási–Albert"
        arg_type = Int
        nargs = '+'
        default = [1]
end

add_arg_group(s, "Scheme Parameters")
@add_arg_table s begin
    "--betas"
    help = "β parameters for SigmoidRule-based Schemes"
    arg_type = Float64
    nargs = '+'
    default = [0.1, 1.0, 10.0]
end

add_arg_group(s, "Arena Parameters")
@add_arg_table s begin
    "--replicates"
        help = "number of times to play each game"
        arg_type = Int
        default = 10
    "--rounds"
        help = "number of rounds to play each game"
        arg_type = Int
        default = 10
end

add_arg_group(s, "Inference Parameters")
@add_arg_table s begin
    "--permutations"
        help = "number of permutations for significance methods"
        arg_type = Int
        default = 100
end

add_arg_group(s, "Phases")
@add_arg_table s begin
    "--force-simulation"
        help = "force the simulation phase to run"
        action = :store_true
    "--force-inference"
        help = "force the inference phase to run"
        action = :store_true
end

add_arg_group(s, "Worker Process Control")
@add_arg_table s begin
    "--procs"
        help = "number of worker processes"
        arg_type = Int
        default = 0
    "--slurm"
        help = "use the SLURM cluster manager"
        action = :store_true
end

args = parse_args(s)

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
    config = Dict(
        "gds" => args["gds"],
        "gdt" => args["gdt"],
        "nodes" => args["nodes"],
        "nrand" => args["nrand"],
        "ps" => args["ps"],
        "ks" => args["ks"],
        "betas" => args["betas"],
        "replicates" => args["replicates"],
        "rounds" => args["rounds"],
        "permutations" => args["permutations"]
    )
    JSON.print(io, config, 2)
end

gomen(games, graphs, schemes, rounds, replicates, methods, rescorers, datadir;
      forcesim = args["force-simulation"], forceinf = args["force-inference"])
