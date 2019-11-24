using ArgParse, Dates, Distributed

function getdatadir(outdir)
    version = 1
    datadir = joinpath(outdir, Dates.format(now(), "Y-m-d"))

    while ispath(datadir * "v$version")
        version += 1
    end
    datadir * "v$version"
end

const s = ArgParseSettings(version = "1.0", add_version = true)

@add_arg_table s begin
    "--datadir", "-o"
        help = "output directory"
        arg_type = String
        default = getdatadir("data")
end

@add_arg_table s begin
    "--replicates", "-n"
        help = "number of times to play each game"
        arg_type = Int
        default = 10
    "--rounds", "-r"
        help = "number of rounds to play each game"
        arg_type = Int
        default = 10
    "--permutations", "-p"
        help = "number of permutations for significance methods"
        arg_type = Int
        default = 100
end

@add_arg_table s begin
    "--nodes"
        help = "number of nodes of in each graph"
        arg_type = Int
        nargs = '+'
        default = [10]
end

@add_arg_table s begin
    "--gdt"
        help = "T-parameter step size for Games generator"
        arg_type = Float64
        default = 0.5
    "--gds"
        help = "S-parameter step size for Games generator"
        arg_type = Float64
        default = 0.5
end

@add_arg_table s begin
    "--nrand", "-N"
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

@add_arg_table s begin
    "--nprocs"
        help = "number of worker processes"
        arg_type = Int
        required = true
    "--slurm"
        help = "use the SLURM cluster manager"
        action = :store_true
end

args = parse_args(s)

if args["slurm"]
    error("SLURM is not yet supported")
end
addprocs(args["nprocs"])

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

gomen(games, graphs, schemes, rounds, replicates, methods, rescorers, args["datadir"])
