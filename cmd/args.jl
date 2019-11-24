using ArgParse, Dates

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
        help = "S-parameter step size for Games generator; must be greater than 0.0"
        arg_type = Float64
        default = 0.5
        range_tester = ds -> ds > 0.0
    "--gdt"
        help = "T-parameter step size for Games generator; must be greater than 0.0"
        arg_type = Float64
        default = 0.5
        range_tester = dt -> dt > 0.0
end

add_arg_group(s, "Graph Parameters")
@add_arg_table s begin
    "--nodes"
        help = "number of nodes of in each graph; must be at least 10"
        arg_type = Int
        nargs = '+'
        default = [10]
        range_tester = n -> n >= 10
    "--nrand"
        help = "number of random networks to generate; must be at least 1"
        arg_type = Int
        default = 1
        range_tester = n -> n >= 1
    "--ps"
       help = "biases for Erdős-Rényi models; bias must be in (0, 1]"
        arg_type = Float64
        nargs = '+'
        default = [0.5]
        range_tester = p -> 0 < p <= 1
    "--ks"
        help = "connectivity parameter for Barabási–Albert; connectivity must be at least 1"
        arg_type = Int
        nargs = '+'
        default = [1]
        range_tester = k -> k >= 1
end

add_arg_group(s, "Scheme Parameters")
@add_arg_table s begin
    "--betas"
        help = "β parameters for SigmoidRule-based Schemes; must be greater than 0.0"
        arg_type = Float64
        nargs = '+'
        default = [0.1, 1.0, 10.0]
        range_tester = β -> β > 0.0
end

add_arg_group(s, "Arena Parameters")
@add_arg_table s begin
    "--replicates"
        help = "number of times to play each game; must be at least 1"
        arg_type = Int
        default = 10
        range_tester = r -> r >= 1
    "--rounds"
        help = "number of rounds to play each game; must be at least 1"
        arg_type = Int
        default = 10
        range_tester = r -> r >= 1
end

add_arg_group(s, "Inference Parameters")
@add_arg_table s begin
    "--permutations"
        help = "number of permutations for significance methods; must be at least 100"
        arg_type = Int
        default = 100
        range_tester = p -> p >= 100
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
        help = "number of worker processes; must be non-negative"
        arg_type = Int
        default = 0
        range_tester = p -> p >= 0
    "--slurm"
        help = "use the SLURM cluster manager"
        action = :store_true
end
