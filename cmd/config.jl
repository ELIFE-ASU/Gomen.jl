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

Base.print(io::IO, config::Config) = JSON.print(io, config, 2)
Base.print(config::Config) = io -> print(io, config)

configfile(datadir) = joinpath(datadir, "config.json")
tmpconfigfile(datadir) = configfile(datadir) * ".tmp"

JSON.parsefile(::Type{Config}, filename) = Config(JSON.parsefile(filename))