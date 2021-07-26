"""
    AbstractRule

A supertype for all decision rules.
"""
abstract type AbstractRule end

"""
    apply(r::AbstractRule, dp)

Determine the probability an agent will switch its strategy based on the rule and the difference in
payoffs of the two strategies.
"""
apply(r::AbstractRule, dp) = error("method must be overloaded")

"""
    AbstractScheme

A supertype for all schemes, mechanisms for agents to choose their next strategy.
"""
abstract type AbstractScheme end

"""
    AbstractArena{Graph, Scheme}

A supertype for all game-playing arenas.
"""
abstract type AbstractArena{Graph <: SimpleGraph, Scheme <: AbstractScheme} end

"""
    game(arena)

The game the agents in the arena play
"""
game(a::AbstractArena) = a.game

"""
    graph(arena)

The graph on which the agents play their games
"""
graph(a::AbstractArena) = a.graph

"""
    scheme(arena)

The scheme the agents use to determine their next strategy
"""
scheme(a::AbstractArena) = a.scheme

"""
    length(arena)

The number of agents in the arena
"""
Base.length(a::AbstractArena) = nv(graph(a))

"""
    edges(arena)

The edges connecting agents in the arena
"""
LightGraphs.edges(a::AbstractArena) = edges(graph(a))

"""
    neighbors(arena, agent)

A vector of the agent's neighbors
"""
LightGraphs.neighbors(a::AbstractArena, i) = neighbors(graph(a), i)

"""
    payoffs(arena, strategies)

Determine the payoff of each agents' possible strategies given the strategies of all of the other
agents.
"""
function payoffs(a::AbstractArena, ss::AbstractVector{Int})
    if length(ss) != length(a)
        throw(ArgumentError("number of strategies should be the same as the number of agents"))
    elseif !all(1 .<= ss .<= 2)
        throw(ArgumentError("all strategies must be either 1 or 2"))
    end
    ps = zeros(Float64, 2, length(ss))
    for i in 1:length(ss), j in neighbors(a, i), s in 1:2
        ps[s, i] += play(game(a), s, ss[j])
    end
    ps
end

"""
    play(arena, strategies)

Play one round of the game in the arena, returning the agents' next strategies.
"""
play(a::AbstractArena, ss::AbstractVector{Int}) = decide(scheme(a), a, ss, payoffs(a, ss))

"""
    play(arena; rounds=1, replicates=1)

Starting from `replicates` many random initial strategies, play the game in the arena `rounds` many
times. The result is a 3-D array of the agents strategies with the first two dimensions of size
`rounds` and `replicates`, respectively.
"""
function play(a::AbstractArena; rounds::Int = 1, replicates::Int = 1)
    if rounds < 1
        throw(DomainError(rounds, "must be at least 1"))
    end
    if replicates < 1
        throw(DomainError(replicates, "must be at least 1"))
    end

    series = Array{Int}(undef, rounds, replicates, length(a))
    @views for i in 1:replicates
        series[1,i,:] = rand(1:2, length(a))
        for j in 2:rounds
            series[j,i,:] = play(a, series[j-1,i,:])
        end
    end
    series
end

"""

    decide(scheme, arena, ss::AbstractVector{Int}, ps::AbstractArray{Float64,2})

Use the scheme to decide what each agent's strategy will be in the next timestep. Here `ss` is a
vector of the agents' current strategies (one for each agent) and `ps` is a matrix of payoffs. Each
column of `ps` should correspond to an agent, and each row is a corresponding strategy.
"""
function decide(scheme::AbstractScheme,
                arena::AbstractArena,
                ss::AbstractVector{Int},
                ps::AbstractArray{Float64,2})
    if !all(1 .<= ss .<= 2)
        throw(ArgumentError("all strategies must be either 1 or 2"))
    end
    if length(ss) != size(ps, 2)
        throw(ArgumentError("length(ss) ≠ size(ps, 2)"))
    elseif size(ps, 1) != 2
        throw(ArgumentError("size(ps, 1) ≠ 2"))
    end
    dps = Vector{Float64}(undef, length(ss))
    for (i, s) in enumerate(ss)
        dps[i] = ps[3 - s, i] - ps[s, i]
    end
    decide(scheme, arena, ss, dps)
end

"""
    decide(scheme, arena, ss::AbstractVector{Int}, dps::AbstractVector{Float64})

Use the scheme to decide what each agent's strategy will be in the next time step. Here `ss` is a
vector of the agents' current strategies (one for each agent) and `dps` is a vector of the
difference between the payoff for the strategy the agent didn't play and the payoff they actually
received.


    decide(scheme, arena, s::Int, dp::Float64)

Use the scheme to decide what an agent's strategy will be in the next time step. Here `s` is the
agent's current strategy and `dp` is the difference between the payoff for the alternative strategy
and the actual payoff they received.
"""
function decide(scheme::AbstractScheme,
                arena::AbstractArena,
                ss::AbstractVector{Int},
                dps::AbstractVector{Float64})
    length(ss) != length(dps) && error("strategies and payoff differences must have the same length")
    map(i -> decide(scheme, arena, ss[i], dps[i]), 1:length(ss))
end

