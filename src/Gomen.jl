module Gomen

export Game, play
export AbstractRule, apply, Sigmoid, Heaviside
export barabasi_albert, erdos_renyi, wheel_graph, star_graph, lattice_graph
export AbstractScheme, CounterFactual, decide

using StaticArrays
using LightGraphs, LightGraphs.SimpleGraphs
using Random

@doc raw"""
    Game(s, t)

A two-player, two-strategy, symmetric game. The games are characterized by the diagonal elements of
a payoff matrix

```math
\begin{bmatrix}
    1 & S \\
    T & 0
\end{bmatrix}
```

with ``-0.5 \\leq S \\leq 0.5`` and ``0.5 \\leq T \\leq 1.5``,
"""
struct Game
    payoffs::SMatrix{2, 2, Float64}
    function Game(s::Number, t::Number)
        if -0.5 ≤ s ≤ 0.5
            if 0.5 ≤ t ≤ 1.5
                return new([1.0 s; t 0.0])
            end
            throw(DomainError(t, "expected 0.5 ≤ t ≤ 1.5"))
        end
        throw(DomainError(s, "expected -0.5 ≤ s ≤ 0.5"))
    end
end

"""
    play(g::Game, a, b)

Play the game with the strategies `a` and `b`, returning the payoff recieved by agent playing `a`.
"""
play(g::Game, a::Int, b::Int) = g.payoffs[a, b]

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

@doc raw"""
    Sigmoid(β)

A rule specifying that the agent will switch its strategy with probability

```math
P(dp) = \frac{1}{1 + e^{-\beta dp}}
```

where ``dp`` is the difference in the two payoffs.
"""
struct Sigmoid <: AbstractRule
    β::Float64
    function Sigmoid(β::Float64 = 1.0)
        if β < zero(β)
            throw(DomainError(β, "must be non-negative"))
        end
        new(β)
    end
end

apply(s::Sigmoid, dp::Float64) = 1.0 / (1.0 + exp(-s.β * dp))

@doc raw"""
    Heaviside(ϵ)

A rule specifying that the agent will switch its strategy with probability

```math
P(dp) = \begin{cases}
    0.0, & dp \leq -\epsilon \\
    0.5, & -\epsilon < dp < \epsilon \\
    1.0, & \epsilon < dp
\end{cases}
```

where ``dp`` is the difference in the two payoffs.
"""
struct Heaviside <: AbstractRule
    ϵ::Float64
    function Heaviside(ϵ::Float64 = 1e-3)
        if ϵ < zero(ϵ)
            throw(DomainError(ϵ, "must be non-negative"))
        end
        new(ϵ)
    end
end

apply(h::Heaviside, dp::Float64) = (abs(dp) < h.ϵ) ? 0.5 : float(dp > 0)

lattice_graph(m::Int, n::Int) = crosspath(m, path_graph(n))

"""
    AbstractScheme

A supertype for all schemes, mechanisms for agents to choose their next strategy.
"""
abstract type AbstractScheme end

"""
    decide(scheme, ss::AbstractVector{Int}, ps::AbstractArray{Float64,2})

Use the scheme to decide what each agent's strategy will be in the next timestep. Here `ss` is a
vector of the agents' current strategies (one for each agent) and `ps` is a matrix of payoffs. Each
column of `ps` should correspond to an agent, and each row is a corresponding strategy.
"""
function decide(scheme::AbstractScheme, ss::AbstractVector{Int}, ps::AbstractArray{Float64,2})
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
    decide(scheme, ss, dps)
end

"""
    decide(scheme, ss::AbstractVector{Int}, dps::AbstractVector{Float64})

Use the scheme to decide what each agent's strategy will be in the next time step. Here `ss` is a
vector of the agents' current strategies (one for each agent) and `dps` is a vector of the
difference between the payoff for the strategy the agent didn't play and the payoff they actually
received.


    decide(scheme, s::Int, dp::Float64)

Use the scheme to decide what an agent's strategy will be in the next time step. Here `s` is the
agent's current strategy and `dp` is the difference between the payoff for the alternative strategy
and the actual payoff they received. 
"""
function decide(scheme::AbstractScheme, ss::AbstractVector{Int}, dps::AbstractVector{Float64})
    map((s,dp) -> decide(scheme, s, dp), ss, dps)
end

@doc raw"""
    CounterFactural([rule = Sigmoid()])

The "counter-factual" update scheme. Under this scheme, the agent will switch their strategy
according to some probability. The probability the agent will switch is determined by applying the
scheme's rule to the payoff difference.
"""
struct CounterFactual{Rule <: AbstractRule} <: AbstractScheme
    rule::Rule
end
CounterFactual() = CounterFactual{Sigmoid}(Sigmoid())

decide(cf::CounterFactual, s::Int, dp::Float64) = (rand() <= apply(cf.rule, dp)) ? (3 - s) : s

end
