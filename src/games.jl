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

with ``-0.5 \leq S \leq 0.5`` and ``0.5 \leq T \leq 1.5``,
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
    sparam(game)

Get the S-parameter of the game.
"""
sparam(g::Game) = g.payoffs[1,2]

"""
    tparam(game)

Get the T-parameter of the game.
"""
tparam(g::Game) = g.payoffs[2,1]

"""
    quadrant(game)

Get the quadrant of S-T space in which the game resides.
"""
function quadrant(g::Game)
    S, T = sparam(g), tparam(g)
    if one(T) ≤ T
        zero(S) ≤ S ? 1 : 4
    else
        zero(S) ≤ S ? 2 : 3
    end
end

"""
    name(quad::Int)

Get the classical name of the games in a quadrant, e.g. Hawk-Dove, Harmony, Stag Hunt or Prisoner's
Dilemma.
"""
function name(quad::Int)
    if quad === 1
        "Hawk-Dove"
    elseif quad === 2
        "Harmony"
    elseif quad === 3
        "Stag Hunt"
    elseif quad === 4
        "Prisoner's Dilemma"
    else
        error("invalid quadrant: $q")
    end
end

"""
    name(game)

Get the classical name of the game, e.g. Hawk-Dove, Harmony, Stag Hunt or Prisoner's Dilemma.
"""
name(g::Game) = name(quadrant(g))

"""
    play(g::Game, a, b)

Play the game with the strategies `a` and `b`, returning the payoff recieved by agent playing `a`.
"""
play(g::Game, a::Int, b::Int) = g.payoffs[a, b]

Base.show(io::IO, g::Game) = print(io, "Game($(sparam(g)), $(tparam(g)))")

const GamesIterator = let ParameterRange = typeof(-0.5:0.1:0.5)
    Iterators.ProductIterator{Tuple{ParameterRange, ParameterRange}}
end

"""
    Games(sstep, tstep)

An iterator over the games with ``S`` and ``T`` parameters varied over their ranges with the
provided step sizes.
"""
struct Games
    iterator::GamesIterator
    Games(sstep, tstep) = let ss = -0.5:sstep:0.5, ts = 0.5:tstep:1.5
        new(product(ss, ts))
    end
end

Base.iterate(g::Games) = let iteree = iterate(g.iterator)
    if iteree === nothing
        iteree
    else
        element, state = iteree
        Game(element...), state
    end
end

Base.iterate(g::Games, state) = let iteree = iterate(g.iterator, state)
    if iteree === nothing
        iteree
    else
        element, state = iteree
        Game(element...), state
    end
end

Base.length(g::Games) = length(g.iterator)

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

lattice_graph(m::Int, n::Int) = if m < 1
    throw(DomainError(m, "lattice must have at least 1 row"))
elseif n < 1
    throw(DomainError(m, "lattice must have at least 1 column"))
else
    crosspath(m, path_graph(n))
end

"""
    RandomScheme(p=0.5)

The "random" update scheme. Under this scheme, the agent will randomly choose strategy 2 with
probability `p`. This is primarily for validation purposes.
"""
struct RandomScheme <: AbstractScheme
    p::Float64
end
RandomScheme() = RandomScheme(0.5)

function decide(r::RandomScheme,
                arena::AbstractArena{G, RandomScheme},
                ss::AbstractVector{Int},
                ps::AbstractArray{Float64,2}) where {G <: SimpleGraph}
    Int.(rand(length(ss)) .> r.p) .+ 1
end

function decide(::RandomScheme, ::AbstractArena, ::AbstractVector{Int}, ::AbstractVector{Float64})
    error("this method is ill-defined")
end
decide(::RandomScheme, ::AbstractArena, ::Int, ::Float64) = error("this method is ill-defined")

"""
    CounterFactural([rule = Sigmoid()])

The "counter-factual" update scheme. Under this scheme, the agent will switch their strategy
according to some probability. The probability the agent will switch is determined by applying the
scheme's rule to the payoff difference.
"""
struct CounterFactual{Rule <: AbstractRule} <: AbstractScheme
    rule::Rule
end
CounterFactual() = CounterFactual{Sigmoid}(Sigmoid())

function decide(cf::S,
                arena::AbstractArena{G, S},
                s::Int,
                dp::Float64) where {G, S <: CounterFactual}
    (rand() <= apply(cf.rule, dp)) ? (3 - s) : s
end

"""
    MimicRandom()

The "mimic random" update scheme. Under this scheme, the agent will switch their strategy at
uniformly at random to the strategy of one of their neighbors or their own.
"""
struct MimicRandom <: AbstractScheme end

function decide(m::MimicRandom,
                arena::AbstractArena{G, MimicRandom},
                ss::AbstractVector{Int},
                dps::AbstractArray{Float64,2}) where {G <: SimpleGraph}
    map(i -> ss[rand([i; neighbors(arena, i)])], 1:length(ss))
end

function decide(::MimicRandom, ::AbstractArena, ::AbstractVector{Int}, ::AbstractVector{Float64})
    error("this method is ill-defined")
end
decide(::MimicRandom, ::AbstractArena, ::Int, ::Float64) = error("this method is ill-defined")

"""
    MimicBest()

The "mimic best" update scheme. Under this scheme, the agent will switch their strategy to that
of their neighbors who recieved the greatest payout, possibly retaining their own.
"""
struct MimicBest <: AbstractScheme end

function decide(m::MimicBest,
                arena::AbstractArena{G, MimicBest},
                ss::AbstractVector{Int},
                ps::AbstractArray{Float64,2}) where {G <: SimpleGraph}
    map(1:length(ss)) do i
        ns = neighbors(arena, i)
        m, j = findmax(ps[ns])
        m > ps[i] ? ss[ns[j]] : ss[i]
    end
end

function decide(::MimicBest, ::AbstractArena, ::AbstractVector{Int}, ::AbstractVector{Float64})
    error("this method is ill-defined")
end
decide(::MimicBest, ::AbstractArena, ::Int, ::Float64) = error("this method is ill-defined")

"""
    MimicBiased()

The "mimic biased" update scheme. Under this scheme, the agent will switch their strategy
randomly to that of their neighbors with a log-probability proportional to their payoffs.
"""
struct MimicBiased <: AbstractScheme end

function decide(m::MimicBiased,
                arena::AbstractArena{G, MimicBiased},
                ss::AbstractVector{Int},
                ps::AbstractArray{Float64,2}) where {G <: SimpleGraph}
    map(1:length(ss)) do i
        idx = [i; neighbors(arena, i)]
        p = exp.(ps[idx])

        perm = sortperm(p)
        p = p[perm]
        idx = idx[perm]

        j = findfirst(cumsum(p) .>= sum(p)*rand())
        ss[idx[j]]
    end
end

function decide(::MimicBiased, ::AbstractArena, ::AbstractVector{Int}, ::AbstractVector{Float64})
    error("this method is ill-defined")
end
decide(::MimicBiased, ::AbstractArena, ::Int, ::Float64) = error("this method is ill-defined")

"""
    Arena(game, graph, scheme) <: AbstractArena

An arena in which the agents, situated on the vertices of a graph, play a game with their neighbors.
Every pair of agents plays the same game. With each round of play, the agents use the scheme to
decide what their next strategy will be.

In this arena, the agents accumulate payoff by playing with every neighbor. The decision of what
strategy to use in the next time step is based on the difference between the total payoffs for each
of the agents' strategies.
"""
struct Arena{Graph, Scheme} <: AbstractArena{Graph, Scheme}
    game::Game
    graph::Graph
    scheme::Scheme
    function Arena(ga::Game, gr::Graph, s::Scheme) where {Graph, Scheme}
        if nv(gr) < 2
            throw(DomainError(gr, "graph cannot be empty"))
        elseif has_self_loops(gr)
            throw(DomainError(gr, "graph cannot have self-loops"))
        elseif any(map(c -> length(c) < 2, connected_components(gr)))
            throw(DomainError(gr, "each component of the graph must have at least 2 elements"))
        end
        new{Graph, Scheme}(ga, gr, s)
    end
end
