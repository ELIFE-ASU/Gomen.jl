module Gomen

export Game, play
export AbstractRule, apply, Sigmoid, Heaviside

using StaticArrays

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

end
