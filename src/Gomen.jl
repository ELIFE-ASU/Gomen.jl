module Gomen

export Game, play

using StaticArrays

"""
    Game(s, t)

A two-player, two-strategy, symmetric game. The games are characterized by the diagonal elements of
a payoff matrix

``\\begin{bmatrix} 1 & S \\\\ T & 0 \\end{bmatrix}``

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

end
