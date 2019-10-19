module Gomen

export Game, Games, play
export AbstractRule, apply, Sigmoid, Heaviside
export barabasi_albert, erdos_renyi, wheel_graph, star_graph, lattice_graph
export AbstractScheme, CounterFactual, decide
export AbstractArena, game, graph, scheme, Arena, payoffs

using Base.Iterators
using StaticArrays
using LightGraphs, LightGraphs.SimpleGraphs
using Random

include("games.jl")

end
