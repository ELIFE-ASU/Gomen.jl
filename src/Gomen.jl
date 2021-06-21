module Gomen

export Game, sparam, tparam, play, Games
export SimpleGraph, GraphGenerator, BarabasiAlbertGenerator, ErdosRenyiGenerator, param
export cycle_graph, wheel_graph, star_graph, lattice_graph, nv
export AbstractRule, apply, Sigmoid, Heaviside
export AbstractScheme, CounterFactual, decide
export AbstractArena, game, graph, scheme, Arena, payoffs
export Aggregator, Sum, HarmonicMean
export Scores, Rescorer, GammaRescorer, CLRRescorer
export EdgeEvidence, Scorer, infer
export SymmetricScorer
export ChisqScorer, LaggedChisqScorer, SymLaggedChisqScorer
export MIScorer, LaggedMIScorer, SymLaggedMIScorer
export TEScorer, SymTEScorer
export SignificanceScorer
export ROC, roc, tpr, fpr, auc
export SimulationError, ScoringError, PerformanceError

using Base.Iterators, Base.Meta, LinearAlgebra, Random, Statistics
using Distributions
using HypothesisTests, Imogen
using LightGraphs, LightGraphs.SimpleGraphs
using MLBase
using RecipesBase
using StaticArrays

include("games.jl")
include("graphs.jl")
include("inference.jl")
include("performance.jl")

struct SimulationError
    rng::AbstractRNG
    arena::Arena
    rounds::Int
    replicates::Int
    err
end

struct ScoringError
    rng::AbstractRNG
    arena::Arena
    series::Array{Int,3}
    scorer::Union{Nothing,Scorer}
    significance::Bool
    rescorer::Union{Nothing,Rescorer}
    err
end

struct PerformanceError
    arena::Arena
    series::Array{Int,3}
    scorer::Union{Nothing,Scorer}
    significance::Bool
    rescorer::Union{Nothing,Rescorer}
    scores::Array{Int,2}
    err
end

end
