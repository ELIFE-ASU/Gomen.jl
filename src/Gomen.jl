module Gomen

export Game, sparam, tparam, play, Games, quadrant, name
export SimpleGraph, GraphGenerator, nv, params
export CycleGraphGenerator, WheelGraphGenerator, StarGraphGenerator, GridGraphGenerator
export BarabasiAlbertGenerator, ErdosRenyiGenerator
export AbstractRule, apply, Sigmoid, Heaviside
export AbstractScheme, CounterFactual, MimicRandom, MimicBest, MimicBiased, decide
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
export gomen

using Base.Iterators, Base.Meta, LinearAlgebra, Random, Statistics
using Distributions
using DrWatson
using HypothesisTests, Imogen
using LightGraphs, LightGraphs.SimpleGraphs
using MLBase
using RecipesBase
using Requires
using StaticArrays

include("core.jl")
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

function gomen(arena, scorer, significance, rescorer, rounds, replicates; kwargs...)
    rng = deepcopy(Random.default_rng())
    series = try
        play(arena; rounds, replicates)
    catch err
        throw(SimulationError(rng, arena, rounds, replicates, err))
    end

    scorerrng = deepcopy(Random.default_rng())
    scores = try
        if isnothing(rescorer)
            infer(significance ? SignificanceScorer(scorer) : scorer, series; kwargs...)
        else
            infer(significance ? SignificanceScorer(scorer) : scorer, rescorer, series; kwargs...)
        end
    catch err
        throw(ScoringError(scorerrng, arena, series, scorer, significance, rescorer, err))
    end

    roc, auc = try
        roc = Gomen.roc(arena, scores)
        roc, Gomen.auc(roc)
    catch err
        throw(PerformanceError(arena, series, scorer, significance, rescorer, scores, err))
    end

    @dict game rng scorerrng series scorer significance rescorer roc auc
end

function __init__()
    @require DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0" begin
        include("dataframes.jl")
    end
end

end
