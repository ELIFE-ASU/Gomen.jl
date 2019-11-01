"""
    EdgeEvidence(src, dst, evidence)

An edge from src to dst with evidence that it exists.
"""
struct EdgeEvidence
    src::Int
    dst::Int
    evidence::Float64
    EdgeEvidence(src, dst, evidence) = if isnan(evidence)
        throw(DomainError(evidence, "cannot be NaN"))
    else
        new(src, dst, evidence)
    end
end

JSON.lower(e::EdgeEvidence) = string(e)

restore(::Type{EdgeEvidence}, j::AbstractString) = eval(Meta.parse(j))

"""
    InferenceMethod

A supertype for all network inference types.
"""
abstract type InferenceMethod end

"""
    edgelist(scores)

Construct a weighted edge list from a scores matrix.
"""
function edgelist(scores::Scores)
    N = size(scores, 1)
    [EdgeEvidence(i, j, scores[i, j]) for i in 1:N for j in i+1:N]
end

function infer(method::InferenceMethod, series::Union{AbstractMatrix{Int}, AbstractArray{Int, 3}})
    edgelist(score(method, series))
end

function infer(method::InferenceMethod, series::Union{AbstractMatrix{Int}, AbstractArray{Int, 3}},
               rescorer::Rescorer)
    edgelist(rescore(rescorer, score(method, series)))
end

function infer(method::InferenceMethod, arena::AbstractArena, rounds::Int)
    infer(method, play(arena, rounds))
end

function infer(method::InferenceMethod, arena::AbstractArena, rounds::Int, replicates::Int)
    infer(method, play(arena, rounds, replicates))
end

function infer(method::InferenceMethod, arena::AbstractArena, rounds::Int, rescorer::Rescorer)
    infer(method, play(arena, rounds), rescorer)
end

function infer(method::InferenceMethod, arena::AbstractArena, rounds::Int, replicates::Int,
               rescorer::Rescorer)
    infer(method, play(arena, rounds, replicates), rescorer)
end
