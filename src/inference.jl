"""
    Edge(src, dst, evidence)

An edge from src to dst with evidence that it exists.
"""
struct Edge
    src::Int
    dst::Int
    evidence::Float64
end

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
    edges = Edge[]
    for i in 1:N, j in i+1:N
        push!(edges, Edge(i, j, scores[i, j]))
    end
    edges
end

function infer(method::InferenceMethod, series::AbstractArray{Int, 2})
    edgelist(score(method, series))
end

function infer(method::InferenceMethod, series::AbstractArray{Int, 2}, rescorer::Rescorer)
    edgelist(rescore(rescorer, score(method, series)))
end

function infer(method::InferenceMethod, arena::AbstractArena, rounds::Int)
    infer(method, play(arena, rounds))
end

function infer(method::InferenceMethod, arena::AbstractArena, rounds::Int, rescorer::Rescorer)
    infer(method, play(arena, rounds), rescorer)
end
