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
    NetworkInference

A supertype for all network inference types.
"""
abstract type NetworkInference end

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
    sort!(edges; by = e -> e.evidence, rev = true)
    edges
end

"""
    infer(::Type{I}, series[, rescorer]) where {I <: NetworkInference}

Infer edges in the network based on a given network inference method.
"""
function infer(::Type{I}, series::AbstractArray{Int, 2}) where {I <: NetworkInference}
    edgelist(score(I, series))
end

function infer(::Type{I}, series::AbstractArray{Int, 2},
               rescorer::AbstractRescorer) where {I <: NetworkInference}
    edgelist(rescore(rescorer, score(I, series)))
end
