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
    MutualInfoInference

Infer based on the mutual information between pairs of nodes.
"""
abstract type MutualInfoInference <: NetworkInference end

function score(::Type{MutualInfoInference}, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = Array{Float64}(undef, N, N)
    @views for i in 1:N, j in i+1:N
        scores[i, j] = scores[j, i] = mutualinfo(series[:, i], series[:, j])
    end
    scores
end

"""
    LaggedMutualInfoInference

Infer based on the net lagged mutual information between pairs of nodes.
"""
abstract type LaggedMutualInfoInference <: NetworkInference end

function score(::Type{LaggedMutualInfoInference}, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = Array{Float64}(undef, N, N)
    @views for i in 1:N, j in i+1:N
        ij = mutualinfo(series[2:end, i], series[1:end-1, j])
        ji = mutualinfo(series[2:end, j], series[1:end-1, i])
        scores[i, j] = scores[j, i] = abs(ij - ji)
    end
    scores
end

"""
    infer(::Type{I}, series) where {I <: NetworkInference}

Infer edges in the network based on a given network inference method.
"""
function infer(::Type{I}, series::AbstractArray{Int, 2}) where {I <: NetworkInference}
    N = size(series, 2)
    scores = score(I, series)
    edges = Edge[]
    for i in 1:N, j in i+1:N
        push!(edges, Edge(i, j, scores[i, j]))
    end
    sort!(edges; by = e -> e.evidence, rev = true)
    edges
end
