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

const Scores = AbstractArray{Float64, 2}

abstract type AbstractRescorer end

struct CLRRescorer <: AbstractRescorer end

function rescore(::CLRRescorer, scores::Scores)
    N = size(scores, 1)
    rescore = zeros(Float64, N, N)
    for i in 1:N
        for j in i-1:N
            score = scores[i, j]
            iscores = vcat((@view scores[1:i-1, i]), (@view scores[i+1:end, i]))
            jscores = vcat((@view scores[1:j-1, j]), (@view scores[j+1:end, j]))

            ibar = score - mean(iscores)
            jbar = score - mean(jscores)
            iσ2 = var(iscore)
            jσ2 = var(jscore)

            irescaled = (iσ2 == zero(iσ2) || ibar < zero(ibar)) ? zero(score) : ibar^2 / iσ2
            jrescaled = (jσ2 == zero(jσ2) || jbar < zero(jbar)) ? zero(score) : jbar^2 / jσ2

            rescore[i, j] = rescore[j, i] = sqrt(irescored + jrescored)
        end
    end
    rescore
end

function edgeslist(scores::Scores)
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
    edge(rescore(rescorer, score(I, series)))
end
