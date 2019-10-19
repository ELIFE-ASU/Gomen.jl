"""
    BasicInference

A supertype for all inference methods that do not consider statistical significance.
"""
abstract type BasicInference <: NetworkInference end

"""
    MutualInfoInference

Infer based on the mutual information between pairs of nodes.
"""
abstract type MutualInfoInference <: BasicInference end

"""
    score(::Type{I}, series) where {I <: BasicInference}

Create a score matrix from a time series using a basic inference method.
"""
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
abstract type LaggedMutualInfoInference <: BasicInference end

function score(::Type{LaggedMutualInfoInference}, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = Array{Float64}(undef, N, N)
    for i in 1:N, j in i+1:N
        scores[i, j] = scores[j, i] = abs(netmutualinfo(xs, ys; l=1))
    end
    scores
end

"""
    SigInference

A supertype for all inference methods that consider statistical significance.
"""
abstract type SigInference <: NetworkInference end

"""
    nperms(method::SigInference)

The number of permutations to use for significance testing
"""
nperms(method::SigInference) = method.nperms

"""
    pvalue(method::SigInference)

The pvalue below which a value is considered significant
"""
pvalue(method::SigInference) = method.pvalue

"""
    SigMutualInfoInference(nperms, pvalue)

An inference method based on simple mutual information, considering the statistical signifiance.
Only edges determined to be significant (p < pvalue) have non-zero evidence.
"""
struct SigMutualInfoInference <: SigInference
    nperms::Int
    pvalue::Float64
    function SigMutualInfoInference(nperms, p)
        if !(zero(p) ≤ p ≤ one(p))
            throw(DomainError(p, "p-value must be in [0.0, 1.0]"))
        elseif nperms < 10
            throw(ArgumentError("number of permutations must be at least 10"))
        end
        new(p, nperms)
    end
end
SigMutualInfoInference(p::Float64) = SigMutualInfoInference(1000, p)
SigMutualInfoInference(nperms::Int) = SigMutualInfoInference(nperms, 0.05)
SigMutualInfoInference() = SigMutualInfoInference(1000, 0.05)

"""
    score(method::I, series) where {I <: SigInference}

Create a score matrix from a time series using a significance inference method.
"""
function score(method::SigMutualInfoInference, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = zeros(Float64, N, N)
    @views for i in 1:N, j in i+1:N
        score, p, _ = significance(mutualinfo, series[:, i], series[:, j]; nperms=nperms(method))
        scores[i, j] = scores[j, i] = (p < pvalue(method)) ? zero(score) : score
    end
    scores
end

"""
    SigLaggedMutualInfoInference(nperms, pvalue)

An inference method based on 1-lagged mutual information, considering the statistical signifiance.
Only edges determined to be significant (p < pvalue) have non-zero evidence.
"""
struct SigLaggedMutualInfoInference <: SigInference
    nperms::Int
    pvalue::Float64
    function SigLaggedMutualInfoInference(nperms, p)
        if !(zero(p) ≤ p ≤ one(p))
            throw(DomainError(p, "p-value must be in [0.0, 1.0]"))
        elseif nperms < 10
            throw(ArgumentError("number of permutations must be at least 10"))
        end
        new(p, nperms)
    end
end
SigLaggedMutualInfoInference(p::Float64) = SigLaggedMutualInfoInference(1000, p)
SigLaggedMutualInfoInference(nperms::Int) = SigLaggedMutualInfoInference(nperms, 0.05)
SigLaggedMutualInfoInference() = SigLaggedMutualInfoInference(1000, 0.05)

function score(method::SigLaggedMutualInfoInference, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = zeros(Float64, N, N)
    @views for i in 1:N, j in i+1:N
        score, p, _ = significance(netmutualinfo, series[:, i], series[:, j];
                                   nperms=nperms(method), l=1)
        scores[i, j] = scores[j, i] = (p < pvalue(method)) ? zero(score) : score
    end
    scores
end

