"""
    MIMethod

Infer based on the mutual information between pairs of nodes.
"""
struct MIMethod <: InferenceMethod end

"""
    score(method, series)

Create a score matrix from a time series using a basic inference method.
"""
function score(::MIMethod, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = Array{Float64}(undef, N, N)
    @views for i in 1:N, j in i+1:N
        scores[i, j] = scores[j, i] = mutualinfo(series[:, i], series[:, j])
    end
    scores
end

"""
    LaggedMIMethod

Infer based on the net lagged mutual information between pairs of nodes.
"""
struct LaggedMIMethod <: InferenceMethod end

function score(::LaggedMIMethod, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = Array{Float64}(undef, N, N)
    @views for i in 1:N, j in i+1:N
        scores[i, j] = scores[j, i] = abs(netmutualinfo(series[:, i], series[:, j]; l=1))
    end
    scores
end

"""
    SignificanceMethod

A supertype for all inference methods that consider statistical significance.
"""
abstract type SignificanceMethod <: InferenceMethod end

"""
    nperms(method::SignificanceMethod)

The number of permutations to use for significance testing
"""
nperms(method::SignificanceMethod) = method.nperms

"""
    pvalue(method::SignificanceMethod)

The pvalue below which a value is considered significant
"""
pvalue(method::SignificanceMethod) = method.pvalue

"""
    SigMIMethod(nperms, pvalue)

An inference method based on simple mutual information, considering the statistical signifiance.
Only edges determined to be significant (p < pvalue) have non-zero evidence.
"""
struct SigMIMethod <: SignificanceMethod
    nperms::Int
    pvalue::Float64
    function SigMIMethod(nperms, p)
        if !(zero(p) ≤ p ≤ one(p))
            throw(DomainError(p, "p-value must be in [0.0, 1.0]"))
        elseif nperms < 10
            throw(ArgumentError("number of permutations must be at least 10"))
        end
        new(nperms, p)
    end
end
SigMIMethod(p::Float64) = SigMIMethod(1000, p)
SigMIMethod(nperms::Int) = SigMIMethod(nperms, 0.05)
SigMIMethod() = SigMIMethod(1000, 0.05)

function score(method::SigMIMethod, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = zeros(Float64, N, N)
    @views for i in 1:N, j in i+1:N
        score, p, _ = significance(mutualinfo, series[:, i], series[:, j]; nperms=nperms(method))
        scores[i, j] = scores[j, i] = (p < pvalue(method)) ? zero(score) : score
    end
    scores
end

"""
    SigLaggedMIMethod(nperms, pvalue)

An inference method based on 1-lagged mutual information, considering the statistical signifiance.
Only edges determined to be significant (p < pvalue) have non-zero evidence.
"""
struct SigLaggedMIMethod <: SignificanceMethod
    nperms::Int
    pvalue::Float64
    function SigLaggedMIMethod(nperms, p)
        if !(zero(p) ≤ p ≤ one(p))
            throw(DomainError(p, "p-value must be in [0.0, 1.0]"))
        elseif nperms < 10
            throw(ArgumentError("number of permutations must be at least 10"))
        end
        new(nperms, p)
    end
end
SigLaggedMIMethod(p::Float64) = SigLaggedMIMethod(1000, p)
SigLaggedMIMethod(nperms::Int) = SigLaggedMIMethod(nperms, 0.05)
SigLaggedMIMethod() = SigLaggedMIMethod(1000, 0.05)

function score(method::SigLaggedMIMethod, series::AbstractArray{Int, 2})
    N = size(series, 2)
    scores = zeros(Float64, N, N)
    @views for i in 1:N, j in i+1:N
        score, p, _ = significance(netmutualinfo, series[:, i], series[:, j];
                                   nperms=nperms(method), l=1)
        scores[i, j] = scores[j, i] = (p < pvalue(method)) ? zero(score) : abs(score)
    end
    scores
end

