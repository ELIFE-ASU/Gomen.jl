"""
    Scores = AbstractArray{Float64, 2}

A type alias for score matrices.
"""
const Scores = AbstractArray{Float64, 2}

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

"""
    Scorer

A supertype for all network inference types.
"""
abstract type Scorer end

function score(scorer::M, series::AbstractArray{Int,4}, args...; self=false, kwargs...) where {M <: Scorer}
    N = size(series, 4)
    scores = Array{Float64}(undef, N, N)
    @views for i in 1:N, j in 1:N
        scores[i,j] = scorer(series[:,:,:,i], series[:,:,:,j], args...; kwargs...)
    end
    !self && foreach(i -> scores[i,i] = zero(scores[i,i]), 1:N)
    scores
end

function score(scorer::M, series::AbstractArray{Int,3}, args...; kwargs...) where {M <: Scorer}
    score(scorer, reshape(series, 1, size(series)...), args...; kwargs...)
end

function score(scorer::M, series::AbstractArray{Int,2}, args...; self=false, kwargs...) where {M <: Scorer}
    N = size(series, 2)
    scores = Array{Float64}(undef, N, N)
    @views for i in 1:N, j in 1:N
        scores[i,j] = scorer(series[:,i], series[:,j], args...; kwargs...)
    end
    !self && foreach(i -> scores[i,i] = zero(scores[i,i]), 1:N)
    scores
end

"""
    SymmetricScorer
"""
abstract type SymmetricScorer <: Scorer end

function score(scorer::M, series::AbstractArray{Int,4}, args...; self=false, kwargs...) where {M <: SymmetricScorer}
    N = size(series, 4)
    scores = Array{Float64}(undef, N, N)
    @views for i in 1:N, j in i+1:N
        scores[i,j] = scores[j,i] = scorer(series[:,:,:,i], series[:,:,:,j], args...; kwargs...)
    end
    !self && foreach(i -> scores[i,i] = zero(scores[i,i]), 1:N)
    scores
end

function score(scorer::M, series::AbstractArray{Int,3}, args...; kwargs...) where {M <: SymmetricScorer}
    score(scorer, reshape(series, 1, size(series)...), args...; kwargs...)
end

function score(scorer::M, series::AbstractMatrix{Int}, args...; self=false, kwargs...) where {M <: SymmetricScorer}
    N = size(series, 2)
    scores = Array{Float64}(undef, N, N)
    @views for i in 1:N, j in i+1:N
        scores[i,j] = scores[j,i] = scorer(series[:,i], series[:,j], args...; kwargs...)
    end
    !self && foreach(i -> scores[i,i] = zero(scores[i,i]), 1:N)
    scores
end

"""
    ChisqScorer

Score based on the χ²-test statistic
"""
struct ChisqScorer <: SymmetricScorer end

function (::ChisqScorer)(xs::AbstractArray{Int,1}, ys::AbstractArray{Int,1}, args...; kwargs...)
    ChisqTest(xs, ys, 2).stat
end

function (::ChisqScorer)(xs::AbstractArray{Int}, ys::AbstractArray{Int}, args...; kwargs...)
    bs = tuple(fill(2, size(xs, 1))...)
    us = vec(mapslices(x -> Imogen.index(x, bs), xs; dims=1))
    vs = vec(mapslices(y -> Imogen.index(y, bs), ys; dims=1))
    ChisqTest(us, vs, max(maximum(us), maximum(vs))).stat
end

"""
    LaggedChisqScorer
"""
struct LaggedChisqScorer <: Scorer
    lag::Int
end
LaggedChisqScorer() = LaggedChisqScorer(1)

function (m::LaggedChisqScorer)(xs::AbstractArray{Int,1}, ys::AbstractArray{Int,1}, args...; kwargs...)
    scorer = ChisqScorer()
    if m.lag < 0
        @views scorer(xs[m.lag+1:end], ys[1:end-m.lag])
    else
        @views scorer(xs[1:end-m.lag], ys[m.lag+1:end])
    end
end

function (m::LaggedChisqScorer)(xs::AbstractArray{Int,2}, ys::AbstractArray{Int,2}, args...; kwargs...)
    scorer = ChisqScorer()
    if m.lag < 0
        @views scorer(xs[:,m.lag+1:end], ys[:,1:end-m.lag])
    else
        @views scorer(xs[:,1:end-m.lag], ys[:,m.lag+1:end])
    end
end

function (m::LaggedChisqScorer)(xs::AbstractArray{Int,3}, ys::AbstractArray{Int,3}, args...; kwargs...)
    scorer = ChisqScorer()
    if m.lag < 0
        @views scorer(xs[:,m.lag+1:end,:], ys[:,1:end-m.lag,:])
    else
        @views scorer(xs[:,1:end-m.lag,:], ys[:,m.lag+1:end,:])
    end
end

"""
    LaggedChisqScorer
"""
struct SymLaggedChisqScorer <: Scorer
    lag::Int
end
SymLaggedChisqScorer() = SymLaggedChisqScorer(1)

function (m::SymLaggedChisqScorer)(xs::AbstractArray{Int}, ys::AbstractArray{Int}, args...; kwargs...)
    scorer = SymLaggedMIScorer(m.lag)
    0.5 * (scorer(xs, ys) + scorer(ys, xs))
end

"""
    MIScorer

Infer based on the mutual information between pairs of nodes.
"""
struct MIScorer <: SymmetricScorer end

function (::MIScorer)(xs::AbstractArray{Int}, ys::AbstractArray{Int}, args...; kwargs...)
    mutualinfo(xs, ys, args...; kwargs...)
end

"""
    LaggedMIScorer

Infer based on the lagged mutual information between pairs of nodes.
"""
struct LaggedMIScorer <: Scorer
    lag::Int
end
LaggedMIScorer() = LaggedMIScorer(1)

function (m::LaggedMIScorer)(xs::AbstractArray{Int}, ys::AbstractArray{Int}, args...; kwargs...)
    mutualinfo(xs, ys, args...; lag=m.lag, kwargs...)
end

"""
    SymLaggedMIScorer

Infer based on the mean, lagged mutual information between pairs of nodes.
"""
struct SymLaggedMIScorer <: SymmetricScorer
    lag::Int
end
SymLaggedMIScorer() = SymLaggedMIScorer(1)

function (m::SymLaggedMIScorer)(xs::AbstractArray{Int}, ys::AbstractArray{Int}, args...; kwargs...)
    xy = mutualinfo(xs, ys, args...; lag=m.lag, kwargs...)
    yx = mutualinfo(ys, xs, args...; lag=m.lag, kwargs...)
    0.5 * (xy + yx)
end

"""
    TEScorer

Infer based on the transfer entropy between pairs of nodes.
"""
struct TEScorer <: Scorer
    k::Int
end
TEScorer() = TEScorer(1)

function (m::TEScorer)(xs::AbstractArray{Int}, ys::AbstractArray{Int}, args...; kwargs...)
    transferentropy(xs, ys, args...; k=m.k, kwargs...)
end

"""
    SymTEScorer

Infer based on the mean, transfer entropy between pairs of nodes.
"""
struct SymTEScorer <: SymmetricScorer
    k::Int
end
SymTEScorer() = SymTEScorer(1)

function (m::SymTEScorer)(xs::AbstractArray{Int}, ys::AbstractArray{Int}, args...; kwargs...)
    xy = transferentropy(xs, ys, args...; k=m.k, kwargs...)
    yx = transferentropy(ys, xs, args...; k=m.k, kwargs...)
    0.5 * (xy + yx)
end

struct SignificanceScorer{S<:Significance, T<:Scorer} <: Scorer
    scorer::T
    nperms::Int
    α::Float64
    function SignificanceScorer(::Type{S}, scorer::T, nperms=1000, α=0.05) where {S<:Significance, T<:Scorer}
        if !(zero(α) ≤ α ≤ one(α))
            throw(DomainError(α, "p-value must be in [0.0, 1.0]"))
        end
        new{S,T}(scorer, nperms, α)
    end
end
SignificanceScorer(scorer::Scorer, args...) = SignificanceScorer(AnalyticSig, scorer, args...)

scorer(m::SignificanceScorer) = (args...; kwargs...) -> m.scorer(args...; kwargs...)

function (m::SignificanceScorer{S})(args...; kwargs...) where S
    try
        dist = @sig S scorer(m)(args...; kwargs...) nperm=m.nperms
        dist.p < m.α ? dist.gt : zero(dist.gt)
    catch
        zero(Float64)
    end
end

"""
    Rescorer

A supertype for all rescoring methods.
"""
abstract type Rescorer end

"""
    CLRRescorer

A rescorer based on the mean and variance of all edges into and out of a pair of nodes.
"""
struct CLRRescorer <: Rescorer end

function rescore(::CLRRescorer, scores::Scores)
    N = size(scores, 1)
    rescore = zeros(Float64, N, N)
    for i in 1:N
        for j in i+1:N
            score = scores[i, j]
            iscores = vcat((@view scores[1:i-1, i]), (@view scores[i+1:end, i]))
            jscores = vcat((@view scores[1:j-1, j]), (@view scores[j+1:end, j]))

            ibar = score - mean(iscores)
            jbar = score - mean(jscores)
            iσ2 = var(iscores)
            jσ2 = var(jscores)

            irescored = (iσ2 == zero(iσ2) || ibar < zero(ibar)) ? zero(score) : ibar^2 / iσ2
            jrescored = (jσ2 == zero(jσ2) || jbar < zero(jbar)) ? zero(score) : jbar^2 / jσ2

            rescore[i, j] = rescore[j, i] = sqrt(irescored + jrescored)
        end
    end
    rescore
end

"""
    GammaRescorer(ϵ=eps(Float64), aggregator=(p,q) -> p + q)

Rescore edges by fitting a Gamma distribution to the incoming and outgoing scores and aggregating
the probability of the score given those distributions.

The scores provided to the Gamma distribution cannot be 0, so all 0-scores are set to ϵ.
"""
struct GammaRescorer <: Rescorer
    ϵ::Float64
    aggregator::Function
end
GammaRescorer() = GammaRescorer(eps(Float64), (p, q) -> p + q)
GammaRescorer(agg::Function) = GammaRescorer(eps(Float64), agg)
GammaRescorer(ϵ::Float64) = GammaRescorer(ϵ, (p, q) -> p + q)

function rescore(gr::GammaRescorer, scores::Scores)
    try
        N = size(scores, 1)
        rescore = zeros(Float64, N, N)
        for i in 1:N
            for j in i+1:N
                score = scores[i, j]
                iscores = vcat((@view scores[1:i-1, i]), (@view scores[i+1:end, i]))
                jscores = vcat((@view scores[1:j-1, j]), (@view scores[j+1:end, j]))

                iscores[iscores .== zero(Float64)] .= gr.ϵ
                jscores[jscores .== zero(Float64)] .= gr.ϵ

                p = cdf(fit(Gamma, iscores), score)
                q = cdf(fit(Gamma, jscores), score)

                rescore[i, j] = rescore[j, i] = gr.aggregator(p, q)
            end
        end
        rescore
    catch
        rescore(CLRRescorer(), scores)
    end
end

"""
    edgelist(scores[; symmetric=false, self=false])

Construct a weighted edge list from a scores matrix.
"""
function evidence(scores::Scores; symmetric=false, self=false)
    N = size(scores, 1)

    evidence = if symmetric
        [EdgeEvidence(i, j, scores[i, j]) for i in 1:N for j in i+1:N]
    else
        [EdgeEvidence(i, j, scores[i, j]) for i in 1:N for j in 1:N]
    end

    !self && filter(e -> e.src != e.dst, evidence)

    evidence
end

function infer(scorer::Scorer, series; kwargs...)
    evidence(score(scorer, series); kwargs...)
end

function infer(scorer::SymmetricScorer, series; kwargs...)
    evidence(score(scorer, series); symmetric=true, kwargs...)
end

function infer(scorer::SignificanceScorer{AnalyticSig,<:SymmetricScorer}, series; kwargs...)
    evidence(score(scorer, series); symmetric=true, kwargs...)
end

function infer(scorer::Scorer, rescorer::Rescorer, series; kwargs...)
    evidence(rescore(rescorer, score(scorer, series)); kwargs...)
end

function infer(scorer::SymmetricScorer, rescorer::Rescorer, series; kwargs...)
    evidence(rescore(rescorer, score(scorer, series)); symmetric=true, kwargs...)
end

function infer(scorer::SignificanceScorer{AnalyticSig,<:SymmetricScorer}, rescorer::Rescorer, series; kwargs...)
    evidence(rescore(rescorer, score(scorer, series)); symmetric=true, kwargs...)
end
