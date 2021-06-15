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

JSON.lower(e::EdgeEvidence) = string(e)

restore(::Type{EdgeEvidence}, j::AbstractString) = eval(Meta.parse(j))

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
    dist = @sig S scorer(m)(args...; kwargs...) nperm=m.nperms parg=2
    dist.p < m.α ? dist.gt : zero(dist.gt)
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
    edgelist(scores)

Construct a weighted edge list from a scores matrix.
"""
function edgelist(scores::Scores)
    N = size(scores, 1)
    [EdgeEvidence(i, j, scores[i, j]) for i in 1:N for j in i+1:N]
end

function infer(scorer::Scorer, series::Union{AbstractMatrix{Int}, AbstractArray{Int, 3}})
    edgelist(score(scorer, series))
end

function infer(scorer::Scorer, series::Union{AbstractMatrix{Int}, AbstractArray{Int, 3}},
               rescorer::Rescorer)
    edgelist(rescore(rescorer, score(scorer, series)))
end

function infer(scorer::Scorer, arena::AbstractArena, rounds::Int)
    infer(scorer, play(arena, rounds))
end

function infer(scorer::Scorer, arena::AbstractArena, rounds::Int, replicates::Int)
    infer(scorer, play(arena, rounds, replicates))
end

function infer(scorer::Scorer, arena::AbstractArena, rounds::Int, rescorer::Rescorer)
    infer(scorer, play(arena, rounds), rescorer)
end

function infer(scorer::Scorer, arena::AbstractArena, rounds::Int, replicates::Int,
               rescorer::Rescorer)
    infer(scorer, play(arena, rounds, replicates), rescorer)
end
