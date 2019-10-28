"""
    Scores = AbstractArray{Float64, 2}

A type alias for score matrices.
"""
const Scores = AbstractArray{Float64, 2}

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
