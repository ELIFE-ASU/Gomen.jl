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

