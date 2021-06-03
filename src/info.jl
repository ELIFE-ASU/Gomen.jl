using Imogen

const Series = Union{AbstractMatrix{Int}, AbstractVector{Int}}


laggedmutualinfo(xs::AbstractVector, ys::AbstractVector; l=0) = @views mutualinfo(xs[1:end-l], ys[l+1:end])
laggedmutualinfo(xs::AbstractMatrix, ys::AbstractMatrix; l=0) = @views mutualinfo(xs[:,1:end-l], ys[:,l+1:end])

function netmutualinfo(xs::Series, ys::Series; l::Int=0)
    if l == zero(l)
        mutualinfo(xs, ys)
    else
        laggedmutualinfo(xs, ys; l) - laggedmutualinfo(ys, xs; l)
    end
end

permute(xs::AbstractVector) = @view xs[randperm(length(xs))]

permute(xs::AbstractMatrix) = @view xs[:,randperm(size(xs,2))]

function significance(measure::Function, xs::Series, ys::Series, args...; nperms=1000, kwargs...)
    gt = measure(xs, ys, args...; kwargs...)

    count = 0
    for _ in 1:nperms
        xsperm = permute(xs)
        count += (measure(xsperm, ys, args...; kwargs...) ≥ gt)
    end
    p = count / (nperms + 1)
    se = sqrt((p * (1 - p)) / (nperms + 1))

    gt, p, se
end
