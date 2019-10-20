const Series = Union{AbstractMatrix{Int}, AbstractVector{Int}}

const Dist = AbstractArray{Float64}

function entropy(dist::Dist)
    h = 0.0
    for p in dist
        if !iszero(p)
            h -= p * log2(p)
        end
    end
    h
end

mutable struct MIDist
    joint::Matrix{Int}
    m1::Vector{Int}
    m2::Vector{Int}
    N::Int
    MIDist() = new(zeros(Int, 2, 2), zeros(Int, 2), zeros(Int, 2), 0)
end

MIDist(xs::AbstractVector{Int}, ys::AbstractVector{Int}) = accumulate!(MIDist(), xs, ys)

entropy(dist::MIDist) = if dist.N == zero(dist.N)
    throw(ArgumentError("distribution has no recorded observations"))
else
    entropy(dist.m1 / dist.N) + entropy(dist.m2 / dist.N) - entropy(dist.joint / dist.N)
end

function accumulate!(dist::MIDist, xs::AbstractVector{Int}, ys::AbstractVector{Int})
    if size(xs) != size(ys)
        throw(ArgumentError("series must have the same size"))
    end

    dist.N += length(xs)
    for (x, y) in zip(xs, ys)
        dist.m1[x] += 1
        dist.m2[y] += 1
        dist.joint[x, y] += 1
    end

    dist
end

function mutualinfo(xs::AbstractVector{Int}, ys::AbstractVector{Int}; l::Int=0)
    if isempty(xs)
        throw(ArgumentError("series is empty"))
    end

    @views entropy(MIDist(xs[1:end-l], ys[l+1:end]))
end

function mutualinfo(xs::AbstractMatrix{Int}, ys::AbstractMatrix{Int}; l::Int=0)
    if isempty(xs)
        throw(ArgumentError("series is empty"))
    end

    dist = MIDist()
    @views for i in 1:size(xs, 1)
        accumulate!(dist, xs[1:end-l], ys[l+1:end])
    end
    entropy(dist)
end

function netmutualinfo(xs::Series, ys::Series; l::Int=0)
    if l == zero(l)
        mutualinfo(xs, ys)
    else
        mutualinfo(xs, ys; l=l) - mutualinfo(ys, xs; l=l)
    end
end

permute(xs::AbstractVector) = @view xs[randperm(length(xs))]

function permute(xs::AbstractMatrix)
    r = xs[:,:]
    @views for i in size(r, 1)
        shuffle!(r[i, :])
    end
    r
end

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
