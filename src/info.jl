const Series = AbstractVector{Int}

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

function mutualinfo(xs::Series, ys::Series; l=0)
    if isempty(xs)
        throw(ArgumentError("series is empty"))
    elseif length(xs) != length(ys)
        throw(ArgumentError("series must have the same length"))
    end

    N = length(xs)
    m1, m2, joint = zeros(2), zeros(2), zeros(2, 2)
    for (x, y) in zip((@view xs[l+1:end]), (@view ys[1:end-l]))
        m1[x] += 1
        m2[y] += 1
        joint[x, y] += 1
    end
    m1 ./= N
    m2 ./= N
    joint ./= N

    entropy(m1) + entropy(m2) - entropy(joint)
end

function netmutualinfo(xs::Series, ys::Series; l=0)
    if l == 0
        mutualinfo(xs, ys)
    else
        mutualinfo(xs, ys; l=l) - mutualinfo(ys, xs; l=1)
    end
end

function significance(measure::Function, xs::Series, ys::Series, args...; nperms=1000, kwargs...)
    gt = measure(xs, ys, args...; kwargs...)

    count = 0
    for _ in 1:nperms
        xsperm = @view xs[randperm(length(xs))]
        count += (measure(xsperm, ys, args...; kwargs...) ≥ gt)
    end
    p = count / (nperms + 1)
    se = sqrt((p * (1 - p)) / (nperms + 1))

    gt, p, se
end
