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

function mutualinfo(xs::Series, ys::Series)
    if length(xs) != length(ys)
        throw(ArgumentError("series must have the same length"))
    end

    N = length(xs)
    m1, m2, joint = zeros(2), zeros(2), zeros(2, 2)
    for (x, y) in zip(xs, ys)
        m1[x] += 1
        m2[y] += 1
        joint[x, y] += 1
    end
    m1 ./= N
    m2 ./= N
    joint ./= N

    entropy(m1) + entropy(m2) - entropy(joint)
end
