using Printf

abstract type PerformanceCurve end

xmetric(::Type{<: PerformanceCurve}) = error("unimplemented")
xmetric(::T) where {T <: PerformanceCurve} = xmetric(T)

ymetric(::Type{<: PerformanceCurve}) = error("unimplemented")
ymetric(::T) where {T <: PerformanceCurve} = ymetric(T)

xlabel(::Type{<: PerformanceCurve}) = error("unimplemented")
xlabel(::T) where {T <: PerformanceCurve} = xlabel(T)

ylabel(::Type{<: PerformanceCurve}) = error("unimplemented")
ylabel(::T) where {T <: PerformanceCurve} = ylabel(T)

xvalues(::PerformanceCurve) = error("unimplemented")
yvalues(::PerformanceCurve) = error("unimplemented")

startpoint(::Type{<: PerformanceCurve}, xs, ys) = error("unimplemented")
startpoint(::T, args...) where {T <: PerformanceCurve} = startpoint(T, args...)

endpoint(::Type{<: PerformanceCurve}, xs, ys) = error("unimplemented")
endpoint(::T, args...) where {T <: PerformanceCurve} = endpoint(T, args...)

sortage!(::Type{<: PerformanceCurve}, xs, ys) = error("unimplemented")

function downsample(xs::AbstractVector{Float64}, ys::AbstractVector{Float64})
    keep = [1]
    for i in 2:length(xs)
        if !(xs[i] ≈ xs[i-1] && ys[i] ≈ ys[i-1])
            push!(keep, i)
        end
    end
    keep
end

function prepare!(::Type{T},
                 xs::AbstractVector{Float64},
                 ys::AbstractVector{Float64};
                 down=true) where {T <: PerformanceCurve}

    if size(xs) != size(ys)
        throw(DimensionMismatch("$(xlabel(T)) and $(ylabel(T)) vectors have different sizes"))
    elseif any(<(0.0), xs)
        throw(ArgumentError("$(xlabel(T)) less than 0.0 provided"))
    elseif any(<(0.0), ys)
        throw(ArgumentError("$(ylabel(T)) less than 0.0 provided"))
    elseif any(>(1.0), xs)
        throw(ArgumentError("$(xlabel(T)) greater than 1.0 provided"))
    elseif any(>(1.0), ys)
        throw(ArgumentError("$(ylabel(T)) greater than 1.0 provided"))
    end

    sortage!(T, xs, ys)

    if down
        d = downsample(xs, ys)
        xs = xs[d]
        ys = ys[d]
    end

    s = startpoint(T, xs, ys)
    if !all((xs[begin], ys[begin]) .≈ s)
        pushfirst!(xs, s[begin])
        pushfirst!(ys, s[end])
    end

    e = endpoint(T, xs, ys)
    if !all((xs[end], ys[end]) .≈ e)
        push!(xs, e[begin])
        push!(ys, e[end])
    end

    xs, ys
end

Base.length(c::PerformanceCurve) = length(xvalues(c))

function MLBase.roc(T::Type{<:PerformanceCurve},
                    arena::AbstractArena,
                    edges::Vector{EdgeEvidence},
                    args...)
    g = graph(arena)
    gt = [Int(has_edge(g, e.src, e.dst)) for e in edges]
    scores = [e.evidence for e in edges]
    if all(==(scores[1]), scores)
        T()
    else
        x, y = xmetric(T), ymetric(T)
        r = roc(gt, scores, args...)
        T(map(x, r), map(y, r))
    end
end

auc(c::PerformanceCurve) = let xs = xvalues(c), ys = yvalues(c)
    @views dot(ys[2:end] + ys[1:end-1], xs[2:end] - xs[1:end-1]) / 2.0
end

struct Curve
    xs::Vector{Float64}
    ys::Vector{Float64}
    function Curve(xs::AbstractVector{Float64}, ys::AbstractVector{Float64})
        if length(xs) != length(ys)
            throw(DimensionMismatch("the x- and y-position vectors must have the same length"))
        end
        new(xs, ys)
    end
end
Curve(r::PerformanceCurve) = Curve(xvalues(r), yvalues(r))

Base.length(c::Curve) = length(c.xs)

function Base.:+(a::Curve, b::Curve)
    xs, ys = Float64[], Float64[]
    i, j = 1, 1
    @views while i <= length(a) && j <= length(b)
        if a.xs[i] ≈ b.xs[j]
            push!(xs, a.xs[i])
            push!(ys, a.ys[i] + b.ys[j])
            i += 1
            j += 1
        elseif a.xs[i] < b.xs[j]
            push!(xs, a.xs[i])
            push!(ys, a.ys[i] + interpolate(b.xs[j-1:j], b.ys[j-1:j], a.xs[i]))
            i += 1
        else
            push!(xs, b.xs[j])
            push!(ys, b.ys[j] + interpolate(a.xs[i-1:i], a.ys[i-1:i], b.xs[j]))
            j += 1
        end
    end
    Curve(xs, ys)
end

function Base.:-(a::Curve, b::Curve)
    xs, ys = Float64[], Float64[]
    i, j = 1, 1
    @views while i <= length(a) && j <= length(b)
        if a.xs[i] ≈ b.xs[j]
            push!(xs, a.xs[i])
            push!(ys, a.ys[i] - b.ys[j])
            i += 1
            j += 1
        elseif a.xs[i] < b.xs[j]
            push!(xs, a.xs[i])
            push!(ys, a.ys[i] - interpolate(b.xs[j-1:j], b.ys[j-1:j], a.xs[i]))
            i += 1
        else
            push!(xs, b.xs[j])
            push!(ys, b.ys[j] - interpolate(a.xs[i-1:i], a.ys[i-1:i], b.xs[j]))
            j += 1
        end
    end
    Curve(xs, ys)
end

function interpolate(xs::AbstractVector{Float64}, ys::AbstractVector{Float64}, x::Float64)
    m = (ys[2] - ys[1]) / (xs[2] - xs[1])
    m * (x - xs[1]) + ys[1]
end

Base.:/(c::Curve, k::Number) = Curve(c.xs, c.ys / k)

Statistics.mean(cs::AbstractVector{T}) where {T <: PerformanceCurve} = T(mean(Curve.(cs)))

struct ROC <: PerformanceCurve
    fpr::Vector{Float64}
    tpr::Vector{Float64}
    function ROC(fpr::Vector{Float64}, tpr::Vector{Float64}; kwargs...)
        fpr, tpr = prepare!(ROC, deepcopy(fpr), deepcopy(tpr); kwargs...)
        new(fpr, tpr)
    end
end
ROC() = ROC([0.0, 1.0], [0.0, 1.0])
ROC(c::Curve) = ROC(c.xs, c.ys)

fpr(r::ROC) = r.fpr

tpr(r::ROC) = r.tpr

xmetric(::Type{ROC}) = false_positive_rate

ymetric(::Type{ROC}) = true_positive_rate

xlabel(::Type{ROC}) = :FPR

ylabel(::Type{ROC}) = :TPR

startpoint(::Type{ROC}, fpr, tpr) = (0.0, 0.0)

endpoint(::Type{ROC}, fpr, tpr) = (1.0, 1.0)

xvalues(r::ROC) = fpr(r)

yvalues(r::ROC) = tpr(r)

function sortage!(::Type{ROC}, fpr, tpr)
    if !(issorted(fpr) && issorted(tpr))
        p = sortperm(collect(zip(fpr, tpr)))
        fpr[:] = fpr[p]
        tpr[:] = tpr[p]
    end

    if !issorted(tpr)
        throw(ArgumentError("the $T curve is not monotonic"))
    end

    fpr, tpr
end

struct PRC <: PerformanceCurve
    recall::Vector{Float64}
    precision::Vector{Float64}
    function PRC(recall::Vector{Float64}, precision::Vector{Float64}; kwargs...)
        recall, precision = prepare!(PRC, deepcopy(recall), deepcopy(precision); kwargs...)
        new(recall, precision)
    end
end
PRC() = new([0.0, 0.0, 1.0, 1.0], [1.0, 0.5, 0.5, 0.0])
PRC(c::Curve) = PRC(c.xs, c.ys)

recall(r::PRC) = r.recall

precision(r::PRC) = r.precision

xmetric(::Type{PRC}) = MLBase.recall

ymetric(::Type{PRC}) = MLBase.precision

xlabel(::Type{PRC}) = :Recall

ylabel(::Type{PRC}) = :Precision

startpoint(::Type{PRC}, recall, precision) = (0.0, 1.0)

endpoint(::Type{PRC}, recall, precision) = (1.0, 0.0)

xvalues(r::PRC) = recall(r)

yvalues(r::PRC) = precision(r)

function prcorder((r1, p1), (r2, p2))
    isequal(r1, r2) && return isless(p2, p1)
    isless(r1, r2)
end

function sortage!(::Type{PRC}, recall, precision)
    p = sortperm(collect(zip(recall, precision)); lt = prcorder)
    recall[:] = recall[p]
    precision[:] = precision[p]

    recall, precision
end

@recipe function plotroc(r::ROC)
    a = auc(r)
    if !get(plotattributes, :noauc, false)
        label := if haskey(plotattributes, :label) && plotattributes[:label] != ""
            @sprintf "%s (AUC = %0.3f)" plotattributes[:label] a
        elseif !haskey(plotattributes, :label)
            @sprintf "AUC = %0.3f" a
        end
    end

    xlims := (0, 1)
    ylims := (0, 1)

    grid -> false
    legend --> ifelse(a ≥ 0.5, :bottomright, :topleft)
    size --> (600, 600)
    xguide --> "FPR"
    yguide --> "TPR"
    linewidth --> 2

    @series begin
        r.fpr, r.tpr
    end

    @series begin
        seriestype := :path
        line := :dash
        linewidth := 2
        seriescolor := :dimgray
        label := nothing
        [0,1], [0,1]
    end
end

safelog(x::Float64) = iszero(x) ? zero(x) : log(x)
function rescale(scores::AbstractArray{Float64})
    a, b = extrema(scores)
    @. (scores - a) / (b - a)
end

function perf(arena::AbstractArena, edges::Vector{EdgeEvidence}, kernel::Function)
    g = graph(arena)
    gt = Float64[has_edge(g, e.src, e.dst) for e in edges]
    scores = rescale([e.evidence for e in edges])
    ll = 0.
    for (g, s) in zip(gt, scores)
        ll += kernel(g, s)
    end
    ll / length(edges)
end

logloss(arena, edges) = perf(arena, edges, (g,s) -> -(g * safelog(s) + (1. - g) * safelog(1. - s)))
brier(arena, edges) = perf(arena, edges, (g,s) -> (g - s)^2)
