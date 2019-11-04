struct ROC
    fpr::Vector{Float64}
    tpr::Vector{Float64}
    ROC() = new([0.0, 1.0], [0.0, 1.0])
    function ROC(fpr::Vector{Float64}, tpr::Vector{Float64}; down=true)
        if size(fpr) != size(tpr)
            throw(DimensionMismatch("TPR and FPR vectors have different sizes"))
        end

        if !(issorted(fpr) && issorted(tpr))
            p = sortperm(collect(zip(fpr, tpr)))
            fpr = fpr[p]
            tpr = tpr[p]
        end

        if !issorted(tpr)
            throw(ArgumentError("the ROC curve is not monotonic increasing"))
        end

        if down
            d = downsample(fpr, tpr)
            fpr = fpr[d]
            tpr = tpr[d]
        end

        if fpr[1] < 0.0
            throw(ArgumentError("negative FPR provided"))
        elseif tpr[1] < 0.0
            throw(ArgumentError("negative TPR provided"))
        elseif !(fpr[1] ≈ tpr[1] ≈ 0.0)
            pushfirst!(fpr, 0.0)
            pushfirst!(tpr, 0.0)
        end

        if fpr[end] > 1.0
            throw(ArgumentError("FPR greater than 1.0 provided"))
        elseif tpr[end] > 1.0
            throw(ArgumentError("FPR greater than 1.0 provided"))
        elseif !(fpr[end] ≈ tpr[end] ≈ 1.0)
            push!(fpr, 1.0)
            push!(tpr, 1.0)
        end

        new(fpr, tpr)
    end
end

fpr(r::ROC) = r.fpr

tpr(r::ROC) = r.tpr

function downsample(xs::AbstractVector{Float64}, ys::AbstractVector{Float64})
    keep = [1]
    for i in 2:length(xs)
        if !(xs[i] ≈ xs[i-1] && ys[i] ≈ ys[i-1])
            push!(keep, i)
        end
    end
    keep
end

Base.length(r::ROC) = length(fpr(r))

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

function interpolate(xs::AbstractVector{Float64}, ys::AbstractVector{Float64}, x::Float64)
    m = (ys[2] - ys[1]) / (xs[2] - xs[1])
    m * (x - xs[1]) + ys[1]
end

Base.:/(c::Curve, k::Number) = Curve(c.xs, c.ys / k)

Statistics.mean(rocs::AbstractVector{ROC}) = let c = mean(map(r -> Curve(fpr(r), tpr(r)), rocs))
    ROC(c.xs, c.ys)
end

function MLBase.roc(arena::AbstractArena, edges::Vector{EdgeEvidence}, args...)
    g = graph(arena)
    gt = [Int(has_edge(g, e.src, e.dst)) for e in edges]
    scores = [e.evidence for e in edges]
    r = roc(gt, scores, args...)
    ROC(map(false_positive_rate, r), map(true_positive_rate, r))
end

auc(r::ROC) = @views dot(r.tpr[2:end] + r.tpr[1:end-1], r.fpr[2:end] - r.fpr[1:end-1]) / 2.0

@recipe function plotroc(r::ROC)
   legend := false
   grid := false
   xlim := (0, 1)
   ylim := (0, 1)

   size --> (600, 600)
   xlabel --> "FPR"
   ylabel --> "TPR"
   linewidth --> 2

   @series begin
       seriestype := :path
       line := :dash
       linewidth := 2
       color := :dimgray
       [0,1], [0,1]
   end

   @series begin
       r.fpr, r.tpr
   end
end