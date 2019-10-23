struct ROC
    fpr::Vector{Float64}
    tpr::Vector{Float64}
    ROC() = new([0.0, 1.0], [0.0, 1.0])
    function ROC(fpr::Vector{Float64}, tpr::Vector{Float64}; down=true)
        if size(fpr) != size(tpr)
            throw(DimensionMismatch("TPR and FPR vectors have different sizes"))
        end

        if !issorted(fpr)
            p = sortperm(fpr)
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

function downsample(xs::AbstractVector, ys::AbstractVector)
    keep = [1]
    for i in 2:length(xs)
        if xs[i] != xs[i-1] || ys[i] != ys[i-1]
            push!(keep, i)
        end
    end
    keep
end

function MLBase.roc(arena::AbstractArena, edges::Vector{EdgeEvidence}, args...)
    g = graph(arena)
    gt = [Int(has_edge(g, e.src, e.dst)) for e in edges]
    scores = [e.evidence for e in edges]
    r = roc(gt, scores, args...)
    ROC(map(false_positive_rate, r), map(true_positive_rate, r))
end

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
