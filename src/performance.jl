struct ROC
    points::Vector{ROCNums}
end

function MLBase.roc(arena::AbstractArena, edges::Vector{EdgeEvidence}, args...)
    g = graph(arena)
    gt = [Int(has_edge(g, e.src, e.dst)) for e in edges]
    scores = [e.evidence for e in edges]
    ROC(roc(gt, scores, args...))
end

MLBase.true_positive_rate(r::ROC) = map(true_positive_rate, r.points)
MLBase.false_positive_rate(r::ROC) = map(false_positive_rate, r.points)

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
