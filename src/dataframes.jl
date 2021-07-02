function DataFrames.DataFrame(roc::ROC, args... ; kwargs...)
    DataFrames.DataFrame(FPR=fpr(roc), TPR=tpr(roc), args...; kwargs...)
end
