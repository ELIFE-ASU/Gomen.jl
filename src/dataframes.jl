function DataFrames.DataFrame(c::PerformanceCurve, args... ; kwargs...)
    df = DataFrames.DataFrame(x=xvalues(c), y=yvalues(c), args...; kwargs...)
    DataFrames.rename!(df, :x => xlabel(c), :y => ylabel(c))
end
