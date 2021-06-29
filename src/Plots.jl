module Plots

using
    ..AcsCertificates,
    CSV,
    DataFrames,
    PGFPlots,
    Query,
    StatsBase,
    TikzPictures

detokenize(x::String) = replace(x, "_" => "\\_") # escape underscores
detokenize(x::Real) = string(round(x; digits=4)) # round numbers

include("plt/tightness.jl") # see Sec. 3.1 and Tab. 1
include("plt/physics.jl") # see Sec. 3.2 and Tab. 2

end # module
