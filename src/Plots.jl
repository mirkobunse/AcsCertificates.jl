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

include("plt/tightness.jl") # see sections 3.1 and B.1
include("plt/physics.jl") # see sections 3.2 and B.2

end # module
