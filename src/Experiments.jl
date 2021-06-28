module Experiments

using
    ..AcsCertificates,
    ..Certificates,
    ..Data,
    ..SigmaTest,
    CSV,
    DataFrames,
    LinearAlgebra,
    LossFunctions,
    MetaConfigurations,
    ProgressMeter,
    Random,
    ScikitLearn,
    StatsBase

include("exp/tightness.jl") # see sections 3.1 and B.1
include("exp/physics.jl") # see sections 3.2 and B.2

end # module
