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

include("exp/tightness.jl") # see Sec. 3.1 and Tab. 1
include("exp/physics.jl") # see Sec. 3.2 and Tab. 2

end # module
