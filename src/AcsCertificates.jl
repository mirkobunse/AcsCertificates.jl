module AcsCertificates

using PyCall, ScikitLearn
export tightness, physics, SkObject

function __init__()
    ScikitLearn.Skcore.import_sklearn() # esure sklearn is installed
    pyimport("warnings").filterwarnings("ignore") # ignore warnings
end

"""
    SkObject(class_name, configuration)
    SkObject(class_name; kwargs...)

Instantiate a scikit-learn `PyObject` by its fully qualified `class_name`.
"""
SkObject(class_name::AbstractString, config::Dict{String, Any}) =
    SkObject(class_name; [ Symbol(k) => v for (k, v) in config ]...)
SkObject(class_name::AbstractString; kwargs...) =
    getproperty(
        pyimport(join(split(class_name, ".")[1:end-1], ".")), # package
        Symbol(split(class_name, ".")[end]) # classname
    )(; kwargs...) # constructor call: package.classname(**kwargs)


# import sub-modules
include("SigmaTest.jl")
using .SigmaTest

include("Data.jl")
using .Data

include("Certificates.jl")
using .Certificates
export Certificate, is_feasible, is_onesided, p_range, max_Δp, optimize_Δℓ, empirical_classwise_risk

include("Experiments.jl")
using .Experiments

include("Plots.jl")
using .Plots


"""
    tightness(config_path, output_path; kwargs...)

Conduct the tightness experiment from Sec. 3.1 and Tab. 1 and generate
LaTeX plots and tables from the results.
"""
function tightness(config_path, output_path;
        crt_path = "plot/.tightness_crt.csv",
        val_path = "plot/.tightness_val.csv",
        tst_path = "plot/.tightness_tst.csv")
    Experiments.tightness(config_path, crt_path, val_path, tst_path)
    Plots.tightness(crt_path, val_path, tst_path, output_path)
end

"""
    physics(config_path, output_path; kwargs...)

Conduct the astro-particle physics experiment from Sec. 3.2 and Tab. 2
and generate a LaTeX table from the results.
"""
function physics(config_path, output_path;
        results_path = "plot/.physics.csv")
    Experiments.physics(config_path, results_path)
    Plots.physics(results_path, output_path)
end


end # module
