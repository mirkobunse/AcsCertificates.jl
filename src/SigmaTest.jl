"""
    module SigmaTest

A thin wrapper for the FACT standard analysis that is implemented at
`https://bitbucket.org/mbunse/sigma-test`.
"""
module SigmaTest

using
    ..AcsCertificates,
    PyCall

export get_training_data, get_crab_data, lima_test

"""
    get_training_data()

Return a tuple `(X, y)` of simulated FACT training data.
"""
function get_training_data()
    config = _sigma_config()
    X, y = _sigma("skl").get_training_data(;
        gamma_dl2_path = _local_copy(config["gamma_diffuse_dl2_url"]),
        hadron_dl2_path = _local_copy(config["hadron_dl2_url"])
    )
    return X, y .* 2 .- 1 # map the labels to ±1
end

"""
    get_crab_data()

Return feature vectors `X` of the open FACT crab sample.
"""
get_crab_data() =
    _sigma("skl").get_crab_data(;
        crab_dl2_path = _local_copy(_sigma_config()["crab_dl2_url"])
    )

"""
    lima_test(crab_gamma_probabilities)

Return the Li&Ma sigma value for the given open FACT crab sample predictions.

**See also:** `AcsCertificates.SigmaTest.get_crab_data()`
"""
lima_test(p::AbstractVector{T}) where T <: Real =
    _sigma("skl").lima_test(p;
        crab_dl2_path = _local_copy(_sigma_config()["crab_dl2_url"]),
        crab_dl3_path = _local_copy(_sigma_config()["crab_dl3_url"]),
        prediction_threshold = "optimize"
    )

# make sure that a local copy is at basename(url)
function _local_copy(url::AbstractString)
    if !isfile(basename(url))
        @info "Downloading $(url)"
        download(url, basename(url))
    end
    return basename(url)
end

# import a sigma sub-module
function _sigma(submodule::AbstractString)
    try
        return pyimport("sigma.$submodule")
    catch exception
        _install_sigma()
        return pyimport("sigma.$submodule") # should work this time
    end
end

# import the sigma.CONFIG dictionary
function _sigma_config()
    try
        return pyimport("sigma").CONFIG
    catch exception
        _install_sigma()
        return pyimport("sigma").CONFIG
    end
end

# install sigma-test via pip
function _install_sigma()
    pyimport_conda("pip", "pip", "anaconda") # make sure pip is installed
    url = "git+https://bitbucket.org/mbunse/sigma-test.git"
    run(`$(pyimport("sys").executable) -m pip install $(url)`)
end

end # module
