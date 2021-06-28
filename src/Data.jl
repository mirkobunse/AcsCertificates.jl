module Data

using
    ..AcsCertificates,
    ..SigmaTest,
    DataFrames,
    PyCall,
    Random,
    Query

export BINARY_LABELS, repository, retrieve, subsample_indices

const BINARY_LABELS = [-1, 1]
_m_y(y::AbstractVector{I}) where {I<:Integer} =
    Int[ sum(y.==y_i) for y_i in BINARY_LABELS ]

"""
    repository(filter=true; m_min=500, nF=100)

List the meta-data of all available data sets that match the filter criteria.
"""
function repository(filter::Bool=true; m_min::Int=500, nF::Int=100)
    df = DataFrame(name=String[], source=String[], m_min=Int[], m_max=Int[], pY=Float64[], nF=Int[])

    # list all data sets from imblearn
    for (name, d) in _imblearn_ds().fetch_datasets()
        m_y = _m_y(d["target"])
        push!(df, [
            name,
            "imblearn",
            minimum(m_y),
            maximum(m_y),
            minimum(m_y) / sum(m_y),
            size(d["data"], 2)
        ]) # assemble the DataFrame row by row
    end

    # add FACT to this repository
    for name in ["fact_balanced", "fact_imbalanced"]
        X, y = retrieve(name)
        m_y = _m_y(y)
        push!(df, [
            name,
            "AcsCertificates",
            minimum(m_y),
            maximum(m_y),
            minimum(m_y) / sum(m_y),
            size(X, 2)
        ])
    end

    # return a sorted DataFrame that is filtered by m_min and nF
    sort!(df, [:m_min, :m_max]; rev=true)
    if filter
        df = df |> @filter(_.m_min >= m_min && _.nF <= nF) |> DataFrame
    end
    return df
end

"""
    retrieve(name)

Retrieve a tuple `(X, y)` of features and labels of a data set `name`.
"""
retrieve(name::AbstractString) =
    if name == "fact_balanced"
        retrieve_fact(; pY=.5)
    elseif name == "fact_imbalanced"
        retrieve_fact()
    else
        retrieve_imblearn(name)
    end

# retrieve data from the imbalanced-learn repository
function retrieve_imblearn(name::AbstractString)
    d = _imblearn_ds().fetch_datasets()[name]
    return d["data"], d["target"]
end

# import the package from a Conda channel
_imblearn_ds() = pyimport_conda("imblearn.datasets", "imbalanced-learn", "conda-forge")

# retrieve and prepare data from the SigmaTest module
function retrieve_fact(; subsample::Bool=true, dropna::Bool=true, pY::Float64=.01, m_maj::Int=120000)
    X, y = SigmaTest.get_training_data()
    i = _subsample_maj(y, m_maj)
    X = X[i, :]
    y = y[i]
    if subsample # physicists sub-sample to pY = 0.5
        i = subsample_indices(y, pY)
        X = X[i, :]
        y = y[i]
    end
    return X, y
end

"""
    subsample_indices(y, pY)

Return indices that represent a sub-sample of `y` according to the class proportions `pY`.
"""
function subsample_indices(y::AbstractVector{Int}, pY::Real)
    m_d = _m_y(y) # the desired number of samples after sub-sampling
    if pY >= m_d[2]/sum(m_d) # sub-sample the majority class
        m_d[1] = round(Int, m_d[2] * (1-pY) / pY)
    else # sub-sample the minority class
        m_d[2] = round(Int, m_d[1] * pY / (1-pY))
    end
    return _subsample_indices(y, m_d) # shuffle and sub-sample
end

# like subsample_indices but so that m_maj is the maximum number of sample per class
_subsample_maj(y::AbstractVector{Int}, m_maj::Int) =
    _subsample_indices(y, min.(_m_y(y), [m_maj, m_maj]))

# shuffle and sub-sample indices from a desired number of samples
function _subsample_indices(y::AbstractVector{Int}, m_d::Vector{Int})
    i = randperm(length(y)) # order after shuffling
    j = sort(vcat(
        (1:length(y))[y[i].==BINARY_LABELS[1]][1:m_d[1]],
        (1:length(y))[y[i].==BINARY_LABELS[2]][1:m_d[2]]
    )) # indices of the shuffled sub-sample
    return i[j] # apply the shuffling and the sub-sampling
end

end # module
