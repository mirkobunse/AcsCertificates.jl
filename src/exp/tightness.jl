"""
    tightness(config_path, crt_path, val_path, tst_path)

Conduct the tightness experiment from sections 3.1 and B.1.
"""
function tightness(config_path, crt_path, val_path, tst_path)
    config = parsefile(config_path)
    @info "Read the configuration at $config_path"

    # select the data to analyze
    repository = Data.repository(;
        m_min=config["data"]["m_min"],
        nF=config["data"]["nF"]
    )
    @info "Analyzing the following data sets:" repository config["data"]
    config["data"] = repository[!, "name"] # replace query with result

    # additional patches for the configuration
    config["pY_tst"] = _logspace(config["pY_tst"]) # logarithmic pY values
    config["rskf"]["n_splits"] = 3 # training, validation, and testing
    config["sample_size_multiplier"] = config["rskf"]["n_splits"] # the effective n_samples when a mean is plotted
    n_steps_per_trial = 2 * length(config["loss"]) * length(config["delta"]) # calls to _advance_progress per trial

    # conduct all trials, collect and store all results
    df_crt, df_val, df_tst = _merge(map(_tightness, _expand(config, n_steps_per_trial)))
    @info "Writing results" crt_path val_path tst_path
    for (df, path) in [(df_crt, crt_path), (df_val, val_path), (df_tst, tst_path)]
        CSV.write(path, df)
    end
    return df_crt, df_val, df_tst
end

# the actual expansion is filtered and further patched
function _expand(config, n_steps_per_trial)
    expansion = Dict{String,Any}[]
    for c in expand(config, "data", "clf", "weight")
        if c["data"] == "fact_balanced" || c["data"] == "fact_imbalanced"
            if c["clf"] == config["clf"][1]
                continue # ignore all but one FACT configuration
            end
            c["clf"] = "sklearn.ensemble.RandomForestClassifier" # patch
        end
        push!(expansion, c)
    end

    # prepare the progress bar
    n_steps = *(
        length(expansion),
        config["rskf"]["n_splits"],
        config["rskf"]["n_repeats"],
        n_steps_per_trial
    ) # multi-line product
    progress = Progress(n_steps; barlen=20)
    for c in expansion
        c["progress"] = progress # all trials update one progress bar
    end
    return expansion
end

# in an array of tuples, concatenate all DataFrames that are at the same tuple index
_merge(v::Vector{NTuple{N, DataFrame}}) where N =
    ( vcat(getproperty.(v, i)...) for i in 1:N )

# n logarithmically equidistant steps between l and u
_logspace(c) = _logspace(c["l"], c["u"], c["n"])
_logspace(l::Real, u::Real, n::Int) = 10. .^ (log10(l):((log10(u)-log10(l))/(n-1)):log10(u))

# a single trial of the experiment
function _tightness(config)
    rskf = SkObject("sklearn.model_selection.RepeatedStratifiedKFold", config["rskf"])
    Random.seed!(config["rskf"]["random_state"]) # always sub-sample the same FACT data
    X, y = Data.retrieve(config["data"])

    # instantiate the classifier
    clf_args = Dict{String,Any}() # classifier config
    if config["clf"] == "sklearn.ensemble.RandomForestClassifier"
        clf_args = Dict{String,Any}(
            "n_estimators" => 20, # 20 instead of 200 for faster reproducibility
            "max_features" => "sqrt",
            "n_jobs" => -1,
            "max_depth" => 15,
            "criterion" => "entropy",
            "random_state" => config["rskf"]["random_state"] # replicable results
        ) # https://github.com/fact-project/open_crab_sample_analysis/blob/b47ff38ada3d44194244e423cf32014ac1f6ee0f/configs/aict.yaml#L62
    end # other classifiers use default parameters
    clf = SkObject(config["clf"], clf_args)

    # shift pY_tst so that one of its values matches the training set proportions
    pY_tst = copy(config["pY_tst"])
    pY_diff = pY_tst .- sum(y .== 1)/length(y)
    pY_tst .-= pY_diff[argmin(abs.(pY_diff))]

    # prepare result storages for the certificates and the test set evaluations
    df_crt = DataFrame(i_rskf=Int[], loss=String[], delta=Float64[], epsilon=Float64[], max_p=Float64[], onesided=Bool[])
    df_val = DataFrame(i_rskf=Int[], loss=String[], delta=Float64[], epsilon=Float64[], pY=Float64[], value=Float64[])
    df_tst = copy(df_val) # same columns

    # repeatedly split data into training, validation, and test sets
    for (i_rskf, (trn_val, tst)) in enumerate(rskf.split(X, y))
        trn, val = _stratified_split(X, y, trn_val) # split trn_val into trn and val indices

        pY_val = sum(y[trn_val.+1] .== 1) / length(y[trn_val.+1]) # training/validation proportions
        w_y = _class_weights(pY_val, config["weight"]) # determine the class weights
        w_trn = _sample_weight(w_y, y[trn.+1]) # sample weights

        # train the classifier and predict the validation and test sets
        ScikitLearn.fit!(clf, X[trn.+1,:], y[trn.+1]; sample_weight=w_trn)
        i_min = findfirst(clf.classes_ .== 1) # index of the minority class (1 or 2)
        y_h_val = ScikitLearn.predict_proba(clf, X[val.+1,:])[:,i_min] .* 2 .- 1 # ∈ [-1, +1]
        y_h_tst = ScikitLearn.predict_proba(clf, X[tst.+1,:])[:,i_min] .* 2 .- 1

        for c in expand(config, "loss", "delta") # for all losses and deltas...
            L = getproperty(LossFunctions, Symbol(c["loss"]))() # instantiate

            # 1) certify wrt the training set predictions
            # [TODO: try to reduce the tol value to have more two-sided certificates]
            certificate = Certificate(L, y_h_val, y[val.+1]; δ=c["delta"], warn=false, w_y=w_y)
            for epsilon in c["epsilon"]
                push!(df_crt, [
                    i_rskf,
                    c["loss"],
                    c["delta"],
                    epsilon,
                    max_Δp(certificate, epsilon), # Δp* at epsilon
                    is_onesided(certificate)
                ]) # assemble the DataFrame row by row
            end
            _advance_progress!(config) # one update per certificate

            # 2) estimate the training error with epsilon
            w_val = _sample_weight(w_y, y[val.+1])
            L_val = mean(LossFunctions.value(L, y[val.+1], y_h_val) .* w_val)
            ϵ_val = _ϵ(length(y[val.+1]) * c["sample_size_multiplier"], c["delta"])
            push!(df_val, [ i_rskf, c["loss"], c["delta"], ϵ_val, pY_val, L_val ])

            # 3) estimate different testing errors with epsilon
            for pY in pY_tst
                i_pY = Data.subsample_indices(y[tst.+1], pY)
                y_pY = y[tst.+1][i_pY]
                y_h_pY = y_h_tst[i_pY]
                w_tst = _sample_weight(_class_weights(pY, c["weight"]), y_pY)
                L_tst = mean(LossFunctions.value(L, y_pY, y_h_pY) .* w_tst)
                δ_tst = c["delta"] * 2 # adjust δ for a fair comparison
                ϵ_tst = _ϵ(length(y_pY) * c["sample_size_multiplier"], δ_tst)
                push!(df_tst, [ i_rskf, c["loss"], c["delta"], ϵ_tst, pY, L_tst ])
            end
            _advance_progress!(config) # another update for all evaluations
        end
    end

    # update all DataFrames with information on the current trial
    for df in [ df_crt, df_val, df_tst ], column in [ "data", "clf", "weight" ]
        df[!, column] .= config[column]
    end
    return df_crt, df_val, df_tst
end

# indices of a single stratified split of X[trn_val.+1,:] and y[trn_val.+1]
function _stratified_split(X, y, trn_val)
    rskf = SkObject("sklearn.model_selection.RepeatedStratifiedKFold"; n_splits=2, n_repeats=1)
    trn, val = rskf.split(X[trn_val.+1,:], y[trn_val.+1]).__next__() # first item of generator
    return trn_val[trn.+1], trn_val[val.+1] # trn, val
end

# determine class weights based on their proportions
function _class_weights(pY, weight)
    w_y = Dict(
        "uniform" => [1., 1.],
        "proportional" => 1 ./ [1-pY, pY],
        "sqrt" => sqrt.(1 ./ [1-pY, pY])
    )[weight]
    return w_y ./ maximum(w_y) # w_y ∈ [0, 1]
end

# weights of samples derived from class weights
_sample_weight(w_y, y) = w_y[convert.(Int, y./2 .+ 1.5)] # map labels ±1 to indices 1 and 2

# update the progress bar
_advance_progress!(config) = ProgressMeter.next!(
    config["progress"];
    showvalues=[(:clf, config["clf"]), (:data, config["data"]), (:weight, config["weight"])]
)

# one-sided maximum error with probability at least 1-δ
_ϵ(m::Int, δ::Float64) = sqrt(-log(δ) / (2*m))
