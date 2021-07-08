"""
    acquisition(config_path, results_path)

Conduct the acquisition experiment from our IAL workshop submission.
"""
function acquisition(config_path, results_path)
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
    config["pY_tst"] = _logspace(config["pY_tst"]) # _logspace is defined in src/exp/tightness.jl
    config["rskf"]["n_splits"] = 3 # training, validation, and testing
    config["sample_size_multiplier"] = config["rskf"]["n_splits"] # the effective n_samples when a mean is plotted

    # conduct all trials, collect and store all results
    df = vcat(map(_acquisition, _expand_acquisition(config))...)
    @info "Writing results" results_path
    CSV.write(results_path, df)
    return df
end

# the actual expansion is filtered and further patched
function _expand_acquisition(config)
    expansion = Dict{String,Any}[]
    for c in expand(config, "data", "strategy", "clf", "weight", "pY_tst", "loss", "delta", "epsilon")
        if c["data"] == "fact_imbalanced"
            continue # ignore the imbalanced subsample; subsample custom pY_tst instances
        end
        push!(expansion, c)
    end

    # prepare the progress bar
    n_steps = *(
        length(expansion),
        config["rskf"]["n_splits"],
        config["rskf"]["n_repeats"],
        config["n_batches"]
    ) # multi-line product
    progress = Progress(n_steps; barlen=20)
    for c in expansion
        c["progress"] = progress # all trials update one progress bar
    end
    return expansion
end

# a single trial of the experiment
function _acquisition(config)
    rskf = SkObject("sklearn.model_selection.RepeatedStratifiedKFold", config["rskf"])
    Random.seed!(config["rskf"]["random_state"]) # always sub-sample the same FACT data
    X, y = Data.retrieve(config["data"])

    # instantiate the classifier and loss function
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
    L = getproperty(LossFunctions, Symbol(config["loss"]))()

    # prepare the result storage
    df = DataFrame(
        i_rskf   = Int[], # iteration of the rskf
        batch    = Int[], # number of the ACS acquisition batch
        N_min    = Int[], # number of minority-class training set instances
        N_maj    = Int[], # number of majority-class instances
        L_tst    = Float64[] # training set loss
    )

    # predetermine Random.seed!s for all i_rskf starts
    seeds = rand(UInt32, rskf.get_n_splits())

    # repeatedly split data into training, validation, and test sets
    for (i_rskf, (trn, tst)) in enumerate(rskf.split(X, y))
        Random.seed!(seeds[i_rskf]) # each split has the same starting point
        config["__cache__"] = nothing # invalidate optional caches of strategies
        i_tst = Data.subsample_indices(y[tst.+1], config["pY_tst"])
        y_tst = y[tst.+1][i_tst]
        X_tst = X[tst.+1,:][i_tst, :]
        w_tst = _sample_weight(_class_weights(config["pY_tst"], config["weight"]), y_tst)

        # set up the training data pool
        X_trn = X[trn.+1,:]
        y_trn = y[trn.+1]
        pY_trn = sum(y_trn .== 1) / length(y_trn) # class proportions of the entire pool
        w_y = _class_weights(pY_trn, config["weight"]) # _class_weights from src/exp/tightness.jl
        w_trn = _sample_weight(w_y, y_trn) # _sample_weight from src/exp/tightness.jl

        # the first batch is uniformly sampled
        i_trn = Data._subsample_maj(y_trn, floor(Int, config["batchsize"]/2))

        # the ACS data acquisition loop
        for batch in 1:config["n_batches"]

            # 1) train and evaluate the classifier
            m_trn = Data._m_y(y_trn[i_trn]) # numbers of training set instances
            ScikitLearn.fit!(clf, X_trn[i_trn, :], y_trn[i_trn]; sample_weight=w_trn[i_trn])
            i_min = findfirst(clf.classes_ .== 1) # index of the minority class (1 or 2)
            y_h_tst = ScikitLearn.predict_proba(clf, X_tst)[:,i_min] .* 2 .- 1 # ∈ [-1, +1]
            L_tst = mean(LossFunctions.value(L, y_tst, y_h_tst) .* w_tst)

            # 2) store and log information
            push!(df, [i_rskf, batch, m_trn[2], m_trn[1], L_tst])
            _progress_acquisition!(config)
            # @info "Iteration complete" strategy=config["strategy"] batch m_trn L_tst

            # 3) acquire new data
            if batch < config["n_batches"]
                y_h_trn = ScikitLearn.predict_proba(clf, X_trn[i_trn, :])[:,i_min] .* 2 .- 1
                m_d = _m_d(L, y_h_trn, y_trn[i_trn], w_y, config)
                m_s = _sanitize_m_d(m_d, config) # error handling
                try
                    i_trn = _acquire(i_trn, y_trn, m_s, config)
                catch some_error
                    if isa(some_error, BoundsError)
                        m_rem = Data._m_y(y_trn[setdiff(1:length(y_trn), i_trn)])
                        if any(m_rem .< m_s) # not enough data remaining; no need to complain
                            _progress_acquisition!(config, config["n_batches"]-batch) # remaining steps
                            break # stop the ACS loop
                        else
                            @error "BoundsError" batch m_trn m_s m_rem m_tst=Data._m_y(y_tst)
                            rethrow()
                        end
                    else
                        rethrow()
                    end
                end
            end

        end # batch
    end # rskf

    # update the results storage with information on the current trial
    for column in [ "data", "strategy", "clf", "weight", "pY_tst", "loss", "delta", "epsilon" ]
        df[!, column] .= config[column]
    end
    return df
end

_m_d(L, y_h, y, w_y, config) =
    if config["strategy"] == "uniform"
        return fill(config["batchsize"] / 2, 2) # = (N/2, N/2)
    elseif config["strategy"] == "proportional"
        p_d = [1-config["pY_tst"], config["pY_tst"]] # desired class proportions
        m_d = p_d .* (length(y) + config["batchsize"]) # desired number of samples
        return m_d - Data._m_y(y) # what to acquire
    elseif config["strategy"] == "inverse"
        empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y)))
        utility = 1 ./ (1 .- empirical_ℓ_y) # inverse accuracy if L==ZeroOneLoss()
        return utility
    elseif config["strategy"] == "improvement"
        empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y)))
        if config["__cache__"] == nothing # first iteration: inverse strategy
            utility = 1 ./ (1 .- empirical_ℓ_y)
        else
            utility = max.(0, config["__cache__"] - empirical_ℓ_y) # reduction in loss
        end
        config["__cache__"] = empirical_ℓ_y # store / update losses
        return utility
    elseif config["strategy"] == "redistriction"
        if config["__cache__"] == nothing # first iteration: inverse strategy
            empirical_ℓ_y = min.(1-1e-4, max.(1e-4, empirical_classwise_risk(L, y_h, y)))
            utility = 1 ./ (1 .- empirical_ℓ_y)
        else
            N = length(config["__cache__"]) # number of samples in previous iteration
            is_redistricted = sign.(y_h[1:N]) .!= config["__cache__"]
            utility = [sum(is_redistricted[y[1:N].==1]), sum(is_redistricted[y[1:N].==-1])]
        end
        config["__cache__"] = sign.(y_h) # store / update predictions
        return utility
    elseif config["strategy"] == "certification"
        c = Certificate(L, y_h, y;
            δ=config["delta"],
            warn=false,
            w_y=w_y,
            allow_onesided=false, # acquisition certificates must be two-sided
            n_trials_extra=7 # allow more trials if 3 random initializations fail
        )
        α, β = beta_parameters(config["pY_tst"], config["pY_tst"])
        return suggest_acquisition(c, config["batchsize"], α, β)
    else
        throw(ValueError("Unknown strategy \"$strategy\""))
    end

function _sanitize_m_d(m_d, config)
    m_s = max.(0.0, m_d)
    m_s[isnan.(m_s)] .= 0
    m_s[m_s .== Inf] .= 1
    if sum(m_s) == 0
        m_s = fill(config["batchsize"]/2, 2)
    end
    m_s = round.(Int, m_s .* (config["batchsize"] / sum(m_s)))
    if sum(m_s) == config["batchsize"]+2
        m_s .-= 1
    elseif sum(m_s) == config["batchsize"]+1
        m_s[findmax(m_s)[2]] -= 1
    elseif sum(m_s) == config["batchsize"]-1
        m_s[findmin(m_s)[2]] += 1
    elseif sum(m_s) == config["batchsize"]-2
        m_s .+= 1
    end
    if sum(m_s) != config["batchsize"]
        @warn "_sanitize_m_d" m_d m_s sum(m_s)
    end
    return m_s
end

function _acquire(i_trn, y_trn, m_s, config)
    i_rem = setdiff(1:length(y_trn), i_trn)
    i_acq = i_rem[Data._subsample_indices(y_trn[i_rem], m_s)]
    return vcat(i_trn, i_acq)
end

_progress_acquisition!(config) = ProgressMeter.next!(
    config["progress"];
    showvalues=[(:strategy, config["strategy"]), (:data, config["data"]), (:clf, config["clf"])]
)
_progress_acquisition!(config, n::Int) = for _ in 1:n _progress_acquisition!(config) end
