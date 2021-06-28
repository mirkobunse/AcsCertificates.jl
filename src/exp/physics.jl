"""
    physics(config_path, results_path)

Conduct the astro-particle physics experiment from sections 3.2 and B.2.
"""
function physics(config_path, results_path)
    config = parsefile(config_path)
    @info "Read the configuration at $config_path"

    # patch the configuration
    config["rskf"]["n_splits"] = 3
    n_steps = *(
        config["rskf"]["n_splits"],
        config["rskf"]["n_repeats"]
    ) # just a multi-line product
    config["progress"] = Progress(n_steps; barlen=20)

    # conduct all trials, collect and store all results
    df = _physics(config) # only a single trial
    @info "Writing results" results_path
    CSV.write(results_path, df)
    return df
end

# a single trial of the experiment
function _physics(config)
    rskf = SkObject("sklearn.model_selection.RepeatedStratifiedKFold", config["rskf"])
    Random.seed!(config["rskf"]["random_state"]) # always sub-sample the same FACT data
    X, y = Data.retrieve("fact_balanced")
    X_wrk = SigmaTest.get_crab_data() # working sample of the Li&Ma sigma test
    L = LossFunctions.ZeroOneLoss() # the loss function

    # instantiate the default FACT classifier
    clf = SkObject(
        "sklearn.ensemble.RandomForestClassifier";
        n_estimators = 200,
        max_features = "sqrt",
        n_jobs = -1,
        max_depth = 15,
        criterion = "entropy",
        random_state = config["rskf"]["random_state"] # replicable results
    ) # https://github.com/fact-project/open_crab_sample_analysis/blob/b47ff38ada3d44194244e423cf32014ac1f6ee0f/configs/aict.yaml#L62

    # prepare the result storage
    df = DataFrame(
        i_rskf = Int[],
        delta = Float64[],
        sigma = Float64[],
        prediction_threshold = Float64[],
        p_S = Float64[],
        p_T = Float64[],
        epsilon = Float64[],
        L_hadron = Float64[],
        L_gamma = Float64[]
    ) # define the types and names of all columns

    # repeatedly split data into training, validation, and test sets
    for (i_rskf, (trn_val, tst)) in enumerate(rskf.split(X, y))
        trn, val = _stratified_split(X, y, trn_val) # split trn_val into trn and val indices

        # train the classifier and predict the validation set
        ScikitLearn.fit!(clf, X[trn.+1,:], y[trn.+1])
        i_min = findfirst(clf.classes_ .== 1) # index of the minority class (1 or 2)
        proba_val = ScikitLearn.predict_proba(clf, X[val.+1,:])[:,i_min] # ∈ [0, 1] this time
        proba_tst = ScikitLearn.predict_proba(clf, X[tst.+1,:])[:,i_min]
        proba_wrk = ScikitLearn.predict_proba(clf, X_wrk)[:,i_min]
        sigma, prediction_threshold = SigmaTest.lima_test(proba_wrk) # optimize the threshold

        # map probabilities to crisp predictions
        y_h_val = ones(Int, length(proba_val))
        y_h_tst = ones(Int, length(proba_tst))
        y_h_wrk = ones(Int, length(proba_wrk))
        y_h_val[proba_val .< prediction_threshold] .= -1
        y_h_tst[proba_tst .< prediction_threshold] .= -1
        y_h_wrk[proba_wrk .< prediction_threshold] .= -1
        L_y = empirical_classwise_risk(L, y_h_tst, y[tst.+1])
        p_S = sum(y_h_val .== 1) / length(y_h_val) # predicted class proportions
        p_T = sum(y_h_wrk .== 1) / length(y_h_wrk)
        required_Δp = abs(p_T - 1e-4) # we need certificates for this Δp*

        for delta in config["delta"]
            certificate = Certificate(L, y_h_val, y[val.+1]; δ=delta, warn=false)
            epsilon = required_Δp * certificate.Δℓ # this ϵ is needed to certify required_Δp

            # store the results of this iteration
            push!(df, [
                i_rskf,
                delta,
                sigma,
                prediction_threshold,
                p_S,
                p_T,
                epsilon,
                L_y[1],
                L_y[2]
            ]) # see the column definition of df above
        end
        _advance_progress!(config, i_rskf, sigma)
    end
    return df
end

# update the progress bar with trial information
_advance_progress!(config, i_rskf, sigma) =
    ProgressMeter.next!(
        config["progress"];
        showvalues = [
            (:i_rskf, i_rskf),
            (:sigma, sigma)
        ]
    )
