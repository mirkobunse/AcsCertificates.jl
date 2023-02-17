"""
    acquisition(filename, strategy_selection)

Generate a LaTeX table from the results of the acquisition experiment.
"""
function acquisition(filename::String, strategy_selection::Vector{String}; 
        results_path::String="plot/.acquisition.csv", base_output_dir="plot/")

    df = CSV.read(results_path, DataFrame)
    idx = map(strategy -> strategy ∈ strategy_selection ? true : false, df[!, :name])
    df = df[idx, :]

    # similarities between strategies
    df[!, :pY_trn] = df[!, :N_min] ./ (df[!, :N_min] .+ df[!, :N_maj])
    df[!, :kl_unif] = _kl.(df[!, :pY_trn], .5)
    df[!, :kl_prop] = _kl.(df[!, :pY_trn], df[!, :pY_tst])
    count_strategies = length(unique(df[!, :name]))
    @info "There were identified $(count_strategies) strategy configurations" 

    # average test error over all CV repetitions
    gid = [:batch, :clf, :weight, :loss, :delta, :epsilon]
    df = combine(
        groupby(df, vcat(gid, [:name, :pY_tst, :data])),
        :L_tst => StatsBase.mean => :L_tst,
        :kl_unif => StatsBase.mean => :kl_unif,
        :kl_prop => StatsBase.mean => :kl_prop
    )
    count_strategies = length(unique(df[!, :name]))
    @info "There were identified $(count_strategies) strategy configurations" 

    # select trials where all strategies were evaluated
    cnt = filter(
        :nrow => x -> x .== length(unique(df[!, :name])),
        combine(groupby(df, [:data, :batch, :pY_tst]), nrow)
    )
    df = semijoin(df, cnt, on=[:data, :batch, :pY_tst])
    df[!, :name] = map(x -> _mapping_names(x), df[!, :name])

    _plot_critical_diagram(df, base_output_dir * filename * "_CD.tex", count_strategies; gid=gid)
    _plot_kl_diagram(df, base_output_dir *  filename * "_KL.tex"; gid=gid)
end

# scalar version for binary classification
_kl(p, q) = Distances.kl_divergence([p, 1-p], [q, 1-q])


function _plot_critical_diagram(df, output_path, count_strategies; gid=[:batch, :clf, :loss, :delta])
    sequence = Pair{String, Vector{Pair{String, Vector}}}[]
    for (key, sdf) in pairs(groupby(df, gid))
        n_data = length(unique(sdf[!, :data]))
        if key.batch ∉ 2:9
            continue
        end
        @info "Batch $(key.batch) is based on $(n_data) data sets"
        title = string(key.batch)
        pairs = CriticalDifferenceDiagrams._to_pairs(sdf, :name, :data, :L_tst)
        push!(sequence, title => pairs)
    end
    plot = CriticalDifferenceDiagrams.plot(sequence...)
    plot.style = join([
        "y dir=reverse",
        "ytick={1,2,3,4,5,6,7,8}",
        "yticklabels={2,3,4,5,6,7,8,9}",
        "ylabel={ACS-Batch}",
        "xlabel={avg. Rang}",
        "ylabel style={font=\\small}",
        "xlabel style={font=\\small}",
        "yticklabel style={font=\\small}",
        "xticklabel style={font=\\small}",
        "grid=both",
        "axis line style={draw=none}",
        "tick style={draw=none}",
        "xticklabel pos=upper",
        "xmin=.5",
        "xmax=$(count_strategies + 0.5)",
        "ymin=.75",
        "ymax=$(length(sequence)).75",
        "clip=false",
        "legend style={draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em}",
        "x dir=reverse",
        "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*},{gray, mark=diamond*}}",
        "width=\\axisdefaultwidth, height=\\axisdefaultheight"
    ], ", ")

    PGFPlots.resetPGFPlotsPreamble()
    PGFPlots.pushPGFPlotsPreamble(join([
        "\\usepackage{amsmath}",
        "\\usepackage{amssymb}",
        "\\definecolor{tu01}{HTML}{84B818}",
        "\\definecolor{tu02}{HTML}{D18B12}",
        "\\definecolor{tu03}{HTML}{1BB5B5}",
        "\\definecolor{tu04}{HTML}{F85A3E}",
        "\\definecolor{tu05}{HTML}{4B6CFC}",
        "\\definecolor{chartreuse(traditional)}{rgb}{0.87, 1.0, 0.0}"
    ], "\n"))
    PGFPlots.save(output_path, plot)
end

function _plot_kl_diagram(df, output_path; gid=[:batch, :clf, :loss, :delta])

    plot = Axis(style = join([
        "title={KL-Divergenz nach \$p_\\mathcal{T}=0.8\$}",
        "legend style={draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em}",
        "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*}}",
        "xtick={1,3,5,7}"
    ], ", "))
    for (key, sdf) in pairs(groupby(df, vcat(setdiff(gid, [:batch]), [:name])))
        if key.name == "proportional"
            continue # KL divergence is usually zero, so that ymode=log does not work in general
        end
        sdf = combine(
            groupby(
                sdf[sdf[!, :batch].<=8, :],
                vcat(gid, [:name])
            ),
            :kl_prop => StatsBase.mean => :kl_prop,
            :kl_prop => StatsBase.std => :kl_prop_std
        )
        push!(plot, PGFPlots.Plots.Linear(
            sdf[!, :batch],
            sdf[!, :kl_prop],
            legendentry=string(key.name),
            errorBars=PGFPlots.ErrorBars(y=sdf[!, :kl_prop_std])
        ))
    end
    
    PGFPlots.resetPGFPlotsPreamble()
    PGFPlots.pushPGFPlotsPreamble(join([
        "\\usepackage{amsmath}",
        "\\usepackage{amssymb}",
        "\\definecolor{tu01}{HTML}{84B818}",
        "\\definecolor{tu02}{HTML}{D18B12}",
        "\\definecolor{tu03}{HTML}{1BB5B5}",
        "\\definecolor{tu04}{HTML}{F85A3E}",
        "\\definecolor{tu05}{HTML}{4B6CFC}",
        "\\definecolor{chartreuse(traditional)}{rgb}{0.87, 1.0, 0.0}"
    ], "\n"))
    
    PGFPlots.save(output_path, plot)
end

function _mapping_names(name)
    if name == "proportional_estimate_B"
        L"$\mathrm{proportional}_{\mathbb{E}_{B}}$"
    elseif name == "proportional_estimate_C"
        L"$\mathrm{proportional}_{\mathbb{E}_{C}}$"
    elseif name == "proportional_estimate_A"
        L"$\mathrm{proportional}_{\mathbb{E}_{A}}$"
    elseif name == "certification_A_low"
        L"$\mathrm{certification}_{\mathbb{E}_{A}}^{\sigma_{low}}$"
    elseif name == "certification_A_high"
        L"$\mathrm{certification}_{\mathbb{E}_{A}}^{\sigma_{high}}$"
    elseif name == "certification_B_low"
        L"$\mathrm{certification}_{\mathbb{E}_{B}}^{\sigma_{low}}$"
    elseif name == "certification_B_high"
        L"$\mathrm{certification}_{\mathbb{E}_{B}}^{\sigma_{high}}$"
    elseif name == "certification_C_low"
        L"$\mathrm{certification}_{\mathbb{E}_{C}}^{\sigma_{low}}$"
    elseif name == "certification_C_high"
        L"$\mathrm{certification}_{\mathbb{E}_{C}}^{\sigma_{high}}$"
    else
        name
    end
end