"""
    acquisition(results_path, output_path)

Generate a LaTeX table from the results of the acquisition experiment.
"""
function acquisition(results_path, output_path)
    df = CSV.read(results_path, DataFrame)

    # similarities between strategies
    df[!, :pY_trn] = df[!, :N_min] ./ (df[!, :N_min] .+ df[!, :N_maj])
    df[!, :kl_unif] = _kl.(df[!, :pY_trn], .5)
    df[!, :kl_prop] = _kl.(df[!, :pY_trn], df[!, :pY_tst])

    # average test error over all CV repetitions
    gid = [:batch, :clf, :weight, :loss, :delta, :epsilon]
    df = combine(
        groupby(df, vcat(gid, [:strategy, :pY_tst, :data])),
        :L_tst => StatsBase.mean => :L_tst,
        :kl_unif => StatsBase.mean => :kl_unif,
        :kl_prop => StatsBase.mean => :kl_prop
    )

    # select trials where all strategies were evaluated
    cnt = filter(
        :nrow => x -> x .== length(unique(df[:strategy])),
        combine(groupby(df, [:data, :batch, :pY_tst]), nrow)
    )
    df = semijoin(df, cnt, on=[:data, :batch, :pY_tst])

    # one CD diagram and one KL divergence diagram per pY_tst value
    groupplot = GroupPlot(2, 2; groupStyle="vertical sep=15mm, horizontal sep=50mm")
    for (key_pY, sdf_pY) in pairs(groupby(df, :pY_tst))

        # sequence of CD diagrams
        sequence = Pair{String, Vector{Pair{String, Vector}}}[]
        for (key, sdf) in pairs(groupby(sdf_pY, gid))
            n_data = length(unique(sdf[!, :data]))
            if key.batch ∉ 3:8
                continue
            end
            @info "Batch $(key.batch) on pY_tst=$(round(key_pY.pY_tst; digits=1)) is based on $(n_data) data sets"
            title = string(key.batch)
            pairs = CriticalDifferenceDiagrams._to_pairs(sdf, :strategy, :data, :L_tst)
            push!(sequence, title => pairs)
        end
        plot = CriticalDifferenceDiagrams.plot(sequence...)
        plot.style = join([
            "y dir=reverse",
            "ytick={1,2,3,4,5,6}",
            "yticklabels={3,4,5,6,7,8}",
            "ylabel={ACS batch}",
            "xlabel={avg. rank \\, (\$p_\\mathcal{T}=$(round(key_pY.pY_tst; digits=1))\$)}",
            "ylabel style={font=\\small}",
            "xlabel style={font=\\small}",
            "yticklabel style={font=\\small}",
            "xticklabel style={font=\\small}",
            "grid=both",
            "axis line style={draw=none}",
            "tick style={draw=none}",
            "xticklabel pos=upper",
            "xmin=.5",
            "xmax=5.5",
            "ymin=.75",
            "ymax=$(length(sequence)).75",
            "clip=false",
            "legend style={draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em}",
            "x dir=reverse",
            "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*}}",
            "width=\\axisdefaultwidth, height=\\axisdefaultheight"
        ], ", ")
        push!(groupplot, plot)

        # KL divergence to proportional sampling
        kl_plot = Axis(style = join([
            "title={KL divergence to \$p_\\mathcal{T}=$(round(key_pY.pY_tst; digits=1))\$}",
            "legend style={draw=none,fill=none,at={(1.1,.5)},anchor=west,row sep=.25em}",
            "cycle list={{tu01,mark=*},{tu02,mark=square*},{tu03,mark=triangle*},{tu04,mark=diamond*},{tu05,mark=pentagon*}}",
            "xtick={1,3,5,7}"
        ], ", "))
        for (key, sdf) in pairs(groupby(sdf_pY, vcat(setdiff(gid, [:batch]), [:strategy])))
            if key.strategy == "proportional"
                continue # KL divergence is usually zero, so that ymode=log does not work in general
            end
            sdf = combine(
                groupby(
                    sdf[sdf[!, :batch].<=8, :],
                    vcat(gid, [:strategy])
                ),
                :kl_prop => StatsBase.mean => :kl_prop,
                :kl_prop => StatsBase.std => :kl_prop_std
            )
            push!(kl_plot, PGFPlots.Plots.Linear(
                sdf[!, :batch],
                sdf[!, :kl_prop],
                legendentry=string(key.strategy),
                errorBars=PGFPlots.ErrorBars(y=sdf[!, :kl_prop_std])
            ))
        end
        push!(groupplot, kl_plot)
    end

    PGFPlots.resetPGFPlotsPreamble()
    PGFPlots.pushPGFPlotsPreamble(join([
        "\\definecolor{tu01}{HTML}{84B818}",
        "\\definecolor{tu02}{HTML}{D18B12}",
        "\\definecolor{tu03}{HTML}{1BB5B5}",
        "\\definecolor{tu04}{HTML}{F85A3E}",
        "\\definecolor{tu05}{HTML}{4B6CFC}"
    ], "\n"))
    PGFPlots.save(output_path, groupplot)
    return nothing
end

# scalar version for binary classification
_kl(p, q) = Distances.kl_divergence([p, 1-p], [q, 1-q])
