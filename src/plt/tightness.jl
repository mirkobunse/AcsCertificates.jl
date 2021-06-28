"""
    tightness(crt_path, val_path, tst_path, output_path)

Generate LaTeX plots and tables from the results of the tightness experiment.
"""
function tightness(crt_path, val_path, tst_path, output_path)
    df_crt = CSV.read(crt_path, DataFrame) # read all inputs
    df_val = CSV.read(val_path, DataFrame)
    df_tst = CSV.read(tst_path, DataFrame)

    # aggregate all relevant quantities
    gid = [:loss, :delta, :data, :clf, :weight] # basic group id
    max_p = _mean_std(df_crt, vcat(gid, :epsilon), :max_p)
    pY_val= _mean_std(df_val, gid, :pY)
    agg_val = _agg_L_ϵ(df_val, gid)
    agg_tst = _agg_L_ϵ(df_tst, vcat(gid, :pY))

    # set up TikzDocument with one page per group plot
    resetPGFPlotsPreamble()
    pushPGFPlotsPreamble("\\usepackage{amsmath,amssymb,booktabs,hyperref}")
    pushPGFPlotsPreamble("\\usepackage[paperheight=14in, paperwidth=7in, margin=1in]{geometry}")
    pushPGFPlotsPreamble("\\usetikzlibrary{calc}")
    pushPGFPlotsPreamble("\\usepgfplotslibrary{hvlines}")

    # the selection of plots for the paper
    paper_axes = Array{Axis}(undef, 2) # empty array of length 2

    # statistics
    n_coords = 0 # number of plot coordinates in all plots
    n_failures = 0 # number of coordinates where the certificate fails
    absolute_errors = Float64[] # collect absolute errors to compute the MAE
    absolute_errors_above = Float64[] # MAE for p_T >= p_S

    # one page per combination of loss and delta
    document = TikzDocument()
    agg_tst[!,:is_fact] = [ d ∈ ["fact_balanced", "fact_imbalanced"] for d in agg_tst[!,:data] ]
    for (key_f, agg_f) in pairs(groupby(sort(agg_tst, :is_fact), :is_fact))
        repo = key_f.is_fact ? "FACT" : "imblearn"
        i_x = Dict(name => i for (i, name) in enumerate(unique(agg_f[!, :clf])))
        i_y = Dict(name => i for (i, name) in enumerate(unique(agg_f[!, :data])))
        for (key_ldw, agg_ldw) in pairs(groupby(agg_f, [:loss, :delta, :weight]))

            # one group plot with all combinations of data and clf
            groupplot_axes = Array{Axis}(undef, length(i_x), length(i_y)) # empty matrix
            for (key_dc, agg_dc) in pairs(groupby(agg_ldw, [:data, :clf]))
                sdf_pY_val = _mean_pY_val(pY_val, agg_dc, gid)
                axis = Axis(; style = join([
                    "title={$(key_ldw.weight) $(key_ldw.loss),\\\\$(_clf(key_dc.clf)) on $(detokenize(key_dc.data))}",
                    "title style={font=\\scriptsize, text width=.4\\linewidth, align=center}",
                    "xlabel={\$p_\\mathcal{T}\$}",
                    "ylabel={\$L_\\mathcal{T}(h)\$}",
                    "xlabel style={font=\\footnotesize}",
                    "ylabel style={font=\\footnotesize}",
                    "xmode=log",
                    "scale=.475",
                    "vertical line={at=$(sdf_pY_val), style={gray, very thin}}",
                    "legend cell align={left}",
                    "legend style={font=\\footnotesize}",
                    "yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}",
                    "scaled y ticks=false"
                ], ", "))
                push!(axis, PGFPlots.Plots.Linear(
                    agg_dc[!,:pY],
                    agg_dc[!,:mean_L];
                    style = "semithick, densely dotted, gray, mark=*, mark options={scale=.8, solid, fill=gray}"
                )) # L_T
                target_domain_bound = agg_dc[!,:mean_L] .+ agg_dc[!,:mean_ϵ]
                push!(axis, PGFPlots.Plots.Linear(
                    agg_dc[!,:pY],
                    target_domain_bound;
                    style = "semithick, dashed, blue, mark=triangle*, mark options={scale=1, solid, fill=blue}"
                )) # L_T + ϵ_T
                max_p_y = _max_p_y(max_p, pY_val, agg_val, agg_dc, gid) # prediction of certificates
                push!(axis, PGFPlots.Plots.Linear(
                    agg_dc[!,:pY],
                    max_p_y;
                    style = "semithick, solid, orange, mark=square*, mark options={scale=.8, solid, fill=orange}"
                )) # certificates
                n_coords += length(max_p_y)
                n_failures += sum(max_p_y .< agg_dc[!,:mean_L]) # such a failure should not happen often
                absolute_errors_axis = abs.(max_p_y .- (target_domain_bound)) # all absolute errors on this axis
                append!(absolute_errors, absolute_errors_axis)
                append!(absolute_errors_above, absolute_errors_axis[agg_dc[!,:pY] .>= sdf_pY_val]) # p_T >= p_S
                groupplot_axes[i_x[key_dc.clf], i_y[key_dc.data]] = axis # add the axis to the current groupplot
                if key_ldw.weight=="uniform" && key_ldw.loss=="L2DistLoss" && key_ldw.delta==.05 && key_dc.data=="coil_2000" && key_dc.clf=="sklearn.linear_model.LogisticRegression"
                    paper_axes[1] = axis
                elseif key_ldw.weight=="sqrt" && key_ldw.loss=="ZeroOneLoss" && key_ldw.delta==.05 && key_dc.data=="letter_img" && key_dc.clf=="sklearn.tree.DecisionTreeClassifier"
                    paper_axes[2] = axis
                end # might want to add it to the paper_axes, too
            end
            groupplot = GroupPlot(
                length(i_x), length(i_y);
                groupStyle = "horizontal sep=18mm, vertical sep=21mm"
            )
            append!(groupplot, reshape(groupplot_axes, length(groupplot_axes))) # flatten
            push!(
                document,
                PGFPlots.plot(groupplot);
                caption = "$(repo): $(key_ldw.weight) $(key_ldw.loss) with \$\\delta = $(key_ldw.delta)\$"
            ) # add one page with the current groupplot
        end
    end
    paper_groupplot = GroupPlot(2, 1; groupStyle = "horizontal sep=18mm")
    append!(paper_groupplot, paper_axes)
    push!(document, PGFPlots.plot(paper_groupplot); caption = "Fig. 4 in our paper")

    # export the plots as a temporary .tex file
    tmp_path = tempname() * ".tex" # temporary path
    save(TEX(tmp_path), document) # .tex export

    # patch the temporary file with the table
    @info "Plotting to $output_path"
    open(output_path, "w") do io
        for l in readlines(tmp_path) # copy line by line
            if occursin("\\end{document}", l)
                _format_tables(io, max_p, pY_val, agg_val, agg_tst, gid)
            end # need to add something after?
            println(io, l)
            if occursin("\\begin{document}", l)
                _format_general_statistics(io, n_coords, n_failures, size(df_crt, 1), mean(absolute_errors), mean(absolute_errors_above))
            end # need to add something before?
        end
    end
    return nothing
end

function _format_general_statistics(io, n_coords, n_failures, n_certificates, mae, mae_above)
    println(io, "\\listoffigures")
    println(io, "\\listoftables")
    println(io, "\\begin{table}[!b]")
    println(io, "  \\caption{General statistics}")
    println(io, "  \\centering")
    println(io, "  \\scriptsize")
    println(io, "  \\begin{tabular}{rl}")
    println(io, "    \\toprule")
    println(io, "      metric & value \\\\") # header
    println(io, "    \\midrule")
    println(io, "      number of certificates & $(n_certificates) \\\\")
    println(io, "      number of plot coordinates & $(n_coords) \\\\")
    println(io, "      number of failures & $(n_failures) \\\\")
    println(io, "      fraction of failures & $(n_failures / n_coords) \\\\")
    println(io, "      mean absolute error (all \$p_{\\mathcal T}\$) & $(mae) \\\\")
    println(io, "      mean absolute error (\$p_{\\mathcal T} \\geq p_{\\mathcal S}\$) & $(mae_above) \\\\")
    println(io, "    \\bottomrule")
    println(io, "  \\end{tabular}")
    println(io, "\\end{table}\n")
end

function _format_tables(io, max_p, pY_val, agg_val, agg_tst, gid)
    n_tables = 0 # count tables to specify page breaks
    for (key_ϵ, max_p_ϵ) in pairs(groupby(max_p, [:epsilon]))
        for (key_ldw, agg_ldw) in pairs(groupby(agg_val, [:loss, :delta, :weight]))
            in_paper = key_ldw.weight=="sqrt" && key_ldw.loss=="ZeroOneLoss" && key_ldw.delta==.05 && key_ϵ.epsilon==0.01
            println(io, "\\begin{table}")
            println(io,
                "  \\caption{",
                "$(key_ldw.weight) $(key_ldw.loss) with ",
                "\$\\delta = $(key_ldw.delta), \\epsilon = $(key_ϵ.epsilon)\$",
                in_paper ? " (Tab. 1 in our paper)}" : "}"
            )
            println(io, "  \\centering")
            println(io, "  \\scriptsize")
            println(io, "  \\begin{tabular}{rrlll}")
            println(io, "    \\toprule")
            println(io,
                "      ",
                join([
                    "data",
                    "classifier",
                    "\$L_{\\mathcal S}(h)\$",
                    "\$p_{\\mathcal S}\$",
                    "\$\\Delta p^\\ast\$"
                ], " & "),
                " \\\\"
            ) # header
            println(io, "    \\midrule")
            for r in eachrow(sort(agg_ldw, [:data, :weight, :clf]))
                pY = semijoin(pY_val, DataFrame(r), on=gid)[1, :mean]
                Δp = semijoin(max_p_ϵ, DataFrame(r), on=gid)[1, :mean]
                println(io,
                    "      ",
                    detokenize(r[:data]), " & ",
                    _clf(r[:clf]), " & ",
                    round(r[:mean_L] + r[:mean_ϵ]; digits=4), " & ",
                    round(pY; digits=4), " & ",
                    round(Δp; digits=4), " \\\\"
                )
            end
            println(io, "    \\bottomrule")
            println(io, "  \\end{tabular}")
            println(io, "\\end{table}\n")

            n_tables += 1
            if (n_tables % 4) == 0
                println(io, "\\clearpage\n")
            end # add a page break for every four tables
        end
    end
end

_clf(x) = replace(split(x, ".")[end], "Classifier" => "")

# join all aggregations for df_val and df_tst: {mean|std}_{L|ϵ}
_agg_L_ϵ(df, gid) =
    innerjoin(
        _mean_std(df, gid, :value), # L_{val|tst}
        _mean_std(df, gid, :epsilon), # ϵ_{val|tst}
        on=gid,
        renamecols="_L"=>"_ϵ"
    )

# mean and std of a column c in a DataFrame
_mean_std(df, gid, c) =
    combine(groupby(df, gid),
        c => StatsBase.mean => :mean,
        c => StatsBase.std => :std
    )

# fetch the mean pY_val value for a SubDataFrame sdf
function _mean_pY_val(pY_val, sdf, gid)
    sdf_pY_val = semijoin(pY_val, sdf, on=gid) # match pY_val to sdf
    if nrow(sdf_pY_val) > 1
        @warn "Semi-join yields more than one result"
    end # should never happen
    return sdf_pY_val[1, :mean]
end

# resample epsilon values for max_p mean values
function _max_p_y(max_p, pY_val, agg_val, sdf, gid)

    # compute the slope of the certificate line
    sdf_max_p = semijoin(max_p, sdf, on=gid) # match max_p to sdf
    slope = mean(sdf_max_p[!, :epsilon] ./ sdf_max_p[!, :mean]) # Δℓ*, actually equal for all rows

    sdf_pY_val = semijoin(pY_val, sdf, on=gid) # match pY_val to sdf
    x = sdf[!, :pY] # x coordinates in the plot
    y = slope .* abs.(x .- sdf_pY_val[1, :mean]) # y coordinates

    sdf_agg_val = semijoin(agg_val, sdf, on=gid)
    if nrow(sdf_agg_val) > 1
        @warn "Semi-join yields more than one result"
    end # should never happen
    y .+= sdf_agg_val[1, :mean_L] + sdf_agg_val[1, :mean_ϵ]
    return y
end
