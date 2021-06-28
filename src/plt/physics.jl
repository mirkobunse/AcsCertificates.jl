"""
    physics(results_path, output_path)

Generate a LaTeX table from the results of the astro-particle physics experiment.
"""
function physics(results_path, output_path)
    df = CSV.read(results_path, DataFrame)

    # estimate the true loss on the target domain
    df[!, :L] = (1 .- df[!, :p_T]) .* df[!, :L_hadron] .+ df[!, :p_T] .* df[!, :L_gamma]

    # aggregate all relevant quantities
    gid = [:delta]
    agg = combine(
        groupby(df, gid),
        :sigma => StatsBase.mean => :sigma, # averages
        :epsilon => StatsBase.mean => :epsilon,
        :L => StatsBase.mean => :L,
        :sigma => StatsBase.std => :sigma_std, # standard deviations
        :L => StatsBase.std => :L_std,
    )

    df_latex = DataFrame(
        Symbol("significance of detection [\$\\sigma\$]") => _format(agg[!, :sigma], agg[!, :sigma_std]),
        Symbol("\$L_{\\mathcal S}(h)\$") => _format(agg[!, :L], agg[!, :L_std]),
        Symbol("\$\\delta\$") => detokenize.(agg[!, :delta]),
        Symbol("\$\\epsilon_\\delta\$") => detokenize.(agg[!, :epsilon]),
    ) # LaTeX-ready DataFrame

    # log and export the aggregation
    @info "Exporting a LaTeX table" output_path df_latex
    open(output_path, "w") do io
        println(io, "\\documentclass{standalone}")
        println(io, "\\usepackage{booktabs,multirow,amsmath,amssymb}")
        println(io, "\\begin{document}\n")
        println(io, "\\begin{tabular}{$(repeat("c", size(df_latex, 2)))}")
        println(io, "  \\toprule")
        println(io, "    ", join(names(df_latex), " & "), " \\\\") # header
        println(io, "  \\midrule")
        println(io, "    \\multirow{4}{*}{$(df_latex[1, 1])} & ",
            "\\multirow{4}{*}{$(df_latex[1, 2])} & ",
            df_latex[1, 3], " & ",
            df_latex[1, 4], "\\\\\n    & & ",
            df_latex[2, 3], " & ",
            df_latex[2, 4], "\\\\\n    & & ",
            df_latex[3, 3], " & ",
            df_latex[3, 4], "\\\\\n    & & ",
            df_latex[4, 3], " & ",
            df_latex[4, 4], "\\\\"
        )
        println(io, "  \\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\n\\end{document}")
    end
    return nothing
end

_format(μ::AbstractVector{T}, σ::AbstractVector{T}) where T <: Real = map(_format, zip(μ, σ))
_format(t::Tuple{T,T}) where T <: Real = _format(t[1], t[2])
_format(μ::Real, σ::Real) = "\$$(round(μ; digits=3)) \\pm $(round(σ; digits=3))\$"
