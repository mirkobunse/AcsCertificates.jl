module Certificates

using
    ..AcsCertificates,
    JuMP,
    NLopt,
    LinearAlgebra,
    LossFunctions,
    SpecialFunctions

export
    beta_parameters,
    Certificate,
    empirical_classwise_risk,
    is_feasible,
    is_onesided,
    max_Δp,
    optimize_Δℓ,
    p_range,
    suggest_acquisition

const BINARY_LABELS = [-1, 1]

# dispatch on the optimization objective wrt Δℓ
abstract type Δℓ_Objective end
struct Δℓ_MinMax <: Δℓ_Objective end
struct Δℓ_MaxMin <: Δℓ_Objective end
struct OneSided_Δℓ_MinMax <: Δℓ_Objective end
onesided(::Type{Δℓ_MinMax}) = OneSided_Δℓ_MinMax
onesided(::Type{T}) where {T<:Δℓ_Objective} =
    throw(ArgumentError("No one-sided version for $T"))

# can be used as any Real number, but stores additional information
struct Δℓ_Result{T <: Δℓ_Objective} <: Real
    ϵ_y::Vector{Float64}
    δ_y::Vector{Float64}
    empirical_ℓ_y::Vector{Float64}
    w_y::Vector{Float64}
    m_y::Vector{Int}
    L::SupervisedLoss
end
result_objective(r::Δℓ_Result{T}) where {T<:Δℓ_Objective} = T
is_feasible(r::Δℓ_Result{T}; kwargs...) where {T<:Δℓ_Objective} =
    is_feasible(T, r.ϵ_y, r.empirical_ℓ_y; w_y=r.w_y, kwargs...)
function is_feasible(
        ::Type{Δℓ_MinMax},
        ϵ_y::Vector{Float64},
        empirical_ℓ_y::Vector{Float64};
        w_y::Vector{Float64}=[1., 1.],
        tol::Float64=1e-4)
    w_ℓ_y = w_y .* empirical_ℓ_y
    o = sortperm(w_ℓ_y) # the order = [1, 2] or [2, 1]
    return w_ℓ_y[o[1]]-ϵ_y[o[1]] > tol # estimate of true ℓ_lower > tol ?
end
function is_feasible(
        ::Type{Δℓ_MaxMin},
        ϵ_y::Vector{Float64},
        empirical_ℓ_y::Vector{Float64};
        w_y::Vector{Float64}=[1., 1.],
        tol::Float64=1e-4)
    w_ℓ_y = w_y .* empirical_ℓ_y
    o = sortperm(w_ℓ_y) # the order = [1, 2] or [2, 1]
    return w_ℓ_y[o[1]]+ϵ_y[o[1]]-tol < w_ℓ_y[o[2]]-ϵ[o[2]] # ℓ_lower-tol < ℓ_upper ?
end
is_feasible(::Type{OneSided_Δℓ_MinMax}, args...; kwargs...) = true
effective_δ(r::Δℓ_Result) = effective_δ(r.δ_y)
effective_δ(δ_y::Vector{Float64}) = δ_y[1] + δ_y[2] - δ_y[1]*δ_y[2]
value(r::Δℓ_Result{T}) where {T<:Δℓ_Objective} =
    is_feasible(r) ? value(T, r.ϵ_y, r.empirical_ℓ_y, r.w_y) : NaN
function value(::Type{Δℓ_MinMax}, ϵ_y::Vector{Float64}, empirical_ℓ_y::Vector{Float64}, w_y::Vector{Float64})
    w_ℓ_y = w_y .* empirical_ℓ_y
    o = sortperm(w_ℓ_y) # the order = [1, 2] or [2, 1]
    return w_ℓ_y[o[2]]+ϵ_y[o[2]] - w_ℓ_y[o[1]]+ϵ_y[o[1]] # minimum upper bound
end
function value(::Type{Δℓ_MaxMin}, ϵ_y::Vector{Float64}, empirical_ℓ_y::Vector{Float64}, w_y::Vector{Float64})
    w_ℓ_y = w_y .* empirical_ℓ_y
    o = sortperm(w_ℓ_y) # the order = [1, 2] or [2, 1]
    return w_ℓ_y[o[2]]-ϵ_y[o[2]] - w_ℓ_y[o[1]]-ϵ_y[o[1]] # maximum lower bound
end
function value(::Type{OneSided_Δℓ_MinMax}, ϵ_y::Vector{Float64}, empirical_ℓ_y::Vector{Float64}, w_y::Vector{Float64})
    w_ℓ_y = w_y .* empirical_ℓ_y
    o = sortperm(w_ℓ_y) # the order = [1, 2] or [2, 1]
    return w_ℓ_y[o[2]]+ϵ_y[o[2]] # one-sided minimum upper bound
end
objective_value(r::Δℓ_Result{T}) where {T<:Δℓ_Objective} = objective_value(T, r.ϵ_y)
objective_value(::Type{Δℓ_MinMax}, ϵ_y::Vector{Float64}) = sum(ϵ_y)
objective_value(::Type{Δℓ_MaxMin}, ϵ_y::Vector{Float64}) = sum(ϵ_y)
objective_value(::Type{OneSided_Δℓ_MinMax}, ϵ_y::Vector{Float64}) = maximum(ϵ_y[isfinite.(ϵ_y)])
Base.convert(::Type{T}, r::Δℓ_Result) where {Float64<:T<:Real} = value(r)
Base.promote_rule(::Type{Δℓ_Result}, ::Type{T}) where {Float64<:T<:Real} = T

"""
    Certificate(L, y_h, y; kwargs...)

A certificate about the robustness of an hypothesis `h` with respect to changes
in the class proportions.

You can inspect this certificate using the *Methods* listed below. It is based
on the predictions `y_h` and ground-truth class labels `y ∈ {-1, 1}` and holds
for the loss function `L` with probability at least `1 - δ`.

### Keyword Arguments

- `δ = 0.05`
- `w_y = [1., 1.]` optional class weights
- `tol = 1e-4` the tolerance, `tol > 0` for the constrained optimization of Δℓ
- `n_trials = 3` number of trials in the multi-start global optimization of Δℓ
- `verbose = false` whether to log additional information to the console
- `warn = true` whether to log warnings to the console
- `m = length(y)` can be used to simulate other data set sizes (discouraged)

### Methods (more detail in their documentation)

- `p_range(c, ϵ=0.05, label=+1)` the range of feasible class proportions
- `max_Δp(c, ϵ=0.05)` the largest feasible distance `Δp = |p_S - p_T|`
"""
struct Certificate{T<:Union{Δℓ_MinMax,OneSided_Δℓ_MinMax}}
    Δℓ::Δℓ_Result{T} # everything can be derived from this result; see Base.show below
end
is_onesided(c::Certificate{Δℓ_MinMax}) = false
is_onesided(c::Certificate{OneSided_Δℓ_MinMax}) = true

Certificate(
        L::SupervisedLoss,
        y_h::AbstractVector{R},
        y::AbstractVector{I};
        kwargs...) where {R<:Real, I<:Integer} =
    Certificate(optimize_Δℓ(L, y_h, y; kwargs...))

function Base.show(io::IO, ::MIME"text/plain", c::Certificate)
    if is_feasible(c.Δℓ)
        println(io, "┌ Certificate on label shift robustness:")
        println(io, "│ * p_range = ", p_range(c), " at ϵ=0.05")
        println(io, "│ * max_Δp = ", max_Δp(c), " at ϵ=0.05")
    else
        println(io, "┌ No certification feasible; increase the amount of data or the value of δ!")
    end
    println(io, "│ * effective_δ = ",
        round(effective_δ(c.Δℓ); digits=4), " (δ_y = ",
        round.(c.Δℓ.δ_y; digits=4), ")")
    println(io, "│ * Δℓ_minmax = ",
        round(value(c.Δℓ); digits=4), " (ϵ_y = ",
        round.(c.Δℓ.ϵ_y; digits=4), ")")
    println(io, "└─┬ Given the following scenario:")
    println(io, "  │ * empirical_ℓ_y = ",
        round.(c.Δℓ.empirical_ℓ_y; digits=6))
    println(io, "  │ * w_y = ",
        round.(c.Δℓ.w_y; digits=6))
    println(io, "  │ * m_y = ", c.Δℓ.m_y)
    println(io, "  └ * L = ", c.Δℓ.L)
end

"""
    p_range(c, ϵ=0.05, label=+1)

The maximum range of values for `p_T` for which the hypothesis certified
by `c` has an ACS-induced error of less than `ϵ`. Namely, this method
returns `(p_S - max_Δp, p_S + max_Δp)`.
"""
function p_range(c::Certificate, ϵ::Float64=0.05, label::Int=BINARY_LABELS[2])
    if label ∉ BINARY_LABELS
        throw(ArgumentError("Only binary labels $BINARY_LABELS are supported"))
    end
    p_S = c.Δℓ.m_y[2] / sum(c.Δℓ.m_y)
    if label != BINARY_LABELS[2]
        p_S = 1-p_S # use the other label as the reference point
    end
    Δp = max_Δp(c, ϵ)
    return max(p_S - Δp, 0.0), min(p_S + Δp, 1.0)
end

"""
    max_Δp(c, ϵ=0.05)

The maximum `Δp = |p_S - p_T|` for which the hypothesis certified by `c`
has an ACS-induced error of less than `ϵ`. Namely,

    | L_T(h) - L_S(h) | < ϵ

holds with probability at least `1-δ`, where `L_D(h)` is the true loss of
`h` in the domain `D` under the loss function `L`.

"""
max_Δp(c::Certificate, ϵ::Float64=0.05) = ϵ / c.Δℓ

"""
    is_feasible(c)

Does `c` represent a feasible certificate?
"""
is_feasible(c::Certificate) = is_feasible(c.Δℓ)

"""
    optimize_Δℓ(L, y_h, y; kwargs...)

Compute the minimum upper bound of `Δℓ = |ℓ_1 - ℓ_2|`, which the hypothesis
`h` can achieve with probability at least `1 - δ`, using the predictions
`y_h` and ground-truth class labels `y ∈ {-1, 1}`.

### Keyword Arguments

- `δ = 0.05`
- `w_y = [1., 1.]` optional class weights
- `tol = 1e-4` the tolerance, `tol > 0` for the constrained optimization of Δℓ
- `n_trials = 3` number of trials in the multi-start global optimization of Δℓ
- `verbose = false` whether to log additional information to the console
- `warn = true` whether to log warnings to the console
- `m = length(y)` can be used to simulate other data set sizes (discouraged)
"""
optimize_Δℓ(
        L::SupervisedLoss,
        y_h::AbstractVector{R},
        y::AbstractVector{I};
        kwargs...) where {R<:Real, I<:Integer} =
    optimize_Δℓ(Δℓ_MinMax, L, y_h, y; kwargs...)

# this general implementation is also able to find maximum lower bounds
function optimize_Δℓ(
        objective::Type{T},
        L::SupervisedLoss,
        y_h::AbstractVector{R},
        y::AbstractVector{I};
        δ::Float64=5e-2,
        w_y::Vector{Float64}=[1., 1.],
        tol::Float64=1e-4,
        n_trials::Int=3,
        n_trials_extra::Int=0,
        allow_onesided::Bool=true,
        m::Int=length(y),
        verbose::Bool=false,
        warn::Bool=true) where {T<:Δℓ_Objective, R<:Real, I<:Integer}
    _check_labels(y)
    m_y = convert.(Int, round.(_m_y(y) .* (m/length(y)))) # number of instances in each class
    empirical_ℓ_y = min.(1-tol, max.(tol, empirical_classwise_risk(L, y_h, y)))

    # do we need to re-order the classes so that ℓ_2 > ℓ_1?
    reorder = w_y[2] * empirical_ℓ_y[2] < w_y[1] * empirical_ℓ_y[1]
    if reorder
        w_y = w_y[[2, 1]]
        m_y = m_y[[2, 1]]
        empirical_ℓ_y = empirical_ℓ_y[[2, 1]]
    end

    # multi-start global optimization#
    args = [ L, empirical_ℓ_y, w_y, m_y, δ, tol, warn ]
    Δℓ = [ _optimize_Δℓ(T, args...) for _ in 1:n_trials ]
    best_Δℓ = Δℓ[argmin(objective_value.(Δℓ))]

    # retry with a one-sided estimate, which never fails but is always worse than a two-sided one
    if allow_onesided && (!is_feasible(best_Δℓ) || effective_δ(best_Δℓ)-tol > δ)
        Δℓ = [ _optimize_Δℓ(onesided(T), args...) for _ in 1:n_trials ]
        best_Δℓ = Δℓ[argmin(objective_value.(Δℓ))]

    # alternative: extra trials for two-sided estimates (the Inf check is less strict than is_feasible)
    elseif !allow_onesided && any(best_Δℓ.ϵ_y .== Inf) && n_trials_extra > 0
        Δℓ = vcat(Δℓ, [ _optimize_Δℓ(T, args...) for _ in 1:n_trials_extra ])
        best_Δℓ = Δℓ[argmin(objective_value.(Δℓ))]
    end

    # debugging information
    if verbose
        feasible = is_feasible.(Δℓ)
        percent_feasible = round(100 * sum(feasible) / n_trials; digits=1)
        percent_close_to_optimum = round(
            100 * sum(isapprox.(Δℓ[feasible], best_Δℓ; atol=tol)) / sum(feasible); digits=1)
        @info "optimize_Δℓ" is_onesided(best_Δℓ) percent_feasible percent_close_to_optimum
    end

    if reorder
        best_Δℓ = typeof(best_Δℓ)(
            best_Δℓ.ϵ_y[[2, 1]],
            best_Δℓ.δ_y[[2, 1]],
            best_Δℓ.empirical_ℓ_y[[2, 1]],
            best_Δℓ.w_y[[2, 1]],
            best_Δℓ.m_y[[2, 1]],
            best_Δℓ.L
        ) # change classes back to their original order
    end
    return best_Δℓ
end

# single-start global optimization repeatedly called for Δℓ_MinMax
function _optimize_Δℓ(
        ::Type{Δℓ_MinMax},
        L::SupervisedLoss,
        empirical_ℓ_y::Vector{Float64},
        w_y::Vector{Float64},
        m_y::Vector{Int},
        δ::Float64,
        tol::Float64,
        warn::Bool)
    w_ℓ_y = w_y .* empirical_ℓ_y # weighted loss

    # helper functions
    δ_1(ϵ_1::Real, ϵ_2::Real) =
        exp(-2*m_y[1]*ϵ_1^2) + exp(-2*m_y[1]*(w_ℓ_y[2]-w_ℓ_y[1]+ϵ_2)^2)
    δ_2(ϵ_1::Real, ϵ_2::Real) =
        exp(-2*m_y[2]*ϵ_2^2) + exp(-2*m_y[2]*(w_ℓ_y[2]-w_ℓ_y[1]+ϵ_1)^2)

    # define the JuMP optimization problem; use the ISRES optimizer
    # https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#isres-improved-stochastic-ranking-evolution-strategy
    model = Model(NLopt.Optimizer)
    register(model, :δ_1, 2, δ_1, autodiff=true)
    register(model, :δ_2, 2, δ_2, autodiff=true)
    set_optimizer_attribute(model, "algorithm", :GN_ISRES) # :LD_MMA, :LD_SLSQP
    set_optimizer_attribute(model, "maxtime", 2.0)
    @variable(model, tol <= ϵ_1 <= 1/tol) # w_ℓ_y[1]-tol
    @variable(model, tol <= ϵ_2 <= 1/tol)
    @NLconstraint(model, δ - (δ_1(ϵ_1, ϵ_2) + δ_2(ϵ_1, ϵ_2) - δ_1(ϵ_1, ϵ_2)*δ_2(ϵ_1, ϵ_2)) >= tol)
    @NLobjective(model, Min, ϵ_1 + ϵ_2)

    # optimize from a random starting point
    ϵ_y = [ Inf, Inf ] # the default for failures of JuMP.optimize!
    set_start_value(ϵ_1, 2*tol + rand() * (w_ℓ_y[1] - 4*tol))
    set_start_value(ϵ_2, 2*tol + rand())
    try
        JuMP.optimize!(model)
        if JuMP.raw_status(model) == "FORCED_STOP"
            throw(InterruptException())
        end
        ϵ_y = [ JuMP.value(ϵ_1), JuMP.value(ϵ_2) ]
    catch some_error
        if isa(some_error, ArgumentError)
            if warn
                @warn some_error.msg
            end
        else
            rethrow()
        end # ignore ArgumentError; likely no feasible region due to tol
    end

    # assemble the result
    δ_y = [ δ_1(ϵ_y[1], ϵ_y[2]), δ_2(ϵ_y[1], ϵ_y[2]) ]
    return Δℓ_Result{Δℓ_MinMax}(ϵ_y, δ_y, empirical_ℓ_y, w_y, m_y, L)
end

# single-start global optimization repeatedly called for OneSided_Δℓ_MinMax
function _optimize_Δℓ(
        ::Type{OneSided_Δℓ_MinMax},
        L::SupervisedLoss,
        empirical_ℓ_y::Vector{Float64},
        w_y::Vector{Float64},
        m_y::Vector{Int},
        δ::Float64,
        tol::Float64,
        warn::Bool)
    w_ℓ_y = w_y .* empirical_ℓ_y # weighted loss

    # helper functions (one-sided probabilities)
    δ_1(ϵ_2::Real) = exp(-2*m_y[1]*(w_ℓ_y[2]-w_ℓ_y[1]+ϵ_2)^2)
    δ_2(ϵ_2::Real) = exp(-2*m_y[2]*ϵ_2^2)

    # define the JuMP optimization problem; use the ISRES optimizer
    # https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#isres-improved-stochastic-ranking-evolution-strategy
    model = Model(NLopt.Optimizer)
    register(model, :δ_1, 1, δ_1, autodiff=true)
    register(model, :δ_2, 1, δ_2, autodiff=true)
    set_optimizer_attribute(model, "algorithm", :GN_ISRES) # :LD_MMA, :LD_SLSQP
    set_optimizer_attribute(model, "maxtime", 2.0)
    @variable(model, tol <= ϵ_2 <= 1/tol)
    @NLconstraint(model, δ - (δ_1(ϵ_2) + δ_2(ϵ_2) - δ_1(ϵ_2)*δ_2(ϵ_2)) >= tol)
    @NLobjective(model, Min, ϵ_2)

    # optimize from a random starting point
    set_start_value(ϵ_2, 2*tol + rand())
    JuMP.optimize!(model) # one-sided estimation should never fail
    ϵ_y = [ NaN, JuMP.value(ϵ_2) ]

    # assemble the result
    δ_y = [ δ_1(ϵ_y[2]), δ_2(ϵ_y[2]) ]
    return Δℓ_Result{OneSided_Δℓ_MinMax}(ϵ_y, δ_y, empirical_ℓ_y, w_y, m_y, L)
end

# single-start global optimization repeatedly called for Δℓ_MaxMin
function _optimize_Δℓ(
        ::Type{Δℓ_MaxMin},
        L::SupervisedLoss,
        empirical_ℓ_y::Vector{Float64},
        w_y::Vector{Float64},
        m_y::Vector{Int},
        δ::Float64,
        tol::Float64,
        warn::Bool)
    w_ℓ_y = w_y .* empirical_ℓ_y # weighted loss

    # helper functions
    δ_1(ϵ_1::Real, ϵ_2::Real) = exp(-2*m_y[1]*ϵ_1^2)
    δ_2(ϵ_1::Real, ϵ_2::Real) = exp(-2*m_y[2]*ϵ_2^2)
    w_ℓ_diff = w_ℓ_y[2] - w_ℓ_y[1]

    # define the JuMP optimization problem; use the ISRES optimizer
    # https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#isres-improved-stochastic-ranking-evolution-strategy
    model = Model(NLopt.Optimizer)
    register(model, :δ_1, 2, δ_1, autodiff=true)
    register(model, :δ_2, 2, δ_2, autodiff=true)
    set_optimizer_attribute(model, "algorithm", :GN_ISRES) # :LD_MMA, :LD_SLSQP
    set_optimizer_attribute(model, "maxtime", 2.0)
    @variable(model, tol <= ϵ_1 <= w_ℓ_diff-tol)
    @variable(model, tol <= ϵ_2 <= w_ℓ_diff-tol)
    @NLconstraint(model, δ - (δ_1(ϵ_1, ϵ_2) + δ_2(ϵ_1, ϵ_2) - δ_1(ϵ_1, ϵ_2)*δ_2(ϵ_1, ϵ_2)) >= tol)
    @NLconstraint(model, w_ℓ_diff - (ϵ_1 + ϵ_2) >= 2*tol)
    @NLobjective(model, Min, ϵ_1 + ϵ_2)

    # optimize from a random starting point
    ϵ_1_start = 2*tol + rand() * (w_ℓ_diff - 4*tol)
    set_start_value(ϵ_1, ϵ_1_start)
    set_start_value(ϵ_2, 2*tol + rand() * (w_ℓ_diff - ϵ_1_start - 2*tol))
    JuMP.optimize!(model)

    # check whether the solution is actually feasible
    ϵ_y = [ JuMP.value(ϵ_1), JuMP.value(ϵ_2) ]
    δ_y = [ δ_1(ϵ_y[1], ϵ_y[2]), δ_2(ϵ_y[1], ϵ_y[2]) ]
    return Δℓ_Result{Δℓ_MaxMin}(ϵ_y, δ_y, empirical_ℓ_y, w_y, m_y, L)
end

"""
    empirical_classwise_risk(ℓ, y_h, y)

The class-wise risk of predictions `y_h` under the loss function `ℓ`.
"""
empirical_classwise_risk(
        L::SupervisedLoss,
        y_h::AbstractVector{R},
        y::AbstractVector{I}) where {R<:Real, I<:Integer} =
    Float64[ LossFunctions.value(L, y[y.==y_i], y_h[y.==y_i], AggMode.Mean()) for y_i in BINARY_LABELS ]

"""
    suggest_acquisition(certificate, n, α, β)  ->  (m_1, m_2)

Suggest what data to acquire in the next batch, in terms of how the class-wise
sample size `(m_1, m_2)` should ideally change when `n = m_1 + m_2` samples are
being added and `α > 0` and `β > 0` are the parameters of a Beta prior.

The outputs `m_1` and `m_2` are allowed to be negative; a negative value amounts
to the suggestion that one of the two classes should be under-sampled or weighted
accordingly.

**See also:** `AcsCertificates.[Certificates.]beta_parameters`
"""
suggest_acquisition(c::Certificate, args...) = suggest_acquisition(c.Δℓ, args...)

function suggest_acquisition(Δℓ_minmax::Δℓ_Result{Δℓ_MinMax}, n::Integer, α::Real, β::Real)
    m_y = Δℓ_minmax.m_y
    δ_y = Δℓ_minmax.δ_y
    empirical_ℓ_y = Δℓ_minmax.empirical_ℓ_y
    w_ℓ_y = Δℓ_minmax.w_y .* empirical_ℓ_y

    # do we need to re-order the classes so that ℓ_2 > ℓ_1?
    reorder = w_ℓ_y[2] < w_ℓ_y[1]
    if reorder
        m_y = m_y[[2, 1]]
        δ_y = δ_y[[2, 1]]
        empirical_ℓ_y = empirical_ℓ_y[[2, 1]]
        w_ℓ_y = w_ℓ_y[[2, 1]]
        α, β = β, α
    elseif Δℓ_minmax.w_y != [1., 1.]
        @warn "Class weights not yet implemented"
    end

    # function and gradient of f(m) = ∫ P(p) * | p_S - p | dp
    p_S(m_1, m_2) = m_1 / (m_1 + m_2) # p_S as a function of m=(m_1, m_2)
    ∇p_S(m_1, m_2) = [m_2, -m_1] ./ (m_1 + m_2)^2 # gradient wrt m
    f(m_1, m_2) = +(
        2/(α+β) * (p_S(m_1, m_2)^α * (1-p_S(m_1, m_2))^β) / beta(α, β),
        (p_S(m_1, m_2) - α/(α+β)) * (2*beta_inc(α, β, p_S(m_1, m_2))[1] - 1)
    ) # just a multi-line addition
    ∂f(m_1, m_2) = 2*beta_inc(α, β, p_S(m_1, m_2))[1] - 1 # partial derivative ∂f / ∂p_S
    ∇f(m_1, m_2) = ∂f(m_1, m_2) .* ∇p_S(m_1, m_2) # gradient by chain rule: (∂f / ∂p_S) * (∂p_S / ∂m)

    # function and gradient of Δℓ(m); CAUTION: this is a simplification
    ϵ(m_1, m_2) = sqrt.(max.(0, -1/2 .* [ log(δ_y[1])/m_1, log(δ_y[2])/m_2 ])) # ϵ as a function of m
    Δℓ(m_1, m_2) = (empirical_ℓ_y[2] - empirical_ℓ_y[1]) + sum(ϵ(m_1, m_2))
    ∇Δℓ(m_1, m_2) = [
        max(0, -log(δ_y[1])/m_1)^(3/2) / (2*sqrt(2)*log(δ_y[1])),
        max(0, -log(δ_y[2])/m_2)^(3/2) / (2*sqrt(2)*log(δ_y[2]))
    ]

    # the negative gradient, as according to the product rule
    acq(m_1, m_2) = - (∇f(m_1, m_2) * Δℓ(m_1, m_2) + f(m_1, m_2) * ∇Δℓ(m_1, m_2))
    suggestion = acq(m_y...) # gradient at the current acquisition point

    # IMPORTANT: only use the negative gradient if we can actually detect a domain gap,
    # i.e. if Δℓ(m_y...) > 0; use a fall-back solution if the certificate is infeasible.
    # In our experiments, the fall-back is almost never used; increasing the n_trials
    # during certification would lower the chance of the fall-back even more.
    if Δℓ(m_y...) == 0 || all(suggestion .<= 0) || all(Δℓ_minmax.ϵ_y .== Inf)
        p_T = α/(α+β) # prior target proportions
        m_d = [p_T, 1-p_T] .* n # desired number of samples, as according to the prior
        suggestion = m_d - m_y # what to acquire (fall-back solution)
    else
        suggestion .*= n / sum(suggestion[suggestion.>0]) # normalize
    end

    # obsolete since the commit 1fdae9e7b1e23be5367dec7cf88c1ce02e4f89f7, but might
    # prove useful in very rare occasions or during future development
    if any(isnan.(suggestion)) # should never happen / last resort
        println("\n\n\n")
        @warn "NaNs in suggest_acquisition" acq(m_y...) ∇f(m_y...) Δℓ(m_y...) f(m_y...) ∇Δℓ(m_y...)
        Base.show(stdout, "text/plain", Certificate(Δℓ_minmax))
        p_T = α/(α+β) # same fall-back as above
        m_d = [p_T, 1-p_T] .* n
        suggestion = m_d - m_y
    end

    if reorder
        suggestion = suggestion[[2, 1]] # recover the original class order
    end
    return suggestion
end

# kind-of a dummy implementation: only request the class that prevents two-sided estimation
suggest_acquisition(Δℓ_minmax::Δℓ_Result{OneSided_Δℓ_MinMax}, n::Int, α::Real, β::Real) =
    throw(ArgumentError("One-sided certificate; try Certificate(L, y_h, y; allow_onesided=false"))

"""
    beta_parameters(μ, σ)

Define the scale parameters `(α, β)` of a Beta distribution via the mean `μ`
and standard deviation `σ`, the last of which is constrained by the variance
`σ^2 < μ*(1-μ)`.
"""
function beta_parameters(μ::Real, σ::Real)
    if σ^2 >= μ*(1-μ)
        throw(DomainError(σ, "σ^2 = $(σ^2) < μ*(1-μ) = $(μ*(1-μ)) is violated"))
    end
    ν = μ * (1-μ) / σ^2 - 1
    return (1-μ)*ν, μ*ν # return a tuple (α, β)
end

# assert that only the two labels +1 and -1 are present in the data
_check_labels(y::AbstractVector{I}) where I<:Integer =
    if sort(unique(y)) != BINARY_LABELS
        throw(ArgumentError("Only binary labels $BINARY_LABELS are supported"))
    end

 # number of instances in each class
_m_y(y::AbstractVector{I}) where {I<:Integer} =
    Int[ sum(y.==y_i) for y_i in BINARY_LABELS ]

end # module
