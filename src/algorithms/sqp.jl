"""
    AbstractSqpOptimizer

Abstract type of SQP solvers
"""
abstract type AbstractSqpOptimizer <: AbstractOptimizer end

macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end

@def sqp_fields begin
    problem::AbstractSqpModel # problem data

    x::TD # primal solution
    p::TD # search direction
    p_soc::TD # direction after SOC
    p_slack::Dict{Int,TD} # search direction at feasibility restoration phase
    lambda::TD # Lagrangian dual multiplier
    mult_x_L::TD # reduced cost for lower bound
    mult_x_U::TD # reduced cost for upper bound

    # Evaluations at `x`
    f::T # objective function
    df::TD # gradient
    E::TD # constraint evaluation
    dE::TD # Jacobian

    j_row::TI # Jacobian matrix row index
    j_col::TI # Jacobian matrix column index
    Jacobian::AbstractMatrix{T} # Jacobian matrix

    h_row::TI # Hessian matrix row index
    h_col::TI # Hessian matrix column index
    h_val::TD # Hessian matrix values
    Hessian::Union{Nothing,AbstractMatrix{T}} # Hessian matrix

    prim_infeas::T # primal infeasibility at `x`
    dual_infeas::T # dual (approximate?) infeasibility
    compl::T # complementary slackness

    optimizer::Union{Nothing,AbstractSubOptimizer} # Subproblem optimizer
    sub_status # subproblem status

    options::Parameters

    feasibility_restoration::Bool # indicator for feasibility restoration
    iter::Int # iteration counter
    ret::Int # solution status
    start_time::Float64 # solution start time
    start_iter_time::Float64 # iteration start time

    tmpx::TD # temporary solution x
    tmpE::TD # temporary constraint evaluation
end

"""
    QpData

Create QP subproblem data
"""
function QpData(sqp::AbstractSqpOptimizer)
	return QpData(
        MOI.MIN_SENSE,
        sqp.Hessian,
		sqp.df,
		sqp.Jacobian,
		sqp.E,
		sqp.problem.g_L,
		sqp.problem.g_U,
		sqp.problem.x_L,
		sqp.problem.x_U,
        sqp.problem.num_linear_constraints
    )
end

"""
    eval_functions!

Evalute the objective, gradient, constraints, and Jacobian.
"""
function eval_functions!(sqp::AbstractSqpOptimizer)
    sqp.f = sqp.problem.eval_f(sqp.x)
    sqp.problem.eval_grad_f(sqp.x, sqp.df)
    sqp.problem.eval_g(sqp.x, sqp.E)
    eval_Jacobian!(sqp)
    if !isnothing(sqp.problem.eval_h)
        sqp.problem.eval_h(sqp.x, :eval, sqp.h_row, sqp.h_col, 1.0, sqp.lambda, sqp.h_val)
        fill!(sqp.Hessian.nzval, 0.0)
        for (i, v) in enumerate(sqp.h_val)
            if sqp.h_col[i] == sqp.h_row[i]
                sqp.Hessian[sqp.h_row[i],sqp.h_col[i]] += v
            else
                sqp.Hessian[sqp.h_row[i],sqp.h_col[i]] += v
                sqp.Hessian[sqp.h_col[i],sqp.h_row[i]] += v
            end
        end
    end
end

"""
    eval_Jacobian!

Evaluate Jacobian matrix.
"""
function eval_Jacobian!(sqp::AbstractSqpOptimizer)
    sqp.problem.eval_jac_g(sqp.x, :eval, sqp.j_row, sqp.j_col, sqp.dE)
    fill!(sqp.Jacobian.nzval, 0.0)
    for (i, v) in enumerate(sqp.dE)
        sqp.Jacobian[sqp.j_row[i],sqp.j_col[i]] += v
    end
end

"""
    norm_violations

Compute the normalized constraint violation
"""
norm_violations(sqp::AbstractSqpOptimizer, p = 1) = norm_violations(
    sqp.E, sqp.problem.g_L, sqp.problem.g_U, 
    sqp.x, sqp.problem.x_L, sqp.problem.x_U, 
    p
)

function norm_violations(sqp::AbstractSqpOptimizer, x::TD, p = 1) where {T, TD<:AbstractArray{T}}
    fill!(sqp.tmpE, 0.0)
    return norm_violations(
        sqp.problem.eval_g(x, sqp.tmpE), sqp.problem.g_L, sqp.problem.g_U, 
        x, sqp.problem.x_L, sqp.problem.x_U, 
        p
    )
end

"""
    KT_residuals

Compute Kuhn-Turck residuals
"""
KT_residuals(sqp::AbstractSqpOptimizer) = KT_residuals(sqp.df, sqp.lambda, sqp.mult_x_U, sqp.mult_x_L, sqp.Jacobian)

"""
    norm_complementarity

Compute the normalized complementeraity
"""
norm_complementarity(sqp::AbstractSqpOptimizer, p = Inf) = norm_complementarity(
    sqp.E, sqp.problem.g_L, sqp.problem.g_U, 
    sqp.x, sqp.problem.x_L, sqp.problem.x_U, 
    sqp.lambda, sqp.mult_x_U, sqp.mult_x_L, 
    p
)

"""
    compute_phi

Evaluate and return the merit function value for a given point x + α * p.

# Arguments
- `sqp`: SQP structure
- `x`: the current solution point
- `α`: step size taken from `x`
- `p`: direction taken from `x`
"""
function compute_phi(sqp::AbstractSqpOptimizer, x::TD, α::T, p::TD) where {T,TD<:AbstractArray{T}}
    sqp.tmpx .= x .+ α * p
    f = sqp.f
    sqp.tmpE .= sqp.E
    if α > 0.0
        f = eval_f(sqp.problem, sqp.tmpx)
        eval_g!(sqp.problem, sqp.tmpx, sqp.tmpE)
    end
    if sqp.feasibility_restoration
        return norm_violations(sqp.tmpE, sqp.problem.g_L, sqp.problem.g_U, sqp.tmpx, sqp.problem.x_L, sqp.problem.x_U, 1)
    else
        return f + sqp.μ * norm_violations(sqp.tmpE, sqp.problem.g_L, sqp.problem.g_U, sqp.tmpx, sqp.problem.x_L, sqp.problem.x_U, 1)
    end
end

"""
    compute_derivative

Compute the directional derivative at current solution for a given direction.
"""
function compute_derivative(sqp::AbstractSqpOptimizer)
    dfp = 0.0
    cons_viol = zeros(sqp.problem.m)
    if sqp.feasibility_restoration
        for (_, v) in sqp.p_slack
            dfp += sum(v)
        end
        for i = 1:sqp.problem.m
            viol = maximum([0.0, sqp.E[i] - sqp.problem.g_U[i], sqp.problem.g_L[i] - sqp.E[i]])
            lhs = sqp.E[i] - viol
            cons_viol[i] += maximum([0.0, lhs - sqp.problem.g_U[i], sqp.problem.g_L[i] - lhs])
        end
    else
        dfp += sqp.df' * sqp.p
        for i = 1:sqp.problem.m
            cons_viol[i] += maximum([
                0.0, 
                sqp.E[i] - sqp.problem.g_U[i],
                sqp.problem.g_L[i] - sqp.E[i]
            ])
        end
    end
    return compute_derivative(dfp, sqp.μ, cons_viol)
end

function terminate_by_iterlimit(sqp::AbstractSqpOptimizer)
    if sqp.iter > sqp.options.max_iter
        sqp.ret = -1
        if sqp.prim_infeas <= sqp.options.tol_infeas
            sqp.ret = 6
        end
        return true
    end
    return false
end

# include("sqp_line_search.jl")
include("sqp_trust_region.jl")