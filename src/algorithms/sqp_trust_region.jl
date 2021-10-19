"""
    Sequential quadratic programming with trust region
"""
abstract type AbstractSqpTrOptimizer <: AbstractSqpOptimizer end

mutable struct SqpTR{T,TD,TI} <: AbstractSqpTrOptimizer
    @sqp_fields

    # directions for multipliers
    p_lambda::TD
    p_mult_x_L::TD
    p_mult_x_U::TD

    E_soc::TD # constraint evaluation for SOC
    soc::TD # second-order correction direction

    phi::T # merit function value
    μ::T # penalty parameter

    Δ::T # current trust region size
    Δ_min::T # minimum trust region size allowed
    Δ_max::T # maximum trust region size allowed

    step_acceptance::Bool

    function SqpTR(problem::Model{T,TD}, TI = Vector{Int}) where {T,TD<:AbstractArray{T}}
        sqp = new{T,TD,TI}()
        sqp.problem = problem
        sqp.x = deepcopy(problem.x)
        sqp.p = zeros(T, problem.n)
        sqp.p_soc = zeros(T, problem.n)
        sqp.p_slack = Dict()
        sqp.lambda = zeros(T, problem.m)
        sqp.mult_x_L = zeros(T, problem.n)
        sqp.mult_x_U = zeros(T, problem.n)
        sqp.df = TD(undef, problem.n)
        sqp.E = TD(undef, problem.m)
        sqp.dE = TD(undef, length(problem.j_str))

        # FIXME: Replace Vector{Int} with TI?
        sqp.j_row = TI(undef, length(problem.j_str))
        sqp.j_col = TI(undef, length(problem.j_str))
        for i = 1:length(problem.j_str)
            sqp.j_row[i] = Int(problem.j_str[i][1])
            sqp.j_col[i] = Int(problem.j_str[i][2])
        end
        sqp.Jacobian =
            sparse(sqp.j_row, sqp.j_col, ones(length(sqp.j_row)), problem.m, problem.n)
        sqp.h_row = TI(undef, length(problem.h_str))
        sqp.h_col = TI(undef, length(problem.h_str))
        for i = 1:length(problem.h_str)
            sqp.h_row[i] = Int(problem.h_str[i][1])
            sqp.h_col[i] = Int(problem.h_str[i][2])
        end
        sqp.h_val = TD(undef, length(problem.h_str))
        sqp.Hessian =
            sparse(sqp.h_row, sqp.h_col, ones(length(sqp.h_row)), problem.n, problem.n)

        sqp.p_lambda = zeros(T, problem.m)
        sqp.p_mult_x_L = zeros(T, problem.n)
        sqp.p_mult_x_U = zeros(T, problem.n)

        sqp.E_soc = TD(undef, problem.m)
        sqp.soc = zeros(T, problem.n)

        sqp.phi = 1.0e+20
        sqp.μ = 1.0e+4
        sqp.Δ = 10.0
        sqp.Δ_min = 1.0e-4
        sqp.Δ_max = 1.0e+4
        sqp.step_acceptance = true

        sqp.prim_infeas = Inf
        sqp.dual_infeas = Inf
        sqp.compl = Inf

        sqp.options = problem.parameters
        sqp.optimizer = nothing
        sqp.sub_status = nothing

        sqp.feasibility_restoration = false
        sqp.iter = 1
        sqp.ret = -5
        sqp.start_time = 0.0
        sqp.start_iter_time = 0.0

        sqp.tmpx = TD(undef, problem.n)
        sqp.tmpE = TD(undef, problem.m)

        return sqp
    end
end

"""
    run!

Run the line-search SQP algorithm
"""
function run!(sqp::AbstractSqpTrOptimizer)

    sqp.start_time = time()

    if sqp.options.OutputFlag == 0
        Logging.disable_logging(Logging.Info)
    end

    print_header(sqp)

    # Find the initial point feasible to linear and bound constraints
    lpviol = violation_of_linear_constraints(sqp, sqp.x)
    if lpviol > sqp.options.tol_infeas
        @info "Initial point not feasible to linear constraints..."
        sub_optimize_lp!(sqp)

        print(sqp, "LP")
    else
        @info "Initial point feasible to linear constraints..." lpviol
    end

    while true

        # Iteration counter limit
        if terminate_by_iterlimit(sqp)
            break
        end

        sqp.start_iter_time = time()

        # evaluate function, constraints, gradient, Jacobian
        if sqp.step_acceptance
            eval_functions!(sqp)
            sqp.prim_infeas = norm_violations(sqp, 1)
            sqp.dual_infeas = KT_residuals(sqp)
            sqp.compl = norm_complementarity(sqp)
        end

        # solve QP subproblem
        QP_time = @elapsed compute_step!(sqp)
        add_statistics(sqp.problem, "QP_time", QP_time)

        if sqp.sub_status ∈ [MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.LOCALLY_SOLVED]
            # do nothing
        elseif sqp.sub_status ∈ [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.NORM_LIMIT, MOI.OBJECTIVE_LIMIT]
            if sqp.feasibility_restoration == true
                @info "Failed to find a feasible direction"
                if sqp.prim_infeas <= sqp.options.tol_infeas
                    sqp.ret = 6
                else
                    sqp.ret = 2
                end
                break
            else
                @info "Feasibility restoration starts..."
                # println("Feasibility restoration ($(sqp.sub_status), |p| = $(norm(sqp.p, Inf))) begins.")
                sqp.feasibility_restoration = true
                print(sqp)
                collect_statistics(sqp)
                sqp.iter += 1
                continue
            end
        else
            sqp.ret == -3
            if sqp.prim_infeas <= sqp.options.tol_infeas
                sqp.ret = 6
            end
            break
        end

        if sqp.step_acceptance
            sqp.phi = compute_phi(sqp, sqp.x, 0.0, sqp.p)
        end

        print(sqp)
        collect_statistics(sqp)

        if norm(sqp.p, Inf) <= sqp.options.tol_direction
            if sqp.feasibility_restoration
                sqp.feasibility_restoration = false
                sqp.iter += 1
                continue
            else
                sqp.ret = 0
                break
            end
        end

        if sqp.prim_infeas <= sqp.options.tol_infeas &&
           sqp.compl <= sqp.options.tol_residual &&
           norm(sqp.p, Inf) <= sqp.options.tol_direction
            if sqp.feasibility_restoration
                sqp.feasibility_restoration = false
                sqp.iter += 1
                continue
            elseif sqp.dual_infeas <= sqp.options.tol_residual
                sqp.ret = 0
                break
            end
        end

        do_step!(sqp)

        # NOTE: This is based on the algorithm of filterSQP.
        if sqp.feasibility_restoration && sqp.step_acceptance
            sqp.feasibility_restoration = false
        end

        sqp.iter += 1
    end
    sqp.problem.obj_val = sqp.problem.eval_f(sqp.x)
    sqp.problem.status = Int(sqp.ret)
    sqp.problem.x .= sqp.x
    sqp.problem.g .= sqp.E
    sqp.problem.mult_g .= sqp.lambda
    sqp.problem.mult_x_U .= sqp.mult_x_U
    sqp.problem.mult_x_L .= sqp.mult_x_L
    add_statistic(sqp.problem, "iter", sqp.iter)
end

"""
    violation_of_linear_constraints

Compute the violation of linear constraints at a given point `x`

# Arguments
- `sqp`: SQP model struct
- `x`: solution to evaluate the violations

# Note
This function assumes that the first `sqp.problem.num_linear_constraints` constraints are linear.
"""
function violation_of_linear_constraints(sqp::AbstractSqpTrOptimizer, x::TD)::T where {T, TD <: AbstractVector{T}}
    # evaluate constraints
    sqp.problem.eval_g(x, sqp.E)

    lpviol = 0.0
    for i = 1:sqp.problem.num_linear_constraints
        lpviol += max(0.0, sqp.problem.g_L[i] - sqp.E[i])
        lpviol -= min(0.0, sqp.problem.g_U[i] - sqp.E[i])
    end
    for i = 1:sqp.problem.n
        lpviol += max(0.0, sqp.problem.x_L[i] - x[i])
        lpviol -= min(0.0, sqp.problem.x_U[i] - x[i])
    end
    return lpviol
end

"""
    sub_optimize_lp!

Compute the initial point that is feasible to linear constraints and variable bounds.

# Arguments
- `sqp`: SQP model struct
"""
function sub_optimize_lp!(sqp::AbstractSqpTrOptimizer)
    sqp.f = sqp.problem.eval_f(sqp.x)
    sqp.problem.eval_grad_f(sqp.x, sqp.df)
    eval_Jacobian!(sqp)
    if 1 == 1
        sqp.x, sqp.lambda, sqp.mult_x_U, sqp.mult_x_L, sqp.sub_status = sub_optimize_lp(
            sqp.options.external_optimizer, 
            sqp.Jacobian, sqp.problem.g_L, sqp.problem.g_U,
            sqp.problem.x_L, sqp.problem.x_U, sqp.x,
            sqp.problem.num_linear_constraints, sqp.problem.m
        )
    else
        fill!(sqp.E, 0.0)
        
        sqp.optimizer =
            SubOptimizer(
                JuMP.Model(sqp.options.external_optimizer), 
                QpData(
                    MOI.MIN_SENSE,
                    nothing,
                    sqp.df,
                    sqp.Jacobian,
                    sqp.E,
                    sqp.problem.g_L,
                    sqp.problem.g_U,
                    sqp.problem.x_L,
                    sqp.problem.x_U,
                    sqp.problem.num_linear_constraints
                )
            )
        create_model!(sqp.optimizer, sqp.Δ)
        sqp.x, sqp.lambda, sqp.mult_x_U, sqp.mult_x_L, sqp.sub_status = sub_optimize_lp(sqp.optimizer, sqp.x)
    end

    # TODO: Do we need to discard small numbers?
    dropzeros!(sqp.x)
    dropzeros!(sqp.lambda)
    dropzeros!(sqp.mult_x_U)
    dropzeros!(sqp.mult_x_L)
    return
end

"""
    sub_optimize!

Solve trust-region QP subproblem. If in feasibility restoration phase, the feasibility restoration subproblem is solved.

# Arguments
- `sqp`: SQP model struct
"""
function sub_optimize!(sqp::AbstractSqpTrOptimizer)
    if isnothing(sqp.optimizer)
        sqp.optimizer = SubOptimizer(
            JuMP.Model(sqp.options.external_optimizer),
            QpData(sqp),
        )
        create_model!(sqp.optimizer, sqp.Δ)
    else
        sqp.optimizer.data = QpData(sqp)
    end
    # TODO: This can be modified to Sl1QP.
    if sqp.feasibility_restoration
        return sub_optimize_FR!(sqp.optimizer, sqp.x, sqp.Δ)
    else
        return sub_optimize!(sqp.optimizer, sqp.x, sqp.Δ)
        # return sub_optimize_L1QP!(sqp.optimizer, sqp.x, sqp.Δ, sqp.μ)
    end
end

"""
    sub_optimize_soc!

Solve second-order correction QP subproblem.

# Arguments
- `sqp`: SQP model struct
"""
function sub_optimize_soc!(sqp::AbstractSqpTrOptimizer)
    sqp.problem.eval_g(sqp.x + sqp.p, sqp.E_soc)
    sqp.E_soc -= sqp.Jacobian * sqp.p
    sqp.optimizer.data = QpData(
        MOI.MIN_SENSE,
        sqp.Hessian,
        sqp.df,
        sqp.Jacobian,
        sqp.E_soc,
        sqp.problem.g_L,
        sqp.problem.g_U,
        sqp.problem.x_L,
        sqp.problem.x_U,
        sqp.problem.num_linear_constraints
    )
    p, _, _, _, _, _ = sub_optimize!(sqp.optimizer, sqp.x, sqp.Δ)
    sqp.p_soc .= sqp.p .+ p
    return nothing
    # return sub_optimize_L1QP!(sqp.optimizer, sqp.x, sqp.Δ, sqp.μ)
end

"""
    compute_step!

Compute the step direction with respect to priaml and dual variables by solving QP subproblem and also updates the penalty parameter μ.

# Arguments
- `sqp`: SQP model struct
"""
function compute_step!(sqp::AbstractSqpTrOptimizer)

    @info "solve QP subproblem..."
    sqp.p, lambda, mult_x_U, mult_x_L, sqp.p_slack, sqp.sub_status = sub_optimize!(sqp)

    sqp.p_lambda = lambda - sqp.lambda
    sqp.p_mult_x_L = mult_x_L - sqp.mult_x_L
    sqp.p_mult_x_U = mult_x_U - sqp.mult_x_U
    sqp.μ = max(sqp.μ, norm(sqp.lambda, Inf))
    @info "...found a direction"
end

"""
    compute_step_Sl1QP!

Compute the step direction with respect to priaml and dual variables by solving an elastic-mode QP subproblem and also updates the penalty parameter μ.

# Arguments
- `sqp`: SQP model struct

# Note
This is not currently used.
"""
function compute_step_Sl1QP!(sqp::AbstractSqpTrOptimizer)

    ϵ_1 = 0.9
    ϵ_2 = 0.1

    sqp.p, lambda, mult_x_U, mult_x_L, sqp.p_slack, sqp.sub_status = sub_optimize!(sqp)

    if sqp.sub_status ∈ [MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.LOCALLY_SOLVED]
        # compute the constraint violation
        m_0 = norm_violations(sqp, 1)
        m_μ = 0.0
        for (_, slacks) in sqp.p_slack
            m_μ += sum(slacks)
        end

        if m_μ > 1.0e-8
            p, infeasibility = sub_optimize_infeas(sqp.optimizer, sqp.x, sqp.Δ)
            # @show m_μ, infeasibility
            if infeasibility < 1.0e-8
                while m_μ > 1.0e-8 && sqp.μ < sqp.options.max_mu
                    sqp.μ = min(10.0 * sqp.μ, sqp.options.max_mu)

                    sqp.p, lambda, mult_x_U, mult_x_L, sqp.p_slack, sqp.sub_status = sub_optimize_L1QP!(sqp.optimizer, sqp.x, sqp.Δ, sqp.μ)

                    m_μ = 0.0
                    for (_, slacks) in sqp.p_slack, s in slacks
                        m_μ += s
                    end
                    @info "L1QP solve for feasible QP" infeasibility sqp.μ sqp.sub_status m_μ
                end
            else
                m_inf = norm_violations(
                    sqp.E + sqp.Jacobian * p,
                    sqp.problem.g_L,
                    sqp.problem.g_U,
                    sqp.x + p,
                    sqp.problem.x_L,
                    sqp.problem.x_U,
                    1,
                )
                while m_0 - m_μ < ϵ_1 * (m_0 - m_inf) && sqp.μ < sqp.options.max_mu
                    sqp.μ = min(10.0 * sqp.μ, sqp.options.max_mu)
                    sqp.p, lambda, mult_x_U, mult_x_L, sqp.p_slack, sqp.sub_status = sub_optimize_L1QP!(sqp.optimizer, sqp.x, sqp.Δ, sqp.μ)

                    m_μ = 0.0
                    for (_, slacks) in sqp.p_slack, s in slacks
                        m_μ += s
                    end
                    @info "L1QP solve for infeasible QP" infeasibility sqp.μ m_0 m_μ
                end
            end
        end

        # q_0 = compute_qmodel(sqp, false)
        # q_k = compute_qmodel(sqp, true)
        # @info "L1QP solve for μ+" q_0 q_k m_0 m_μ
        # while q_0 - q_k < ϵ_2 * sqp.μ * (m_0 - m_μ)
        #     sqp.μ = min(2.0 * sqp.μ, sqp.options.max_mu)
        #     sqp.p, lambda, mult_x_U, mult_x_L, sqp.p_slack, sqp.sub_status = sub_optimize_L1QP!(sqp.optimizer, sqp.x, sqp.Δ, sqp.μ)

        #     m_μ = 0.0
        #     for (_, slacks) in sqp.p_slack, s in slacks
        #         m_μ += s
        #     end
        #     q_k = compute_qmodel(sqp, true)
        #     @info "L1QP solve for μ+" q_0 q_k m_0 m_μ
        # end

    else
        @error "Unexpected QP subproblem status $(sqp.sub_status)"
    end

    @info "...solved QP subproblem"

    sqp.p_lambda = lambda - sqp.lambda
    sqp.p_mult_x_L = mult_x_L - sqp.mult_x_L
    sqp.p_mult_x_U = mult_x_U - sqp.mult_x_U
    sqp.μ = max(sqp.μ, norm(sqp.lambda, Inf))
end

"""
    compute_qmodel

Evaluate the quadratic model q(p) with ℓ₁ penalty term, which is given by
q(p) = fₖ + ∇fₖᵀp + 0.5 pᵀ ∇ₓₓ²Lₖ p + μ ∑ᵢ|cᵢ(xₖ) + ∇cᵢ(xₖ)ᵀp| + μ ∑ᵢ[cᵢ(xₖ) + ∇cᵢ(xₖ)ᵀp]⁻

# Arguments
- `sqp::SqpTR`: SQP model struct
- `p::TD`: direction vector
- `with_step::Bool`: `true` for q(p); `false` for `q(0)`

# Note
For p=0, the model is simplified to q(0) = μ ∑ᵢ|cᵢ(xₖ)| + μ ∑ᵢ[cᵢ(xₖ)]⁻
"""
function compute_qmodel(sqp::AbstractSqpTrOptimizer, p::TD, with_step::Bool = false) where {T, TD<:AbstractArray{T}}
    qval = 0.0
    if with_step
        qval += sqp.df' * p + 0.5 * p' * sqp.Hessian * p 
        sqp.tmpx .= sqp.x .+ p
        sqp.tmpE .= sqp.E .+ sqp.Jacobian * p
    else
        sqp.tmpx .= sqp.x
        sqp.tmpE .= sqp.E
    end
    qval += sqp.μ * norm_violations(
        sqp.tmpE,
        sqp.problem.g_L,
        sqp.problem.g_U,
        sqp.tmpx,
        sqp.problem.x_L,
        sqp.problem.x_U,
        1,
    )
    return qval
end
compute_qmodel(sqp::AbstractSqpTrOptimizer, with_step::Bool = false) = compute_qmodel(sqp, sqp.p, with_step)

"""
    do_step!

Test the step `p` whether to accept or reject.
"""
function do_step!(sqp::AbstractSqpTrOptimizer)

    ϕ_k = compute_phi(sqp, sqp.x, 1.0, sqp.p)
    ared = sqp.phi - ϕ_k
    # @show sqp.phi, ϕ_k

    pred = 1.0
    if !sqp.feasibility_restoration
        q_0 = compute_qmodel(sqp, false)
        q_k = compute_qmodel(sqp, true)
        pred = q_0 - q_k
        # @show q_0, q_k
    end

    ρ = ared / pred
    if ared > 0 && ρ > 0
        sqp.x .+= sqp.p
        sqp.lambda .+= sqp.p_lambda
        sqp.mult_x_L .+= sqp.p_mult_x_L
        sqp.mult_x_U .+= sqp.p_mult_x_U
        if sqp.Δ == norm(sqp.p, Inf)
            sqp.Δ = min(2 * sqp.Δ, sqp.Δ_max)
        end
        sqp.step_acceptance = true
    else
        sqp.tmpx .= sqp.x .+ sqp.p
        c_k = norm_violations(sqp, sqp.tmpx)

        perform_soc = false
        if sqp.options.use_soc
            if c_k > 0 && sqp.feasibility_restoration == false
                @info "Try second-order correction..."

                # sqp.p should be adjusted inside sub_optimize_soc!
                sub_optimize_soc!(sqp)

                ϕ_soc = compute_phi(sqp, sqp.x, 1.0, sqp.p_soc)
                ared = sqp.phi - ϕ_soc
                pred = 1.0
                if !sqp.feasibility_restoration
                    q_soc = compute_qmodel(sqp, sqp.p_soc, true)
                    pred = q_0 - q_soc
                end
                ρ_soc = ared / pred
                if ared > 0 && ρ_soc > 0
                    @info "SOC" ϕ_k ϕ_soc ared pred ρ_soc
                    @info "...second-order correction added"
                    sqp.x .+= sqp.p_soc
                    sqp.lambda .+= sqp.p_lambda
                    sqp.mult_x_L .+= sqp.p_mult_x_L
                    sqp.mult_x_U .+= sqp.p_mult_x_U
                    sqp.step_acceptance = true
                    perform_soc = true
                else
                    @info "...second-order correction discarded"
                end
            end
        end

        if !perform_soc
            sqp.Δ = 0.5 * min(sqp.Δ, norm(sqp.p, Inf))
            sqp.step_acceptance = false
        end
    end
end

function print_header(sqp::AbstractSqpTrOptimizer)
    if sqp.options.OutputFlag == 0
        return
    end
    @printf("  %6s", "iter")
    @printf(" ")
    @printf("  %15s", "f(x_k)")
    @printf("  %15s", "ϕ(x_k)")
    @printf("  %15s", "μ")
    @printf("  %15s", "|λ|∞")
    @printf("  %14s", "Δ")
    @printf("  %14s", "|p|")
    @printf("  %14s", "inf_pr")
    @printf("  %14s", "inf_du")
    @printf("  %14s", "compl")
    @printf("  %10s", "time")
    @printf("\n")
end

"""
    print

Print iteration information.
"""
function print(sqp::AbstractSqpTrOptimizer, status_mark = "  ")
    if sqp.options.OutputFlag == 0
        return
    end
    if sqp.iter > 1 && (sqp.iter - 1) % 25 == 0
        print_header(sqp)
    end
    st = ifelse(sqp.feasibility_restoration, "FR", status_mark)
    @printf("%2s%6d", st, sqp.iter)
    @printf("%1s", ifelse(sqp.step_acceptance, "a", "r"))
    @printf("  %+6.8e", sqp.f)
    @printf("  %+6.8e", sqp.phi)
    @printf("  %+6.8e", sqp.μ)
    @printf("  %+6.8e", norm(sqp.lambda,Inf))
    @printf("  %6.8e", sqp.Δ)
    @printf("  %6.8e", norm(sqp.p, Inf))
    if isinf(sqp.prim_infeas)
        @printf("  %14s", "Inf")
    else
        @printf("  %6.8e", sqp.prim_infeas)
    end
    if isinf(sqp.dual_infeas)
        @printf("  %14s", "Inf")
    else
        @printf("  %6.8e", sqp.dual_infeas)
    end
    if isinf(sqp.compl)
        @printf("  %14s", "Inf")
    else
        @printf("  %6.8e", sqp.compl)
    end
    @printf("  %10.2f", time() - sqp.start_time)
    @printf("\n")
end

"""
    collect_statistics

Collect iteration information.
"""
function collect_statistics(sqp::AbstractSqpTrOptimizer)
    if sqp.options.StatisticsFlag == 0
        return
    end
    add_statistics(sqp.problem, "f(x)", sqp.f)
    add_statistics(sqp.problem, "ϕ(x_k))", sqp.phi)
    add_statistics(sqp.problem, "D(ϕ,p)", sqp.directional_derivative)
    add_statistics(sqp.problem, "|p|", norm(sqp.p, Inf))
    add_statistics(sqp.problem, "|J|2", norm(sqp.dE, 2))
    add_statistics(sqp.problem, "|J|inf", norm(sqp.dE, Inf))
    add_statistics(sqp.problem, "inf_pr", sqp.prim_infeas)
    # add_statistics(sqp.problem, "inf_du", dual_infeas)
    add_statistics(sqp.problem, "compl", sqp.compl)
    add_statistics(sqp.problem, "alpha", sqp.alpha)
    add_statistics(sqp.problem, "iter_time", time() - sqp.start_iter_time)
    add_statistics(sqp.problem, "time_elapsed", time() - sqp.start_time)
end
