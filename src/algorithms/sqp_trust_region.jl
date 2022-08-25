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

    sqp.μ = sqp.options.init_mu
    sqp.Δ = sqp.options.tr_size

    sqp.start_time = time()

    if sqp.options.OutputFlag == 0
        Logging.disable_logging(Logging.Info)
    end

    print_header(sqp)

    # TODO: warm start

    # truncate initial x to lie inside simple bounds
    for i = 1:sqp.problem.n
        sqp.x[i] = min(sqp.problem.x_U[i], max(sqp.problem.x_L[i], sqp.x[i]))
    end

    # evaluate the constraints at the initial point
    sqp.problem.eval_g(sqp.x, sqp.E)
    sqp.problem.eval_grad_f(sqp.x, sqp.df)
    eval_Jacobian!(sqp)

    # ensure that the initial point is feasible wrt linear c/s
    lpviol = violation_of_linear_constraints(sqp, sqp.x)
    if lpviol > sqp.options.tol_infeas
        sqp.f = sqp.problem.eval_f(sqp.x)
        sqp.prim_infeas = norm_violations(sqp.E, sqp.problem.g_L, sqp.problem.g_U)
        sqp.phi = sqp.f + sqp.prim_infeas
        print(sqp, "L-INF")
        @info "Initial point not feasible in linear c/s..."
        @info "...solving a phase 1 problem for linear c/s"
        @info "...setting Hessian = I and gradient = 0"

        sub_optimize_lp!(sqp)

        if sqp.sub_status == :Optimal
            sqp.x .+= sqp.p
    
            sqp.f = sqp.problem.eval_f(sqp.x)
            sqp.problem.eval_g(sqp.x, sqp.E)
            sqp.prim_infeas = norm_violations(sqp.E, sqp.problem.g_L, sqp.problem.g_U)
    
            # update QP
            sqp.problem.eval_grad_f(sqp.x, sqp.df)
            eval_Jacobian!(sqp)

            sqp.lambda .= sqp.problem.mult_g
            # TODO: combine mult_x_U and mult_x_L
            sqp.mult_x_U .= sqp.problem.mult_x_U
            sqp.mult_x_L .= sqp.problem.mult_x_L
            eval_Hessian!(sqp)
    
            @info "Feasible point for linear c/s found"
            @info "f(x), h(c(x)) = $(sqp.f), $(sqp.prim_infeas)"
        elseif sqp.sub_status == :Infeasible
            @info "Linear c/s are inconsistent: STOP"
            # TODO: return
        else
            @info "Unexpected status: $(sqp.sub_status)"
            # TODO: return
        end
    else
        @info "Initial point is feasible wrt linear c/s"
        sqp.f = sqp.problem.eval_f(sqp.x)
        eval_Hessian!(sqp)
        sqp.prim_infeas = norm_violations(sqp.E, sqp.problem.g_L, sqp.problem.g_U)
    end

    sqp.μ = 1.0
    sqp.phi = sqp.f + sqp.prim_infeas

    # If filter were used, this is the place to initialize the filter.

    print(sqp, "START")

    while true

        # set up and solve QP subproblem
        sub_optimize!(sqp)
        if sqp.optimizer.status ∈ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.LOCALLY_SOLVED]
            d_norm = norm(sqp.optimizer.xsol, Inf)
            # update x, f(x+d), c(x+d)
            sqp.x .+= sqp.optimizer.xsol
            sqp.f = sqp.problem.eval_f(sqp.x)
            sqp.problem.eval_g(sqp.x, sqp.E)
            # compute constraint violation
            sqp.prim_infeas = norm_violations(sqp.E, sqp.problem.g_L, sqp.problem.g_U)

            penalty_estimate!(sqp)
            check_accept!(sqp)

            if d_norm < sqp.options.tol_direction
                sqp.step_acceptance = true
                @info "Zero step from QP: accept"
            end
            if sqp.step_acceptance
                @info "Step acceptable to filter, Δ = $(sqp.Δ)"
            else
                if sqp.feasibility_restoration
                    best_phi = sqp.phi
                    # ...
                end
            end

            print(sqp)

            if sqp.step_acceptance == false && sqp.prim_infeas < Inf && sqp.prim_infeas > 0.0
                @info "Step not accepted, try SOC steps"
                while true
                    sub_optimize_soc!(sqp)
                end
            end

        elseif sqp.optimizer.status ∈ [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.NORM_LIMIT, MOI.OBJECTIVE_LIMIT]
            # solve phase I SQP
        else
            @info "Unexpected QP status: $(sqp.optimizer.status)"
            sqp.ret == -3
            if sqp.prim_infeas <= sqp.options.tol_infeas
                sqp.ret = 6
            end
            break
        end

        # Iteration counter limit
        if terminate_by_iterlimit(sqp)
            break
        end

        sqp.start_iter_time = time()

        # evaluate function, constraints, gradient, Jacobian
        if sqp.step_acceptance
            eval_functions!(sqp)
            sqp.prim_infeas = norm_violations(sqp.E, sqp.problem.g_L, sqp.problem.g_U)
            sqp.dual_infeas = KT_residuals(sqp)
            sqp.compl = norm_complementarity(sqp)
        end
        # @info "iteration status" sqp.f sqp.prim_infeas sqp.compl
        @debug begin
            print_vector(sqp.x, "new x values")
            print_vector(sqp.lambda, "Multipliers")
        end

        print(sqp)
        collect_statistics(sqp)

        # solve QP subproblem
        QP_time = @elapsed compute_step!(sqp)
        add_statistics(sqp.problem, "QP_time", QP_time)

        if sqp.optimizer.status ∈ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.LOCALLY_SOLVED]
            # do nothing
        elseif sqp.optimizer.status ∈ [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.NORM_LIMIT, MOI.OBJECTIVE_LIMIT]
            if sqp.feasibility_restoration == true
                @info "Failed to find a feasible direction"
                if sqp.prim_infeas <= sqp.options.tol_infeas
                    sqp.ret = 6
                else
                    sqp.ret = 2
                end
                break
            else
                @info "Feasibility restoration starts... (status: $(sqp.optimizer.status))"
                # println("Feasibility restoration ($(sqp.optimizer.status), |p| = $(norm(sqp.p, Inf))) begins.")
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

function penalty_estimate!(sqp::AbstractSqpTrOptimizer)
    μ = maximum(10.0 .^ round.(log10.(abs.(sqp.lambda)) .+ 1))
    sqp.μ = clamp(μ, 1.e-6, 1.e+6)
    # estimate of l_infty exact penalty function
    sqp.phi = sqp.f + sqp.μ * sqp.prim_infeas
end

function bound_shift!(n, m, blo, bup, bl, bu, x, c, cstype, Δ, shift, w, socs = false)
    if shift > 0.0
        for i=1:m
            if cstype[i] == "N"
                bl[n+i] = blo[n+i] - c[i] - shift
                bu[n+i] = bup[n+i] - c[i] - shift
            else
                bl[n+i] = blo[n+i] - c[i]
                bu[n+i] = bup[n+i] - c[i]
            end
        end
        if socs
            @error "bound_shift: STOP"
        end
    else
        if socs
            for i=1:m
                bl[n+i] = blo[n+i] - c[i] + w[i]
                bu[n+i] = bup[n+i] - c[i] + w[i]
            end
        else
            for i=1:m
                bl[n+i] = blo[n+i] - c[i]
                bu[n+i] = bup[n+i] - c[i]
            end
        end
    end

    for i=1:n
        bl[i] = max(blo[i]-x[i], -Δ)
        bu[i] = min(bup[i]-x[i],  Δ)
    end

    if shift == 0.0
        for i=1:m
            if cstype[i] == "L"
                if blo[n+i] == bup[n+i]
                    bu[n+i] = 0.0
                    bl[n+i] = 0.0
                else
                    bu[n+i] = max(bu[n+i], 0.0)
                    bl[n+i] = min(bl[n+i], 0.0)
                end
            end
        end
        for i=1:n
            if blo[i] == bup[i]
                bu[i] = 0.0
                bl[i] = 0.0
            else
                bu[i] = max(bu[i], 0.0)
                bl[i] = min(bl[i], 0.0)
            end
        end
    end
end

function sub_optimize_lp!(sqp::AbstractSqpTrOptimizer)
    c = zeros(sqp.problem.n)
    Q = sparse(I, sqp.problem.n, sqp.problem.n)
    bl = zeros(sqp.problem.n+sqp.problem.m)
    bu = zeros(sqp.problem.n+sqp.problem.m)
    bound_shift!(
        sqp.problem.n, sqp.problem.m,
        [sqp.problem.x_L; sqp.problem.g_L],
        [sqp.problem.x_U; sqp.problem.g_U],
        bl, bu,
        sqp.x, sqp.E, cstype, sqp.Δ, Inf, sqp.E
    )
    QPsolve!(
        sqp.problem.n,
        sqp.problem.m,
        c, Q, sqp.Jacobian, bl, bu,
        sqp.p, sqp.r, sqp.sub_status
    )
end

function sub_optimize_qp!(sqp::AbstractSqpTrOptimizer)
    c = zeros(sqp.problem.n)
    Q = sparse(I, sqp.problem.n, sqp.problem.n)
    bl = [sqp.problem.x_L; sqp.problem.g_L]
    bu = [sqp.problem.x_U; sqp.problem.g_U]
    for i = 1:sqp.problem.m
        bl[sqp.problem.n+i] -= sqp.E[i]
        bu[sqp.problem.n+i] -= sqp.E[i]
    end
    for i = 1:sqp.problem.n
        bl[i] = max(bl[i] - sqp.x[i], -sqp.Δ)
        bu[i] = min(bu[i] - sqp.x[i],  sqp.Δ)
    end
    QPsolve!(
        sqp.problem.n,
        sqp.problem.m,
        c, Q, sqp.Jacobian, bl, bu,
        sqp.p, sqp.r, sqp.sub_status
    )
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
    # println("Hessian:")
    # print_matrix(sqp.Hessian)
    # println("Jacobian:")
    # print_matrix(sqp.Jacobian)
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
    sub_optimize!(sqp.optimizer, sqp.x, sqp.Δ)
    sqp.p_soc .= sqp.p .+ sqp.optimizer.xsol
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

    @debug "solve QP subproblem..."
    sub_optimize!(sqp)

    if sqp.optimizer.status ∈ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.LOCALLY_SOLVED]
        sqp.p .= sqp.optimizer.xsol
        sqp.p_lambda .= sqp.optimizer.λ .- sqp.lambda
        sqp.p_mult_x_L .= max.(0.0, sqp.optimizer.μ) .- sqp.mult_x_L
        sqp.p_mult_x_U .= min.(0.0, sqp.optimizer.μ) .- sqp.mult_x_U
        sqp.μ = max(sqp.μ, norm(sqp.lambda, Inf))
        @debug "...found a direction"
    end
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
        sqp.problem.g_U
    )
    return qval
end
compute_qmodel(sqp::AbstractSqpTrOptimizer, with_step::Bool = false) = compute_qmodel(sqp, sqp.p, with_step)

function check_accept!(sqp::AbstractSqpTrOptimizer)
    ϕ_k = compute_phi(sqp, sqp.x, 1.0, sqp.p)
    ared = sqp.phi - ϕ_k
    pred = 1.0
    if !sqp.feasibility_restoration
        q_0 = compute_qmodel(sqp, false)
        q_k = compute_qmodel(sqp, true)
        pred = q_0 - q_k
    end
    ρ = ared / pred
    if ared > 0 && ρ > 0
        sqp.step_acceptance = true
    else
        sqp.step_acceptance = false
    end
end

"""
    do_step!

Test the step `p` whether to accept or reject.
"""
function do_step!(sqp::AbstractSqpTrOptimizer)

    ϕ_k = compute_phi(sqp, sqp.x, 1.0, sqp.p)
    ared = sqp.phi - ϕ_k
    # @show sqp.phi, ϕ_k
    @debug begin
        sqp.tmpx .= sqp.x .+ sqp.p
        f = sqp.problem.eval_f(sqp.tmpx)
        sqp.problem.eval_g(sqp.tmpx, sqp.tmpE)
        viol = norm_violations(sqp.tmpE, sqp.problem.g_L, sqp.problem.g_U)
        "new ϕ = $f + $(sqp.μ) * $(viol)"
    end

    pred = 1.0
    if !sqp.feasibility_restoration
        q_0 = compute_qmodel(sqp, false)
        q_k = compute_qmodel(sqp, true)
        pred = q_0 - q_k
        # @show q_0, q_k
    end

    ρ = ared / pred
    @debug "test step acceptance" sqp.phi ϕ_k ared pred ρ
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
    @printf("%11s", "iter")
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
    @printf("%5s%6d", st, sqp.iter)
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
