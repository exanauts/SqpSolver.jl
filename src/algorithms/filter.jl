"""
    filter-SQP
"""
mutable struct filterSQP{T,TD,TI} <: AbstractSqpOptimizer
    @sqp_fields

    blo::TD
    bup::TD
    qp::QpData

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
        sqp.blo = [sqp.problem.x_L; sqp.problem.g_L]
        sqp.bup = [sqp.problem.x_U; sqp.problem.g_U]
        sqp.qp = QpData{T,TD,SparseMatrixCSC}(problem.n, problem.m)
        sqp.qp.cstype = repeat("L", problem.num_linear_constraints) * repeat("N", problem.m - problem.num_linear_constraints)

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

        sqp.hc = Inf
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

Run filter-SQP algorithm
"""
function run!(sqp::filterSQP)

    convergence = false
    termination = false

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
        sqp.x[i] = clamp(sqp.x[i], sqp.problem.x_L[i], sqp.problem.x_U[i])
    end

    # evaluate the constraints at the initial point
    sqp.problem.eval_g(sqp.x, sqp.E)
    sqp.problem.eval_grad_f(sqp.x, sqp.df)
    eval_Jacobian!(sqp)

    # ensure that the initial point is feasible wrt linear c/s
    lpviol = violation_of_linear_constraints(sqp, sqp.x)
    if lpviol > sqp.options.tol_infeas
        sqp.f = sqp.problem.eval_f(sqp.x)
        sqp.hc = norm_violations(sqp.E, sqp.problem.g_L, sqp.problem.g_U)
        sqp.phi = sqp.f + sqp.hc
        print(sqp, "L-INF")
        @info "Initial point not feasible in linear c/s..."
        @info "...solving a phase 1 problem for linear c/s"
        @info "...setting Hessian = I and gradient = 0"

        # ensure Hessian = I and c = 0
        sqp.qp.c = zeros(sqp.problem.n)
        sqp.qp.Q = sparse(I, sqp.problem.n, sqp.problem.n)
        # shift the bounds (nonlinear c/s to infty)
        bound_shift!(
            sqp.qp.n, sqp.qp.m,
            sqp.blo, sqp.bup, sqp.qp.bl, sqp.qp.bu,
            sqp.x, sqp.E, sqp.qp.cstype, Inf, Inf, sqp.E
        )
        # solve feasibility problem for linear c/s only
        solve!(sqp.qp)

        if sqp.qp.status <= 1
            # update x, also if initial LP is infeasible
            sqp.xnew .= sqp.x + sqp.p
    
            # evaluate f, c at new starting point
            sqp.fnew = sqp.problem.eval_f(sqp.xnew)
            sqp.problem.eval_g(sqp.xnew, sqp.Enew)

            # compute h(c(x)) = max ( c_i(x) , 0 ) for c_i(x) <= 0
            sqp.hcnew = norm_violations(sqp.Enew, sqp.problem.g_L, sqp.problem.g_U)
    
            # update x, c(x), Jacobian and Hessian matrix
            update_QP!(sqp)
    
            @info "Feasible point for linear c/s found"
            @info "f(x), h(c(x)) = $(sqp.f), $(sqp.hc)"
        elseif sqp.qp.status == 2
            @warn "Linear c/s BOUNDS inconsistent: STOP"
            termination = true
        elseif sqp.qp.status == 3
            @warn "Linear c/s are inconsistent: STOP"
            termination = true
        else
            @warn "Unexpected status: $(sqp.sub_status)"
            termination = true
        end
    else
        @info "Initial point is feasible wrt linear c/s"
        sqp.f = sqp.problem.eval_f(sqp.x)
        eval_Hessian!(sqp)
        sqp.hc = norm_violations(sqp.E, sqp.problem.g_L, sqp.problem.g_U)
    end

    if !termination
        sqp.μ = 1.0
        sqp.phi = sqp.f + sqp.hc

        # add upper bound on c/s violation to filter
        cs_ubd = max(100.0, 0.125*hc) # TODO: need to be parameterized
        add_to_filter(-Inf, cs_bud, mu, filter, cs_ubd)

        # place initial point on filter
        NWSE = add_to_filter(sqp.f, sqp.hc, sqp.μ, filter, cs_ubd)

        print(sqp, "START")
    end

    while !termination

        iiter = 0

        # set up and solve QP subproblem
        bound_shift!(
            sqp.qp.n, sqp.qp.m,
            sqp.blo, sqp.blu, sqp.qp.bl, sqp.qp.bu,
            sqp.x, sqp.E, sqp.qp.cstype, sqp.Δ, 0.0, sqp.E
        )
        sub_optimize_qp!(sqp)
        iiter += 1

        if sqp.qp.status == 1
            @warn "unbounded QP: reject step"
            sqp.Δ /= 4.0
            if sqp.feasibility_restoration == false
                if sqp.Δ < sqp.options.tol_infeas / 1e+2 && !convergence
                    @warn "Δ < epsilon: TERMINATE SQP"
                    convergence = true
                end
            end
            if convergence
                termination = true
            end
        elseif sqp.qp.status == 9
            @warn "STOP: unexpected termination from QP solver"
            termination = true
        end

        if termination
            break
        end

        if sqp.qp.status == 3
            # solve phase I SQP
        elseif sqp.qp.status == 0
            # update the sufficient reduction entry in the filter
            if !sqp.feasibility_restoration
                filter.items[filter.pos,REDN] = -sqp.qp.f
            end

            # compute the infty norm of the step
            d_norm = norm(sqp.optimizer.xsol, Inf)
            # update x, f(x+d), c(x+d)
            sqp.x .+= sqp.optimizer.xsol
            sqp.f = sqp.problem.eval_f(sqp.x)
            sqp.problem.eval_g(sqp.x, sqp.E)
            # compute constraint violation
            sqp.hc = norm_violations(sqp.E, sqp.problem.g_L, sqp.problem.g_U)
            penalty_estimate!(sqp)
            NWSE = add_to_filter(sqp.f, sqp.hc, sqp.μ, filter, cs_ubd)
            if d_norm <= sqp.options.tol_infeas
                accept_step = true
                @info "Zero step from QP: accept"
            end
            if accept_step
                @info "Step acceptable to filter, Δ = $(sqp.Δ)"
                step = "+22"
            else
                step = "-22"
                # save step in work, in case SOCs fail & unblocking occurs
                if sqp.feasibility_restoration
                    best_phi = sqp.phi
                    best_NWSE = NWSE
                    best_Step = "+22"
                end
            end

            print(sqp)

            if sqp.step_acceptance == false && sqp.hc < Inf && sqp.hc > 0.0
                @debug "Step not accepted, try SOC steps"
                SOCiter = 0
                SOCexit = false
                avg_rate = 1.0
                while true
                    SOCiter += 1
                    SOCS_QP!(sqp, 2, 1, [i])
                    SOCScount += 1
                    iiter += 1

                    if sqp.qp.status == 1
                        @debug "WARNING: unbdd QP: phase II SOCS"
                    elseif sqp.qp.status == 2 || sqp.qp.status >= 4
                        sqp.status = 6
                        @info "STOP: SOCS unexpect status:" sqp.qp.status
                        termination = true
                        break
                    end

                    # compute infinity norm of step 
                    d_norm = norm(sqp.qp.x, Inf)
    
                    if sqp.qp.status != 3
                        # update x provisionally & evaluate f(x+d), c(x+d)
                        sqp.xnew .= sqp.x + sqp.qp.x
                        sqp.fnew = sqp.problem.eval_f(sqp.xnew)
                        sqp.problem.eval_g(sqp.xnew, sqp.Enew)
                        sqp.hcnew = norm_violations(sqp.Enew, sqp.problem.g_L, sqp.problem.g_U)
                        penalty_estimate!(sqp)
                        NWSE = add_to_filter(sqp.fnew, sqp.hcnew, sqp.μ, filter, cs_ubd)
                        @debug "SOC STEP: $(SOCiter)"
                        @debug "f, h(c) after SOSC " sqp.fnew sqp.hcnew
                        @debug "penalty after SOSC " sqp.phi

                        # compute linear rate of convergence for SOC step
                        if hchat <= 0.0
                            rate = 0.1
                        else
                            rate = hcnew / hchat
                        end
                        avg_rate = ((SOCiter-1) * avg_rate + rate) / SOCiter

                        # check for reduction in c/s l_1 sum
                        if accept_step
                            SOCexit = true
                            step = "+23"
                            @debug "Accept SOC step $(SOCiter)"
                        else
                            # check whether another SOC step would be helpful
                            if hcnew > eps && rate <= 0.25
                                SOCexit = false
                                @debug "Next SOC step, suff. reduction:" rate, hcnew
                                hchat = hcnew
                                if feas_rest
                                    if phinew < best_phi
                                        best_phi = phinew
                                        best_NWSE = NWSE
                                        best_step = "+23"
                                        @error "TODO: save best step up to now"
                                    end
                                end
                            else
                                SOCexit = true
                                @debug "STOP SOC: not suff. reduction:" rate, hcnew
                            end
                            step = "-23"
                        end
                    else
                        ###########################
                    end
                end

                if termination
                    break
                end
            end

            # try to unblock filter (for 1st step after feas. restn.)
            if sqp.feasibility_restoration && !accept_step
                unblocked, NWSE = unblock_filter(sqp.fnew, sqp.hcnew, sqp.μ, filter, cs_ubd, 2, NWSE)
                if unblocked
                    @debug "1st step after feasibty-restn:accept"
                    accept_step = true
                    step = "+24"
                else
                    @debug "Unblocking unsuccessful"
                    accept_step = false
                    step = "-24"
                end
                step1 = step * NWSE
                print(sqp)
            end

            if accept_step
                # reset indicator for feasibility restoration to false
                sqp.feasibility_restoration = false

                # take step modify TR (if necessary), new gradients
                if step == "+23" && avg_rate > 0.1
                    @debug "Do not increase TR radius: SOC rate too large $(avg_rate)"
                elseif step == "+24"
                    @debug "Do not increase TR radius after unblocking"
                else
                    sqp.Δ = enlarge_TR(sqp.Δ, d_norm)
                end
                # save position in filter of new entry for update of q
                filter.pos = f_pos
                # update penalty function value
                sqp.phi = phinew
                @debug "update x, f(x), c(x) A(x), and W(x)"
                update_QP!(sqp)
                # convergence test (for termination)
                convergence = conv_test(sqp, d_norm)
            else
                # reduce Trust region
                sqp.Δ = reduce_TR(sqp.Δ, d_norm)
                @debug "Step rejected after SOCS: reduce TR radius $(sqp.Δ)"
            end

            if sqp.iter >= sqp.options.max_iter && !convergence
                termination = true
                sqp.status = 6
                @info "STOP: ITERATION LIMIT REACHED"
            end
        end
    end

    if sqp.iter == 0 && sqp.qp.status ∈ [2,3]
        @info "Problem is linear infeasible."
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
    sqp.phi = sqp.f + sqp.μ * sqp.hc
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

function sub_optimize_qp!(sqp::AbstractSqpTrOptimizer)
    sqp.qp.c = copy(sqp.df)
    sqp.qp.Q = copy(sqp.Hessian)
    sqp.qp.A = copy(sqp.Jacobian)
    solve!(sqp.qp)
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
    if isinf(sqp.hc)
        @printf("  %14s", "Inf")
    else
        @printf("  %6.8e", sqp.hc)
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
    add_statistics(sqp.problem, "inf_pr", sqp.hc)
    # add_statistics(sqp.problem, "inf_du", dual_infeas)
    add_statistics(sqp.problem, "compl", sqp.compl)
    add_statistics(sqp.problem, "alpha", sqp.alpha)
    add_statistics(sqp.problem, "iter_time", time() - sqp.start_iter_time)
    add_statistics(sqp.problem, "time_elapsed", time() - sqp.start_time)
end


function add_to_filter(f, hc, mu, filter::Filter, cs_ubd)
    # check whether the new entry can be accepted by the filter
    acceptable, NWSE = check_accept(f, hc, filter)

    # add the new entry to the filter
    if acceptable
        add_entry!(f, hc, mu, cs_ubd, filter)
    end

    return NWSE
end

function check_accept(f, hc, filter::Filter)
    acceptable = true
    NWSE = "  "

    i = 0
    found = false
    if i + 1 < filter.len
        while i < filter.len && found == false
            i += 1
            found = sufficient_reduction(
                -Inf, hc, Inf, 0.0, 0.0, 1.0, filter.items[i,CONS], true
            )
        end
    else
        found = true
    end

    if found
        if i == 0 || (i == 1 && filter.len == 1)
            acceptable = true
        elseif i == 1
        elseif i < filter.len - 1
            acceptable = sufficient_reduction(
                f, hc, 
                filter.items[i-1,FUNC], filter.items[i-1,CONS], 
                filter.items[i-1,REDN], filter.items[i-1,PEN],
                filter.items[i,CONS], true
            )
            k = i - 1
        else
            acceptable = sufficient_reduction(
                f, hc, 
                filter.items[i-1,FUNC], filter.items[i-1,CONS], 
                filter.items[i-1,REDN], filter.items[i-1,PEN],
                filter.items[i,CONS], true
            )
            k = i - 1
        end
    else
        acceptable = false
        NWSE = "UB"
    end

    filter.pos = i

    return acceptable, NWSE
end

function sufficient_reduction(f, hc, func1, cons1, redn1, mu1, cons2, both)::Bool

    alpha = 0.25
    alpha2 = 1e-4
    beta = 0.99
    acceptable = false

    # sufficient reduction in f
    redn_f = max(alpha * redn1, alpha2 * cons1 * mu1)
    suff_redn_f = f <= (func1 - redn_f)

    # sufficient reduction in ||c||
    suff_redn_hc = hc < (beta * cons2)

    if hc == 0.0 && cons2 == 0.0
        both = true
    end

    if both
        acceptable = suff_redn_f && suff_redn_hc
    else
        acceptable = suff_redn_f || suff_redn_hc
    end

    return acceptable
end

FUNC = 1
CONS = 2
REDN = 3
PEN  = 4

mutable struct Filter{Tv}
    pos::Int
    len::Int
    maxf::Int
    items::Tv

    function Filter(maxf)
        items = Tv(undef, (maxf,4))
        return new(1, 0, maxf, items)
    end
end

function add_entry!(f, hc, mu, cs_ubd, filter::Filter)
    i = filter.pos

    # add an acceptable (f, hc) to the filter
    if i == 0
        # 1st point on filter
        replace = false
        i = 1
    elseif i == 1
        if f <= filter.items[1,FUNC]
            # replace the left most entry on the filter
            replace = true
        else
            replace = false
        end
    elseif hc == filter.items[i-1,CONS]
        # replace entry i-1 on filter
        replace = true
        i = i - 1
    elseif i <= filter.len && f <= filter.items[i,FUNC]
        # replace entry i on filter
        replace = true
    else
        # add entry before upper bound
        replace = false
    end

    if replace
        # replace entry i on the filter
        @debug "acceptable to filter..."
        @debug " ...replace τ by $(filter.items[i,FUNC]), $(f)"
        @debug " ...replace θ by $(filter.items[i,CONS]), $(hc)"
        filter.items[i,FUNC] = f
        filter.items[i,CONS] = hc
        filter.items[i,REDN] = 0.0
        filter.items[i,RED] = mu
        filter.pos = i

        # remove redundant entries from the filter
        ii = 0
        for k = i+1:filter.len
            if f <= filter.items[k,FUNC] && hc <= filter.items[k,CONS]
                ii += 1
            else
                break
            end
        end

        if ii > 0
            rshift!(filter.items, ii, filter.len - ii, 1)
            filter.len -= ii
        end
    else
        if filter.len + 1 > filter.maxf
            @debug "WARNING: filter too small: max $(maxf), length $(filter.len)"
            # remove last entry (upper bound)
            filter.len -= 1
            # transform last-1 entry into upper bound
            @debug "Add upper bnd on c/s violation $(cs_ubd)"
            filter.items[filter.len,FUNC] = -Inf
            filter.items[filter.len,CONS] = cs_ubd
            filter.items[filter.len,REDN] = 0.0
            filter.items[filter.len,PEN] = 1.0
            # if new entry was in last interval then need to change 
            if i == filter.maxf
                i -= 1
            end
        end
        filter.len += 1
        # shift entries to right (new entry in center)
        shift_right!(filter.items, i, filter.len - i)
        # add new entry to filter
        filter.items[i,FUNC] = f
        filter.items[i,CONS] = hc
        filter.items[i,REDN] = 0.0
        filter.items[i,PEN] = mu
        filter.pos = i
        @debug "acceptable to filter; add $(f), $(hc)"
    end
end

function rshift!(r, i0, n, k)
    if k > 0
        for i=i0:(n+i0), j=1:4
            r[i,j] = r[i+k,j]
        end
    elseif k < 0, j=1:4
        for i = (n+i0):-1:i0
            r[i,j] - r[i+k,j]
        end
    end
end

function shift_right!(r, i0, n)
    for i=(n+i0):-1:i0, j=1:4
        r[i,j] = r[i-1,j]
    end
end

function update_QP!(sqp::filterSQP)
    sqp.problem.eval_grad_f(sqp.x, sqp.df)
    eval_Jacobian!(sqp)

    sqp.f = sqp.fnew
    sqp.hc = sqp.hcnew
    sqp.E .= sqp.Enew
    sqp.x .= sqp.xnew

    if sqp.qp.status != 1
        # copy multipliers from r into lam (re--order)
        # TODO
    else
        @warn "Multipliers NOT updated, unbdd QP: $(sqp.qp.status)"
    end

    eval_Hessian!(sqp)
end

function conv_test(sqp::filterSQP, d_norm)
    convergence = false

    if sqp.hc <= sqp.eps
        # w = g - A.lam(n+1:n+m) - lam(1:n) is the KT residual vector
        # (i) w = 0.D0
        w = zeros(sqp.problem.n)
        # (ii) w = w + g 
        w = deepcopy(sqp.df)
        almx = max(1.0, norm(w))
        maxlam = 0.0
        # (iii) w = w - A.lam
        w .-= sqp.mult_x # TODO: mult_x_L + mult_x_U
        almx = max(almx, maximum(sqp.mult_x))
        maxlam = max(maxlam, maximum(sqp.mult_x))
        for i=1:sqp.problem.m
            w .-= sqp.lambda[i] * sqp.Jacobian[i,:]'
            almx = max(almx, abs(sqp.lambda[i]) * sum(sqp.Jacobian[i,:]))
            maxlam = max(maxlam, abs(sqp.lambda[i]))
        end

        rKT = norm(w)

        @debug "Convergence Test:"
        @debug " Norm of KT residual     = $(rKT)"
        @debug " max( |g|, |a_i*lam_i| ) = $(almx)"
        @debug " largest multiplier      = $(maxlam)"
        @debug " Norm of step            = $(d_norm)"
        @debug " Norm of c/s violation   = $(sqp.hc)"

        if sqp.f < sqp.options.fmin
            convergence = true
            sqp.status = 0
            @info "STOP: UNBOUNDED NLP DETECTED"
            @info "      f = $(sqp.f) < fmin = $(sqp.options.fmin)"
        elseif d_norm <= sqp.eps || rKT <= sqp.eps
            convergence = true
            sqp.status = 0
            @info "CONVERGENCE: ALGORITHM STOPS"
        end
    elseif d_norm <= sqp.eps
        convergence = true
        if sqp.hc <= sqp.eps
            sqp.status = 0
        else
            sqp.status = 3
        end
        @info "STOP: small step; solution may not be optimal"
    end

    return convergence
end

function enlarge_TR(Δ, d_norm)
    if Δ == d_norm
        @debug "Increase TR radius from $(Δ) to $(2.0*Δ)"
        Δ *= 2.0
    end
    return Δ
end

function reduce_TR(Δ, d_norm)
    return min(Δ, d_norm) / 2.0
end

function unblock_filter(f, hc, mu, filter::Filter, cs_ubd, phase, NWSE)

    # printed output depends on phase being used
    if phase == 1
        string1 = "||c||_J"
        string2 = "|c|_J  "
        string3 = "|c|_Jt "
        min_f = -Inf
        hc1 = f + hc
    elseif phase == 2
        string1 = "[f,|c|]"
        string2 = "   f   "
        string3 = "  |c|  "
        min_f = -Inf
        hc1 = hc
    else
        @error "Wrong phase in add_to_filter; STOP" phase
    end

    unblocked = false
    # check whether upper bound is blocking entry OR total c/s violn
    ubd_block = false
    acceptable = sufficient_reduction(0.0, hc1, Inf, 0.0, 0.0, 1.0, filter.items[filter.len,CONS], true)

    if !acceptable
        @debug "  Blocking entry is upper bnd on theta:  DO NOT REMOVE THIS ENTRY."
        NWSE = "UB"
        ubd_block = true
    else
        # find all blocking (thetap, taup) in the filter
        n_block = 0
        for i=1:filter.len-1
            acceptable = sufficient_reduction(
                f, hc, 
                filter.items[i,FUNC], filter.items[i,CONS], 
                filter.items[i,REDN], filter.items[i,PEN], 
                filter.items[i,CONS], 
                false
            )
            if !acceptable
                unblocked = true
                @debug "Blocking [taup,thetap] in filter found"
                # mark entry for removal from filter
                n_block += 1
                block_index[n_block] = i
            end
        end
    end

    if unblocked
        cs_ubd = max(cs_ubd / 10.0, hc)
        @debug "UNBLOCKING FILTER:"
        @debug "Reduce c/s upper bound to $(cs_bud)"
        add_entry!(min_f, cs_ubd, 1.0, cs_ubd, filter)

        # remove blocking entries (in list) from filter & find new f_posn
        filter.pos = filter.len + 1
        for i = n_block:-1:1
            ii = block_index[i]
            filter.pos = min(filter.pos, ii)
            @debug "Remove entry $(ii) : " filter.items[ii,FUNC] filter.items[ii,CONS]
            rshift!(filter.items, ii, filter.len, 1)
            filter.len -= 1
        end
        if abs(cs_ubd-hc) <= eps
            @debug "Blocked entry lies on upper bound"
            add_entry!(min_f, cs_ubd, 1.0, cs_ubd, filter)
        else
            @debug "Add blocked entry to filter:" f hc
            add_entry!(f, hc, mu, cs_ubd, filter)
        end
    elseif !ubd_block
        if NWSE == "NW" || NWSE == "SE"
            unblocked = true
            @debug "Blocking caused by $(NWSE) corner"
            if NWSE == "NW"
                filter.pos = 1
            else
                filter.pos = filter.len
            end
            add_entry!(f, hc, mu, cs_ubd, filter)
        else
            unblocked = true
            @warn "NOT NW or SE"
        end
    else
        if !ubd_block
            @warn "Cannot unblock filter"
        end
    end

    return unblocked, NWSE
end

function SOCS_QP!(sqp::filterSQP, phase, n_iatt, iatt)
    # compute w = A'.d  
    sqp.problem.eval_g(sqp.p, sqp.E_soc)

    # shift the bounds, also computing the new bounds on x
    bound_shift!(
        sqp.qp.n, sqp.qp.m,
        sqp.blo, sqp.blu, sqp.qp.bl, sqp.qp.bu,
        sqp.x, sqp.Enew, sqp.qp.cstype, sqp.Δ, 0.0, sqp.E_soc, true
    )

    # relax general c/s in phase I sum if phase==1
    if phase ==1
        @debug "Relaxing general bounds for SOCS"
        for i=1:n_iatt
            j = abs(iatt[i])
            if iatt[i] > 0
                sqp.qp.bl[j] = -Inf
            else
                sqp.qp.bu[j] = +Inf
            end
        end
    end

    # solve QP to get Second Order Correction Step
    solve!(sqp.qp)

    # check for unbounded QP; take step if necessary
    if sqp.qp.status == 1
        # TODO: we need to know the indices of actvie/inactive constraints: ls
        @error "TODO: unbddQP"
    end

    @debug "SOCS status" sqp.qp.status
end

function unbddQP(n, m, k, alpha, d, ls, w)
    # compute the search direction which led to an unbounded QP
    for j = n-k+1:n+m # going over inactive constraints
        alsj = abs(ls[j])
        if alsj <= n
            d[alsj] -= alpha * sign(ls[j] * w[alsj])
        end
    end
end
