mutable struct QpJuMP{T,Tv<:AbstractArray{T},Tm<:AbstractMatrix{T}} <: AbstractSubOptimizer
    model::JuMP.Model
    data::QpData{T,Tv,Tm}
    x::Vector{JuMP.VariableRef}
    constr::Vector{JuMP.ConstraintRef}
    rngbdcons::Vector{Int}
    rngcons::Vector{Int}
    slack_vars::Dict{Int,Vector{JuMP.VariableRef}}

    xsol::Tv
    λ::Tv
    μ::Tv
    status

    function QpJuMP(model::JuMP.AbstractModel, data::QpData{T,Tv,Tm}) where {T,Tv,Tm}
        qp = new{T,Tv,Tm}()
        qp.model = model
        qp.data = data
        qp.x = []
        qp.constr = []
        qp.rngbdcons = []
        qp.rngcons = []
        qp.slack_vars = Dict()
        qp.xsol = []
        qp.λ = []
        qp.μ = []
        qp.status = MOI.OPTIMIZE_NOT_CALLED
        return qp
    end
end

SubOptimizer(model::JuMP.AbstractModel, data::QpData{T,Tv,Tm}) where {T,Tv,Tm} =
    QpJuMP(model, data)

"""
    create_model!

Initialize QP subproblem in JuMP.Model. The model assumes that the first `qp.data.num_linear_constraints` constraints are linear.
The slack variables are not introduced for the linear constraints.

# Arguments
- `qp`
- `Δ`: trust-region size
"""
function create_model!(
    qp::QpJuMP{T,Tv,Tm}, 
    Δ::T,
) where {T,Tv,Tm}

    qp.constr = []
    qp.rngbdcons = []
    qp.rngcons = []
    empty!(qp.slack_vars)

    n = length(qp.data.c)
    m = length(qp.data.c_lb)

    qp.xsol = Tv(undef, n)
    qp.λ = Tv(undef, m)
    qp.μ = Tv(undef, n)

    # create nominal variables
    qp.x = @variable(
        qp.model,
        [i = 1:n],
        base_name = "x",
    )

    set_trust_region!(qp, Δ)

    # add slack variables only for nonlinear constraints
    for i = (qp.data.num_linear_constraints+1):m
        qp.slack_vars[i] = []
        push!(qp.slack_vars[i], @variable(qp.model, base_name = "u[$i]", lower_bound = 0.0))
        if qp.data.c_lb[i] > -Inf && qp.data.c_ub[i] < Inf
            push!(qp.slack_vars[i], @variable(qp.model, base_name = "v[$i]", lower_bound = 0.0))
        end
    end

    # dummy objective function
    @objective(qp.model, Min, 0.0)

    # create affine constraints
    for i = 1:m
        c_ub = qp.data.c_ub[i]
        c_lb = qp.data.c_lb[i]
        if abs(qp.data.b[i]) < Inf
            c_ub -= qp.data.b[i]
            c_lb -= qp.data.b[i]
        end

        if qp.data.c_lb[i] == qp.data.c_ub[i] #This means the constraint is equality
            if i <= qp.data.num_linear_constraints
                Arow = qp.data.A[i,:]
                push!(
                    qp.constr, 
                    @constraint(qp.model, sum(v * qp.x[Arow.nzind[j]] for (j,v) in enumerate(Arow.nzval)) == c_lb)
                )
            else
                push!(qp.constr, @constraint(qp.model, qp.slack_vars[i][1] - qp.slack_vars[i][2] == c_lb))
            end
        elseif qp.data.c_lb[i] > -Inf && qp.data.c_ub[i] < Inf
            if i <= qp.data.num_linear_constraints
                Arow = qp.data.A[i,:]
                # push!(qp.constr, @constraint(qp.model, c_lb <= sum(A[i,j] * qp.x[j] for j in A[i,:].nzind) <= c_ub))
                push!(qp.constr, @constraint(qp.model, sum(v * qp.x[Arow.nzind[j]] for (j,v) in enumerate(Arow.nzval)) >= c_lb))
            else
                push!(qp.constr, @constraint(qp.model, qp.slack_vars[i][1] >= c_lb))
            end
            push!(qp.rngcons, i)
        elseif qp.data.c_lb[i] > -Inf
            if i <= qp.data.num_linear_constraints
                Arow = qp.data.A[i,:]
                push!(qp.constr, @constraint(qp.model, sum(v * qp.x[Arow.nzind[j]] for (j,v) in enumerate(Arow.nzval)) >= c_lb))
            else
                push!(qp.constr, @constraint(qp.model, qp.slack_vars[i][1] >= c_lb))
            end
        elseif qp.data.c_ub[i] < Inf
            if i <= qp.data.num_linear_constraints
                Arow = qp.data.A[i,:]
                push!(qp.constr, @constraint(qp.model, sum(v * qp.x[Arow.nzind[j]] for (j,v) in enumerate(Arow.nzval)) <= c_ub))
            else
                push!(qp.constr, @constraint(qp.model, -qp.slack_vars[i][1] <= c_ub))
            end
        end
    end

    # create ranged affine constraints
    for i in qp.rngcons
        c_ub = qp.data.c_ub[i] - qp.data.b[i]
        if i <= qp.data.num_linear_constraints
            Arow = qp.data.A[i,:]
            push!(qp.constr, @constraint(qp.model, sum(v * qp.x[Arow.nzind[j]] for (j,v) in enumerate(Arow.nzval)) <= c_ub))
        else
            push!(qp.constr, @constraint(qp.model, -qp.slack_vars[i][2] <= c_ub))
        end
    end
end

function sub_optimize!(
    qp::QpJuMP{T,Tv,Tm},
    x_k::Tv,
    Δ::T,
) where {T,Tv,Tm}

    # dimension of LP
    m, n = size(qp.data.A)

    # modify objective function
    if isnothing(qp.data.Q)
        @objective(qp.model, qp.data.sense, 
            sum(qp.data.c[i] * qp.x[i] for i = 1:n)
        )
    else
        @objective(
            qp.model, 
            qp.data.sense, 
            sum(qp.data.c[i] * qp.x[i] for i = 1:n)
            + 0.5 * sum(
                qp.data.Q.nzval[i] * qp.x[qp.data.Q.rowval[i]] * qp.x[j] 
                for j = 1:qp.data.Q.n for i in nzrange(qp.data.Q, j) #if abs(qp.data.Q.nzval[i]) > 1.e-5
            )
        )
    end

    # fix slack variables to zeros
    for (_, slacks) in qp.slack_vars, s in slacks
        if JuMP.has_lower_bound(s)
            JuMP.delete_lower_bound(s)
        end
        JuMP.fix(s, 0.0)
    end

    set_trust_region!(qp, x_k, Δ)
    modify_constraints!(qp)

    # @show x_k
    # JuMP.print(qp.model)
    JuMP.optimize!(qp.model)
    qp.status = termination_status(qp.model)
    collect_solution!(qp)

    return
end

function sub_optimize_lp(
    optimizer,
    A::Tm, x_k::Tv,
    problem::AbstractSqpModel
) where {T, Tv<:AbstractArray{T}, Tm<:AbstractMatrix{T}}

    n = length(x_k)

    model = JuMP.Model(optimizer)
    @variable(model, problem.x_L[i] <= x[i=1:n] <= problem.x_U[i])
    @objective(model, Min, sum(x[i]^2 for i=1:n))
    constr = Vector{JuMP.ConstraintRef}(undef, problem.num_linear_constraints)
    for i = 1:problem.num_linear_constraints
        arow = A[i,:]
        if problem.g_L[i] == problem.g_U[i]
            constr[i] = @constraint(model, sum(a * x[arow.nzind[j]] for (j, a) in enumerate(arow.nzval)) == problem.g_L[i])
        elseif problem.g_L[i] > -Inf && problem.g_U[i] < Inf
            constr[i] = @constraint(model, problem.g_L[i] <= sum(a * x[arow.nzind[j]] for (j, a) in enumerate(arow.nzval)) <= problem.g_U[i])
        elseif problem.g_L[i] > -Inf
            constr[i] = @constraint(model, sum(a * x[arow.nzind[j]] for (j, a) in enumerate(arow.nzval)) >= problem.g_L[i])
        elseif problem.g_U[i] < Inf
            constr[i] = @constraint(model, sum(a * x[arow.nzind[j]] for (j, a) in enumerate(arow.nzval)) <= problem.g_U[i])
        end
    end
    JuMP.optimize!(model)
    status = termination_status(model)

    if status ∈ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.LOCALLY_SOLVED]
        problem.x .= JuMP.value.(x)

        # extract the multipliers to constraints
        for i = 1:problem.num_linear_constraints
            problem.mult_g[i] = JuMP.dual(constr[i])
        end

        # extract the multipliers to column bounds
        for i = 1:n
            redcost = JuMP.reduced_cost(x[i])
            if redcost > 0
                problem.mult_x_L[i] = redcost
            elseif redcost < 0
                problem.mult_x_U[i] = redcost
            end
        end
        status = :Optimal
    elseif status ∈ [MOI.LOCALLY_INFEASIBLE, MOI.INFEASIBLE]
        fill!(problem.x, 0.0)
        fill!(problem.mult_x_L, 0.0)
        fill!(problem.mult_x_U, 0.0)
        status = :Infeasible
    else
        @error "Unexpected status: $(status)"
    end
    return status
end

"""
Solve QP subproblem for feasibility restoration
"""
function sub_optimize_FR!(
    qp::QpJuMP{T,Tv,Tm},
    x_k::Tv,
    Δ::T,
) where {T,Tv,Tm}

    # dimension of LP
    m, n = size(qp.data.A)

    # modify objective function
    @objective(qp.model, Min, sum(s for (_, slacks) in qp.slack_vars, s in slacks))

    # modify slack variable bounds
    for (i, slacks) in qp.slack_vars
        if qp.data.b[i] >= qp.data.c_lb[i] && qp.data.b[i] <= qp.data.c_ub[i]
            for s in slacks
                if JuMP.is_fixed(s) == false
                    JuMP.fix(s, 0.0, force = true)
                end
            end
        else
            for s in slacks
                if JuMP.is_fixed(s)
                    JuMP.unfix(s)
                end
                set_lower_bound(s, 0.0)
            end
        end
    end

    set_trust_region!(qp, x_k, Δ)
    modify_constraints!(qp)

    # JuMP.write_to_file(qp.model, "debug_jump.lp", format = MOI.FileFormats.FORMAT_LP)
    # @show x_k
    # JuMP.print(qp.model)
    JuMP.optimize!(qp.model)
    qp.status = termination_status(qp.model)
    collect_solution!(qp)

    return
end

"""
Compute the infeasibility of the linearized model
"""
function sub_optimize_infeas(
    qp::QpJuMP{T,Tv,Tm},
    x_k::Tv,
    Δ::T,
) where {T,Tv,Tm}

    # modify objective function
    @objective(qp.model, Min, sum(s for (_, slacks) in qp.slack_vars, s in slacks))

    # modify slack variable bounds
    for (_, slacks) in qp.slack_vars, s in slacks
        if JuMP.is_fixed(s)
            JuMP.unfix(s)
        end
        set_lower_bound(s, 0.0)
    end

    set_trust_region!(qp, x_k, Δ)
    modify_constraints!(qp)

    JuMP.optimize!(qp.model)
    status = termination_status(qp.model)

    Xsol = Tv(undef, length(qp.x))
    infeasibility = Inf
    if status ∈ [MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.LOCALLY_SOLVED]
        Xsol .= JuMP.value.(qp.x)
        infeasibility = JuMP.objective_value(qp.model)
    end

    return Xsol, infeasibility
end


function set_trust_region!(
    x::Vector{JuMP.VariableRef},
    v_lb::Tv,
    v_ub::Tv,
    Δ::T
) where {T,Tv}
    for i in eachindex(x)
        set_lower_bound(x[i], max(-Δ, v_lb[i]))
        set_upper_bound(x[i], min(+Δ, v_ub[i]))
    end
end

function set_trust_region!(
    qp::QpJuMP{T,Tv,Tm},
    x_k::Tv,
    Δ::T
) where {T,Tv,Tm} 
    return set_trust_region!(qp.x, qp.data.v_lb - x_k, qp.data.v_ub - x_k, Δ)
end

function set_trust_region!(
    qp::QpJuMP{T,Tv,Tm},
    Δ::T
) where {T,Tv,Tm} 
    return set_trust_region!(qp.x, qp.data.v_lb, qp.data.v_ub, Δ)
end

function modify_constraints!(qp::QpJuMP{T,Tv,Tm}) where {T,Tv,Tm}

    # problem dimension
    m, n = size(qp.data.A)

    # modify the nonlinear constraint coefficients
    for j = 1:qp.data.A.n, i in nzrange(qp.data.A, j)
        if qp.data.A.rowval[i] > qp.data.num_linear_constraints
            set_normalized_coefficient(
                qp.constr[qp.data.A.rowval[i]],
                qp.x[j],
                qp.data.A.nzval[i],
            )
        end
    end

    # modify the coefficients for the other part of ranged constraints
    for (ind, val) in enumerate(qp.rngcons)
        if val > qp.data.num_linear_constraints
            row_of_A = qp.data.A[val, :]
            for (i,j) = enumerate(row_of_A.nzind)
                set_normalized_coefficient(qp.constr[m+ind], qp.x[j], row_of_A.nzval[i])
            end
        end
    end

    # modify RHS
    for i in 1:m
        c_ub = qp.data.c_ub[i] - qp.data.b[i]
        c_lb = qp.data.c_lb[i] - qp.data.b[i]

        if qp.data.c_lb[i] == qp.data.c_ub[i]
            set_normalized_rhs(qp.constr[i], c_lb)
        elseif qp.data.c_lb[i] > -Inf && qp.data.c_ub[i] < Inf
            set_normalized_rhs(qp.constr[i], c_lb)
        elseif qp.data.c_lb[i] > -Inf
            set_normalized_rhs(qp.constr[i], c_lb)
        elseif qp.data.c_ub[i] < Inf
            set_normalized_rhs(qp.constr[i], c_ub)
        end
    end

    # modify the RHS for the other part of ranged constraints
    for (i, val) in enumerate(qp.rngcons)
        c_ub = qp.data.c_ub[val] - qp.data.b[val]
        set_normalized_rhs(qp.constr[i+m], c_ub)
    end
end

function collect_solution!(qp::QpJuMP{T,Tv,Tm}) where {T,Tv,Tm}

    # problem dimension
    m, n = size(qp.data.A)

    if qp.status ∈ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.LOCALLY_SOLVED]
        qp.xsol .= JuMP.value.(qp.x)

        # extract the multipliers to constraints
        for i = 1:m
            qp.λ[i] = JuMP.dual(qp.constr[i])
        end
        for (i, val) in enumerate(qp.rngcons)
            qp.λ[val] += JuMP.dual(qp.constr[i+m])
        end
        # @show MOI.get(qp.model, MOI.ConstraintDual(1), qp.constr)

        # extract the multipliers to column bounds
        for i = 1:n
            qp.μ[i] = JuMP.reduced_cost(qp.x[i])
            # if redcost > 0
            #     mult_x_L[i] = redcost
            # elseif redcost < 0
            #     mult_x_U[i] = redcost
            # end
        end
    elseif qp.status ∈ [MOI.LOCALLY_INFEASIBLE, MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.NORM_LIMIT, MOI.OBJECTIVE_LIMIT]
        fill!(qp.xsol, 0.0)
        fill!(qp.λ, 0.0)
        fill!(qp.μ, 0.0)
    else
        @error "Unexpected status: $(qp.status)"
    end
end
