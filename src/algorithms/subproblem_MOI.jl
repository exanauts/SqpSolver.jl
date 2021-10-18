mutable struct QpModel{T,Tv<:AbstractArray{T},Tm<:AbstractMatrix{T}} <: AbstractSubOptimizer
    model::MOI.AbstractOptimizer
    data::QpData{T,Tv,Tm}
    adj::Vector{Int}
    x::Vector{MOI.VariableIndex}
    constr_v_ub::Vector{MOI.ConstraintIndex}
    constr_v_lb::Vector{MOI.ConstraintIndex}
    constr::Vector{MOI.ConstraintIndex}
    slack_vars::Dict{Int,Vector{MOI.VariableIndex}}
    constr_slack::Vector{MOI.ConstraintIndex}

    function QpModel(
        model::MOI.AbstractOptimizer,
        data::QpData{T,Tv,Tm},
    ) where {T,Tv,Tm}
        qp = new{T,Tv,Tm}()
        qp.model = model
        qp.data = data
        qp.adj = []
        qp.x = []
        qp.constr_v_ub = []
        qp.constr_v_lb = []
        qp.constr = []
        qp.constr_slack = []
        qp.slack_vars = Dict()
        return qp
    end
end

SubOptimizer(model::MOI.AbstractOptimizer, data::QpData{T,Tv,Tm}) where {T,Tv,Tm} = QpModel(model, data)

function create_model!(qp::QpModel{T,Tv,Tm}, x_k::Tv, Δ::T, tol_error = 0.0) where {T,Tv,Tm}

    # empty optimizer just in case
    MOI.empty!(qp.model)
    qp.adj = []
    qp.constr_v_ub = []
    qp.constr_v_lb = []
    qp.constr = []
    qp.constr_slack = []
    empty!(qp.slack_vars)

    n = length(qp.data.c)
    m = length(qp.data.c_lb)

    @assert n > 0
    @assert m >= 0
    @assert length(qp.data.c) == n
    @assert length(qp.data.c_lb) == m
    @assert length(qp.data.c_ub) == m
    @assert length(qp.data.v_lb) == n
    @assert length(qp.data.v_ub) == n
    @assert length(x_k) == n

    # variables
    qp.x = MOI.add_variables(qp.model, n)

    # objective function
    obj_terms = Array{MOI.ScalarAffineTerm{T},1}()
    for i = 1:n
        push!(obj_terms, MOI.ScalarAffineTerm{T}(qp.data.c[i], MOI.VariableIndex(i)))
    end

    for i = 1:m
        # add slack variables
        qp.slack_vars[i] = []
        push!(qp.slack_vars[i], MOI.add_variable(qp.model))
        if qp.data.c_lb[i] > -Inf && qp.data.c_ub[i] < Inf
            push!(qp.slack_vars[i], MOI.add_variable(qp.model))
        end

        # Set slack bounds and objective coefficient
        push!(
            qp.constr_slack,
            MOI.add_constraint(
                qp.model,
                MOI.SingleVariable(qp.slack_vars[i][1]),
                MOI.GreaterThan(0.0),
            ),
        )
        push!(obj_terms, MOI.ScalarAffineTerm{T}(1.0, qp.slack_vars[i][1]))
        if length(qp.slack_vars[i]) == 2
            push!(
                qp.constr_slack,
                MOI.add_constraint(
                    qp.model,
                    MOI.SingleVariable(qp.slack_vars[i][2]),
                    MOI.GreaterThan(0.0),
                ),
            )
            push!(obj_terms, MOI.ScalarAffineTerm{T}(1.0, qp.slack_vars[i][2]))
        end
    end

    # set objective function
    if isnothing(qp.data.Q)
        MOI.set(
            qp.model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            MOI.ScalarAffineFunction(obj_terms, 0.0),
        )
    else
        Q_terms = Array{MOI.ScalarQuadraticTerm{T},1}()
        for j = 1:qp.data.Q.n, i in nzrange(qp.data.Q, j)
            if i >= j
                push!(
                    Q_terms, 
                    MOI.ScalarQuadraticTerm{T}(
                        qp.data.Q.nzval[i], 
                        MOI.VariableIndex(qp.data.Q.rowval[i]),
                        MOI.VariableIndex(j)
                    )
                )
            end
        end
		MOI.set(qp.model,
			MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}(),
			MOI.ScalarQuadraticFunction(obj_terms, Q_terms, 0.0))
    end
    MOI.set(qp.model, MOI.ObjectiveSense(), qp.data.sense)

    for i = 1:n
        ub = min(Δ, qp.data.v_ub[i] - x_k[i])
        lb = max(-Δ, qp.data.v_lb[i] - x_k[i])
        ub = (abs(ub) <= tol_error) ? 0.0 : ub
        lb = (abs(lb) <= tol_error) ? 0.0 : lb
        push!(
            qp.constr_v_ub,
            MOI.add_constraint(qp.model, MOI.SingleVariable(qp.x[i]), MOI.LessThan(ub)),
        )
        push!(
            qp.constr_v_lb,
            MOI.add_constraint(qp.model, MOI.SingleVariable(qp.x[i]), MOI.GreaterThan(lb)),
        )
    end

    for i = 1:m
        c_ub = qp.data.c_ub[i] - qp.data.b[i]
        c_lb = qp.data.c_lb[i] - qp.data.b[i]
        c_ub = (abs(c_ub) <= tol_error) ? 0.0 : c_ub
        c_lb = (abs(c_lb) <= tol_error) ? 0.0 : c_lb

        if qp.data.c_lb[i] == qp.data.c_ub[i] #This means the constraint is equality
            push!(
                qp.constr,
                MOI.add_constraint(
                    qp.model,
                    MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.(
                            [1.0; -1.0],
                            [qp.slack_vars[i][1]; qp.slack_vars[i][2]],
                        ),
                        0.0,
                    ),
                    MOI.EqualTo(c_lb),
                ),
            )
        elseif qp.data.c_lb[i] != -Inf &&
               qp.data.c_ub[i] != Inf &&
               qp.data.c_lb[i] < qp.data.c_ub[i]
            push!(
                qp.constr,
                MOI.add_constraint(
                    qp.model,
                    MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.([1.0], [qp.slack_vars[i][1]]),
                        0.0,
                    ),
                    MOI.GreaterThan(c_lb),
                ),
            )
            push!(qp.adj, i)
        elseif qp.data.c_lb[i] != -Inf
            push!(
                qp.constr,
                MOI.add_constraint(
                    qp.model,
                    MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.([1.0], [qp.slack_vars[i][1]]),
                        0.0,
                    ),
                    MOI.GreaterThan(c_lb),
                ),
            )
        elseif qp.data.c_ub[i] != Inf
            push!(
                qp.constr,
                MOI.add_constraint(
                    qp.model,
                    MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.([-1.0], [qp.slack_vars[i][1]]),
                        0.0,
                    ),
                    MOI.LessThan(c_ub),
                ),
            )
        end
    end

    for i in qp.adj
        c_ub = qp.data.c_ub[i] - qp.data.b[i]
        c_ub = (abs(c_ub) <= tol_error) ? 0.0 : c_ub
        push!(
            qp.constr,
            MOI.add_constraint(
                qp.model,
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.([-1.0], [qp.slack_vars[i][2]]),
                    0.0,
                ),
                MOI.LessThan(c_ub),
            ),
        )
    end
end

"""
	sub_optimize!

Solve subproblem

# Arguments
- `qp`: QP model
- `x_k`: trust region center
- `Δ`: trust region size
- `feasibility`: indicator for feasibility restoration phase 
- `tol_error`: threshold to drop small numbers to zeros
"""
function sub_optimize!(
    qp::QpModel{T,Tv,Tm},
    x_k::Tv,
    Δ::T,
    feasibility = false,
    tol_error = 0.0,
) where {T,Tv,Tm}

    # dimension of LP
    m, n = size(qp.data.A)
    @assert n > 0
    @assert m >= 0
    @assert length(qp.data.c) == n
    @assert length(qp.data.c_lb) == m
    @assert length(qp.data.c_ub) == m
    @assert length(qp.data.v_lb) == n
    @assert length(qp.data.v_ub) == n
    @assert length(x_k) == n

    b = deepcopy(qp.data.b)

    if feasibility
        if isnothing(qp.data.Q)
            # modify objective coefficient
            for i = 1:n
                MOI.modify(
                    qp.model,
                    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                    MOI.ScalarCoefficientChange(MOI.VariableIndex(i), 0.0),
                )
            end

            # modify slack objective coefficient
            for (_, slacks) in qp.slack_vars, s in slacks
                MOI.modify(
                    qp.model,
                    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                    MOI.ScalarCoefficientChange(s, 1.0),
                )
            end
        else
            # Set new QP objective function again
            obj_terms = Array{MOI.ScalarAffineTerm{T},1}()
            for (_, slacks) in qp.slack_vars, s in slacks
                push!(obj_terms, MOI.ScalarAffineTerm{T}(1.0, s))
            end
            MOI.set(
                qp.model, 
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}(),
			    MOI.ScalarQuadraticFunction(
                    obj_terms,
                    Array{MOI.ScalarQuadraticTerm{T},1}(),
                    0.0
                )
            )
        end

        # set optimization sense
        MOI.set(qp.model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        do_transform = false
        for cons in qp.constr_slack
            if typeof(cons) == MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{T}}
                do_transform = true
                break
            end
        end

        # set slack variable bounds
        constr_index = 1
        for i = 1:m
            # Adjust parameters for feasibility problem
            viol = 0.0
            if qp.data.b[i] > qp.data.c_ub[i]
                viol = qp.data.c_ub[i] - qp.data.b[i]
            elseif qp.data.b[i] < qp.data.c_lb[i]
                viol = qp.data.c_lb[i] - qp.data.b[i]
            end
            b[i] -= abs(viol)

            # Add bound constraints
            if length(qp.slack_vars[i]) == 2
                if viol < 0
                    if do_transform
                        qp.constr_slack[constr_index] = MOI.transform(
                            qp.model,
                            qp.constr_slack[constr_index],
                            MOI.GreaterThan(0.0),
                        )
                    else
                        MOI.set(
                            qp.model,
                            MOI.ConstraintSet(),
                            qp.constr_slack[constr_index],
                            MOI.GreaterThan(0.0),
                        )
                    end
                    constr_index += 1
                    if do_transform
                        qp.constr_slack[constr_index] = MOI.transform(
                            qp.model,
                            qp.constr_slack[constr_index],
                            MOI.GreaterThan(viol),
                        )
                    else
                        MOI.set(
                            qp.model,
                            MOI.ConstraintSet(),
                            qp.constr_slack[constr_index],
                            MOI.GreaterThan(viol),
                        )
                    end
                    constr_index += 1
                else
                    if do_transform
                        qp.constr_slack[constr_index] = MOI.transform(
                            qp.model,
                            qp.constr_slack[constr_index],
                            MOI.GreaterThan(-viol),
                        )
                    else
                        MOI.set(
                            qp.model,
                            MOI.ConstraintSet(),
                            qp.constr_slack[constr_index],
                            MOI.GreaterThan(-viol),
                        )
                    end
                    constr_index += 1
                    if do_transform
                        qp.constr_slack[constr_index] = MOI.transform(
                            qp.model,
                            qp.constr_slack[constr_index],
                            MOI.GreaterThan(0.0),
                        )
                    else
                        MOI.set(
                            qp.model,
                            MOI.ConstraintSet(),
                            qp.constr_slack[constr_index],
                            MOI.GreaterThan(0.0),
                        )
                    end
                    constr_index += 1
                end
            elseif length(qp.slack_vars[i]) == 1
                if do_transform
                    qp.constr_slack[constr_index] = MOI.transform(
                        qp.model,
                        qp.constr_slack[constr_index],
                        MOI.GreaterThan(-abs(viol)),
                    )
                else
                    MOI.set(
                        qp.model,
                        MOI.ConstraintSet(),
                        qp.constr_slack[constr_index],
                        MOI.GreaterThan(-abs(viol)),
                    )
                end
                # @show i, viol, length(qp.slack_vars[i]), qp.constr_slack[constr_index]
                constr_index += 1
            else
                @error "unexpected slack_vars"
            end
        end
    else
        if isnothing(qp.data.Q)
            # modify objective coefficient
            for i = 1:n
                MOI.modify(
                    qp.model,
                    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                    MOI.ScalarCoefficientChange(MOI.VariableIndex(i), qp.data.c[i]),
                )
            end

            # set slack objective coefficient
            for (_, slacks) in qp.slack_vars, s in slacks
                MOI.modify(
                    qp.model,
                    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                    MOI.ScalarCoefficientChange(s, 0.0),
                )
            end
        else
            # Set new QP objective function again
            obj_terms = Array{MOI.ScalarAffineTerm{T},1}()
            for i = 1:n
                push!(obj_terms, MOI.ScalarAffineTerm{T}(qp.data.c[i], MOI.VariableIndex(i)))
            end
            for (_, slacks) in qp.slack_vars, s in slacks
                push!(obj_terms, MOI.ScalarAffineTerm{T}(0.0, s))
            end
            Q_terms = Array{MOI.ScalarQuadraticTerm{T},1}()
            for j = 1:qp.data.Q.n, i in nzrange(qp.data.Q, j)
                if i >= j
                    push!(
                        Q_terms, 
                        MOI.ScalarQuadraticTerm{T}(
                            qp.data.Q.nzval[i], 
                            MOI.VariableIndex(qp.data.Q.rowval[i]),
                            MOI.VariableIndex(j)
                        )
                    )
                end
            end
            MOI.set(
                qp.model, 
                MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}(),
			    MOI.ScalarQuadraticFunction(obj_terms, Q_terms, 0.0)
            )
        end

        # set optimization sense
        MOI.set(qp.model, MOI.ObjectiveSense(), qp.data.sense)

        # set slack variable bounds
        do_transform = false
        for cons in qp.constr_slack
            if typeof(cons) != MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{T}}
                do_transform = true
                break
            end
        end
        if do_transform
            for i in eachindex(qp.constr_slack)
                qp.constr_slack[i] =
                    MOI.transform(qp.model, qp.constr_slack[i], MOI.EqualTo(0.0))
            end
        end
    end

    # set variable bounds
    for i = 1:n
        ub = min(Δ, qp.data.v_ub[i] - x_k[i])
        lb = max(-Δ, qp.data.v_lb[i] - x_k[i])
        ub = (abs(ub) <= tol_error) ? 0.0 : ub
        lb = (abs(lb) <= tol_error) ? 0.0 : lb
        MOI.set(qp.model, MOI.ConstraintSet(), qp.constr_v_ub[i], MOI.LessThan(ub))
        MOI.set(qp.model, MOI.ConstraintSet(), qp.constr_v_lb[i], MOI.GreaterThan(lb))
    end
    # @show Δ, qp.data.v_lb, qp.data.v_ub, x_k

    # modify the constraint coefficients
    for j = 1:qp.data.A.n, i in nzrange(qp.data.A, j)
        coeff = abs(qp.data.A.nzval[i]) <= tol_error ? 0.0 : qp.data.A.nzval[i]
        MOI.modify(
            qp.model,
            qp.constr[qp.data.A.rowval[i]],
            MOI.ScalarCoefficientChange(MOI.VariableIndex(j), coeff),
        )
    end
    for (ind, val) in enumerate(qp.adj)
        row_of_A = qp.data.A[val, :]
        for i = 1:row_of_A.n
            j = row_of_A.nzind[i]
            coeff = abs(row_of_A.nzval[i]) <= tol_error ? 0.0 : row_of_A.nzval[i]
            MOI.modify(
                qp.model,
                qp.constr[m+ind],
                MOI.ScalarCoefficientChange(MOI.VariableIndex(j), coeff),
            )
        end
    end

    # modify RHS
    for i = 1:m
        c_ub = qp.data.c_ub[i] - b[i]
        c_lb = qp.data.c_lb[i] - b[i]
        c_ub = (abs(c_ub) <= tol_error) ? 0.0 : c_ub
        c_lb = (abs(c_lb) <= tol_error) ? 0.0 : c_lb

        if qp.data.c_lb[i] == qp.data.c_ub[i]
            MOI.set(qp.model, MOI.ConstraintSet(), qp.constr[i], MOI.EqualTo(c_lb))
        elseif qp.data.c_lb[i] != -Inf &&
               qp.data.c_ub[i] != Inf &&
               qp.data.c_lb[i] < qp.data.c_ub[i]
            MOI.set(qp.model, MOI.ConstraintSet(), qp.constr[i], MOI.GreaterThan(c_lb))
        elseif qp.data.c_lb[i] != -Inf
            MOI.set(qp.model, MOI.ConstraintSet(), qp.constr[i], MOI.GreaterThan(c_lb))
        elseif qp.data.c_ub[i] != Inf
            MOI.set(qp.model, MOI.ConstraintSet(), qp.constr[i], MOI.LessThan(c_ub))
        end
    end
    @show qp.data.c_lb-b, qp.data.c_ub-b, b
    for (i, val) in enumerate(qp.adj)
        c_ub = qp.data.c_ub[val] - b[val]
        c_ub = (abs(c_ub) <= tol_error) ? 0.0 : c_ub
        MOI.set(qp.model, MOI.ConstraintSet(), qp.constr[i+m], MOI.LessThan(c_ub))
    end

    # dest = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_LP)
    # MOI.copy_to(dest, qp.model)
    # MOI.write_to_file(dest, "debug_moi.lp")

    MOI.optimize!(qp.model)
    status = MOI.get(qp.model, MOI.TerminationStatus())

    # TODO: These can be part of data.
    Xsol = Tv(undef, n)
    p_slack = Dict{Int,Vector{Float64}}()
    lambda = Tv(undef, m)
    mult_x_U = Tv(undef, n)
    mult_x_L = Tv(undef, n)

    if status == MOI.OPTIMAL
        # @show MOI.get(qp.model, MOI.ObjectiveValue())
        Xsol .= MOI.get(qp.model, MOI.VariablePrimal(), qp.x)
        for (i, slacks) in qp.slack_vars
            p_slack[i] = MOI.get(qp.model, MOI.VariablePrimal(), slacks)
        end
        @show MOI.get(qp.model, MOI.ObjectiveValue()), Xsol
        # @show p_slack

        # extract the multipliers to constraints
        for i = 1:m
            lambda[i] = MOI.get(qp.model, MOI.ConstraintDual(1), qp.constr[i])
        end
        for (i, val) in enumerate(qp.adj)
            lambda[val] += MOI.get(qp.model, MOI.ConstraintDual(1), qp.constr[i+m])
        end
        # @show MOI.get(qp.model, MOI.ConstraintDual(1), qp.constr)

        # extract the multipliers to column bounds
        mult_x_U .= MOI.get(qp.model, MOI.ConstraintDual(1), qp.constr_v_ub)
        mult_x_L .= MOI.get(qp.model, MOI.ConstraintDual(1), qp.constr_v_lb)
        # careful because of the trust region
        for j = 1:n
            if Xsol[j] < qp.data.v_ub[j] - x_k[j]
                mult_x_U[j] = 0.0
            end
            if Xsol[j] > qp.data.v_lb[j] - x_k[j]
                mult_x_L[j] = 0.0
            end
        end
    elseif status == MOI.DUAL_INFEASIBLE
        @error "Trust region must be employed."
    elseif status == MOI.INFEASIBLE
        fill!(Xsol, 0.0)
        fill!(lambda, 0.0)
        fill!(mult_x_U, 0.0)
        fill!(mult_x_L, 0.0)
    else
        @error "Unexpected status: $(status)"
    end

    return Xsol, lambda, mult_x_U, mult_x_L, p_slack, status
end
