"""
"""

mutable struct _ConstraintInfo{F,S}
    func::F
    set::S
    dual_start::Union{Nothing,Float64}
end

_ConstraintInfo(func, set) = _ConstraintInfo(func, set, nothing)

"""
    Optimizer()

Create a new SQP optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{Model,Nothing}
    name::String
    invalid_model::Bool
    variables::MOI.Utilities.VariablesContainer{Float64}
    variable_primal_start::Vector{Union{Nothing,Float64}}
    variable_lower_start::Vector{Union{Nothing,Float64}}
    variable_upper_start::Vector{Union{Nothing,Float64}}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{
        Nothing,
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    }
    linear_le_constraints::Vector{
        _ConstraintInfo{
            MOI.ScalarAffineFunction{Float64},
            MOI.LessThan{Float64},
        },
    }
    linear_ge_constraints::Vector{
        _ConstraintInfo{
            MOI.ScalarAffineFunction{Float64},
            MOI.GreaterThan{Float64},
        },
    }
    linear_eq_constraints::Vector{
        _ConstraintInfo{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},
    }
    quadratic_le_constraints::Vector{
        _ConstraintInfo{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.LessThan{Float64},
        },
    }
    quadratic_ge_constraints::Vector{
        _ConstraintInfo{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.GreaterThan{Float64},
        },
    }
    quadratic_eq_constraints::Vector{
        _ConstraintInfo{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.EqualTo{Float64},
        },
    }
    nlp_dual_start::Union{Nothing,Vector{Float64}}
    silent::Bool
    options::Parameters
    solve_time::Float64
    callback::Union{Nothing,Function}

    function Optimizer(; kwargs...)
        prob = new(
            nothing,
            "",
            false,
            MOI.Utilities.VariablesContainer{Float64}(),
            Union{Nothing,Float64}[],
            Union{Nothing,Float64}[],
            Union{Nothing,Float64}[],
            MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
            MOI.FEASIBILITY_SENSE,
            nothing,
            _ConstraintInfo{
                MOI.ScalarAffineFunction{Float64},
                MOI.LessThan{Float64},
            }[],
            _ConstraintInfo{
                MOI.ScalarAffineFunction{Float64},
                MOI.GreaterThan{Float64},
            }[],
            _ConstraintInfo{
                MOI.ScalarAffineFunction{Float64},
                MOI.EqualTo{Float64},
            }[],
            _ConstraintInfo{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.LessThan{Float64},
            }[],
            _ConstraintInfo{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.GreaterThan{Float64},
            }[],
            _ConstraintInfo{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.EqualTo{Float64},
            }[],
            nothing,
            false,
            Parameters(),
            NaN,
            nothing,
        )
        for (k, v) in kwargs
            set_parameter(prob.options, string(k), v)
        end
        return prob
    end
end

MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.1.0"

### _EmptyNLPEvaluator

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing
MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing
MOI.jacobian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing
MOI.eval_hessian_lagrangian(::_EmptyNLPEvaluator, H, x, σ, μ) = nothing

function MOI.empty!(model::Optimizer)
    model.inner = nothing
    model.invalid_model = false
    MOI.empty!(model.variables)
    empty!(model.variable_primal_start)
    empty!(model.variable_lower_start)
    empty!(model.variable_upper_start)
    model.nlp_data = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    empty!(model.linear_le_constraints)
    empty!(model.linear_ge_constraints)
    empty!(model.linear_eq_constraints)
    empty!(model.quadratic_le_constraints)
    empty!(model.quadratic_ge_constraints)
    empty!(model.quadratic_eq_constraints)
    model.nlp_dual_start = nothing
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.variables) &&
           isempty(model.variable_primal_start) &&
           isempty(model.variable_lower_start) &&
           isempty(model.variable_upper_start) &&
           model.nlp_data.evaluator isa _EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE &&
           isempty(model.linear_le_constraints) &&
           isempty(model.linear_ge_constraints) &&
           isempty(model.linear_eq_constraints) &&
           isempty(model.quadratic_le_constraints) &&
           isempty(model.quadratic_ge_constraints) &&
           isempty(model.quadratic_eq_constraints)
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "SQP"

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
        },
    },
    ::Type{
        <:Union{
            MOI.LessThan{Float64},
            MOI.GreaterThan{Float64},
            MOI.EqualTo{Float64},
        },
    },
)
    return true
end

function MOI.get(model::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    ret = MOI.get(model.variables, MOI.ListOfConstraintTypesPresent())
    constraints = Set{Tuple{Type,Type}}()
    for F in (
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    )
        for S in (
            MOI.LessThan{Float64},
            MOI.GreaterThan{Float64},
            MOI.EqualTo{Float64},
        )
            if !isempty(_constraints(model, F, S))
                push!(constraints, (F, S))
            end
        end
    end
    return append!(ret, collect(constraints))
end

### MOI.Name

MOI.supports(::Optimizer, ::MOI.Name) = true

function MOI.set(model::Optimizer, ::MOI.Name, value::String)
    model.name = value
    return
end

MOI.get(model::Optimizer, ::MOI.Name) = model.name

### MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

### MOI.TimeLimitSec

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), Float64(value))
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), 1.0e+10)
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get_parameter(model.options, "time_limit")
end

### MOI.RawOptimizerAttribute

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    set_parameter(model.options, p.name, value)
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    return get_parameter(model.options, p.name)
end

### Variables

"""
    column(x::MOI.VariableIndex)
Return the column associated with a variable.
"""
column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(model::Optimizer)
    push!(model.variable_primal_start, nothing)
    push!(model.variable_lower_start, nothing)
    push!(model.variable_upper_start, nothing)
    return MOI.add_variable(model.variables)
end

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    return MOI.is_valid(model.variables, x)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.NumberOfVariables,MOI.ListOfVariableIndices},
)
    return MOI.get(model.variables, attr)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
) where {S<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo}}
    return MOI.is_valid(model.variables, ci)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.NumberOfConstraints{MOI.VariableIndex,S},
        MOI.ListOfConstraintIndices{MOI.VariableIndex,S},
    },
) where {S<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo}}
    return MOI.get(model.variables, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    c::MOI.ConstraintIndex{MOI.VariableIndex,S},
) where {S<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo}}
    return MOI.get(model.variables, attr, c)
end

function MOI.add_constraint(
    model::Optimizer,
    x::MOI.VariableIndex,
    set::Union{
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.EqualTo{Float64},
    },
)
    return MOI.add_constraint(model.variables, x, set)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    set::S,
) where {S<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo}}
    MOI.set(model.variables, MOI.ConstraintSet(), ci, set)
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
) where {S<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo}}
    MOI.delete(model.variables, ci)
    return
end

### ScalarAffineFunction and ScalarQuadraticFunction constraints

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo},
}
    return 1 <= ci.value <= length(_constraints(model, F, S))
end

function _constraints(
    model::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{MOI.LessThan{Float64}},
)
    return model.linear_le_constraints
end

function _constraints(
    model::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{MOI.GreaterThan{Float64}},
)
    return model.linear_ge_constraints
end

function _constraints(
    model::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{MOI.EqualTo{Float64}},
)
    return model.linear_eq_constraints
end

function _constraints(
    model::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction{Float64}},
    ::Type{MOI.LessThan{Float64}},
)
    return model.quadratic_le_constraints
end

function _constraints(
    model::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction{Float64}},
    ::Type{MOI.GreaterThan{Float64}},
)
    return model.quadratic_ge_constraints
end

function _constraints(
    model::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction{Float64}},
    ::Type{MOI.EqualTo{Float64}},
)
    return model.quadratic_eq_constraints
end

function _check_inbounds(model::Optimizer, var::MOI.VariableIndex)
    MOI.throw_if_not_valid(model, var)
    return
end

function _check_inbounds(model::Optimizer, aff::MOI.ScalarAffineFunction)
    for term in aff.terms
        MOI.throw_if_not_valid(model, term.variable)
    end
    return
end

function _check_inbounds(model::Optimizer, quad::MOI.ScalarQuadraticFunction)
    for term in quad.affine_terms
        MOI.throw_if_not_valid(model, term.variable)
    end
    for term in quad.quadratic_terms
        MOI.throw_if_not_valid(model, term.variable_1)
        MOI.throw_if_not_valid(model, term.variable_2)
    end
    return
end

function MOI.add_constraint(
    model::Optimizer,
    func::F,
    set::S,
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:MOI.AbstractScalarSet,
}
    _check_inbounds(model, func)
    constraints = _constraints(model, F, S)
    push!(constraints, _ConstraintInfo(func, set))
    return MOI.ConstraintIndex{F,S}(length(constraints))
end

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    return length(_constraints(model, F, S))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    return MOI.ConstraintIndex{F,S}[
        MOI.ConstraintIndex{F,S}(i) for
        i in eachindex(_constraints(model, F, S))
    ]
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    return _constraints(model, F, S)[c.value].func
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    return _constraints(model, F, S)[c.value].set
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{F,S},
    value::Union{Real,Nothing},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    MOI.throw_if_not_valid(model, ci)
    constraints = _constraints(model, F, S)
    constraints[ci.value].dual_start = value
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    MOI.throw_if_not_valid(model, ci)
    constraints = _constraints(model, F, S)
    return constraints[ci.value].dual_start
end

### MOI.VariablePrimalStart

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, vi)
    model.variable_primal_start[column(vi)] = value
    return
end

### MOI.ConstraintDualStart

_dual_start(::Optimizer, ::Nothing, ::Int = 1) = 0.0

function _dual_start(model::Optimizer, value::Real, scale::Int = 1)
    return _dual_multiplier(model) * value * scale
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{
        MOI.ConstraintIndex{
            MOI.VariableIndex,
            <:Union{MOI.GreaterThan,MOI.LessThan,MOI.EqualTo},
        },
    },
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_lower_start[ci.value] = value
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.variable_lower_start[ci.value]
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_upper_start[ci.value] = value
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.variable_upper_start[ci.value]
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    if value === nothing
        model.variable_lower_start[ci.value] = nothing
        model.variable_upper_start[ci.value] = nothing
    elseif value >= 0.0
        model.variable_lower_start[ci.value] = value
        model.variable_upper_start[ci.value] = 0.0
    else
        model.variable_lower_start[ci.value] = 0.0
        model.variable_upper_start[ci.value] = value
    end
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    l = model.variable_lower_start[ci.value]
    u = model.variable_upper_start[ci.value]
    return (l === u === nothing) ? nothing : (l + u)
end

### MOI.NLPBlockDualStart

MOI.supports(::Optimizer, ::MOI.NLPBlockDualStart) = true

function MOI.set(
    model::Optimizer,
    ::MOI.NLPBlockDualStart,
    values::Union{Nothing,Vector},
)
    model.nlp_dual_start = values
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlockDualStart) = model.nlp_dual_start

### MOI.NLPBlock

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    return
end

### ObjectiveSense

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

### ObjectiveFunction

MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType) = typeof(model.objective)

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunction{F}) where {F}
    return convert(F, model.objective)::F
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
        },
    },
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction{F},
    func::F,
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    _check_inbounds(model, func)
    model.objective = func
    return
end

### SQP callback functions
### In setting up the data for SQP, we order the constraints as follows:
### - linear_le_constraints
### - linear_ge_constraints
### - linear_eq_constraints
### - quadratic_le_constraints
### - quadratic_ge_constraints
### - quadratic_eq_constraints
### - nonlinear constraints from nlp_data

const _CONSTRAINT_ORDERING = (
    :linear_le_constraints,
    :linear_ge_constraints,
    :linear_eq_constraints,
    :quadratic_le_constraints,
    :quadratic_ge_constraints,
    :quadratic_eq_constraints,
)

function _offset(
    ::Optimizer,
    ::Type{<:MOI.ScalarAffineFunction},
    ::Type{<:MOI.LessThan},
)
    return 0
end

function _offset(
    model::Optimizer,
    ::Type{<:MOI.ScalarAffineFunction},
    ::Type{<:MOI.GreaterThan},
)
    return length(model.linear_le_constraints)
end

function _offset(
    model::Optimizer,
    F::Type{<:MOI.ScalarAffineFunction},
    ::Type{<:MOI.EqualTo},
)
    return _offset(model, F, MOI.GreaterThan{Float64}) +
           length(model.linear_ge_constraints)
end

function _offset(
    model::Optimizer,
    ::Type{<:MOI.ScalarQuadraticFunction},
    ::Type{<:MOI.LessThan},
)
    x = _offset(model, MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})
    return x + length(model.linear_eq_constraints)
end

function _offset(
    model::Optimizer,
    F::Type{<:MOI.ScalarQuadraticFunction},
    ::Type{<:MOI.GreaterThan},
)
    return _offset(model, F, MOI.LessThan{Float64}) +
           length(model.quadratic_le_constraints)
end

function _offset(
    model::Optimizer,
    F::Type{<:MOI.ScalarQuadraticFunction},
    ::Type{<:MOI.EqualTo},
)
    return _offset(model, F, MOI.GreaterThan{Float64}) +
           length(model.quadratic_ge_constraints)
end

function _nlp_constraint_offset(model::Optimizer)
    x = _offset(
        model,
        MOI.ScalarQuadraticFunction{Float64},
        MOI.EqualTo{Float64},
    )
    return x + length(model.quadratic_eq_constraints)
end

_eval_function(::Nothing, ::Any) = 0.0

_eval_function(f, x) = MOI.Utilities.eval_variables(xi -> x[xi.value], f)

### Eval_F_CB

function _eval_objective(model::Optimizer, x)
    if model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    end
    return _eval_function(model.objective, x)
end

### Eval_Grad_F_CB

_fill_gradient(::Any, ::Any, ::Nothing) = nothing

function _fill_gradient(grad, ::Vector, f::MOI.VariableIndex)
    grad[f.value] = 1.0
    return
end

function _fill_gradient(grad, ::Vector, f::MOI.ScalarAffineFunction{Float64})
    for term in f.terms
        grad[term.variable.value] += term.coefficient
    end
    return
end

function _fill_gradient(
    grad,
    x::Vector,
    quad::MOI.ScalarQuadraticFunction{Float64},
)
    for term in quad.affine_terms
        grad[term.variable.value] += term.coefficient
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        if row_idx == col_idx
            grad[row_idx.value] += term.coefficient * x[row_idx.value]
        else
            grad[row_idx.value] += term.coefficient * x[col_idx.value]
            grad[col_idx.value] += term.coefficient * x[row_idx.value]
        end
    end
    return
end

function _eval_objective_gradient(model::Optimizer, grad, x)
    if model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    else
        fill!(grad, 0.0)
        _fill_gradient(grad, x, model.objective)
    end
    return
end

### Eval_G_CB

function _eval_constraint(model::Optimizer, g, x)
    row = 1
    for key in _CONSTRAINT_ORDERING
        for info in getfield(model, key)
            g[row] = _eval_function(info.func, x)
            row += 1
        end
    end
    nlp_g = view(g, row:length(g))
    MOI.eval_constraint(model.nlp_data.evaluator, nlp_g, x)
    return
end

### Eval_Jac_G_CB

function _append_to_jacobian_sparsity(J, f::MOI.ScalarAffineFunction, row)
    for term in f.terms
        push!(J, (row, term.variable.value))
    end
    return
end

function _append_to_jacobian_sparsity(J, f::MOI.ScalarQuadraticFunction, row)
    for term in f.affine_terms
        push!(J, (row, term.variable.value))
    end
    for term in f.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        if row_idx == col_idx
            push!(J, (row, row_idx.value))
        else
            push!(J, (row, row_idx.value))
            push!(J, (row, col_idx.value))
        end
    end
    return
end

function _jacobian_structure(model::Optimizer)
    J = Tuple{Int64,Int64}[]
    row = 1
    for key in _CONSTRAINT_ORDERING
        for info in getfield(model, key)
            _append_to_jacobian_sparsity(J, info.func, row)
            row += 1
        end
    end
    if length(model.nlp_data.constraint_bounds) > 0
        for (nlp_row, col) in MOI.jacobian_structure(model.nlp_data.evaluator)
            push!(J, (nlp_row + row - 1, col))
        end
    end
    return J
end

function _fill_constraint_jacobian(
    values,
    offset,
    ::Vector,
    f::MOI.ScalarAffineFunction,
)
    num_coefficients = length(f.terms)
    for i in 1:num_coefficients
        values[offset+i] = f.terms[i].coefficient
    end
    return num_coefficients
end

function _fill_constraint_jacobian(
    values,
    offset,
    x,
    f::MOI.ScalarQuadraticFunction,
)
    nterms = 0
    for term in f.affine_terms
        nterms += 1
        values[offset+nterms] = term.coefficient
    end
    for term in f.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        if row_idx == col_idx
            nterms += 1
            values[offset+nterms] = term.coefficient * x[col_idx.value]
        else
            # Note that the order matches the Jacobian sparsity pattern.
            nterms += 2
            values[offset+nterms-1] = term.coefficient * x[col_idx.value]
            values[offset+nterms] = term.coefficient * x[row_idx.value]
        end
    end
    return nterms
end

function _eval_constraint_jacobian(model::Optimizer, values, x)
    offset = 0
    for key in _CONSTRAINT_ORDERING
        for info in getfield(model, key)
            offset += _fill_constraint_jacobian(values, offset, x, info.func)
        end
    end
    nlp_values = view(values, (1+offset):length(values))
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, nlp_values, x)
    return
end

### Eval_H_CB

_append_to_hessian_sparsity(::Any, ::Any) = nothing

function _append_to_hessian_sparsity(H, f::MOI.ScalarQuadraticFunction)
    for term in f.quadratic_terms
        push!(H, (term.variable_1.value, term.variable_2.value))
    end
    return
end

function _append_hessian_lagrangian_structure(H, model::Optimizer)
    if !model.nlp_data.has_objective
        _append_to_hessian_sparsity(H, model.objective)
    end
    for info in model.quadratic_le_constraints
        _append_to_hessian_sparsity(H, info.func)
    end
    for info in model.quadratic_ge_constraints
        _append_to_hessian_sparsity(H, info.func)
    end
    for info in model.quadratic_eq_constraints
        _append_to_hessian_sparsity(H, info.func)
    end
    append!(H, MOI.hessian_lagrangian_structure(model.nlp_data.evaluator))
    return
end

_fill_hessian_lagrangian(::Any, ::Any, ::Any, ::Any) = 0

function _fill_hessian_lagrangian(H, offset, λ, f::MOI.ScalarQuadraticFunction)
    for term in f.quadratic_terms
        H[offset+1] = λ * term.coefficient
        offset += 1
    end
    return length(f.quadratic_terms)
end

function _eval_hessian_lagrangian(
    ::Type{S},
    model::Optimizer,
    H,
    μ,
    offset,
) where {S}
    F = MOI.ScalarQuadraticFunction{Float64}
    offset_start = _offset(model, F, S)
    for (i, info) in enumerate(_constraints(model, F, S))
        offset +=
            _fill_hessian_lagrangian(H, offset, μ[offset_start+i], info.func)
    end
    return offset
end

function _eval_hessian_lagrangian(model::Optimizer, H, x, σ, μ)
    offset = 0
    if !model.nlp_data.has_objective
        offset += _fill_hessian_lagrangian(H, 0, σ, model.objective)
    end
    # Handles any quadratic constraints that are present. The order matters.
    offset =
        _eval_hessian_lagrangian(MOI.LessThan{Float64}, model, H, μ, offset)
    offset =
        _eval_hessian_lagrangian(MOI.GreaterThan{Float64}, model, H, μ, offset)
    offset = _eval_hessian_lagrangian(MOI.EqualTo{Float64}, model, H, μ, offset)
    # Handles the Hessian in the nonlinear block
    MOI.eval_hessian_lagrangian(
        model.nlp_data.evaluator,
        view(H, 1+offset:length(H)),
        x,
        σ,
        view(μ, 1+_nlp_constraint_offset(model):length(μ)),
    )
    return
end

### MOI.optimize!

_bounds(s::MOI.LessThan) = (-Inf, s.upper)
_bounds(s::MOI.GreaterThan) = (s.lower, Inf)
_bounds(s::MOI.EqualTo) = (s.value, s.value)

function MOI.optimize!(model::Optimizer)
    # TODO: Reuse model.inner for incremental solves if possible.
    num_linear_constraints = 
        length(model.linear_le_constraints) +
        length(model.linear_ge_constraints) +
        length(model.linear_eq_constraints)
    num_quadratic_constraints =
        length(model.quadratic_le_constraints) +
        length(model.quadratic_ge_constraints) +
        length(model.quadratic_eq_constraints)
    num_nlp_constraints = length(model.nlp_data.constraint_bounds)
    has_hessian = :Hess in MOI.features_available(model.nlp_data.evaluator)
    init_feat = [:Grad]
    if has_hessian
        push!(init_feat, :Hess)
    end
    if num_nlp_constraints > 0
        push!(init_feat, :Jac)
    end
    MOI.initialize(model.nlp_data.evaluator, init_feat)
    jacobian_sparsity = _jacobian_structure(model)
    hessian_sparsity = Tuple{Int,Int}[]
    if has_hessian
        _append_hessian_lagrangian_structure(hessian_sparsity, model)
    end
    if model.sense == MOI.MIN_SENSE
        objective_scale = 1.0
    elseif model.sense == MOI.MAX_SENSE
        objective_scale = -1.0
    else # FEASIBILITY_SENSE
        # TODO: This could produce confusing solver output if a nonzero
        # objective is set.
        objective_scale = 0.0
    end
    eval_f_cb(x) = objective_scale * _eval_objective(model, x)
    function eval_grad_f_cb(x, grad_f)
        if model.sense == MOI.FEASIBILITY_SENSE
            grad_f .= zero(eltype(grad_f))
        else
            _eval_objective_gradient(model, grad_f, x)
            rmul!(grad_f,objective_scale)
        end
        return
    end
    eval_g_cb(x, g) = _eval_constraint(model, g, x)
    function eval_jac_g_cb(x, rows, cols, values)
        if values === nothing
            for i in eachindex(jacobian_sparsity)
                rows[i], cols[i] = jacobian_sparsity[i]
            end
        else
            _eval_constraint_jacobian(model, values, x)
        end
        return
    end
    function eval_h_cb(x, rows, cols, obj_factor, lambda, values)
        if values === nothing
            for i in eachindex(hessian_sparsity)
                rows[i], cols[i] = hessian_sparsity[i]
            end
        else
            obj_factor *= objective_scale
            _eval_hessian_lagrangian(model, values, x, obj_factor, lambda)
        end
        return
    end
    g_L, g_U = Float64[], Float64[]
    for key in _CONSTRAINT_ORDERING
        for info in getfield(model, key)
            l, u = _bounds(info.set)
            push!(g_L, l)
            push!(g_U, u)
        end
    end
    for bound in model.nlp_data.constraint_bounds
        push!(g_L, bound.lower)
        push!(g_U, bound.upper)
    end
    start_time = time()
    if length(model.variables.lower) == 0
        model.invalid_model = true
        return
    end

    model.inner = Model(
        length(model.variables.lower),
        length(g_L),
        model.variables.lower,
        model.variables.upper,
        g_L,
        g_U,
        jacobian_sparsity, 
        hessian_sparsity,
        eval_f_cb, 
        eval_g_cb, 
        eval_grad_f_cb, 
        eval_jac_g_cb, 
        has_hessian ? eval_h_cb : nothing,
        num_linear_constraints,
        model.options
    )
    options = model.inner.parameters
    if !has_hessian
   	    set_parameter(options, "hessian_type", "none")
    end
    if model.silent
        set_parameter(options, "OutputFlag", 0)
    end
    # Initialize the starting point, projecting variables from 0 onto their
    # bounds if VariablePrimalStart  is not provided.
    for (i, v) in enumerate(model.variable_primal_start)
        if v !== nothing
            model.inner.x[i] = v
        else
            model.inner.x[i] = max(0.0, model.variables.lower[i])
            model.inner.x[i] = min(model.inner.x[i], model.variables.upper[i])
        end
    end
    # Initialize the dual start to 0.0 if NLPBlockDualStart is not provided.
    if model.nlp_dual_start === nothing
        model.nlp_dual_start = zeros(Float64, num_nlp_constraints)
    end
    # ConstraintDualStart
    row = 1
    for key in _CONSTRAINT_ORDERING
        for info in getfield(model, key)
            model.inner.mult_g[row] = _dual_start(model, info.dual_start, -1)
            row += 1
        end
    end
    for dual_start in model.nlp_dual_start
        model.inner.mult_g[row] = _dual_start(model, dual_start, -1)
        row += 1
    end
    # ConstraintDualStart for variable bounds
    for i in 1:length(model.inner.n)
        model.inner.mult_x_L[i] =
            _dual_start(model, model.variable_lower_start[i])
        model.inner.mult_x_U[i] =
            _dual_start(model, model.variable_upper_start[i], -1)
    end
    optimize!(model.inner)
    # Store SolveTimeSec.
    model.solve_time = time() - start_time
    return
end

### MOI.ResultCount

# SQP always has an iterate available.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.inner !== nothing) ? 1 : 0
end

### MOI.TerminationStatus

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.invalid_model
        return MOI.INVALID_MODEL
    elseif model.inner === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    status = ApplicationReturnStatus[model.inner.status]
    if status == :Solve_Succeeded || status == :Feasible_Point_Found
        return MOI.LOCALLY_SOLVED
    elseif status == :Infeasible_Problem_Detected
        return MOI.LOCALLY_INFEASIBLE
    elseif status == :Solved_To_Acceptable_Level
        return MOI.ALMOST_LOCALLY_SOLVED
    elseif status == :Search_Direction_Becomes_Too_Small
        return MOI.NUMERICAL_ERROR
    elseif status == :Diverging_Iterates
        return MOI.NORM_LIMIT
    elseif status == :User_Requested_Stop
        return MOI.INTERRUPTED
    elseif status == :Maximum_Iterations_Exceeded
        return MOI.ITERATION_LIMIT
    elseif status == :Maximum_CpuTime_Exceeded
        return MOI.TIME_LIMIT
    elseif status == :Restoration_Failed
        return MOI.NUMERICAL_ERROR
    elseif status == :Error_In_Step_Computation
        return MOI.NUMERICAL_ERROR
    elseif status == :Invalid_Option
        return MOI.INVALID_OPTION
    elseif status == :Not_Enough_Degrees_Of_Freedom
        return MOI.INVALID_MODEL
    elseif status == :Invalid_Problem_Definition
        return MOI.INVALID_MODEL
    elseif status == :Invalid_Number_Detected
        return MOI.INVALID_MODEL
    elseif status == :Unrecoverable_Exception
        return MOI.OTHER_ERROR
    else
        return MOI.MEMORY_LIMIT
    end
end

### MOI.RawStatusString

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.invalid_model
        return "The model has no variable"
    elseif model.inner === nothing
        return "Optimize not called"
    else
        return string(ApplicationReturnStatus[model.inner.status])
    end
end

### MOI.PrimalStatus

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = ApplicationReturnStatus[model.inner.status]
    if status == :Solve_Succeeded
        return MOI.FEASIBLE_POINT
    elseif status == :Feasible_Point_Found
        return MOI.FEASIBLE_POINT
    elseif status == :Solved_To_Acceptable_Level
        # Solutions are only guaranteed to satisfy the "acceptable" convergence
        # tolerances.
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == :Infeasible_Problem_Detected
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

### MOI.DualStatus

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = ApplicationReturnStatus[model.inner.status]
    if status == :Solve_Succeeded
        return MOI.FEASIBLE_POINT
    elseif status == :Feasible_Point_Found
        return MOI.FEASIBLE_POINT
    elseif status == :Solved_To_Acceptable_Level
        # Solutions are only guaranteed to satisfy the "acceptable" convergence
        # tolerances.
        return MOI.NEARLY_FEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

### MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

### MOI.ObjectiveValue

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    scale = (model.sense == MOI.MAX_SENSE) ? -1 : 1
    return scale * model.inner.obj_val
end

### MOI.VariablePrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return model.inner.x[column(vi)]
end

### MOI.ConstraintPrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.inner.g[_offset(model, F, S)+ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{
        MOI.VariableIndex,
        <:Union{
            MOI.LessThan{Float64},
            MOI.GreaterThan{Float64},
            MOI.EqualTo{Float64},
        },
    },
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.inner.x[ci.value]
end

### MOI.ConstraintDual

_dual_multiplier(model::Optimizer) = 1.0

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S,
}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    s = -_dual_multiplier(model)
    return s * model.inner.mult_g[_offset(model, F, S)+ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return min(0.0, _dual_multiplier(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return max(0.0, _dual_multiplier(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return _dual_multiplier(model) * rc
end

### MOI.NLPBlockDual

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    s = -_dual_multiplier(model)
    return s .* model.inner.mult_g[(1+_nlp_constraint_offset(model)):end]
end
