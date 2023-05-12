abstract type AbstractSqpModel end

mutable struct Model{T,TD} <: AbstractSqpModel
    n::Int  # Num vars
    m::Int  # Num cons
    x::TD  # Starting and final solution
    x_L::TD # Variables Lower Bound
    x_U::TD # Variables Upper Bound
    g::TD  # Final constraint values
    g_L::TD # Constraints Lower Bound
    g_U::TD # Constraints Upper Bound
    j_str::Array{Tuple{Int,Int}}
    h_str::Array{Tuple{Int,Int}}
    mult_g::TD # lagrange multipliers on constraints
    mult_x_L::TD # lagrange multipliers on lower bounds
    mult_x_U::TD # lagrange multipliers on upper bounds
    obj_val::T  # Final objective
    status::Int  # Final status

    # Callbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Union{Function,Nothing}

    num_linear_constraints::Int # number of linear constraints

    intermediate  # Can be nothing

    # For MathProgBase
    sense::Symbol

    parameters::Parameters
    statistics::Dict{String,Any}   # collects parameters of all iterations inside the algorithm if StatisticsFlag > 0

    Model(
        n::Int, 
        m::Int, 
        x_L::TD, 
        x_U::TD,
        g_L::TD,
        g_U::TD,
        j_str::Array{Tuple{Int,Int}},
        h_str::Array{Tuple{Int,Int}},
        eval_f::Function,
        eval_g::Function,
        eval_grad_f::Function,
        eval_jac_g::Function,
        eval_h::Union{Function,Nothing},
        num_linear_constraints::Int,
        sense::Symbol, # {:Min, :Max}
        parameters::Parameters
    ) where {T, TD<:AbstractArray{T}} = new{T,TD}(
        n, m,
        zeros(n), x_L, x_U,
        zeros(m), g_L, g_U,
        j_str, h_str,
        zeros(m), zeros(n), zeros(n),
        0.0,
        -5,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, 
        num_linear_constraints,
        nothing, sense,
        parameters,
        Dict{String,Any}()
    )
end

function optimize!(model::Model)
    if isnothing(model.parameters.external_optimizer)
    	model.status = -12;
        @error "`external_optimizer` parameter must be set for subproblem solutions."
    else
        if model.parameters.algorithm == "SQP-TR"
            sqp = SqpTR(model)
            run!(sqp)
        # elseif model.parameters.algorithm == "SLP-TR"
        #     model.eval_h = nothing
        #     slp = SlpTR(model)
        #     run!(slp)
        # elseif model.parameters.algorithm == "SLP-LS"
        #     model.eval_h = nothing
        #     slp = SlpLS(model)
        #     run!(slp)
        else
            @warn "$(model.parameters.algorithm) is not defined"
        end
    end
    return nothing
end

function add_statistic(model::AbstractSqpModel, name::String, value)
    if model.parameters.StatisticsFlag == 0
        return
    end
    model.statistics[name] = value
end

function add_statistics(model::AbstractSqpModel, name::String, value::T) where T
    if model.parameters.StatisticsFlag == 0
        return
    end
    if !haskey(model.statistics, name)
        model.statistics[name] = Array{T,1}()
    end
    push!(model.statistics[name], value)
end
