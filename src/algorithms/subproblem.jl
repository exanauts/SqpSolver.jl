abstract type AbstractSubOptimizer end

"""
sense 0.5 x'Qx + c'x + μ (s1 + s2)
subject to
c_lb <= Ax + b + s1 - s2 + s <= c_ub
v_lb <= x + x_k <= v_ub
-Δ <= x <= Δ
s1 + max(0,s) >= 0
s2 - min(0,s) >= 0
"""
struct QpData{T,Tv<:AbstractArray{T},Tm<:AbstractMatrix{T}}
    sense::MOI.OptimizationSense
    Q::Union{Nothing,Tm}
    c::Tv
    A::Tm
    b::Tv
    c_lb::Tv
    c_ub::Tv
    v_lb::Tv
    v_ub::Tv
    num_linear_constraints::Int
end

SubModel = Union{
    MOI.AbstractOptimizer,
    JuMP.AbstractModel,
}

include("subproblem_MOI.jl")
include("subproblem_JuMP.jl")
