abstract type AbstractSubOptimizer end

"""
minimize    0.5 x^T Q x + c^T x
subject to  bl <= [I : A]^T x <= bu
"""
mutable struct QpData{T,Tv<:AbstractArray{T},Tm<:AbstractMatrix{T}}
    n::Int
    m::Int
    Q::Tm
    c::Tv
    A::Tm
    bl::Tv
    bu::Tv
    cstype::String # L: linear or N: nonlinear constraints

    f::T  # objective value
    x::Tv # primal solution
    y::Tv # dual multipler

    # 0 = solution obtained
    # 1 = unbounded
    # 2 = bl[i] > bu[i] for some i
    # 3 = infeasible detected
    # 9 = unknown
    status::Int

    function QpData{T,Tv,Tm}(n::Int, m::Int)
        Q = spzeros(n, n)
        c = zeros(T, n)
        A = spzeros(m, n)
        bl = zeros(T, n+m)
        bu = zeros(T, n+m)
        x = zeros(T, n)
        y = zeros(T, n+m)
        qp = new{T,Tv,Tm}(
            n, m, Q, c, A, bl, bu, 
            repeat("N", m),
            Inf, x, y, 0
        )
        return qp
    end
end

SubModel = Union{
    MOI.AbstractOptimizer,
    JuMP.AbstractModel,
}

include("subproblem_MOI.jl")
include("subproblem_JuMP.jl")


"""
minimize    0.5 x^T Q x + c^T x
subject to  bl <= [I : A]^T x <= bu
"""
function solve!(qp::QpData)
    #
end
