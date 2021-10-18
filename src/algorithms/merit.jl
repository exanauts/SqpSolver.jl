"""
    compute_derivative

Compute and return directional derivative

# Arguments
- `∇f`: evaluation of the objective gradient
- `p`: search direction
- `∇fp`: objective gradient times times search direction, i.e., `∇f' * p`
- `μ`: penalty parameter
- `cons_viol`: constraint violations
"""
compute_derivative(∇fp::T, μ::T, cons_viol::T) where {T} = ∇fp - μ * cons_viol
compute_derivative(∇fp::T, μ::Tv, cons_viol::Tv) where {T, Tv<:AbstractArray{T}} = ∇fp - μ' * cons_viol
compute_derivative(∇fp::T, μ::T, cons_viol::Tv) where {T, Tv<:AbstractArray{T}} = ∇fp - μ * sum(cons_viol)
compute_derivative(∇f::Tv, p::Tv, μ::T, cons_viol::Tv) where {T, Tv<:AbstractArray{T}} = ∇f' * p - μ * sum(cons_viol)
compute_derivative(∇f::Tv, p::Tv, μ::Tv, cons_viol::Tv) where {T, Tv<:AbstractArray{T}} = ∇f' * p - μ' * cons_viol
