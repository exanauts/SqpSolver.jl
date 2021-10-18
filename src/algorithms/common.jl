"""
    KT_residuals

Compute Kuhn-Turck residuals

# Arguments
- `df`: gradient
- `lambda`: Lagrangian multipliers with respect to the constraints
- `mult_x_U`: reduced cost with respect to the upper bounds
- `mult_x_L`: reduced cost with respect to the lower bounds
- `Jac`: Jacobian matrix
- `norm`: whether the residual is normalized or not
"""
function KT_residuals(
    df::Tv, lambda::Tv, mult_x_U::Tv, mult_x_L::Tv, Jac::Tm
) where {T, Tv<:AbstractArray{T}, Tm<:AbstractMatrix{T}}
    KT_res = norm(df - Jac' * lambda - mult_x_U - mult_x_L)
    scalar = max(1.0, norm(df))
    for i = 1:size(Jac,1)
        scalar = max(scalar, abs(lambda[i]) * norm(Jac[i,:]))
    end
    return KT_res / scalar
end

"""
    norm_complementarity

Compute the normalized complementeraity
"""
function norm_complementarity(
    E::Tv, g_L::Tv, g_U::Tv, x::Tv, x_L::Tv, x_U::Tv, 
    lambda::Tv, mult_x_U::Tv, mult_x_L::Tv,
    p = Inf
) where {T, Tv <: AbstractArray{T}}
    m = length(E)
    compl = Tv(undef, m)
    denom = 0.0
    for i = 1:m
        if g_L[i] == g_U[i]
            compl[i] = 0.0
        else
            compl[i] = min(E[i] - g_L[i], g_U[i] - E[i]) * lambda[i]
            denom += lambda[i]^2
        end
    end
    return norm(compl, p) / (1 + sqrt(denom))
end

"""
    norm_violations

Compute the normalized constraint violation
"""
function norm_violations(
    E::Tv, g_L::Tv, g_U::Tv, x::Tv, x_L::Tv, x_U::Tv, p = 1
) where {T, Tv <: AbstractArray{T}}

    m = length(E)
    n = length(x)
    viol = Tv(undef, m+n)
    fill!(viol, 0.0)
    for i = 1:m
        if E[i] > g_U[i]
            viol[i] = E[i] - g_U[i]
        elseif E[i] < g_L[i]
            viol[i] = g_L[i] - E[i]
        end
    end
    for j = 1:n
        if x[j] > x_U[j]
            viol[m+j] = x[j] - x_U[j]
        elseif x[j] < x_L[j]
            viol[m+j] = x_L[j] - x[j]
        end
    end
    return norm(viol, p)
end