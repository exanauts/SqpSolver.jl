"""
Initialize variable values by taking the mean of lower and upper bounds.
"""
function init_vars(pm::AbstractPowerModel)
    init_branch_vars(pm)
    init_dc_vars(pm)
    init_gen_vars(pm)
    init_voltage_vars(pm)
end

"""
Initialize variable values for ACPPowerModel from Ipopt solution.
"""
function init_vars_from_ipopt(pm::T, pm2::T) where T<:AbstractPowerModel
    optimize_model!(pm2, optimizer = Ipopt.Optimizer)
    init_branch_vars(pm, pm2)
    init_dc_vars(pm, pm2)
    init_gen_vars(pm, pm2)
    init_voltage_vars(pm, pm2)
end

"""
Set initial variable value to JuMP, if the variable has both lower and upper bounds.
"""
function set_start_value(v::JuMP.VariableRef)
    if has_lower_bound(v) && has_upper_bound(v)
        if upper_bound(v) < Inf && lower_bound(v) > -Inf
            JuMP.set_start_value(v, (upper_bound(v)+lower_bound(v))/2)
        elseif upper_bound(v) < Inf
            JuMP.set_start_value(v, upper_bound(v))
        elseif lower_bound(v) > -Inf
            JuMP.set_start_value(v, lower_bound(v))
        end
    elseif has_lower_bound(v)
        if lower_bound(v) > -Inf
            JuMP.set_start_value(v, lower_bound(v))
        else
            JuMP.set_start_value(v, 0.0)
        end
    elseif has_upper_bound(v)
        if upper_bound(v) < Inf
            JuMP.set_start_value(v, upper_bound(v))
        else
            JuMP.set_start_value(v, 0.0)
        end
    end
end

"""
Initilize branch variable values
"""

function init_branch_vars(pm::AbstractPowerModel)
    for (l,i,j) in ref(pm,:arcs)
        set_start_value(var(pm,:p)[(l,i,j)])
        set_start_value(var(pm,:q)[(l,i,j)])
    end
end

function init_branch_vars(pm::AbstractPowerModel, pm_solved::AbstractPowerModel)
    for (l,i,j) in ref(pm,:arcs)
        JuMP.set_start_value(var(pm,:p)[(l,i,j)], JuMP.value(var(pm_solved,:p)[(l,i,j)]))
        JuMP.set_start_value(var(pm,:q)[(l,i,j)], JuMP.value(var(pm_solved,:q)[(l,i,j)]))
    end
end

function init_branch_vars(pm::IVRPowerModel)
    for (l,i,j) in ref(pm,:arcs)
        set_start_value(var(pm,:cr)[(l,i,j)])
        set_start_value(var(pm,:ci)[(l,i,j)])
    end
    for l in ids(pm,:branch)
        set_start_value(var(pm,:csr)[l])
        set_start_value(var(pm,:csi)[l])
    end
end

function init_branch_vars(pm::IVRPowerModel, pm_solved::IVRPowerModel)
    for (l,i,j) in ref(pm,:arcs)
        JuMP.set_start_value(var(pm,:cr)[(l,i,j)], JuMP.value(var(pm_solved,:cr)[(l,i,j)]))
        JuMP.set_start_value(var(pm,:ci)[(l,i,j)], JuMP.value(var(pm_solved,:ci)[(l,i,j)]))
    end
    for l in ids(pm,:branch)
        JuMP.set_start_value(var(pm,:csr)[l], JuMP.value(var(pm_solved,:csr)[l]))
        JuMP.set_start_value(var(pm,:csi)[l], JuMP.value(var(pm_solved,:csi)[l]))
    end
end

"""
Initilize direct current branch variable values
"""

function init_dc_vars(pm::AbstractPowerModel)
    for arc in ref(pm,:arcs_dc)
        set_start_value(var(pm,:p_dc)[arc])
        set_start_value(var(pm,:q_dc)[arc])
    end
end

function init_dc_vars(pm::AbstractPowerModel, pm_solved::AbstractPowerModel)
    for arc in ref(pm,:arcs_dc)
        JuMP.set_start_value(var(pm,:p_dc)[arc], JuMP.value(var(pm_solved,:p_dc)[arc]))
        JuMP.set_start_value(var(pm,:q_dc)[arc], JuMP.value(var(pm_solved,:q_dc)[arc]))
    end
end

function init_dc_vars(pm::IVRPowerModel)
    for arc in ref(pm,:arcs_dc)
        set_start_value(var(pm,:crdc)[arc])
        set_start_value(var(pm,:cidc)[arc])
    end
end

function init_dc_vars(pm::IVRPowerModel, pm_solved::IVRPowerModel)
    for arc in ref(pm,:arcs_dc)
        JuMP.set_start_value(var(pm,:crdc)[arc], JuMP.value(var(pm_solved,:crdc)[arc]))
        JuMP.set_start_value(var(pm,:crdc)[arc], JuMP.value(var(pm_solved,:crdc)[arc]))
    end
end

"""
Initilize generation variable values
"""

function init_gen_vars(pm::AbstractPowerModel)
    for (i,gen) in ref(pm,:gen)
        set_start_value(var(pm,:pg)[i])
        set_start_value(var(pm,:qg)[i])
    end
end

function init_gen_vars(pm::AbstractPowerModel, pm_solved::AbstractPowerModel)
    for (i,gen) in ref(pm,:gen)
        JuMP.set_start_value(var(pm,:pg)[i], JuMP.value(var(pm_solved,:pg)[i]))
        JuMP.set_start_value(var(pm,:qg)[i], JuMP.value(var(pm_solved,:qg)[i]))
    end
end

function init_gen_vars(pm::IVRPowerModel)
    for (i,gen) in ref(pm,:gen)
        set_start_value(var(pm,:crg)[i])
        set_start_value(var(pm,:cig)[i])
    end
end

function init_gen_vars(pm::IVRPowerModel, pm_solved::IVRPowerModel)
    for (i,gen) in ref(pm,:gen)
        JuMP.set_start_value(var(pm,:crg)[i], JuMP.value(var(pm_solved,:crg)[i]))
        JuMP.set_start_value(var(pm,:crg)[i], JuMP.value(var(pm_solved,:crg)[i]))
    end
end

"""
Initilize voltage variable values
"""

function init_voltage_vars(pm::AbstractACPModel)
    for (i,bus) in ref(pm,:bus)
        set_start_value(var(pm,:va)[i])
        set_start_value(var(pm,:vm)[i])
    end
end

function init_voltage_vars(pm::AbstractACPModel, pm_solved::AbstractACPModel)
    for (i,bus) in ref(pm,:bus)
        JuMP.set_start_value(var(pm,:va)[i], JuMP.value(var(pm_solved,:va)[i]))
        JuMP.set_start_value(var(pm,:vm)[i], JuMP.value(var(pm_solved,:vm)[i]))
    end
end

function init_voltage_vars(pm::AbstractACRModel)
    for (i,bus) in ref(pm,:bus)
        set_start_value(var(pm,:vr)[i])
        set_start_value(var(pm,:vi)[i])
    end
end

function init_voltage_vars(pm::AbstractACRModel, pm_solved::AbstractACRModel)
    for (i,bus) in ref(pm,:bus)
        JuMP.set_start_value(var(pm,:vr)[i], JuMP.value(var(pm_solved,:vr)[i]))
        JuMP.set_start_value(var(pm,:vi)[i], JuMP.value(var(pm_solved,:vi)[i]))
    end
end
