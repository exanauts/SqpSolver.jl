mutable struct ACWRPowerModel <: PowerModels.AbstractWRModel
    PowerModels.@pm_fields
end

function PowerModels.variable_bus_voltage(pm::ACWRPowerModel; kwargs...)
    variable_bus_voltage_magnitude_sqr(pm; kwargs...)
    variable_buspair_voltage_product(pm; kwargs...)

    nw = pm.cnw
    PowerModels.var(pm, nw)[:vr] = JuMP.@variable(
        pm.model, 
        [i in PowerModels.ids(pm, nw, :bus)], 
        base_name="$(nw)_vr", 
        start = PowerModels.comp_start_value(PowerModels.ref(pm, nw, :bus, i), "vr_start", 1.0))
    PowerModels.var(pm, nw)[:vi] = JuMP.@variable(
        pm.model, 
        [i in PowerModels.ids(pm, nw, :bus)], 
        base_name="$(nw)_vi", 
        start = PowerModels.comp_start_value(PowerModels.ref(pm, nw, :bus, i), "vi_start"))
end

function PowerModels.constraint_model_voltage(pm::ACWRPowerModel, n::Int)
    w  = var(pm, n, :w)
    wr = var(pm, n, :wr)
    wi = var(pm, n, :wi)
    vr = var(pm, n, :vr)
    vi = var(pm, n, :vi)
    for i in ids(pm, n, :bus)
        JuMP.@constraint(pm.model, w[i] == vr[i]^2 + vi[i]^2)
    end
    for (i,j) in ids(pm, n, :buspairs)
        JuMP.@constraint(pm.model, wr[(i,j)] == vr[i] * vr[j] + vi[i] * vi[j])
        JuMP.@constraint(pm.model, wi[(i,j)] == vi[i] * vr[j] - vr[i] * vi[j])
    end
end

build_acwr(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACWRPowerModel, PowerModels.build_opf)
