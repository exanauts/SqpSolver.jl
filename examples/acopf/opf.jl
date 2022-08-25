using Revise
using SQP
using PowerModels, JuMP, Ipopt
using filterSQP
using CPLEX

PowerModels.silence()

include("acwr.jl")
include("init_opf.jl")

function build_opf(pm::PowerModels.AbstractPowerModel)
    PowerModels.variable_bus_voltage(pm)
    PowerModels.variable_gen_power(pm)
    PowerModels.variable_branch_power(pm)
    PowerModels.variable_dcline_power(pm)

    PowerModels.objective_min_fuel_and_flow_cost(pm)

    PowerModels.constraint_model_voltage(pm)

    for i in PowerModels.ids(pm, :ref_buses)
        PowerModels.constraint_theta_ref(pm, i)
    end

    for i in PowerModels.ids(pm, :bus)
        PowerModels.constraint_power_balance(pm, i)
    end

    for i in PowerModels.ids(pm, :branch)
        PowerModels.constraint_ohms_yt_from(pm, i)
        PowerModels.constraint_ohms_yt_to(pm, i)

        # constraint_voltage_angle_difference(pm, i)

        PowerModels.constraint_thermal_limit_from(pm, i)
        PowerModels.constraint_thermal_limit_to(pm, i)
    end

    for i in PowerModels.ids(pm, :dcline)
        PowerModels.constraint_dcline_power_losses(pm, i)
    end
end

build_acp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACPPowerModel, build_opf)
build_acr(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACRPowerModel, build_opf)
build_iv(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), IVRPowerModel, PowerModels.build_opf_iv)
build_dcp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), DCPPowerModel, PowerModels.build_opf_iv)
build_socp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), SOCWRPowerModel, PowerModels.build_opf)

##
function run_sqp_opf(data_file::String, max_iter::Int = 100)
    # pm = build_socp(data_file)
    pm = build_acp(data_file)
    init_vars(pm)
    # pm2 = build_acp(data_file)
    # JuMP.@objective(pm2.model, Min, 0)
    # init_vars_from_ipopt(pm, pm2)

    # choose an internal QP solver
    qp_solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "warm_start_init_point" => "yes",
        "linear_solver" => "ma27",
        # CPLEX.Optimizer,
        # "CPX_PARAM_SCRIND" => 0,
        # "CPX_PARAM_THREADS" => 1,
        # "CPXPARAM_OptimalityTarget" => 2, # 1: convex, 2: local, 3: global
        # "CPX_PARAM_QPMETHOD" => 1,
        # "CPXPARAM_Barrier_ConvergeTol" => 1.0e-4,
    )

    result = optimize_model!(pm, optimizer = optimizer_with_attributes(
        SQP.Optimizer, 
        "algorithm" => "SQP-TR",
        "external_optimizer" => qp_solver,
        # "tol_infeas" => 1.e-6,
        # "tol_residual" => 1.e-4,
        "init_mu" => 1.e+4,
        "max_iter" => max_iter,
        # "use_soc" => true,
    ))
    return pm, result
end

# run_sqp_opf("../data/case118.m", 20);

##
function run_ipopt!(data_file::String)
    pm = build_acp(data_file)
    init_vars(pm)
    # pm2 = build_acp(data_file)
    # JuMP.@objective(pm2.model, Min, 0)
    # init_vars_from_ipopt(pm, pm2)
    solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        "warm_start_init_point" => "yes",
        "linear_solver" => "ma27",
    )
    optimize_model!(pm, optimizer = solver)
    return
end

# run_ipopt!("../data/case30.m");

##
function run_filter_sqp!(data_file::String)
    pm = build_acp(data_file)
    init_vars(pm)
    # pm2 = build_socp(data_file)
    # JuMP.@objective(pm2.model, Min, 0)
    # init_vars_from_ipopt(pm, pm2)
    solver = optimizer_with_attributes(
        filterSQP.Optimizer,
        "iprint" => 1,
    )
    optimize_model!(pm, optimizer = solver)
    return
end

# run_filter_sqp!("../data/case30.m");
# run_filter_sqp!("../data/case300.m");

###
# data_file = "../data/case118.m"
# pm = build_acp(data_file);
# init_vars(pm);


# qp_solver = optimizer_with_attributes(
#     Ipopt.Optimizer,
#     "print_level" => 0,
#     "warm_start_init_point" => "yes",
#     "linear_solver" => "ma27",
# );
# solver = optimizer_with_attributes(
#     SQP.Optimizer, 
#     "algorithm" => "SQP-TR",
#     "external_optimizer" => qp_solver,
#     # "tol_infeas" => 1.e-6,
#     # "tol_residual" => 1.e-4,
#     "init_mu" => 1.e+2,
#     "max_iter" => 20,
#     # "use_soc" => true,
# );

# solver = optimizer_with_attributes(
#     filterSQP.Optimizer,
#     "iprint" => 1,
# );
# optimize_model!(pm, optimizer = solver);

# for (_, g) in pm.data["gen"]
#     g["cost"][1] *= rand() * 0.2 + 0.9;
#     g["cost"][2] *= rand() * 0.2 + 0.9;
# end
# objective_min_fuel_cost(pm);

# JuMP.optimize!(pm.model);
