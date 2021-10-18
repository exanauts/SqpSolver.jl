using Revise
using SQP
using PowerModels, JuMP, Ipopt
using filterSQP
using CPLEX

PowerModels.silence()

include("acwr.jl")
include("init_opf.jl")

build_acp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACPPowerModel, PowerModels.build_opf)
build_acr(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACRPowerModel, PowerModels.build_opf)
build_iv(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), IVRPowerModel, PowerModels.build_opf_iv)
build_dcp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), DCPPowerModel, PowerModels.build_opf_iv)

##
function run_sqp_opf(data_file::String, max_iter::Int = 100)
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
        "linear_solver" => "ma57",
        # "ma57_pre_alloc" => 5.0,
        # CPLEX.Optimizer,
        # "CPX_PARAM_SCRIND" => 1,
        # "CPX_PARAM_THREADS" => 1,
        # "CPXPARAM_OptimalityTarget" => 2, # 1: convex, 2: local, 3: global
        # "CPXPARAM_Barrier_ConvergeTol" => 1.0e-4,
    )

    result = optimize_model!(pm, optimizer = optimizer_with_attributes(
        SQP.Optimizer, 
        "algorithm" => "SQP-TR",
        "external_optimizer" => qp_solver,
        "tol_infeas" => 1.e-6,
        "tol_residual" => 1.e-4,
        "max_iter" => max_iter,
        "use_soc" => true,
    ))
    return pm, result
end

run_sqp_opf("../data/case30.m", 50);

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
        "linear_solver" => "ma57",
    )
    optimize_model!(pm, optimizer = solver)
    return
end

run_ipopt!("../data/case2869pegase.m");

# ##
# function run_filter_sqp!(data_file::String)
#     pm = build_acp(data_file)
#     init_vars(pm)
#     # pm2 = build_acp(data_file)
#     # JuMP.@objective(pm2.model, Min, 0)
#     # init_vars_from_ipopt(pm, pm2)
#     solver = optimizer_with_attributes(
#         filterSQP.Optimizer,
#         "iprint" => 1,
#     )
#     optimize_model!(pm, optimizer = solver)
#     return
# end

# run_filter_sqp!("../data/case1354pegase.m");