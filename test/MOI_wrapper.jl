using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

const optimizer = SQP.Optimizer()
const ipopt_optimizer = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "warm_start_init_point" => "yes",
)

MOI.set(optimizer, MOI.RawParameter("external_optimizer"), ipopt_optimizer)
# MOI.set(optimizer, MOI.RawParameter("max_iter"), 3000)
MOI.set(optimizer, MOI.RawParameter("OutputFlag"), 0)

const config_no_duals = MOIT.TestConfig(atol=1e-1, rtol=1e-1, duals=false, optimal_status=MOI.LOCALLY_SOLVED)

@testset "SolverName" begin
    @test MOI.get(optimizer, MOI.SolverName()) == "SQP"
end

@testset "supports_default_copy_to" begin
    @test MOIU.supports_default_copy_to(optimizer, false)
    @test !MOIU.supports_default_copy_to(optimizer, true)
end

@testset "Unit ($algo)" for algo in ["SQP-TR"]
    MOI.set(optimizer, MOI.RawParameter("algorithm"), algo)
    bridged = MOIB.full_bridge_optimizer(optimizer, Float64)
    # A number of test cases are excluded because loadfromstring! works only
    # if the solver supports variable and constraint names.
    exclude = [
        "delete_variable", # Deleting not supported.
        "delete_variables", # Deleting not supported.
        "getvariable", # Variable names not supported.
        "solve_zero_one_with_bounds_1", # Variable names not supported.
        "solve_zero_one_with_bounds_2", # Variable names not supported.
        "solve_zero_one_with_bounds_3", # Variable names not supported.
        "getconstraint", # Constraint names not suported.
        "variablenames", # Variable names not supported.
        "solve_with_upperbound", # loadfromstring!
        "solve_with_lowerbound", # loadfromstring!
        "solve_integer_edge_cases", # loadfromstring!
        "solve_affine_lessthan", # loadfromstring!
        "solve_affine_greaterthan", # loadfromstring!
        "solve_affine_equalto", # loadfromstring!
        "solve_affine_interval", # loadfromstring!
        "get_objective_function", # Function getters not supported.
        "solve_constant_obj",  # loadfromstring!
        "solve_blank_obj", # loadfromstring!
        "solve_singlevariable_obj", # loadfromstring!
        "solve_objbound_edge_cases", # ObjectiveBound not supported.
        "solve_affine_deletion_edge_cases", # Deleting not supported.
        "solve_unbounded_model", # `NORM_LIMIT`
        "number_threads", # NumberOfThreads not supported
        "delete_nonnegative_variables", # get ConstraintFunction n/a.
        "update_dimension_nonnegative_variables", # get ConstraintFunction n/a.
        "delete_soc_variables", # VectorOfVar. in SOC not supported
        "solve_result_index", # DualObjectiveValue not supported
        "solve_farkas_interval_lower",
        "solve_farkas_lessthan",
        "solve_farkas_interval_upper",
        "solve_farkas_greaterthan",
        "solve_farkas_variable_lessthan_max",
        "solve_farkas_variable_lessthan",
        "solve_farkas_equalto_upper",
        "solve_farkas_equalto_lower",
    ]
    MOIT.unittest(bridged, config_no_duals, exclude)
    MOI.empty!(optimizer)
end

@testset "MOI Linear tests ($algo)" for algo in ["SQP-TR"]
    MOI.set(optimizer, MOI.RawParameter("algorithm"), algo)
    exclude = ["linear8a", # Behavior in infeasible case doesn't match test.
               "linear12", # Same as above.
               "linear8b", # Behavior in unbounded case doesn't match test.
               "linear8c", # Same as above.
               "linear7",  # VectorAffineFunction not supported.
               "linear15", # VectorAffineFunction not supported.
               ]
    model_for_SQP = MOIU.UniversalFallback(MOIU.Model{Float64}())
    linear_optimizer = MOI.Bridges.Constraint.SplitInterval{Float64}(
                         MOIU.CachingOptimizer(model_for_SQP, optimizer))
    MOIT.contlineartest(linear_optimizer, config_no_duals, exclude)
    # Tests setting bounds of `SingleVariable` constraint
    # MOIT.linear3test(linear_optimizer, config_no_duals)
    MOI.empty!(optimizer)
end

# FIXME: LP subproblems are numerically instable in the trust region method.
@testset "MOI QP tests ($algo)" for algo in []
    MOI.set(optimizer, MOI.RawParameter("algorithm"), algo)
    qp_optimizer = MOIU.CachingOptimizer(MOIU.Model{Float64}(), optimizer)
    MOIT.qptest(qp_optimizer, config_no_duals)
    # MOIT.qp1test(qp_optimizer, config_no_duals)
    MOI.empty!(optimizer)
end

@testset "MOI QCQP tests ($algo)" for algo in ["SQP-TR"]
    MOI.set(optimizer, MOI.RawParameter("algorithm"), algo)
    qp_optimizer = MOIU.CachingOptimizer(MOIU.Model{Float64}(), optimizer)
    exclude = ["qcp1"] # VectorAffineFunction not supported.
    MOIT.qcptest(qp_optimizer, config_no_duals, exclude)
    # MOIT.qcp2test(qp_optimizer, config_no_duals)
    MOI.empty!(optimizer)
end

# FIXME: LP subproblems are numerically instable in the trust region method.
@testset "MOI NLP tests ($algo)" for algo in ["SQP-TR"]
    MOI.set(optimizer, MOI.RawParameter("algorithm"), algo)
    MOIT.nlptest(optimizer, config_no_duals)
    # MOIT.hs071_test(optimizer, config_no_duals)
    MOI.empty!(optimizer)
end

@testset "Testing getters" begin
    MOI.Test.copytest(MOI.instantiate(SQP.Optimizer, with_bridge_type=Float64), MOIU.Model{Float64}())
end

@testset "Bounds set twice" begin
    MOI.Test.set_lower_bound_twice(optimizer, Float64)
    MOI.Test.set_upper_bound_twice(optimizer, Float64)
end
