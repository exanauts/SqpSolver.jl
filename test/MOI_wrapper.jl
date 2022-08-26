module TestMOIWrapper

using SQP
using Ipopt
using JuMP
using Test

const MOI = SQP.MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

const optimizer = SQP.Optimizer()
const ipopt_optimizer = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "warm_start_init_point" => "yes",
)

MOI.set(optimizer, MOI.RawOptimizerAttribute("external_optimizer"), ipopt_optimizer)
# MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), 3000)
MOI.set(optimizer, MOI.RawOptimizerAttribute("OutputFlag"), 1)

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_MOI_Test()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Bridges.full_bridge_optimizer(optimizer, Float64),
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol = 1e-4,
            rtol = 1e-4,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintDual,
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
            ],
        );
        exclude = String[
            # Tests purposefully excluded:
            #  - NORM_LIMIT when run on macOS-M1. See #315
            "test_linear_transform",
            #  - Upstream: ZeroBridge does not support ConstraintDual
            "test_conic_linear_VectorOfVariables_2",
            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
            #  - Excluded because Ipopt returns NORM_LIMIT instead of
            #    DUAL_INFEASIBLE
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            #  - Excluded because Ipopt returns INVALID_MODEL instead of
            #    LOCALLY_SOLVED
            "test_linear_VectorAffineFunction_empty_row",
            #  - Excluded because Ipopt returns LOCALLY_INFEASIBLE instead of
            #    INFEASIBLE
            "INFEASIBLE",
            "test_conic_linear_INFEASIBLE",
            "test_conic_linear_INFEASIBLE_2",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_",
            #  - Excluded due to upstream issue
            "test_model_LowerBoundAlreadySet",
            "test_model_UpperBoundAlreadySet",
            #  - CachingOptimizer does not throw if optimizer not attached
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
        ],
    )
    return
end

end
TestMOIWrapper.runtests()
