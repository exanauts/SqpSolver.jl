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
    "mu_strategy" => "adaptive",
    "warm_start_init_point" => "yes",
)

MOI.set(optimizer, MOI.RawOptimizerAttribute("external_optimizer"), ipopt_optimizer)
MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), 1000)
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
            infeasible_status = MOI.LOCALLY_INFEASIBLE,
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
            #  - Convex after reformulation; but we cannot find a global optimum.
            "test_quadratic_SecondOrderCone_basic",
        ],
    )
    return
end

end
TestMOIWrapper.runtests()
