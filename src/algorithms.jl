"""
    AbstractOptimizer

Abstract type of solvers
"""
abstract type AbstractOptimizer end

"""
    run!
    
Abstract function of running algorithm
"""
function run! end


include("algorithms/common.jl")
include("algorithms/merit.jl")
include("algorithms/subproblem.jl")
include("algorithms/sqp.jl")
