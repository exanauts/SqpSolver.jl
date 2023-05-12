# SqpSolver.jl

![Run tests](https://github.com/exanauts/SqpSolver.jl/workflows/Run%20tests/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/exanauts/SqpSolver.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/exanauts/SqpSolver.jl)

This is a Julia package that implements sequantial quadratic programming algorithms for continuous nonlinear optimization.

## Installation

```julia
]add SqpSolver
```

## Example

Consider the following quadratic optimization problem

```
min   x^2 + x 
s.t.  x^2 - x = 2
```

This problem can be solved by the following code snippet:
```julia
# Load packages
using SqpSolver, JuMP
using Ipopt # can be any QP solver

# Number of variables
n = 1

# Build nonlinear problem model via JuMP
model = Model(optimizer_with_attributes(
    SqpSolver.Optimizer, 
    "external_optimizer" => Ipopt.Optimizer,
))
@variable(model, x)
@objective(model, Min, x^2 + x)
@NLconstraint(model, x^2 - x == 2)

# Solve optimization problem
JuMP.optimize!(model)

# Retrieve solution
Xsol = JuMP.value.(X)
```

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
