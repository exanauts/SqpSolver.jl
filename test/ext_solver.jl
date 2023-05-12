
qp_solver = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "warm_start_init_point" => "yes",
)
optimizer_solver = optimizer_with_attributes(
    SqpSolver.Optimizer,
    "external_optimizer" => qp_solver,
    "algorithm" => "SQP-TR",
    "OutputFlag" => 0,
)

model = Model(optimizer_solver)

@variable(model, X);
@variable(model, Y);
@objective(model, Min, X^2 + X);
@NLconstraint(model, X^2 - X == 2);
@NLconstraint(model, X * Y == 1);
@NLconstraint(model, X * Y >= 0);
@constraint(model, X >= -2);

JuMP.optimize!(model);

xsol = JuMP.value.(X)
ysol = JuMP.value.(Y)
status = termination_status(model)
