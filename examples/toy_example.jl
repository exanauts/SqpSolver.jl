using SQP, Ipopt
using JuMP

ipopt_solver = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "warm_start_init_point" => "yes",
)
optimizer = optimizer_with_attributes(
    SQP.Optimizer, 
    "external_optimizer" => ipopt_solver,
    "max_iter" => 100,
    "algorithm" => "SQP-TR",
)
model = Model(optimizer)

@variable(model, X);
@variable(model, Y);
@objective(model, Min, X^2 + X);
@NLconstraint(model, X^2 - X == 2);
@NLconstraint(model, X*Y == 1);
@NLconstraint(model, X*Y >= 0);
@constraint(model, X >= -2);

println("________________________________________");
print(model);
println("________________________________________");
JuMP.optimize!(model);

xsol = JuMP.value.(X)
ysol = JuMP.value.(Y)
status = termination_status(model)

println("Xsol = ", xsol);
println("Ysol = ", ysol);

println("Status: ", status);
