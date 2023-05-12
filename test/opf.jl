using PowerModels

PowerModels.silence()

build_acp(data_file::String) = instantiate_model(
    PowerModels.parse_file(data_file), 
    ACPPowerModel, 
    PowerModels.build_opf
)

function run_sqp_opf(data_file::String, max_iter::Int = 100)
    pm = build_acp(data_file)
    qp_solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "warm_start_init_point" => "yes",
    )
    result = optimize_model!(pm, optimizer = optimizer_with_attributes(
        SqpSolver.Optimizer, 
        "algorithm" => "SQP-TR",
        "external_optimizer" => qp_solver,
        "max_iter" => max_iter,
    ))
    return result
end