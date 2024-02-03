using ArgParse
using Dates: now

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--gridsize"
            arg_type = Int
            default = 8
            help = "Length of the an edge of the GridWorld (equal width and length)."

        "--t_prob"
            arg_type = Float64
            default = 0.7
            help = "Probability of a correct transition."

        "--o_prob"
            arg_type = Float64
            default = 0.9
            help = "Probability of a correct observation."

        "--exploration_const"
            arg_type = Float64
            default = 0.1
            help = "Exploration constant `k_ucb` (Section A.2 in paper) during action selection."

        "--z_dist_exp_const"
            arg_type = Float64
            default = 20.0
            help = "Exponential constant `k_z-exp` (Section A.4 in paper) to tune the steepness of the z-curve." 

        "--threads"
            arg_type = Int
            default = 20
            help = "Simultaneous number of branches spawned across multiple threads. I.e. how many threads to use?"

        "--sims_per_thread"
            arg_type = Int
            default = 10
            help = "How many branches should each thread spawn?"

        "--max_timesteps"
            arg_type = Int
            default = 10
            help = "Maximum number of timesteps to go back in time while constructing the BackwardTree."

        "--rollout_random"
            arg_type = Bool
            default = true
            help = "true: Select random action if rolling out; false: Select action with smallest linear program cost (computationally significantly more expensive!)."

        "--val_epochs"
            arg_type = Int
            default = 10000
            help = "Number of epochs for forward simulations (during validation)."

        "--noise_seed"
            arg_type = Int
            default = 149
            help = "Seed for RNG. Should be a prime number, due to how parallelization is setup across threads."

        "--savename"
            arg_type = String
            default = "benchmarkRun_" * string(now())
            help = "Save name to locally save results. Any valid String accepted. Pass no arguments to skip saving."    end

    return parse_args(s, as_symbols=true)
end

function show_args(parsed_args)
    println("## Arguments Parsed ##")
    for (arg,val) in parsed_args
        println("  $arg \t =>  $val")
    end
    println(" ")
end

macro show_args(parsed_args)
    return :( show_args($parsed_args) )
end
