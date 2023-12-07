using ArgParse
using Dates: now

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--gridsize"
            arg_type = Int
            default = 8

        "--t_prob"
            arg_type = Float64
            default = 0.7

        "--o_prob"
            arg_type = Float64
            default = 0.9

        "--z_dist_exp_const"
            arg_type = Float64
            default = 20.0

        "--exploration_const"
            arg_type = Float64
            default = 0.1

        "--sims_per_thread"
            arg_type = Int
            default = 10

        "--no_of_threads"
            arg_type = Int
            default = 20

        "--max_timesteps"
            arg_type = Int
            default = 10

        "--rollout_random"
            arg_type = Bool
            default = true

        "--val_epochs"
            arg_type = Int
            default = 1000

        "--noise_seed"   # should be a prime number, due to how RNG is setup on different threads.
            arg_type = Int
            default = 149

        "--savename"
            help = "Save name of file for results. Any valid String accepted. Pass no arguments to skip saving."
            arg_type = String
            default = "benchmarkRun_" * string(now())
    end

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
