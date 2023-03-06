using ArgParse
using Dates: now

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--gridsize"
            arg_type = Int
            default = 6

        "--t_and_o_prob"
            arg_type = Float64
            default = 0.8

        "--z_dist_exp_const"
            arg_type = Float64
            default = 3.0

        "--no_of_simulations"
            arg_type = Int
            default = 100

        "--exploration_const"
            arg_type = Float64
            default = 1.0

        "--timesteps"
            arg_type = Int
            default = 4

        # "--obs_N"
        #     arg_type = Int
        #     default = 2

        # "--belief_N"
        #     arg_type = Int
        #     default = 10

        "--val_epochs"
            arg_type = Int
            default = 100_000

        "--noise_seed"
            arg_type = Int
            default = 123

        "--savename"
            help = "Save name of file for results. Any valid String accepted. Pass no arguments to skip saving."
            arg_type = String
            default = "benchmarkRun_" * string(now()) * ".csv"
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
