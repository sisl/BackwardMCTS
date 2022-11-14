using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--gridsize"
            arg_type = Int
            default = 6

        "--t_and_o_prob"
            arg_type = Float64
            default = 0.9

        "--z_val"
            arg_type = Float64
            default = 0.5

        "--timesteps"
            arg_type = Int
            default = 4

        "--obs_N"
            arg_type = Int
            default = 1

        "--belief_N"
            arg_type = Int
            default = 1

        "--noise_seed"
            arg_type = Int
            default = 123

        "--savename"
            help = "Save name of file for results. Any valid String accepted. Pass no arguments to skip saving."
            arg_type = String
            default = nothing
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
