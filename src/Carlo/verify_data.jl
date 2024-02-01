using CSV
using DataFrames: DataFrame, nrow
using ProgressBars: tqdm
using Statistics: mean, median

### Params ###
verify_csv = true
verify_python_to_julia = false
##############


function verify(;plot_data=true)

    files = readdir("../csvfiles"; join=true)

    X = zeros(length(files))
    Y = zeros(length(files))

    @info "Using $(Threads.nthreads()) threads."
    Threads.@threads for i = tqdm(1:length(files))
        df = DataFrame(CSV.File(files[i]))
        X[i] = df[end, "X_Position"]
        Y[i] = df[end, "Y_Position"]
    end
    
    if plot_data
        fig = scatter(X, Y, legend=false)
        return (X, Y, fig)
    else
        return (X, Y)
    end
end


function velocities()

    files = readdir("../csvfiles_rival"; join=true)

    vels = zeros(length(files))

    @info "Using $(Threads.nthreads()) threads."
    Threads.@threads for i = tqdm(1:length(files))
        df = DataFrame(CSV.File(files[i]))
        X = df[:,"X_Velocity"]
        Y = df[:,"Y_Velocity"]
        vels[i] = mean(sqrt.(X.^2 + Y.^2))
    end
    
    return vels
end

###########################################################

function compare_python_tick()
    files = readdir("../csvfiles"; join=true)
    X_err = zeros(length(files))
    Y_err = zeros(length(files))
    
    @info "Using $(Threads.nthreads()) threads."
    Threads.@threads for i = tqdm(1:length(files))
        
        df = DataFrame(CSV.File(files[i]))
        state = CarloCar(center=[df[1, :X_Position], df[1, :Y_Position]], heading=df[1, :Heading])
        
        X_Position_jl = []
        Y_Position_jl = []

        for r in 1:nrow(df)
            state.inputSteering = df[r, :U_Steering]
            state.inputAcceleration = df[r, :U_Throttle]
            tick!(state, dt=0.1)
            append!(X_Position_jl, state.center[1])
            append!(Y_Position_jl, state.center[2])
        end

        X_err[i] = mean(abs.(X_Position_jl - df.X_Position))
        Y_err[i] = mean(abs.(Y_Position_jl - df.Y_Position))

    end

    return X_err, Y_err
end

###########################################################

if verify_csv 
    using Plots
    X, Y, fig = verify(plot_data=true)
    @show vels = velocities()
    Plots.display(fig)
end

if verify_python_to_julia
    using Statistics: mean, median, std
    include("CarloCar_dynamics.jl")
    X_err, Y_err = compare_python_tick()
    @show mean(X_err), median(X_err)
    @show mean(Y_err), median(Y_err)
end