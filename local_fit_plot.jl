using QuantumCumulants
using OrdinaryDiffEq, ModelingToolkit
# using Optim
using DifferentialEquations
using PyPlot
# using NPZ
# using ArgParse
# using BenchmarkTools
using DelimitedFiles
using BeepBeep
import LinearAlgebra as la
# Define parameters
multiplier = 1e9
M_1 = 10900
M_3 = 1750
# M_3 = 0.175
y1 = 2*pi*2.94e-3*multiplier
y3 = 1.76e-2*multiplier

@cnumbers ω1 ω2 ω3 g1 g2 γ1 γ2 γ3 Ω1 Ω2 Ω3  # 2-magnon, 2-photon
h1 = FockSpace(:cavity);h2 = FockSpace(:cavity);h3 = FockSpace(:cavity)
h=h1⊗h2⊗h3
# Define the fundamental operators
@qnumbers b1::Destroy(h,1) b2::Destroy(h,2) b3::Destroy(h,3)
#            magnon PY          resonator        magnon YIG

Ham = ω1*(b1'*b1) + ω2*(b2'*b2) + ω3*(b3'*b3) + g1*((b1'*b2)+(b2'*b1)) + g2*((b3'*b2)+(b2'*b3)) + Ω1*( b1'+b1) + Ω2*(b2'+b2) + Ω3*(b3'+b3)
# Collapse operators
J = [b1,b2,b3]; rates = [2γ1,2γ2,2γ3]
# Derive a set of equations
ops = [b1,b2,b3]; eqs = meanfield(ops,Ham,J;rates=rates,order=1)

# Complete equations
eqs_completed = complete(eqs);
@named sys = ODESystem(eqs_completed);
A = calculate_jacobian(sys); B=[eqs_completed[1].rhs.dict[Ω1] * Ω1; eqs_completed[2].rhs.dict[Ω2] * Ω2; eqs_completed[3].rhs.dict[Ω3] * Ω3];
Ainv=inv(A); X=Ainv*B; b1=X[1]; b2=X[2]; b3=X[3];

function main(type, optimized_params)

    println("Running main for $type")

    # type = "strong"
    root = joinpath(pwd(),"data","yig_t_sweep_outputs")
    # Read the CSV file into a DataFrame
    # file_path = joinpath(root,"strong_peaks_widths.csv")
    file_path = joinpath(root, "peaks_widths", "$type"*"_peaks_widths.csv")
    file_path_full = joinpath(root,"$type.csv")
    df = readdlm(file_path, ',', Float64, '\n',skipstart=1)
    full_data = readdlm(file_path_full,',',Float64,'\n')

    # Display the first few rows of the DataFrame
    frequencies = full_data[2:end,1] * 2e9 * pi;
    s21 = full_data[2:end,2:end];
    locs = df[:,1:2] * 2e9 * pi;
    locs = sort(locs, dims=2)
    Hlist = full_data[1,2:end];

    γ1n=0.1; γ2n=0.00469; γ3n=1.4e-4;
    ω3n = H -> y3 * (H*(H+M_3))^.5
    ω1n = H -> y1 * (H*(H+M_1))^.5

    function main_calc_real_part_opt(Hlist,ω2n,g1n,g2n)
        occupationList1 = Float64[]; occupationList2 = Float64[];
        for H in Hlist
                An=substitute( A, Dict(ω1=>ω1n(H),ω2=>ω2n*2e9*pi,ω3=>ω3n(H),g1=>g1n*2e9*pi,g2=>g2n*2e9*pi,γ1=>γ1n*2e9*pi,γ2=>γ2n*2e9*pi,γ3=>γ3n*2e9*pi))
                Ann = 1im * zeros(3,3)
                for i=1:3
                    for j=1:3
                        Ann[i,j] = real(An[i,j]) + 1im * imag(An[i,j])
                    end
                
                end
                # if abs(H-0.8) < 1e-2
                #     println(Ann*1im)
                # end
                if any(isinf, Ann) || any(isnan, Ann)
                    println("Inf or NaN detected!")
                    println("Parameters: H=$H, g1n=$g1n, g2n=$g2n, γ1n=$γ1n, γ2n=$γ2n, γ3n=$γ3n")
                    println("Matrix Ann: $Ann")
                end
                data0=Ann * 1im
                data1 = la.eigen(data0)
            
                datar=la.real(data1.values)
                sort!(datar,rev=true)
                datar = filter(x -> x >= minimum(locs) && x <= maximum(locs), datar)
                if length(datar) == 1
                    datar = [datar[1], datar[1]]
                end
                if length(datar) == 0
                    # println(H)# = [datar[1], datar[1]]
                    datar = [ω2n, ω2n]

                end
                # println(size(datar))
                # r1n=datar[1]; r2n=datar[2];# r3n=datar[3]; 
                #print(An)
            push!(occupationList1, datar[1]); push!(occupationList2, datar[2]);# push!(occupationList3, r3n)
        end
        occupationList = [occupationList1 occupationList2]
        return occupationList
    end

    function main_calc_real_part_full(Hlist,ω2n,g1n,g2n)
        occupationList1 = Float64[]; occupationList2 = Float64[]; occupationList3 = Float64[];
        for H in Hlist
                An=substitute( A, Dict(ω1=>ω1n(H),ω2=>ω2n*1e10,ω3=>ω3n(H),g1=>g1n*2e9*pi,g2=>g2n*2e9*pi,γ1=>γ1n*2e9*pi,γ2=>γ2n*2e9*pi,γ3=>γ3n*2e9*pi))
                Ann = 1im * zeros(3,3)
                for i=1:3
                    for j=1:3
                        Ann[i,j] = real(An[i,j]) + 1im * imag(An[i,j])
                    end
                
                end
                if any(isinf, Ann) || any(isnan, Ann)
                    println("Inf or NaN detected!")
                    println("Parameters: H=$H, g1n=$g1n, g2n=$g2n, γ1n=$γ1n, γ2n=$γ2n, γ3n=$γ3n")
                    println("Matrix Ann: $Ann")
                end
                data0=Ann * 1im
                data1 = la.eigen(data0)
            
                datar=la.real(data1.values)
            if !issorted(datar, rev=false)
                println("Warning: datar is not sorted in descending order at H=$H")
            end
            push!(occupationList1, datar[1]); push!(occupationList2, datar[2]); push!(occupationList3, datar[3]);
        end
        occupationList = [occupationList1 occupationList2 occupationList3]
        return occupationList
    end

    
    root = joinpath(pwd(),"results")

    function inter(Hlist, params)
        theoretical_values = main_calc_real_part_opt(Hlist, params...)
        
        # Compute the sum of squared differences for each occupation list
        # sum_sq_error = sum((theoretical_values[1] .- locs[:,1]).^2) +
                    #    sum((theoretical_values[2] .- locs[:,2]).^2)
                    #    sum((theoretical_values[3] .- locs[:,3]).^2)

        # score3arr = (theoretical_values[3] .- locs[2,:]).^2
        # score1arr = (theoretical_values[1] .- locs[1,:]).^2
        # score2arr1 = (theoretical_values[2] .- locs[1,:]).^2
        # score2arr2 = (theoretical_values[2] .- locs[2,:]).^2
        # score2arr = min(score2arr1, score2arr2)
        # sq_error = score1arr + score2arr + score3arr

        # println(size(theoretical_values[1]))
        score1 = (theoretical_values[:,1] .- locs[:,2]).^2;
        score2 = (theoretical_values[:,2] .- locs[:,1]).^2;
        
        sq_error = score1 .+ score2
        
        return sum(sq_error)
    end


    println("Optimized parameters: ", optimized_params)

    occupationList = main_calc_real_part_full(Hlist,optimized_params...)

    return (Hlist, frequencies, s21, occupationList)
end

# types = ["strong20","strong25","strong50","strong2"]
# types = ["strong50","strong2","strong25"]

# types = ["yig_t_0.06","yig_t_0.1"]


# function parse_command_line()
#     s = ArgParseSettings()
#     @add_arg_table s begin
#         "--thickness"
#         help = "Set the thickness value"
#         arg_type = Int64
#         required = true
#     end
#     return parse_args(s)
# end

function parallel_main(files)
    Threads.@threads for file in files
        main(file)
    end
    return nothing
end

function serial_main(files, optimized_params)
    for i in 1:length(files)
        main(files[i], optimized_params[i])
    end
    return nothing
end

function plot_multiple_calculations(params)
    # files = keys(params)
    files = sort(collect(keys(params)))
    num_plots = length(files)
    ncols = 2
    nrows = ceil(Int, num_plots / ncols)

    # Create a figure and a grid of subplots
    fig, axes = subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    idx = 6
    
    for (i, file) in enumerate(files)
    # for file in files
        param = params[file]
        Hlist, frequencies, s21, occupationList = main(file, param)
        # if idx == 5
        #     ax = axes[7]
        # else
        #     ax = axes[(idx+2)%7]
        # end
        # idx = (idx+2)%7
        ax=axes[i]

        t = round(parse(Float64, split(file, "_")[3]), digits=3)

        im = ax.pcolormesh(Hlist, frequencies, s21, cmap=:inferno_r, shading="auto", norm=matplotlib.colors.Normalize(vmin=minimum(s21), vmax=maximum(s21)))
        ax.plot(Hlist, occupationList[:,1], "w")
        ax.plot(Hlist, occupationList[:,2], "w")
        ax.plot(Hlist, occupationList[:,3], "w")
        ax.set_title("t = $t"*raw"$\mu$m"*"; Params = $param")
        # ax.set_xlabel("Magnetic Field (Oe)")
        # ax.set_ylabel("Frequency (GHz)")
        ax.set_ylim(2.75e10, 3.95e10)
        # if i % ncols == 0
            # fig.colorbar(im, ax=ax)
        # end
    end
    # fig.colorbar(im, ax=axes[end-1], orientation="horizontal")

    # If there are unused subplots, hide them
    for j in (num_plots+1):length(axes)
        axes[j].axis("off")
    end

    tight_layout()
    savefig("combined_plots_vertical.png")
    println("Saved figure to combined_plots_vertical.png")
    close(fig)  # Close the figure if you don't want to display it
end

# if Threads.nthreads() == 1
#     println("Running in serial mode, set environment variable!\nExiting...")
#     exit()
# end

params = Dict(
    "yig_t_0.02" => [3.2, 0.1, 0.0], 
    "yig_t_0.033333333333333" => [3.165, 0.1, 0.075],
    "yig_t_0.046666666666667" => [3.16, 0.12, 0.12], 
    "yig_t_0.06" => [3.18, 0.14, 0.135],
    "yig_t_0.073333333333333" => [3.165, 0.13, 0.155], 
    "yig_t_0.086666666666667" => [3.17, 0.15, 0.17], 
    "yig_t_0.1" => [3.17, 0.17, 0.18]
    )

println("Threads allocated: ", Threads.nthreads())

plot_multiple_calculations(params)

beep(4)
