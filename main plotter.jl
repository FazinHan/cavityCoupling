# using QuantumCumulants
# using OrdinaryDiffEq, ModelingToolkit
ENV["MPLBACKEND"] = "qt5agg"
using QuantumCumulants
using OrdinaryDiffEq, ModelingToolkit
using PyPlot
using DelimitedFiles
import LinearAlgebra as la
using Statistics



multiplier = 1
M_1 = 10900
M_2 = 1750
# M_3 = 0.175
y1 = 2.94e-3*multiplier
y2 = 1.76e-2*multiplier/2/pi

@cnumbers ω1 ω2 ωc g1 g2 g3 λ1 λ2 λc κ1 κ2 κc α1 α2 β  # 2-magnon, 2-photon
@cnumbers Ω1 Ω2 Ωc 
h1 = FockSpace(:cavity);h2 = FockSpace(:cavity);hc = FockSpace(:cavity)
h=h1⊗h2⊗hc
# Define the fundamental operators
@qnumbers b1::Destroy(h,1) b2::Destroy(h,2) bc::Destroy(h,3)
#            magnon PY       magnon YIG        resonator

Ham = ω1*(1-im*α1)*(b1'*b1) + ω2*(1-im*α2)*(b2'*b2) + ωc*(1-im*β)*(bc'*bc) + g1*((b1'*bc)+(bc'*b1)) + g2*((bc'*b2)+(b2'*bc)) + Ω1*( b1'+b1) + Ω2*(b2'+b2) + Ωc*(bc'+bc) + g3*((b1'*b2)+(b2'*b1))
# Collapse operators
J = [b1,b2,bc]; rates = [λ1,λ2,λc]
# Derive a set of equations
ops = [b1,b2,bc]; eqs = meanfield(ops,Ham,J;rates=rates,order=1)

# Complete equations
eqs_completed = complete(eqs);
@named sys = ODESystem(eqs_completed);
display(sys)
A = calculate_jacobian(sys); B=[eqs_completed[1].rhs.dict[Ω1] * Ω1; eqs_completed[2].rhs.dict[Ω2] * Ω2; eqs_completed[3].rhs.dict[Ωc] * Ωc];
# Ainv=inv(A); X=Ainv*B; b1=X[1]; b2=X[2]; bc=X[3];

function main_calc_real_part(Hlist,ωcn,g1n,g2n,g3n,λ1n,λ2n,λcn,α1n,α2n,βn, M_1=M_1, y1=y1,M_2_=M_2, y2_=y2)
    ω2n = H -> y2_ * (H*(H+M_2_))^.5
    ω1n = H -> y1 * (H*(H+M_1))^.5
    # println("Running main_calc_real_part")
    occupationList1 = Float64[]; occupationList2 = Float64[]; occupationList3 = Float64[];
    for H in Hlist
        # if H
        An=substitute( A, Dict(ω1=>ω1n(H),ωc=>ωcn,ω2=>ω2n(H),g1=>g1n,g2=>g2n,g3=>g3n,λ1=>λ1n,λ2=>λ2n,λc=>λcn,α1=>α1n,α2=>α2n,β=>βn))
        data0 = 1im .* An
        data1 = la.eigen(data0)
        datar=la.real(data1.values)
        sort!(datar,rev=true)
        # datar = filter(x -> x >= minimum(locs) && x <= maximum(locs), datar)
        if length(datar) == 1
            datar = [datar[1], datar[1], datar[1]]
        elseif length(datar) == 0
            # println(H)# = [datar[1], datar[1]]
            datar = [ω2n, ω2n, ω2n]
        elseif length(datar) == 2
            datar = [datar[1], datar[2], datar[2]]
        end
        # println(size(datar))
        # r1n=datar[1]; r2n=datar[2];# r3n=datar[3]; 
        #print(An)
        push!(occupationList1, datar[1]); push!(occupationList2, datar[2]); push!(occupationList3, datar[3]);
    end
    occupationList = [occupationList1 occupationList2 occupationList3]

    return occupationList

end

function main(type, optimized_params)#, lone=false)

    println("Running main for $type")

    # type = "strong"
    root = joinpath(pwd(),"data","yig_t_sweep_new")
    # if lone
    #     root = joinpath(pwd(),"data","lone_t_sweep_yig")
    # end
    # Read the CSV file into a DataFrame
    # file_path = joinpath(root,"strong_peaks_widths.csv")
    # file_path = joinpath(root, "peaks_widths", "$type"*"_peaks_widths.csv")
    file_path_full = joinpath(root,"$type.csv")
    # df = readdlm(file_path, ',', Float64, '\n',skipstart=1)
    full_data = readdlm(file_path_full,',',Float64,'\n')

    # Display the first few rows of the DataFrame
    frequencies = full_data[2:end,1];
    s21 = full_data[2:end,2:end];
    # locs = df[:,1:2] * 2e9 * pi;
    # locs = sort(locs, dims=2)
    Hlist = full_data[1,2:end];

    # for i in 1:length(Hlist)
    s21_H = s21
    # if i == 150
    #     println("s21_H: ", s21_H)
    # end
    s21_H[ (s21_H .- mean(s21_H)) ./ std(s21_H) .> 3 ] .= 0
    s21 = s21_H
    # end

    function inter(Hlist, ωcn, params)
        n_field = length(Hlist)
        n_freq = length(frequencies)
        array = zeros(n_field, n_freq)
        data_points = main_calc_real_part(Hlist,ωcn, params...)
        for i in 1:3
            for (idx,point) in enumerate(data_points[:,i])
                index = findmin(abs.(frequencies .- point))[2]
                array[idx,index] += 1
            end
        end
        array = array'  # Transpose the array to match the dimensions of the s21 array
        
        sq_error = (s21.+array).^2
        
        return sum(sq_error)
    end

    

    root = joinpath(pwd(),"results")

   
    # ωcn = optimized_params[1]
    # optimized_params = optimized_params[2:end]

    # objective(params) = inter(Hlist, ωcn, params)

    # # Perform the optimization
    # lower = [5.06, 0, 0]
    # upper = [3.3, 1, 1]
    # inner_optimizer = LBFGS()
    # result = optimize(objective,lower,upper,initial_params,Fminbox(inner_optimizer))
    # result = optimize(objective,optimized_params,inner_optimizer)
    # Extract optimized parameters
    # optimized_params = Optim.minimizer(result)
    # optimized_params = [0.05, 0.19]

    println("Optimized parameters: ",  optimized_params)
    #Numerical calculations of dispersion spectra for case-1 (J > Γ)

    occupationList = main_calc_real_part(Hlist,optimized_params...)

    return (Hlist, frequencies, s21, occupationList)#, optimized_params)
end

function s21_theoretical(w, H, ωc, g1, g2, g3, alpha_1, alpha_2, lambda_1, lambda_2, lambda_r, beta)
    # Constants
    gyro1 = 2.94e-3
    gyro2 = 1.76e-2 / (2π)
    M1 = 10900.0  # Py
    M2 = 1750.0   # YIG

    # gamma_1 = 0.0001
    gamma_1 = 2*pi*lambda_1^2
    # gamma_2 = 0.008
    gamma_2 = 2*pi*lambda_2^2
    # gamma_r = 0.02
    gamma_r = 2*pi*lambda_r^2

    # alpha_1 = 0.0
    # alpha_2 = 0.0
    alpha_r = beta

    omega_1 = gyro1 * sqrt(H * (H + M1))
    omega_2 = gyro2 * sqrt(H * (H + M2))
    omega_r = ωc

    tomega_1 = omega_1 - 1im * (alpha_1 + gamma_1)
    tomega_2 = omega_2 - 1im * (alpha_2 + gamma_2)
    tomega_r = omega_r - 1im * (alpha_r + gamma_r)

    M = 1im * [
        w - tomega_1                       -g1 + 1im * sqrt(gamma_1 * gamma_r)    -g3 + 1im * sqrt(gamma_1 * gamma_2);
    -g1 + 1im * sqrt(gamma_1 * gamma_r) w - tomega_r                           -g2 + 1im * sqrt(gamma_2 * gamma_r);
        -g3 + 1im * sqrt(gamma_1 * gamma_2)      -g2 + 1im * sqrt(gamma_2 * gamma_r)    w - tomega_2
    ]

    B = sqrt(2) * [sqrt(gamma_1); sqrt(gamma_r); sqrt(gamma_2)]

    result = la.transpose(B) * la.inv(M) * B

    return abs(result[1,1])
end

function plot_multiple_calculations(params, save_file, plot_size=3, width_excess=0, lone=false, nrows=2, theo=true)
    # files = keys(params)
    files = sort(collect(keys(params)))
    num_plots = length(files)*(2^theo)
    # nrows = 2
    ncols = floor(Int, num_plots / nrows)
    # ncols = max(ceil(Int, num_plots / nrows),2)
    println("ncols:",ncols)

    # plot_size = 4

    # Create a figure and a grid of subplots
    # fig, axes = subplots()

    fig, axes = subplots(nrows, ncols, figsize=(plot_size*ncols+width_excess,plot_size*nrows), sharey=theo, sharex=true)
    # idx = 6

    conversion = 1e3

    images = []
    for (idx, file) in enumerate(files)
        param = params[file]

        if lone
            M_1 = 10900
            M_2 = 1750
            # M_3 = 0.175
            y1 = 2.94e-3
            y2 = 1.76e-2/2/pi
            if idx == 1
                # M_2 = .1
                y2 = 1
            elseif idx == 2
                M_1 = 1e9
                M_2 = 1500
                # println("M_1: ", M_1)
            else
                # y2 = 1.6e-2/2/pi
                M_2 = 1650
                # M_2 = 1750
            end
            # occupationList = main_calc_real_part(Hlist,param...,M_2, y2)
            param = vcat(param, [M_1, y1, M_2, y2])
        end

        Hlist, frequencies, s21, occupationList = main(file, param)
        # Hlist, frequencies, s21, occupationList, param = main(file, param)

        Hlist_old = Hlist

        Hlist = Hlist/conversion # Convert to kOe

        ωcn, g1n,g2n,g3n,λ1n,λ2n,λcn,α1n,α2n,βn = param

        s21_theoretical_array = zeros(length(frequencies), length(Hlist))

        if theo
            for i in eachindex(Hlist)
                for j in eachindex(frequencies)
                    s21_theoretical_array[j,i] = s21_theoretical(frequencies[j],Hlist_old[i],ωcn,g1n,g2n,g3n,α1n,α2n,λ1n,λ2n,λcn,βn) # s21(frequencies[j], Hlist[i], gyro1=y1, gyro2=y2, M1=M_1, M2=M_2, g1=g1n, g2=g2n, λ1=λ1n, λ2=λ2n, λc=λcn, α1=α1n, α2=α2n, β=βn)
                end
            end
            ax = axes[2idx-1]
        else
            ax = axes[idx]
        end

        println(size(s21_theoretical_array))

        if theo || idx==1
            for h in 1:length(Hlist)
                s21[:,h] .= (s21[:,h] .- minimum(s21[:,h])) ./ (maximum(s21[:,h]) .- minimum(s21[:,h]))
            end
        end
        
        if lone && idx == 3
            ax.text(220/conversion, 5.0, "P1", color="white", fontsize=12, ha="left")
            ax.text(1090/conversion, 4.96, "P2", color="white", fontsize=12, ha="left")
        end

        t = round(parse(Float64, split(file, "_")[3]), digits=3)
        tt = Int64(t*1e3)

        im = ax.pcolormesh(Hlist, frequencies, s21.-1, cmap=:inferno_r)
        push!(images, im)

        ax.plot(Hlist, occupationList[:,1], "w",alpha=.5)
        ax.plot(Hlist, occupationList[:,2], "w",alpha=.5)
        ax.plot(Hlist, occupationList[:,3], "w",alpha=.5)
        ax.set_ylim(4.3,6.2)

        if theo
            axes[2idx].pcolormesh(Hlist, frequencies, s21_theoretical_array, cmap=:inferno)
            ax.plot(Hlist, occupationList[:,1], "w",alpha=.5)
            ax.plot(Hlist, occupationList[:,2], "w",alpha=.5)
            ax.plot(Hlist, occupationList[:,3], "w",alpha=.5)
            ax.text(1150/conversion, 5.5, "t = $tt μm", color="white", fontsize=15, ha="right")
        end
        param = round.(param, digits=2)
        
        if idx==2 && !theo && lone
            lower, upper = 4.5, 6.3
            ax.set_ylim(lower, upper)
            ax.set_yticks([lower,lower + (upper-lower)/3, lower + (upper-lower)/3*2, upper])
        elseif !theo && lone
            lower, upper = 4.3, 5.8
            ax.set_ylim(lower, upper)
            ax.set_yticks([lower,lower + (upper-lower)/3, lower + (upper-lower)/3*2, upper])
            # ax.set_ylim(4.3,6)
        end

        # axes[idx].tick_params(axis="both", which="major", labelsize=15)
        # axes[idx].tick_params(axis="both", which="minor", labelsize=8)
        
    end

    for idx in eachindex(axes)
        axes[idx].tick_params(axis="both", which="major", labelsize=12)
        axes[idx].tick_params(axis="both", which="minor", labelsize=8)
        # if !lone && theo
        #     axes[idx].set_xticks([0,1])
        #     # axes[idx].set_yticks([5,6])
        # end
    end

    if lone
        axes[1].text(120/conversion, 4.4, "(a)", color="white", fontsize=15, ha="center")
        axes[2].text(120/conversion, 4.6, "(b)", color="white", fontsize=15, ha="center")
        axes[3].text(120/conversion, 4.4, "(c)", color="white", fontsize=15, ha="center")
        axes[1].text(550/conversion, 5.5, "Py", color="white", fontsize=15, ha="center")
        axes[2].text(1050/conversion, 5.7, "YIG", color="white", fontsize=15, ha="center")
        axes[3].text(800/conversion, 5.5, "Py+YIG", color="white", fontsize=15, ha="center")
    end
   
    fig.supxlabel("       Magnetic Field (kOe)", fontsize=12)#,ha="right")
    fig.supylabel("Frequency (GHz)", fontsize=12)
    tight_layout()
    
    for j in (num_plots+1):length(axes)
        axes[j].axis("off")
    end

    if lone
        asp = 25
        padding = .1
    else
        asp = 20
        padding = .02
    end

    cbar = fig.colorbar(mappable=images[1], ax=axes[1:end], orientation="vertical",aspect=asp,pad=padding)
    cbar.set_label("\$S_{21}\$ (a.u.)", fontsize=12)
    cbar.ax.tick_params(labelsize=12)  # Set colorbar tick label size

    # println(fig.tick_params)

    save_path = joinpath(pwd(),"tentative","images")

    savefig(joinpath(save_path,save_file),dpi=300,bbox_inches="tight")  # Save the figure with a tight layout
    # show()
    println("Saved figure to $save_file")
    close(fig)  # Close the figure if you don't want to display it
end

params = Dict( # ωcn  g1n  g2n g3n λ1n  λ2n  λcn  α1  α2  β
            #  "yig_t_0.000" => [5.09, .11, 0.0, .001,5e-2,1e-5,.08, 2e-2, 1e-5, 1e-5],  
             "yig_t_0.005" => [5.06, .12, 0.04, .001,5e-2,1e-5,.08, 2e-2, 1e-5, 1e-5],  
            #  "yig_t_0.013" => [5.01, .13, 0.075, .001,5e-2,1e-5,.08, 2e-2, 1e-5, 1e-5], 
             "yig_t_0.027" => [5.01, .14, 0.12, .001,5e-2,1e-5,.08, 2e-2, 1e-5, 1e-5],  
            #  "yig_t_0.040" => [5.04, .155, 0.13, .001,5e-2,1e-5,.08, 2e-2, 1e-5, 1e-5], 
             "yig_t_0.053" => [5.01, .16, 0.15, .001,5e-2,1e-5,.08, 2e-2, 1e-5, 1e-5], 
            #  "yig_t_0.067" => [5.02, .18, 0.18, .001,5e-2,1e-5,.08, 2e-2, 1e-5, 1e-5],  
             "yig_t_0.100" => [5.01, .2,  0.25, .001,5e-2,1e-5,.08, 2e-2, 1e-5, 1e-5], 
             )

println("Threads allocated: ", Threads.nthreads())

plot_multiple_calculations(params,"combined_plots.png",3,1)

params = Dict( # ωcn  g1n  g2n g3n λ1n  λ2n  λcn  α1  α2  β
             "yig_t_0.000" => [5.09, .11, 0.0, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
             "yig_t_0.100_lone" => [5.27, .0, 0.25, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
            #  "yig_t_0.005" => [5.06, .12, 0.04, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
            #  "yig_t_0.013" => [5.01, .13, 0.075, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
            #  "yig_t_0.027" => [5.01, .14, 0.12, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
            #  "yig_t_0.040" => [5.04, .155, 0.13, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
            #  "yig_t_0.053" => [5.01, .16, 0.15, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
            #  "yig_t_0.067" => [5.02, .18, 0.18, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
             "yig_t_0.100_z" => [5.01, .2,  0.21, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
             )

println("Threads allocated: ", Threads.nthreads())

plot_multiple_calculations(params,"combined_plots_isolate.png",3,1,true,3,false)