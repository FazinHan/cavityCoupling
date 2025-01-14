using QuantumCumulants
using OrdinaryDiffEq, ModelingToolkit
using DifferentialEquations
using PyPlot
using BeepBeep
using DelimitedFiles
import LinearAlgebra as la

function main(type, optimized_params)

    println("Running main for $type")

    root = joinpath(pwd(),"data","yig_t_sweep_outputs")
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

    function main_calc_real_part_full(Hlist,yig_t,ω2n)
        occupationList1 = Float64[]; occupationList2 = Float64[]; occupationList3 = Float64[];
        # Derive a set of equations
        multiplier = 1e9
        M_1 = 10900
        M_3 = 1750
        # M_3 = 0.175
        y1 = 2*pi*2.94e-3*multiplier
        y3 = 1.76e-2*multiplier

        ## fit coefficients
        a1, b1 = .39, .04
        a2, b2 = .98,-.11

        @cnumbers ω1 ω2 ω3 γ1 γ2 γ3 Ω1 Ω2 Ω3  # 2-magnon, 2-photon
        h1 = FockSpace(:cavity);h2 = FockSpace(:cavity);h3 = FockSpace(:cavity)
        h=h1⊗h2⊗h3
        # Define the fundamental operators
        @qnumbers m1::Destroy(h,1) a_r::Destroy(h,2) m2::Destroy(h,3)
        #            magnon PY          resonator        magnon YIG
        @syms t::Real
        @register_symbolic g1(t) g2(t)

        # Define g1 and g2 as functions of yig_t
        g1(t) = (a1 * log(t) + b1)*2e9*pi
        g2(t) = (a2 * log(t) + b2)*2e9*pi

        Ham(t) = ω1*(m1'*m1) + ω2*(a_r'*a_r) + ω3*(m2'*m2) + g1(t)*((m1'*a_r)+(a_r'*m1)) + g2(t)*((m2'*a_r)+(a_r'*m2)) + Ω1*( m1'+m1) + Ω2*(a_r'+a_r) + Ω3*(m2'+m2)
        # Collapse operators
        J = [m1,a_r,m2]; rates = [2γ1,2γ2,2γ3]

        ops = [m1,a_r,m2]; eqs = meanfield(ops,Ham(yig_t),J;rates=rates,order=1)

        # Complete equations
        eqs_completed = complete(eqs);
        @named sys = ODESystem(eqs_completed);
        A = calculate_jacobian(sys); B=[eqs_completed[1].rhs.dict[Ω1] * Ω1; eqs_completed[2].rhs.dict[Ω2] * Ω2; eqs_completed[3].rhs.dict[Ω3] * Ω3];
        Ainv=inv(A); X=Ainv*B; m1=X[1]; a_r=X[2]; m2=X[3];
        for H in Hlist
                An=substitute( A, Dict(ω1=>ω1n(H),ω2=>ω2n*1e10,ω3=>ω3n(H),γ1=>γ1n*2e9*pi,γ2=>γ2n*2e9*pi,γ3=>γ3n*2e9*pi))
                # An=substitute( A, Dict(ω1=>ω1n(H),ω2=>ω2n*2e9*pi,ω3=>ω3n(H),g1=>g1n*2e9*pi,g2=>g2n*2e9*pi,γ1=>γ1n*2e9*pi,γ2=>γ2n*2e9*pi,γ3=>γ3n*2e9*pi))
                Ann = 1im * zeros(3,3)
                for i=1:3
                    for j=1:3
                        Ann[i,j] = real(An[i,j]) + 1im * imag(An[i,j])
                    end
                
                end
                if any(isinf, Ann) || any(isnan, Ann)
                    println("Inf or NaN detected!")
                    println("Parameters: H=$H, yig_t=$yig_t, γ1n=$γ1n, γ2n=$γ2n, γ3n=$γ3n")
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

    initial_params = optimized_params

    objective(params) = inter(Hlist, params)

    # # Perform the optimization

    println("Optimized parameters: ", optimized_params)
    #Numerical calculations of dispersion spectra for case-1 (J > Γ)

    occupationList = main_calc_real_part_full(Hlist,optimized_params...)

    return (Hlist, frequencies, s21, occupationList)
end

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
    nrows = 2
    ncols = ceil(Int, num_plots / nrows)

    # Create a figure and a grid of subplots
    fig, axes = subplots(nrows, ncols, figsize=(5*ncols, 10))
    idx = 6
    
    for (i, file) in enumerate(files)
    # for file in files
        param = params[file]
        Hlist, frequencies, s21, occupationList = main(file, param)
        if idx == 5
            ax = axes[7]
        else
            ax = axes[(idx+2)%7]
        end
        idx = (idx+2)%7

        t = round(parse(Float64, split(file, "_")[3]), digits=3)

        im = ax.pcolormesh(Hlist, frequencies, s21, cmap=:inferno_r)
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
    savefig("combined_plots_remodel.png")
    println("Saved figure to combined_plots_remodel.png")
    close(fig)  # Close the figure if you don't want to display it
end

# if Threads.nthreads() == 1
#     println("Running in serial mode, set environment variable!\nExiting...")
#     exit()
# end

params = Dict("yig_t_0.02" => [0.02,3.2], 
             "yig_t_0.033333333333333" => [0.033333333333333,3.165],
             "yig_t_0.046666666666667" => [0.046666666666667,3.16], 
             "yig_t_0.06" => [0.06,3.18],
             "yig_t_0.073333333333333" => [0.073333333333333,3.165], 
             "yig_t_0.086666666666667" => [0.086666666666667,3.17], 
             "yig_t_0.1" => [0.1,3.17])

println("Threads allocated: ", Threads.nthreads())

plot_multiple_calculations(params)

beep(4)
