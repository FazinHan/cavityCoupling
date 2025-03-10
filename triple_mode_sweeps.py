import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
sp.init_printing()

# Define other symbolic variables
b1,b2,b3,w,w1,w2,w3,a,b,v,v1,v2,v3,g1,g2,g3,pin,pout,s21,H = sp.symbols('b_1,b_2,b_3,w,w_1,w_2,w_3,a,b,v,v_1,v_2,v_3,g_1,g_2,g_3,p_{in},p_{out},s_{21},H')
#a=alpha b=beta v=gamma,v1=gamma1,v2=gamma2,v3=gamma3 b1,b2=annihilation op for magnon and b3= annihilattion op for photon
# Define imaginary unit
j = sp.I

# Define equations
eq11 = sp.Eq(j*((w-w1+j*a)*b1)-j*g2*b2-j*g3*b3-j*(sp.sqrt(v1)*pin)-v1*b1-sp.sqrt(v1*v2)*b2-sp.sqrt(v1*v3)*b3, 0)
eq22 = sp.Eq(j*((w-w2+j*b)*b2)-j*g1*b3-j*g2*b1-j*(sp.sqrt(v2)*pin)-sp.sqrt(v1*v2)*b1-v2*b2-sp.sqrt(v2*v3)*b3, 0)
eq33 = sp.Eq(j*((w-w3+j*v)*b3)-j*g1*b2-j*g3*b1-j*(sp.sqrt(v3)*pin)-sp.sqrt(v1*v3)*b1-sp.sqrt(v2*v3)*b2-v3*b3, 0)

eq44 = sp.Eq(j*((w-w1+j*a)*b1)-j*g2*b2-j*g3*b3-j*(sp.sqrt(v1)*pout)+v1*b1+sp.sqrt(v1*v2)*b2+sp.sqrt(v1*v3)*b3, 0)
eq55 = sp.Eq(j*((w-w2+j*b)*b2)-j*g1*b3-j*g2*b1-j*(sp.sqrt(v2)*pout)+sp.sqrt(v1*v2)*b1+v2*b2+sp.sqrt(v2*v3)*b3, 0)
eq66 = sp.Eq(j*((w-w3+j*v)*b3)-j*g1*b2-j*g3*b1-j*(sp.sqrt(v3)*pout)+sp.sqrt(v1*v3)*b1+sp.sqrt(v2*v3)*b2+v3*b3, 0)
eq7 = sp.Eq(s21, (pout/pin)-1)

#eq6

#t = 0
t = sp.pi/3

v1_value=0.0001     #gamma1 extrinsic damping of magnon1 mode 
# v1_value = sys.argv[1]
v2_value=0.008 #* (sp.cos(t))**2      #gamma1 extrinsic damping of magnon2 mode
# v2_value = sys.argv[2]

v3_value=0.02 #* (sp.sin(t))**2      #gamma1 extrinsic damping of cavity mode
# v3_value = sys.argv[2]

# w2_value=3.9790176    # cavity 2 resonance frequency
w3_value=5.71             # cavity resonance frequency
a_value=0.00014  # intrinsic damping of magnon1 mode
b_value=0.00369 # intrinsic damping of magnon2 mode
v_value=0.005268  # intrinsic damping of cavity mode
g1_value=0.038
g2_value=0.033
g3_value=0.058

def compute_s21(v1_value, v2_value):
    eq1=eq11.subs({v1:v1_value,
                    v2:v2_value,
                    v3:v3_value,
                    # w2:w2_value,
                    w3:w3_value,
                    a:a_value,
                    b:b_value,
                    v:v_value,
                    g1:g1_value,
                    g2:g2_value,
                    g3:g3_value})
    eq2=eq22.subs({v1:v1_value,
                    v2:v2_value,
                    v3:v3_value,
                    # w2:w2_value,
                    w3:w3_value,
                    a:a_value,
                    b:b_value,
                    v:v_value,
                    g1:g1_value,
                    g2:g2_value,
                    g3:g3_value})
    eq3=eq33.subs({v1:v1_value,
                    v2:v2_value,
                    v3:v3_value,
                    # w2:w2_value,
                    w3:w3_value,
                    a:a_value,
                    b:b_value,
                    v:v_value,
                    g1:g1_value,
                    g2:g2_value,
                    g3:g3_value})
    eq4=eq44.subs({v1:v1_value,
                    v2:v2_value,
                    v3:v3_value,
                    # w2:w2_value,
                    w3:w3_value,
                    a:a_value,
                    b:b_value,
                    v:v_value,
                    g1:g1_value,
                    g2:g2_value,
                    g3:g3_value})
    eq5=eq55.subs({v1:v1_value,
                    v2:v2_value,
                    v3:v3_value,
                    # w2:w2_value,
                    w3:w3_value,
                    a:a_value,
                    b:b_value,
                    v:v_value,
                    g1:g1_value,
                    g2:g2_value,
                    g3:g3_value})
    eq6=eq66.subs({v1:v1_value,
                    v2:v2_value,
                    v3:v3_value,
                    # w2:w2_value,
                    w3:w3_value,
                    a:a_value,
                    b:b_value,
                    v:v_value,
                    g1:g1_value,
                    g2:g2_value,
                    g3:g3_value})


    # Solve eq1 for b1 in terms of b2 and b3
    sol_b1 = sp.solve(eq1, b1)[0]

    # Substitute sol_b1 into eq2 and solve for b2 in terms of b3
    eq2_sub = eq2.subs(b1, sol_b1)
    sol_b2 = sp.solve(eq2_sub, b2)[0]

    # Substitute sol_b1 and sol_b2 into eq3 and solve for b3
    eq3_sub = eq3.subs({b1: sol_b1, b2: sol_b2})
    sol_b3 = sp.solve(eq3_sub, b3)[0]

    # Substitute sol_b3 back into sol_b2 to get the final expression for b2
    sol_b2_final = sol_b2.subs(b3, sol_b3)

    # Substitute sol_b3 and sol_b2_final back into sol_b1 to get the final expression for b1
    sol_b1_final = sol_b1.subs({b2: sol_b2_final, b3: sol_b3})

    # Compute the difference between eq1 and eq3
    difference_eq1_eq3 = eq1.lhs - eq4.lhs

    # Define the new equation
    new_eq = sp.Eq(difference_eq1_eq3, 0)
    # Substitute the values of a and b into new_eq
    new_eq_substituted = new_eq.subs({b1: sol_b1_final, b2: sol_b2_final, b3: sol_b3})
    solution3 = sp.solve(new_eq_substituted, pout)

    # Solve equation eq5 for s21 after substituting pout from solution3
    try:
        solution4 = sp.solve(eq7.subs(pout, solution3[0]), s21)
    except IndexError:
        # solution4 = sp.solve(eq7.subs(pout, solution3), s21)
        print("no solution found for v1 = ", v1_value, " and v2 = ", v2_value)

    # Assign the value of s21 to a variable
    S = solution4[0]

    print("solution computed")
    # print("solution computed\nstarting plotter...")

    import pickle
    # S = pickle.load(file:=open('s21_analytical.p', 'rb'))
    jobid = os.popen("echo $SLURM_JOB_ID").read()
    pickle.dump(S, file:=open(f's21_analytical_v1={v1_value}_v2={v2_value}_{jobid}.p', 'wb'))
    print("solution saved")
    file.close()
# import pandas as pd
# import os



# # Define parameter ranges
# type = "yig_t_0.02"


# root = os.path.join("C:\\Users\\freak\\OneDrive\\Documents\\core\\Projects\\cavityCoupling","data","lone_t_sweep_yig")
# file_path_full = os.path.join(root,f"{type}.csv")

# full_data = pd.read_csv(file_path_full)

# # Display the first few rows of the DataFrame
# frequencies = full_data.to_numpy()[:,0]
# # frequencies = np.linspace(4.5, 6, frequencies.size) 
# Hlist = np.array(full_data.columns)[1:].astype(float) # Skip the first column which is 'Frequency'
# # Hlist = np.linspace(1.050, 1.350, Hlist.size)
# s21 = full_data.to_numpy()[:,1:] # Skip the first column which is 'Frequency'

# print("all data loaded")

# # H_values = np.linspace(0.600, 1.500, 201)
# H_values = Hlist
# # w_values = np.linspace(3.5, 6.5, 501)
# w_values = frequencies

# # Convert symbolic expression s21 to a NumPy function
# s21_func = sp.lambdify((w,w1,w2), sp.Abs(S), modules='numpy')

# print("starting lambda computation...")

# # Initialize a list to store s21_f
# s21_f = []

# # Loop over each H value
# for H in H_values:
#     # Calculate wb_value for the current H value
#     wb1_value = 1.68e-2/2/np.pi*np.sqrt(H * (H + 1750))
#     wb2_value = 2.94e-3*np.sqrt(H * (H + 10900))
#     # wb_value = 2.8*np.sqrt(H * (H + 0.172))
#     # w1_value=1.38284384+(3.19289744)*H
    
#     # Initialize a list to store s21_3 values for the current H value
#     s21_f_H = []
    
#     for w_val in w_values:
#         s21_2 = s21_func(w_val, wb1_value, wb2_value)
#         s21_f_H.append(s21_2)
      
#     # Append s21_f_H to s21_f as a row
#     s21_f.append(s21_f_H)

# # Convert s21_f to a numpy array and transpose it
# s21_f = np.array(s21_f).T  # Transpose s21_f here
# H_f = np.array(H_values)
# w_f = np.array(w_values)

# print("arrays created\nplotting...")

# # Create a contour plot
# #contour = plt.contourf(H_f, w_f, s21_f, )
# contour = plt.contourf(H_f, w_f, s21_f, cmap='inferno', levels=np.linspace(0, 2, 501))
# plt.colorbar(contour)
# plt.xlabel('H')
# plt.ylim([5.5,6])    
# plt.ylabel('w')
# # plt.title('Contour Plot of s21')
# plt.tight_layout()
# plt.savefig('s21_contour_plot_3mode_new.png')
# print("plot saved")

# #print(s21_f)
# #print(H_f)
# #print(w_f)
# #s21_f.shape[0]
# #H_f.shape[0]
# #plt.ylim([3.6, 4.4])
# #plt.xlim([1250, 1450])
# # plt.clim(-7, 0)
