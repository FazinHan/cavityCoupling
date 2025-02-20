import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
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
v2_value=0.008 * (sp.cos(t))**2      #gamma1 extrinsic damping of magnon2 mode
v3_value=0.02 * (sp.sin(t))**2      #gamma1 extrinsic damping of cavity mode
# w2_value=3.9790176    # cavity 2 resonance frequency
w3_value=5.71             # cavity 3 resonance frequency
a_value=0.00014  # intrinsic damping of magnon1 mode
b_value=0.00369 # intrinsic damping of magnon2 mode
v_value=0.005268  # intrinsic damping of cavity mode
g1_value=0.038
g2_value=0.033
g3_value=0.058

eq1=eq11.subs({v3:v3_value,w3:w3_value,a:a_value,b:b_value,v:v_value,g1:g1_value,g2:g2_value,g3:g3_value})
eq2=eq22.subs({v3:v3_value,w3:w3_value,a:a_value,b:b_value,v:v_value,g1:g1_value,g2:g2_value,g3:g3_value})
eq3=eq33.subs({v3:v3_value,w3:w3_value,a:a_value,b:b_value,v:v_value,g1:g1_value,g2:g2_value,g3:g3_value})

eq4=eq44.subs({v3:v3_value,w3:w3_value,a:a_value,b:b_value,v:v_value,g1:g1_value,g2:g2_value,g3:g3_value})
eq5=eq55.subs({v3:v3_value,w3:w3_value,a:a_value,b:b_value,v:v_value,g1:g1_value,g2:g2_value,g3:g3_value})
eq6=eq66.subs({v3:v3_value,w3:w3_value,a:a_value,b:b_value,v:v_value,g1:g1_value,g2:g2_value,g3:g3_value})

# Solve eq1 for b1 in terms of b2 and b3
sol_b1 = sp.solve(eq1, (b1,v1,v2))[0]

# Substitute sol_b1 into eq2 and solve for b2 in terms of b3
eq2_sub = eq2.subs(b1, sol_b1)
sol_b2 = sp.solve(eq2_sub, (v1,v2,b2))[0]

# Substitute sol_b1 and sol_b2 into eq3 and solve for b3
eq3_sub = eq3.subs({b1: sol_b1, b2: sol_b2})
sol_b3 = sp.solve(eq3_sub, (v1,v2,b3))[0]

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
solution4 = sp.solve(eq7.subs(pout, solution3[0]), s21)

# Assign the value of s21 to a variable
S = solution4[0]

import pickle
pickle.dump(S, open('S.p', 'wb'))