import numpy as np
from scipy.optimize import fsolve

################################
# Constant parameters
###############################
MM=0.028
mm=0.04
JJ=0.001
gg=9.81
LL=0.2
ll=0.05
Ms=MM+mm
m_M=mm/MM
    
MM_plus_mm=1/(MM + mm)
M_plus_m_times_mm_times_LL=(mm * LL) * MM_plus_mm; 
M_plus_m_times_mm_over_MM=(mm/MM)*MM_plus_mm 
MM_times_LL=(MM*LL); 
LL_over_JJ=(ll/JJ); 

# Define a system of nonlinear equations
def equations(x):

    alpha = x[0]
    theta = x[1]
    omega_alpha = x[2]
    Fs = Ms*gg; 
    Fd = 0 

    omega_alpha_2=omega_alpha**2
    sin_alpha=np.sin(alpha)
    sin_theta=np.sin(theta); 
    sin_alpha_minus_theta=np.sin(alpha - theta)
    cos_alpha=np.cos(alpha)
    cos_theta=np.cos(theta)
    
    eq1= M_plus_m_times_mm_times_LL * omega_alpha_2 * sin_alpha - MM_plus_mm*Fs*sin_theta + M_plus_m_times_mm_over_MM*Fs*sin_alpha_minus_theta*cos_alpha
    eq2= -M_plus_m_times_mm_times_LL*omega_alpha_2*cos_alpha + MM_plus_mm*Fs*cos_theta + M_plus_m_times_mm_over_MM*Fs*sin_alpha_minus_theta*sin_alpha -gg
    eq3= (-Fs* sin_alpha_minus_theta)/MM_times_LL
    #eq4= LL_over_JJ*Fd

    return [eq1, eq2, eq3 ]
# Initial guess
initial_guess = [1, 1, 1 ]

# Use fsolve to find the roots
roots = fsolve(equations, initial_guess)

print("Roots:", roots)
