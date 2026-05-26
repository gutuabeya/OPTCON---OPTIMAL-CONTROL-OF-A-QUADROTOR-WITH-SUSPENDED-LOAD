import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import dynamics as dyn

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

#######################################
# Dynamics parameters
#######################################
dt = dyn.dt #1e-3 #get discretization step from dynamics
ns = dyn.ns #8    #get dimension of state from dynamics
ni = dyn.ni #2    #get dimension of input from dynamics

######################################
# Reference curve Parameters
######################################
tf = 1.0 #10 # final time in seconds
TT = int(tf/dt) #returns integer discrete-time samples
xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))
xx_des = np.zeros((ns, 2))
uu_des = np.zeros((ni, 2))
# Generate a bell-shaped reference curve
# Curve parameters
a = 1
b = tf/2 + dt
# sampling time
t_samp_temp = np.linspace(dt, 2*b - dt, int(tf/dt))

def bell_function(t, x, a):
    tau = t - b
    exp_arg = -(2*b**2)*(tau**2 + b**2)/(tau**2 - b**2)**2
    beta = a * np.exp(exp_arg)
    return beta*np.ones((1,))

def bell_create(des_curve,dimension):  
    ref=np.zeros((dimension, TT))
    for i in range(dimension):
        x0=(des_curve[i][0])*np.ones((1,)) 
        xf=(des_curve[i][1])*np.ones((1,))    
    #using solve_ivp for xp----------------------------------------------------------------
        sigma_temp = solve_ivp(bell_function, 
                        (dt, 2*b - dt), #time_span 
                        x0, 
                        t_eval = t_samp_temp, #the equation will be evalution at each of this time
                        args = (1,), 
                        method = 'Radau')
        norm_factor = sigma_temp.y[0][-1] # normalization factor

        sigma = solve_ivp(bell_function, 
                    (dt, 2*b - dt), #time_span start from dt upto 2b*-dt
                    x0, 
                    t_eval = t_samp_temp, #the equation will be evalution at each of this times
                    args = ((xf-x0)/norm_factor,), 
                    method = 'Radau').y[0]
        ref[i,:]=sigma
    return ref

# xx_ref=bell_create(xx_des, ns)
# uu_ref=bell_create(uu_des, ni)

# t_samp = t_samp_temp - dt

# def plott(ref, dimen):
#     for ff in range(dimen):
#         # Create a figure with two subplots side by side
#         fig, axs = plt.subplots(1, 2, figsize=(10, 4))
#         # Plot the first figure
#         axs[0].plot(t_samp_temp, ref[ff,:], label='desired_state')
#         axs[0].set_title('state_var bell Vs time')
#         axs[0].legend()
#       #Plot the second figure
#         axs[1].plot(t_samp_temp, ref[ff+1,:], label='desired_state', color='orange')
#         axs[1].set_title('state_var bell Vs time')
#         axs[1].legend()
#         plt.tight_layout() # Adjust layout for better spacing
#         #plt.axis('equal')
#         plt.show()

# plott(xx_ref, ns)
# plott(uu_ref, ni)

# # Plot the trajectory
# plt.plot(xx_ref[0,:], xx_ref[1,:])
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Quadrotor Trajectory')
# #plt.axis('equal')
# plt.show()
