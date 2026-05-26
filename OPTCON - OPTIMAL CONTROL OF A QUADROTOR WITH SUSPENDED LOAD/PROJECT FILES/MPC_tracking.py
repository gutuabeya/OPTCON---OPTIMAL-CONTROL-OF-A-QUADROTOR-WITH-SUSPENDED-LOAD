import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import dynamics as dyn
from referance_curve import tf
from solver import linear_mpc
dt = dyn.dt
TT = int(tf/dt)

# Load the array from the saved file
loaded_data = np.load('xx_opt.npz')
xx_opt = loaded_data['result_array']

loaded_data = np.load('uu_opt.npz')
uu_opt = loaded_data['result_array']

# Load the array from the saved file
loaded_data = np.load('xx_ref.npz')
xx_ref = loaded_data['result_array']

loaded_data = np.load('uu_ref.npz')
uu_ref = loaded_data['result_array']

ns = dyn.ns
ni = dyn.ni

xx = np.zeros((ns,1))
uu = np.zeros((ni,1))
xx0 = np.zeros((ns,1))
xx0 = np.array([[0.1],[0.1],[0],[0],[0],[0.1],[0.1],[0.2]]) #considered perturbed initial condition
# xx0=xx0.squeeze()

AAnom= np.zeros((ns,ns,TT))
BBnom= np.zeros((ns,ni,TT))

T_pred = 20      # MPC Prediction horizon
Tsim = TT-T_pred    # simulation horizon

########################
# Linearize the dynamics about the optimal trajectory to get nominal A,B matrices
########################
for tt in range(TT): 
  fx, fu = dyn.dynamics(xx_opt[:,tt],uu_opt[:,tt])[1:]
  AAnom[:,:,tt] = fx.T  
  BBnom[:,:,tt] = fu.T

########################
# Cost
########################
QQ = 1*np.diag([20.0, 5.0, 5.0, 5.0, 0.1, 0.1, 0.1, 0.1 ]) 
RR = 0.1*np.eye(ni)  #create matrix of RR
QQf = 1*QQ

#############################
# Model Predictive Control
#############################

#bounds on inputs not tight
umax = 5
umin = -umax

x1max = 10; 
x1min = -x1max; 

x2max = 10   #bounds on state
x2min = -x2max

######################################################
#Array to store the states and inputs of real dynamics
######################################################
xx_real_mpc = np.zeros((ns,Tsim))  
uu_real_mpc = np.zeros((ni,Tsim))  

############################################################################
#Array to store the states and inputs of optimal of mpc starting from time t
############################################################################
xx_mpc = np.zeros((ns, T_pred, Tsim))
uu_mpc = np.zeros((ni, T_pred, Tsim))

xx_real_mpc[:,0] = xx0  #Initilize the xx_real_mpc to xinit

for tt in range(Tsim-1):
    # System evolution - real with MPC
    xx_t_mpc = xx_real_mpc[:,tt] # get initial condition
    
    xopt_temp= xx_opt[:,tt:tt+T_pred]
    uopt_temp= uu_opt[:,tt:tt+T_pred]
    
    AAA=AAnom[:,:,tt:tt+T_pred]
    BBB=BBnom[:,:,tt:tt+T_pred]

    # Solve MPC problem and apply first input to the system
    if tt%10 == 0: # print every 5 time instants
      print('MPC:\t t = {}'.format(tt))

    uu_real_mpc[:,tt], xx_mpc[:,:,tt] = linear_mpc(AAA, BBB, xopt_temp, uopt_temp, QQ, RR, QQf, xx_t_mpc, umax=umax, umin=umin, x1_max=x1max, x1_min=x1min, x2_min = x2min, x2_max = x2max, T_pred = T_pred)[:2]
    xx_real_mpc[:,tt+1] = dyn.dynamics(xx_real_mpc[:,tt], uu_real_mpc[:,tt])[0]

#######################################
# Plots
#######################################

time = np.arange(Tsim)
fig, axs = plt.subplots(ns, 1, sharex='all')

axs[0].plot(time, xx_real_mpc[0,:], linewidth=2)
axs[0].plot(time, xx_opt[0,:Tsim],'--r', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

if 1 or np.amax(xx_opt[0,:]) > 100: # set lims only if neededs
  axs[0].set_ylim([0,5])

axs[0].set_xlim([-1,Tsim])
axs[0].legend(['MPC', 'REF.'])

axs[1].plot(time, xx_real_mpc[1,:], linewidth=2)
axs[1].plot(time, xx_opt[1,:Tsim], '--r', linewidth=2)

if x2max < 1.1*np.amax(xx_real_mpc[1,:]): # draw constraints only if active
  axs[1].plot(time, np.ones(Tsim)*x2max, '--g', linewidth=1)
  axs[1].plot(time, np.ones(Tsim)*x2min, '--g', linewidth=1)

axs[1].grid()
axs[1].set_ylabel('$x_2$')

if 1 or np.amax(xx_opt[1,:]) > 100: # set lims only if neededs
  axs[1].set_ylim([0,5])

axs[1].set_xlim([-1,Tsim])
axs[1].legend(['MPC', 'REF.'])

axs[2].plot(time, xx_real_mpc[2,:], linewidth=2)
axs[2].plot(time, xx_opt[2,:Tsim], '--r', linewidth=2)

# if x2max < 1.1*np.amax(xx_real_mpc[1,:]): # draw constraints only if active
#   axs[1].plot(time, np.ones(Tsim)*x2max, '--g', linewidth=1)
#   axs[1].plot(time, np.ones(Tsim)*x2min, '--g', linewidth=1)

axs[2].grid()
axs[2].set_ylabel('$x_3$')

# if 1 or np.amax(xx_real_opt[2,:]) > 100: # set lims only if neededs
#   axs[2].set_ylim([-10,10])

axs[2].set_xlim([-1,Tsim])
axs[2].legend(['MPC', 'REF.'])

axs[3].plot(time, xx_real_mpc[3,:], linewidth=2)
axs[3].plot(time, xx_opt[3,:Tsim], '--r', linewidth=2)

# if x2max < 1.1*np.amax(xx_real_mpc[1,:]): # draw constraints only if active
#   axs[1].plot(time, np.ones(Tsim)*x2max, '--g', linewidth=1)
#   axs[1].plot(time, np.ones(Tsim)*x2min, '--g', linewidth=1)

axs[3].grid()
axs[3].set_ylabel('$x_4$')

# if 1 or np.amax(xx_real_opt[3,:]) > 100: # set lims only if neededs
#   axs[3].set_ylim([-10,10])

axs[3].set_xlim([-1,Tsim])
axs[3].legend(['MPC', 'REF.'])

axs[4].plot(time, xx_real_mpc[4,:], linewidth=2)
axs[4].plot(time, xx_opt[4,:Tsim], '--r', linewidth=2)

# if x2max < 1.1*np.amax(xx_real_mpc[1,:]): # draw constraints only if active
#   axs[1].plot(time, np.ones(Tsim)*x2max, '--g', linewidth=1)
#   axs[1].plot(time, np.ones(Tsim)*x2min, '--g', linewidth=1)

axs[4].grid()
axs[4].set_ylabel('$x_5$')

# if 1 or np.amax(xx_real_opt[4,:]) > 100: # set lims only if neededs
#   axs[4].set_ylim([-10,10])

axs[4].set_xlim([-1,Tsim])
axs[4].legend(['MPC', 'REF.'])

axs[5].plot(time, xx_real_mpc[5,:], linewidth=2)
axs[5].plot(time, xx_opt[5,:Tsim], '--r', linewidth=2)

# if x2max < 1.1*np.amax(xx_real_mpc[1,:]): # draw constraints only if active
#   axs[1].plot(time, np.ones(Tsim)*x2max, '--g', linewidth=1)
#   axs[1].plot(time, np.ones(Tsim)*x2min, '--g', linewidth=1)

axs[5].grid()
axs[5].set_ylabel('$x_6$')

# if 1 or np.amax(xx_real_opt[5,:]) > 100: # set lims only if neededs
#   axs[5].set_ylim([-10,10])

axs[5].set_xlim([-1,Tsim])
axs[5].legend(['MPC', 'REF.'])

axs[6].plot(time, xx_real_mpc[6,:], linewidth=2)
axs[6].plot(time, xx_opt[6,:Tsim], '--r', linewidth=2)

# if x2max < 1.1*np.amax(xx_real_mpc[1,:]): # draw constraints only if active
#   axs[1].plot(time, np.ones(Tsim)*x2max, '--g', linewidth=1)
#   axs[1].plot(time, np.ones(Tsim)*x2min, '--g', linewidth=1)

axs[6].grid()
axs[6].set_ylabel('$x_7$')

# if 1 or np.amax(xx_real_opt[6,:]) > 100: # set lims only if neededs
#   axs[6].set_ylim([-10,10])

axs[6].set_xlim([-1,Tsim])
axs[6].legend(['MPC', 'REF.'])

axs[7].plot(time, xx_real_mpc[7,:], linewidth=2)
axs[7].plot(time, xx_opt[7,:Tsim], '--r', linewidth=2)

# if x2max < 1.1*np.amax(xx_real_mpc[1,:]): # draw constraints only if active
#   axs[1].plot(time, np.ones(Tsim)*x2max, '--g', linewidth=1)
#   axs[1].plot(time, np.ones(Tsim)*x2min, '--g', linewidth=1)

axs[7].grid()
axs[7].set_ylabel('$x_8$')

# if 1 or np.amax(xx_real_opt[7,:]) > 100: # set lims only if neededs
#   axs[7].set_ylim([-10,10])

axs[7].set_xlim([-1,Tsim])
axs[7].legend(['MPC', 'REF.'])

fig.align_ylabels(axs)
fig, input = plt.subplots(ni, 1, sharex='all')

input[0].plot(time, uu_real_mpc[0,:],'g', linewidth=2)
input[0].plot(time, uu_opt[0,:Tsim],'--r', linewidth=2)

# if umax < 1.1*np.amax(uu_real_mpc[0,:]): # draw constraints only if active
#   input[0].plot(time, np.ones(Tsim)*umax, '--g', linewidth=1)
#   input[0].plot(time, np.ones(Tsim)*umin, '--g', linewidth=1)

input[0].grid()
input[0].set_ylabel('$u_1$')
input[0].set_xlabel('time')

# if 1 or np.amax(xx_real_opt[0,:]) > 100: # set lims only if neededs
#   input[0].set_ylim([-10,10])

input[0].set_xlim([-1,Tsim])
input[0].legend(['MPC', 'REF.'])

##########################################################################################
input[1].plot(time, uu_real_mpc[1,:],'g', linewidth=2)
input[1].plot(time, uu_opt[1,:Tsim],'--r', linewidth=2)

if umax < 1.1*np.amax(uu_real_mpc[1,:]): # draw constraints only if active
  input[1].plot(time, np.ones(Tsim)*umax, '--g', linewidth=1)
  input[1].plot(time, np.ones(Tsim)*umin, '--g', linewidth=1)

input[1].grid()
input[1].set_ylabel('$u_2$')
input[1].set_xlabel('time')

# if 1 or np.amax(xx_real_opt[1,:]) > 100: # set lims only if neededs
#   input[1].set_ylim([-10,10])

input[1].set_xlim([-1,Tsim])
input[1].legend(['MPC', 'REF.'])

fig.align_ylabels(input)
plt.show()
##########################################################################################
