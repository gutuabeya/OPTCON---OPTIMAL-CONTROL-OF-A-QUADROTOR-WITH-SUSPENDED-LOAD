
import numpy as np
import cvxpy as cp
from solver_ltv_LQR import ltv_LQR

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import dynamics as dyn
from referance_curve import tf
from referance_curve import bell_create
from step_referance import step_create

import control as ctrl
from solver import unconstrained_lqr
dt = dyn.dt
TT = int(tf/dt)
mm = dyn.mm 
MM = dyn.MM 
gg = dyn.gg 
ns = dyn.ns
ni = dyn.ni

# Load the array from the saved file
loaded_data = np.load('xx_opt.npz')
xx_opt = loaded_data['result_array']

loaded_data = np.load('uu_opt.npz')
uu_opt = loaded_data['result_array']

xx_ref = xx_opt; 
uu_ref = uu_opt; 

Tsim = TT   # simulation horizon

xx_lqr = np.zeros((ns,Tsim))
uu_lqr = np.zeros((ni,Tsim))
#xx0 = np.zeros((ns,1))
xx0 = np.array([[1.0],[0.5],[0],[0],[0.2],[0.3],[0.1],[0.2]]) #considered perturbed initial condition
xx0=xx0.squeeze()

AAnom= np.zeros((ns,ns,TT))
BBnom= np.zeros((ns,ni,TT))

########################
# Linear Dynamics - get nominal A,B matrices
########################
for tt in range(TT-1): 
  fx, fu = dyn.dynamics(xx_opt[:,tt],uu_opt[:,tt])[1:]
  AAnom[:,:,tt] = fx.T  # Matrix A around nominal trajectory xx-optimal
  BBnom[:,:,tt] = fu.T  # Matrix B around nominal trajectory uu-optimal

########################
# Cost Matrix for tracking
########################
QQ = 1*np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0 ]) #create matrix of QQ
RR = 1*np.eye(ni)  #create matrix of RR
QQf = 10*QQ
SS = np.zeros((ni, ns, TT))

###################################
# No Affine terms is used (for tracking)
###################################
qq = np.zeros((ns, Tsim))
rr = np.zeros((ni, Tsim))
qqf = np.zeros((ns))

####################################
## LQR Tracking
####################################

KK = ltv_LQR(AAnom,BBnom,QQ,RR, SS, QQf, TT, xx0,qq, rr, qqf)[0]

xx_lqr[:,0] = xx0
for tt in range(Tsim-1):

  if tt%10 == 0: # print every 10 time instants
    print('LQR:\t t = {}'.format(tt))

  uu_lqr[:,tt]   = uu_opt[:,tt] + KK[:,:,tt]@(xx_lqr[:,tt]- xx_opt[:,tt])
  xx_lqr[:,tt+1] = dyn.dynamics(xx_lqr[:,tt], uu_lqr[:,tt])[0]

#######################################
# Plots
#######################################

time = np.arange(Tsim)
fig, axs = plt.subplots(ns, 1, sharex='all')

axs[0].plot(time, xx_ref[0,:], linewidth=2)
axs[0].plot(time, xx_lqr[0,:],'--r', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

if 1 or np.amax(xx_lqr[0,:]) > 100: # set lims only if neededs
  axs[0].set_ylim([0,5])

axs[0].set_xlim([-1,Tsim])
axs[0].legend(['REF.', 'LQR'])

axs[1].plot(time, xx_ref[1,:], linewidth=2)
axs[1].plot(time, xx_lqr[1,:], '--r', linewidth=2)

axs[1].grid()
axs[1].set_ylabel('$x_2$')

if 1 or np.amax(xx_lqr[0,:]) > 100: # set lims only if neededs
  axs[1].set_ylim([0,5])

axs[1].set_xlim([-1,Tsim])
axs[1].legend(['REF.', 'LQR'])

axs[2].plot(time, xx_ref[2,:], linewidth=2)
axs[2].plot(time, xx_lqr[2,:], '--r', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$x_3$')
axs[2].set_xlim([-1,Tsim])
axs[2].legend(['REF.', 'LQR'])

axs[3].plot(time, xx_ref[3,:], linewidth=2)
axs[3].plot(time, xx_lqr[3,:], '--r', linewidth=2)
axs[3].grid()
axs[3].set_ylabel('$x_4$')
axs[3].set_xlim([-1,Tsim])
axs[3].legend(['REF.', 'LQR'])

axs[4].plot(time, xx_ref[4,:], linewidth=2)
axs[4].plot(time, xx_lqr[4,:], '--r', linewidth=2)
axs[4].grid()
axs[4].set_ylabel('$x_5$')
axs[4].set_xlim([-1,Tsim])
axs[4].legend(['REF.', 'LQR'])

axs[5].plot(time, xx_ref[5,:], linewidth=2)
axs[5].plot(time, xx_lqr[5,:], '--r', linewidth=2)
axs[5].grid()
axs[5].set_ylabel('$x_6$')
axs[5].set_xlim([-1,Tsim])
axs[5].legend(['REF.', 'LQR'])

axs[6].plot(time, xx_ref[6,:], linewidth=2)
axs[6].plot(time, xx_lqr[6,:], '--r', linewidth=2)
axs[6].grid()
axs[6].set_ylabel('$x_7$')
axs[6].set_xlim([-1,Tsim])
axs[6].legend(['REF.', 'LQR'])

axs[7].plot(time, xx_ref[7,:], linewidth=2)
axs[7].plot(time, xx_lqr[7,:], '--r', linewidth=2)
axs[7].grid()
axs[7].set_ylabel('$x_8$')
axs[7].set_xlim([-1,Tsim])
axs[7].legend(['REF.', 'LQR'])

fig.align_ylabels(axs)
fig, input = plt.subplots(ni, 1, sharex='all')

input[0].plot(time, uu_ref[0,:],'g', linewidth=2)
input[0].plot(time, uu_lqr[0,:],'--r', linewidth=2)
input[0].grid()
input[0].set_ylabel('$u_1$')
input[0].set_xlabel('time')
input[0].set_xlim([-1,Tsim])
input[0].legend(['REF.', 'LQR'])

input[1].plot(time, uu_ref[1,:],'g', linewidth=2)
input[1].plot(time, uu_lqr[1,:],'--r', linewidth=2)
input[1].grid()
input[1].set_ylabel('$u_2$')
input[1].set_xlabel('time')
input[1].set_xlim([-1,Tsim])
input[1].legend(['REF.', 'LQR'])

fig.align_ylabels(input)
plt.show()

# Save the array to a file (use .npz format)
np.savez('xx_lqr.npz', result_array=xx_lqr)
np.savez('uu_lqr.npz', result_array=uu_lqr)
