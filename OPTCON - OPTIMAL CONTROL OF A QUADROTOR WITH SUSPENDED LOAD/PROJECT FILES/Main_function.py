import numpy as np
import dynamics as dyn
from trajectory_optimazation import newton_method
from referance_curve import bell_create
from step_referance import step_create

from referance_curve import tf
from plot_trajectory import plot_traj
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

#######################################
# Dynamics parameters
#######################################
dt = dyn.dt #1e-3 #get discretization step from dynamics
ns = dyn.ns #8    #get dimension of state from dynamics
ni = dyn.ni #2    #get dimension of input from dynamics
mm = dyn.mm #0.04
MM = dyn.MM #0.028
gg = dyn.gg #9.81

########################
#Desired curve parameters
#######################
xx_des=[[0,5.0],[0,5.0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] #state
des_input=(mm+MM)*gg #inputs necessary input to hold the drone at each points
uu_des=[[des_input,des_input],[0.0,0.0]]
xx_ref = bell_create(xx_des, ns)
uu_ref = bell_create(uu_des, ni)
# xx_ref, uu_ref = step_create(xx_des, uu_des)

#######################################
# ARMIJO and Algorithm parameters
#######################################
stepsize_0 = 0.7
cc = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations
term_cond = 1e-6
max_iters = 100
visu_armijo = False
# visu_armijo = True
algoparam=[stepsize_0, cc, beta, armijo_maxiters,max_iters,term_cond,visu_armijo ]

#######################################
# Trajectory parameters
#######################################
TT = int(tf/dt) # discrete-time samples
trajparam=[tf, dt, ns, ni, TT]

######################################
# Generate optimal trajectory using Newthon method
######################################
xx_opt, uu_opt,JJ,descent = newton_method(xx_ref, uu_ref, algoparam, trajparam)

#############################
# Plots
#############################
plotparam=[ns,ni,tf,TT,JJ,descent,max_iters,xx_opt,xx_ref,uu_opt,uu_ref]
plot_traj(plotparam)

#####################################
# Save the xx_opt. and uu_opt as well the referance
#####################################
np.savez('xx_opt.npz', result_array=xx_opt)
np.savez('xx_ref.npz', result_array=xx_ref)

np.savez('uu_opt.npz', result_array=uu_opt)
np.savez('uu_ref.npz', result_array=uu_ref)
