# Model Predictive Control for Linear Systems
# Solvers
# OPTCON 2022
# Lorenzo Sforni
# 20 Dec 2022
#

import numpy as np
import cvxpy as cp

#case of unconstarined i.e there is no constarint in xx and uu.
def unconstrained_lqr(AA, BB, xx_ref, uu_ref, QQ, RR, QQf, xx0, T_hor = 100):  
    """
        LQR - given init condition and time horizon, optimal state-input trajectory

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xx0: initial condition
          - T_hor: time horizon
    """

    xx0 = xx0.squeeze()
    #xxf = xxf.squeeze()

    ns=8; ni =2

    xx_lqr = cp.Variable((ns, T_hor))
    uu_lqr = cp.Variable((ni, T_hor))

    cost = 0
    constr = []

    for tt in range(T_hor-1):
        cost += cp.quad_form(xx_lqr[:,tt] - xx_ref[:,tt], QQ) + cp.quad_form(uu_lqr[:,tt] - uu_ref[:,tt], RR)
        constr += [xx_lqr[:,tt+1] == AA[:,:,tt]@xx_lqr[:,tt] + BB[:,:,tt]@uu_lqr[:,tt]]
    
    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_lqr[:,T_hor-1] - xx_ref[:,T_hor-1], QQf)
    constr += [xx_lqr[:,0] == xx0]

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! ")

    return xx_lqr.value, uu_lqr.value

#2. case of constarined i.e there is constarint in xx or uu.
def linear_mpc(AA, BB, xx_ref, uu_ref, QQ, RR, QQf, xxt, umax = 1, umin = -1, x1_max = 20, x1_min = -20, x2_max = 20, x2_min = -20,  T_pred = 20):
    """
        Linear MPC solver - Constrained LQR

        Given a measured state xxt measured at t
        gives back the optimal input to be applied at t

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xxt: initial condition (at time t)
          - T: time (prediction) horizon

        Returns
          - u_t: input to be applied at t
          - xx, uu predicted trajectory

    """

    xxt = xxt.squeeze()
    #xxf = xxf.squeeze()

    ns = 8; 
    ni = 2; 

    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))

    cost = 0
    constr = []

    for tt in range(T_pred-1):
        cost += cp.quad_form((xx_mpc[:,tt] - xx_ref[:,tt] ), QQ) + cp.quad_form((uu_mpc[:,tt] - uu_ref[:,tt]), RR)
        constr += [xx_mpc[:,tt+1] == AA[:,:,tt]@xx_mpc[:,tt] + BB[:,:,tt]@uu_mpc[:,tt], # dynamics constrauu_mpc[0,tt] <= umax, # inputs constraints, i.e bounds on inputs
                # uu_mpc[0,tt] <= umax, 
                # uu_mpc[0,tt] >= umin, 
                # uu_mpc[1,tt] <= umax, 
                # uu_mpc[1,tt] >= umin,  
                # xx_mpc[0,tt] <= x1_max, #boundas on x1 state  
                # xx_mpc[0,tt] >= x1_min,
                # xx_mpc[1,tt] <= x2_max, #boundas on x2 state 
                # xx_mpc[1,tt] >= x2_min,
                ]
    
    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form((xx_mpc[:,T_pred-1] - xx_ref[:,T_pred-1]), QQf)
    constr += [xx_mpc[:,0] == xxt] #initial condition constarint

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()
    # problem.solve(verbose=True)
    # problem.solve(solver=cp.ECOS, verbose=True)


    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value