from solver_ltv_LQR import ltv_LQR
from armijo_step_selection import armijo_step_size_selection
import numpy as np
import matplotlib.pyplot as plt
import dynamics as dyn
import cost as cst  # import cost functions
import signal  # Allow Ctrl-C to work despite plotting
signal.signal(signal.SIGINT, signal.SIG_DFL)

def newton_method(xx_ref, uu_ref, algoparam, trajparam):
  #######################################
  #Algorithm parameters
  #######################################
  stepsize_0, cc, beta, armijo_maxiters,max_iters,term_cond,visu_armijo =algoparam
  
  #######################################
  # Trajectory parameters
  #######################################
  tf, dt, ns, ni, TT = trajparam
  
  #######################################
  # INITIALIZATION of the Algorithm 
  #######################################

  xx_init = np.zeros((ns,TT))
  x0 = xx_ref[:,0]
  u0 = uu_ref[:,0]
  uu_init = uu_ref
  xx_init[:,0] = x0 #inputing the initial value to xx_init
  for tt in range(TT-1):
    xx_init[:,tt+1] = dyn.dynamics(xx_init[:,tt], uu_init[:,tt])[0] # from initial value we create an matrix of every state with reference input
 
  ######################################
  # Create and Initilize 3D Arrays to store data
  ######################################
  
  xx = np.zeros((ns, TT, max_iters))   # state seq.
  uu = np.zeros((ni, TT, max_iters))   # input seq.
  
  xx[:,:,0] = xx_init[:,:]
  uu[:,:,0] = uu_init[:,:]
  
  lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.
  dJ = np.zeros((ni,TT, max_iters))  
  JJ = np.zeros(max_iters)      # collect cost
  descent = np.zeros(max_iters) # collect norm of descent direction
  descent_arm = np.zeros(max_iters) # collect descent direction at each iteration

  #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  #             ALGORITHM  (NEWTON'S)                   *
  #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  
  print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
  print("*             ALGORITHM  (NEWTON METHOD)              *")
  print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
  
  kk = 0
  for kk in range(max_iters-1):

    ##################################
    #Define Arrays
    ##################################
    AA = np.zeros((ns, ns, TT))
    BB = np.zeros((ns, ni, TT))
    QQ = np.zeros((ns, ns, TT))
    RR = np.zeros((ni, ni, TT))
    SS = np.zeros((ni, ns, TT))
    qq = np.zeros((ns, TT))
    rr = np.zeros((ni, TT))
    qqf = np.zeros((ns))
    QQ_f = np.zeros((ns, ns))

    ###########################################
    # calculate cost at xx_k and uu_k iteration
    ###########################################
    JJ[kk] = 0
    for tt in range(TT-1):
      temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
      JJ[kk] += temp_cost
    
    temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
    JJ[kk] += temp_cost
    
    ##################################
    # compute lamda_T, Q_T, q_T 
    ##################################
    lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1] #
    lmbd[:,TT-1,kk] = lmbd_temp.squeeze() #

    QQ_f=cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[2]  #lxx= Q_T
    qqf = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1].squeeze()  #lx= q_T

    ##################################
    # compute qq_t, rr_t, AA_t, BB_t
    ##################################  
    for tt in range(TT-1):
      aa, bb, QQ[:,:,tt], RR[:,:,tt], SS[:,:,tt] = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:]
      fx, fu = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:] 
  
      qq[:,tt] = aa.squeeze()
      rr[:,tt] = bb.squeeze()
      
      AA[:,:,tt] = fx.T
      BB[:,:,tt] = fu.T 

  ##############################################################################
  #Compute KK, SIGMA using Riccati equation
  ##############################################################################
    x_0 = np.zeros((ns))
    KK, sigma, deltax, deltau = ltv_LQR(AA,BB,QQ,RR,SS,QQ_f, TT, x_0, qq, rr, qqf)

  ##############################################################################
  # compute lamd and gradient of J, and Descent direction in backward
  ##############################################################################
    for tt in reversed(range(TT-1)):  # integration backward in time
      lmbd_temp = AA[:,:,tt].T@lmbd[:,tt+1,kk] + qq[:,tt]      # costate equation
      dJ_temp =  BB[:,:,tt].T@lmbd[:,tt+1,kk] + rr[:,tt]

      lmbd[:,tt,kk] = lmbd_temp.squeeze()
      dJ[:,tt,kk] = dJ_temp.squeeze()

      # Descent direction calculation
      descent[kk] += deltau[:,tt].T@deltau[:,tt]
      descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt]

    ##################################
    # Stepsize selection - ARMIJO
    ##################################
    armijo_param=[stepsize_0,beta,cc, armijo_maxiters,visu_armijo, TT, ns, ni,kk, x0, uu, deltau, xx_ref, uu_ref, JJ,descent_arm]
    stepsize=armijo_step_size_selection(armijo_param)
    
    ##############################################################################
    #Update the current solution  XXt and UUt
    ##############################################################################  
  
    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))
    
    xx_temp[:,0] = x0

    for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt] - xx[:,tt,kk]) + stepsize*sigma[:,tt]
      xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

    xx[:,:,kk+1] = xx_temp
    uu[:,:,kk+1] = uu_temp
  
  ##############################################################################
  #Termination condition
  ##############################################################################    
    print('Iter = {}\t Descent = {}\t Cost = {}'.format(kk, descent[kk], JJ[kk]))
  
    if descent[kk] <= term_cond:
      max_iters = kk
      break
  
  xx_star = xx[:,:,max_iters-1]
  uu_star = uu[:,:,max_iters-1]
  uu_star[:,-1] = uu_star[:,-2] # for plot
    
  return xx_star, uu_star,JJ, descent