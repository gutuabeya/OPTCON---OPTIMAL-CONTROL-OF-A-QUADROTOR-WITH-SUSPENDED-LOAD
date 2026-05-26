import numpy as np
import dynamics as dyn
import cost as cst 
import matplotlib.pyplot as plt

def armijo_step_size_selection(armijo_param):
  ##################################
  # Stepsize selection - ARMIJO
  ##################################
  stepsize_0,beta,cc, armijo_maxiters,visu_armijo, TT, ns, ni,kk, x0, uu, deltau, xx_ref, uu_ref, JJ,descent_arm=armijo_param
  
  stepsizes = []  # list of stepsizes
  costs_armijo = []

  stepsize = stepsize_0

  for ii in range(armijo_maxiters):

    # temp solution update
    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))
    
    xx_temp[:,0] = x0

    for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt]
      xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

    # temp cost calculation
    JJ_temp = 0

    for tt in range(TT-1):
      temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
      JJ_temp += temp_cost

    temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
    JJ_temp += temp_cost

    stepsizes.append(stepsize)      # save the stepsize
    costs_armijo.append(np.min([JJ_temp, 100*JJ[kk]]))    # save the cost associated to the stepsize

    if JJ_temp > JJ[kk]  + cc*stepsize*descent_arm[kk]:
        stepsize = beta*stepsize # update the stepsize
    
    else:
        print('Armijo stepsize = {}'.format(stepsize))
        break
  # plt.plot(xx_temp[0,:])
  # plt.show()
    
  ############################
  # ARMIJO plot
  ############################

  if visu_armijo:# and kk==54: #and kk==56: 

    steps = np.linspace(0,stepsize_0,int(1e1))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):

      step = steps[ii]

      # temp solution update
      xx_temp = np.zeros((ns,TT))
      uu_temp = np.zeros((ni,TT))

      xx_temp[:,0] = x0

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt,kk] + step*deltau[:,tt]
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

      # temp cost calculation
      JJ_temp = 0

      for tt in range(TT-1):
        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ_temp += temp_cost

      temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
      JJ_temp += temp_cost

      costs[ii] = np.min([JJ_temp, 100*JJ[kk]])


    plt.figure(1)
    plt.clf()

    plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
    plt.plot(steps, JJ[kk] + descent_arm[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(steps, JJ[kk] + cc*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

    plt.grid()
    plt.xlabel('stepsize')
    plt.legend()
    plt.draw()

    plt.show()

  return stepsize