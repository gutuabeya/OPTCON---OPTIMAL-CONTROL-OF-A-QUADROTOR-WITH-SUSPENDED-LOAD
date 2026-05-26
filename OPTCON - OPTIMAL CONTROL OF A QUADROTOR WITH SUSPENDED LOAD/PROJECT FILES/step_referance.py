import numpy as np
import dynamics as dyn
# import matplotlib.pyplot as plt
from referance_curve import tf

ns = dyn.ns
ni = dyn.ni
dt = dyn.dt
TT = int(tf/dt)

def step_create(xx_des, uu_des):
  
  xx_ref = np.zeros((ns, TT))
  uu_ref = np.zeros((ni, TT))

  for ii in range(ns):
    x0=(xx_des[ii][0])*np.ones((1,)) 
    xf=(xx_des[ii][1])*np.ones((1,)) 
    
    for tt in range(TT):
        if(tt<int(TT/2)):
            xx_ref[ii,tt] = x0
        else:
            xx_ref[ii,tt] = xf

  for jj in range(ni):
    u0=(uu_des[jj][0])*np.ones((1,)) 
    uf=(uu_des[jj][1])*np.ones((1,)) 
        
    for tt in range(TT):
        if(tt<int(TT/2)):
            uu_ref[jj,tt] = u0
        else:
            uu_ref[jj,tt] = uf

    
  return xx_ref, uu_ref