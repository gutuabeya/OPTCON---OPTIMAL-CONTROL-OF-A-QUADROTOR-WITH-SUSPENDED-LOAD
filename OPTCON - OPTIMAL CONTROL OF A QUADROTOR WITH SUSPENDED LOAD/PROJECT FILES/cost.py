# Cost functions
import numpy as np
import dynamics as dyn

ns = dyn.ns  #dimension of state
ni = dyn.ni

#User defined cost for trajectory generate
QQt = 0.1*np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01 ]) #create matrix of QQt
RRt = 0.001*np.eye(ni)  #create matrix of RRt
QQT = QQt 
SSt = np.zeros((ni, ns)) 
 
def stagecost(xx,uu, xx_ref, uu_ref):

  xx = xx[:,None]
  uu = uu[:,None]
  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)
  lx = QQt@(xx - xx_ref)   #diff. of ll wrt x = lx = nabula_x,l =at
  lu = RRt@(uu - uu_ref)   #diff. of ll wrt u = lu = nabula_u,l =bt
  lxx = QQt
  luu = RRt
  lux = SSt 
  return ll, lx, lu, lxx, luu, lux

def termcost(xx,xx_ref):

  xx = xx[:,None]
  xx_ref = xx_ref[:,None]

  llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)
  lTx = QQT@(xx - xx_ref)  # diff. of llT wrt x =nabula_x = l terminal
  lTxx=QQT; 
  return llT.squeeze(), lTx,lTxx