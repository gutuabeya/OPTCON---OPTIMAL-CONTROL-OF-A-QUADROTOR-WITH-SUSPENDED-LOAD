import numpy as np
ns = 8
ni = 2
dt = 1e-2 # discretization step

#Constant parameters
MM=0.028
mm=0.04
JJ=0.001
gg=9.81
LL=0.2
ll=0.05

MM_plus_mm=1/(MM + mm)
M_plus_m_times_mm_times_LL=(mm * LL) * MM_plus_mm; 
M_plus_m_times_mm_over_MM=(mm/MM)*MM_plus_mm 
MM_times_LL=(MM*LL); 
LL_over_JJ=(ll/JJ); 


def dynamics(xx,uu):
 
  xx = xx[:,None]
  uu = uu[:,None]

  #xp, yp, alpha, theta, vx, vy, omega_alpha, omega_theta = xx
  #Fs, Fd = u

  xp = xx[0,0]
  yp = xx[1,0]
  alpha = xx[2,0]
  theta = xx[3,0]
  vx = xx[4,0]
  vy = xx[5,0]
  omega_alpha = xx[6,0]
  omega_theta =xx[7,0]

  Fs = uu[0,0]
  Fd = uu[1,0]

  xx_plus = np.zeros((ns, 1))

  omega_alpha_2=omega_alpha**2
  sin_alpha=np.sin(alpha)
  sin_theta=np.sin(theta); 
  sin_alpha_minus_theta=np.sin(alpha - theta)
  cos_alpha=np.cos(alpha)
  cos_theta=np.cos(theta)
  cos_alpha_minus_theta=np.cos(alpha - theta)

  xx_plus[0]=xp + dt*vx; #xp
  xx_plus[1]=yp + dt*vy; #yp
  xx_plus[2]=alpha + dt*omega_alpha; #alpha
  xx_plus[3]=theta + dt*omega_theta; #theta
  xx_plus[4]=vx + dt * (M_plus_m_times_mm_times_LL * omega_alpha_2 * sin_alpha - MM_plus_mm*Fs*sin_theta + M_plus_m_times_mm_over_MM*Fs*sin_alpha_minus_theta*cos_alpha)
  xx_plus[5]=vy + dt * (-M_plus_m_times_mm_times_LL*omega_alpha_2*cos_alpha + MM_plus_mm*Fs*cos_theta + M_plus_m_times_mm_over_MM*Fs*sin_alpha_minus_theta*sin_alpha -gg)
  xx_plus[6]=omega_alpha + dt * ((-Fs* sin_alpha_minus_theta)/MM_times_LL)
  xx_plus[7]=omega_theta + dt *LL_over_JJ*Fd

  fx = np.zeros((ns, ns))
  fu = np.zeros((ni, ns))

  #Gradient 
    #df1dx

  fx[0,0] = 1
  fx[1,0] = 0
  fx[2,0] = 0
  fx[3,0] = 0
  fx[4,0] = dt
  fx[5,0] = 0
  fx[6,0] = 0
  fx[7,0] = 0

    #df2dx

  fx[0,1] = 0
  fx[1,1] = 1
  fx[2,1] = 0
  fx[3,1] = 0
  fx[4,1] = 0
  fx[5,1] = dt
  fx[6,1] = 0
  fx[7,1] = 0

  #df3dx

  fx[0,2] = 0
  fx[1,2] = 0
  fx[2,2] = 1
  fx[3,2] = 0
  fx[4,2] = 0
  fx[5,2] = 0
  fx[6,2] = dt
  fx[7,2] = 0

  #df4dx

  fx[0,3] = 0
  fx[1,3] = 0
  fx[2,3] = 0
  fx[3,3] = 1
  fx[4,3] = 0
  fx[5,3] = 0
  fx[6,3] = 0
  fx[7,3] = dt  

#df5dx

  fx[0,4] = 0
  fx[1,4] = 0
  fx[2,4] = dt*M_plus_m_times_mm_times_LL*omega_alpha_2*cos_alpha + dt*M_plus_m_times_mm_over_MM*Fs*(cos_alpha*cos_alpha_minus_theta - sin_alpha*sin_alpha_minus_theta); #ok
  fx[3,4] = -dt*Fs*(M_plus_m_times_mm_over_MM*cos_alpha*cos_alpha_minus_theta + MM_plus_mm*cos_theta);#ok
  fx[4,4] = 1
  fx[5,4] = 0
  fx[6,4] = 2*dt*M_plus_m_times_mm_times_LL*omega_alpha*sin_alpha;#ok
  fx[7,4] = 0

#df6dx

  fx[0,5] = 0
  fx[1,5] = 0
  fx[2,5] = dt*M_plus_m_times_mm_times_LL*omega_alpha_2*sin_alpha+ dt*M_plus_m_times_mm_over_MM*Fs*(cos_alpha_minus_theta*sin_alpha + cos_alpha*sin_alpha_minus_theta); #ok
  fx[3,5] = -dt*Fs*(M_plus_m_times_mm_over_MM*cos_alpha_minus_theta*sin_alpha + MM_plus_mm*sin_theta); #ok
  fx[4,5] = 0
  fx[5,5] = 1
  fx[6,5] =-2*dt*M_plus_m_times_mm_times_LL*omega_alpha*cos_alpha; #0k 
  fx[7,5] = 0

  #df7dx

  fx[0,6] = 0
  fx[1,6] = 0
  fx[2,6] = -(dt/MM_times_LL)*Fs*cos_alpha_minus_theta; #ok
  fx[3,6] =  (dt/MM_times_LL)*Fs*cos_alpha_minus_theta; #ok
  fx[4,6] = 0
  fx[5,6] = 0
  fx[6,6] = 1
  fx[7,6] = 0

 #df8dx

  fx[0,7] = 0
  fx[1,7] = 0
  fx[2,7] = 0
  fx[3,7] = 0
  fx[4,7] = 0
  fx[5,7] = 0
  fx[6,7] = 0
  fx[7,7] = 1  

#df1du

  fu[0,0] = 0
  fu[1,0] = 0

#df2du

  fu[0,1] = 0
  fu[1,1] = 0

#df3du

  fu[0,2] = 0
  fu[1,2] = 0

#df4du

  fu[0,3] = 0
  fu[1,3] = 0

#df5du

  fu[0,4] = -dt*MM_plus_mm*sin_theta + dt*M_plus_m_times_mm_over_MM*sin_alpha_minus_theta*cos_alpha; #ok
  fu[1,4] = 0

#df6du
  fu[0,5] = dt*M_plus_m_times_mm_over_MM*sin_alpha_minus_theta*sin_alpha + dt*MM_plus_mm*cos_theta;  
  fu[1,5] = 0

 #df7du
  fu[0,6] = -(dt/MM_times_LL)*sin_alpha_minus_theta;
  fu[1,6] = 0

 #df8du

  fu[0,7] = 0
  fu[1,7] = dt*LL_over_JJ; 
  
  xx_plus = xx_plus.squeeze()
  return xx_plus, fx, fu



