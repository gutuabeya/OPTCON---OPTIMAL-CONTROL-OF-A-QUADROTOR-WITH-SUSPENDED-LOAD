import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

  ############################
  # Plots
  ############################

  # cost and descent
def plot_traj(plotparam):
  ns,ni,tf,TT,JJ,descent,max_iters,xx_star,xx_ref,uu_star,uu_ref=plotparam

  plt.figure('descent direction')
  plt.plot(np.arange(max_iters), descent[:max_iters])
  plt.xlabel('$k$')
  plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
  plt.yscale('log')
  plt.grid()
  plt.show(block=False)
  plt.figure('cost')
  plt.plot(np.arange(max_iters), JJ[:max_iters])
  plt.xlabel('$k$')
  plt.ylabel('$J(\\mathbf{u}^k)$')
  plt.yscale('log')
  plt.grid()
  plt.show(block=False)

  # Optimal trajectory - States
  tt_hor = np.linspace(0, tf, TT)
  fig, axs_states = plt.subplots(ns, 1, sharex='all', figsize=(10, 12))
  state_labels = ['$xp$', '$yp$', '$alp$', '$the$', '$Vx$', '$Vy$', '$om_alp$', '$om_the$']
  control_labels = ['$Fs$', '$Fd$']
  # Define colors for xx_star (states) and uu_star (controls)
  state_colors = ['g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']
  control_colors = ['r', 'brown']

  # Plot states
  for i in range(ns):
    axs_states[i].plot(tt_hor, xx_star[i, :], state_colors[i], linewidth=2, label=f'{state_labels[i]} (opt.)')
    axs_states[i].plot(tt_hor, xx_ref[i, :], 'g--', linewidth=2, label=f'{state_labels[i]} (ref.)')

    axs_states[i].grid()
    axs_states[i].legend()
    axs_states[i].set_ylabel(f'{state_labels[i]}')
  axs_states[-1].set_xlabel('Time')

  # Optimal trajectory - Controls
  fig, axs_controls = plt.subplots(ni, 1, sharex='all', figsize=(10, 6))

  # Plot control inputs
  for i in range(ni):
    axs_controls[i].plot(tt_hor, uu_star[i, :], control_colors[i], linewidth=2, label=f'{control_labels[i]} (optimal)')
    axs_controls[i].plot(tt_hor, uu_ref[i, :], 'g--', linewidth=2, label=f'{control_labels[i]} (reference)')

    axs_controls[i].grid()
    axs_controls[i].legend()
    axs_controls[i].set_ylabel(f'{control_labels[i]}')
  axs_controls[-1].set_xlabel('Time')

  plt.show()