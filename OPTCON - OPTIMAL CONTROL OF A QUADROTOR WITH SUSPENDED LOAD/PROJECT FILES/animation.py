import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from referance_curve import tf
from dynamics import dt

TT = int(tf/dt)
tt_hor = np.linspace(0,tf,TT)
time = np.arange(len(tt_hor))*dt

# Load the array from the saved file
loaded_data = np.load('xx_lqr.npz')
xx_lqr = loaded_data['result_array']

loaded_data = np.load('xx_opt.npz')
xx_opt = loaded_data['result_array']

# Assuming you have arrays for reference and computed data
# Replace these with your actual data
reference_x = xx_opt[0,:]
reference_y = xx_opt[1,:] 

computed_x = xx_lqr[0,:] 
computed_y = xx_lqr[1,:] 

# Set up the figure and axes
fig, (ax_main, ax_track_x, ax_track_y) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]})
ax_main.set_xlim(min(min(reference_x), min(computed_x)), max(max(reference_x), max(computed_x)))
ax_main.set_ylim(min(min(reference_y), min(computed_y)), max(max(reference_y), max(computed_y)))
ax_main.set_xlabel('X-axis')
ax_main.set_ylabel('Y-axis')
ax_main.set_title('Drone Animation')

# Initialize the drone marker, reference line, and computed line
drone_marker, = ax_main.plot([], [], 'o', markersize=10, label='Drone')
reference_line, = ax_main.plot(reference_x, reference_y, 'g-', label='Reference')
computed_line, = ax_main.plot([], [], 'r-', label='Computed')

# Add a legend
ax_main.legend()

# Function to initialize the animation
def init():
    drone_marker.set_data([], [])
    computed_line.set_data([], [])
    return drone_marker, reference_line, computed_line

# Function to update the main animation
def update(frame):
    x_computed = computed_x[frame]
    y_computed = computed_y[frame]

    drone_marker.set_data(x_computed, y_computed)
    computed_line.set_data(computed_x[:frame], computed_y[:frame])  # Update the computed line
    return drone_marker, reference_line, computed_line

# Create the main animation
ani_main = animation.FuncAnimation(fig, update, frames=len(computed_x), init_func=init, blit=True, interval=5)

# Set up the axes for tracking animations
ax_track_x.set_xlim(0, len(xx_lqr[0,:]))
ax_track_x.set_ylim(min(xx_lqr[0,:]), max(xx_lqr[0,:]))
ax_track_x.set_xlabel('Frame Index')
ax_track_x.set_ylabel('X-axis')
ax_track_x.set_title('X-axis Tracking')

ax_track_y.set_xlim(0, len(xx_lqr[1,:]))
ax_track_y.set_ylim(min(xx_lqr[1,:]), max(xx_lqr[1,:]))
ax_track_y.set_xlabel('Frame Index')
ax_track_y.set_ylabel('Y-axis')
ax_track_y.set_title('Y-axis Tracking')

# Initialize the tracking markers and lines
drone_marker_x, = ax_track_x.plot([], [], 'o', markersize=5, label='Drone X')
tracking_line_x, = ax_track_x.plot([], [], 'r-', label='Tracking X')
reference_line_x, = ax_track_x.plot([], [], 'g-', label='Reference X')

drone_marker_y, = ax_track_y.plot([], [], 'o', markersize=5, label='Drone Y')
tracking_line_y, = ax_track_y.plot([], [], 'r-', label='Tracking Y')
reference_line_y, = ax_track_y.plot([], [], 'g-', label='Reference Y')

# Add legends
ax_track_x.legend()
ax_track_y.legend()

# Function to initialize the tracking animations
def init_track():
    drone_marker_x.set_data([], [])
    tracking_line_x.set_data([], [])
    reference_line_x.set_data([], [])

    drone_marker_y.set_data([], [])
    tracking_line_y.set_data([], [])
    reference_line_y.set_data([], [])

    return drone_marker_x, tracking_line_x, reference_line_x, drone_marker_y, tracking_line_y, reference_line_y

# Function to update the X-axis tracking animation
def update_track_x(frame):
    x_track = xx_lqr[0,:frame]

    drone_marker_x.set_data(frame, xx_lqr[0, frame])
    tracking_line_x.set_data(range(frame), x_track)  # Update the tracking line
    reference_line_x.set_data(range(len(reference_x)), reference_x)  # Update the reference line
    return drone_marker_x, tracking_line_x, reference_line_x

# Function to update the Y-axis tracking animation
def update_track_y(frame):
    y_track = xx_lqr[1,:frame]

    drone_marker_y.set_data(frame, xx_lqr[1, frame])
    tracking_line_y.set_data(range(frame), y_track)  # Update the tracking line
    reference_line_y.set_data(range(len(reference_y)), reference_y)  # Update the reference line
    return drone_marker_y, tracking_line_y, reference_line_y

# Create the main animation with a slower interval (e.g., 50 milliseconds)
ani_main = animation.FuncAnimation(fig, update, frames=len(computed_x), init_func=init, blit=True, interval=20)

# Create the tracking animations with a slower interval (e.g., 50 milliseconds)
ani_track_x = animation.FuncAnimation(fig, update_track_x, frames=len(xx_lqr[0,:]), init_func=init_track, blit=True, interval=20)
ani_track_y = animation.FuncAnimation(fig, update_track_y, frames=len(xx_lqr[1,:]), init_func=init_track, blit=True, interval=20)

# Show the animations
plt.show()
