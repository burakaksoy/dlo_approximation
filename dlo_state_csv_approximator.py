import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import traceback

from dlo_state_approximator import dlo_state_approximator

# Set the default DPI for all images
plt.rcParams['figure.dpi'] = 100  # e.g. 300 dpi
# Set the default figure size
plt.rcParams['figure.figsize'] = [25.6, 19.2]  # e.g. 6x4 inches
# plt.rcParams['figure.figsize'] = [12.8, 9.6]  # e.g. 6x4 inches
# plt.rcParams['figure.figsize'] = [6.4, 4.8]  # e.g. 6x4 inches

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def read_n_plot_csv_data(file, plot=True):
    # Load the data from the CSV file
    df = pd.read_csv(file)
    
    # Extract position data
    p_x = df['p_x']
    p_y = df['p_y']
    p_z = df['p_z']

    # Extract orientation data (quaternion)
    o_x = df['o_x']
    o_y = df['o_y']
    o_z = df['o_z']
    o_w = df['o_w']

    if plot:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the position data
        ax.scatter(p_x, p_y, p_z, c='r', marker='o')

        # Plot the orientation data
        for i in range(len(p_x)):
            # Create a rotation object from the quaternion
            r = R.from_quat([o_x[i], o_y[i], o_z[i], o_w[i]])
            
            # Define a unit vector along the z-axis
            vec = np.array([0, 0, 1])
            
            # Rotate the vector
            vec_rotated = r.apply(vec)
            
            # Plot the orientation vector
            ax.quiver(p_x[i], p_y[i], p_z[i], vec_rotated[0], vec_rotated[1], vec_rotated[2], length=0.01, normalize=True)

        # Set labels
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title('3D Position and Orientation Plot')

        # Ensure the aspect ratio is equal
        ax.set_box_aspect([np.ptp(p_x), np.ptp(p_y), np.ptp(p_z)])  # aspect ratio is 1:1:1
        
        
        # Show the plot
        plt.show()
    
    return p_x.to_numpy(), p_y.to_numpy(), p_z.to_numpy(), o_x.to_numpy(), o_y.to_numpy(), o_z.to_numpy(), o_w.to_numpy()

# File paths
file = 'dlo_state_example_1.csv'
# file = 'dlo_state_example_2.csv'

# Length of the DLO
l_dlo = 0.5 # meters

# Read the data for the DLO state from the CSV file
p_x, p_y, p_z, o_x, o_y, o_z, o_w = read_n_plot_csv_data(file, plot=False)
# p_z[:] = 0.0
dlo_state = np.array([p_x, p_y, p_z, o_x, o_y, o_z, o_w]).T # Nx7 numpy array
# print("dlo_state =", dlo_state)
# print("dlo_state.shape =", dlo_state.shape)

avr_errors = []
joint_poss = [] # 
max_angles = [] # (radians)
run_times = []

for num_seg_d in range(1, len(dlo_state)+1):
    try:
        print("num_seg_d =", num_seg_d)
        
        time.sleep(0.5) # Needed to avoid the wrong timing results
        start_time = time.time()
        approximated_pos, joint_pos, max_angle, avg_error = dlo_state_approximator(l_dlo, dlo_state, num_seg_d, start_from_beginning=True)
        end_time = time.time()
        
        # # Find the cumulative length of the approximated positions
        # cum_len = np.cumsum(np.linalg.norm(approximated_pos[1:,:] - approximated_pos[:-1,:], axis=1))
        # print("cum_len =", cum_len[-1])
        
        avr_errors.append(100*avg_error)
        print("avg_error =", 100*avg_error, "cm.")
        
        joint_poss.append(joint_pos)
        print("joint_pos =", joint_pos, "(first 3 are in meters, the next are in radians).")
        
        max_angles.append(np.rad2deg(max_angle))
        print("max_angle =", np.rad2deg(max_angle), "degrees.")
        
        run_times.append(end_time - start_time)
        print("run_time =", (end_time - start_time)*1000, "ms.")
            
        print("-------------------------------------------")

        # # plot the results
        # ax = plt.figure().add_subplot(projection='3d')
        
        # # Add title with the number of segments
        # ax.set_title("Approx. w/ Number of Segments = " + str(num_seg_d), fontsize=30)
        
        # ax.plot(p_x, p_y, p_z, 'o', label='original', markersize=6)
        # ax.plot(approximated_pos[:,0], approximated_pos[:,1],approximated_pos[:,2], '-', label='approximated', markersize=12, linewidth=3)

        # ax.legend(fontsize=20)
        # ax.tick_params(axis='both', which='major', labelsize=20)
        # # ax.set_aspect('equal')
        # set_axes_equal(ax)
        # plt.show()
        
    
    except:
        # Print the traceback
        print(traceback.format_exc())
        
        # Print the error message
        print("Error occurred for num_seg_d =", num_seg_d)
        
        continue

# ------------------------------------------------------------------------------
# Plot the average errors for each number of segments
plt.figure()

plt.plot(range(1, len(dlo_state)+1), avr_errors, linewidth=5)

plt.xlabel('Number of Segments', fontsize=25)
plt.ylabel('Average Error (cm)', fontsize=25)
plt.title('Average Error vs. Number of Segments', fontsize=30)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Plot the maximum angles for each number of segments
plt.figure()
plt.plot(range(1, len(dlo_state)+1), max_angles, linewidth=5)

plt.xlabel('Number of Segments', fontsize=25)
plt.ylabel('Maximum Angle (degrees)', fontsize=25)
plt.title('Maximum Rotation Angle btw Segments vs. Number of Segments', fontsize=30)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Plot the run times for each number of segments
plt.figure()

plt.plot(range(1, len(dlo_state)+1), 1000*np.array(run_times), linewidth=5)
# # Also plot the average run time
# run_time_avg = 1000*np.mean(run_times)
# plt.plot(range(1, len(dlo_state)+1), [run_time_avg]*len(dlo_state), linewidth=5, linestyle='--')
# plt.legend(['Run Time', 'Average Run Time'])

plt.xlabel('Number of Segments', fontsize=25)
plt.ylabel('Run Time (milliseconds)', fontsize=25)
plt.title('Run Time vs. Number of Segments', fontsize=30)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()