import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def plot_csv_data(file):
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
    set_axes_equal(ax)
    # ax.set_box_aspect([np.ptp(p_x), np.ptp(p_y), np.ptp(p_z)])  # aspect ratio is 1:1:1
    
    # Show the plot
    plt.show()


# Example usage
file = 'dlo_state_example_1.csv'
# file = 'dlo_state_example_2.csv'
plot_csv_data(file)
