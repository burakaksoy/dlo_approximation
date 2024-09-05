import warnings
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def dlo_fwd_kin(joint_pos, dlo_l, return_rot_matrices=False):
    """
        Compute the polyline (vertices) of the DLO given the joint angles of the DLO from the Kinematics model.
        Kinematics model is defined as follows:
        - The first 3 elements of joint_pos are the translational joint values as prismatic joints (x, y, z)
        - The last 3N elements of joint_pos are the rotational joint angles around x, y, and z axes for each segment.
        - The zero-config is the straight line along the +z-axis.
        - The DLO has EQUAL length N segments. The length of each segment is dlo_l/N.
        - N can be calculated as (len(joint_pos) - 3) // 3
        
    Args:
        joint_pos: (3+3N)x1 numpy array of the joint angles of the DLO. 
            The first 3 elements are the translational joint angles (x, y, z)
            and the last 3N elements are the rotational joint angles around x, y, and z axes for each segment.
        dlo_l: The length of the DLO.
        return_rot_matrices: If True, the rotation matrices for each segment are also returned. (optional)

    Returns:
        polyline: A (N+1)x3 numpy array of points that represent the vertices of the DLO where N is the number of segments.
        rot_matrices: (OPTIONAL) A Nx3x3 numpy array of rotation matrices for each segment. 
                        Each Rotation matrix from the origin to the given point n
    """
    
    # Get the number of joints
    num_joints = len(joint_pos)
    
    # Get the number of segments
    N = (num_joints - 3) // 3
    
    # Segment length
    seg_l = dlo_l / N
    
    # Initialize the polyline as the vertex positions of the DLO in 3D space (x, y, z)
    polyline = np.zeros((N + 1, 3))
    
    # Initialize the rotation matrices
    rot_matrices = np.zeros((N, 3, 3))
    
    # Initialize the current rotation matrix 
    current_rot = np.eye(3) # Identity matrix
    
    # Initialize the current position vector from the first 3 elements of joint_pos 
    current_pos = joint_pos[:3]
    
    # Assign the first point to the polyline
    polyline[0] = current_pos
    
    # Iterate through the revolute joints
    for i in range(3,num_joints,3):
        # Get the current joint angles
        current_joint_angles = joint_pos[i:i+3]
        
        # Create a rotation matrix for the revolute joint
        rotation = R.from_euler('XYZ', joint_pos[i:i+3], degrees=False).as_matrix()
        
        current_rot = current_rot @ rotation # Update the current rotation matrix
        
        # Get the current position vector
        current_pos = current_pos + current_rot @ np.array([0, 0, seg_l]) # Move along the z-axis of the current frame
        
        # Update the polyline
        polyline[i//3] = current_pos
        
        # Update the rotation matrices
        rot_matrices[i//3 - 1] = current_rot
        
    if return_rot_matrices:
        return polyline, rot_matrices
    else:
        return polyline

    
def test_approximated_dlo_fwd_kin1():
    # Define the joint angles of the DLO
    joint_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Define the length of the DLO
    dlo_l = 1
    
    # Compute the polyline
    polyline = dlo_fwd_kin(joint_pos, dlo_l)
    
    print(polyline)
    
    # Define the expected polyline
    expected_polyline = np.array([
        [0, 0, 0],
        [0, 0, 0.5],
        [0, 0, 1]
    ])
    
    # Compare the results
    assert np.allclose(polyline, expected_polyline), "Test failed!"
    
    print("Test passed!")

def test_approximated_dlo_fwd_kin2():
    # Define the joint angles of the DLO
    joint_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Define the length of the DLO
    dlo_l = 1
    
    # Compute the polyline
    polyline, rot_matrices = dlo_fwd_kin(joint_pos, dlo_l, return_rot_matrices=True)
    
    print("polyline:", polyline)
    print("rot_matrices:", rot_matrices)
    
    # Define the expected polyline
    expected_polyline = np.array([
        [0, 0, 0],
        [0, 0, 0.5],
        [0, 0, 1]
    ])
    
    expected_rot_matrices = np.array([
        np.eye(3),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    ])
    
    # Compare the results
    assert np.allclose(polyline, expected_polyline), "Test failed!"
    assert np.allclose(rot_matrices, expected_rot_matrices), "Test failed!"
    
    print("Test passed!")

# Run the tests
if __name__ == "__main__":
    # test_approximated_dlo_fwd_kin1()
    test_approximated_dlo_fwd_kin2()