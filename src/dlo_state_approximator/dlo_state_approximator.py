import pandas as pd

import numpy as np
from scipy.spatial.transform import Rotation as R

from .dists_to_line_segments import min_dist_to_polyline
from .weighting_functions import generate_weighting, generate_middle_peak_weighting_function
from .dlo_inv_kin import dlo_inv_kin


def average_quaternions(Q, weights):
    '''
    Averaging Quaternions using Markley's method with weights.[1]
    See: https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
    
    [1] Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. 
    "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, no. 4 (2007): 1193-1197.

    Arguments:
        Q (ndarray): An Mx4 ndarray of quaternions. Each quaternion has [w, x, y, z] format.
        weights (list): An M elements list, a weight for each quaternion.

    Returns:
        ndarray: The weighted average of the input quaternions in [w, x, y, z] format. 
    '''
    # Use the optimized numpy functions for a more efficient computation
    A = np.einsum('ij,ik,i->...jk', Q, Q, weights)
    # Compute the eigenvectors and eigenvalues, and return the eigenvector corresponding to the largest eigenvalue
    quaternion = np.linalg.eigh(A)[1][:, -1]
    
    return normalize_quaternion(quaternion)

def normalize_quaternion(quaternion):
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Cannot normalize a quaternion with zero norm.")
    return quaternion / norm

# Start from the beginning of the DLO
def dlo_state_approximator_from_beginning(l_dlo, dlo_state, num_seg_d):
    num_seg = len(dlo_state) # number of segments
    # print("num_seg = ", num_seg)
    
    l_seg_d = l_dlo / num_seg_d # meters # desired segment length

    # Make sure the desired number of segments is less than or equal to the original number of segments
    assert 1 <= num_seg_d, "The desired number of segments must be more than or equal to 1."  
    assert num_seg_d <= num_seg, "The desired number of segments must be less than or equal to the original number of segments."

    # Number of original segments that each group of desired segments represents
    num_seg_group = num_seg / num_seg_d   
    # print("num_seg_group = ", num_seg_group)

    # Initialize the approximated position array
    approximated_pos = np.zeros((num_seg_d+1, 3)) # (N+1)x3, (x, y, z) order.
    approximated_ori = np.zeros((num_seg_d, 3, 3)) # Nx3x3, rotation matrix for each segment.

    start_tip = None
    end_tip   = None
    
    for i in range(num_seg_d):
        # Calculate the starting and ending indices for the current group
        start_idx = int(np.round(  i   * num_seg_group))
        end_idx   = int(np.round((i+1) * num_seg_group))
        
        # print("start_idx = ", start_idx)
        # print("end_idx = ", end_idx)
                        
        # Calculate the mean orientation for the current group
        group_ori = dlo_state[start_idx:end_idx,(6,3,4,5)] # Nx4, (w, x, y, z) order.
        weights = np.ones(end_idx-start_idx)  # Assuming equal weight for simplicity
        
        group_ori_mean = average_quaternions(group_ori, weights) # [w, x, y, z]
        
        # Calculate the main direction vector for the current group
        # Create a rotation object from the quaternion
        r = R.from_quat([group_ori_mean[1], group_ori_mean[2], group_ori_mean[3], group_ori_mean[0]]) # [x, y, z, w]
        
        # Convert to rotation matrix and store it in the approximated orientation array
        approximated_ori[i] = r.as_matrix() # 3x3
        
        # Define a unit vector along the z-axis (Assuming the z-axis is the main direction of the dlo)
        vec = np.array([0, 0, 1])        
        # Rotate the vector
        group_dir = r.apply(vec) # the main direction vector for the current group
                        
        # Determine the start and end tip positions for the current group using the main direction vector
        if start_tip is None:
            # Convert the group positions into a Nx3 array
            group_pos = dlo_state[start_idx:end_idx,(0,1,2)] # Nx3, (x, y, z) order.
            
            # Calculate the mean position for the current group
            group_pos_mean = np.mean(group_pos, axis=0) # (,3)
        
            start_tip = group_pos_mean - group_dir * l_seg_d/2.0 # Nx3
            end_tip = group_pos_mean + group_dir * l_seg_d/2.0 # Nx3
        else:
            start_tip = end_tip                
            end_tip = start_tip + group_dir * l_seg_d
            
        approximated_pos[i,:] = start_tip
        
        if i == num_seg_d - 1:
            approximated_pos[i+1,:] = end_tip

    # print("approximated_pos =", approximated_pos)
                
    # joint_pos = dlo_inv_kin_old(approximated_pos) 
    joint_pos = dlo_inv_kin(approximated_pos, approximated_ori) 
    
    # the maximum absolute rotation angle between the approximated line segments
    # when there are three consecutive joints rotating around x, y, and z axes with Rx, Ry, and Rz respectively.
    if len(joint_pos) <= 6:
        max_angle = 0.0
    else:
        max_angle = np.max(np.abs(joint_pos[6:])) # ignore the first 3 joints (translational joints), and the next 3 joints (the three rotational joints) that are used for inital orientation segment of the DLO. Hence we start from the 7th joint (index 6).
            
    errors = min_dist_to_polyline(points=dlo_state[:,0:3], polyline=approximated_pos) 
    # print("errors = ", errors)
    avg_error = np.mean(errors)
            
    return approximated_pos, joint_pos, max_angle, avg_error


# def dlo_state_approximator_from_middle(l_dlo, dlo_state, num_seg_d):
#     num_seg = len(dlo_state) # number of segments
#     # print("num_seg = ", num_seg)
    
#     l_seg_d = l_dlo / num_seg_d # meters # desired segment length

#     # Make sure the desired number of segments is less than or equal to the original number of segments
#     assert 1 <= num_seg_d, "The desired number of segments must be more than or equal to 1."  
#     assert num_seg_d <= num_seg, "The desired number of segments must be less than or equal to the original number of segments."

#     # Number of original segments that each group of desired segments represents
#     num_seg_group = num_seg / num_seg_d   
#     # # print("num_seg_group = ", num_seg_group)

#     # Initialize the approximated position array
#     approximated_pos = np.zeros((num_seg_d+1, 3)) # (N+1)x3, (x, y, z) order.
#     approximated_ori = np.zeros((num_seg_d, 3,3)) # Nx3x3, rotation matrix for each segment.    

#     start_tip = None
#     end_tip   = None
    
#     mid_idx_d = int(np.round(num_seg_d/2))

#     # POSITIVE DIRECTION FROM THE CENTER OF THE DLO
#     for i in range(mid_idx_d, num_seg_d, 1):
#         # Calculate the starting and ending indices for the current group
#         start_idx = int(np.round(  i   * num_seg_group))
#         end_idx   = int(np.round((i+1) * num_seg_group))
        
#         # print("start_idx = ", start_idx)
#         # print("end_idx = ", end_idx)
        
#         group_ori = dlo_state[start_idx:end_idx,(6,3,4,5)] # Nx4, (w, x, y, z) order.
#         weights = np.ones(end_idx-start_idx)  # Assuming equal weight for simplicity
#         # weights = generate_weighting(end_idx-start_idx, alpha=10)
#         # weights = generate_middle_peak_weighting(end_idx-start_idx, sigma=0.4)
        
        
#         # Calculate the mean orientation for the current group
#         group_ori_mean = average_quaternions(group_ori, weights) # [w, x, y, z]
        
#         # Calculate the main direction vector for the current group
#         # Create a rotation object from the quaternion
#         r = R.from_quat([group_ori_mean[1], group_ori_mean[2], group_ori_mean[3], group_ori_mean[0]]) # [x, y, z, w]
        
#         # Convert to rotation matrix and store it in the approximated orientation array
#         approximated_ori[i] = r.as_matrix() # 3x3
        
#         # Define a unit vector along the z-axis (Assuming the z-axis is the main direction of the dlo)
#         vec = np.array([0, 0, 1])        
#         # Rotate the vector
#         group_dir = r.apply(vec) # the main direction vector for the current group
                        
#         # Determine the start and end tip positions for the current group using the main direction vector
#         if start_tip is None:
#             # Convert the group positions into a Nx3 array
#             group_pos = dlo_state[start_idx:end_idx,(0,1,2)] # Nx3, (x, y, z) order.
            
#             # Calculate the mean position for the current group
#             group_pos_mean = np.mean(group_pos, axis=0) # (,3)
            
#             start_tip = group_pos_mean - group_dir * l_seg_d/2.0 # Nx3
#             end_tip = group_pos_mean + group_dir * l_seg_d/2.0 # Nx3
#         else:
#             start_tip = end_tip                
#             end_tip = start_tip + group_dir * l_seg_d
            
#         approximated_pos[i,:] = start_tip
        
#         if i == num_seg_d - 1:
#             approximated_pos[i+1,:] = end_tip
            
#     # print("starting the negative direction")
    
#     start_tip = approximated_pos[mid_idx_d,:]
#     end_tip   = approximated_pos[mid_idx_d,:]
        
#     # NEGATIVE DIRECTION FROM THE CENTER OF THE DLO
#     for i in range(mid_idx_d-1, -1, -1):
    
#         # Calculate the starting and ending indices for the current group
#         start_idx = int(np.round((i+1) * num_seg_group))
#         end_idx   = int(np.round(  i   * num_seg_group))
                    
#         # print("start_idx = ", start_idx)
#         # print("end_idx = ", end_idx)
        
#         group_ori = dlo_state[end_idx:start_idx,(6,3,4,5)] # Nx4, (w, x, y, z) order.
#         weights = np.ones(start_idx-end_idx)  # Assuming equal weight for simplicity
#         # weights = generate_weighting(start_idx-end_idx, alpha=10)
#         # weights = generate_middle_peak_weighting(start_idx-end_idx, sigma=0.4)
        
#         # Calculate the mean orientation for the current group
#         group_ori_mean = average_quaternions(group_ori, weights) # [w, x, y, z]
        
#         # Calculate the main direction vector for the current group
#         # Create a rotation object from the quaternion
#         r = R.from_quat([group_ori_mean[1], group_ori_mean[2], group_ori_mean[3], group_ori_mean[0]]) # [x, y, z, w]
        
#         # Convert to rotation matrix and store it in the approximated orientation array
#         approximated_ori[i] = r.as_matrix() # 3x3
        
#         # Define a unit vector along the z-axis (Assuming the z-axis is the main direction of the dlo)
#         vec = np.array([0, 0, -1])        
#         # Rotate the vector
#         group_dir = r.apply(vec) # the main direction vector for the current group
        
#         # Determine the start and end tip positions for the current group using the main direction vector
#         start_tip = end_tip                
#         end_tip = start_tip + group_dir * l_seg_d
        
#         approximated_pos[i,:] = end_tip
        
#     # joint_pos = dlo_inv_kin_old(approximated_pos) 
#     joint_pos = dlo_inv_kin(approximated_pos, approximated_ori) 
    
#     # the maximum absolute rotation angle between the approximated line segments
#     # when there are three consecutive joints rotating around x, y, and z axes with Rx, Ry, and Rz respectively.
#     if len(joint_pos) <= 6:
#         max_angle = 0.0
#     else:
#         max_angle = np.max(np.abs(joint_pos[6:])) # ignore the first 3 joints (translational joints), and the next 3 joints (the three rotational joints) that are used for inital orientation segment of the DLO. Hence we start from the 7th joint (index 6).
            
#     errors = min_dist_to_polyline(points=dlo_state[:,0:3], polyline=approximated_pos) 
#     # print("errors = ", errors)
#     avg_error = np.mean(errors)
            
#     return approximated_pos, joint_pos, max_angle, avg_error


def dlo_state_approximator(l_dlo, dlo_state, num_seg_d, start_from_beginning=True):    
    # if start_from_beginning:
    return dlo_state_approximator_from_beginning(l_dlo, dlo_state, num_seg_d)
    # else:
    #     return dlo_state_approximator_from_middle(l_dlo, dlo_state, num_seg_d)
    