import pandas as pd

import numpy as np
from scipy.spatial.transform import Rotation as R

from .dists_to_line_segments import min_dist_to_polyline


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

def generate_weighting(n, alpha=4):
    """
    Generate a discrete weighting function where the last element has a very high weight
    and the first element has a very low weight. The sum of all weights is normalized to 1.
    
    Parameters:
    n (int): Number of discrete sections.
    alpha (float): Common ratio of the geometric series, must be greater than 1.
    
    Returns:
    weights (np.ndarray): Normalized weights.
    """
    # Generate the weights using a geometric series
    weights = np.array([alpha**i for i in range(n)], dtype=float)
    
    # Normalize the weights to sum up to 1
    weights /= np.sum(weights)
    
    return weights

def generate_middle_peak_weighting(n, sigma=1.0):
    """
    Generate a discrete weighting function where the middle element has the highest weight
    and the first and last elements have the lowest weight. The sum of all weights is normalized to 1.
    
    Parameters:
    n (int): Number of discrete sections.
    sigma (float): Standard deviation of the Gaussian distribution.
    
    Returns:
    weights (np.ndarray): Normalized weights.
    """
    # Generate the x values centered around the middle of the range
    x = np.linspace(-1, 1, n)
    
    # Generate the weights using a Gaussian function
    weights = np.exp(-0.5 * (x / sigma)**2)
    
    # Normalize the weights to sum up to 1
    weights /= np.sum(weights)
    
    return weights

# Start from the beginning of the DLO
def dlo_state_approximator_from_beginning(l_dlo, dlo_state, num_seg_d):

    p_x, p_y, p_z, o_x, o_y, o_z, o_w = dlo_state[0], dlo_state[1], dlo_state[2], dlo_state[3], dlo_state[4], dlo_state[5], dlo_state[6]

    num_seg = len(p_x) # 40 # number of segments
    # print("num_seg = ", num_seg)
    
    # print("num_seg_d = ", num_seg_d)
    
    l_seg_d = l_dlo / num_seg_d # meters # desired segment length

    # Make sure the desired number of segments is less than or equal to the original number of segments
    assert 1 <= num_seg_d, "The desired number of segments must be more than or equal to 1."  
    assert num_seg_d <= num_seg, "The desired number of segments must be less than or equal to the original number of segments."

    # Number of original segments that each group of desired segments represents
    num_seg_group = num_seg / num_seg_d   
    # print("num_seg_group = ", num_seg_group)

    # Initialize the approximated position array
    approximated_pos = np.zeros((num_seg_d+1, 3))

    start_tip = None
    end_tip   = None
    
    max_angle = 0

    for i in range(num_seg_d):
        # Calculate the starting and ending indices for the current group
        start_idx = int(np.round(  i   * num_seg_group))
        end_idx   = int(np.round((i+1) * num_seg_group))
        
        # print("start_idx = ", start_idx)
        # print("end_idx = ", end_idx)
                
        # Convert the group positions into a Nx3 array
        group_pos = np.array([p_x[start_idx:end_idx], 
                            p_y[start_idx:end_idx], 
                            p_z[start_idx:end_idx]]).T  # Nx3
        
        group_ori = np.array([o_w[start_idx:end_idx],
                              o_x[start_idx:end_idx], 
                              o_y[start_idx:end_idx], 
                              o_z[start_idx:end_idx]]).T  # Nx4
        weights = np.ones(end_idx-start_idx)  # Assuming equal weight for simplicity
        
        # Calculate the mean position for the current group
        group_pos_mean = np.mean(group_pos, axis=0) # (,3)
        
        # Calculate the mean orientation for the current group
        group_ori_mean = average_quaternions(group_ori, weights) # [w, x, y, z]
        
        # Calculate the main direction vector for the current group
        # Create a rotation object from the quaternion
        r = R.from_quat([group_ori_mean[1], group_ori_mean[2], group_ori_mean[3], group_ori_mean[0]]) # [x, y, z, w]
        # Define a unit vector along the z-axis (Assuming the z-axis is the main direction of the dlo)
        vec = np.array([0, 0, 1])        
        # Rotate the vector
        group_dir = r.apply(vec) # the main direction vector for the current group
                        
        # Determine the start and end tip positions for the current group using the main direction vector
        if start_tip is None:
            start_tip = group_pos_mean - group_dir * l_seg_d/2.0 # Nx3
            end_tip = group_pos_mean + group_dir * l_seg_d/2.0 # Nx3
        else:
            start_tip = end_tip                
            end_tip = start_tip + group_dir * l_seg_d
            
        approximated_pos[i,:] = start_tip
        
        if i == num_seg_d - 1:
            approximated_pos[i+1,:] = end_tip
            
        # TODO: Return the max rotation angle between the approximated line segments
            
    errors = min_dist_to_polyline(points=np.array(dlo_state[0:3]).T, polyline=approximated_pos)
    # print("errors = ", errors)
    avg_error = np.mean(errors)
            
    return approximated_pos, max_angle, avg_error


def dlo_state_approximator_from_middle(l_dlo, dlo_state, num_seg_d):
    
    p_x, p_y, p_z, o_x, o_y, o_z, o_w = dlo_state[0], dlo_state[1], dlo_state[2], dlo_state[3], dlo_state[4], dlo_state[5], dlo_state[6]

    num_seg = len(p_x) # 40 # number of segments
    # print("num_seg = ", num_seg)
    
    l_seg_d = l_dlo / num_seg_d # meters # desired segment length

    ## -------- End Parameters ---------- ##      

    # Make sure the desired number of segments is less than or equal to the original number of segments
    assert 1 <= num_seg_d, "The desired number of segments must be more than or equal to 1."  
    assert num_seg_d <= num_seg, "The desired number of segments must be less than or equal to the original number of segments."

    # Number of original segments that each group of desired segments represents
    num_seg_group = num_seg / num_seg_d   
    # # print("num_seg_group = ", num_seg_group)

    # Initialize the approximated position array
    approximated_pos = np.zeros((num_seg_d+1, 3))

    start_tip = None
    end_tip   = None
    
    max_angle = 0

    mid_idx_d = int(np.round(num_seg_d/2))

    # POSITIVE DIRECTION FROM THE CENTER OF THE DLO
    for i in range(mid_idx_d, num_seg_d, 1):
        # Calculate the starting and ending indices for the current group
        start_idx = int(np.round(  i   * num_seg_group))
        end_idx   = int(np.round((i+1) * num_seg_group))
        
        # print("start_idx = ", start_idx)
        # print("end_idx = ", end_idx)
            
        # Convert the group positions into a Nx3 array
        group_pos = np.array([p_x[start_idx:end_idx], 
                            p_y[start_idx:end_idx], 
                            p_z[start_idx:end_idx]]).T  # Nx3
        
        group_ori = np.array([o_w[start_idx:end_idx],
                              o_x[start_idx:end_idx], 
                              o_y[start_idx:end_idx], 
                              o_z[start_idx:end_idx]]).T  # Nx4
        weights = np.ones(end_idx-start_idx)  # Assuming equal weight for simplicity
        # weights = generate_weighting(end_idx-start_idx, alpha=10)
        # weights = generate_middle_peak_weighting(end_idx-start_idx, sigma=0.4)
        
        # Calculate the mean position for the current group
        group_pos_mean = np.mean(group_pos, axis=0) # (,3)
        
        # Calculate the mean orientation for the current group
        group_ori_mean = average_quaternions(group_ori, weights) # [w, x, y, z]
        
        # Calculate the main direction vector for the current group
        # Create a rotation object from the quaternion
        r = R.from_quat([group_ori_mean[1], group_ori_mean[2], group_ori_mean[3], group_ori_mean[0]]) # [x, y, z, w]
        # Define a unit vector along the z-axis (Assuming the z-axis is the main direction of the dlo)
        vec = np.array([0, 0, 1])        
        # Rotate the vector
        group_dir = r.apply(vec) # the main direction vector for the current group
                        
        # Determine the start and end tip positions for the current group using the main direction vector
        if start_tip is None:
            start_tip = group_pos_mean - group_dir * l_seg_d/2.0 # Nx3
            end_tip = group_pos_mean + group_dir * l_seg_d/2.0 # Nx3
        else:
            start_tip = end_tip                
            end_tip = start_tip + group_dir * l_seg_d
            
        approximated_pos[i,:] = start_tip
        
        if i == num_seg_d - 1:
            approximated_pos[i+1,:] = end_tip
            
    # print("starting the negative direction")
    
    start_tip = approximated_pos[mid_idx_d,:]
    end_tip   = approximated_pos[mid_idx_d,:]
        
    # NEGATIVE DIRECTION FROM THE CENTER OF THE DLO
    for i in range(mid_idx_d-1, -1, -1):
    
        # Calculate the starting and ending indices for the current group
        start_idx = int(np.round((i+1) * num_seg_group))
        end_idx   = int(np.round(  i   * num_seg_group))
                    
        # print("start_idx = ", start_idx)
        # print("end_idx = ", end_idx)
            
        # Convert the group positions into a Nx3 array
        group_pos = np.array([p_x[end_idx:start_idx], 
                              p_y[end_idx:start_idx], 
                              p_z[end_idx:start_idx]]).T  # Nx3
        
        group_ori = np.array([o_w[end_idx:start_idx],
                              o_x[end_idx:start_idx], 
                              o_y[end_idx:start_idx], 
                              o_z[end_idx:start_idx]]).T  # Nx4
        weights = np.ones(start_idx-end_idx)  # Assuming equal weight for simplicity
        # weights = generate_weighting(start_idx-end_idx, alpha=10)
        # weights = generate_middle_peak_weighting(start_idx-end_idx, sigma=0.4)
        
        # Calculate the mean position for the current group
        group_pos_mean = np.mean(group_pos, axis=0) # (,3)
        
        # Calculate the mean orientation for the current group
        group_ori_mean = average_quaternions(group_ori, weights) # [w, x, y, z]
        
        # Calculate the main direction vector for the current group
        # Create a rotation object from the quaternion
        r = R.from_quat([group_ori_mean[1], group_ori_mean[2], group_ori_mean[3], group_ori_mean[0]]) # [x, y, z, w]
        # Define a unit vector along the z-axis (Assuming the z-axis is the main direction of the dlo)
        vec = np.array([0, 0, -1])        
        # Rotate the vector
        group_dir = r.apply(vec) # the main direction vector for the current group
        
        # Determine the start and end tip positions for the current group using the main direction vector
        start_tip = end_tip                
        end_tip = start_tip + group_dir * l_seg_d
        
        approximated_pos[i,:] = end_tip
        
        # TODO: Return the max rotation angle between the approximated line segments

    errors = min_dist_to_polyline(points=np.array(dlo_state[0:3]).T, polyline=approximated_pos)
    # print("errors = ", errors)
    avg_error = np.mean(errors)

    return approximated_pos, max_angle, avg_error


def dlo_state_approximator(l_dlo, dlo_state, num_seg_d, start_from_beginning=False):    
    if start_from_beginning:
        return dlo_state_approximator_from_beginning(l_dlo, dlo_state, num_seg_d)
    else:
        return dlo_state_approximator_from_middle(l_dlo, dlo_state, num_seg_d)
    