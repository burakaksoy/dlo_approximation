import warnings
import math
import numpy as np
 

def dlo_inv_kin(polyline):
    """
    Compute the joint angles of the DLO given the vertices of the DLO from the approximated DLO state.

    Args:
        polyline: A (N+1)x3 numpy array of points that represent the vertices of the DLO where N is the number of segments.

    Returns:
        Q: (3+2N)x1 numpy array of the joint angles of the DLO. 
            The first 3 elements are the translational joint angles (x, y, z)
            and the last 2N elements are the rotational joint angles around x and y axes for each segment.
    """
    
    ex = np.array([1.,0,0])
    ey = np.array([0,1.,0])
    
    # Number of segments
    N = len(polyline) - 1
    
    # Initialize the joint angles
    Q = np.zeros(3+2*N)
        
    # Translational joint angles
    Q[0:3] = polyline[0,:] # the first point is the starting point of the first segment
    
    R_0_to_Pn = np.eye(3) # Rotation matrix from the origin to the given point n

    for n in range(1,N+1):
        # print("n = ", n)
        q = polyline[n] - polyline[n - 1]
        # print("q = ", q)
        p = np.array([0,0,np.linalg.norm(q)])
        # print("p = ", p)
        q = (R_0_to_Pn.T).dot(q)
        
        q_sol = subproblem2(p, q, ex, ey)        
        # print("q_sol = ", q_sol)
        
        Q[2*n+1 : 2*n+3] = least_rotation_angles(q_sol)        
        
        R_0_to_Pn = np.dot(R_0_to_Pn, ( np.dot(rot(ex,Q[2*n+1]), rot(ey, Q[2*n+2])) ) )
                
    return Q

def subproblem0(p, q, k):
    """
    Solves canonical geometric subproblem 0, theta subtended between p and q according to
    
        q = rot(k, theta)*p
           ** assumes k'*p = 0 and k'*q = 0
           
    Requires that p and q are perpendicular to k. Use subproblem 1 if this is not
    guaranteed.

    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotation
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotation
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis unit vector
    :rtype:  number
    :return: theta angle as scalar in radians
    """
    
    eps = np.finfo(np.float64).eps    
    
    assert (np.dot(k,p) < np.sqrt(eps)) and (np.dot(k,q) < np.sqrt(eps)), "k must be perpendicular to p and q"
    
    norm = np.linalg.norm
    
    ep = p / norm(p)
    eq = q / norm(q)
    
    theta = 2 * np.arctan2( norm(ep - eq), norm (ep + eq))
    
    if (np.dot(k,np.cross(p , q)) < 0):
        return -theta
        
    return theta

def subproblem1(p, q, k):
    """
    Solves canonical geometric subproblem 1, theta subtended between p and q according to
    
        q = rot(k, theta)*p
    
    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotation
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotation
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis unit vector
    :rtype:  number
    :return: theta angle as scalar in radians
    """
    
    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
        
    if norm (np.subtract(p, q)) < np.sqrt(eps):
        return 0.0
    
    k = np.divide(k,norm(k))
    
    pp = np.subtract(p,np.dot(p, k)*k)
    qp = np.subtract(q,np.dot(q, k)*k)
    
    # pp = p - np.dot(np.dot(p, k),k)
    # qp = q - np.dot(np.dot(q, k),k)
    
    epp = np.divide(pp, norm(pp))    
    eqp = np.divide(qp, norm(qp))
    
    # epp = pp/norm(pp)
    # eqp = qp/norm(qp)
    
    theta = subproblem0(epp, eqp, k)
    
    # if (np.abs(norm(p) - norm(q)) > norm(p)*1e-2):
    if (np.abs(norm(p) - norm(q)) > np.sqrt(eps)):
        warnings.warn("||p|| and ||q|| must be the same!!!")
    
    return theta

def subproblem2(p, q, k1, k2):
    """
    Solves canonical geometric subproblem 2, solve for two coincident, nonparallel
    axes rotation a link according to
    
        q = rot(k1, theta1) * rot(k2, theta2) * p
    
    solves by looking for the intersection between cones of
    
        rot(k1,-theta1)q = rot(k2, theta2) * p
        
    may have 0, 1, or 2 solutions
       
    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotations
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotations
    :type    k1: numpy.array
    :param   k1: 3 x 1 rotation axis 1 unit vector
    :type    k2: numpy.array
    :param   k2: 3 x 1 rotation axis 2 unit vector
    :rtype:  list of number pairs
    :return: theta angles as list of number pairs in radians
    """
    
    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
    
    k12 = np.dot(k1, k2)
    pk = np.dot(p, k2)
    qk = np.dot(q, k1)
    
    # check if solution exists
    if (np.abs( 1 - k12**2) < eps):
        warnings.warn("No solution - k1 and k2 are parallel!")
        return []
    
    a = np.matmul([[k12, -1], [-1, k12]],[pk, qk]) / (k12**2 - 1) # a = [alpha, beta]
    
    bb = (np.dot(p,p) - np.dot(a,a) - 2*a[0]*a[1]*k12)
    if (np.abs(bb) < eps): bb=0
    
    if (bb < 0):
        warnings.warn("No solution - no intersection found between cones")
        return []
    
    gamma = np.sqrt(bb) / norm(np.cross(k1,k2))
    if (np.abs(gamma) < eps):
        cm=np.array([k1, k2, np.cross(k1,k2)]).T
        c1 = np.dot(cm, np.hstack((a, gamma))) # v
        theta2 = subproblem1(p, c1, k2)
        theta1 = -subproblem1(q, c1, k1)
        return [(theta1, theta2)]
    
    cm=np.array([k1, k2, np.cross(k1,k2)]).T
    c1 = np.dot(cm, np.hstack((a, gamma)))
    c2 = np.dot(cm, np.hstack((a, -gamma)))
    theta1_1 = -subproblem1(q, c1, k1)
    theta1_2 = -subproblem1(q, c2, k1)
    theta2_1 =  subproblem1(p, c1, k2)
    theta2_2 =  subproblem1(p, c2, k2)
    return [(theta1_1, theta2_1), (theta1_2, theta2_2)]

def wrap_to_pi(theta):
    """
    Wraps theta to the range [-pi, pi] efficiently for numpy arrays.
    
    Parameters:
    theta (np.ndarray): Input array of angles in radians.
    
    Returns:
    np.ndarray: Array of angles wrapped to the range [-pi, pi].
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi

def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector
    
             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]
    
    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix    
    """
    
    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]    
    return khat

def rot(k, theta):
    """
    Generates a 3 x 3 rotation matrix from a unit 3 x 1 unit vector axis
    and an angle in radians using the Euler-Rodrigues formula
    
        R = I + sin(theta)*hat(k) + (1 - cos(theta))*hat(k)^2
        
    :type    k: numpy.array
    :param   k: 3 x 1 unit vector axis
    :type    theta: number
    :param   theta: rotation about k in radians
    :rtype:  numpy.array
    :return: the 3 x 3 rotation matrix 
        
    """
    I = np.identity(3)
    khat = hat(k)
    khat2 = khat.dot(khat)
    return I + math.sin(theta)*khat + (1.0 - math.cos(theta))*khat2

def least_rotation_angles(q):
    """
    Compute the least rotation angles that are required to rotate p to the x-y plane.
    
    Input:
        q: theta angles as list of number pairs in radians as returned by subproblem2
        - Can be an empty list if no solution exists. But we know that for our DLO, there is always a solution.
        - Can be a list of length 1 if only one solution exists.
        - Can be a list of length 2 if two solutions exist.
    
    Returns:
        q_sol: single pair of angles in radians
    """
    
    if len(q) == 0:
        warnings.warn("No solution exists")
        return np.zeros(2)
    elif len(q) == 1:
        return q[0]
    else: # len(q) == 2
        # Select the solution with the smallest angle
        solution1 = wrap_to_pi(np.array(q[0]))
        solution2 = wrap_to_pi(np.array(q[1]))
    
        # Calculate the sum of absolute values of angles for both solutions
        sum1 = abs(solution1[0]) + abs(solution1[1])
        sum2 = abs(solution2[0]) + abs(solution2[1])
        
        # Compare the sums and select the one with the least rotation
        if sum1 <= sum2:
            return solution1
        else:
            return solution2

# ------------------------------------------------------------------------------

# Test case for approximated dlo inverse kinematics
def test_approximated_dlo_inv_kin1():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin(polyline)
    
    print("The following q must be all zeros") 
    print("q = ", q)
    
def test_approximated_dlo_inv_kin2():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin(polyline)
    
    print("The following q must be zeros in the first three elements,")
    print("then the next two elements must be (zero, pi/2),")
    print("and the remaining elements must be (zero, zero) series")
    print("q = ", q)
    
def test_approximated_dlo_inv_kin3():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 1]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin(polyline)
    
    print("The following q must be zeros in the first three elements")
    print("then the other elements must be [-pi/2  pi/4  pi/2  zero]")
    print("q = ", q)
    
def test_approximated_dlo_inv_kin4():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [0, 0, 0],
        [-1, 1, 0],
        [-1, 1, 1]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin(polyline)
    
    print("The following q must be zeros in the first three elements")
    print("then the other elements must be [-pi/2  -pi/4  pi/2  zero]")
    print("q = ", q)

# Run the tests
if __name__ == "__main__":
    test_approximated_dlo_inv_kin1()
    print("-----------------------------")
    test_approximated_dlo_inv_kin2()
    print("-----------------------------")
    test_approximated_dlo_inv_kin3()
    print("-----------------------------")
    test_approximated_dlo_inv_kin4()