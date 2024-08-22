import warnings
import math
import numpy as np
 

def dlo_inv_kin(polyline, rot_matrices):
    """
    Compute the joint angles of the DLO given the vertices of the DLO from the approximated DLO state.

    Args:
        polyline: A (N+1)x3 numpy array of points that represent the vertices of the DLO where N is the number of segments.
        rot_matrices: A Nx3x3 numpy array of rotation matrices for each segment.

    Returns:
        Q: (3+3N)x1 numpy array of the joint angles of the DLO. 
            The first 3 elements are the translational joint angles (x, y, z)
            and the last 3N elements are the rotational joint angles around x, y, and z axes for each segment.
    """
    
    ex = np.array([1.,0,0])
    ey = np.array([0,1.,0])
    ez = np.array([0,0,1.])
    
    # Number of segments
    N = len(polyline) - 1
    
    # Initialize the joint angles
    Q = np.zeros(3+3*N)
        
    # Translational joint angles
    Q[0:3] = polyline[0,:] # the first point is the starting point of the first segment
    
    R_0_to_Pn_1 = np.eye(3) # Rotation matrix from the origin to the given point n-1

    for n in range(1,N+1):
        # print("n = ", n)
        
        R_0_to_Pn = rot_matrices[n-1] # Rotation matrix from the origin to the given point n
        # print("Rotation matrix from the origin to the given point n")
        # print("R_0_to_Pn = ", R_0_to_Pn)
        
        R = R_0_to_Pn_1.T @ R_0_to_Pn # Rotation matrix from the previous segment to the current segment (n-1 to n)
        # print("Rotation matrix from the previous segment to the current segment (n-1 to n):")
        # print("R = ", R)
        
        ## Now we will basically solve for Euler XYZ (current, intrinsic, body centric) angles R = Rx * Ry * Rz
        ## or equivalently, Euler zyx (static, fixed, inertial) angles 

        # ## OPTION 1:
        """
        # DO NOT USE THIS OPTION, IT IS NOT WORKING, 
        # There is an error somewhere! USE OPTION 2 INSTEAD
        
        # Solve for q_y (rotation around y-axis) with Subproblem 4
        q_y = subproblem4(p=ex, q=ez, k=ey, d=ex.T @ R @ ez) # d = p.T*rot(k, theta)*q
        # print("q_y = ", q_y)
        
        # Potentially it has 0, 1, or 2 solutions
        if len(q_y) == 0:
            warnings.warn("No solution for q_y")        
        
        q_sol = np.zeros((len(q_y),3)) # To store the solutions as (q_x, q_y, q_z)
        q_sol[:,1] = np.array(q_y) # Store q_y
        
        # Solve for q_x (rotation around x-axis) with Subproblem 1
        for i in range(len(q_y)):
            q_x = -subproblem1(p=R @ ez, q = rot(ey,q_y[i]) @ ez, k=ex) # q = rot(k, theta)*p
            q_sol[i,0] = q_x # Store q_x
        
        # Solve for q_z (rotation around z-axis) with Subproblem 1
        for i in range(len(q_y)):
            q_z = subproblem1(p=R.T @ ex, q = rot(ey,-q_y[i]) @ ex, k=ez) # q = rot(k, theta)*p
            q_sol[i,2] = q_z # Store q_z
        """
            
        ## OPTION 2:
        # Solve for q_x and q_y (rotation around x-axis and y-axis) with Subproblem 2
        q_x_q_y = subproblem2(p=ez, q=R @ ez, k1= ex, k2=ey) # q = rot(k1, theta1) * rot(k2, theta2) * p
        # print("q_x_q_y = ", q_x_q_y)
        
        # Potentially it has 0, 1, or 2 solutions
        if len(q_x_q_y) == 0:
            warnings.warn("No solution for q_x and q_y")        
        
        q_sol = np.zeros((len(q_x_q_y),3)) # To store the solutions as (q_x, q_y, q_z) as (n solns x 3) array
        for i in range(len(q_x_q_y)):
            q_sol[i,0] = q_x_q_y[i][0] # Store q_x
            q_sol[i,1] = q_x_q_y[i][1] # Store q_y
        
        # Solve for q_z (rotation around z-axis) with Subproblem 1
        for i in range(len(q_x_q_y)):
            q_x = q_sol[i,0]
            q_y = q_sol[i,1]    
            q_z = subproblem1(p=ex, q = rot(ey,-q_y) @ rot(ex,-q_x) @ R @ ex, k=ez) # q = rot(k, theta)*p
            q_sol[i,2] = q_z # Store q_z
             
        # print("q_sol = ", q_sol)
        
        Q[3*n : 3*n+3] = least_rotation_angles(q_sol)        
        
        R_0_to_Pn_1 = R_0_to_Pn # Update the rotation matrix for the next iteration
                
    return Q

def dlo_inv_kin_old(polyline):
    """
    Compute the joint angles of the DLO given the vertices of the DLO from the approximated DLO state.
    DEPRECATED FUNCTION, use dlo_inv_kin instead. 
    The reason is this function consider 2 revolute joints for each segment.
    While the new function considers 3 joints (spherical joint) for each segment.

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

def subproblem3(p, q, k, d):
    """
    WARNING: MAYBE WRONG, DO NOT USE WITHOUT VERIFICATION
    
    Solves canonical geometric subproblem 3,solve for theta in
    an elbow joint according to
    
        || q + rot(k, theta)*p || = d
        
    may have 0, 1, or 2 solutions
    
    :type    p: numpy.array
    :param   p: 3 x 1 position vector of point p
    :type    q: numpy.array
    :param   q: 3 x 1 position vector of point q
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis for point p
    :type    d: number
    :param   d: desired distance between p and q after rotation
    :rtype:  list of numbers
    :return: list of valid theta angles in radians        
    
    """
    
    norm=np.linalg.norm
    
    pp = np.subtract(p,np.dot(np.dot(p, k),k))
    qp = np.subtract(q,np.dot(np.dot(q, k),k))
    dpsq = d**2 - ((np.dot(k, np.add(p,q)))**2)
    
    bb=-(np.dot(pp,pp) + np.dot(qp,qp) - dpsq)/(2*norm(pp)*norm(qp))
    
    if dpsq < 0 or np.abs(bb) > 1:
        warnings.warn("No solution - no rotation can achieve specified distance")
        return []
    
    theta = subproblem1(pp/norm(pp), qp/norm(qp), k)
    
    phi = np.arccos(bb)
    if np.abs(phi) > 0:
        return [theta + phi, theta - phi]
    else:
        return [theta]
    
def subproblem4(p, q, k, d):
    """
    WARNING: MAYBE WRONG, DO NOT USE WITHOUT VERIFICATION
    
    Solves canonical geometric subproblem 4, theta for static
    displacement from rotation axis according to
    
        d = p.T*rot(k, theta)*q
        
    may have 0, 1, or 2 solutions
        
    :type    p: numpy.array
    :param   p: 3 x 1 position vector of point p
    :type    q: numpy.array
    :param   q: 3 x 1 position vector of point q
    :type    k: numpy.array
    :param   k: 3x1 rotation axis for point p
    :type    d: number
    :param   d: desired displacement
    :rtype:  list of numbers
    :return: list of valid theta angles in radians    
    """
        
    # a = np.matmul(np.matmul(p,hat(k)),q)
    a = p.T @ (hat(k) @ q)
    # b = -np.matmul(p, np.matmul(hat(k),hat(k).dot(q)))
    b = -p.T @ hat(k) @ (hat(k) @ q)
    # c = np.subtract(d, (np.dot(p,q) -b))
    c = d - (p.T @ q + b)
    
    phi = np.arctan2(b, a)
    
    d = c / np.sqrt(a**2 + b**2)
    
    if d > 1:
        return []
        
    psi = np.arcsin(d)
    
    return [-phi+psi, -phi-psi+np.pi]

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
    Compute the least rotation angles that are required to rotate p
    
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
        sum1 = np.sum(np.abs(solution1)) # abs(solution1[0]) + abs(solution1[1]) + abs(solution1[2])
        sum2 = np.sum(np.abs(solution2)) # abs(solution2[0]) + abs(solution2[1]) + abs(solution1[2])
        
        # # Look only the min z rotation (i.e. select the min twist solution)
        # sum1 = abs(solution1[2])
        # sum2 = abs(solution1[2])
        
        # Compare the sums and select the one with the least rotation
        if sum1 <= sum2:
            return solution1
        else:
            return solution2

# ------------------------------------------------------------------------------

# Test case for approximated dlo inverse kinematics
def test_approximated_dlo_inv_kin_old1():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin_old(polyline)
    
    print("The following q must be all zeros") 
    print("q = ", q)
    
def test_approximated_dlo_inv_kin_old2():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin_old(polyline)
    
    print("The following q must be zeros in the first three elements,")
    print("then the next two elements must be (zero, pi/2),")
    print("and the remaining elements must be (zero, zero) series")
    print("q = ", q)
    
def test_approximated_dlo_inv_kin_old3():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 1]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin_old(polyline)
    
    print("The following q must be zeros in the first three elements")
    print("then the other elements must be [-pi/2  pi/4  pi/2  zero]")
    print("q = ", q)
    
def test_approximated_dlo_inv_kin_old4():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [0, 0, 0],
        [-1, 1, 0],
        [-1, 1, 1]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin_old(polyline)
    
    print("The following q must be zeros in the first three elements")
    print("then the other elements must be [-pi/2  -pi/4  pi/2  zero]")
    print("q = ", q)
    
def test_approximated_dlo_inv_kin_old5():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [1, -1, 1],
        [-1, -1, 1]])
    
    # pritn shape of polyline   
    print("polyline shape = ", polyline.shape)
    
    q = dlo_inv_kin_old(polyline)
    
    print("The following q must be [1,-1,1] in the first three elements")
    print("then the other elements must be [zero -pi/2]")
    print("q = ", q)

def test_approximated_dlo_inv_kin1():
    # define a DLO state assumed to be approximated from the original DLO state
    polyline = np.array([
        [1, -1, 1],
        [-1, -1, 1]])
    
    rot_mat = np.array([[0, 0, -1], 
                        [-1, 0, 0], 
                        [0, 1, 0]])
    
    rot_matrices = np.array([rot_mat])
    
    # pritn shape of polyline and rot_matrices
    print("polyline shape = ", polyline.shape)
    print("rot_matrices shape = ", rot_matrices.shape)
    
    q = dlo_inv_kin(polyline, rot_matrices)
    
    print("The following q must be [1,-1,1] in the first three elements")
    print("then the other elements must be [zero -pi/2 ??]")
    print("q = ", q)


# Run the tests
if __name__ == "__main__":
    # print("-----------------------------")
    # test_approximated_dlo_inv_kin_old1()
    # print("-----------------------------")
    # test_approximated_dlo_inv_kin_old2()
    # print("-----------------------------")
    # test_approximated_dlo_inv_kin_old3()
    # print("-----------------------------")
    # test_approximated_dlo_inv_kin_old4()
    # print("-----------------------------")
    # test_approximated_dlo_inv_kin_old5()
    print("-----------------------------")
    test_approximated_dlo_inv_kin1()
    