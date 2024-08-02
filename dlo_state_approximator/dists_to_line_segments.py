import numpy as np
 

def min_dist_to_polyline(points, polyline):
    """
    Compute the minimum distances from a set of points to a polyline in 3D space.
    This function utilizes the lineseg_dists function to compute the distances from the points to each segment of the polyline, and then determines the minimum distance for each point.

    Args:
        points: _description_
        polyline: _description_

    Returns:
        min_distances: _description_
    """
    
    num_segments = len(polyline) - 1
    min_distances = np.full(len(points), np.inf)

    for i in range(num_segments):
        a = polyline[i]
        b = polyline[i + 1]
        distances = lineseg_dists(points, a, b)
        min_distances = np.minimum(min_distances, distances)

    return min_distances

def lineseg_dists(p, a, b):
    """
    Compute the distances from a set ofpoints to a single line segment in 3D space.
    
    Obtained from https://stackoverflow.com/a/54442561/10993229
    
    Method:
        Take the dot-product of P - A with normalize(A - B) to obtain the signed parallel distance component s from A. Likewise with B and t.

        Take the maximum of these two numbers and zero to get the clamped parallel distance component. This will only be non-zero if the point is outside the "boundary" (Voronoi region?) of the segment.

        Calculate the perpendicular distance component as before, using the cross-product.

        Use Pythagoras to compute the required closest distance (gray line from P to A).

        The above is branchless and thus easy to vectorize with numpy.

    Args:
        p : Set of points to compute distances from.
        a : Segment start point.
        b : Segment end point.

    Returns:
        _type_: _description_
    """
    
    # Handle case where p is a single point, i.e. 1d array.
    p = np.atleast_2d(p)

    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, np.linalg.norm(c, axis=1))

# ------------------------------------------------------------------------------
# Test case for 3D points
def test_lineseg_dists():
    # Define points in 3D space
    points = np.array([
        [0, 4, 0],
        [0, -4, 0],
        [0, 0, 4],
        [0, 0, -4]
    ])
    
    #  Add some Translate in x-direction 
    points[:,0] += 4 + 3 

    # Define line segment in 3D space
    a = np.array([3, 0, 0])
    b = np.array([4, 0, 0])

    # Compute distances
    distances = lineseg_dists(points, a, b)

    # Print results
    print("Points:")
    print(points)
    print("Line segment endpoints:")
    print("a:", a)
    print("b:", b)
    print("Distances from points to line segment:")
    print(distances)

# Test case for min_dist_to_polyline
def test_min_dist_to_polyline():
    # Define points in 3D space
    points = np.array([
        [0, 4, 0],
        [0, -4, 0],
        [0, 0, 4],
        [0, 0, -4]
    ])
    
    #  Add some Translate in x-direction 
    points[:,0] += 0
    
    # Define polyline in 3D space
    polyline = np.array([
        [3, 0, 0],
        [4, 0, 0],
        [4, 4, 0]
    ])

    # Compute minimum distances
    min_distances = min_dist_to_polyline(points, polyline)

    # Print results
    print("Points:")
    print(points)
    print("Polyline vertices:")
    print(polyline)
    print("Minimum distances from points to polyline:")
    print(min_distances)

# Run the tests
if __name__ == "__main__":
    # test_lineseg_dists()
    
    test_min_dist_to_polyline()