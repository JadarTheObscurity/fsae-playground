import numpy as np
"""
input: path - Nx2 np array
output: steering angle
"""

def intersection_circle_and_line(r, xc, yc, x1, y1, x2, y2):
    """
    return the intersection of the circle and the line. return none if no intersection
    x1, y1: start point
    x2, y2: end point
    ax + by + c = 0
    y = (y2-y1)/(x2-x1) x + b

    """
    a = y2 - y1
    b = -(x2-x1)
    c = - a * x1 - b * y1
    if a == 0:
        a = 1e-5

    A = (a**2 + b**2)
    B = 2 * b * c
    D = B**2 - 4 * A * (c**2 - a**2 * r**2)

    print(a, b, c,D)

    if D < 0:
        return None
    ya = (-B + np.sqrt(D)) / (2 * A) + yc
    xa = (-b * ya - c) / a + xc
    yb = (-B - np.sqrt(D)) / (2 * A) + yc
    xb = (-b * yb - c) / a + xc

    # Check if x1, y1, x2, y2 are in the line segment. Only return the points in the line segments
    intersections = []
    if (xa - x1) * (xa - x2) < 0 or (ya - y1) * (ya - y2) < 0:
        intersections.append([xa, ya])
    if (xb - x1) * (xb - x2) < 0 or (yb - y1) * (yb - y2) < 0:
        intersections.append([xb, yb])
    if len(intersections) == 0:
        return None

    # only leave the point which direction is the same as the path's direction
    path_dir = np.array([x2-x1, y2-y1])
    for p in intersections:
        p_dir = np.array([p[0] - xc, p[1] - yc])
        if np.dot(path_dir, p_dir) > 0:
            return p
    
    return intersections

def pure_pursuit(paths, r, L):
    """
    paths: 
    r: pure pursuit radius
    L: car length
    """
    print(paths)
    if paths.shape[0] < 2:
        return None

    p_follow = None
    for idx in range(len(paths) - 1):
        p_start = paths[idx]
        p_end = paths[idx+1]
        tmp_p_follow = intersection_circle_and_line(r, 0, 0, p_start[0], p_start[1], p_end[0], p_end[1])
        print(tmp_p_follow)
        if tmp_p_follow is not None:
            p_follow = tmp_p_follow
    if p_follow is None:
        return 0
    else:
        print(p_follow)
        return np.rad2deg(np.arctan2(2 * L * p_follow[1], p_follow[0]**2 + p_follow[1]**2-L**2))
    return 0

    
def test():
    # print(intersection_circle_and_line(5, 0, 0, 0, 0, 0, 10))
    # print(intersection_circle_and_line(5, 0, 0, 0, 0, 0, 2))
    # print(intersection_circle_and_line(5, 0, 10, 0, 0, 0, 10))
    # print(intersection_circle_and_line(7, 0, 10, 0, 0, 10, 10))
    print(intersection_circle_and_line(5, 0, 10, 6, -10, 6, 10))

if __name__ == '__main__':
    test()

