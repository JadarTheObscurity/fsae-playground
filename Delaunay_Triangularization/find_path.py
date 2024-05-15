from BowyerWatson import BowyerWatson, Edge
import numpy as np
import matplotlib.pyplot as plt



def intersection(edge1: Edge, edge2: Edge):
    """
    return the intersection of the ray(sp -> ep) and the edge. return none if no intersection
    """
    if abs(edge1.p2[0] - edge1.p1[0]) > 1e-5:
        m1 = (edge1.p2[1] - edge1.p1[1]) / (edge1.p2[0] - edge1.p1[0])
    else:
        m1 = (edge1.p2[1] - edge1.p1[1]) / 1e-5
    b1 = edge1.p1[1] - m1 * edge1.p1[0]

    if abs(edge2.p2[0] - edge2.p1[0]) > 1e-5:
        m2 = (edge2.p2[1] - edge2.p1[1]) / (edge2.p2[0] - edge2.p1[0])
    else:
        m2 = (edge2.p2[1] - edge2.p1[1]) / 1e-5
    b2 = edge2.p1[1] - m2 * edge2.p1[0]

    if m1 == m2:
        return None
    x = (b1 - b2) / (m2 - m1)
    y = m1 * x + b1
    
    # check if (x, y) in two segments
    min_x1 = min([edge1.p1[0], edge1.p2[0]])
    max_x1 = max([edge1.p1[0], edge1.p2[0]])
    min_y1 = min([edge1.p1[1], edge1.p2[1]])
    max_y1 = max([edge1.p1[1], edge1.p2[1]])
    min_x2 = min([edge2.p1[0], edge2.p2[0]])
    max_x2 = max([edge2.p1[0], edge2.p2[0]])
    min_y2 = min([edge2.p1[1], edge2.p2[1]])
    max_y2 = max([edge2.p1[1], edge2.p2[1]])
    if (
        (min_x1 <= x <= max_x1 and min_y1 <= y <= max_y1) and
        (min_x2 <= x <= max_x2 and min_y2 <= y <= max_y2)
    ):
        return (x, y)
    return None

def grade_path(path_points):
    total = 0
    for idx in range(len(path_points) - 2):
        first_vector = path_points[idx+1] - path_points[idx]
        second_vector = path_points[idx+2] - path_points[idx+1]
        # calculate the angle between first vector and second vector
        cos_theta = np.dot(first_vector, second_vector) / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
        theta = np.arccos(cos_theta)
        total += abs(theta)
    return total

def get_best_path_random(triangulation):
    edges = []
    for tri in triangulation:
        for e in tri.edges:
            if e not in edges:
                edges.append(e)
    mid_points = np.array([(e.p1 + e.p2) / 2 for e in edges])
    mid_points = mid_points[np.argsort(np.sum(np.square(mid_points), axis=1))]
    # randomly pick m integer from [0, n]
    n = len(mid_points)
    m = n // 3
    min_grade = 100
    best_path = None
    for i in range(1):
        idx = np.sort(np.random.choice(n, m, replace=False))
        path_points = mid_points[idx]
        path_points = np.concatenate(([[0, 0]], path_points))
        grade = grade_path(path_points)
        if grade < min_grade:
            best_path = path_points
            min_grade = grade
    return best_path

def get_best_path_greedy(triangulation):
    edges = []
    for tri in triangulation:
        for e in tri.edges:
            if e not in edges:
                edges.append(e)
    anchor = np.array([0, 0])
    search_length = 4
    direction = np.array([1, 0])
    paths = [anchor]
    for _ in range(10):
        search_edge = Edge(anchor, anchor + search_length * direction)
        min_d = search_length
        selected_edge = None
        for e in edges:
            p = intersection(search_edge, e)
            if p is not None:
                dis_to_anchor = np.linalg.norm(p - anchor)
                if dis_to_anchor < min_d:
                    min_d = dis_to_anchor
                    selected_edge = e

        if selected_edge is None:
            break
        new_anchor = (selected_edge.p1 + selected_edge.p2)/2
        direction = (new_anchor - anchor) / np.linalg.norm(new_anchor - anchor)
        anchor = new_anchor + 0.01 * direction
        paths.append(anchor)
    return np.array(paths)


if __name__ == '__main__':
    # points = np.array([[1, 1], [-1, 1], [1, 2], [-1, 2], [1.1, 3], [-0.9, 2.9], [1.7, 4.7], [-0.5, 4.3]])
    R = 6
    w = 3
    l = 3
    theta = 0
    dtheta = l / R
    left_l = (R - w/2) * dtheta
    right_l = (R + w/2) * dtheta
    left_cones = [[1, w/2]]
    right_cones = [[1, -w/2]]
    for i in range(4):
        left_cones += [[left_l * np.cos(theta) + left_cones[-1][0], left_l * np.sin(theta) + left_cones[-1][1]]]
        right_cones += [[right_l * np.cos(theta) + right_cones[-1][0], right_l * np.sin(theta) + right_cones[-1][1]]]
        theta += dtheta
    points = np.array(left_cones + right_cones)

    triangulation = BowyerWatson(points)
    paths = get_best_path_greedy(triangulation)
    plt.plot(-paths[:, 1], paths[:, 0])


    for t in triangulation:
        plt.plot([-t.p1[1], -t.p2[1], -t.p3[1], -t.p1[1]], [t.p1[0], t.p2[0], t.p3[0], t.p1[0]], 'b-')
    # plt.plot([p[0] for p in points], [p[1] for p in points], 'ro')
    plt.plot(-points[:, 1], points[:,0], 'ro')
    plt.axis("equal")
    # plt.scatter(mid_points[:,0], mid_points[:, 1])
    plt.show()