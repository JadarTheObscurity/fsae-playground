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

def bfs(triangulation):
    edges = []
    for tri in triangulation:
        for e in tri.edges:
            if e not in edges:
                edges.append(e)
    # find the mid point that is closest to (0, 0)
    mid_point = np.array([(e.p1 + e.p2) / 2 for e in edges])
    sort_idx = np.argsort(np.sum(np.square(mid_point), axis=1))
    init_edge = edges[sort_idx[0]]
    init_triangle = None
    for tri in triangulation:
        if init_edge in tri.edges:
            init_triangle = tri
            break
    # get the edge with the closest midpoint
    
    def greedy_search(start_triangle, start_edge, ttl):
        # Find the triangles contains start_edge that is not start_triangle
        next_triangles = []
        mid_point = (start_edge.p1 + start_edge.p2) / 2
        for tri in triangulation:
            if start_edge in tri.edges and tri is not start_triangle:
                next_triangles.append(tri)
        if ttl == 0 or len(next_triangles) == 0:
            return [start_edge]
        next_triangle = next_triangles[0]
        other_edges = [e for e in next_triangle.edges if e != start_edge]
        path1 = greedy_search(next_triangle, other_edges[0], ttl-1)
        path2 = greedy_search(next_triangle, other_edges[1], ttl-1)
        path = path1 if len(path1) > len(path2) else path2
        return [start_edge] + path
    
    init_other_edges = [e for e in init_triangle.edges if e != init_edge]
    path_edge_1 = greedy_search(init_triangle, init_other_edges[0], 10)
    path_edge_2 = greedy_search(init_triangle, init_other_edges[1], 10)
    path_edge = [init_edge] + (path_edge_1 if len(path_edge_1) > len(path_edge_2) else path_edge_2)
    path = [(e.p1 + e.p2) / 2 for e in path_edge]

    # modify the last point to determine the best path
    if len(path_edge) >= 2:
        last_edge = path_edge[-1]
        last_triangle = [t for t in triangulation if last_edge in t.edges][0]
        other_edges = [e for e in last_triangle.edges if e != last_edge and e != path_edge[-2]]
        path_edge_3 = path_edge[:-1] + other_edges
        path_2 = [(e.p1 + e.p2) / 2 for e in path_edge_3]
        path_points = [grade_path(path), grade_path(path_2)]
        best_idx = np.argmin(path_points)
        return path if best_idx == 0 else path_2
         
    return path
    

def get_best_path_greedy(triangulation):
    edges = []
    for tri in triangulation:
        for e in tri.edges:
            if e not in edges:
                edges.append(e)
    # pick the longest one
    # anchor = np.array([0, 0])
    mid_point = [(e.p1 + e.p2) / 2 for e in edges]
    # find the mid point that is closest to (0, 0)
    mid_point = np.array(mid_point)
    sort_idx = np.argsort(np.sum(np.square(mid_point), axis=1))
    init_edge = edges[sort_idx[0]]
    init_mid = (init_edge.p1 + init_edge.p2) / 2
    init_triangle = None
    for tri in triangulation:
        if init_edge in tri.edges:
            init_triangle = tri
            break
    # get the edge with the closest midpoint
    anchors = []
    directions = []
    for e in init_triangle.edges:
        if e is not init_edge:
            mid = (e.p1 + e.p2) / 2
            dir = (mid - init_mid) / np.linalg.norm(mid - init_mid)
            directions.append(dir)
            anchors.append(mid + 0.01 * dir)
    
    def greedy_search(anchor, direction, search_length):
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
        return paths
    search_length = 5
    path1 = [[0, 0], init_mid] + greedy_search(anchors[0], directions[0], search_length)
    path2 = [[0, 0], init_mid] + greedy_search(anchors[1], directions[1], search_length)
    if len(path1) > len(path2):
        return np.array(path1)
    else:
        return np.array(path2)


if __name__ == '__main__':
    # points = np.array([[1, 1], [-1, 1], [1, 2], [-1, 2], [1.1, 3], [-0.9, 2.9], [1.7, 4.7], [-0.5, 4.3]])
    R = 15
    w = 3
    l = 3
    theta = np.deg2rad(5)
    dtheta = l / R
    left_l = (R - w/2) * dtheta
    right_l = (R + w/2) * dtheta
    left_cones = [[1 + w/2 * np.sin(theta),  w/2 * np.cos(theta)]]
    right_cones = [[1 - w/2 * np.sin(theta),-w/2 * np.cos(theta)]]
    # rotate left_cones and right_cones by 15 degrees
    for i in range(4):
        left_cones += [[left_l * np.cos(theta) + left_cones[-1][0], left_l * np.sin(theta) + left_cones[-1][1]]]
        right_cones += [[right_l * np.cos(theta) + right_cones[-1][0], right_l * np.sin(theta) + right_cones[-1][1]]]
        theta += dtheta
    points = np.array(left_cones + right_cones)

    import os
    import sys
    print(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), "../PurePursuit"))
    from PurePursuit import pure_pursuit

    triangulation = BowyerWatson(points)
    paths = get_best_path_greedy(triangulation)

    follow_radius = 4
    print(pure_pursuit(paths, follow_radius, 1))


    # plt.plot(-paths[:, 1], paths[:, 0])
    # for t in triangulation:
    #     plt.plot([-t.p1[1], -t.p2[1], -t.p3[1], -t.p1[1]], [t.p1[0], t.p2[0], t.p3[0], t.p1[0]], 'b-')
    # plt.plot([p[0] for p in points], [p[1] for p in points], 'ro')
    plt.plot(-points[:, 1], points[:,0], 'ro')
    plt.axis("equal")
    # plt.scatter(mid_points[:,0], mid_points[:, 1])
    plt.show()