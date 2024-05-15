from BowyerWatson import BowyerWatson, Edge
import numpy as np
import matplotlib.pyplot as plt



def intersection(sp, ep, edge):
    """
    return the intersection of the ray(sp -> ep) and the edge. return none if no intersection
    """

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

def get_best_path(triangulation):
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
    for i in range(500):
        idx = np.sort(np.random.choice(n, m, replace=False))
        path_points = mid_points[idx]
        path_points = np.concatenate(([[0, 0]], path_points))
        grade = grade_path(path_points)
        if grade < min_grade:
            best_path = path_points
            min_grade = grade
    return best_path



if __name__ == '__main__':
    points = np.array([[1, 1], [-1, 1], [1, 2], [-1, 2], [1.1, 3], [-0.9, 2.9], [1.7, 4.7], [-0.5, 4.3]])
    triangulation = BowyerWatson(points)
    best_path = get_best_path(triangulation)
    # plot path_points
    plt.plot([best_path[0, 0]] + [p[0] for p in best_path] + [best_path[-1, 0]], [best_path[0, 1]] + [p[1] for p in best_path] + [best_path[-1, 1]], 'r-')


    for t in triangulation:
        plt.plot([t.p1[0], t.p2[0], t.p3[0], t.p1[0]], [t.p1[1], t.p2[1], t.p3[1], t.p1[1]], 'b-')
    plt.plot([p[0] for p in points], [p[1] for p in points], 'ro')
    # plt.scatter(mid_points[:,0], mid_points[:, 1])
    plt.show()