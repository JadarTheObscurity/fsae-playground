import json
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("../Delaunay_Triangularization")
from BowyerWatson import BowyerWatson
from find_path import get_best_path_greedy

def fit_curve(points):
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)
    if x.shape[0] == 0:
        return None
    if x.shape[0] == 2:
        return [0] + np.polyfit(x.flatten(), y.flatten(), 1).tolist()
    return np.polyfit(x.flatten(), y.flatten(), 2)

with open("hist_cone_list.json", "r") as f:
    hist_cone_list = json.load(f)



fig = plt.figure(1)
ax1 = fig.add_subplot(111, aspect='equal')

cone_list_idx = 56
def plot(event):
    global cone_list_idx
    if event.key == 'j':
        cone_list_idx = max(cone_list_idx - 1, 0)
    sys.stdout.flush()
    if event.key == 'k':
        cone_list_idx = min(cone_list_idx + 1, len(hist_cone_list) - 1)
    print(f"Current index: {cone_list_idx}")
    
    ax1.cla()
    points = np.array(hist_cone_list[cone_list_idx])
    triangulation = BowyerWatson(points)
    filtered_triangulation = []
    for t in triangulation:
        # find the smallest angle in the triangle
        angles = []
        for i in range(3):
            first_vector = t.points[(i+1)%3] - t.points[i]
            second_vector = t.points[(i+2)%3] - t.points[(i+1)%3]
            cos_theta = np.dot(first_vector, second_vector) / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
            theta = np.arccos(cos_theta)
            angles.append(theta)
        min_angle = min(angles)
        if min_angle > np.pi / 5:
            filtered_triangulation.append(t)
    paths = get_best_path_greedy(filtered_triangulation)[1:]

    C = fit_curve(paths)
    # Sample 100 points from the curve
    x = points[:, 0].reshape(-1, 1)
    c_x = np.linspace(min(x), max(x), 100)
    c_y = C[0] * c_x**2 + C[1] * c_x + C[2]
    curve_points = np.hstack((c_x.reshape(-1, 1), c_y.reshape(-1, 1)))
    # plot the curve_points
    ax1.plot(-curve_points[:, 1], curve_points[:, 0], 'ro')

    ax1.plot(-paths[:, 1], paths[:, 0], 'bo')
    ax1.plot(-paths[:, 1], paths[:, 0], 'b-')
    for t in filtered_triangulation:
        ax1.plot([-t.p1[1], -t.p2[1], -t.p3[1], -t.p1[1]], [t.p1[0], t.p2[0], t.p3[0], t.p1[0]], 'b-')
    ax1.plot(-points[:, 1], points[:,0], 'ro')
    ax1.axis("equal")
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('key_press_event', plot)

plt.show()
exit()
points = np.array(hist_cone_list[99])

triangulation = BowyerWatson(points)
paths = get_best_path_greedy(triangulation)[1:]

C = fit_curve(paths)
# Sample 100 points from the curve
x = points[:, 0].reshape(-1, 1)
c_x = np.linspace(min(x), max(x), 100)
c_y = C[0] * c_x**2 + C[1] * c_x + C[2]
curve_points = np.hstack((c_x.reshape(-1, 1), c_y.reshape(-1, 1)))
# plot the curve_points
plt.plot(-curve_points[:, 1], curve_points[:, 0], 'ro')

plt.plot(-paths[:, 1], paths[:, 0], 'bo')
plt.plot(-paths[:, 1], paths[:, 0], 'b-')
for t in triangulation:
    plt.plot([-t.p1[1], -t.p2[1], -t.p3[1], -t.p1[1]], [t.p1[0], t.p2[0], t.p3[0], t.p1[0]], 'b-')
plt.plot(-points[:, 1], points[:,0], 'ro')
plt.axis("equal")
plt.show()