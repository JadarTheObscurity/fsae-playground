import json
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("../Delaunay_Triangularization")
from BowyerWatson import filtered_BowyerWatson
from find_path import get_best_path_greedy, bfs
from scipy.spatial.transform import Rotation

def fit_curve(points):
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)
    if x.shape[0] == 0:
        return None
    if x.shape[0] == 2:
        return [0, 0] + np.polyfit(x.flatten(), y.flatten(), 1).tolist()
    elif x.shape[0] >= 3:
        return [0] + np.polyfit(x.flatten(), y.flatten(), 2).tolist()
    # else:
    #     return np.polyfit(x.flatten(), y.flatten(), 3).tolist()

def match_points(points, prev_points, threshold=2):
    # match points -> prev_points
    now_prev_match = []
    now_prev_dist = []
    prev_now_match = []
    if len(points) == 0 or len(prev_points) == 0:
        return []
    for p in points:
        now_prev_match.append(np.argmin(np.sum(np.square(p[None, :] - prev_points), axis=1)))
        now_prev_dist.append(np.min(np.sum(np.square(p[None, :] - prev_points), axis=1)))
    for p in prev_points:
        prev_now_match.append(np.argmin(np.sum(np.square(p[None, :] - points), axis=1)))
    matched_pair = []
    for i in range(len(now_prev_match)):
        if i == prev_now_match[now_prev_match[i]] and now_prev_dist[i] < threshold**2:
            matched_pair.append((i, now_prev_match[i]))
    return matched_pair

def draw_match_points(curr_points, prev_points, ax):
    matched_pair = match_points(curr_points, prev_points)
    ax.cla()
    ax.plot([0], [ 0], 'o', color='#000000')
    ax.plot(-curr_points[:, 1], curr_points[:,0], 'ro')
    ax.plot(-prev_points[:, 1], prev_points[:,0], 'o', color='#800000')
    ax.axis("equal")
    for pair in matched_pair:
        p1 = curr_points[pair[0]]
        p2 = prev_points[pair[1]]
        ax.plot([-p1[1], -p2[1]], [p1[0], p2[0]], 'b-')

def icp(curr_points, prev_points):
    matched_pair = match_points(curr_points, prev_points)
    if len(matched_pair) == 0: 
        return prev_points
    matched_curr_points = curr_points[[pair[0] for pair in matched_pair]]
    matched_prev_points = prev_points[[pair[1] for pair in matched_pair]]
    curr_points_center = np.mean(matched_curr_points, axis=0)
    prev_points_center = np.mean(matched_prev_points, axis=0)

    # Shift
    offsets = []
    for pair in matched_pair:
        p1 = curr_points[pair[0]]
        p2 = prev_points[pair[1]]
        offsets.append(p1 - p2)
    weights = 1 / (np.sum(np.square(offsets), axis=1) + 0.1)
    ts = np.average(offsets, weights=weights, axis=0)
    print(ts, weights)
    # ts = curr_points_center - prev_points_center
    prev_points = prev_points + ts

    # Rotate
    angles = []
    for i in range(len(matched_pair)):
        curr_vec = matched_curr_points[i] - curr_points_center 
        prev_vec = matched_prev_points[i] - prev_points_center 
        angle = np.arctan2(curr_vec[1], curr_vec[0]) - np.arctan2(prev_vec[1], prev_vec[0])
        if angle > np.pi: angle -= 2 * np.pi
        if angle < -np.pi: angle += 2 * np.pi
        angles.append(angle)
    # TODO: Regect outliers
    if len(angles) > 1:
        angle_means = []
        angle_stds = []
        for i in range(len(angles)):
            angle_mean = np.mean(angles[0:i] + angles[i+1:])
            angle_std = np.std(angles[0:i] + angles[i+1:])
            angle_means.append(angle_mean)
            angle_stds.append(angle_std)
        angle = angle_means[np.argmin(angle_stds)]
    else:
        angle = angles[0]
    print(angle, angles)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return  (R @ (prev_points - prev_points_center).T).T + prev_points_center

def plot(event):
    global cone_list_idx
    global accum_points
    if event.key == 'j':
        cone_list_idx = max(cone_list_idx - 1, 0)
    sys.stdout.flush()
    if event.key == 'k':
        cone_list_idx = min(cone_list_idx + 1, len(hist_cone_list) - 1)

    print(f"Current index: {cone_list_idx}")
    print(f"dt: {hist_timestamp[cone_list_idx] - hist_timestamp[cone_list_idx - 1]}")
    print(f"Speed: {hist_speed[cone_list_idx]}")
    print(f"Steering: {hist_steering[cone_list_idx]}")
    curr_points = np.array(hist_cone_list[cone_list_idx])
    dt = hist_timestamp[cone_list_idx] - hist_timestamp[cone_list_idx - 1]
    # Get the speed and steering from previous timestamp
    speed = hist_speed[cone_list_idx-1]
    steering = hist_steering[cone_list_idx-1]
    ds = speed * dt
    theta = ds / 2.0 * np.tan(steering)
    prior_motion = np.array([ds * np.cos(theta), ds * np.sin(theta)])

    accum_points -= prior_motion
    draw_match_points(curr_points, accum_points, axs[0])
    for i in range(4):
        accum_points = icp(curr_points, accum_points)
    # Add new points to the accumulated points if the distance is large
    for p in curr_points:
        if np.min(np.sum(np.square(p[None, :] - accum_points), axis=1)) > 0.5**2:
            accum_points = np.concatenate([accum_points, p[None, :]])

    # draw_match_points(curr_points, accum_points, axs[1])

    distances = np.sqrt(np.sum(np.square(accum_points), axis=1))
    # Filter points that are approximately 5 units apart from the origin
    desired_distance = 10
    accum_points = accum_points[(distances < desired_distance)]

    # draw_match_points(curr_points, partial_accum_points, axs[3])
    axs[1].cla()
    axs[1].plot([0], [ 0], 'o', color='#000000')
    axs[1].plot(-accum_points[:, 1], accum_points[:,0], 'o', color='#800000')
    axs[1].plot(-curr_points[:, 1], curr_points[:,0], 'ro')
    axs[1].axis("equal")
    axs[1].set_xlim(-10, 10)
    axs[1].set_ylim(-10, 10)

    triangulation = filtered_BowyerWatson(accum_points)
    paths = np.array(bfs(triangulation))
    C = fit_curve(paths)
    # Sample 100 points from the curve
    c_x = np.linspace(-10, 10, 100)
    c_y = C[0] * c_x**3 + C[1] * c_x**2 + C[2] * c_x + C[3]
    curve_points = np.hstack((c_x.reshape(-1, 1), c_y.reshape(-1, 1)))
    # plot the curve_points

    axs[2].cla()
    axs[2].plot(-curve_points[:, 1], curve_points[:, 0], 'ro')
    axs[2].plot([0], [ 0], 'o', color='#000000')
    axs[2].plot(-accum_points[:, 1], accum_points[:,0], 'ro')
    axs[2].plot(-paths[:, 1], paths[:, 0], 'bo')
    axs[2].plot(-paths[:, 1], paths[:, 0], 'b-')
    for t in triangulation:
        axs[2].plot([-t.p1[1], -t.p2[1], -t.p3[1], -t.p1[1]], [t.p1[0], t.p2[0], t.p3[0], t.p1[0]], 'b-')
    axs[2].axis("equal")
    axs[2].set_xlim(-10, 10)
    axs[2].set_ylim(-10, 10)

    fig.canvas.draw()
    # plt.savefig(f'__tmp_pic/frame_{cone_list_idx:03d}.jpeg')


if __name__ == '__main__':
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs = [ax1, ax2, ax3]

    with open("2024_06_07_08_12_log.json", "r") as f:
        log_file = json.load(f)

    hist_timestamp = [log["timestamp"] for log in log_file]
    hist_speed = [log["speed"] for log in log_file]
    hist_steering = [log["steering_angle"] for log in log_file]
    hist_cone_list = [log["cones"] for log in log_file]
    cone_list_idx = 10
    print(len(hist_cone_list))
    accum_points = np.array(hist_cone_list[cone_list_idx])
    cid = fig.canvas.mpl_connect('key_press_event', plot)
    plt.show()