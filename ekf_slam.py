# This program demonstrate EKF SLAM using range and bearing measurements
# The main purpose of this program is to visualize how different matrix changes
# We assume EKF with known correspondence to make our life easier

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


"""
x: state variable
P: state covariance matrix
g: motion model
h: sensor model
G: Jacobian of g
H: Jacobian of h
R: uncertainity of control
Q: uncertainity of sensor
Algorithm EKF_SLAM(x, P, u, z):
    # Predition
    x[t] = g(x[t-1], u[t])
    P = G @ P @ G.T + R

    # Update
    K = P @ H.T @ (H @ P @ H.T + Q) ^ -1
    x[t] = x[t] + K @ (z - h(x[t]))
    P = (I - K @ H) @ P
    return x[t], P
"""

def sensor_observation(x, map):
    """
    Return the landmarks robot would observed within range r
    """
    observation_radius = 5
    observations = []
    car_x = x[0]
    car_y = x[1]
    car_theta = x[2]
    for landmark in map:
        r = np.sqrt((landmark[0] - car_x) ** 2 + (landmark[1] - car_y) ** 2)
        if r < observation_radius:
            theta = np.arctan2(landmark[1] - car_y, landmark[0] - car_x) - car_theta
            observations.append((r, theta))
    return observations



def motion_model(x, u, dt=0.01):
    d_theta = u[1] * dt
    turn_radius = u[0] / u[1]
    dx = -turn_radius * np.sin(x[2]) + turn_radius * np.sin(x[2] + d_theta)
    dy = turn_radius * np.cos(x[2]) - turn_radius * np.cos(x[2] + d_theta)

    F = np.eye(3)
    F = np.concatenate((F, np.zeros((3, 2 * num_landmark))), axis=1)

    G = np.array([
        [0, 0, turn_radius * np.cos(x[2]) - turn_radius * np.cos(x[2] + d_theta)],
        [0, 0, turn_radius * np.sin(x[2]) - turn_radius * np.sin(x[2] + d_theta)],
        [0, 0, 0]
    ])
    G = np.eye(3 + 2 * num_landmark) + F.T @ G @ F

    x[0] += dx
    x[1] += dy
    x[2] += d_theta
    return x, G



def sensor_model(x, landmark):
    delta = np.array([landmark[0] - x[0], landmark[1] - x[1]])
    q = np.dot(delta.T, delta).flatten()[0]
    q_sqrt = np.sqrt(q)
    theta = np.arctan2(delta[1], delta[0]) - x[2]
    if q < 1e-6: q=1e-6
    
    z_hat = np.array([q_sqrt, theta])
    H = 1 / q * np.array([
        [-q_sqrt * delta[0], -q_sqrt * delta[1], 0, q_sqrt * delta[0], q_sqrt * delta[1]],
        [delta[1], -delta[0], -q, -delta[1], delta[0]]
    ])
    return z_hat, H


def motion_predition(x, P, u, R):
    x, G = motion_model(x, u)

    F = np.eye(3)
    F = np.concatenate((F, np.zeros((3, 2 * num_landmark))), axis=1)
    P = G @ P @ G.T + F.T @ R @ F
    return x, P

def angle_clip(angle):
    angle %= 2 * np.pi
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def find_closest_landmark(x, z):
    d = []
    map = []
    for i in range(num_landmark):
        mx = x[3 + 2 * i]
        my = x[3 + 2 * i + 1]
        map.append((mx, my))
    for mx, my in map:
        z_hat, H = sensor_model(x, [mx, my])
        z_diff = z - z_hat
        z_diff[1] = angle_clip(z_diff[1])
        d.append(np.sum(np.square(z_diff)))
    closest_idx = np.argmin(np.array(d))
    return map[closest_idx], closest_idx



def sensor_update(x, P, z, Q):
    # Find the correspondent landmark on the map
    closest_lmk, lmk_idx = find_closest_landmark(x, z)

    # Calculate z_hat
    z_hat, H = sensor_model(x, closest_lmk)
    z_diff = z - z_hat
    z_diff[1] = angle_clip(z_diff[1])
    # F
    F1 = np.concatenate((np.eye(3), np.zeros((3, 2 * num_landmark))), axis=1)
    F2 = np.concatenate((np.zeros((2, 3 + 2 * lmk_idx)), np.eye(2), np.zeros((2, 2 * (num_landmark - lmk_idx - 1)))), axis=1)
    F = np.concatenate((F1, F2), axis=0)
    H = H @ F
    # Update
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Q)
    x = x + K @ z_diff 
    P = (np.eye(x.shape[0]) - K @ H) @ P
    print(z_diff)
    return x, P

def ekf_slam(x, P, u, observations):
    R = 0.1 * np.eye(3)
    Q = 0.01 * np.eye(2)
    x, P = motion_predition(x, P, u, R)
    for z in observations:
        x, P = sensor_update(x, P, z, Q)
    
    return x, P

global_map = np.array([
    [7, 0],
    [-7, 0],
    [0, 7],
    [0, -7],
    [7, 7],
    [7, -7],
    [-7, 7],
    [-7, -7],
])

x_gt = np.array([0, -10, 0], dtype=np.float32)
x_raw = x_gt.copy()
x = x_gt.copy()

for m in global_map:
    x = np.concatenate((x, np.array([m[0], m[1]])))

num_landmark = int((x.shape[0]-3) // 2)
landmark_sigma = 100

init_P = np.eye((3))

# Enlarge P from 3x3 to 3+2N x 3+2N
P = np.concatenate((init_P, np.zeros((3, 2 * num_landmark))), axis=1)
P = np.concatenate((P, np.zeros((2 * num_landmark, 3 + 2 * num_landmark))), axis=0)
for i in range(3, 3 + 2 * num_landmark):
    P[i][i] = landmark_sigma


print(x)
print(P)
path_gt_hist = np.empty((0, 3))
path_raw_hist = np.empty((0, 3))
path_hist = np.empty((0, 3))

ims = []
fig = plt.figure(figsize=(10, 5))
for i in range(100):
    u_gt = np.array([10, 10/10])
    u = u_gt + 1 * (np.random.random(2) - 0.3)

    observations = sensor_observation(x_gt, global_map)
    x_gt, _ = motion_model(x_gt, u_gt, dt=0.01)
    x_raw, _ = motion_model(x_raw, u, dt=0.01)
    x, P = ekf_slam(x, P, u, observations)
    print(f"x pose: {x[:3]}")

    path_gt_hist = np.vstack((path_gt_hist, x_gt[:3].reshape(1, 3)))
    path_raw_hist = np.vstack((path_raw_hist, x_raw[:3].reshape(1, 3)))
    path_hist = np.vstack((path_hist, x[:3].reshape(1, 3)))

    if (i > 300 and path_gt_hist.shape[0] > 300):
        path_gt_hist = path_gt_hist[-300:]
        path_raw_hist = path_raw_hist[-300:]
        path_hist = path_hist[-300:]

    # Animation

    # Subplot 1: Original plot
    plt.clf()
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    plt.gcf().canvas.mpl_connect('key_release_event', 
        lambda event: [exit(0) if event.key == 'escape' else None])

    # Plot landmark
    for z in observations:
        theta = z[1] + x[2]
        mx = x[0] + z[0] * np.cos(theta)
        my = x[1] + z[0] * np.sin(theta)
        # plot (mx, my) as a X point and a line from (x[0], x[1]) to (mx, my) in blue line
        plt.plot([x[0], mx], [x[1], my], '-b')
        plt.plot(mx, my, 'x')

        # Theory
        closest_lmk, lmk_idx = find_closest_landmark(x, z)
        z_hat, H = sensor_model(x, closest_lmk)
        theta = z_hat[1] + x[2]
        mx = x[0] + z_hat[0] * np.cos(theta)
        my = x[1] + z_hat[0] * np.sin(theta)
        # plot (mx, my) as a X point and a line from (x[0], x[1]) to (mx, my) in blue line
        plt.plot([x[0], mx], [x[1], my], '-b')
        plt.plot(mx, my, 'o')

    # Plot path
    plt.scatter(global_map[:, 0], global_map[:, 1], c='k', marker='*')
    plt.scatter(path_gt_hist[:, 0], path_gt_hist[:, 1], c='k', marker='.')
    plt.scatter(path_raw_hist[:, 0], path_raw_hist[:, 1], c='b', marker='.')
    plt.scatter(path_hist[:, 0], path_hist[:, 1], c='r', marker='.')
    # plt.plot(global_map[:, 0], global_map[:, 1], '*k')
    # plt.plot(path_gt_hist[:, 0], path_gt_hist[:, 1], '.k')
    # plt.plot(path_raw_hist[:, 0], path_raw_hist[:, 1], '.b')
    # plt.plot(path_hist[:, 0], path_hist[:, 1], '.r')
    # Plot sensor reading

    plt.axis("equal")
    plt.grid(True)

    # Subplot 2: Heatmap of P
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
    plt.cla()
    plt.matshow(P, cmap='hot', fignum=0)  # Display P as a heatmap using matshow
    plt.colorbar()  # Add a colorbar to the heatmap
    plt.title('Heatmap of P')  # Add a title to the heatmap
    if i % 5 == 0:
        plt.savefig(f"./__tmp_pic/{i}.png")
    plt.pause(0.0001)
    # while not plt.waitforbuttonpress():  # Wait for 'a' key press to continue
    #     pass