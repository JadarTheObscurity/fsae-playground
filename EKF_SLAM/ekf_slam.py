# This program demonstrate EKF SLAM using range and bearing measurements
# The main purpose of this program is to visualize how different matrix changes
# We assume EKF with known correspondence to make our life easier

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
np.set_printoptions(precision=4, suppress=True)

# EKF SLAM


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
        theta = np.arctan2(landmark[1] - car_y, landmark[0] - car_x) - car_theta
        theta = angle_clip(theta)
        if r < observation_radius and abs(theta) < np.pi / 2:
            r += 0.04 * np.random.random(1)[0]
            theta += angle_clip(0.1 * (np.random.random() - 0.5))
            observations.append((r, theta))
    return observations



def motion_model(x, u, dt=0.01):
    num_landmark = get_num_landmark(x)
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

def inverse_sensor_model(x, z):
    theta = z[1] + x[2]
    mx = x[0] + z[0] * np.cos(theta)
    my = x[1] + z[0] * np.sin(theta)
    return mx, my


def motion_predition(x, P, u, R):
    num_landmark = get_num_landmark(x)
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

def find_closest_landmark(x, P, z, Q, alpha=30.0):
    d = []
    map = []
    num_landmark = get_num_landmark(x) 
    for i in range(num_landmark):
        mx = x[3 + 2 * i]
        my = x[3 + 2 * i + 1]
        map.append((mx, my))
     # Include current observation

    for i, (mx, my) in enumerate(map):
        z_hat, H = sensor_model(x, [mx, my])
        z_diff = z - z_hat
        z_diff[1] = angle_clip(z_diff[1])
        # Calculate the mahalanobis distance
        F1 = np.concatenate((np.eye(3), np.zeros((3, 2 * num_landmark))), axis=1)
        F2 = np.concatenate((np.zeros((2, 3 + 2 * i)), np.eye(2), np.zeros((2, 2 * (num_landmark - i - 1)))), axis=1)
        F = np.concatenate((F1, F2), axis=0)
        H = H @ F
        Psi = H @ P @ H.T + Q
        md = z_diff.T @ np.linalg.inv(Psi) @ z_diff

        # Calculate the mahalanobis
        d.append(md)
    d.append(alpha)
    map.append(inverse_sensor_model(x, z))
    closest_idx = np.argmin(np.array(d))
    if closest_idx == num_landmark:
        print(f"Add new landmark: {sorted(d)}")
    return map[closest_idx], closest_idx



def sensor_update(x, P, z, Q):
    # Find the correspondent landmark on the map
    closest_lmk, lmk_idx = find_closest_landmark(x, P, z, Q)
    num_landmark = get_num_landmark(x)
    # If found a new landmark
    if lmk_idx == num_landmark:
        x = np.concatenate((x, np.array(closest_lmk)))
        P = expand_P(P, landmark_sigma)
        num_landmark += 1
        print(f"Add new landmark, {num_landmark}")

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
    return x, P

def ekf_slam(x, P, u, observations):
    x, P = motion_predition(x, P, u, R)
    for z in observations:
        x, P = sensor_update(x, P, z, Q)
    
    return x, P

def get_num_landmark(x):
    return  int((x.shape[0]-3) // 2)

def expand_P(P, sigma):
    P_dim = P.shape[0]
    P = np.concatenate((P, np.zeros((P_dim, 2))), axis=1)
    P = np.concatenate((P, np.zeros((2, P_dim + 2))), axis=0)
    P[P_dim:, P_dim:] = sigma *  np.eye(2)
    return P




# Create global map
track_width = 3.0
track_radius = 6.0
cone_number = 10
global_map = np.empty((0, 2))
for t in np.linspace(0, 2 * np.pi, cone_number, endpoint=False):
    inner_r = track_radius - track_width / 2
    outer_r = track_radius + track_width / 2
    global_map = np.vstack((global_map, np.array([inner_r * np.cos(t), inner_r * np.sin(t)])))
    global_map = np.vstack((global_map, np.array([outer_r * np.cos(t), outer_r * np.sin(t)])))

# print(global_map.shape)
# exit()
x_gt = np.array([0, -track_radius, 0], dtype=np.float32)
x_raw = x_gt.copy()
x = x_gt.copy()

# for m in global_map:
#     x = np.concatenate((x, np.array([m[0], m[1]])))

num_landmark = int((x.shape[0]-3) // 2)
landmark_sigma = 0.1
R = 0.01 * np.eye(3)
Q = 0.01 * np.eye(2)

# Create P 3x3 to 3+2N x 3+2N
P = np.eye(3)
for i in range(num_landmark):
    P = expand_P(P, landmark_sigma)



path_gt_hist = np.empty((0, 3))
path_raw_hist = np.empty((0, 3))
path_hist = np.empty((0, 3))

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
fig.canvas.mpl_connect('key_release_event', 
    lambda event: [exit(0) if event.key == 'escape' else None])
im = axs[1].matshow(P, cmap='hot')  # Display P as a heatmap using matshow
fig.colorbar(im, ax=axs[1])  # Add a colorbar to the heatmap

for iter in range(501):
    u_gt = np.array([10.0, 10 / track_radius])
    u = u_gt + 1 * (np.random.random(2) - np.array([0.5, 0.3]))

    observations = sensor_observation(x_gt, global_map)
    x_gt, _ = motion_model(x_gt, u_gt, dt=0.01)
    x_raw, _ = motion_model(x_raw, u, dt=0.01)
    x, P = ekf_slam(x, P, u, observations)

    path_gt_hist = np.vstack((path_gt_hist, x_gt[:3].reshape(1, 3)))
    path_raw_hist = np.vstack((path_raw_hist, x_raw[:3].reshape(1, 3)))
    path_hist = np.vstack((path_hist, x[:3].reshape(1, 3)))

    if (iter > 300 and path_gt_hist.shape[0] > 300):
        path_gt_hist = path_gt_hist[-300:]
        path_raw_hist = path_raw_hist[-300:]
        path_hist = path_hist[-300:]

    # Animation
    # Create a figure and a 1x2 subplot grid
    # fig.clf()
    axs[0].cla()
    axs[1].cla()
    
    # Subplot 1: Map and path
    axs[0].scatter(global_map[:, 0], global_map[:, 1], c='k', marker='*')
    self_map = x[3:]
    if self_map.shape[0] > 0:
        axs[0].scatter(self_map[::2], self_map[1::2], c='r', marker='*')

    # Plot landmark
    for z in observations:
        mx, my = inverse_sensor_model(x, z)
        # plot (mx, my) as a X point and a line from (x[0], x[1]) to (mx, my) in blue line
        axs[0].plot([x[0], mx], [x[1], my], '-b')
        axs[0].plot(mx, my, 'x')
    
    # Plot car uncertainty
    cov = P[:2, :2]
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 1]))
    width, height = 2 * np.sqrt(eigvals)
    ellipse = Ellipse((x[0], x[1]), width, height, angle=angle, edgecolor='r', fc='none')
    axs[0].add_patch(ellipse)

    # Plog landmark uncertainity
    if self_map.shape[0] > 0:
        for i in range(self_map.shape[0] // 2):
            mx, my = self_map[2*i:2*i+2]
            cov = P[3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigvecs[:, 1]))
            width, height = 2 * np.sqrt(eigvals)
            ellipse = Ellipse((mx, my), width, height, angle=angle, edgecolor='r', fc='none')
            axs[0].add_patch(ellipse)

    
    # Plot path
    axs[0].scatter(path_gt_hist[:, 0], path_gt_hist[:, 1], c='b', marker='.')
    axs[0].scatter(path_raw_hist[:, 0], path_raw_hist[:, 1], c='k', marker='.')
    axs[0].scatter(path_hist[:, 0], path_hist[:, 1], c='r', marker='.')
    
    axs[0].axis("equal")
    axs[0].grid(True)
    
    # Subplot 2: Heatmap of P
    im = axs[1].matshow(P, cmap='hot')  # Display P as a heatmap using matshow
    # fig.colorbar(im, ax=axs[1])  # Add a colorbar to the heatmap
    axs[1].set_title('Heatmap of P')  # Add a title to the heatmap
    
    if iter % 5 == 0:
        print(f"Save as {iter:04}.png")
        fig.savefig(f"./__tmp_pic/{iter:04}.png")
    plt.pause(0.001)
    """
    # Subplot 1: Original plot
    plt.clf()
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    plt.gcf().canvas.mpl_connect('key_release_event', 
        lambda event: [exit(0) if event.key == 'escape' else None])

    # Plot landmark
    for z in observations:
        mx, my = inverse_sensor_model(x, z)
        # plot (mx, my) as a X point and a line from (x[0], x[1]) to (mx, my) in blue line
        plt.plot([x[0], mx], [x[1], my], '-b')
        plt.plot(mx, my, 'x')

        # # Theory
        # closest_lmk, lmk_idx = find_closest_landmark(x, z)
        # z_hat, H = sensor_model(x, closest_lmk)
        # theta = z_hat[1] + x[2]
        # mx = x[0] + z_hat[0] * np.cos(theta))3)3)3)3
        # my = x[1] + z_hat[0] * np.sin(theta)
        # # plot (mx, my) as a X point and a line from (x[0], x[1]) to (mx, my) in blue line
        # plt.plot([x[0], mx], [x[1], my], '-b')
        # plt.plot(mx, my, 'o')

        # Create a figure and a 1x2 subplot grid
    # Plot map
    plt.scatter(global_map[:, 0], global_map[:, 1], c='k', marker='*')
    self_map = x[3:]
    if self_map.shape[0] > 0:
        plt.scatter(self_map[::2], self_map[1::2], c='r', marker='*')
    # Plot car uncertainity
    # Covariance matrix
    cov = np.array([[4.0, 1.3], [1.3, 3.5]])
    
    # Calculate the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Calculate the angle of rotation and the scale of the x and y axes
    angle = np.degrees(np.arctan2(*eigvecs[:, 1]))
    width, height = 2 * np.sqrt(eigvals)
    
    # Create the ellipse
    ellipse = Ellipse((x[0], x[1]), width, height, angle=angle)
    ax = plt.gca()
    ax.add_patch(ellipse)

    # Plot the ellipse
    plt.add_patch(ellipse)
    # Plot path
    plt.scatter(path_gt_hist[:, 0], path_gt_hist[:, 1], c='k', marker='.')
    plt.scatter(path_raw_hist[:, 0], path_raw_hist[:, 1], c='b', marker='.')
    plt.scatter(path_hist[:, 0], path_hist[:, 1], c='r', marker='.')

    plt.axis("equal")
    plt.grid(True)

    # Subplot 2: Heatmap of P
    # plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
    # plt.cla()
    # plt.matshow(P, cmap='hot', fignum=0)  # Display P as a heatmap using matshow
    # plt.colorbar()  # Add a colorbar to the heatmap
    # plt.title('Heatmap of P')  # Add a title to the heatmap
    # if i % 5 == 0:
        # plt.savefig(f"./__tmp_pic/{i:04}.png")
    plt.pause(0.0001)
    """