# Extended Kalman Filter SLAM

This Python script implements an Extended Kalman Filter (EKF) for Simultaneous Localization and Mapping (SLAM).

## Extended Kalman Filter (EKF)

The Extended Kalman Filter (EKF) is a variant of the Kalman Filter which linearizes about an estimate of the current mean and covariance. In the context of the EKF SLAM algorithm, the EKF is used to estimate the robot's pose and the position of landmarks in the environment.

The EKF algorithm can be summarized as follows:

1. **Initialization:**

    Initialize the state estimate $\mathbf{x}$ and the covariance matrix $\mathbf{P}$.

2. **Prediction:**

    Predict the state estimate and the covariance matrix based on the motion model and control inputs.

    $$
    \mathbf{x}_{pred} = f(\mathbf{x}_{t-1}, \mathbf{u}_t) \\
    \mathbf{P}_{pred} = \mathbf{G}_t \mathbf{P}_{t-1} \mathbf{G}_t^T + \mathbf{Q}_t
    $$

    where $\mathbf{F}_t$ is the Jacobian of the motion model with respect to the state, and $\mathbf{Q}_t$ is the process noise covariance matrix.

3. **Update:**

    Update the state estimate and the covariance matrix based on the measurement model and the observed measurements.

    $$
    \mathbf{K}_t = \mathbf{P}_{pred} \mathbf{H}_t^T (\mathbf{H}_t \mathbf{P}_{pred} \mathbf{H}_t^T + \mathbf{R}_t)^{-1} \\
    \mathbf{x}_t = \mathbf{x}_{pred} + \mathbf{K}_t (\mathbf{z}_t - h(\mathbf{x}_{pred})) \\
    \mathbf{P}_t = (I - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{pred}
    $$

    where $\mathbf{H}_t$ is the Jacobian of the measurement model with respect to the state, $\mathbf{R}_t$ is the measurement noise covariance matrix, $\mathbf{z}_t$ is the observed measurement, and $h(\cdot)$ is the measurement model.

## Functions

- `get_num_landmark(x)`: Returns the number of landmarks in the state vector `x`.
- `expand_P(P, sigma)`: Expands the covariance matrix `P` by adding a new landmark with initial uncertainty `sigma`.

## Variables

- `track_width`: The width of the track.
- `track_radius`: The radius of the track.
- `cone_number`: The number of cones around the track.
- `global_map`: A 2D array representing the global map of landmarks.
- `x_gt`: The ground truth state vector.
- `x_raw`: The raw state vector.
- `x`: The estimated state vector.
- `num_landmark`: The number of landmarks in the state vector.
- `landmark_sigma`: The initial uncertainty of a new landmark.
- `R`: The measurement noise covariance matrix.
- `Q`: The process noise covariance matrix.
- `P`: The state covariance matrix.
- `path_gt_hist`: A history of the ground truth path.
- `path_raw_hist`: A history of the raw path.
- `path_hist`: A history of the estimated path.

## Usage

Run the script with Python 3:

```bash
python3 ekf_slam.py