import numpy as np
from scipy.optimize import minimize

class StanleyController:
    def __init__(self, k_e=0.5, k_s=0.1):
        self.k_e = k_e  # Lateral error control gain
        self.k_s = k_s  # Speed stabilization factor

    def calculate_curvatures(self, mid_lane):
        """
        Calculate curvatures for each point in the path
        :param mid_lane: Path points, shape (n, 2) array
        :return: Curvatures, shape (n,) array
        """
        curvatures = np.zeros(len(mid_lane))
        for i in range(1, len(mid_lane) - 1):
            p1, p2, p3 = mid_lane[i - 1], mid_lane[i], mid_lane[i + 1]
            v1 = p2 - p1
            v2 = p3 - p2
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-pi, pi]
            distance = np.linalg.norm(v1)
            if distance != 0:
                curvatures[i] = angle / distance
        return curvatures

    def control(self, current_pos, current_yaw, velocity, mid_lane):
        """
        Stanley controller to calculate the steering angle
        :param current_pos: Vehicle current position (x, y)
        :param current_yaw: Vehicle heading angle (radians)
        :param velocity: Vehicle speed
        :param mid_lane: Path points, shape (n, 2) array
        :return: Steering angle delta
        """
        # Step 1: Calculate lateral error 'e'
        # Compute distance from vehicle position to all path points
        distances = np.linalg.norm(mid_lane - current_pos, axis=1)
        nearest_indices = distances.argsort()[:5]  # Consider the nearest 5 points
        nearest_points = mid_lane[nearest_indices]

        # Calculate the average lateral error using the nearest points
        e_sum = 0.0
        for point in nearest_points:
            vector_path_to_vehicle = current_pos - point
            path_direction = np.arctan2(mid_lane[min(nearest_indices[0] + 1, len(mid_lane) - 1), 1] - point[1],
                                        mid_lane[min(nearest_indices[0] + 1, len(mid_lane) - 1), 0] - point[0])
            error_direction = np.arctan2(vector_path_to_vehicle[1], vector_path_to_vehicle[0])
            e = np.linalg.norm(vector_path_to_vehicle) * np.sign(np.sin(error_direction - path_direction))
            e_sum += e
        e_avg = e_sum / len(nearest_points)

        # Step 2: Calculate heading error 'theta_e'
        # Calculate the average heading error using the nearest points
        theta_e_sum = 0.0
        for i in nearest_indices:
            nearest_point = mid_lane[i]
            path_direction = np.arctan2(mid_lane[min(i + 1, len(mid_lane) - 1), 1] - nearest_point[1],
                                        mid_lane[min(i + 1, len(mid_lane) - 1), 0] - nearest_point[0])
            theta_e = path_direction - current_yaw
            theta_e = np.arctan2(np.sin(theta_e), np.cos(theta_e))  # Normalize heading error to [-pi, pi]
            theta_e_sum += theta_e
        theta_e_avg = theta_e_sum / len(nearest_indices)

        # Step 3: Calculate curvature 'k'
        curvatures = self.calculate_curvatures(mid_lane)
        nearest_curvature = curvatures[nearest_indices[0]]

        # Step 4: Calculate steering angle 'delta'
        delta = theta_e_avg + np.arctan(self.k_e * e_avg / (velocity + self.k_s)) + 0.1 * nearest_curvature
        delta = np.arctan2(np.sin(delta), np.cos(delta))  # Normalize steering angle to [-pi, pi]
        
        return delta

# Example usage
stanley_controller = StanleyController(k_e=1.0, k_s=0.1)

# Assume vehicle's current position, heading angle, and speed
current_pos = np.array([0.0, 0.0])  # Current position in vehicle coordinates
current_yaw = 0.0  # Heading angle
velocity = 5.0  # Vehicle speed (in m/s)

# Assume path coordinates mid_lane
mid_lane = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.5],
    [3.0, 4.0],
    [4.0, 6.0],
    [5.0, 8.0],
    # More path points...
])

# Calculate steering angle
delta = stanley_controller.control(current_pos, current_yaw, velocity, mid_lane)
print(f"Calculated steering angle: {np.degrees(delta):.2f} degrees")


import numpy as np
from scipy.optimize import minimize

class MPCController:
    def __init__(self, steering_limit=0.5, accel_limit=2.0, horizon=10):
        self.steering_limit = steering_limit  # Steering angle limit in radians
        self.accel_limit = accel_limit  # Acceleration limit in m/s^2
        self.horizon = horizon  # Prediction horizon

    def objective(self, u, *args):
        """
        Objective function for MPC
        :param u: Control inputs (steering and acceleration)
        :param args: Additional arguments (current state, path points, curvatures)
        :return: Cost
        """
        current_pos, current_yaw, velocity, mid_lane, curvatures = args
        cost = 0.0

        # Simulate vehicle dynamics for the prediction horizon
        x, y, yaw, v = current_pos[0], current_pos[1], current_yaw, velocity
        dt = 0.1  # Time step for simulation
        for i in range(self.horizon):
            delta = u[2 * i]  # Steering angle
            a = u[2 * i + 1]  # Acceleration

            # Vehicle dynamics model
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
            yaw += v / 2.5 * np.tan(delta) * dt  # 2.5 is the wheelbase
            v += a * dt

            # Calculate lateral error to the nearest path point
            distances = np.linalg.norm(mid_lane - np.array([x, y]), axis=1)
            nearest_index = np.argmin(distances)
            nearest_point = mid_lane[nearest_index]
            e = np.linalg.norm(np.array([x, y]) - nearest_point)

            # Calculate curvature cost
            curvature = curvatures[nearest_index]

            # Cost function: penalize lateral error, steering, acceleration, and curvature
            cost += e**2 + 0.1 * delta**2 + 0.1 * a**2 + 10.0 * curvature**2

        return cost

    def control(self, current_pos, current_yaw, velocity, mid_lane):
        """
        MPC controller to calculate the steering angle and acceleration
        :param current_pos: Vehicle current position (x, y)
        :param current_yaw: Vehicle heading angle (radians)
        :param velocity: Vehicle speed
        :param mid_lane: Path points, shape (n, 2) array
        :return: Steering angle and acceleration
        """
        # Calculate curvatures along the path
        curvatures = self.calculate_curvatures(mid_lane)

        # Initial guess for control inputs
        u0 = np.zeros(2 * self.horizon)

        # Bounds for control inputs (steering angle and acceleration)
        bounds = [(-self.steering_limit, self.steering_limit), (-self.accel_limit, self.accel_limit)] * self.horizon

        # Optimize control inputs
        result = minimize(self.objective, u0, args=(current_pos, current_yaw, velocity, mid_lane, curvatures), bounds=bounds, method='SLSQP')

        # Extract optimized steering angle and acceleration
        if result.success:
            optimal_u = result.x
            steering_angle = optimal_u[0]
            acceleration = optimal_u[1]
        else:
            steering_angle = 0.0
            acceleration = 0.0

        return steering_angle, acceleration

    def calculate_curvatures(self, mid_lane):
        """
        Calculate curvatures for each point in the path
        :param mid_lane: Path points, shape (n, 2) array
        :return: Curvatures, shape (n,) array
        """
        curvatures = np.zeros(len(mid_lane))
        for i in range(1, len(mid_lane) - 1):
            p1, p2, p3 = mid_lane[i - 1], mid_lane[i], mid_lane[i + 1]
            v1 = p2 - p1
            v2 = p3 - p2
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-pi, pi]
            distance = np.linalg.norm(v1)
            if distance != 0:
                curvatures[i] = angle / distance
        return curvatures

# Example usage
mpc_controller = MPCController(steering_limit=0.5, accel_limit=2.0, horizon=10)

# Assume vehicle's current position, heading angle, and speed
current_pos = np.array([0.0, 0.0])  # Current position in vehicle coordinates
current_yaw = 0.0  # Heading angle
velocity = 5.0  # Vehicle speed (in m/s)

# Assume path coordinates mid_lane
mid_lane = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.5],
    [3.0, 4.0],
    [4.0, 6.0],
    [5.0, 8.0],
    # More path points...
])

# Calculate steering angle and acceleration
steering_angle, acceleration = mpc_controller.control(current_pos, current_yaw, velocity, mid_lane)
print(f"Calculated steering angle: {np.degrees(steering_angle):.2f} degrees")
print(f"Calculated acceleration: {acceleration:.2f} m/s^2")
