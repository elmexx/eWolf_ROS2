import numpy as np

class StanleyController:
    def __init__(self, k_e=0.5, k_s=0.1):
        self.k_e = k_e  # Lateral error control gain
        self.k_s = k_s  # Speed stabilization factor

    def control(self, velocity, mid_lane):
        """
        Stanley controller to calculate the steering angle
        :param velocity: Vehicle speed
        :param mid_lane: Path points, shape (n, 2) array
        :return: Steering angle delta
        """
        # Step 1: Calculate lateral error 'e'
        current_pos = np.array([0.0, 0.0])  # Vehicle is always at the origin in its own coordinate system

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
        # Use the first nearest point to determine the direction
        nearest_point = mid_lane[nearest_indices[0]]
        path_direction = np.arctan2(mid_lane[min(nearest_indices[0] + 1, len(mid_lane) - 1), 1] - nearest_point[1],
                                    mid_lane[min(nearest_indices[0] + 1, len(mid_lane) - 1), 0] - nearest_point[0])
        theta_e = path_direction  # In vehicle coordinate system, current_yaw is always 0
        theta_e = np.arctan2(np.sin(theta_e), np.cos(theta_e))  # Normalize heading error to [-pi, pi]

        # Step 3: Calculate steering angle 'delta'
        delta = theta_e + np.arctan(self.k_e * e_avg / (velocity + self.k_s))
        delta = np.arctan2(np.sin(delta), np.cos(delta))  # Normalize steering angle to [-pi, pi]
        
        return delta

# Example usage
stanley_controller = StanleyController(k_e=1.0, k_s=0.1)

# Assume vehicle's speed
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
delta = stanley_controller.control(velocity, mid_lane)
print(f"Calculated steering angle: {np.degrees(delta):.2f} degrees")
