import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, N, dt, L, max_steering_angle, max_acceleration, ref_v):
        self.N = N  # Prediction horizon
        self.dt = dt  # Time step
        self.L = L  # Wheelbase
        self.max_steering_angle = max_steering_angle
        self.max_acceleration = max_acceleration
        self.ref_v = ref_v  # Reference velocity

        # State variables: [x, y, psi, v]
        self.x = np.zeros(N+1)
        self.y = np.zeros(N+1)
        self.psi = np.zeros(N+1)
        self.v = np.zeros(N+1)

        # Control inputs: [delta, a]
        self.delta = np.zeros(N)
        self.a = np.zeros(N)

        # Optimization variables
        self.delta_opt = cp.Variable(N)
        self.a_opt = cp.Variable(N)

    def optimize(self, waypoints, current_state):
        # Unpack current state
        x0, y0, psi0, v0 = current_state

        # Define constraints and cost function
        constraints = []
        cost = 0

        for t in range(self.N):
            if t == 0:
                self.x[t] = x0
                self.y[t] = y0
                self.psi[t] = psi0
                self.v[t] = v0

            self.x[t+1] = self.x[t] + self.v[t] * np.cos(self.psi[t]) * self.dt
            self.y[t+1] = self.y[t] + self.v[t] * np.sin(self.psi[t]) * self.dt
            self.psi[t+1] = self.psi[t] + (self.v[t] / self.L) * cp.tan(self.delta_opt[t]) * self.dt
            self.v[t+1] = self.v[t] + self.a_opt[t] * self.dt

            # Add constraints
            constraints += [
                self.delta_opt[t] <= self.max_steering_angle,
                self.delta_opt[t] >= -self.max_steering_angle,
                self.a_opt[t] <= self.max_acceleration,
                self.a_opt[t] >= -self.max_acceleration
            ]

            # Cost function
            cost += cp.norm(self.x[t] - waypoints[t][0])**2
            cost += cp.norm(self.y[t] - waypoints[t][1])**2
            cost += cp.norm(self.v[t] - self.ref_v)**2
            cost += cp.norm(self.delta_opt[t])**2
            cost += cp.norm(self.a_opt[t])**2

        # Define problem and solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Extract optimized control inputs
        self.delta = self.delta_opt.value
        self.a = self.a_opt.value

        return self.delta[0], self.a[0]

# Example usage
N = 10  # Prediction horizon
dt = 0.1  # Time step
L = 2.5  # Wheelbase of the vehicle in meters
max_steering_angle = 0.26  # Maximum steering angle in radians
max_acceleration = 1.0  # Maximum acceleration in m/s^2
ref_v = 15.0  # Reference velocity in m/s

controller = MPCController(N, dt, L, max_steering_angle, max_acceleration, ref_v)

# Waypoints representing the desired path in global coordinates
waypoints = [(10, 2), (20, 3), (30, 4), (40, 4), (50, 4), (60, 4), (70, 4), (80, 4), (90, 4), (100, 4)]

# Current state of the vehicle in global coordinates [x, y, psi, v]
current_state = (0, 0, 0, 10)

# Compute control inputs using MPC
steering_angle, acceleration = controller.optimize(waypoints, current_state)
print(f"Computed steering angle: {steering_angle} radians")
print(f"Computed acceleration: {acceleration} m/s^2")
