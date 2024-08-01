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

        # Optimization variables
        self.delta_opt = cp.Variable(N)
        self.a_opt = cp.Variable(N)

    def optimize(self, waypoints, current_state):
        # Unpack current state
        psi0, v0 = current_state

        # Initialize state variables
        x = cp.Variable(self.N + 1)
        y = cp.Variable(self.N + 1)
        psi = cp.Variable(self.N + 1)
        v = cp.Variable(self.N + 1)

        # Initial state constraints
        constraints = [x[0] == 0, y[0] == 0, psi[0] == psi0, v[0] == v0]
        cost = 0

        for t in range(self.N):
            # Vehicle model constraints
            constraints += [
                x[t+1] == x[t] + v[t] * cp.cos(psi[t]) * self.dt,
                y[t+1] == y[t] + v[t] * cp.sin(psi[t]) * self.dt,
                psi[t+1] == psi[t] + (v[t] / self.L) * cp.tan(self.delta_opt[t]) * self.dt,
                v[t+1] == v[t] + self.a_opt[t] * self.dt,
                self.delta_opt[t] <= self.max_steering_angle,
                self.delta_opt[t] >= -self.max_steering_angle,
                self.a_opt[t] <= self.max_acceleration,
                self.a_opt[t] >= -self.max_acceleration
            ]

            # Cost function: track the middle lane and reference velocity
            cost += cp.norm(x[t+1] - waypoints[t, 0])**2
            cost += cp.norm(y[t+1] - waypoints[t, 1])**2
            cost += cp.norm(v[t] - self.ref_v)**2
            cost += cp.norm(self.delta_opt[t])**2
            cost += cp.norm(self.a_opt[t])**2

        # Define problem and solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Extract optimized control inputs
        steering_angle = self.delta_opt.value[0]
        acceleration = self.a_opt.value[0]

        return steering_angle, acceleration

# Example usage
N = 10  # Prediction horizon
dt = 0.1  # Time step
L = 2.5  # Wheelbase of the vehicle in meters
max_steering_angle = 0.26  # Maximum steering angle in radians
max_acceleration = 1.0  # Maximum acceleration in m/s^2
ref_v = 15.0  # Reference velocity in m/s

controller = MPCController(N, dt, L, max_steering_angle, max_acceleration, ref_v)

# Example waypoints representing the middle lane in vehicle coordinates
waypoints = np.array([(i, 0.1 * i) for i in range(60)])

# Initial state of the vehicle [psi, v]
psi = 0  # Example value: psi = 0 radians
v = 10  # Example value: v = 10 m/s

# Simulate for a few steps
for t in range(50):
    current_state = (psi, v)

    # Compute control inputs using MPC
    steering_angle, acceleration = controller.optimize(waypoints, current_state)
    print(f"Time step {t}:")
    print(f"Computed steering angle: {steering_angle} radians")
    print(f"Computed acceleration: {acceleration} m/s^2")

    # Update vehicle state for next iteration
    psi += (v / L) * np.tan(steering_angle) * dt
    v += acceleration * dt
