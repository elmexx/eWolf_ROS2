import numpy as np
import matplotlib.pyplot as plt

WB = 2.9
Kp = 1.0

class PurePursuitController:
    def __init__(self, lookahead_distance, max_steering_angle):
        self.lookahead_distance = lookahead_distance
        self.max_steering_angle = max_steering_angle

    def global_to_vehicle_coordinates(self, vehicle, point):
        dx = point[0] - vehicle.x
        dy = point[1] - vehicle.y

        local_x = dx*np.cos(vehicle.theta) + dy*np.sin(vehicle.theta)
        local_y = -dx*np.sin(vehicle.theta) + dy*np.cos(vehicle.theta)

        return local_x, local_y

    def is_point_front(self, vehicle, point):
        dx = point[0] - vehicle.x
        dy = point[1] - vehicle.y

        local_x = dx*np.cos(vehicle.theta) + dy*np.sin(vehicle.theta)
        local_y = -dx*np.sin(vehicle.theta) + dy*np.cos(vehicle.theta)

        return local_x > 0

    def calculate_control(self, vehicle, path):
        
        lookahead_point = None

        for point in path:
            # x, y = point[0]-vehicle.x, point[1]-vehicle.y
            # distance = np.sqrt(x**2 + y**2)
            distance = np.hypot(point[0]-vehicle.x,point[1]-vehicle.y)
            if distance > self.lookahead_distance and self.is_point_front(vehicle, point):
                lookahead_point = point #(x,y)
                break

        if lookahead_point is None:
            lookahead_point = path[-1]
        
        dx = lookahead_point[0] - vehicle.x
        dy = lookahead_point[1] - vehicle.y

        angle_to_lookahead = np.arctan2(dy, dx) - vehicle.theta
        # angle_to_lookahead = max(min(angle_to_lookahead, self.max_steering_angle), -self.max_steering_angle)    

        steering_angle = 2*np.sin(angle_to_lookahead)/self.lookahead_distance
        steering_angle = max(min(steering_angle, self.max_steering_angle), -self.max_steering_angle)    

        return steering_angle, lookahead_point

class Vehicle:
    def __init__(self, x=0, y=0, theta=0, v = 0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.rear_x = self.x - ((WB/2)*np.cos(self.theta))
        self.rear_y = self.y - ((WB/2)*np.sin(self.theta))

    def update(self, a, steering_angle, dt):
        self.x += self.v*np.cos(self.theta)*dt
        self.y += self.v*np.sin(self.theta)*dt
        self.theta += self.v/WB*np.tan(steering_angle)*dt #steering_angle*dt
        self.v += a*dt
        self.rear_x = self.x - ((WB/2)*np.cos(self.theta))
        self.rear_y = self.y - ((WB/2)*np.sin(self.theta))

    def get_pose(self):
        return self.x, self.y, self.theta
    
def globalpath_to_vehiclepath(pure_pursuit,vehicle, path):
    vehicle_path = [pure_pursuit.global_to_vehicle_coordinates(vehicle, point) for point in path]
    return vehicle_path
    
def proportional_control(target, current):
    a = Kp*(target - current)
    return a

from scipy.interpolate import make_interp_spline

vehicle = Vehicle()
#path = [(i,0) for i in range(100)]
control_points = np.array([[0,0],[50,5],[100,0]])

x_new = np.linspace(0,100,500)
spl = make_interp_spline(control_points[:,0],control_points[:,1],k=2)
y_new = spl(x_new)

path = list(zip(x_new,y_new))
# path = [(i,0) for i in range(100)]
velocity = 5.0

max_steering_angle = 0.26
lookahead_distance = 5
dt = 0.1
num_steps = 300
target_speed = 30/3.6

pure_pursuit = PurePursuitController(lookahead_distance, max_steering_angle)

trajectory = []
lookahead_points = []
for _ in range(num_steps):
    ai = proportional_control(target_speed, vehicle.v)
    #vehicle_path = globalpath_to_vehiclepath(pure_pursuit, vehicle, path)
    control_angle, lookahead_point = pure_pursuit.calculate_control(vehicle, path)
    vehicle.update(ai, control_angle, dt)
    # print(vehicle.x,vehicle.y)
    trajectory.append(vehicle.get_pose())
    lookahead_points.append(lookahead_point)

    control_angle_mpc, acc_control_mpc = mpc_controller.optimize(mid_lane, current_state)
            
    # Update vehicle state for next iteration
    current_state = mpc_controller.vehicle_model(current_state, [control_angle_mpc, acc_control_mpc], dt)
    current_state[0], current_state[1] = 0, 0


    plt.cla()
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key =='escape' else None]
    )
    # plot_arrow(vehicle.x, vehicle.y, vehicle.theta)
    plt.plot(vehicle.x, vehicle.y, '-r')
    plt.plot(trajectory[0][0],trajectory[0][1],'-b')
    plt.plot(lookahead_point[0], lookahead_point[1], 'xg')
    plt.title('Speed:'+str(vehicle.v)[:4])
    plt.pause(0.001)
    
x_data = [pose[0] for pose in trajectory]
y_data = [pose[1] for pose in trajectory]

plt.figure(figsize=(10,6))
plt.plot(x_data, y_data,'r')
plt.plot(*zip(*path))


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline

# Constants
WB = 2.9
Kp = 1.0

class PurePursuitController:
    def __init__(self, lookahead_distance, max_steering_angle):
        self.lookahead_distance = lookahead_distance
        self.max_steering_angle = max_steering_angle

    def global_to_vehicle_coordinates(self, vehicle, point):
        dx = point[0] - vehicle.x
        dy = point[1] - vehicle.y

        local_x = dx * np.cos(vehicle.theta) + dy * np.sin(vehicle.theta)
        local_y = -dx * np.sin(vehicle.theta) + dy * np.cos(vehicle.theta)

        return local_x, local_y

    def is_point_front(self, vehicle, point):
        dx = point[0] - vehicle.x
        dy = point[1] - vehicle.y

        local_x = dx * np.cos(vehicle.theta) + dy * np.sin(vehicle.theta)
        local_y = -dx * np.sin(vehicle.theta) + dy * np.cos(vehicle.theta)

        return local_x > 0

    def calculate_control(self, vehicle, path):
        lookahead_point = None

        for point in path:
            distance = np.hypot(point[0] - vehicle.x, point[1] - vehicle.y)
            if distance > self.lookahead_distance and self.is_point_front(vehicle, point):
                lookahead_point = point
                break

        if lookahead_point is None:
            lookahead_point = path[-1]

        dx = lookahead_point[0] - vehicle.x
        dy = lookahead_point[1] - vehicle.y

        angle_to_lookahead = np.arctan2(dy, dx) - vehicle.theta
        steering_angle = 2 * np.sin(angle_to_lookahead) / self.lookahead_distance
        steering_angle = max(min(steering_angle, self.max_steering_angle), -self.max_steering_angle)

        return steering_angle, lookahead_point

class Vehicle:
    def __init__(self, x=0, y=0, theta=0, v=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.rear_x = self.x - ((WB / 2) * np.cos(self.theta))
        self.rear_y = self.y - ((WB / 2) * np.sin(self.theta))

    def update(self, a, steering_angle, dt):
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.v / WB * np.tan(steering_angle) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * np.cos(self.theta))
        self.rear_y = self.y - ((WB / 2) * np.sin(self.theta))

    def get_pose(self):
        return self.x, self.y, self.theta

def proportional_control(target, current):
    a = Kp * (target - current)
    return a

# MPC Controller class
class MPCController:
    def __init__(self, N, dt, L, max_steering_angle, max_acceleration, ref_v):
        self.N = N  # Prediction horizon
        self.dt = dt  # Time step
        self.L = L  # Wheelbase
        self.max_steering_angle = max_steering_angle
        self.max_acceleration = max_acceleration
        self.ref_v = ref_v  # Reference velocity

    def vehicle_model(self, state, control, dt):
        x, y, psi, v = state
        delta, a = control
        x_next = x + v * np.cos(psi) * dt
        y_next = y + v * np.sin(psi) * dt
        psi_next = psi + (v / self.L) * np.tan(delta) * dt
        v_next = v + a * dt
        return np.array([x_next, y_next, psi_next, v_next])

    def nearest_point_index(self, path, x, y):
        distances = [np.hypot(point[0] - x, point[1] - y) for point in path]
        nearest_index = np.argmin(distances)
        return nearest_index

    def optimize(self, path, current_state):
        def objective(U):
            U = U.reshape((self.N, 2))
            state = np.array(current_state)
            cost = 0
            for t in range(self.N):
                state = self.vehicle_model(state, U[t], self.dt)
                x, y, psi, v = state
                nearest_index = self.nearest_point_index(path, x, y)
                nearest_point = path[nearest_index]
                cost += 10 * ((x - nearest_point[0]) ** 2 + (y - nearest_point[1]) ** 2)  # Increase weight on path error
                cost += 10 * (v - self.ref_v) ** 2  # Increase weight on velocity error
                cost += 0.1 * U[t, 0] ** 2 + 0.1 * U[t, 1] ** 2  # Decrease weight on control effort
            return cost

        U0 = np.zeros((self.N, 2)).flatten()
        bounds = []
        for t in range(self.N):
            bounds.append((-self.max_steering_angle, self.max_steering_angle))  # Bounds for delta
            bounds.append((-self.max_acceleration, self.max_acceleration))  # Bounds for a

        result = minimize(objective, U0, bounds=bounds, method='SLSQP', options={'ftol': 1e-6, 'disp': True})
        if result.success:
            optimal_U = result.x.reshape((self.N, 2))
            steering_angle, acceleration = optimal_U[0]
            return steering_angle, acceleration
        else:
            raise ValueError(f"Optimization failed: {result.message}")

# Set up the simulation
control_points = np.array([[0, 0], [50, 5], [100, 0]])
x_new = np.linspace(0, 100, 500)
spl = make_interp_spline(control_points[:, 0], control_points[:, 1], k=2)
y_new = spl(x_new)
path = list(zip(x_new, y_new))
velocity = 5.0
max_steering_angle = 0.26
lookahead_distance = 5
dt = 0.1
num_steps = 300
target_speed = 30 / 3.6

# Initialize controllers
pure_pursuit = PurePursuitController(lookahead_distance, max_steering_angle)
mpc_controller = MPCController(N=20, dt=dt, L=WB, max_steering_angle=max_steering_angle, max_acceleration=0.5, ref_v=target_speed)

# Simulation loop for Pure Pursuit and MPC
vehicle_pp = Vehicle()
vehicle_mpc = Vehicle()
trajectory_pure_pursuit = []
trajectory_mpc = []
lookahead_points = []

current_state_mpc = [0, 0, 0, target_speed]  # Initial state for MPC with target speed

for _ in range(num_steps):
    # Pure Pursuit control
    ai_pp = proportional_control(target_speed, vehicle_pp.v)
    control_angle_pp, lookahead_point = pure_pursuit.calculate_control(vehicle_pp, path)
    vehicle_pp.update(ai_pp, control_angle_pp, dt)
    trajectory_pure_pursuit.append(vehicle_pp.get_pose())
    lookahead_points.append(lookahead_point)

    # MPC control
    try:
        control_angle_mpc, acc_control_mpc = mpc_controller.optimize(path, current_state_mpc)  # Use global path for MPC
        vehicle_mpc.update(acc_control_mpc, control_angle_mpc, dt)
        current_state_mpc = [vehicle_mpc.x, vehicle_mpc.y, vehicle_mpc.theta, vehicle_mpc.v]
        trajectory_mpc.append(vehicle_mpc.get_pose())
    except ValueError as e:
        print(f"Optimization failed at time step {_} with error: {e}")
        break

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(*zip(*path), label='Path')
plt.plot([pose[0] for pose in trajectory_pure_pursuit], [pose[1] for pose in trajectory_pure_pursuit], 'r', label='Pure Pursuit Trajectory')
plt.plot([pose[0] for pose in trajectory_mpc], [pose[1] for pose in trajectory_mpc], 'b', label='MPC Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory Comparison: Pure Pursuit vs MPC')
plt.legend()
plt.grid(True)
plt.show()

plt.title('Trajectory Comparison: Pure Pursuit vs MPC')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from numba import njit
import casadi as ca
import time

# Constants
WB = 2.9
Kp = 1.0

class Vehicle:
    def __init__(self, x=0, y=0, theta=0, v=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.rear_x = self.x - ((WB / 2) * np.cos(self.theta))
        self.rear_y = self.y - ((WB / 2) * np.sin(self.theta))

    def update(self, a, steering_angle, dt):
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.v / WB * np.tan(steering_angle) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * np.cos(self.theta))
        self.rear_y = self.y - ((WB / 2) * np.sin(self.theta))

    def get_pose(self):
        return self.x, self.y, self.theta

@njit
def nearest_point_index(path, x, y, theta):
    distances = np.hypot(path[:, 0] - x, path[:, 1] - y)
    nearest_index = np.argmin(distances)
    
    # Ensure the nearest point is in front of the vehicle
    for i in range(nearest_index, len(path)):
        dx = path[i, 0] - x
        dy = path[i, 1] - y
        angle_to_point = np.arctan2(dy, dx) - theta
        if np.cos(angle_to_point) > 0:
            nearest_index = i
            break
            
    return nearest_index

class MPCController:
    def __init__(self, N, dt, L, max_steering_angle, max_acceleration, ref_v):
        self.N = N  # Prediction horizon
        self.dt = dt  # Time step
        self.L = L  # Wheelbase
        self.max_steering_angle = max_steering_angle
        self.max_acceleration = max_acceleration
        self.ref_v = ref_v  # Reference velocity

        # Define model
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        v = ca.SX.sym('v')
        state = ca.vertcat(x, y, psi, v)

        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')
        control = ca.vertcat(delta, a)

        x_next = x + v * ca.cos(psi) * dt
        y_next = y + v * ca.sin(psi) * dt
        psi_next = psi + (v / L) * ca.tan(delta) * dt
        v_next = v + a * dt
        state_next = ca.vertcat(x_next, y_next, psi_next, v_next)

        self.model = ca.Function('model', [state, control], [state_next])

        # Define optimizer
        U = ca.SX.sym('U', N, 2)
        X = ca.SX.sym('X', N+1, 4)
        P = ca.SX.sym('P', 4 + 2 * N)

        Q = ca.diagcat(10, 10, 0, 10)
        R = ca.diagcat(0.1, 0.1)

        obj = 0
        g = []

        for k in range(N):
            st = X[k, :]
            con = U[k, :]
            ref = P[4 + 2*k: 6 + 2*k]
            obj = obj + ca.mtimes([(st - ref).T, Q, (st - ref)]) + ca.mtimes([con.T, R, con])
            st_next = X[k+1, :]
            f_value = self.model(st, con)
            g = ca.vertcat(g, st_next - f_value)

        opt_variables = ca.vertcat(U.reshape((-1, 1)), X.reshape((-1, 1)))

        nlp_prob = {
            'f': obj,
            'x': opt_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-6,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def optimize(self, path, current_state):
        N = self.N
        dt = self.dt
        L = self.L
        ref_v = self.ref_v

        ref_path = np.zeros((N, 2))
        current_pose = current_state[:2]
        current_theta = current_state[2]

        for k in range(N):
            index = nearest_point_index(path, current_pose[0], current_pose[1], current_theta)
            ref_path[k, :] = path[index, :]
            current_pose = ref_path[k, :]

        c_p = np.concatenate((current_state, ref_path.flatten()))

        lbg = np.zeros((N * 4, 1))
        ubg = np.zeros((N * 4, 1))

        lbx = np.concatenate([np.tile([-self.max_steering_angle, -self.max_acceleration], N), np.tile([-np.inf, -np.inf, -np.inf, -np.inf], N+1)])
        ubx = np.concatenate([np.tile([self.max_steering_angle, self.max_acceleration], N), np.tile([np.inf, np.inf, np.inf, np.inf], N+1)])

        u0 = np.zeros((N, 2))
        x0 = np.tile(current_state, (N+1, 1))

        init_opt = np.concatenate([u0.flatten(), x0.flatten()])

        sol = self.solver(
            x0=init_opt,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=c_p
        )

        u_opt = sol['x'].full().flatten()[:N*2].reshape(N, 2)

        return u_opt[0, 0], u_opt[0, 1]

# Set up the simulation
control_points = np.array([[0, 0], [50, 5], [100, 0]])
x_new = np.linspace(0, 100, 500)
spl = make_interp_spline(control_points[:, 0], control_points[:, 1], k=2)
y_new = spl(x_new)
path = np.array(list(zip(x_new, y_new)))  # Convert path to numpy array for numba compatibility
max_steering_angle = 0.26
dt = 0.1
num_steps = 300
target_speed = 30 / 3.6

# Initialize MPC controller and vehicle
mpc_controller = MPCController(N=10, dt=dt, L=WB, max_steering_angle=max_steering_angle, max_acceleration=0.5, ref_v=target_speed)
vehicle_mpc = Vehicle()
trajectory_mpc = []

current_state_mpc = np.array([0, 0, 0, target_speed])  # Initial state for MPC with target speed

for step in range(num_steps):
    try:
        if step % 5 == 0:  # Reduce optimization frequency
            control_angle_mpc, acc_control_mpc = mpc_controller.optimize(path, current_state_mpc)  # Use global path for MPC
        vehicle_mpc.update(acc_control_mpc, control_angle_mpc, dt)
        current_state_mpc = np.array([vehicle_mpc.x, vehicle_mpc.y, vehicle_mpc.theta, vehicle_mpc.v])
        trajectory_mpc.append(vehicle_mpc.get_pose())
    except ValueError as e:
        print(f"Optimization failed at time step {step} with error: {e}")
        break

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(path[:, 0], path[:, 1], label='Path')
plt.plot([pose[0] for pose in trajectory_mpc], [pose[1] for pose in trajectory_mpc], 'b', label='MPC Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('MPC Trajectory Following Path')
plt.legend()
plt.grid(True)
plt.show()

