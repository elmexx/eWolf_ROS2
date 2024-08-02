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


