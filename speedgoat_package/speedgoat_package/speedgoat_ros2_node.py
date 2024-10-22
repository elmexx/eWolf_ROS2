# ROS2 imports 
import rclpy
from rclpy.node import Node

# python imports
import numpy as np
import socket
import cv2
import os, sys, time

# message imports
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import String
from lane_parameter_msg.msg import LaneParams
from sensor_msgs.msg import CompressedImage
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer


def UDP_send(socket_UDP, server_address, msg):
    socket_UDP.sendto(msg,server_address)
    return None
    
class PurePursuitController:
    def __init__(self, lookahead_distance, max_steering_angle):
        self.lookahead_distance = lookahead_distance
        self.max_steering_angle = max_steering_angle

    def calculate_control(self, mid_lane):
        current_pos = (0,0)
        lookahead_point = None

        for x, y in mid_lane:
            distance = np.sqrt(x**2 + y**2)
            if distance > self.lookahead_distance:
                lookahead_point = (x,y)
                break

        if lookahead_point is None:
            lookahead_point = mid_lane[-1]
        
        dx = lookahead_point[0] - current_pos[0]
        dy = lookahead_point[1] - current_pos[1]

        angle_to_lookahead = np.arctan2(dy, dx)

        angle_to_lookahead = max(min(angle_to_lookahead, self.max_steering_angle), -self.max_steering_angle)    
        return angle_to_lookahead

import numpy as np

class StanleyController:
    def __init__(self, k, max_steering_angle, vehicle_length):
        self.k = k  # 控制增益，影响横向误差的作用
        self.max_steering_angle = max_steering_angle
        self.vehicle_length = vehicle_length

    def calculate_control(self, mid_lane, vehicle_yaw, vehicle_pos):
        # 车辆的当前位置
        current_pos = vehicle_pos

        # 计算横向误差，选择最近的点
        nearest_point = None
        min_distance = float('inf')
        for x, y in mid_lane:
            distance = np.sqrt((x - current_pos[0]) ** 2 + (y - current_pos[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = (x, y)
        
        # 横向误差是当前车辆位置到路径的距离
        lateral_error = min_distance
        
        # 计算路径点的航向角
        dx = nearest_point[0] - current_pos[0]
        dy = nearest_point[1] - current_pos[1]
        path_yaw = np.arctan2(dy, dx)
        
        # 计算航向误差（路径的角度与车辆的角度差异）
        yaw_error = path_yaw - vehicle_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # 将角度标准化到[-pi, pi]

        # Stanley控制器公式：转向角 = 航向误差 + arctan(k * 横向误差 / 速度)
        # 这里假设速度恒定为 1 来简化
        steering_angle = yaw_error + np.arctan2(self.k * lateral_error, 1)

        # 限制转向角在最大范围内
        steering_angle = max(min(steering_angle, self.max_steering_angle), -self.max_steering_angle)

        return steering_angle

from scipy.optimize import minimize
import numpy as np

class MPCController:
    def __init__(self, horizon, dt, max_steering_angle, vehicle_length):
        self.horizon = horizon  # 预测时域
        self.dt = dt  # 时间步长
        self.max_steering_angle = max_steering_angle
        self.vehicle_length = vehicle_length

    def vehicle_model(self, state, steering_angle, velocity):
        # 状态为[x, y, yaw]
        x, y, yaw = state
        x_next = x + velocity * np.cos(yaw) * self.dt
        y_next = y + velocity * np.sin(yaw) * self.dt
        yaw_next = yaw + (velocity / self.vehicle_length) * np.tan(steering_angle) * self.dt
        return np.array([x_next, y_next, yaw_next])

    def calculate_control(self, mid_lane, vehicle_state):
        # 目标是最小化预测状态和路径之间的误差
        def objective(steering_sequence):
            state = np.array(vehicle_state)  # [x, y, yaw]
            total_error = 0

            for i in range(self.horizon):
                steering_angle = steering_sequence[i]
                state = self.vehicle_model(state, steering_angle, 1)  # 假设速度恒定为1
                nearest_point = min(mid_lane, key=lambda p: np.linalg.norm(np.array(p) - state[:2]))
                error = np.linalg.norm(np.array(nearest_point) - state[:2])  # 横向误差
                total_error += error

            return total_error

        # 初始转向角序列猜测
        initial_guess = [0] * self.horizon

        # 约束转向角的最大最小值
        bounds = [(-self.max_steering_angle, self.max_steering_angle)] * self.horizon

        # 使用最小化函数来求解最优转向角序列
        result = minimize(objective, initial_guess, bounds=bounds)

        # 返回第一个时间步的最优转向角
        return result.x[0]

# Example of usage
vehicle_state = [0, 0, 0]  # 假设车辆的初始状态是 [x, y, yaw]
mid_lane = [(5, 2), (10, 3), (15, 4)]  # 路径上的点

mpc = MPCController(horizon=10, dt=0.1, max_steering_angle=np.pi/4, vehicle_length=2.5)
steering_angle = mpc.calculate_control(mid_lane, vehicle_state)
print(f'MPC Steering Angle: {steering_angle}')


from scipy.optimize import minimize

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

    def optimize(self, waypoints, current_state):
        # Objective function to minimize
        def objective(U):
            U = U.reshape((self.N, 2))
            state = np.array([0, 0, current_state[2], current_state[3]])  # x, y are always 0 in vehicle coordinates
            cost = 0
            prev_u = np.zeros(2)
            for t in range(self.N):
                state = self.vehicle_model(state, U[t], self.dt)
                x, y, psi, v = state
                cost += 10*((x - waypoints[t, 0])**2 + (y - waypoints[t, 1])**2)  # Minimize path error
                cost += (v - self.ref_v)**2  # Minimize velocity error
                cost += (U[t, 0]**2 + U[t, 1]**2)  # Minimize control effort
                cost += 10*np.sum((U[t]-prev_u)**2)
                prev_u = U[t]
            return cost

        # Initial guess for control inputs
        U0 = np.zeros((self.N, 2)).flatten()

        # Bounds for control inputs
        bounds = []
        for t in range(self.N):
            bounds.append((-self.max_steering_angle, self.max_steering_angle))  # Bounds for delta
            bounds.append((-self.max_acceleration, self.max_acceleration))      # Bounds for a

        # Solve the optimization problem
        result = minimize(objective, U0, bounds=bounds, method='SLSQP', options={'ftol': 1e-6})

        if result.success:
            optimal_U = result.x.reshape((self.N, 2))
            steering_angle, acceleration = optimal_U[0]
            return steering_angle, acceleration
        else:
            raise ValueError(f"Optimization failed: {result.message}")


class Speedgoat(Node):
    def __init__(self):
        super().__init__('speedgoat_node')
        self.declare_parameter('method', 'mpc')
        self.declare_parameter('ref_v', 7.0)

        self.method = self.get_parameter('method').get_parameter_value().string_value
        self.ref_v = self.get_parameter('ref_v').get_parameter_value().double_value

        left_sub = Subscriber(self, LaneParams, 'leftlanedetection')
        right_sub = Subscriber(self, LaneParams, 'rightlanedetection')
        # ts = TimeSynchronizer([left_sub, right_sub], 10)
        ts = ApproximateTimeSynchronizer([left_sub, right_sub], 10, 1)
        ts.registerCallback(self.callback)
        
        #self.host = "10.42.0.10"
        self.host = "169.254.18.189"
        self.port = 5500
        self.send_port = 8080
        self.buffersize = 1024
        self.server_address = (self.host, self.port) 
       
        # create Socket UDP
        self.socket_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_UDP_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receive_addr = ("", self.send_port)
        self.socket_UDP_receive.bind(self.receive_addr)
        self.socket_UDP_receive.settimeout(1)
        
        max_steering_angle = 0.26
        lookahead_distance = 5
        self.x_vehicle = np.linspace(0,20,50)

        N = 10  # Prediction horizon
        dt = 0.1  # Time step
        L = 2.5  # Wheelbase of the vehicle in meters
        max_acceleration = 1.0  # Maximum acceleration in m/s^2
        #ref_v = 7.0  # Reference velocity in m/s

        if self.method == 'pure_pursuit':
            self.pure_pursuit = PurePursuitController(lookahead_distance, max_steering_angle)
        elif self.method == 'mpc':
            self.mpc = MPCController(N, dt, L, max_steering_angle, max_acceleration, self.ref_v)
    
    def callback(self, left_sub, right_sub):
        # self.get_logger().info('left heard: "%f"' % left_sub.a)
        # self.get_logger().info('left heard: "%f"' % left_sub.b)
        # self.get_logger().info('right heard: "%f"' % right_sub.a)

        #lane_params = np.array([left_sub.a,left_sub.b,left_sub.c, right_sub.a, right_sub.b, right_sub.c]).astype(np.float64)
        
        leftparam = np.array([left_sub.a,left_sub.b,left_sub.c])
        rightparam = np.array([right_sub.a, right_sub.b, right_sub.c])
        m_laneparam = (leftparam + rightparam)/2
        # left_lane = np.vstack((self.x_vehicle,np.polyval(leftparam, self.x_vehicle))).T
        # right_lane = np.vstack((self.x_vehicle,np.polyval(rightparam, self.x_vehicle))).T
        mid_lane = np.vstack((self.x_vehicle,np.polyval(m_laneparam, self.x_vehicle))).T
        control_angle = self.pure_pursuit.calculate_control(mid_lane)

        udp_data = np.array([left_sub.a,left_sub.b,left_sub.c, right_sub.a, right_sub.b, right_sub.c, control_angle]).astype(np.float64)
        UDP_msg = udp_data.tobytes()
        # print(UDP_msg)
        UDP_send(socket_UDP=self.socket_UDP, server_address=self.server_address, msg=UDP_msg)
        # receive
        """
        try:
            recv_data, addr = self.socket_UDP_receive.recvfrom(64)
            outdata = np.frombuffer(recv_data,dtype=np.single)
            print('Receive Daten: ',outdata)
        except socket.timeout as e:
            pass   
        """     
        return True 

