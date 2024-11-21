# system time publish
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time

class TimePublisher(Node):
    def __init__(self):
        super().__init__('time_publisher_node')
        self.publisher_ = self.create_publisher(Time, 'system_time', 10)
        self.timer = self.create_timer(0.01, self.publish_time) 

    def publish_time(self):
        now = self.get_clock().now()
        time_msg = Time()
        time_msg.sec = int(now.seconds()) 
        time_msg.nanosec = int(now.nanoseconds() % 1e9)  
        self.publisher_.publish(time_msg)
        self.get_logger().info(f'Published time: {time_msg.sec}.{time_msg.nanosec:09d}')

def main(args=None):
    rclpy.init(args=args)
    node = TimePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# time subscription -> delay
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time

class TimeDelayCalculator(Node):
    def __init__(self):
        super().__init__('time_delay_calculator')
        self.subscription = self.create_subscription(
            Time,
            'system_time', 
            self.time_callback,
            10
        )

    def time_callback(self, msg):
        current_time = self.get_clock().now()
        received_time = rclpy.time.Time(seconds=msg.sec, nanoseconds=msg.nanosec)

        delay = (current_time - received_time).nanoseconds / 1e6 
        self.get_logger().info(f'Delay: {delay:.3f} ms')

def main(args=None):
    rclpy.init(args=args)
    node = TimeDelayCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# multi topic delay
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from message_filters import Subscriber, ApproximateTimeSynchronizer
from lane_parameter_msg.msg import LaneParams

class MultiTopicDelayCalculator(Node):
    def __init__(self):
        super().__init__('multi_topic_delay_calculator')

        self.system_time_sub = Subscriber(self, Time, 'system_time')
        self.camera_image_sub = Subscriber(self, SensorImage, '/camera/color/image_raw')
        self.lane_info_sub = Subscriber(self, LaneParams, 'leftlanedetection')
        
        self.sync = ApproximateTimeSynchronizer(
            [self.system_time_sub, self.camera_image_sub, self.lane_info_sub],
            queue_size=10,
            slop=0.1 
        )
        self.sync.registerCallback(self.sync_callback)

    def sync_callback(self, system_time_msg, camera_msg, lane_msg):

        system_time = rclpy.time.Time(seconds=system_time_msg.sec, nanoseconds=system_time_msg.nanosec)
        camera_time = rclpy.time.Time.from_msg(camera_msg.header.stamp)
        lane_time = rclpy.time.Time.from_msg(lane_msg.stamp)

        delay_camera_to_system = (camera_time - system_time).nanoseconds / 1e6  
        delay_lane_to_system = (lane_time - system_time).nanoseconds / 1e6  
        delay_camera_to_lane = (camera_time - lane_time).nanoseconds / 1e6  

        self.get_logger().info(f'Delay (Camera Image-> System): {delay_camera_to_system:.3f} ms')
        self.get_logger().info(f'Delay (Lane -> System): {delay_lane_to_system:.3f} ms')
        self.get_logger().info(f'Delay (Speedgoat -> Lane): {delay_camera_to_lane:.3f} ms')

def main(args=None):
    rclpy.init(args=args)
    node = MultiTopicDelayCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# speedgoat udp add system time
import time
current_time = time.time()
combined_array = np.concatenate((udp_data, [current_time]), axis=0).astype(np.float64)
