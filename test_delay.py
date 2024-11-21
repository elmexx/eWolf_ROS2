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
