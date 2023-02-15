import rclpy
from speedgoat_package.speedgoat_ros2_node import Speedgoat

def main(args=None):
    rclpy.init(args=args)
    speedgoat_node = Speedgoat()
    rclpy.spin(speedgoat_node)
    
    speedgoat_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
