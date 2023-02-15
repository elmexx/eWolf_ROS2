import rclpy
from gps_ros2.nmea_tcp_driver import ReachGPS

def main(args=None):
    rclpy.init(args=args)
    
    host = "128.4.89.123"
    port = 2234

    gps_node = ReachGPS(host, port)
    gps_node.start()

    rclpy.spin(gps_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gps_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

