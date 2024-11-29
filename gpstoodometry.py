import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

class GpsToOdometry(Node):
    def __init__(self):
        super().__init__('gps_to_odometry')
        self.subscription = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10
        )
        self.publisher = self.create_publisher(Odometry, '/gps/odometry', 10)

    def gps_callback(self, msg: NavSatFix):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'

        odom_msg.pose.pose.position.x = msg.latitude
        odom_msg.pose.pose.position.y = msg.longitude
        odom_msg.pose.pose.position.z = msg.altitude

        odom_msg.pose.pose.orientation = Quaternion(w=1.0)

        self.publisher.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GpsToOdometry()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
