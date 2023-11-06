import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class OdometryGeneratorNode(Node):
    def __init__(self):
        super().__init__('odometry_generator_node')
        self.camera_pose_subscription = self.create_subscription(
            PoseStamped, '/orb_slam2/pose', self.camera_pose_callback, 10)
        self.odometry_publisher = self.create_publisher(Odometry, 'camera_odom', 10)
        self.previous_pose = None  # Store previous camera pose

    def camera_pose_callback(self, msg):
        if self.previous_pose is None:
            self.previous_pose = msg.pose
            return

        # Calculate odometry from camera pose data
        odometry_msg = Odometry()
        odometry_msg.header = msg.header
        odometry_msg.child_frame_id = 'camera_link'  # Change to appropriate frame

        # Calculate position and orientation changes
        position_change = (
            msg.pose.position.x - self.previous_pose.position.x,
            msg.pose.position.y - self.previous_pose.position.y,
            msg.pose.position.z - self.previous_pose.position.z
        )

        # Set odometry pose and twist
        odometry_msg.pose.pose = msg.pose
        odometry_msg.twist.twist.linear.x = position_change[0]
        odometry_msg.twist.twist.linear.y = position_change[1]
        odometry_msg.twist.twist.linear.z = position_change[2]

        # Publish odometry
        self.odometry_publisher.publish(odometry_msg)

        # Update previous pose
        self.previous_pose = msg.pose

def main(args=None):
    rclpy.init(args=args)
    odometry_generator_node = OdometryGeneratorNode()
    rclpy.spin(odometry_generator_node)
    odometry_generator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
