import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class PathPublisher(Node):

    def __init__(self):
        super().__init__('path_publisher')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/orb_slam2/pose',
            self.callback,
            10)
        self.path_publisher = self.create_publisher(Path, 'pub_path', 10)
        self.path_msg = Path()

    def callback(self, data):
        pose = data.pose
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header = data.header
        self.path_msg.header = data.header
        self.path_msg.poses.append(pose_stamped)
        self.path_publisher.publish(self.path_msg)

def main(args=None):
    rclpy.init(args=args)

    path_publisher = PathPublisher()

    rclpy.spin(path_publisher)

    path_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
