import rclpy
from lanedet_ros2.lanedet_node_helper import Lanedet

def main(args=None):
    rclpy.init(args=args)

    detection_node = Lanedet()

    rclpy.spin(detection_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
