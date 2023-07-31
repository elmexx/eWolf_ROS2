import rclpy
from yolo3d_ros2.yolo3d_trt import Yolo3dDetector

def main(args=None):
    rclpy.init(args=args)

    detection_node = Yolo3dDetector()

    rclpy.spin(detection_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
