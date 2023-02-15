import rclpy
from yolo_trt_ros2.yolo_trt import YoloDetector

def main(args=None):
    rclpy.init(args=args)

    detection_node = YoloDetector()

    rclpy.spin(detection_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
