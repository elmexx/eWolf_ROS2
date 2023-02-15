import rclpy
from video_to_image.pub_video2img_helper import Video2Img

def main(args=None):
    rclpy.init(args=args)

    image_node = Video2Img()

    rclpy.spin(image_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
