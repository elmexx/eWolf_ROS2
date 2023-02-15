def main():
    print('Hi from ros2bag2images.')


if __name__ == '__main__':
    main()

import rclpy
from ros2bag2images.bag2images_helper import ToImg

def main(args=None):
    rclpy.init(args=args)

    image_node = ToImg()

    rclpy.spin(image_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
