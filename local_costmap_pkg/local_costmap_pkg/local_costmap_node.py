import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class LocalCostmapPublisherNode(Node):
    def __init__(self):
        super().__init__('local_costmap_publisher_node')
        self.costmap_publisher = self.create_publisher(OccupancyGrid, 'local_costmap', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):

        # map_data = np.random.randint(0, 100, size=(128, 128), dtype=np.int8)
        # map_data[50:70,50:70] = 0
        map_data = np.zeros((2454,678)).astype(np.int8)
        costmap_msg = OccupancyGrid()
        costmap_msg.header.stamp = self.get_clock().now().to_msg()
        costmap_msg.header.frame_id = 'base_link'  # Set appropriate frame ID
        costmap_msg.info.resolution = 0.0151  # Set appropriate resolution
        costmap_msg.info.width = 678
        costmap_msg.info.height = 2454
        costmap_msg.info.origin.position.x = 0  # Set appropriate origin
        costmap_msg.info.origin.position.y = 0
        costmap_msg.data = map_data.flatten().tolist() #[0] * (costmap_msg.info.width * costmap_msg.info.height)

        # Populate the costmap data based on your obstacle position matrix
        # For each element in your matrix, set the corresponding cost value

        self.costmap_publisher.publish(costmap_msg)

def main(args=None):
    rclpy.init(args=args)
    local_costmap_publisher_node = LocalCostmapPublisherNode()
    rclpy.spin(local_costmap_publisher_node)
    local_costmap_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
