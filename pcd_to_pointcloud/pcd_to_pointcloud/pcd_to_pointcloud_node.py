import os
from pyntcloud import PyntCloud
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

class MyNode(Node):
    def __init__(self):
        super().__init__('pcl_node')
        self.publisher = self.create_publisher(PointCloud2, 'processed_point_cloud', 10)

        # Get the list of PCD files
        # pcd_folder = os.path.join(os.path.dirname(__file__), 'data')
        pcd_folder = '/media/fmon005/Data/slam/'
        pcd_files = [f for f in os.listdir(pcd_folder) if f.endswith('.pcd')]

        # Process each PCD file
        for pcd_file in pcd_files:
            self.process_pcd(os.path.join(pcd_folder, pcd_file))

    def process_pcd(self, pcd_file):
        # Load PCD file
        pcl_cloud = PyntCloud.from_file(pcd_file)

        # Perform point cloud processing using PCL library
        # ...

        # Convert processed PCL point cloud to PointCloud2 message
        pcl_msg = self.pcl_to_msg(pcl_cloud)

        # Publish the processed point cloud
        self.publisher.publish(pcl_msg)

    def pcl_to_msg(self, pcl_cloud):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        pcl_data = pc2.create_cloud(header, fields, pcl_cloud.to_array())

        return pcl_data

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

