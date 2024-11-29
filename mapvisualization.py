import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker, MarkerArray
import osmnx as ox
import networkx as nx
import pyproj
import numpy as np

class MapAndGPSVisualizationNode(Node):
    def __init__(self):
        super().__init__('map_and_gps_visualization_node')

        # Declare parameters
        self.declare_parameter('radius', 500)  # 半径 (以米为单位) 用于提取局部道路网络

        # Get parameters
        self.radius = self.get_parameter('radius').get_parameter_value().integer_value

        # Set up publishers for MarkerArray and Marker
        self.map_publisher = self.create_publisher(MarkerArray, 'map_markers', 10)
        self.gps_publisher = self.create_publisher(Marker, 'gps_marker', 10)

        # Initialize variables to store the graph and GPS coordinates
        self.graph = None
        self.latitude = None
        self.longitude = None

        # Create a subscriber for GPS data
        self.subscription = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10
        )

        # Setup coordinate transformer
        self.project = pyproj.Transformer.from_proj(
            pyproj.Proj(init='epsg:4326'),  # 原始坐标系：WGS84
            pyproj.Proj(proj='utm', zone=10, ellps='WGS84')  # 目标坐标系：UTM (根据实际情况调整zone)
        )

    def gps_callback(self, msg):
        # Update GPS coordinates
        self.latitude = msg.latitude
        self.longitude = msg.longitude

        # Convert GPS coordinates to UTM coordinates
        x, y = self.project.transform(self.latitude, self.longitude)
        
        # Create and publish a Marker for the GPS position
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.id = 0
        self.gps_publisher.publish(marker)

        # If the graph has not been initialized, initialize it using the current GPS coordinates
        if self.graph is None:
            self.initialize_map()

    def initialize_map(self):
        if self.latitude is not None and self.longitude is not None:
            self.get_logger().info(f"Initializing map around GPS coordinates: {self.latitude}, {self.longitude}")
            # Extract the local graph around the current GPS position
            self.graph = ox.graph_from_point((self.latitude, self.longitude), dist=self.radius, network_type='drive')

            # Convert graph nodes to UTM coordinates
            self.nodes, self.edges = ox.graph_to_gdfs(self.graph)
            self.nodes['x'], self.nodes['y'] = self.project.transform(self.nodes['y'].values, self.nodes['x'].values)
            
            # Publish the map as markers
            self.publish_map()

    def publish_map(self):
        marker_array = MarkerArray()

        # Create markers for nodes
        for idx, node in self.nodes.iterrows():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = node['x']
            marker.pose.position.y = node['y']
            marker.pose.position.z = 0.0
            marker.id = int(idx)
            marker_array.markers.append(marker)

        # Create markers for edges
        for idx, edge in self.edges.iterrows():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.5
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.points = []

            start_node = self.graph.nodes[edge['u']]
            end_node = self.graph.nodes[edge['v']]
            start_x, start_y = self.project.transform(start_node['y'], start_node['x'])
            end_x, end_y = self.project.transform(end_node['y'], end_node['x'])

            start_point = Marker()
            start_point.x = start_x
            start_point.y = start_y
            start_point.z = 0.0

            end_point = Marker()
            end_point.x = end_x
            end_point.y = end_y
            end_point.z = 0.0

            marker.points.append(start_point)
            marker.points.append(end_point)
            marker.id = int(idx) + len(self.nodes)
            marker_array.markers.append(marker)

        self.map_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MapAndGPSVisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



###
#!/bin/bash

while true; do
  ros2 topic pub /fix sensor_msgs/NavSatFix "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'gps'}, status: {status: 0, service: 1}, latitude: 37.7749, longitude: -122.4194, altitude: 10.0, position_covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], position_covariance_type: 0}" -1
  sleep 1
done
###
