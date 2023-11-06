import launch
import launch_ros.actions
from launch_ros.actions import Node

def generate_launch_description():
    return launch.LaunchDescription([

        Node(
            package='lanedet_ros2',
            executable='lanedet_node',
            name='lanedet_ros'),
            
        Node(
            package='lane_detection_visualization',
            executable='lane_detection_visualization_node',
            name='lane_detection_visualization'),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2'),
  ])
