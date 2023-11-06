import launch
import launch_ros.actions
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

robot_localization_params = os.path.join(
            get_package_share_directory('camera_odometry'),
            'config',
            'vehicle_ekf.yaml'
        ),


def generate_launch_description():
    return launch.LaunchDescription([

        Node(
            package='camera_odometry',
            executable='pub_odom',
            name='pub_odom'),
            
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[robot_localization_params]
        )
  ])


