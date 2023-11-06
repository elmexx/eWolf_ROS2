import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os

def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='vehicle_description').find('vehicle_description')
    default_model_path = os.path.join(pkg_share, 'urdf/estima_white.urdf')
    # default_rviz_config_path = os.path.join(pkg_share, 'rviz/urdf_config.rviz')

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', LaunchConfiguration('model')])}]
    )
    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        condition=launch.conditions.UnlessCondition(LaunchConfiguration('gui'))
    )

    tf2_ros_publisher_node1 = launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ["0", "0", "0", "0", "0", "0", "map", "odom"],
        output="screen"    
    )
    tf2_ros_publisher_node2 = launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ["10", "0", "0", "0", "0", "0", "odom", "base_link"],
        output="screen"    
    )

    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2')

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='gui', default_value='True',
                                            description='Flag to enable joint_state_publisher_gui'),
        launch.actions.DeclareLaunchArgument(name='model', default_value=default_model_path,
                                            description='Absolute path to robot urdf file'),

        joint_state_publisher_node,
        robot_state_publisher_node,
        # tf2_ros_publisher_node1,
        # tf2_ros_publisher_node2
        # rviz_node
    ])
