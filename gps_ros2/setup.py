from setuptools import setup

package_name = 'gps_ros2'
reach_ros_node = "gps_ros2/reach_ros_node"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, reach_ros_node],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gps_node = gps_ros2.gps_node:main'
        ],
    },
)
