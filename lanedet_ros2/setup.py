from setuptools import setup
import os
from glob import glob

package_name = 'lanedet_ros2'
tools = "lanedet_ros2/tools"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, tools],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fmon005',
    maintainer_email='gaokunbit@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lanedet_node = lanedet_ros2.lanedet_node:main'
        ],
    },
)
