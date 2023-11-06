from setuptools import setup

package_name = 'yolo3d_ros2'
library = 'yolo3d_ros2/library'
torch_lib = 'yolo3d_ros2/torch_lib'
utils = 'yolo3d_ros2/utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, library, torch_lib, utils],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'yolo3d_node = yolo3d_ros2.yolo3d_node:main'
        ],
    },
)
