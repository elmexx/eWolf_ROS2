from setuptools import setup

package_name = 'yolo_trt_ros2'
utils = 'yolo_trt_ros2/utils'
plugins = 'yolo_trt_ros2/plugins'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, utils, plugins],
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
            'yolo_trt_node = yolo_trt_ros2.yolo_trt_node:main'
        ],
    },
)
