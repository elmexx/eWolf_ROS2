from setuptools import setup

package_name = 'lane_detection_package'
unet = "lane_detection_package/unet"
utils = "lane_detection_package/utils"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, unet, utils],
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
        'lane_detector = lane_detection_package.lane_detector:main'
        ],
    },
)
