from setuptools import setup

package_name = 'pcd_to_pointcloud'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
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
            'pcd_to_pointcloud_node = pcd_to_pointcloud.pcd_to_pointcloud_node:main'
        ],
    },
)
