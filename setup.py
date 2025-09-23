from setuptools import setup
import os
from glob import glob


package_name = 'object_detection'


setup(
name=package_name,
version='0.1.0',
packages=[package_name],
data_files=[
# Register package with the ament index
('share/ament_index/resource_index/packages', ['resource/' + package_name]),
# Install package.xml
('share/' + package_name, ['package.xml']),
# Install launch files (optional, include if you have them)
('share/' + package_name + '/launch', glob('launch/*.launch.xml')),
],
install_requires=['setuptools'],
zip_safe=True,
author='Andrew Hellman',
author_email='andrewhellman@missouri.edu',
maintainer='Andrew Hellman',
maintainer_email='andrewhellman@missouri.edu',
description='Two-camera fused centroid detector with plane projection and true 3D triangulation.',
license='Apache 2.0',
tests_require=['pytest'],
entry_points={
'console_scripts': [
# maps: ros2 run object_detection object_detector -> object_detection/object_detector.py:main
'object_detector = object_detection.object_detection:main',
],
},
)