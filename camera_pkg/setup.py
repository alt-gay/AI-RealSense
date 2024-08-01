from setuptools import find_packages, setup

package_name = 'camera_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raus',
    maintainer_email='raus@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'intelpub = camera_pkg.intelpub:main',
            'intelsub = camera_pkg.intelsub:main',
            'intelsubb = camera_pkg.intelsubb:main',
            'intel_sub = camera_pkg.intel_sub:main',
            'cam_action_server = camera_pkg.cam_action_server:main',
            'cam_action_client = camera_pkg.cam_action_client:main',
            'webcam_pub = camera_pkg.webcam_pub:main',
            'webcam_sub = camera_pkg.webcam_sub:main',
            'llm_request = camera_pkg.llm_request:main'
        ],
    },
)
