from launch import LaunchDescription
from launch_ros.actions import Node
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    
    rain_forklift_params = os.path.join(
        get_package_share_directory('rain_autonomous_forklift_det'),
        'config',
        'rain_autonomous_forklift_det_param.yaml'
    )

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=rain_forklift_params,
        description='Path to the ROS2 parameters file to use'
    )

    rviz_param_dir = os.path.join(
        get_package_share_directory('rain_autonomous_forklift_det'),
        'rviz',
        'rain_autonomous_forklift.rviz'
    )

    detector_node = Node(
        package='rain_autonomous_forklift_det',
        executable='rain_autonomous_forklift_det',
        name='object_detector',
        output='screen',
        parameters=[rain_forklift_params]
    )

    rviz = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_param_dir],
    )
    return LaunchDescription([
        params_declare,
        rviz,
        detector_node,
    ])
