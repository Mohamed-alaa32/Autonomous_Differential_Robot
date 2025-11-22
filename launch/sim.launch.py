
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.event_handlers import OnProcessExit

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def generate_launch_description():


    # --- 1. GET THE PACKAGE SHARE DIRECTORY ---
    pkg_share = get_package_share_directory('autonomous_differential_rob')

    # --- 2. BUILD THE FULL PATH TO YOUR SDF FILE ---
    world_file_path = os.path.join(pkg_share, 'worlds', 'world.sdf')

    # --- 3. PASS THE FULL PATH TO GAZEBO ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        # Use an f-string to inject the full path
        launch_arguments={'gz_args': f'-r {world_file_path}'}.items()
    )
 

    # Robot Description
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([get_package_share_directory('autonomous_differential_rob'), "urdf", "robot.xacro"]),
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

 
    # Spawn Robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'robot', '-x', '-1.0','-y', '0.0','-z', '0.5'],
        output='screen'
    )

    # This bridge connects Gazebo's /lidar topic to ROS 2's /scan topic
    laser_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Syntax: Gz Topic@ROS 2 Msg Type@Gz Msg Type
            '/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Bridge Image topic (Gazebo -> ROS)
            '/front_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            # Bridge Camera Info (Gazebo -> ROS) - Optional but recommended
            '/front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        # Remap the ROS 2 topic from /lidar to /scan for convention
        remappings=[
            ('/lidar', '/scan')
        ],
        output='screen'
    )

    bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        
    ],
    output='screen'
)

    # RViz Node
    rviz_node = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        output="log",
        arguments=["-d", os.path.join(get_package_share_directory('autonomous_differential_rob'), "rviz", "config.rviz")],
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # 5. Bridge (Optional: for Lidar/Camera, not needed for cmd_vel with ros2_control)
    # This bridge forwards the clock so ROS knows the sim time
 
    # 6. Spawners for Controllers
    spawn_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    spawn_diff_drive = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_base_controller"],
        output="screen",
    )


    hough_marker = Node(
        package='autonomous_differential_rob',
        executable='lane_keeping.py',
        name='line_follower_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    control_node = Node(
        package='autonomous_differential_rob',
        executable='control.py',
        name='control_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    # pkg_my_package = get_package_share_directory('dual_arms')
    
    # # # Path to your marker's SDF file
    # model_path = os.path.join(pkg_my_package, 'worlds', 'aruco.sdf')

    # # # Spawn the marker
    # spawn_marker = Node(
    #     package='ros_gz_sim',
    #     executable='create',
    #     arguments=[
    #         '-file', model_path,
    #         '-name', 'my_marker',
    #         '-x', '1.0',
    #         '-y', '0.5',
    #         '-z', '0.2'
    #     ],
    #     output='screen'
    # )

    # Path to your new controller config file
    controller_config = os.path.join(
      get_package_share_directory('autonomous_differential_rob'), 
      'config', 
      'diff_drive.yaml'
    )

    # --- Load the controller_manager ---
    # This is started by the Gazebo <plugin> in your URDF
    # But we need to pass it the config file
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{'robot_description': robot_description},
                    controller_config],
        output="screen"
    )
    # NOTE: If you are using Gazebo, the controller_manager
    # is often launched by the gz_ros2_control plugin,
    # so you might not need the node above, but you MUST
    # pass the 'controller_config' to the 'gz_sim' node
    # or the 'ros2_control_node'.

    # --- Spawner for joint_state_broadcaster ---
    # This loads the JSS.
    # spawn_joint_state_broadcaster = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["joint_state_broadcaster", 
    #                "--controller-manager", "/controller_manager"],
    #     output="screen",
    # )

    # --- Spawner for diff_drive_controller ---
    # This loads the diff drive controller.
    # spawn_diff_drive_controller = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["diff_drive_controller", 
    #                "--controller-manager", "/controller_manager"],
    #     output="screen",
    # )



    # Launch Description Assembly
    return LaunchDescription([
        gazebo,
        robot_state_publisher_node,
        spawn_entity,
        rviz_node,
        laser_bridge_node,
        hough_marker,
        control_node,
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_entity, on_exit=[spawn_jsb])
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_diff_drive])
        ),
    ])