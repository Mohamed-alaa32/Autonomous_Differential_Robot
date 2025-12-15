#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path
# --- NEW: Import Marker message ---
from visualization_msgs.msg import Marker

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        # Clamp integral to prevent windup
        self.integral = max(min(self.integral, 10.0), -10.0)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

class LocalPathFollower(Node):
    def __init__(self):
        super().__init__('local_path_follower')

        # --- Parameters ---
        self.lookahead_dist = 1.5   # Meters to look ahead
        self.max_linear_vel = 0.1
        self.max_angular_vel = 0.3
        self.goal_tolerance = 0.1
        self.frame_id = "front_camera_link" # Define frame ID once
        
        # PID Gains (Tunable)
        self.pid_angular = PIDController(kp=1.5, ki=0.0, kd=0.05)
        
        # --- Subscribers & Publishers ---
        self.path_sub = self.create_subscription(
            Path, 
            '/lane_trajectory', 
            self.path_callback, 
            10
        )
        
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped, 
            '/cmd_vel', 
            10
        )

        # --- NEW: Marker Publisher for Target Point ---
        self.marker_pub = self.create_publisher(
            Marker,
            '/lookahead_marker',
            10
        )

        self.current_path = None
        self.timer = self.create_timer(0.05, self.control_loop) # 20 Hz control loop

        self.get_logger().info("Local Path Follower Started with Visualization")

    def path_callback(self, msg):
        self.current_path = msg

    def get_lookahead_point(self):
        if not self.current_path or not self.current_path.poses:
            return None
        
        poses = self.current_path.poses
        last_x = poses[-1].pose.position.x
        last_y = poses[-1].pose.position.y
        dist_to_goal = math.hypot(last_x, last_y)
        
        if dist_to_goal < self.goal_tolerance:
            return "GOAL_REACHED"

        for pose_stamped in poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            dist = math.hypot(x, y)
            
            if dist >= self.lookahead_dist:
                return (x, y)
        
        return (last_x, last_y)

    def publish_debug_marker(self, x, y):
        """Helper to publish a red sphere at the target location"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0 # On the ground
        
        # Scale (Size of the red ball)
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        
        # Color (Red, opaque)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0 # Alpha (transparency)
        
        self.marker_pub.publish(marker)

    def control_loop(self):
        if self.current_path is None:
            return

        # 1. Find Target
        target = self.get_lookahead_point()

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.frame_id

        if target == "GOAL_REACHED":
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            # Optional: Hide marker or change color when done
            return

        if target is None:
            return

        target_x, target_y = target

        # --- NEW: Publish the marker at the target coordinates ---
        self.publish_debug_marker(target_x, target_y)

        # 2. Calculate Error
        heading_error = math.atan2(target_y, target_x)

        # 3. PID Control
        angular_vel = self.pid_angular.compute(heading_error, dt=0.05)
        angular_vel = max(min(angular_vel, self.max_angular_vel), -self.max_angular_vel)

        # 4. Linear Velocity Logic
        if abs(heading_error) > 0.5:
            linear_vel = 0.1
        else:
            linear_vel = self.max_linear_vel * (1.0 - abs(heading_error))
            linear_vel = max(0.0, linear_vel)

        cmd.twist.linear.x = linear_vel
        cmd.twist.angular.z = angular_vel

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = LocalPathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = TwistStamped()
        stop_cmd.header.frame_id = "front_camera_link"
        node.cmd_vel_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()