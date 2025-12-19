#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool # <--- NEW IMPORT

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
        self.lookahead_dist = 1.5   
        self.max_linear_vel = 1.0
        self.max_angular_vel = 1.3
        self.goal_tolerance = 0.1
        self.frame_id = "front_camera_link" 
        
        # PID Gains
        self.pid_angular = PIDController(kp=1.5, ki=0.0, kd=0.05)
        
        # --- Subscribers & Publishers ---
        self.path_sub = self.create_subscription(Path, '/lane_trajectory', self.path_callback, 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)
        
        # --- NEW: Finish Line Subscriber ---
        self.finish_sub = self.create_subscription(Bool, '/finish_line', self.finish_cb, 10)

        self.current_path = None
        self.timer = self.create_timer(0.05, self.control_loop) 

        # --- NEW: Finish Line State ---
        self.finish_triggered = False
        self.finish_start_time = None
        self.deceleration_duration = 3.5 # Seconds
        self.fully_stopped = False

        self.get_logger().info("Local Path Follower Started with Visualization")

    def path_callback(self, msg):
        self.current_path = msg

    def finish_cb(self, msg):
        """Callback for /finish_line topic"""
        if msg.data and not self.finish_triggered:
            self.finish_triggered = True
            self.finish_start_time = self.get_clock().now()
            self.get_logger().info("Finish Line Detected! Decelerating...")

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
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0 
        
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0 
        
        self.marker_pub.publish(marker)

    def control_loop(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.frame_id

        # --- 0. CHECK FINISH LINE STATUS ---
        if self.fully_stopped:
            # Send 0 command continuously to keep it stopped
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        # 1. Path Logic
        if self.current_path is None: return

        target = self.get_lookahead_point()

        if target == "GOAL_REACHED":
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        if target is None: return

        target_x, target_y = target
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

        # --- 5. FINISH LINE DECELERATION OVERRIDE ---
        if self.finish_triggered:
            now = self.get_clock().now()
            elapsed_time = (now - self.finish_start_time).nanoseconds / 1e9
            
            if elapsed_time >= self.deceleration_duration:
                # Time is up, stop completely
                linear_vel = 0.0
                angular_vel = 0.0
                self.fully_stopped = True
                self.get_logger().info("Car Stopped.")
            else:
                # Ramp down velocity: Speed * (1 - ratio of time passed)
                decel_factor = 1.0 - (elapsed_time / self.deceleration_duration)
                linear_vel = linear_vel * decel_factor
                # Optional: Dampen angular velocity too so it doesn't jerk while stopping
                angular_vel = angular_vel * decel_factor 

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