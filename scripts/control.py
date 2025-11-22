#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path

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
        self.lookahead_dist = 0.5   # Meters to look ahead
        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0
        self.goal_tolerance = 0.1
        
        # PID Gains (Tunable)
        # Since error is just the angle to the point, Kp acts directly on steering
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
            '/diff_drive_base_controller/cmd_vel', 
            10
        )

        self.current_path = None
        self.timer = self.create_timer(0.05, self.control_loop) # 20 Hz control loop

        self.get_logger().info("Local Path Follower Started")

    def path_callback(self, msg):
        # We assume this path is already in 'base_link' or similar local frame
        self.current_path = msg

    def get_lookahead_point(self):
        """
        Finds the first point in the path that is further than 'lookahead_dist'
        from the robot origin (0,0).
        """
        if not self.current_path or not self.current_path.poses:
            return None
        
        # inverted_poses = self.current_path.poses[::-1]

        # poses = inverted_poses
        poses = self.current_path.poses
        # Check if the end of the path is very close (Goal Reached)
        last_x = poses[-1].pose.position.x
        last_y = poses[-1].pose.position.y
        dist_to_goal = math.hypot(last_x, last_y)
        # self.get_logger().info("Distance to goal: {}".format(dist_to_goal))
        self.get_logger().info("Last Point: x={}, y={}, length of path: {}".format(last_x, last_y, len(poses)))
        
        if dist_to_goal < self.goal_tolerance:
            self.get_logger().info("Distance to goal: {}".format(dist_to_goal))
            return "GOAL_REACHED"

        # Find the point that satisfies the lookahead distance
        for pose_stamped in poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            
            # Distance from robot (0,0) to point (x,y)
            dist = math.hypot(x, y)
            
            if dist >= self.lookahead_dist:
                return (x, y)
        
        # If all points are closer than lookahead, but we aren't at goal, 
        # aim for the last point.
        return (last_x, last_y)

    def control_loop(self):
        if self.current_path is None:
            return

        # 1. Find Target
        target = self.get_lookahead_point()

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "front_camera_link"

        if target == "GOAL_REACHED":
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info("Goal Reached - Stopping", throttle_duration_sec=2.0)
            return

        if target is None:
            return

        target_x, target_y = target

        # 2. Calculate Error
        # Since robot is at (0,0) facing 0 rads:
        # The angle to the target is simply atan2(y, x)
        # The error is (target_angle - robot_yaw) -> (atan2(y,x) - 0)
        heading_error = math.atan2(target_y, target_x)

        # 3. PID Control
        angular_vel = self.pid_angular.compute(heading_error, dt=0.05)
        
        # Clamp angular velocity
        angular_vel = max(min(angular_vel, self.max_angular_vel), -self.max_angular_vel)

        # 4. Linear Velocity Logic
        # Slow down if the turn is sharp
        if abs(heading_error) > 0.5: # ~30 degrees
            linear_vel = 0.1 # Creep speed while turning
        else:
            # Scale speed: faster when error is small
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
        # Safety stop
        stop_cmd = TwistStamped()
        stop_cmd.header.frame_id = "front_camera_link"
        node.cmd_vel_pub.publish(stop_cmd)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()