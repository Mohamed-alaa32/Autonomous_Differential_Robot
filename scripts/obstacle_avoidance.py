#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped  # <--- NEW IMPORT

class BlueBoxDetector(Node):
    def __init__(self):
        super().__init__('blue_box_detector')

        # === Configuration ===
        self.BOX_REAL_SIZE = 0.3  
        self.CAMERA_HEIGHT = 0.2  
        self.CAMERA_PITCH = 0.0   

        self.blue_lower = np.array([100, 150, 50])
        self.blue_upper = np.array([140, 255, 255])

        self.img_sub = self.create_subscription(Image, '/front_camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/front_camera/camera_info', self.camera_info_callback, 10)
        
        self.marker_pub = self.create_publisher(Marker, '/obstacle/blue_box_marker', 10)
        self.debug_pub = self.create_publisher(Image, '/obstacle/debug_view', 10)
        
        # --- NEW PUBLISHER ---
        self.pose_pub = self.create_publisher(PoseStamped, '/obstacle/blue_box_pose', 10)

        self.bridge = CvBridge()
        self.fx = 600.0
        self.cx = 320.0
        self.cy = 240.0
        self.fy = 600.0
        self.camera_info_received = False

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.camera_info_received = True
            self.destroy_subscription(self.info_sub)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 500:
                rect = cv2.minAreaRect(largest_contour)
                box_points = cv2.boxPoints(rect)
                
                # --- FIX: Replace np.int0 with np.int32 ---
                box_points = np.int32(box_points) 

                lowest_point = max(box_points, key=lambda p: p[1])
                bottom_pixel_y = lowest_point[1]
                
                alpha = math.atan2(bottom_pixel_y - self.cy, self.fy)
                total_angle = self.CAMERA_PITCH + alpha
                
                if total_angle < 0.01: total_angle = 0.01 
                
                dist_to_front = self.CAMERA_HEIGHT / math.tan(total_angle)
                final_z = dist_to_front + (self.BOX_REAL_SIZE / 2.0)

                center_x_px = rect[0][0]
                real_x = (center_x_px - self.cx) * final_z / self.fx
                
                angle_deg = rect[2]
                if rect[1][0] < rect[1][1]: 
                    angle_deg += 90 
                yaw = math.radians(angle_deg)

                # --- 1. Publish Marker ---
                self.publish_marker(msg.header, real_x, final_z, yaw)
                
                # --- 2. Publish PoseStamped ---
                self.publish_pose(msg.header, real_x, final_z, yaw)
                
                cv2.drawContours(cv_image, [box_points], 0, (0, 0, 255), 2)
                cv2.circle(cv_image, tuple(lowest_point), 5, (0, 255, 0), -1) 
                cv2.putText(cv_image, f"{final_z:.2f}m", (lowest_point[0], lowest_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

    def publish_pose(self, header, x, z, yaw):
        """
        Publishes the PoseStamped of the box center.
        Note: Coordinates are in Optical Frame (Z=Forward, X=Right, Y=Down)
        """
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "front_camera_link"
        
        # Position
        pose_msg.pose.position.x = z  # Forward distance
        pose_msg.pose.position.y = -(x) # Lateral (ROS Y is usually Left, Optical X is Right -> flip sign)
        pose_msg.pose.position.z = 0.0 # On the floor (relative to robot base usually, but here relative to camera frame)
        
        # Orientation
        # Assuming simple yaw rotation around vertical axis
        q = R.from_euler('z', yaw, degrees=False).as_quat()
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self.pose_pub.publish(pose_msg)

    def publish_marker(self, header, x, z, yaw):
        marker = Marker()
        marker.header = header
        marker.header.frame_id = "front_camera_link"
        marker.ns = "blue_box_rotated"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Rviz Visualization Coordinates
        marker.pose.position.x = z
        marker.pose.position.y = -x # Negate because Camera Link Y is LEFT, Image X is RIGHT
        marker.pose.position.z = 0.0 # Floor level roughly relative to camera frame if we ignore height offset
        
        q = R.from_euler('z', yaw, degrees=False).as_quat()
        
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        
        marker.scale.x = self.BOX_REAL_SIZE
        marker.scale.y = self.BOX_REAL_SIZE
        marker.scale.z = self.BOX_REAL_SIZE
        
        marker.color.b = 1.0
        marker.color.a = 0.8
        marker.lifetime.nanosec = 200000000 # 0.2s

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = BlueBoxDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()