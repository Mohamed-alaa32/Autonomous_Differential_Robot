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

class BlueBoxDetector(Node):
    def __init__(self):
        super().__init__('blue_box_detector')

        # === Configuration ===
        self.BOX_REAL_SIZE = 0.4  # Meters (40cm)
        self.CAMERA_HEIGHT = 0.2  # Height of camera from floor (meters)
        self.CAMERA_PITCH = 0.0   # Radians (0 if looking straight forward)

        self.blue_lower = np.array([100, 150, 50])
        self.blue_upper = np.array([140, 255, 255])

        self.img_sub = self.create_subscription(Image, '/front_camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/front_camera/camera_info', self.camera_info_callback, 10)
        
        self.marker_pub = self.create_publisher(Marker, '/obstacle/blue_box_marker', 10)
        self.debug_pub = self.create_publisher(Image, '/obstacle/debug_view', 10)

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
                # 1. Get Rotated Rectangle (Handles the "not facing perfectly" issue)
                # rect returns ((center_x, center_y), (width, height), angle)
                rect = cv2.minAreaRect(largest_contour)
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)

                # 2. Find the "Bottom-Most" Point (The point touching the ground closest to us)
                # We look for the point with the highest Y pixel value
                lowest_point = max(box_points, key=lambda p: p[1])
                bottom_pixel_y = lowest_point[1]
                
                # 3. Calculate Distance to that Front Contact Point
                alpha = math.atan2(bottom_pixel_y - self.cy, self.fy)
                total_angle = self.CAMERA_PITCH + alpha
                
                # Protect against division by zero (horizon)
                if total_angle < 0.01: total_angle = 0.01 
                
                dist_to_front = self.CAMERA_HEIGHT / math.tan(total_angle)

                # 4. Apply Depth Offset
                # We detected the front face/corner. The center is 'Size/2' further back.
                # Note: This is a simplification. For 45-degree rotation, the offset differs slightly,
                # but Size/2 is a safe robust average for obstacle avoidance.
                final_z = dist_to_front + (self.BOX_REAL_SIZE / 2.0)

                # 5. Calculate X Position
                # Use the visual center of the rotated box for X
                center_x_px = rect[0][0]
                real_x = (center_x_px - self.cx) * final_z / self.fx
                
                # 6. Calculate Orientation (Yaw)
                # OpenCV angle is usually -90 to 0. We convert this to radians.
                angle_deg = rect[2]
                if rect[1][0] < rect[1][1]: 
                    angle_deg += 90 # Adjust based on which side is "width"
                
                # Convert to Radians. Note: Image rotation is around Z-axis in image plane, 
                # which maps to Y-axis rotation in the Optical Frame.
                yaw = math.radians(angle_deg)

                # 7. Visualization
                self.publish_marker(msg.header, real_x, final_z, yaw)
                
                # Draw rotated box on debug image
                cv2.drawContours(cv_image, [box_points], 0, (0, 0, 255), 2)
                cv2.circle(cv_image, tuple(lowest_point), 5, (0, 255, 0), -1) # Green dot at bottom contact
                cv2.putText(cv_image, f"{final_z:.2f}m", (lowest_point[0], lowest_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

    def publish_marker(self, header, x, z, yaw):
        marker = Marker()
        marker.header = header
        marker.header.frame_id = "front_camera_link"
        marker.ns = "blue_box_rotated"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = z
        # Fixed Y: Camera Height (Ground) - Half Box Height
        marker.pose.position.y = -(self.CAMERA_HEIGHT - (self.BOX_REAL_SIZE / 2.0))
        marker.pose.position.z = x
        
        # Create Quaternion for Rotation
        # In Optical Frame (X-Right, Y-Down, Z-Forward), "Yaw" is rotation around Y-axis.
        q = R.from_euler('y', yaw, degrees=False).as_quat()
        
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