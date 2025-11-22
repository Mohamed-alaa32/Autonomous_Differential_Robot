#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math

class LaneDetector:
    def __init__(self):
        self.lane_width_meters = 0.4
        # METERS_PER_PIXEL needs calibration based on camera height/angle
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        
        self.current_fit_left = None
        self.current_fit_right = None

    def preprocess(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(hls, lower_white, upper_white)
        lower_yellow = np.array([10, 0, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
        combined = cv2.bitwise_or(white_mask, yellow_mask)
        return combined

    def perspective_transform(self, img):
        h, w = img.shape[:2]
        src = np.float32([
            [w * 0.4, h * 0.65],
            [w * 0.6, h * 0.65],
            [w, h],
            [0, h]
        ])
        dst = np.float32([
            [w * 0.2, 0],
            [w * 0.8, 0],
            [w * 0.8, h],
            [w * 0.2, h]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped, Minv

    def find_lane_lines(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        if histogram[leftx_base] < 10 or histogram[rightx_base] < 10:
            return None, None

        nwindows = 9
        window_height = int(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        try:
            left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
            right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
            self.current_fit_left = left_fit
            self.current_fit_right = right_fit
            return left_fit, right_fit
        except Exception:
            return self.current_fit_left, self.current_fit_right

    def generate_trajectory_points(self, left_fit, right_fit, height, offset_meters=0):
        """
        Generate points limited to 2 METERS ahead of the robot.
        """
        # 1. Calculate how many pixels represent 2 meters
        pixels_needed = 2.0 / self.ym_per_pix
        
        # 2. Define Y range
        # Image Bottom (Robot) = height
        # 2 Meters away = height - pixels_needed
        start_y = height - 1
        end_y = max(0, height - pixels_needed) # Ensure we don't go off image
        
        # Generate 20 points within this specific 2m range
        # Note: We generate from Bottom (Near) to Top (Far)
        ploty = np.linspace(start_y, end_y, num=20)
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        center_fitx = (left_fitx + right_fitx) / 2
        
        offset_pixels = offset_meters / self.xm_per_pix
        target_fitx = center_fitx + offset_pixels
        
        return target_fitx, ploty

class LaneNode(Node):
    def __init__(self):
        super().__init__('lane_keeping_node')
        
        self.declare_parameter('camera_topic', '/front_camera/image_raw')
        self.declare_parameter('camera_frame_id', 'camera_optical_frame')
        
        topic_name = self.get_parameter('camera_topic').value
        self.sub_img = self.create_subscription(Image, topic_name, self.image_cb, 10)
        self.pub_path = self.create_publisher(Path, '/lane_trajectory', 10)
        self.pub_debug_img = self.create_publisher(Image, '/lane_debug', 10)
        
        self.bridge = CvBridge()
        self.detector = LaneDetector()
        
        self.lane_state = 0 
        self.switch_start_time = 0
        self.lane_width = 0.4 
        
        self.create_timer(0.1, self.control_loop)
        self.switch_timer = self.create_timer(15.0, self.trigger_switch)

        self.latest_image = None
        self.get_logger().info("Lane Keeping Node Started - Max Path: 2m, Mode: Sharp Switch")

    def trigger_switch(self):
        if self.lane_state == 0:
            self.get_logger().info("INITIATING SHARP LANE SWITCH RIGHT")
            self.lane_state = 2 
            self.switch_start_time = time.time()
        elif self.lane_state != 0:
            self.get_logger().info("LANE SWITCH COMPLETE - RESUMING KEEP")
            self.lane_state = 0

    def image_cb(self, msg):
        self.latest_image = msg

    def control_loop(self):
        if self.latest_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        binary = self.detector.preprocess(cv_image)
        warped, Minv = self.detector.perspective_transform(binary)
        left_fit, right_fit = self.detector.find_lane_lines(warped)
        
        if left_fit is None or right_fit is None:
            return

        # --- LOGIC FOR SHARP SWITCHING ---
        offset = 0.0
        target_offset = 0.0
        
        if self.lane_state == 1: target_offset = -self.lane_width
        elif self.lane_state == 2: target_offset = self.lane_width
            
        if self.lane_state != 0:
            elapsed = time.time() - self.switch_start_time
            # Reduced duration for sharper reaction (1.5 seconds total)
            duration = 1.5 
            
            if elapsed < duration:
                t = elapsed / duration
                # Smoothstep function (3x^2 - 2x^3) creates an S-curve.
                # This makes the curvature sharp in the middle of the transition.
                ratio = t * t * (3 - 2 * t)
                offset = target_offset * ratio
            else:
                offset = target_offset
        # ----------------------------------

        target_x, target_y = self.detector.generate_trajectory_points(
            left_fit, right_fit, warped.shape[0], offset_meters=offset
        )

        path_msg = Path()
        path_msg.header = self.latest_image.header
        path_msg.header.frame_id = "front_camera_link" 
        
        for x_pix, y_pix in zip(target_x, target_y):
            img_h, img_w = warped.shape
            
            # Coordinates: Y pixel is Forward (X metric), X pixel is Lateral (Y metric)
            metric_x = (img_h - y_pix) * self.detector.ym_per_pix
            metric_y = (img_w/2 - x_pix) * self.detector.xm_per_pix
            
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(metric_x)
            pose.pose.position.y = float(metric_y) 
            pose.pose.position.z = 0.0 
            pose.pose.orientation.w = 1.0 
            
            path_msg.poses.append(pose)

        self.pub_path.publish(path_msg)
        
        for i in range(len(target_x)):
            # Draw visual feedback
            if 0 <= int(target_x[i]) < warped.shape[1] and 0 <= int(target_y[i]) < warped.shape[0]:
                cv2.circle(warped, (int(target_x[i]), int(target_y[i])), 5, (150), -1)
            
        debug_msg = self.bridge.cv2_to_imgmsg(warped, "mono8")
        self.pub_debug_img.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()