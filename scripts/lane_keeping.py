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

class LaneDetector:
    def __init__(self):
        # Physical parameters
        self.lane_width_meters = 3.5 # Standard road lane width (adjust for your robot scale, e.g. 0.4 for small bots)
        
        # Camera Calibration (Tune these!)
        self.ym_per_pix = 30 / 720  
        self.xm_per_pix = 3.7 / 700 

    def preprocess(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(hls, lower_white, upper_white)
        return mask

    def perspective_transform(self, img):
        h, w = img.shape[:2]
        # Trapizoid for "Bird's Eye View"
        src = np.float32([[w * 0.4, h * 0.65], [w * 0.6, h * 0.65], [w, h], [0, h]])
        dst = np.float32([[w * 0.2, 0], [w * 0.8, 0], [w * 0.8, h], [w * 0.2, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped, Minv

    def find_peaks(self, histogram, threshold=10, min_dist=100):
        indices = np.where(histogram > threshold)[0]
        peaks = []
        if len(indices) > 0:
            current_cluster = [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] - indices[i-1] < 50:
                    current_cluster.append(indices[i])
                else:
                    peaks.append(current_cluster[np.argmax(histogram[current_cluster])])
                    current_cluster = [indices[i]]
            peaks.append(current_cluster[np.argmax(histogram[current_cluster])])
        
        # Filter strictly by distance
        peaks = sorted(peaks)
        final_peaks = []
        if peaks:
            final_peaks.append(peaks[0])
            for p in peaks[1:]:
                if p - final_peaks[-1] > min_dist:
                    final_peaks.append(p)
        return final_peaks

    def get_fits(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        peaks = self.find_peaks(histogram)
        
        fits = []
        if len(peaks) < 1: return fits # Need at least one line

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        for start_x in peaks:
            lane_inds = []
            current_x = start_x
            window_height = int(binary_warped.shape[0] / 9)
            
            for window in range(9):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_x_low = current_x - 80
                win_x_high = current_x + 80
                
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                lane_inds.append(good_inds)
                if len(good_inds) > 50:
                    current_x = int(np.mean(nonzerox[good_inds]))
            
            lane_inds = np.concatenate(lane_inds)
            if len(lane_inds) > 0:
                try:
                    fit = np.polyfit(nonzeroy[lane_inds], nonzerox[lane_inds], 2)
                    fits.append(fit)
                except: pass
        return fits

    def generate_path(self, fit_center, height, lookahead_meters=3.0):
        """Generates trajectory points for 3 meters ahead"""
        pixels_needed = lookahead_meters / self.ym_per_pix
        
        # Generate points from bottom (robot) to 3m ahead
        start_y = height - 1
        end_y = max(0, height - pixels_needed)
        
        ploty = np.linspace(start_y, end_y, num=20)
        fitx = fit_center[0]*ploty**2 + fit_center[1]*ploty + fit_center[2]
        
        return fitx, ploty

class LaneNode(Node):
    def __init__(self):
        super().__init__('lane_trajectory_manager')
        
        # Topics
        self.sub_img = self.create_subscription(Image, '/front_camera/image_raw', self.image_cb, 10)
        self.sub_trigger = self.create_subscription(PoseStamped, '/lane_switch_trigger', self.trigger_cb, 10)
        
        self.pub_path = self.create_publisher(Path, '/lane_trajectory', 10)
        self.pub_debug = self.create_publisher(Image, '/lane_debug', 10)
        
        self.bridge = CvBridge()
        self.detector = LaneDetector()
        self.latest_image = None
        
        # === STATE MACHINE ===
        # 0 = Left Lane, 1 = Right Lane
        self.target_lane_index = 0 
        self.last_switch_time = 0
        self.switch_cooldown = 15.0 # Seconds before allowing another switch
        
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Lane Manager Started. Default: Left Lane (0). Waiting for Trigger...")

    def trigger_cb(self, msg):
        """Logic: If x < 0.4, Switch Lanes"""
        current_time = time.time()
        
        # Check Cooldown to prevent flickering
        if (current_time - self.last_switch_time) < self.switch_cooldown:
            return

        if msg.pose.position.x < 0.8:
            # TOGGLE LANE (0 -> 1, or 1 -> 0)
            prev_lane = self.target_lane_index
            self.target_lane_index = 1 - self.target_lane_index
            
            self.last_switch_time = current_time
            self.get_logger().warn(f"TRIGGER RECEIVED (x={msg.pose.position.x:.2f}). Switching Lane: {prev_lane} -> {self.target_lane_index}")

    def image_cb(self, msg):
        self.latest_image = msg

    def control_loop(self):
        if self.latest_image is None: return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        except: return

        # 1. Detect Lines
        binary = self.detector.preprocess(cv_image)
        warped, Minv = self.detector.perspective_transform(binary)
        fits = self.detector.get_fits(warped)
        
        if len(fits) == 0: return # No lines found

        height, width = warped.shape[:2]
        
        # 2. Determine "Real" Lanes from Lines
        # A "Lane" is the space between two fit lines.
        # We need to map detected lines to our abstract "Lane 0" and "Lane 1" concept.
        
        # Heuristic: Calculate x-position of all lines at the bottom of image
        bottom_x_positions = []
        for fit in fits:
            x_val = fit[0]*(height-1)**2 + fit[1]*(height-1) + fit[2]
            bottom_x_positions.append(x_val)
        
        # Sort fits by their position from Left to Right
        sorted_indices = np.argsort(bottom_x_positions)
        sorted_fits = [fits[i] for i in sorted_indices]
        
        # 3. Select Target Path
        target_poly_center = None
        
        # Case A: We see 3 lines (Perfect visibility of both lanes)
        if len(sorted_fits) >= 3:
            if self.target_lane_index == 0: # Target Left
                left_line = sorted_fits[0]
                right_line = sorted_fits[1]
            else: # Target Right
                left_line = sorted_fits[1]
                right_line = sorted_fits[2]
                
            # Average to find center
            target_poly_center = (left_line + right_line) / 2.0

        # Case B: We see 2 lines (Only 1 Lane visible)
        elif len(sorted_fits) == 2:
            # Is this the left lane or right lane?
            lane_center_x = (bottom_x_positions[sorted_indices[0]] + bottom_x_positions[sorted_indices[1]]) / 2
            
            is_left_visual_lane = lane_center_x < (width / 2)
            
            detected_poly_center = (sorted_fits[0] + sorted_fits[1]) / 2.0
            
            if self.target_lane_index == 0: # WE WANT LEFT
                if is_left_visual_lane:
                    target_poly_center = detected_poly_center # We have it
                else:
                    # We see Right, but want Left -> Generate Virtual Left Lane
                    # Subtract lane width (in pixels)
                    pixel_offset = self.detector.lane_width_meters / self.detector.xm_per_pix
                    target_poly_center = detected_poly_center.copy()
                    target_poly_center[2] -= pixel_offset # Shift Left
                    
            else: # WE WANT RIGHT
                if not is_left_visual_lane:
                    target_poly_center = detected_poly_center # We have it
                else:
                    # We see Left, but want Right -> Generate Virtual Right Lane
                    pixel_offset = self.detector.lane_width_meters / self.detector.xm_per_pix
                    target_poly_center = detected_poly_center.copy()
                    target_poly_center[2] += pixel_offset # Shift Right

        # Case C: 1 Line (Guess work, usually assume current lane center offset)
        else:
             target_poly_center = sorted_fits[0].copy() # Fallback to tracking the line itself
             target_poly_center[2] += (self.detector.lane_width_meters / self.detector.xm_per_pix) / 2 # Offset to center

        # 4. Generate & Publish Path (3 Meters Ahead)
        if target_poly_center is not None:
            tx, ty = self.detector.generate_path(target_poly_center, height, lookahead_meters=3.0)
            
            path_msg = Path()
            path_msg.header = self.latest_image.header
            path_msg.header.frame_id = "front_camera_link"
            
            for i in range(len(tx)):
                # Convert Pixels to Meters for ROS
                metric_x = (height - ty[i]) * self.detector.ym_per_pix # Forward
                metric_y = (width/2 - tx[i]) * self.detector.xm_per_pix # Lateral
                
                pose = PoseStamped()
                pose.pose.position.x = float(metric_x)
                pose.pose.position.y = float(metric_y)
                path_msg.poses.append(pose)
            
            self.pub_path.publish(path_msg)
            
            # --- DEBUG VISUALIZATION ---
            # Draw Target Lane in GREEN
            debug_img = cv_image.copy()
            pts = np.column_stack((tx, ty)).astype(np.int32)
            cv2.polylines(debug_img, [pts], False, (0, 255, 0), 5)
            
            # Text Status
            status = f"Mode: {'RIGHT' if self.target_lane_index else 'LEFT'} Lane"
            cv2.putText(debug_img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()