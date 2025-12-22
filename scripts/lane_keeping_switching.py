#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class LaneDetector:
    def __init__(self):
        # === TUNING SECTION ===
        
        # 1. Lane Width: Match this to your simulation (e.g., 0.5 or 0.6)
        self.lane_width_meters = 0.4  
        
        # 2. Manual Bias: Force the path left or right.
        # Positive (+) shifts path LEFT (fixes robot driving too far Right)
        # Negative (-) shifts path RIGHT (fixes robot driving too far Left)
        self.manual_bias = 0.00 
        
        # 3. Calibration
        self.ym_per_pix = 30 / 720  
        self.xm_per_pix = 2.7 / 700 

    def preprocess(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(hls, lower_white, upper_white)
        return mask

    def check_finish_line(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # --- CHANGED TO GREEN ---
        # Green is typically centered around Hue 60.
        # Range: Hue 40-80 covers most shades of green.
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # ------------------------

        h, w = mask.shape
        roi = mask[int(h * 0.75):h, :]
        return cv2.countNonZero(roi) > (roi.size * 0.05)

    def perspective_transform(self, img):
        h, w = img.shape[:2]
        src = np.float32([[w * 0.4, h * 0.65], [w * 0.6, h * 0.65], [w, h], [0, h]])
        dst = np.float32([[w * 0.2, 0], [w * 0.8, 0], [w * 0.8, h], [w * 0.2, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped, Minv

    # --- RESTORED METHOD ---
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
        
        peaks = sorted(peaks)
        final_peaks = []
        if peaks:
            final_peaks.append(peaks[0])
            for p in peaks[1:]:
                if p - final_peaks[-1] > min_dist:
                    final_peaks.append(p)
        return final_peaks

    # --- UPDATED ROBUST FITTING ---
    def get_fits(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        peaks = self.find_peaks(histogram)
        fits = []
        if len(peaks) < 1: return fits 

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
            
            # Robust Logic: Only fit if we have enough pixels
            if len(lane_inds) > 100:
                y_span = np.max(nonzeroy[lane_inds]) - np.min(nonzeroy[lane_inds])
                try:
                    # If line is short (<150px), use linear fit (degree 1) to avoid wild curves
                    if y_span < 150:
                        linear_fit = np.polyfit(nonzeroy[lane_inds], nonzerox[lane_inds], 1)
                        fit = np.array([0, linear_fit[0], linear_fit[1]])
                    else:
                        fit = np.polyfit(nonzeroy[lane_inds], nonzerox[lane_inds], 2)
                    fits.append(fit)
                except: pass
                    
        return fits

    def generate_path(self, fit_center, height, lookahead_meters=2.0):
        pixels_needed = lookahead_meters / self.ym_per_pix
        start_y = height - 1
        end_y = max(0, height - pixels_needed)
        ploty = np.linspace(start_y, end_y, num=20)
        fitx = fit_center[0]*ploty**2 + fit_center[1]*ploty + fit_center[2]
        return fitx, ploty

class LaneNode(Node):
    def __init__(self):
        super().__init__('lane_trajectory_manager')
        
        self.sub_img = self.create_subscription(Image, '/front_camera/image_raw', self.image_cb, 10)
        self.sub_trigger = self.create_subscription(PoseStamped, '/obstacle/blue_box_pose', self.trigger_cb, 10)
        
        self.pub_path = self.create_publisher(Path, '/lane_trajectory', 10)
        self.pub_debug = self.create_publisher(Image, '/lane_debug', 10)
        self.pub_finish = self.create_publisher(Bool, '/finish_line', 10)
        
        self.bridge = CvBridge()
        self.detector = LaneDetector()
        self.latest_image = None
        
        self.target_lane_index = 1 
        self.last_switch_time = 0
        self.switch_cooldown = 6.0 
        self.finish_line_triggered = False

        self.create_timer(0.01, self.control_loop)
        self.get_logger().info("Lane Manager Started.")

    def trigger_cb(self, msg:PoseStamped):
        current_time = time.time()
        if (current_time - self.last_switch_time) < self.switch_cooldown:
            return
        if msg.pose.position.x < 1.6 and abs(msg.pose.position.y) < 0.2:
            self.target_lane_index = 1 - self.target_lane_index
            self.last_switch_time = current_time
            self.get_logger().info("Switching Lane!")

    def image_cb(self, msg):
        self.latest_image = msg

    def control_loop(self):
        if self.latest_image is None: return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        except: return

        # 0. Finish Line
        is_finish = self.detector.check_finish_line(cv_image)
        self.pub_finish.publish(Bool(data=is_finish))
        if is_finish and not self.finish_line_triggered:
            self.get_logger().info("!!! FINISH LINE !!!")
            self.finish_line_triggered = True
        elif not is_finish:
            self.finish_line_triggered = False

        # 1. Detect Lanes
        binary = self.detector.preprocess(cv_image)
        warped, Minv = self.detector.perspective_transform(binary)
        fits = self.detector.get_fits(warped)
        
        height, width = warped.shape[:2]
        debug_img = cv_image.copy()

        if len(fits) == 0:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
            return 

        # 2. Logic: Ensure Equidistance
        bottom_x_positions = []
        for fit in fits:
            x_val = fit[0]*(height-1)**2 + fit[1]*(height-1) + fit[2]
            bottom_x_positions.append(x_val)
        
        sorted_indices = np.argsort(bottom_x_positions)
        sorted_fits = [fits[i] for i in sorted_indices]
        
        target_poly_center = None
        
        # --- Strict Logic ---
        if len(sorted_fits) >= 2:
            lane_center_x = (bottom_x_positions[sorted_indices[0]] + bottom_x_positions[sorted_indices[1]]) / 2
            is_left_visual_lane = lane_center_x < (width / 2)
            if len(sorted_fits) >= 3:
                self.get_logger().info(f"Mode: 3 Lines (Gap: {abs(bottom_x_positions[0] - bottom_x_positions[1]):.1f}px)")
                if self.target_lane_index == 0:
                     target_poly_center = (sorted_fits[0] + sorted_fits[1]) / 2.0
                else:
                     target_poly_center = (sorted_fits[1] + sorted_fits[2]) / 2.0
            else:
                self.get_logger().info(f"Mode: 2 Lines (Gap: {abs(bottom_x_positions[0] - bottom_x_positions[1]):.1f}px)")
                detected_center = (sorted_fits[0] + sorted_fits[1]) / 2.0
                if self.target_lane_index == 0: 
                    if is_left_visual_lane: target_poly_center = detected_center 
                    else:
                        shift_px = self.detector.lane_width_meters / self.detector.xm_per_pix
                        target_poly_center = detected_center.copy()
                        target_poly_center[2] -= shift_px
                else: 
                    if not is_left_visual_lane: target_poly_center = detected_center 
                    else:
                        shift_px = self.detector.lane_width_meters / self.detector.xm_per_pix
                        target_poly_center = detected_center.copy()
                        target_poly_center[2] += shift_px

        else:
            # 1 Line Visible
            self.get_logger().info("Mode: 1 Line (Using Width Guess)")
            target_poly_center = sorted_fits[0].copy()
            shift_px = (self.detector.lane_width_meters / self.detector.xm_per_pix) / 2
            line_x = bottom_x_positions[0]
            if line_x < width/2:
                target_poly_center[2] += shift_px 
            else:
                target_poly_center[2] -= shift_px 

        # 4. Generate Path
        if target_poly_center is not None:
            tx, ty = self.detector.generate_path(target_poly_center, height)
            
            path_msg = Path()
            path_msg.header = self.latest_image.header
            path_msg.header.frame_id = "front_camera_link"
            
            for i in range(len(tx)):
                metric_x = (height - ty[i]) * self.detector.ym_per_pix 
                
                # --- APPLY MANUAL BIAS HERE ---
                raw_metric_y = (width/2 - tx[i]) * self.detector.xm_per_pix
                final_metric_y = raw_metric_y + self.detector.manual_bias
                
                pose = PoseStamped()
                pose.pose.position.x = float(metric_x)
                pose.pose.position.y = float(final_metric_y)
                path_msg.poses.append(pose)
            
            self.pub_path.publish(path_msg)

            # --- DEBUG VISUALIZATION ---
            cv2.line(debug_img, (int(width/2), 0), (int(width/2), height), (255, 0, 0), 2)
            bias_px = self.detector.manual_bias / self.detector.xm_per_pix
            visual_tx = tx - bias_px 
            pts = np.column_stack((visual_tx, ty)).astype(np.int32)
            cv2.polylines(debug_img, [pts], False, (0, 255, 0), 4)
            if self.finish_line_triggered:
                cv2.putText(debug_img, "FINISH LINE!", (int(width/2) - 150, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            # status = f"Bias: {self.detector.manual_bias:.2f}m"
            # cv2.putText(debug_img, status, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()