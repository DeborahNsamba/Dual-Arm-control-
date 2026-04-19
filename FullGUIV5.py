
"""
#!/usr/bin/env python3
Simplified Dual-Arm Robot Controller - Auto Detection Only
Uses YOLO for ping-pong ball detection and moves the appropriate arm
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import math
import serial
import cv2
import numpy as np
from PIL import Image, ImageTk
import argparse
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

# Set to True to show debug overlay on camera frames
DEBUG_MODE = False

# Stereo calibration parameters
MTX_LEFT = np.array([[388.04660798, 0, 304.37138089],
                     [0, 389.20101089, 235.73470932],
                     [0, 0, 1]])
MTX_RIGHT = np.array([[424.01881092, 0, 310.6237663],
                      [0, 425.86194639, 233.50324676],
                      [0, 0, 1]])
DIST_LEFT = np.array([0.00738529, -0.05270355, -0.00116587, -0.00049622, 0.050802])
DIST_RIGHT = np.array([-0.02672905, 0.1576315, 0.00056655, -0.00172501, -0.39624642])
R = np.array([[ 9.99966820e-01, -7.02165770e-04,  8.11572139e-03],
              [ 8.31979959e-04,  9.99871596e-01, -1.60031054e-02],
              [-8.10344247e-03,  1.60093264e-02,  9.99839005e-01]])
T = np.array([[-7.53400414], [0.24089944], [2.92889796]])

# Rectified parameters
FX = 558.3599
CX = 580.0999
CY = 238.4877
BASELINE = 8.09

# Robot geometry
L1 = 18.0
L2 = 14.0

# Right arm
RIGHT_CAM_OFFSET_RIGHT_CM = 16.0
RIGHT_CAM_OFFSET_UP_CM = 10.0
RIGHT_HOME_ANGLES = {"J0": 0.0, "J1": 135.0, "J2": 98.6, "J3": 86.0}
RIGHT_SERVO_CHANNELS = [16, 17, 18, 19]

# Left arm
LEFT_CAM_OFFSET_RIGHT_CM = -16.0
LEFT_CAM_OFFSET_UP_CM = 10.0
LEFT_HOME_ANGLES = {"J0": 0.0, "J1": 135.0, "J2": 97, "J3": 98.6}
LEFT_SERVO_CHANNELS = [28, 29, 30, 31]

# Servo configuration
SERVOS_CONFIG = {
    "RIGHT_J0": {"id": 16, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
    "RIGHT_J1": {"id": 17, "min_angle": 120.0, "max_angle": 270, "min_pulse": 1390, "max_pulse": 24, "offset": 0.0, "invert": False},
    "RIGHT_J2": {"id": 18, "min_angle": 0.0, "max_angle": 133.6, "min_pulse": 530, "max_pulse": 1490, "offset": 0.0, "invert": False},
    "RIGHT_J3": {"id": 19, "min_angle": 26.77, "max_angle": 117.23, "min_pulse": 790, "max_pulse": 1770, "offset": 0.0, "invert": False},
    "LEFT_J0": {"id": 28, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": True},
    "LEFT_J1": {"id": 29, "min_angle": 55.6, "max_angle": 270.0, "min_pulse": 925, "max_pulse": 2450, "offset": 0.0, "invert": False},
    "LEFT_J2": {"id": 30, "min_angle": 0.0, "max_angle": 142.1, "min_pulse": 530, "max_pulse": 1550, "offset": 0.0, "invert": False},
    "LEFT_J3": {"id": 31, "min_angle": 26.85, "max_angle": 172.62, "min_pulse": 1430, "max_pulse": 2370, "offset": 0.0, "invert": True},
}

# YOLO detection configuration
YOLO_MODEL_PATH = "model_nano.pt"
YOLO_CONF_THRESH = 0.5
YOLO_CLASS_NAME = "ping-pong-ball"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def clamp(v, a, b):
    return max(a, min(b, v))

def setup_stereo_rectification(image_size=(640, 480)):
    """Configure stereo rectification"""
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        MTX_LEFT, DIST_LEFT, MTX_RIGHT, DIST_RIGHT, 
        image_size, R, T, alpha=0
    )
    
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        MTX_LEFT, DIST_LEFT, R1, P1, image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        MTX_RIGHT, DIST_RIGHT, R2, P2, image_size, cv2.CV_16SC2
    )
    
    return left_map1, left_map2, right_map1, right_map2

def rectify_images(left_img, right_img, left_map1, left_map2, right_map1, right_map2):
    """Rectify stereo images"""
    left_rect = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    return left_rect, right_rect

def detect_balls(frame, model):
    """Run YOLO detection on a frame and return the best ping-pong-ball detection"""
    results = model(frame, conf=YOLO_CONF_THRESH, verbose=False)
    
    best_det = None
    best_conf = 0.0
    
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        result = results[0]
        class_names = result.names
        
        for box in result.boxes:
            class_id = int(box.cls.item())
            cls_name = class_names.get(class_id, "").lower().replace(" ", "-")
            if cls_name != YOLO_CLASS_NAME:
                continue
            
            conf = box.conf.item()
            if conf <= best_conf:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            radius = int(min(x2 - x1, y2 - y1) / 2)
            if radius < 5:
                continue
            
            best_conf = conf
            best_det = (int((x1 + x2) / 2), int((y1 + y2) / 2), radius)
    
    return best_det

def stereo_to_xyz_cm_rectified(detL, detR):
    """Convert stereo detection to 3D coordinates"""
    disparity = float(detL[0] - detR[0])
    if disparity <= 0.1:
        return None
        
    Z_cm = (FX * BASELINE) / disparity
    X_cm = (detL[0] - CX) * Z_cm / FX
    Y_cm = (detL[1] - CY) * Z_cm / FX
    
    return (X_cm, Y_cm, Z_cm)

def camera_to_robot_base(X_cam, Y_cam, Z_cam, camera_offset_right):
    """Convert camera coordinates to robot base frame"""
    forward = Z_cam
    right = X_cam + camera_offset_right
    up = -Y_cam + 10.0
    
    return forward, up, right

def compute_yaw_and_planar_coords(forward, right):
    """Compute yaw angle and planar distance"""
    yaw_rad = math.atan2(right, forward)
    yaw_deg = math.degrees(yaw_rad)
    r = math.hypot(forward, right)
    return yaw_deg, r

def planar_2link_ik(r_cm, up_cm, L1cm=18.0, L2cm=14.0, elbow_down=True):
    """Solve planar 2-link IK"""
    x = r_cm
    y = up_cm
    
    D_sq = x*x + y*y
    denom = 2 * L1cm * L2cm
    cos_q2 = (D_sq - L1cm*L1cm - L2cm*L2cm) / denom
    
    if cos_q2 < -1.0 or cos_q2 > 1.0:
        return None
        
    if elbow_down:
        q2 = math.acos(cos_q2)
    else:
        q2 = -math.acos(cos_q2)
        
    k1 = L1cm + L2cm * math.cos(q2)
    k2 = L2cm * math.sin(q2)
    q1 = math.atan2(y, x) - math.atan2(k2, k1)
    
    shoulder_deg = math.degrees(q1)
    elbow_deg = math.degrees(q2)
    
    return shoulder_deg, elbow_deg

# ============================================================================
# ROBOT ARM CLASS
# ============================================================================
class RobotArm:
    def __init__(self, side="right", ser=None):
        self.side = side
        self.ser = ser
        
        if side == "right":
            self.servo_channels = RIGHT_SERVO_CHANNELS
            self.home_angles = RIGHT_HOME_ANGLES.copy()
            self.camera_offset_right = RIGHT_CAM_OFFSET_RIGHT_CM
            self.prefix = "RIGHT"
        else:
            self.servo_channels = LEFT_SERVO_CHANNELS
            self.home_angles = LEFT_HOME_ANGLES.copy()
            self.camera_offset_right = LEFT_CAM_OFFSET_RIGHT_CM
            self.prefix = "LEFT"
        
        self.current_angles = self.home_angles.copy()
    
    def angle_to_pulse(self, angle_deg, joint_name):
        """Convert angle to pulse width"""
        servo_key = f"{self.prefix}_J{joint_name[-1]}"
        cfg = SERVOS_CONFIG[servo_key]
        
        angle = angle_deg + cfg.get("offset", 0.0)
        if cfg.get("invert", False):
            angle = 270 - angle
        
        angle = clamp(angle, cfg["min_angle"], cfg["max_angle"])
        ratio = (angle - cfg["min_angle"]) / (cfg["max_angle"] - cfg["min_angle"])
        pulse = int(cfg["min_pulse"] + ratio * (cfg["max_pulse"] - cfg["min_pulse"]))
        return pulse
    
    def send_servo_positions(self, angles_deg, duration_ms=500):
        """Send servo positions to hardware"""
        if self.ser is None:
            return
        
        parts = []
        for joint_name, angle in angles_deg.items():
            servo_key = f"{self.prefix}_J{joint_name[-1]}"
            cfg = SERVOS_CONFIG[servo_key]
            pulse = self.angle_to_pulse(angle, joint_name)
            parts.append(f"#{cfg['id']} P{pulse}")
        
        cmd = " ".join(parts) + f" T{int(duration_ms)}\r"
        try:
            self.ser.write(cmd.encode('ascii'))
        except Exception as e:
            print(f"Serial write error ({self.side}):", e)
    
    def smooth_move(self, target_angles, step_deg=4.0, step_time_ms=400):
        """Move smoothly to target angles"""
        deltas = [abs(target_angles[j] - self.current_angles[j]) for j in ("J0", "J1", "J2", "J3")]
        max_delta = max(deltas)
        
        if max_delta < 0.5:
            self.current_angles = target_angles.copy()
            return
        
        steps = max(1, int(math.ceil(max_delta / step_deg)))
        
        for s in range(1, steps + 1):
            interp = {}
            for j in ("J0", "J1", "J2", "J3"):
                interp[j] = self.current_angles[j] + (target_angles[j] - self.current_angles[j]) * (s/steps)
            
            self.send_servo_positions(interp, duration_ms=step_time_ms)
            time.sleep(max(0.01, step_time_ms/1000.0 * 0.9))
        
        self.current_angles = target_angles.copy()
    
    def move_to_home(self):
        """Move arm to home position"""
        self.smooth_move(self.home_angles, step_deg=5, step_time_ms=500)
    
    def calculate_ik(self, xyz_cam):
        """Calculate IK for target position"""
        if xyz_cam is None:
            return None
        
        Xc, Yc, Zc = xyz_cam
        forward, up, right = camera_to_robot_base(Xc, Yc, Zc, self.camera_offset_right)
        yaw_deg, r_cm = compute_yaw_and_planar_coords(forward, right)
        
        ik = planar_2link_ik(r_cm, up, L1cm=18.0, L2cm=14.0, elbow_down=True)
        if ik is None:
            return None
        
        shoulder_deg, elbow_deg = ik
        
        # Map to servo angles
        if self.side == "right":
            target_angles = {
                "J0": clamp(shoulder_deg + 90.0, 0, 270),
                "J1": clamp(yaw_deg + 135.0, 0, 157.27),
                "J2": clamp(elbow_deg, 0, 133.6),
                "J3": self.home_angles["J3"]
            }
        else:
            target_angles = {
                "J0": clamp(shoulder_deg + 90.0, 0, 270),
                "J1": clamp(yaw_deg + 135.0, 55.6, 270),
                "J2": clamp(elbow_deg, 0, 142.1),
                "J3": self.home_angles["J3"]
            }
        
        return target_angles

# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================
class SimpleRobotController:
    def __init__(self, root):
        self.root = root
        root.title("Auto Ball Detection & Grab")
        root.geometry("1000x700")
        
        # Colors
        self.bg_color = "#252323"
        self.frame_bg = "#3c3f41"
        self.text_color = "#ffffff"
        root.configure(bg=self.bg_color)
        
        # Initialize serial
        self.ser = None
        self.init_serial()
        
        # Initialize robot arms
        self.right_arm = RobotArm("right", self.ser)
        self.left_arm = RobotArm("left", self.ser)
        
        # Load YOLO model
        print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded.")
        
        # Camera variables
        self.cap = None
        self.camera_running = False
        self.auto_tracking = False
        self.left_map1 = self.left_map2 = self.right_map1 = self.right_map2 = None
        self.left_frame_rect = None
        self.right_frame_rect = None
        
        # Create UI
        self.create_ui()
        
        # Move arms to home position
        self.move_arms_to_home()
    
    def init_serial(self):
        """Initialize serial connection"""
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5)
            time.sleep(0.1)
            print(f"Opened serial on {SERIAL_PORT}")
        except Exception as e:
            print(f"Could not open serial port: {e}")
            print("Continuing in debug mode (no hardware moves).")
            self.ser = None
    
    def create_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Camera display frames
        camera_frame = tk.Frame(main_frame, bg=self.bg_color)
        camera_frame.pack(fill="both", expand=True)
        camera_frame.grid_columnconfigure(0, weight=1)
        camera_frame.grid_columnconfigure(1, weight=1)
        camera_frame.grid_rowconfigure(0, weight=1)
        
        # Left camera label
        left_frame = tk.LabelFrame(camera_frame, text="Left Camera", bg=self.frame_bg, fg=self.text_color, font=("Arial", 10, "bold"))
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        self.left_cam_label = tk.Label(left_frame, bg="black", relief="solid", bd=2)
        self.left_cam_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Right camera label
        right_frame = tk.LabelFrame(camera_frame, text="Right Camera", bg=self.frame_bg, fg=self.text_color, font=("Arial", 10, "bold"))
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        self.right_cam_label = tk.Label(right_frame, bg="black", relief="solid", bd=2)
        self.right_cam_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg=self.frame_bg, relief="ridge", bd=2)
        control_frame.pack(fill="x", pady=10)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg=self.frame_bg)
        button_frame.pack(pady=10)
        
        self.start_stop_button = tk.Button(
            button_frame,
            text="Start Auto Detection",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.toggle_auto_tracking,
            width=20
        )
        self.start_stop_button.pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_frame,
            text="Move Arms to Home",
            bg="#ff8c00",
            fg="black",
            font=("Arial", 12, "bold"),
            command=self.move_arms_to_home,
            width=20
        ).pack(side=tk.LEFT, padx=10)
        
        # Status labels
        status_frame = tk.Frame(control_frame, bg=self.frame_bg)
        status_frame.pack(pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="Status: Ready",
            bg=self.frame_bg,
            fg="white",
            font=("Arial", 11, "bold")
        )
        self.status_label.pack()
        
        self.right_status = tk.Label(
            status_frame,
            text="Right Arm: Idle",
            bg=self.frame_bg,
            fg="#888888",
            font=("Arial", 10)
        )
        self.right_status.pack()
        
        self.left_status = tk.Label(
            status_frame,
            text="Left Arm: Idle",
            bg=self.frame_bg,
            fg="#888888",
            font=("Arial", 10)
        )
        self.left_status.pack()
        
        self.coord_label = tk.Label(
            status_frame,
            text="Ball Position: --, --, -- cm",
            bg=self.frame_bg,
            fg="#888888",
            font=("Arial", 10)
        )
        self.coord_label.pack()
    
    def move_arms_to_home(self):
        """Move both arms to home position"""
        self.right_arm.move_to_home()
        self.left_arm.move_to_home()
        self.right_status.config(text="Right Arm: Moving to home", fg="#ff8c00")
        self.left_status.config(text="Left Arm: Moving to home", fg="#ff8c00")
        self.root.after(3000, lambda: self.right_status.config(text="Right Arm: Idle", fg="#888888"))
        self.root.after(3000, lambda: self.left_status.config(text="Left Arm: Idle", fg="#888888"))
    
    def toggle_auto_tracking(self):
        """Toggle auto tracking on/off"""
        if self.auto_tracking:
            self.stop_auto_tracking()
        else:
            self.start_auto_tracking()
    
    def start_auto_tracking(self):
        """Start auto tracking"""
        if not self.camera_running:
            if not self.initialize_camera():
                self.status_label.config(text="Status: Camera Error", fg="red")
                return
        
        self.auto_tracking = True
        self.start_stop_button.config(text="Stop Auto Detection", bg="red")
        self.status_label.config(text="Status: Tracking", fg="green")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def stop_auto_tracking(self):
        """Stop auto tracking"""
        self.auto_tracking = False
        self.start_stop_button.config(text="Start Auto Detection", bg="#4CAF50")
        self.status_label.config(text="Status: Stopped", fg="red")
        self.right_status.config(text="Right Arm: Idle", fg="#888888")
        self.left_status.config(text="Left Arm: Idle", fg="#888888")
        self.coord_label.config(text="Ball Position: --, --, -- cm", fg="#888888")
        
        # Move arms to home
        self.move_arms_to_home()
    
    def initialize_camera(self):
        """Initialize stereo camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                print("Camera not found")
                return False
            
            # Setup stereo rectification
            image_size = (640, 480)
            self.left_map1, self.left_map2, self.right_map1, self.right_map2 = setup_stereo_rectification(image_size)
            
            self.camera_running = True
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def get_arm_side(self, x_cm):
        """Determine which arm to use based on X coordinate"""
        if x_cm > 0:
            return "right"
        else:
            return "left"
    
    def camera_loop(self):
        """Main camera processing loop"""
        frame_count = 0
        
        while self.camera_running and self.auto_tracking:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, lambda: self.status_label.config(text="Status: Camera Error", fg="red"))
                break
            
            frame_count += 1
            
            # Split stereo frame
            h, w = frame.shape[:2]
            mid = w // 2
            left_raw = frame[:, :mid]
            right_raw = frame[:, mid:]
            
            # Resize if needed
            if left_raw.shape[1] != 640 or left_raw.shape[0] != 480:
                left_raw = cv2.resize(left_raw, (640, 480))
                right_raw = cv2.resize(right_raw, (640, 480))
            
            # Rectify images
            left_rect, right_rect = rectify_images(
                left_raw, right_raw,
                self.left_map1, self.left_map2,
                self.right_map1, self.right_map2
            )
            
            # Run YOLO detection
            if frame_count < 2:
                self.left_frame_rect = detect_balls(left_rect, self.yolo_model)
                self.right_frame_rect = detect_balls(right_rect, self.yolo_model)
            else:
                if frame_count % 2 == 0:
                    self.left_frame_rect = detect_balls(left_rect, self.yolo_model)
                else:
                    self.right_frame_rect = detect_balls(right_rect, self.yolo_model)
            
            det_L = self.left_frame_rect
            det_R = self.right_frame_rect
            
            # Process detections
            if det_L and det_R:
                # Both cameras see the ball - compute stereo 3D position
                xyz = stereo_to_xyz_cm_rectified(det_L, det_R)
                if xyz:
                    X_cm, Y_cm, Z_cm = xyz
                    self.root.after(0, lambda x=X_cm, y=Y_cm, z=Z_cm: 
                        self.coord_label.config(text=f"Ball Position: X:{x:+.1f}, Y:{y:+.1f}, Z:{z:.1f} cm", fg="#4CAF50"))
                    
                    # Determine which arm to use
                    arm_side = self.get_arm_side(X_cm)
                    
                    if arm_side == "right" and self.right_arm:
                        self.root.after(0, lambda: self.right_status.config(text="Right Arm: Ball detected - moving to grab", fg="green"))
                        target_angles = self.right_arm.calculate_ik(xyz)
                        if target_angles:
                            self.right_arm.smooth_move(target_angles, step_deg=4.0, step_time_ms=300)
                            time.sleep(2)  # Pause to grab
                            self.right_arm.move_to_home()
                            self.root.after(0, lambda: self.right_status.config(text="Right Arm: Returned to home", fg="#888888"))
                    
                    elif arm_side == "left" and self.left_arm:
                        self.root.after(0, lambda: self.left_status.config(text="Left Arm: Ball detected - moving to grab", fg="green"))
                        target_angles = self.left_arm.calculate_ik(xyz)
                        if target_angles:
                            self.left_arm.smooth_move(target_angles, step_deg=4.0, step_time_ms=300)
                            time.sleep(2)  # Pause to grab
                            self.left_arm.move_to_home()
                            self.root.after(0, lambda: self.left_status.config(text="Left Arm: Returned to home", fg="#888888"))
                    
                    self.root.after(0, lambda: self.right_status.config(text="Right Arm: Idle", fg="#888888"))
                    self.root.after(0, lambda: self.left_status.config(text="Left Arm: Idle", fg="#888888"))
                else:
                    self.root.after(0, lambda: self.coord_label.config(text="Ball Position: Invalid disparity", fg="red"))
            else:
                self.root.after(0, lambda: self.coord_label.config(text="Ball Position: Not detected", fg="red"))
                self.root.after(0, lambda: self.right_status.config(text="Right Arm: No ball", fg="red"))
                self.root.after(0, lambda: self.left_status.config(text="Left Arm: No ball", fg="red"))
            
            # Draw detections on display frames
            display_left = left_rect.copy()
            display_right = right_rect.copy()
            
            if det_L:
                cx, cy, r = det_L
                cv2.circle(display_left, (cx, cy), r, (0, 255, 0), 2)
                cv2.putText(display_left, "BALL", (cx - 20, cy - r - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if det_R:
                cx, cy, r = det_R
                cv2.circle(display_right, (cx, cy), r, (0, 255, 0), 2)
                cv2.putText(display_right, "BALL", (cx - 20, cy - r - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Debug overlay
            if DEBUG_MODE and det_L and det_R:
                disparity = float(det_L[0] - det_R[0])
                cv2.putText(display_left, f"Disparity: {disparity:.1f} px", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
            
            # Convert to RGB for display
            left_rgb = cv2.cvtColor(display_left, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(display_right, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            left_img = Image.fromarray(left_rgb)
            right_img = Image.fromarray(right_rgb)
            
            # Convert to ImageTk
            left_tk = ImageTk.PhotoImage(image=left_img)
            right_tk = ImageTk.PhotoImage(image=right_img)
            
            # Update GUI in main thread
            self.root.after(0, self.update_camera_display, left_tk, right_tk)
            
            # Small delay
            time.sleep(0.03)
    
    def update_camera_display(self, left_img, right_img):
        """Update camera display in GUI"""
        self.left_cam_label.config(image=left_img)
        self.left_cam_label.image = left_img
        self.right_cam_label.config(image=right_img)
        self.right_cam_label.image = right_img
    
    def cleanup(self):
        """Cleanup resources"""
        self.camera_running = False
        self.auto_tracking = False
        
        if self.cap:
            self.cap.release()
        
        if self.ser:
            self.ser.close()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Ball Detection Robot")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlay on camera frames")
    args = parser.parse_args()
    
    if args.debug:
        DEBUG_MODE = True
        print("[DEBUG MODE ENABLED] Camera frame debug overlay is ON")
    
    root = tk.Tk()
    app = SimpleRobotController(root)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()