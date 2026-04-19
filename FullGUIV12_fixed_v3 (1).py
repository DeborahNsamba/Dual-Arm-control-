
"""
#!/usr/bin/env python3
Simplified Dual-Arm Robot Controller - Auto Detection Only
Uses YOLO for ping-pong ball detection and moves the appropriate arm
Includes battery voltage and current monitoring with TTS for unreachable objects
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
import random
import pyttsx3
from inference_tflite import TFLiteDetector
import tensorflow as tf

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

# Reachable workspace limits (in cm)
MIN_REACH = 10.0  # Minimum reachable distance
MAX_REACH = 30.0  # Maximum reachable distance
MIN_HEIGHT = 5.0  # Minimum height (negative Y means above camera)
MAX_HEIGHT = 25.0  # Maximum height

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
    "RIGHT_J1": {"id": 17, "min_angle": 120.0, "max_angle": 270, "min_pulse": 1390, "max_pulse": 2450, "offset": 0.0, "invert": False},
    "RIGHT_J2": {"id": 18, "min_angle": 0.0, "max_angle": 133.6, "min_pulse": 530, "max_pulse": 1490, "offset": 0.0, "invert": False},
    "RIGHT_J3": {"id": 19, "min_angle": 26.77, "max_angle": 117.23, "min_pulse": 790, "max_pulse": 1770, "offset": 0.0, "invert": False},
    "LEFT_J0": {"id": 28, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": True},
    "LEFT_J1": {"id": 29, "min_angle": 55.6, "max_angle": 270.0, "min_pulse": 925, "max_pulse": 2450, "offset": 0.0, "invert": False},
    "LEFT_J2": {"id": 30, "min_angle": 0.0, "max_angle": 142.1, "min_pulse": 530, "max_pulse": 1550, "offset": 0.0, "invert": False},
    "LEFT_J3": {"id": 31, "min_angle": 26.85, "max_angle": 172.62, "min_pulse": 1430, "max_pulse": 2370, "offset": 0.0, "invert": True},
}

# YOLO detection configuration
YOLO_MODEL_PATH = "model_nano_saved_model/model_nano_int8.tflite"
YOLO_CONF_THRESH = 0.5
YOLO_CLASS_NAME = "ping-pong-ball"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def clamp(v, a, b):
    return max(a, min(b, v))

def is_position_reachable(x_cm, y_cm, z_cm):
    """Check if the ball position is within reachable workspace"""
    # Calculate distance from arm base
    distance = math.sqrt(x_cm**2 + z_cm**2)
    
    # Check reach constraints
    if distance < MIN_REACH:
        return False, f"Too close: {distance:.1f}cm < {MIN_REACH}cm"
    if distance > MAX_REACH:
        return False, f"Too far: {distance:.1f}cm > {MAX_REACH}cm"
    if y_cm < MIN_HEIGHT:
        return False, f"Too low: {y_cm:.1f}cm < {MIN_HEIGHT}cm"
    if y_cm > MAX_HEIGHT:
        return False, f"Too high: {y_cm:.1f}cm > {MAX_HEIGHT}cm"
    
    return True, "Reachable"

def speak_unreachable():
    """Use text-to-speech to announce unreachable object"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say("Object introuvable.")
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

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
    """Run TFLite detection on a frame and return the best ping-pong-ball detection"""
    return model.detect(frame)

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
        root.title("Auto Ball Detection & Grab with Battery Monitor")
        root.geometry("1100x800")
        
        # Colors
        self.bg_color = "#252323"
        self.frame_bg = "#3c3f41"
        self.text_color = "#ffffff"
        self.voltage_color = "#4CAF50"
        self.current_color = "#2196F3"
        self.power_color = "#FF9800"
        root.configure(bg=self.bg_color)
        
        # Initialize serial
        self.ser = None
        self.init_serial()
        
        # Initialize robot arms
        self.right_arm = RobotArm("right", self.ser)
        self.left_arm = RobotArm("left", self.ser)
        
        # Load TFLite model
        print(f"Loading TFLite model from: {YOLO_MODEL_PATH}")
        self.yolo_model = TFLiteDetector(YOLO_MODEL_PATH, conf_thresh=YOLO_CONF_THRESH)
        print("TFLite model loaded.")
        
        # Battery monitoring variables
        self.total_power_used = 0.0  # Watt-hours
        self.last_update_time = time.time()
        
        # Camera variables
        self.cap = None
        self.camera_running = False
        self.auto_tracking = False
        self.left_map1 = self.left_map2 = self.right_map1 = self.right_map2 = None
        self.left_frame_rect = None
        self.right_frame_rect = None
        
        # TTS engine (initialize once)
        self.tts_engine = None
        self.init_tts()
        
        # Create UI
        self.create_ui()
        
        # Start battery monitoring
        self.update_battery()
        
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
    
    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 1.0)
            print("TTS initialized")
        except Exception as e:
            print(f"TTS initialization error: {e}")
            self.tts_engine = None
    
    def speak(self, message):
        """Speak a message using TTS"""
        if self.tts_engine:
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def create_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Split into left (camera) and right (battery) panels
        panels_frame = tk.Frame(main_frame, bg=self.bg_color)
        panels_frame.pack(fill="both", expand=True)
        panels_frame.grid_columnconfigure(0, weight=3)
        panels_frame.grid_columnconfigure(1, weight=1)
        panels_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Camera feeds
        camera_panel = tk.Frame(panels_frame, bg=self.bg_color)
        camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Camera display frames
        camera_frame = tk.Frame(camera_panel, bg=self.bg_color)
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
        
        # Right panel - Battery monitoring
        battery_panel = tk.LabelFrame(panels_frame, text="Battery Monitor", bg=self.frame_bg, fg=self.text_color, font=("Arial", 12, "bold"))
        battery_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        
        # Battery readings
        readings_frame = tk.Frame(battery_panel, bg=self.frame_bg)
        readings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid for readings
        readings_frame.grid_columnconfigure(0, weight=1)
        readings_frame.grid_columnconfigure(1, weight=1)
        
        # Voltage display
        voltage_frame = tk.Frame(readings_frame, bg=self.frame_bg, relief=tk.RIDGE, bd=2)
        voltage_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        voltage_frame.grid_rowconfigure(0, weight=1)
        voltage_frame.grid_rowconfigure(1, weight=1)
        voltage_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            voltage_frame,
            text="VOLTAGE",
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 11, "bold")
        ).grid(row=0, column=0, pady=(10, 0))
        
        self.voltage_label = tk.Label(
            voltage_frame,
            text="--.- V",
            bg=self.frame_bg,
            fg=self.voltage_color,
            font=("Arial", 18, "bold")
        )
        self.voltage_label.grid(row=1, column=0, pady=(0, 10))
        
        # Current display
        current_frame = tk.Frame(readings_frame, bg=self.frame_bg, relief=tk.RIDGE, bd=2)
        current_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        current_frame.grid_rowconfigure(0, weight=1)
        current_frame.grid_rowconfigure(1, weight=1)
        current_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            current_frame,
            text="CURRENT",
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 11, "bold")
        ).grid(row=0, column=0, pady=(10, 0))
        
        self.current_label = tk.Label(
            current_frame,
            text="--.- A",
            bg=self.frame_bg,
            fg=self.current_color,
            font=("Arial", 18, "bold")
        )
        self.current_label.grid(row=1, column=0, pady=(0, 10))
        
        # Power display
        power_frame = tk.Frame(readings_frame, bg=self.frame_bg, relief=tk.RIDGE, bd=2)
        power_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        power_frame.grid_rowconfigure(0, weight=1)
        power_frame.grid_rowconfigure(1, weight=1)
        power_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            power_frame,
            text="POWER",
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 11, "bold")
        ).grid(row=0, column=0, pady=(10, 0))
        
        self.power_label = tk.Label(
            power_frame,
            text="--.- W",
            bg=self.frame_bg,
            fg=self.power_color,
            font=("Arial", 18, "bold")
        )
        self.power_label.grid(row=1, column=0, pady=(0, 10))
        
        # Battery percentage display
        percentage_frame = tk.Frame(readings_frame, bg=self.frame_bg, relief=tk.RIDGE, bd=2)
        percentage_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        percentage_frame.grid_rowconfigure(0, weight=1)
        percentage_frame.grid_rowconfigure(1, weight=1)
        percentage_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            percentage_frame,
            text="BATTERY",
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 11, "bold")
        ).grid(row=0, column=0, pady=(10, 0))
        
        self.percentage_label = tk.Label(
            percentage_frame,
            text="--%",
            bg=self.frame_bg,
            fg="#FF9800",
            font=("Arial", 18, "bold")
        )
        self.percentage_label.grid(row=1, column=0, pady=(0, 10))
        
        # Total power used
        total_power_frame = tk.Frame(readings_frame, bg=self.frame_bg, relief=tk.RIDGE, bd=2)
        total_power_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        total_power_frame.grid_rowconfigure(0, weight=1)
        total_power_frame.grid_rowconfigure(1, weight=1)
        total_power_frame.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            total_power_frame,
            text="TOTAL POWER USED",
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 11, "bold")
        ).grid(row=0, column=0, pady=(10, 0))
        
        self.total_power_label = tk.Label(
            total_power_frame,
            text="--.- Wh",
            bg=self.frame_bg,
            fg="white",
            font=("Arial", 16, "bold")
        )
        self.total_power_label.grid(row=1, column=0, pady=(0, 10))
        
        # Battery status
        self.battery_status_text = tk.Label(
            battery_panel,
            text="Battery Status: Normal",
            bg=self.frame_bg,
            fg="green",
            font=("Arial", 10)
        )
        self.battery_status_text.pack(pady=5)
        
        # Control panel (below battery readings)
        control_frame = tk.LabelFrame(battery_panel, text="Controls", bg=self.frame_bg, fg=self.text_color, font=("Arial", 11, "bold"))
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg=self.frame_bg)
        button_frame.pack(pady=10)
        
        self.start_stop_button = tk.Button(
            button_frame,
            text="Start Auto Detection",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.toggle_auto_tracking,
            width=18
        )
        self.start_stop_button.pack(pady=5)
        
        tk.Button(
            button_frame,
            text="Move Arms to Home",
            bg="#ff8c00",
            fg="black",
            font=("Arial", 10, "bold"),
            command=self.move_arms_to_home,
            width=18
        ).pack(pady=5)
        
        # Status labels
        status_frame = tk.Frame(control_frame, bg=self.frame_bg)
        status_frame.pack(pady=10, fill="x")
        
        self.status_label = tk.Label(
            status_frame,
            text="Status: Ready",
            bg=self.frame_bg,
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.status_label.pack()
        
        self.right_status = tk.Label(
            status_frame,
            text="Right Arm: Idle",
            bg=self.frame_bg,
            fg="#888888",
            font=("Arial", 9)
        )
        self.right_status.pack()
        
        self.left_status = tk.Label(
            status_frame,
            text="Left Arm: Idle",
            bg=self.frame_bg,
            fg="#888888",
            font=("Arial", 9)
        )
        self.left_status.pack()
        
        self.coord_label = tk.Label(
            status_frame,
            text="Ball Position: --, --, -- cm",
            bg=self.frame_bg,
            fg="#888888",
            font=("Arial", 9)
        )
        self.coord_label.pack()
        
        self.reachability_label = tk.Label(
            status_frame,
            text="Reachability: --",
            bg=self.frame_bg,
            fg="#888888",
            font=("Arial", 9)
        )
        self.reachability_label.pack()
        
        # Connection status
        self.connection_status = tk.Label(
            battery_panel,
            text="Serial: DISCONNECTED",
            bg=self.frame_bg,
            fg="red",
            font=("Arial", 9)
        )
        self.connection_status.pack(pady=5)
    
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
        self.reachability_label.config(text="Reachability: --", fg="#888888")
        
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
    
    def read_voltage(self):
        """Read voltage from sensor (replace with actual ADC reading)"""
        # TODO: Replace with actual SPI reading from LTC1864
        # For now, simulate realistic voltage (7.0V to 8.4V range)
        return 7.6 + random.uniform(-0.3, 0.3)
    
    def read_current(self):
        """Read current from sensor (replace with actual ADC reading)"""
        # TODO: Replace with actual SPI reading from LTC1864
        # For now, simulate realistic current (0.5A to 3.0A range)
        return 1.5 + random.uniform(-0.5, 0.8)
    
    def update_battery(self):
        """Update battery readings (live display only, no history)"""
        try:
            current_time = time.time()
            elapsed_time = current_time - self.last_update_time
            
            # Read voltage and current
            voltage = self.read_voltage()
            current = self.read_current()
            
            # Calculate power (Watts)
            power = voltage * current
            
            # Calculate battery percentage
            FULL_CHARGE_VOLTAGE = 8.4
            EMPTY_CHARGE_VOLTAGE = 6.2
            
            if voltage >= FULL_CHARGE_VOLTAGE:
                percent = 100
            elif voltage <= EMPTY_CHARGE_VOLTAGE:
                percent = 0
            else:
                percent = int(((voltage - EMPTY_CHARGE_VOLTAGE) / 
                            (FULL_CHARGE_VOLTAGE - EMPTY_CHARGE_VOLTAGE)) * 100)
            
            # Calculate total power used (Watt-hours)
            elapsed_hours = elapsed_time / 3600.0
            self.total_power_used += power * elapsed_hours
            
            # Update display
            self.voltage_label.config(text=f"{voltage:.2f} V")
            self.current_label.config(text=f"{current:.2f} A")
            self.power_label.config(text=f"{power:.2f} W")
            self.percentage_label.config(text=f"{percent}%")
            self.total_power_label.config(text=f"{self.total_power_used:.2f} Wh")
            
            # Update battery status
            if percent >= 80:
                status_text = "Battery Status: FULL"
                status_color = "green"
                percent_color = "#4CAF50"
            elif percent <= 10:
                status_text = "Battery Status: LOW - CHARGE NOW!"
                status_color = "red"
                percent_color = "#f44336"
            elif percent <= 35:
                status_text = f"Battery Status: {percent}% - MEDIUM"
                status_color = "orange"
                percent_color = "#FF9800"
            else:
                status_text = f"Battery Status: {percent}% - GOOD"
                status_color = "green"
                percent_color = "#4CAF50"
            
            # Update percentage label color
            self.percentage_label.config(fg=percent_color)
            self.battery_status_text.config(text=status_text, fg=status_color)
            
            # Update connection status
            if self.ser and self.ser.is_open:
                self.connection_status.config(text="Serial: CONNECTED", fg="green")
            else:
                self.connection_status.config(text="Serial: DISCONNECTED", fg="red")
            
            # Update last update time
            self.last_update_time = current_time
            
        except Exception as e:
            print(f"Error updating battery: {e}")
            self.voltage_label.config(text="Error", fg="red")
            self.current_label.config(text="Error", fg="red")
            self.power_label.config(text="Error", fg="red")
            self.percentage_label.config(text="Error", fg="red")
            self.battery_status_text.config(text="Sensor Error", fg="red")
        
        # Schedule next update
        self.root.after(1000, self.update_battery)
    
    def get_arm_side(self, x_cm):
        """Determine which arm to use based on X coordinate"""
        if x_cm > 0:
            return "right"
        else:
            return "left"
    
    def camera_loop(self):
        """Main camera processing loop"""
        frame_count = 0
        last_unreachable_alert_time = 0  # For rate-limiting TTS alerts
        # FPS calculation (smoothed)
        last_fps_time = time.time()
        fps = 0.0
        fps_alpha = 0.9
        
        while self.camera_running and self.auto_tracking:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, lambda: self.status_label.config(text="Status: Camera Error", fg="red"))
                break
            
            frame_count += 1

            # Update FPS (smoothed instant FPS)
            now_fps = time.time()
            dt = now_fps - last_fps_time
            if dt > 0:
                inst_fps = 1.0 / dt
                if fps <= 0.0:
                    fps = inst_fps
                else:
                    fps = fps_alpha * fps + (1.0 - fps_alpha) * inst_fps
            last_fps_time = now_fps
            
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
                    
                    # Check if position is reachable
                    reachable, reason = is_position_reachable(X_cm, Y_cm, Z_cm)
                    
                    self.root.after(0, lambda x=X_cm, y=Y_cm, z=Z_cm: 
                        self.coord_label.config(text=f"Ball Position: X:{x:+.1f}, Y:{y:+.1f}, Z:{z:.1f} cm", fg="#4CAF50"))
                    
                    if reachable:
                        self.root.after(0, lambda: self.reachability_label.config(text=f"Reachability: REACHABLE", fg="green"))
                        
                        # Determine which arm to use
                        arm_side = self.get_arm_side(X_cm)
                        
                        if arm_side == "right" and self.right_arm:
                            self.root.after(0, lambda: self.right_status.config(text="Right Arm: Moving to grab", fg="green"))
                            target_angles = self.right_arm.calculate_ik(xyz)
                            if target_angles:
                                self.right_arm.smooth_move(target_angles, step_deg=4.0, step_time_ms=300)
                                time.sleep(2)  # Pause to grab
                                self.right_arm.move_to_home()
                                self.root.after(0, lambda: self.right_status.config(text="Right Arm: Returned to home", fg="#888888"))
                        
                        elif arm_side == "left" and self.left_arm:
                            self.root.after(0, lambda: self.left_status.config(text="Left Arm: Moving to grab", fg="green"))
                            target_angles = self.left_arm.calculate_ik(xyz)
                            if target_angles:
                                self.left_arm.smooth_move(target_angles, step_deg=4.0, step_time_ms=300)
                                time.sleep(2)  # Pause to grab
                                self.left_arm.move_to_home()
                                self.root.after(0, lambda: self.left_status.config(text="Left Arm: Returned to home", fg="#888888"))
                        
                        self.root.after(0, lambda: self.right_status.config(text="Right Arm: Idle", fg="#888888"))
                        self.root.after(0, lambda: self.left_status.config(text="Left Arm: Idle", fg="#888888"))
                    else:
                        # Object is unreachable
                        self.root.after(0, lambda r=reason: self.reachability_label.config(text=f"Reachability: UNREACHABLE - {r}", fg="red"))
                        
                        # Rate-limit TTS alerts (only once every 5 seconds)
                        current_time = time.time()
                        if current_time - last_unreachable_alert_time > 5.0:
                            last_unreachable_alert_time = current_time
                            # Speak in a separate thread to avoid blocking
                            tts_thread = threading.Thread(target=speak_unreachable, daemon=True)
                            tts_thread.start()
                else:
                    self.root.after(0, lambda: self.coord_label.config(text="Ball Position: Invalid disparity", fg="red"))
                    self.root.after(0, lambda: self.reachability_label.config(text="Reachability: Invalid position", fg="red"))
            else:
                self.root.after(0, lambda: self.coord_label.config(text="Ball Position: Not detected", fg="red"))
                self.root.after(0, lambda: self.reachability_label.config(text="Reachability: No ball detected", fg="red"))
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

            # FPS overlay (top-left)
            try:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(display_left, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_right, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except Exception:
                pass
            
            # Convert to RGB for display
            left_rgb = cv2.cvtColor(display_left, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(display_right, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            left_img = Image.fromarray(left_rgb)
            right_img = Image.fromarray(right_rgb)
            
            # Resize for better display
            left_img = left_img.resize((480, 360), Image.Resampling.LANCZOS)
            right_img = right_img.resize((480, 360), Image.Resampling.LANCZOS)
            
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
        
        # Stop TTS engine
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Ball Detection Robot with Battery Monitor")
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
