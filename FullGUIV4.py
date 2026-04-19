"""
#!/usr/bin/env python3
UNIFIED DUAL-ARM ROBOT CONTROLLER
Tab 1: Manual Control (2x2 grid layout)
Tab 2: Auto Mode (ball tracking with camera feeds)
Tab 3: Recording/Playback
Tab 4: Settings
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import math
import serial
import cv2
import numpy as np
from PIL import Image, ImageTk
import queue
#import spidev
import random
import json
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

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

# Right arm (green ball)
RIGHT_CAM_OFFSET_RIGHT_CM = 16.0
RIGHT_CAM_OFFSET_UP_CM = 10.0
RIGHT_HOME_ANGLES = {"J0": 0.0, "J1": 135.0, "J2": 98.6, "J3": 86.0}
RIGHT_SERVO_CHANNELS = [16, 17, 18, 19]

# Left arm (blue ball)
LEFT_CAM_OFFSET_RIGHT_CM = -16.0
LEFT_CAM_OFFSET_UP_CM = 10.0
LEFT_HOME_ANGLES = {"J0": 0.0, "J1": 135.0, "J2": 97, "J3": 98.6}
LEFT_SERVO_CHANNELS = [28, 29, 30, 31]

# Servo configuration - UPDATED WITH CORRECT VALUES
SERVOS_CONFIG = {
    # Right hand servos (from your first code)
    "RIGHT_J0": {"id": 16, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
    "RIGHT_J1": {"id": 17, "min_angle": 120.0, "max_angle": 270, "min_pulse": 1390, "max_pulse": 24, "offset": 0.0, "invert": False},
    "RIGHT_J2": {"id": 18, "min_angle": 0.0, "max_angle": 133.6, "min_pulse": 530, "max_pulse": 1490, "offset": 0.0, "invert": False},
    "RIGHT_J3": {"id": 19, "min_angle": 26.77, "max_angle": 117.23, "min_pulse": 790, "max_pulse": 1770, "offset": 0.0, "invert": False},

    # Left hand servos (from your second code)
    "LEFT_J0": {"id": 28, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": True},
    "LEFT_J1": {"id": 29, "min_angle": 55.6, "max_angle": 270.0, "min_pulse": 925, "max_pulse": 2450, "offset": 0.0, "invert": False},
    "LEFT_J2": {"id": 30, "min_angle": 0.0, "max_angle": 142.1, "min_pulse": 530, "max_pulse": 1550, "offset": 0.0, "invert": False},
    "LEFT_J3": {"id": 31, "min_angle": 26.85, "max_angle": 172.62, "min_pulse": 1430, "max_pulse": 2370, "offset": 0.0, "invert": True},

    # Camera servos
    "CAM_YAW": {"id": 8, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": True},
    "CAM_PITCH": {"id": 0, "min_angle": 0.0, "max_angle": 180.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
}

# Camera search parameters
CAMERA_SEARCH_ENABLED = True
CAMERA_YAW_RANGE = 30  # degrees left/right from center
CAMERA_PITCH_RANGE = 15  # degrees up/down from center
CAMERA_SEARCH_SPEED = 1  # degrees per step
CAMERA_SEARCH_DELAY = 0.200  # seconds between steps
CAMERA_HOME_POSITION = {"yaw": 135, "pitch": 144.5}  # Center position

# Ball color HSV ranges
GREEN_LOWER1 = np.array([31, 54, 59])
GREEN_UPPER1 = np.array([81, 253, 255])
GREEN_LOWER2 = np.array([30, 65, 161])
GREEN_UPPER2 = np.array([102, 255, 255])

# Blue ball HSV ranges
BLUE_LOWER1 = np.array([90, 50, 50])
BLUE_UPPER1 = np.array([130, 255, 255])
BLUE_LOWER2 = np.array([85, 60, 100])
BLUE_UPPER2 = np.array([140, 255, 255])

# Battery monitoring
BATTERY_VOLTAGE_HISTORY = []
BATTERY_CURRENT_HISTORY = []
POWER_HISTORY = []
MAX_HISTORY_POINTS = 100

# ============================================================================
# UTILITY FUNCTIONS (for auto mode)
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

def detect_ball(frame, color="green"):
    """Detect ball of specified color"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if color == "green":
        mask = cv2.inRange(hsv, GREEN_LOWER1, GREEN_UPPER1) | cv2.inRange(hsv, GREEN_LOWER2, GREEN_UPPER2)
    else:  # blue
        mask = cv2.inRange(hsv, BLUE_LOWER1, BLUE_UPPER1) | cv2.inRange(hsv, BLUE_LOWER2, BLUE_UPPER2)
    
    mask = cv2.medianBlur(mask, 5)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    best = max(contours, key=cv2.contourArea)
    (xc, yc), r = cv2.minEnclosingCircle(best)
    
    if r < 5:
        return None
    
    cx = int(xc)
    cy = int(yc)
    return (cx, cy, int(r))

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
    forward = Z_cm
    right = X_cm + camera_offset_right
    up = -Y_cm + 10.0
    
    return forward, up, right

def compute_yaw_and_planar_coords(forward, right):
    """Compute yaw angle and planar distance"""
    yaw_rad = math.atan2(right, forward)
    yaw_deg = math.degrees(yaw_rad)
    r = math.hypot(forward, right)
    return yaw_deg, r

def planar_2link_ik(r_cm, up_cm, L1cm=L1, L2cm=L2, elbow_down=True):
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
# MOVEMENT RECORDER CLASS
# ============================================================================
class MovementRecorder:
    def __init__(self):
        self.is_recording = False
        self.recorded_movements = []
        self.start_time = None
        self.current_recording_name = ""
        
    def start_recording(self, name=None):
        """Start recording movements"""
        if name is None:
            name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.is_recording = True
        self.recorded_movements = []
        self.start_time = time.time()
        self.current_recording_name = name
        print(f"Started recording: {name}")
        
    def stop_recording(self):
        """Stop recording movements"""
        self.is_recording = False
        print(f"Stopped recording. Captured {len(self.recorded_movements)} movements")
        
    def record_movement(self, arm_side, joint_name, angle, timestamp=None):
        """Record a single movement"""
        if not self.is_recording:
            return
        
        if timestamp is None:
            timestamp = time.time() - self.start_time
        
        movement = {
            'timestamp': timestamp,
            'arm_side': arm_side,
            'joint_name': joint_name,
            'angle': angle
        }
        self.recorded_movements.append(movement)
        
    def record_multiple(self, arm_side, angles, timestamp=None):
        """Record multiple joints at once"""
        if not self.is_recording:
            return
        
        if timestamp is None:
            timestamp = time.time() - self.start_time
        
        for joint_name, angle in angles.items():
            movement = {
                'timestamp': timestamp,
                'arm_side': arm_side,
                'joint_name': joint_name,
                'angle': angle
            }
            self.recorded_movements.append(movement)
    
    def save_recording(self, filename=None):
        """Save recording to file"""
        if not self.recorded_movements:
            return False
        
        if filename is None:
            filename = f"{self.current_recording_name}.json"
        
        data = {
            'name': self.current_recording_name,
            'timestamp': datetime.now().isoformat(),
            'movements': self.recorded_movements
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Recording saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False
    
    def load_recording(self, filename):
        """Load recording from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.current_recording_name = data['name']
            self.recorded_movements = data['movements']
            print(f"Loaded recording: {data['name']} with {len(self.recorded_movements)} movements")
            return data
        except Exception as e:
            print(f"Error loading recording: {e}")
            return None
    
    def get_unique_arm_sides(self):
        """Get unique arm sides in recording"""
        sides = set()
        for movement in self.recorded_movements:
            sides.add(movement['arm_side'])
        return list(sides)
    
    def get_movements_by_time(self):
        """Group movements by timestamp"""
        time_groups = {}
        for movement in self.recorded_movements:
            ts = round(movement['timestamp'], 2)
            if ts not in time_groups:
                time_groups[ts] = []
            time_groups[ts].append(movement)
        return time_groups

# ============================================================================
# CAMERA SEARCH CLASS
# ============================================================================
class CameraSearch:
    def __init__(self, ser=None):
        self.ser = ser
        self.current_yaw = CAMERA_HOME_POSITION["yaw"]
        self.current_pitch = CAMERA_HOME_POSITION["pitch"]
        self.searching = False
        self.search_thread = None
        self.search_direction = 1  # 1 for right/up, -1 for left/down
        self.search_phase = "yaw"  # "yaw" or "pitch"
        self.yaw_target_left = CAMERA_HOME_POSITION["yaw"] - CAMERA_YAW_RANGE
        self.yaw_target_right = CAMERA_HOME_POSITION["yaw"] + CAMERA_YAW_RANGE
        self.pitch_target_up = CAMERA_HOME_POSITION["pitch"] - CAMERA_PITCH_RANGE
        self.pitch_target_down = CAMERA_HOME_POSITION["pitch"] + CAMERA_PITCH_RANGE
        self.object_detected = False
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # seconds before resuming search after detection
    
    def angle_to_pulse(self, angle_deg, servo_name):
        """Convert angle to pulse width for camera servos"""
        cfg = SERVOS_CONFIG[servo_name]
        
        angle = angle_deg + cfg.get("offset", 0.0)
        if cfg.get("invert", False):
            angle = 270 - angle
        
        angle = clamp(angle, cfg["min_angle"], cfg["max_angle"])
        ratio = (angle - cfg["min_angle"]) / (cfg["max_angle"] - cfg["min_angle"])
        pulse = int(cfg["min_pulse"] + ratio * (cfg["max_pulse"] - cfg["min_pulse"]))
        return pulse
    
    def move_camera(self, yaw=None, pitch=None, duration_ms=200):
        """Move camera servos to specified angles"""
        if self.ser is None:
            return
        
        parts = []
        
        if yaw is not None:
            yaw = clamp(yaw, 0, 270)
            pulse = self.angle_to_pulse(yaw, "CAM_YAW")
            parts.append(f"#{SERVOS_CONFIG['CAM_YAW']['id']} P{pulse}")
            self.current_yaw = yaw
        
        if pitch is not None:
            pitch = clamp(pitch, 0, 270)
            pulse = self.angle_to_pulse(pitch, "CAM_PITCH")
            parts.append(f"#{SERVOS_CONFIG['CAM_PITCH']['id']} P{pulse}")
            self.current_pitch = pitch
        
        if parts:
            cmd = " ".join(parts) + f" T{int(duration_ms)}\r"
            try:
                self.ser.write(cmd.encode('ascii'))
            except Exception as e:
                print(f"Camera move error: {e}")
    
    def move_to_home(self):
        """Move camera to home position (135° for both)"""
        self.move_camera(
            yaw=CAMERA_HOME_POSITION["yaw"],
            pitch=CAMERA_HOME_POSITION["pitch"],
            duration_ms=500
        )
    
    def start_search(self):
        """Start camera search pattern"""
        if self.searching:
            return
        
        self.searching = True
        self.search_phase = "yaw"
        self.search_direction = 1
        self.object_detected = False
        self.search_thread = threading.Thread(target=self._search_pattern, daemon=True)
        self.search_thread.start()
    
    def stop_search(self):
        """Stop camera search pattern"""
        self.searching = False
        if self.search_thread:
            self.search_thread.join(timeout=1)
    
    def object_found(self):
        """Signal that an object has been found"""
        self.object_detected = True
        self.last_detection_time = time.time()
    
    def _search_pattern(self):
        """Camera search pattern with object detection pause"""
        while self.searching:
            # Check if object was recently detected
            if self.object_detected:
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_timeout:
                    # Still in pause period after detection
                    time.sleep(0.1)
                    continue
                else:
                    # Resume search after timeout
                    self.object_detected = False
            
            if self.search_phase == "yaw":
                # Move yaw
                if self.search_direction > 0:
                    target = self.yaw_target_right
                else:
                    target = self.yaw_target_left
                
                # Move towards target
                step = CAMERA_SEARCH_SPEED * self.search_direction
                new_yaw = self.current_yaw + step
                
                # Check if reached target
                if (self.search_direction > 0 and new_yaw >= target) or \
                   (self.search_direction < 0 and new_yaw <= target):
                    # Reverse direction
                    self.search_direction *= -1
                    # Check if we've completed a full yaw sweep
                    if self.search_direction > 0:
                        # Start pitch search
                        self.search_phase = "pitch"
                        self.search_direction = 1
                        continue
                
                self.move_camera(yaw=new_yaw, duration_ms=50)
                
            elif self.search_phase == "pitch":
                # Move pitch
                if self.search_direction > 0:
                    target = self.pitch_target_down
                else:
                    target = self.pitch_target_up
                
                # Move towards target
                step = CAMERA_SEARCH_SPEED * self.search_direction
                new_pitch = self.current_pitch + step
                
                # Check if reached target
                if (self.search_direction > 0 and new_pitch >= target) or \
                   (self.search_direction < 0 and new_pitch <= target):
                    # Reverse direction
                    self.search_direction *= -1
                    # Check if we've completed a full pitch sweep
                    if self.search_direction > 0:
                        # Return to yaw search
                        self.search_phase = "yaw"
                        continue
                
                self.move_camera(pitch=new_pitch, duration_ms=50)
            
            time.sleep(CAMERA_SEARCH_DELAY)

# ============================================================================
# MAIN APPLICATION CLASS - WITH THREADED ARCHITECTURE
# ============================================================================
class UnifiedRobotController:
    def __init__(self, root):
        self.root = root
        root.title("Dual-Arm Robot Controller - Manual/Auto/Recording Modes")
        root.geometry("1200x800")
        
        # ========= DARK THEME COLORS =========
        self.bg_color = "#252323"
        self.frame_bg = "#3c3f41"
        self.text_color = "#ffffff"
        self.button_bg = "#5c5c5c"
        self.reset_button_bg = "#ff8c00"
        self.record_button_bg = "#d32f2f"
        self.play_button_bg = "#388e3c"
        self.slider_width = 180
        
        # Battery colors
        self.voltage_color = "#4CAF50"  # Green
        self.current_color = "#2196F3"   # Blue
        self.power_color = "#FF9800"     # Orange
        
        root.configure(bg=self.bg_color)
        self.configure_styles()
        
        # Initialize serial
        self.ser = None
        self.init_serial()
        
        # Initialize SPI sensors
        self.init_spi_sensors()
        
        # Initialize robot arms for auto mode
        self.right_arm = None
        self.left_arm = None
        
        # Initialize camera search
        self.camera_search = CameraSearch(self.ser)
        
        # Initialize movement recorder
        self.recorder = MovementRecorder()
        
        # Playback control variables
        self.is_playing = False
        self.playback_thread = None
        
        # Camera handling for auto mode - THREADED ARCHITECTURE
        self.cap = None
        self.camera_running = False
        self.auto_tracking = False
        
        # Threaded camera pipeline
        self.raw_frame_queue = queue.Queue(maxsize=2)
        self.processed_frame_queue = queue.Queue(maxsize=2)
        self.capture_running = False
        self.processing_running = False
        self.display_running = False
        self.capture_thread = None
        self.processing_thread = None
        self.display_thread = None
        self.left_map1 = self.left_map2 = self.right_map1 = self.right_map2 = None
        
        # Performance monitoring
        self.display_counter = 0
        self.frame_timestamps = []
        
        # Battery monitoring variables
        self.total_power_used = 0.0  # Watt-hours
        self.last_update_time = time.time()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_manual_tab()      # 2x2 grid layout
        self.create_auto_tab()        # Auto mode with camera
        self.create_recording_tab()   # Recording/Playback
        self.create_settings_tab()    # Settings
        
        # Initialize manual mode
        self.setup_manual_variables()
        
        # Start battery monitoring
        self.update_battery()
        
        # Start performance monitoring
        self.root.after(1000, self.monitor_performance)
        
        # Set default mode to manual
        self.switch_to_manual()
    
    def init_serial(self):
        """Initialize serial connection"""
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5)
            time.sleep(0.1)
            print(f"Opened serial on {SERIAL_PORT}")
            self.serial_connected = True
            
            # Initialize robot arms
            self.right_arm = self.RobotArm("right", self.ser)
            self.left_arm = self.RobotArm("left", self.ser)
        except Exception as e:
            print(f"Could not open serial port: {e}")
            print("Continuing in debug mode (no hardware moves).")
            self.ser = None
            self.serial_connected = False
            # Initialize robot arms with None serial
            self.right_arm = self.RobotArm("right", None)
            self.left_arm = self.RobotArm("left", None)
    
    def init_spi_sensors(self):
        """Initialize SPI connections for voltage and current sensors"""
        try:
            # Voltage sensor (LTC1864 on CE0)
            self.spi_voltage = spidev.SpiDev()
            self.spi_voltage.open(0, 0)  # bus 0, device 0 (CE0)
            self.spi_voltage.max_speed_hz = 1000000  # 1 MHz
            self.spi_voltage.mode = 1  # Mode SPI 1
            
            # Current sensor (LTC1864 on CE0)
            self.spi_current = spidev.SpiDev()
            self.spi_current.open(1, 0)  # bus 1, device 0 (CE0)
            self.spi_current.max_speed_hz = 1000000  # 1 MHz
            self.spi_current.mode = 1  # Mode SPI 1
            
            print("SPI sensors initialized successfully")
            self.spi_available = True
        except Exception as e:
            print(f"Could not initialize SPI sensors: {e}")
            self.spi_voltage = None
            self.spi_current = None
            self.spi_available = False
    
    def configure_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color)
        style.configure("TNotebook", background=self.bg_color)
        style.configure("TNotebook.Tab", background=self.button_bg, foreground=self.text_color,
                       padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#5c5c5c")])
    
    # ============================================================================
    # ROBOT ARM CLASS (for auto mode)
    # ============================================================================
    class RobotArm:
        def __init__(self, side="right", ser=None):
            self.side = side
            self.ser = ser
            
            if side == "right":
                self.servo_channels = RIGHT_SERVO_CHANNELS
                self.home_angles = RIGHT_HOME_ANGLES.copy()
                self.camera_offset_right = RIGHT_CAM_OFFSET_RIGHT_CM
                self.color = "green"
                self.prefix = "RIGHT"
            else:
                self.servo_channels = LEFT_SERVO_CHANNELS
                self.home_angles = LEFT_HOME_ANGLES.copy()
                self.camera_offset_right = LEFT_CAM_OFFSET_RIGHT_CM
                self.color = "blue"
                self.prefix = "LEFT"
            
            self.current_angles = self.home_angles.copy()
            self.is_tracking = False
        
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
            
            ik = planar_2link_ik(r_cm, up, L1cm=L1, L2cm=L2, elbow_down=True)
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
    # TAB 1: MANUAL MODE (2x2 GRID LAYOUT) - FIXED CAMERA CONTROL
    # ============================================================================
    def create_manual_tab(self):
        """Create manual control tab with 2x2 grid layout"""
        manual_frame = ttk.Frame(self.notebook)
        self.notebook.add(manual_frame, text='Manual Control')
        
        # ========= SERVO CONFIGURATION =========
        self.SERIAL_PORT = '/dev/serial0'
        self.BAUD_RATE = 9600
        
        # Right hand servos
        self.RIGHT_BASE_CH, self.RIGHT_SHOULDER_CH, self.RIGHT_ELBOW_CH, self.RIGHT_GRIPPER_CH = 16, 17, 18, 19
        # Left hand servos
        self.LEFT_BASE_CH, self.LEFT_SHOULDER_CH, self.LEFT_ELBOW_CH, self.LEFT_GRIPPER_CH = 28, 29, 30, 31
        
        self.SERVOS = {
            # Right hand servos
            "RIGHT_J0": {"id": self.RIGHT_BASE_CH, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
            "RIGHT_J1": {"id": self.RIGHT_SHOULDER_CH, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
            "RIGHT_J2": {"id": self.RIGHT_ELBOW_CH, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
            "RIGHT_J3": {"id": self.RIGHT_GRIPPER_CH, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
            
            # Left hand servos
            "LEFT_J0": {"id": self.LEFT_BASE_CH, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": True},
            "LEFT_J1": {"id": self.LEFT_SHOULDER_CH, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
            "LEFT_J2": {"id": self.LEFT_ELBOW_CH, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
            "LEFT_J3": {"id": self.LEFT_GRIPPER_CH, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": True},
            
            # Camera servos
            "CAM_YAW": {"id": 8, "min_angle": 0.0, "max_angle": 270.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": True},
            "CAM_PITCH": {"id": 0, "min_angle": 0.0, "max_angle": 180.0, "min_pulse": 530, "max_pulse": 2450, "offset": 0.0, "invert": False},
        }
        
        # ========= HOME POSITIONS =========
        self.home_positions = {
            'left_base': 0,
            'left_shoulder': 135,  
            'left_elbow': 98.6,
            'left_gripper': 135,
            'right_base': 0,
            'right_shoulder': 135,
            'right_elbow': 98.6,
            'right_gripper': 135,
            'cam_pitch': 135,
            'cam_yaw': 135
        }
        
        self.servo_home_angles = {
            'right': {"RIGHT_J0": 0.0, "RIGHT_J1": 135.0, "RIGHT_J2": 97, "RIGHT_J3": 135.0},
            'left': {"LEFT_J0": 0.0, "LEFT_J1": 135.0, "LEFT_J2": 98.6, "LEFT_J3": 135.0}
        }

        # ========= GRID SETUP (2x2 with equal sizing) =========
        manual_frame.grid_rowconfigure(0, weight=1, uniform="row")
        manual_frame.grid_rowconfigure(1, weight=1, uniform="row")
        manual_frame.grid_columnconfigure(0, weight=1, uniform="col")
        manual_frame.grid_columnconfigure(1, weight=1, uniform="col")

        # ========= CREATE 4 EQUAL GRIDS =========
        left_arm_frame = self.make_frame(manual_frame, "Left Arm Control", 0, 0)
        right_arm_frame = self.make_frame(manual_frame, "Right Arm Control", 0, 1)
        camera_control_frame = self.make_frame(manual_frame, "Camera Gimbal Control", 1, 0)
        battery_frame = self.make_frame(manual_frame, "Battery Monitoring", 1, 1)

        # ========= LEFT ARM CONTROL =========
        left_arm_frame.grid_rowconfigure(0, weight=1)
        left_arm_frame.grid_rowconfigure(1, weight=0)
        left_arm_frame.grid_columnconfigure(0, weight=1, uniform="slider")
        left_arm_frame.grid_columnconfigure(1, weight=1, uniform="slider")
        left_arm_frame.grid_columnconfigure(2, weight=1, uniform="slider")
        left_arm_frame.grid_columnconfigure(3, weight=1, uniform="slider")
        
        self.left_base = self.create_slider_with_grid(left_arm_frame, "Left Base", "left_base", 0, 0)
        self.left_shoulder = self.create_slider_with_grid(left_arm_frame, "Left Shoulder", "left_shoulder", 0, 1)
        self.left_elbow = self.create_slider_with_grid(left_arm_frame, "Left Elbow", "left_elbow", 0, 2)
        self.left_gripper = self.create_slider_with_grid(left_arm_frame, "Left Gripper", "left_gripper", 0, 3)
        
        reset_left_frame = tk.Frame(left_arm_frame, bg=self.frame_bg)
        reset_left_frame.grid(row=1, column=0, columnspan=4, pady=(20, 0), sticky="ew")
        reset_left_frame.grid_columnconfigure(0, weight=1)
        
        tk.Button(
            reset_left_frame,
            text="Reset Left Arm",
            bg=self.reset_button_bg,
            fg="black",
            font=("Arial", 10, "bold"),
            command=lambda: self.reset_arm_to_home("left")
        ).pack(pady=10, padx=20)

        # ========= RIGHT ARM CONTROL =========
        right_arm_frame.grid_rowconfigure(0, weight=1)
        right_arm_frame.grid_rowconfigure(1, weight=0)
        right_arm_frame.grid_columnconfigure(0, weight=1, uniform="slider")
        right_arm_frame.grid_columnconfigure(1, weight=1, uniform="slider")
        right_arm_frame.grid_columnconfigure(2, weight=1, uniform="slider")
        right_arm_frame.grid_columnconfigure(3, weight=1, uniform="slider")
        
        self.right_base = self.create_slider_with_grid(right_arm_frame, "Right Base", "right_base", 0, 0)
        self.right_shoulder = self.create_slider_with_grid(right_arm_frame, "Right Shoulder", "right_shoulder", 0, 1)
        self.right_elbow = self.create_slider_with_grid(right_arm_frame, "Right Elbow", "right_elbow", 0, 2)
        self.right_gripper = self.create_slider_with_grid(right_arm_frame, "Right Gripper", "right_gripper", 0, 3)
        
        reset_right_frame = tk.Frame(right_arm_frame, bg=self.frame_bg)
        reset_right_frame.grid(row=1, column=0, columnspan=4, pady=(20, 0), sticky="ew")
        reset_right_frame.grid_columnconfigure(0, weight=1)
        
        tk.Button(
            reset_right_frame,
            text="Reset Right Arm",
            bg=self.reset_button_bg,
            fg="black",
            font=("Arial", 10, "bold"),
            command=lambda: self.reset_arm_to_home("right")
        ).pack(pady=10, padx=20)

        # ========= CAMERA GIMBAL CONTROL =========
        camera_control_frame.grid_rowconfigure(0, weight=1)
        camera_control_frame.grid_rowconfigure(1, weight=0)
        camera_control_frame.grid_columnconfigure(0, weight=1, uniform="slider")
        camera_control_frame.grid_columnconfigure(1, weight=1, uniform="slider")
        
        # Create camera pitch slider (servo ID 0)
        self.cam_pitch_slider = self.create_camera_slider_with_grid(
            camera_control_frame, 
            "Camera Pitch", 
            "cam_pitch", 
            servo_name="CAM_PITCH",
            row=0, 
            col=0
        )
        
        # Create camera yaw slider (servo ID 8)
        self.cam_yaw_slider = self.create_camera_slider_with_grid(
            camera_control_frame, 
            "Camera Yaw", 
            "cam_yaw", 
            servo_name="CAM_YAW",
            row=0, 
            col=1
        )
        
        # Camera control buttons
        camera_button_frame = tk.Frame(camera_control_frame, bg=self.frame_bg)
        camera_button_frame.grid(row=1, column=0, columnspan=2, pady=(20, 0), sticky="ew")
        camera_button_frame.grid_columnconfigure(0, weight=1)
        camera_button_frame.grid_columnconfigure(1, weight=1)
        
        # Camera Reset button (moves to 135° for both)
        self.camera_reset_button = tk.Button(
            camera_button_frame,
            text="Camera Reset",
            bg=self.button_bg,
            fg=self.text_color,
            font=("Arial", 10, "bold"),
            command=self.camera_reset
        )
        self.camera_reset_button.grid(row=0, column=0, padx=5, pady=10)
        
        # Auto Search button (press and hold)
        self.auto_search_button = tk.Button(
            camera_button_frame,
            text="Auto Search",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.start_auto_search
        )
        self.auto_search_button.grid(row=0, column=1, padx=5, pady=10)
        
        # Bind button events for press and hold functionality
        self.auto_search_button.bind("<ButtonPress-1>", self.on_auto_search_press)
        self.auto_search_button.bind("<ButtonRelease-1>", self.on_auto_search_release)

        # ========= BATTERY MONITORING =========
        battery_frame.grid_rowconfigure(0, weight=0)  # Title
        battery_frame.grid_rowconfigure(1, weight=1)  # Main content
        battery_frame.grid_rowconfigure(2, weight=0)  # Status
        battery_frame.grid_columnconfigure(0, weight=1)

        # Battery readings section
        readings_frame = tk.Frame(battery_frame, bg=self.frame_bg)
        readings_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Configure grid to fill space uniformly
        readings_frame.grid_columnconfigure(0, weight=1, uniform="col")
        readings_frame.grid_columnconfigure(1, weight=1, uniform="col")
        readings_frame.grid_rowconfigure(0, weight=1, uniform="row")
        readings_frame.grid_rowconfigure(1, weight=1, uniform="row")
        readings_frame.grid_rowconfigure(2, weight=1, uniform="row")

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
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, pady=(10, 0))

        self.voltage_label = tk.Label(
            voltage_frame,
            text="--.- V",
            bg=self.frame_bg,
            fg=self.voltage_color,
            font=("Arial", 20, "bold")
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
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, pady=(10, 0))

        self.current_label = tk.Label(
            current_frame,
            text="--.- A",
            bg=self.frame_bg,
            fg=self.current_color,
            font=("Arial", 20, "bold")
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
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, pady=(10, 0))

        self.power_label = tk.Label(
            power_frame,
            text="--.- W",
            bg=self.frame_bg,
            fg=self.power_color,
            font=("Arial", 20, "bold")
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
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, pady=(10, 0))

        self.percentage_label = tk.Label(
            percentage_frame,
            text="--%",
            bg=self.frame_bg,
            fg="#FF9800",  # Orange color for percentage
            font=("Arial", 20, "bold")
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
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, pady=(10, 0))

        self.total_power_label = tk.Label(
            total_power_frame,
            text="--.- Wh",
            bg=self.frame_bg,
            fg="white",
            font=("Arial", 20, "bold")
        )
        self.total_power_label.grid(row=1, column=0, pady=(0, 10))

        # Battery status
        self.battery_status_text = tk.Label(
            battery_frame,
            text="Battery Status: Normal",
            bg=self.frame_bg,
            fg="green",
            font=("Arial", 11)
        )
        self.battery_status_text.grid(row=2, column=0, pady=10, sticky="ew")
        
        # Connection status
        self.status_label = tk.Label(
            battery_frame,
            text="Serial: DISCONNECTED",
            bg=self.frame_bg,
            fg="red",
            font=("Arial", 10)
        )
        self.status_label.grid(row=3, column=0, pady=5, sticky="ew")
    
    def create_camera_slider_with_grid(self, parent, label, joint_name, servo_name, row, col):
        """Create a camera-specific slider that directly controls camera servos"""
        slider_frame = tk.Frame(parent, bg=self.frame_bg)
        slider_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        slider_frame.grid_propagate(False)
        
        slider_frame.grid_rowconfigure(0, weight=0)
        slider_frame.grid_rowconfigure(1, weight=1)
        slider_frame.grid_columnconfigure(0, weight=1)
        
        # Get home position for this camera servo
        home_angle = self.home_positions.get(joint_name, 135)
        
        lbl = tk.Label(
            slider_frame,
            text=f"{label}: {home_angle}°",
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 9)
        )
        lbl.grid(row=0, column=0, pady=(0, 5))
        
        slider = tk.Scale(
            slider_frame,
            from_=270,
            to=0,
            orient="vertical",
            length=200,
            width=20,
            command=lambda v, lbl=lbl, name=label, servo=servo_name: 
                self.on_camera_slider_change(v, lbl, name, servo),
            bg=self.frame_bg,
            troughcolor="#5c5c5c",
            fg=self.text_color,
            highlightthickness=0,
            sliderlength=30
        )
        slider.set(home_angle)
        slider.grid(row=1, column=0, sticky="ns")
        
        return slider
    
    def on_camera_slider_change(self, slider_value, label_widget, label_name, servo_name):
        """Handle camera slider changes - directly control camera servos"""
        angle = float(slider_value)
        label_widget.config(text=f"{label_name}: {angle}°")
        
        # Move the camera servo directly
        if servo_name == "CAM_YAW":
            self.camera_search.move_camera(yaw=angle, duration_ms=200)
        elif servo_name == "CAM_PITCH":
            self.camera_search.move_camera(pitch=angle, duration_ms=200)
        
        print(f"{label_name}: {angle}°")
    
    def camera_reset(self):
        """Reset camera to home position (135° for both)"""
        # Set sliders to 135°
        if hasattr(self, 'cam_pitch_slider'):
            self.cam_pitch_slider.set(135)
        if hasattr(self, 'cam_yaw_slider'):
            self.cam_yaw_slider.set(135)
        
        # Update slider labels
        for widget in self.cam_pitch_slider.master.winfo_children():
            if isinstance(widget, tk.Label):
                widget.config(text="Camera Pitch: 135°")
                break
        
        for widget in self.cam_yaw_slider.master.winfo_children():
            if isinstance(widget, tk.Label):
                widget.config(text="Camera Yaw: 135°")
                break
        
        # Move camera to home position
        self.camera_search.move_to_home()
        print("Camera reset to 135° for both axes")
    
    def on_auto_search_press(self, event):
        """Handle auto search button press"""
        self.auto_search_button.config(bg="red", text="Auto Search (ON)")
        self.camera_search.start_search()
        print("Auto search started (press and hold)")
    
    def on_auto_search_release(self, event):
        """Handle auto search button release"""
        self.auto_search_button.config(bg="#4CAF50", text="Auto Search")
        self.camera_search.stop_search()
        print("Auto search stopped (button released)")
    
    def start_auto_search(self):
        """Start auto search (called when button is clicked normally)"""
        # This is just a backup in case the button events don't work
        if not hasattr(self, '_auto_search_active'):
            self._auto_search_active = False
        
        if not self._auto_search_active:
            self._auto_search_active = True
            self.auto_search_button.config(bg="red", text="Auto Search (ON)")
            self.camera_search.start_search()
        else:
            self._auto_search_active = False
            self.auto_search_button.config(bg="#4CAF50", text="Auto Search")
            self.camera_search.stop_search()
    
    def setup_manual_variables(self):
        """Initialize variables needed for manual mode"""
        self.build_connection_status = self.serial_connected
    
    # ============================================================================
    # MANUAL MODE FUNCTIONS
    # ============================================================================
    def make_frame(self, parent, title, row, col):
        frame = tk.LabelFrame(
            parent,
            text=title,
            padx=10,
            pady=10,
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 11, "bold")
        )
        frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        return frame
    
    def create_slider_with_grid(self, parent, label, joint, row, col):
        slider_frame = tk.Frame(parent, bg=self.frame_bg)
        slider_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        slider_frame.grid_propagate(False)
        
        slider_frame.grid_rowconfigure(0, weight=0)
        slider_frame.grid_rowconfigure(1, weight=1)
        slider_frame.grid_columnconfigure(0, weight=1)
        
        if "left" in joint:
            arm_side = "left"
            if "base" in joint:
                joint_idx = 0
            elif "shoulder" in joint:
                joint_idx = 1
            elif "elbow" in joint:
                joint_idx = 2
            else:
                joint_idx = 3
        else:
            arm_side = "right"
            if "base" in joint:
                joint_idx = 0
            elif "shoulder" in joint:
                joint_idx = 1
            elif "elbow" in joint:
                joint_idx = 2
            else:
                joint_idx = 3
        
        lbl = tk.Label(
            slider_frame,
            text=f"{label}: {self.home_positions[joint]}°",
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 9)
        )
        lbl.grid(row=0, column=0, pady=(0, 5))
        
        slider = tk.Scale(
            slider_frame,
            from_=270,
            to=0,
            orient="vertical",
            length=200,
            width=20,
            command=lambda v, lbl=lbl, name=label, joint_name=joint_idx, side=arm_side: 
                self.on_slider_change(v, lbl, name, joint_name, side),
            bg=self.frame_bg,
            troughcolor="#5c5c5c",
            fg=self.text_color,
            highlightthickness=0,
            sliderlength=30
        )
        slider.set(self.home_positions[joint])
        slider.grid(row=1, column=0, sticky="ns")
        
        return slider
    
    def on_slider_change(self, slider_value, label_widget, label_name, joint_name, arm_side):
        angle = float(slider_value)
        label_widget.config(text=f"{label_name}: {angle}°")
        
        # Record movement if recording
        if self.recorder.is_recording:
            joint_map = {0: "J0", 1: "J1", 2: "J2", 3: "J3"}
            joint_key = joint_map[joint_name]
            self.recorder.record_movement(arm_side, joint_key, angle)
        
        servo_angle = self.move_servo_from_slider(slider_value, joint_name, arm_side)
        print(f"{label_name}: Slider={slider_value}°, Servo Angle={servo_angle:.1f}°")
    
    def slider_to_servo_angle(self, slider_value):
        servo_angle = float(slider_value)
        return servo_angle
    
    def servo_angle_to_pulse(self, angle_deg, servo_config):
        angle = angle_deg + servo_config.get("offset", 0.0)
        if servo_config.get("invert", False):
            angle = 270 - angle
        
        angle = max(servo_config["min_angle"], min(servo_config["max_angle"], angle))
        ratio = (angle - servo_config["min_angle"]) / (servo_config["max_angle"] - servo_config["min_angle"])
        pulse = int(servo_config["min_pulse"] + ratio * (servo_config["max_pulse"] - servo_config["min_pulse"]))
        return pulse
    
    def move_servo_from_slider(self, slider_value, joint_name, arm_side):
        servo_angle = self.slider_to_servo_angle(slider_value)
        
        if arm_side == "right":
            servo_key = f"RIGHT_J{joint_name}"
        else:
            servo_key = f"LEFT_J{joint_name}"
        
        servo_config = self.SERVOS[servo_key]
        pulse = self.servo_angle_to_pulse(servo_angle, servo_config)
        self.send_servo_command(servo_config["id"], pulse, duration_ms=200)
        
        return servo_angle
    
    def send_servo_command(self, servo_id, pulse, duration_ms=500):
        if self.ser is None:
            print(f"[DEBUG] #{servo_id} P{pulse} T{duration_ms}")
            return
        
        cmd = f"#{servo_id} P{pulse} T{int(duration_ms)}\r"
        try:
            self.ser.write(cmd.encode('ascii'))
            print(f"Sent: {cmd.strip()}")
        except Exception as e:
            print(f"Serial write error: {e}")
    
    def send_multiple_servos(self, servo_data, duration_ms=500):
        if self.ser is None:
            print(f"[DEBUG] Multi-servo: {servo_data}")
            return
        
        parts = []
        for servo_id, pulse in servo_data:
            parts.append(f"#{servo_id} P{pulse}")
        
        if parts:
            cmd = " ".join(parts) + f" T{int(duration_ms)}\r"
            try:
                self.ser.write(cmd.encode('ascii'))
                print(f"Sent: {cmd.strip()}")
            except Exception as e:
                print(f"Serial write error: {e}")
    
    def reset_arm_to_home(self, arm_side):
        if arm_side == "left":
            target_angles = self.servo_home_angles['left']
            
            slider_values = {
                'left_base': target_angles["LEFT_J0"],
                'left_shoulder': target_angles["LEFT_J1"],
                'left_elbow': target_angles["LEFT_J2"],
                'left_gripper': target_angles["LEFT_J3"]
            }
            
            self.left_base.set(slider_values['left_base'])
            self.left_shoulder.set(slider_values['left_shoulder'])
            self.left_elbow.set(slider_values['left_elbow'])
            self.left_gripper.set(slider_values['left_gripper'])
            
            self.update_slider_labels("left", target_angles)
            self.move_arm_smoothly("left", target_angles)
            
        else:
            target_angles = self.servo_home_angles['right']
            
            slider_values = {
                'right_base': target_angles["RIGHT_J0"],
                'right_shoulder': target_angles["RIGHT_J1"],
                'right_elbow': target_angles["RIGHT_J2"],
                'right_gripper': target_angles["RIGHT_J3"]
            }
            
            self.right_base.set(slider_values['right_base'])
            self.right_shoulder.set(slider_values['right_shoulder'])
            self.right_elbow.set(slider_values['right_elbow'])
            self.right_gripper.set(slider_values['right_gripper'])
            
            self.update_slider_labels("right", target_angles)
            self.move_arm_smoothly("right", target_angles)
    
    def update_slider_labels(self, arm_side, angles):
        if arm_side == "left":
            labels = [
                ("Left Base", self.left_base),
                ("Left Shoulder", self.left_shoulder),
                ("Left Elbow", self.left_elbow),
                ("Left Gripper", self.left_gripper)
            ]
            angle_keys = ["LEFT_J0", "LEFT_J1", "LEFT_J2", "LEFT_J3"]
        else:
            labels = [
                ("Right Base", self.right_base),
                ("Right Shoulder", self.right_shoulder),
                ("Right Elbow", self.right_elbow),
                ("Right Gripper", self.right_gripper)
            ]
            angle_keys = ["RIGHT_J0", "RIGHT_J1", "RIGHT_J2", "RIGHT_J3"]
        
        for i, (name, slider) in enumerate(labels):
            slider_frame = slider.master
            for widget in slider_frame.winfo_children():
                if isinstance(widget, tk.Label):
                    widget.config(text=f"{name}: {angles[angle_keys[i]]}°")
                    break
    
    def move_arm_smoothly(self, arm_side, target_angles, step_time_ms=400):
        if self.ser is None:
            print(f"[DEBUG] Would move {arm_side} arm to {target_angles}")
            return
        
        servo_data = []
        
        if arm_side == "left":
            for joint_name, angle in target_angles.items():
                servo_config = self.SERVOS[joint_name]
                pulse = self.servo_angle_to_pulse(angle, servo_config)
                servo_data.append((servo_config["id"], pulse))
        else:
            for joint_name, angle in target_angles.items():
                servo_config = self.SERVOS[joint_name]
                pulse = self.servo_angle_to_pulse(angle, servo_config)
                servo_data.append((servo_config["id"], pulse))
        
        self.send_multiple_servos(servo_data, step_time_ms)
        print(f"Moved {arm_side} arm to home position")
    
    def read_voltage(self, channel=0):
        if self.spi_voltage is None:
            return random.uniform(8.4, 6.2)  # Simulate data
            
        try:
            if channel == 0:
                cmd = [0x00, 0x00]
            else:
                cmd = [0x80, 0x00]

            resp = self.spi_voltage.xfer2(cmd)
            raw = (resp[0] << 8) | resp[1]
            
            voltage = (raw / 65535.0) * 20.0
            return voltage
        except Exception as e:
            print(f"Error reading voltage: {e}")
            return random.uniform(8.4, 6.2)
    
    def read_current(self, channel=1):
        if self.spi_current is None:
            return random.uniform(0.5, 2.5)  # Simulate data
            
        try:
            if channel == 0:
                cmd = [0x00, 0x00]
            else:
                cmd = [0x80, 0x00]

            resp = self.spi_current.xfer2(cmd)
            raw = (resp[0] << 8) | resp[1]
            
            current = ((raw - 32767.5) * 30.0) / 32767.5
            return abs(current)  # Return absolute value
        except Exception as e:
            print(f"Error reading current: {e}")
            return random.uniform(0.5, 2.5)
    
    def update_battery(self):
        try:
            current_time = time.time()
            elapsed_time = current_time - self.last_update_time
            
            if self.spi_available:
                voltage = self.read_voltage(0)
                current = self.read_current(1)
            else:
                # Simulate data if SPI not available
                voltage = 0.90 + random.uniform(-0.2, 0.2)
                current = 0.51 + random.uniform(-0.05, 0.05)
            
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
            
            # Update history
            BATTERY_VOLTAGE_HISTORY.append(voltage)
            BATTERY_CURRENT_HISTORY.append(current)
            POWER_HISTORY.append(power)
            
            # Keep only last N points
            if len(BATTERY_VOLTAGE_HISTORY) > MAX_HISTORY_POINTS:
                BATTERY_VOLTAGE_HISTORY.pop(0)
                BATTERY_CURRENT_HISTORY.pop(0)
                POWER_HISTORY.pop(0)
            
            # Calculate total power used (Watt-hours)
            # Convert elapsed time from seconds to hours
            elapsed_hours = elapsed_time / 3600.0
            avg_power = np.mean(POWER_HISTORY) if POWER_HISTORY else 0
            self.total_power_used += avg_power * elapsed_hours
            
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
                # Update percentage color based on level
                percent_color = "#4CAF50"  # Green for high charge
            elif percent <= 10:
                status_text = "Battery Status: LOW"
                status_color = "red"
                percent_color = "#f44336"  # Red for low charge
            elif percent <= 35:
                status_text = f"Battery Status: {percent}%"
                status_color = "orange"
                percent_color = "#FF9800"  # Orange for medium charge
            else:
                status_text = f"Battery Status: {percent}%"
                status_color = "green"
                percent_color = "#4CAF50"  # Green for good charge
            
            # Update percentage label color
            self.percentage_label.config(fg=percent_color)
            self.battery_status_text.config(text=status_text, fg=status_color)
            
            # Update connection status
            if self.ser and self.ser.is_open:
                self.status_label.config(text="Serial: CONNECTED", fg="green")
            else:
                self.status_label.config(text="Serial: DISCONNECTED", fg="red")
            
            # Update last update time
            self.last_update_time = current_time
            
            print(f"Battery: {voltage:.2f}V, {current:.2f}A, {power:.2f}W, {percent}%, Total: {self.total_power_used:.2f}Wh")
                
        except Exception as e:
            print(f"Error reading battery sensors: {e}")
            self.voltage_label.config(text="Error", fg="red")
            self.current_label.config(text="Error", fg="red")
            self.power_label.config(text="Error", fg="red")
            self.percentage_label.config(text="Error", fg="red")
            self.total_power_label.config(text="Error", fg="red")
            self.battery_status_text.config(text="Sensor Error", fg="red")

        # Schedule next update
        self.root.after(1000, self.update_battery)
    
    # ============================================================================
    # THREADED CAMERA ARCHITECTURE
    # ============================================================================
    def initialize_camera_threading(self):
        """Initialize threaded camera pipeline"""
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
            print("Camera initialized for threading")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def start_camera_threads(self):
        """Start all camera processing threads"""
        if not self.cap or not self.cap.isOpened():
            return False
        
        # Set flags
        self.capture_running = True
        self.processing_running = True
        self.display_running = True
        
        # Clear queues
        self.clear_queues()
        
        # Start threads
        self.capture_thread = threading.Thread(
            target=self._capture_frames, 
            daemon=True,
            name="CaptureThread"
        )
        
        self.processing_thread = threading.Thread(
            target=self._process_frames,
            daemon=True,
            name="ProcessingThread"
        )
        
        self.display_thread = threading.Thread(
            target=self._display_frames,
            daemon=True,
            name="DisplayThread"
        )
        
        self.capture_thread.start()
        self.processing_thread.start()
        self.display_thread.start()
        
        print("Camera threads started successfully")
        return True
    
    def stop_camera_threads(self):
        """Stop all camera threads gracefully"""
        self.capture_running = False
        self.processing_running = False
        self.display_running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        # Clear queues
        self.clear_queues()
        
        print("Camera threads stopped")
    
    def clear_queues(self):
        """Clear all queues"""
        while not self.raw_frame_queue.empty():
            try:
                self.raw_frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.processed_frame_queue.empty():
            try:
                self.processed_frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def _capture_frames(self):
        """Thread 1: Just capture frames as fast as possible"""
        print("Capture thread started")
        frame_count = 0
        
        while self.capture_running and self.camera_running:
            try:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Put in queue if not full
                if not self.raw_frame_queue.full():
                    self.raw_frame_queue.put(frame)
                    frame_count += 1
                else:
                    # Queue full, drop frame to prevent lag
                    # This keeps the stream responsive
                    pass
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Capture thread error: {e}")
                time.sleep(0.01)
        
        print(f"Capture thread stopped, captured {frame_count} frames")
    
    def _process_frames(self):
        """Thread 2: Process frames for tracking and display"""
        print("Processing thread started")
        
        while self.processing_running and self.camera_running:
            try:
                # Get frame from queue (with timeout to check running flag)
                try:
                    frame = self.raw_frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the frame
                processed_data = self._process_single_frame(frame)
                
                # Put processed frame in display queue
                if not self.processed_frame_queue.full():
                    self.processed_frame_queue.put(processed_data)
                
            except Exception as e:
                print(f"Processing thread error: {e}")
        
        print("Processing thread stopped")
    
    def _process_single_frame(self, frame):
        """Process a single frame"""
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
        
        # Detect balls (only if auto tracking is enabled)
        detection_results = {}
        if self.auto_tracking:
            green_detL = detect_ball(left_rect, "green")
            green_detR = detect_ball(right_rect, "green")
            blue_detL = detect_ball(left_rect, "blue")
            blue_detR = detect_ball(right_rect, "blue")
            
            detection_results = {
                'green': (green_detL, green_detR),
                'blue': (blue_detL, blue_detR),
                'any_detection': (green_detL and green_detR) or (blue_detL and blue_detR)
            }
            
            # Handle camera search based on detection
            if self.camera_search_var.get():
                if detection_results['any_detection']:
                    self.camera_search.object_found()
            
            # Process arm movements if needed
            self._handle_arm_movements(detection_results)
        
        # Prepare display frames
        display_left, display_right = self._prepare_display_frames(
            left_rect, right_rect, detection_results
        )
        
        return {
            'left_display': display_left,
            'right_display': display_right,
            'detections': detection_results
        }
    
    def _handle_arm_movements(self, detection_results):
        """Handle arm movements based on detections"""
        tracking_mode = self.tracking_mode.get()
        
        # Right arm (green ball)
        if tracking_mode in ["both", "right"] and detection_results['green'][0] and detection_results['green'][1]:
            xyz = stereo_to_xyz_cm_rectified(
                detection_results['green'][0], 
                detection_results['green'][1]
            )
            if xyz:
                if self.right_arm:
                    target_angles = self.right_arm.calculate_ik(xyz)
                    if target_angles:
                        # Use threading for arm movement to not block processing
                        threading.Thread(
                            target=self.right_arm.smooth_move,
                            args=(target_angles,),
                            kwargs={'step_deg': 4.0, 'step_time_ms': 300},
                            daemon=True
                        ).start()
                else: print("Right arm not connected.")
        
        # Left arm (blue ball)
        if tracking_mode in ["both", "left"] and detection_results['blue'][0] and detection_results['blue'][1]:
            xyz = stereo_to_xyz_cm_rectified(
                detection_results['blue'][0], 
                detection_results['blue'][1]
            )
            if xyz:
                if self.left_arm:
                    target_angles = self.left_arm.calculate_ik(xyz)
                    if target_angles:
                        threading.Thread(
                            target=self.left_arm.smooth_move,
                            args=(target_angles,),
                            kwargs={'step_deg': 4.0, 'step_time_ms': 300},
                            daemon=True
                        ).start()
                else: print("Left arm not connected.")
    
    def _prepare_display_frames(self, left_rect, right_rect, detection_results):
        """Prepare frames for display with minimal processing"""
        # Copy frames for display
        display_left = left_rect.copy()
        display_right = right_rect.copy()
        
        # Draw detections if available
        if 'green' in detection_results:
            green_detL, green_detR = detection_results['green']
            if green_detL:
                cv2.circle(display_left, green_detL[:2], green_detL[2], (0, 255, 0), 2)
            if green_detR:
                cv2.circle(display_right, green_detR[:2], green_detR[2], (0, 255, 0), 2)
        
        if 'blue' in detection_results:
            blue_detL, blue_detR = detection_results['blue']
            if blue_detL:
                cv2.circle(display_left, blue_detL[:2], blue_detL[2], (0, 255, 255), 2)
            if blue_detR:
                cv2.circle(display_right, blue_detR[:2], blue_detR[2], (0, 255, 255), 2)
        
        # Add minimal status text (only essential)
        yaw_offset = self.camera_search.current_yaw - CAMERA_HOME_POSITION["yaw"]
        pitch_offset = self.camera_search.current_pitch - CAMERA_HOME_POSITION["pitch"]
        
        cv2.putText(display_left, f"Y:{yaw_offset:+3.0f}°", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_left, f"P:{pitch_offset:+3.0f}°", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add search status
        if self.camera_search.object_detected:
            status_text = "Found!"
            status_color = (0, 255, 0)
        elif self.camera_search.searching:
            status_text = "Searching"
            status_color = (0, 255, 255)
        else:
            status_text = "Idle"
            status_color = (255, 255, 255)
        
        cv2.putText(display_left, status_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(display_right, status_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Resize for display (smaller = faster)
        display_left = cv2.resize(display_left, (400, 300))
        display_right = cv2.resize(display_right, (400, 300))
        
        return display_left, display_right
    
    def _display_frames(self):
        """Thread 3: Update GUI display"""
        print("Display thread started")
        
        while self.display_running and self.camera_running:
            try:
                # Get processed frame from queue
                try:
                    processed_data = self.processed_frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Track frame rate
                self.frame_timestamps.append(time.time())
                # Keep only last 30 timestamps (for FPS calculation)
                if len(self.frame_timestamps) > 30:
                    self.frame_timestamps.pop(0)
                
                # Convert to display format
                left_rgb = cv2.cvtColor(processed_data['left_display'], cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(processed_data['right_display'], cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image (use faster resize if needed)
                left_img = Image.fromarray(left_rgb)
                right_img = Image.fromarray(right_rgb)
                
                # Convert to ImageTk
                left_tk = ImageTk.PhotoImage(image=left_img)
                right_tk = ImageTk.PhotoImage(image=right_img)
                
                # Update GUI in main thread
                self.root.after(0, self.update_camera_display, left_tk, right_tk)
                
                # Update status labels less frequently to reduce GUI load
                self.display_counter += 1
                if self.display_counter % 10 == 0:
                    self._update_status_labels(processed_data['detections'])
                
            except Exception as e:
                print(f"Display thread error: {e}")
        
        print("Display thread stopped")
    
    def _update_status_labels(self, detections):
        """Update status labels on GUI"""
        if not self.auto_tracking:
            return
        
        # Calculate FPS
        if len(self.frame_timestamps) > 1:
            time_diff = self.frame_timestamps[-1] - self.frame_timestamps[0]
            fps = len(self.frame_timestamps) / time_diff if time_diff > 0 else 0
        else:
            fps = 0
        
        # Update detection status
        if detections.get('any_detection', False):
            self.camera_status_label.config(text=f"Camera: Target Found (FPS: {fps:.1f})", fg="green")
        else:
            self.camera_status_label.config(text=f"Camera: Searching (FPS: {fps:.1f})", fg="yellow")
        
        # Update arm status
        tracking_mode = self.tracking_mode.get()
        
        if tracking_mode in ["both", "right"] and detections['green'][0] and detections['green'][1]:
            self.right_status_label.config(text="Right Arm: Ball detected", fg="green")
        else:
            self.right_status_label.config(text="Right Arm: No ball", fg="red")
        
        if tracking_mode in ["both", "left"] and detections['blue'][0] and detections['blue'][1]:
            self.left_status_label.config(text="Left Arm: Ball detected", fg="green")
        else:
            self.left_status_label.config(text="Left Arm: No ball", fg="red")
    
    def monitor_performance(self):
        """Monitor and print performance metrics"""
        if hasattr(self, 'raw_frame_queue'):
            capture_q_size = self.raw_frame_queue.qsize()
        else:
            capture_q_size = 0
            
        if hasattr(self, 'processed_frame_queue'):
            display_q_size = self.processed_frame_queue.qsize()
        else:
            display_q_size = 0
        
        # Calculate FPS from timestamps
        if len(self.frame_timestamps) > 1:
            time_diff = self.frame_timestamps[-1] - self.frame_timestamps[0]
            fps = len(self.frame_timestamps) / time_diff if time_diff > 0 else 0
        else:
            fps = 0
        
        print(f"Performance: Capture Q={capture_q_size}/2, Display Q={display_q_size}/2, FPS={fps:.1f}")
        
        # Schedule next monitoring
        self.root.after(2000, self.monitor_performance)

    # ============================================================================
    # TAB 2: AUTO MODE
    # ============================================================================
    def create_auto_tab(self):
        """Create auto mode tab with camera feeds"""
        auto_frame = ttk.Frame(self.notebook)
        self.notebook.add(auto_frame, text='Auto Mode')
        
        # Configure grid
        auto_frame.grid_columnconfigure(0, weight=1)
        auto_frame.grid_columnconfigure(1, weight=1)
        auto_frame.grid_rowconfigure(0, weight=3)
        auto_frame.grid_rowconfigure(1, weight=1)
        
        # Camera display frames
        self.left_cam_label = tk.Label(auto_frame, bg="black", relief="solid", bd=2)
        self.left_cam_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.right_cam_label = tk.Label(auto_frame, bg="black", relief="solid", bd=2)
        self.right_cam_label.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Add labels for camera feeds
        left_label = tk.Label(auto_frame, text="Left Camera View", bg=self.bg_color, fg="white",
                             font=("Arial", 10, "bold"))
        left_label.grid(row=0, column=0, sticky="nw", padx=10, pady=5)
        
        right_label = tk.Label(auto_frame, text="Right Camera View", bg=self.bg_color, fg="white",
                              font=("Arial", 10, "bold"))
        right_label.grid(row=0, column=1, sticky="nw", padx=10, pady=5)
        
        # Control frame
        control_frame = tk.Frame(auto_frame, bg=self.frame_bg)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew", padx=5)
        
        # Mode selection
        mode_frame = tk.Frame(control_frame, bg=self.frame_bg)
        mode_frame.pack(pady=10)
        
        tk.Label(
            mode_frame,
            text="Tracking Mode:",
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 10)
        ).pack(side=tk.LEFT, padx=5)
        
        self.tracking_mode = tk.StringVar(value="both")
        tk.Radiobutton(
            mode_frame,
            text="Both Arms",
            variable=self.tracking_mode,
            value="both",
            bg=self.frame_bg,
            fg=self.text_color,
            selectcolor=self.frame_bg
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            mode_frame,
            text="Right Arm Only (Green)",
            variable=self.tracking_mode,
            value="right",
            bg=self.frame_bg,
            fg=self.text_color,
            selectcolor=self.frame_bg
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            mode_frame,
            text="Left Arm Only (Blue)",
            variable=self.tracking_mode,
            value="left",
            bg=self.frame_bg,
            fg=self.text_color,
            selectcolor=self.frame_bg
        ).pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg=self.frame_bg)
        button_frame.pack(pady=10)
        
        self.start_stop_button = tk.Button(
            button_frame,
            text="Start Auto Tracking",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.toggle_auto_tracking,
            width=20
        )
        self.start_stop_button.pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_frame,
            text="Switch to Manual Mode",
            bg="#2196F3",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.switch_to_manual_mode,
            width=20
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_frame,
            text="Move Arms to Home",
            bg=self.reset_button_bg,
            fg="black",
            font=("Arial", 12, "bold"),
            command=self.move_arms_to_home,
            width=20
        ).pack(side=tk.LEFT, padx=10)
        
        # Status display
        status_frame = tk.Frame(control_frame, bg=self.frame_bg)
        status_frame.pack(pady=5)
        
        self.auto_status_label = tk.Label(
            status_frame,
            text="Status: Ready",
            bg=self.frame_bg,
            fg="white",
            font=("Arial", 10)
        )
        self.auto_status_label.pack()
        
        self.right_status_label = tk.Label(
            status_frame,
            text="Right Arm (Green): Idle",
            bg=self.frame_bg,
            fg="white",
            font=("Arial", 9)
        )
        self.right_status_label.pack()
        
        self.left_status_label = tk.Label(
            status_frame,
            text="Left Arm (Blue): Idle",
            bg=self.frame_bg,
            fg="white",
            font=("Arial", 9)
        )
        self.left_status_label.pack()
    
    def switch_to_manual_mode(self):
        """Switch from auto to manual mode"""
        self.stop_auto_tracking()
        self.notebook.select(0)  # Switch to manual tab
        self.move_arms_to_home()
    
    def move_arms_to_home(self):
        """Move both arms to home position"""
        self.right_arm.move_to_home()
        self.left_arm.move_to_home()
    
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
                self.auto_status_label.config(text="Status: Camera Error", fg="red")
                return
        
        self.auto_tracking = True
        self.start_stop_button.config(text="Stop Auto Tracking", bg="red")
        self.auto_status_label.config(text="Status: Tracking", fg="green")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def stop_auto_tracking(self):
        """Stop auto tracking"""
        self.auto_tracking = False
        self.start_stop_button.config(text="Start Auto Tracking", bg="#4CAF50")
        self.auto_status_label.config(text="Status: Stopped", fg="red")
        self.right_status_label.config(text="Right Arm (Green): Idle", fg="white")
        self.left_status_label.config(text="Left Arm (Blue): Idle", fg="white")
        
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
    
    def camera_loop(self):
        """Main camera processing loop"""
        frame_count = 0
        start_time = time.time()
        
        while self.camera_running and self.auto_tracking:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, lambda: self.auto_status_label.config(text="Status: Camera Error", fg="red"))
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
            
            # Detect balls
            green_detL = detect_ball(left_rect, "green")
            green_detR = detect_ball(right_rect, "green")
            blue_detL = detect_ball(left_rect, "blue")
            blue_detR = detect_ball(right_rect, "blue")
            
            # Process detections
            tracking_mode = self.tracking_mode.get()
            
            # Right arm (green ball)
            if tracking_mode in ["both", "right"] and green_detL and green_detR:
                self.root.after(0, lambda: self.right_status_label.config(
                    text="Right Arm (Green): Ball detected", fg="green"))
                
                xyz = stereo_to_xyz_cm_rectified(green_detL, green_detR)
                if xyz:
                    if self.right_arm:
                        target_angles = self.right_arm.calculate_ik(xyz)
                        if target_angles:
                            self.root.after(0, lambda: self.right_status_label.config(
                                text="Right Arm (Green): Moving to target", fg="green"))
                            
                            # Move to target
                            self.right_arm.smooth_move(target_angles, step_deg=4.0, step_time_ms=300)
                            
                            # Hold for 3 seconds
                            time.sleep(3)
                            
                            # Return to home
                            self.root.after(0, lambda: self.right_status_label.config(
                                text="Right Arm (Green): Returning to home", fg="blue"))
                            self.right_arm.move_to_home()
                            
                            self.root.after(0, lambda: self.right_status_label.config(
                                text="Right Arm (Green): Returned to home", fg="blue"))
                        else: print("Right arm not connected.")
            elif tracking_mode in ["both", "right"]:
                self.root.after(0, lambda: self.right_status_label.config(
                    text="Right Arm (Green): No green ball detected", fg="red"))
            
            # Left arm (blue ball)
            if tracking_mode in ["both", "left"] and blue_detL and blue_detR:
                self.root.after(0, lambda: self.left_status_label.config(
                    text="Left Arm (Blue): Ball detected", fg="green"))
                
                xyz = stereo_to_xyz_cm_rectified(blue_detL, blue_detR)
                if xyz:
                    if self.left_arm:
                        target_angles = self.left_arm.calculate_ik(xyz)
                        if target_angles:
                            self.root.after(0, lambda: self.left_status_label.config(
                                text="Left Arm (Blue): Moving to target", fg="green"))
                            
                            # Move to target
                            self.left_arm.smooth_move(target_angles, step_deg=4.0, step_time_ms=300)
                            
                            # Hold for 3 seconds
                            time.sleep(3)
                            
                            # Return to home
                            self.root.after(0, lambda: self.left_status_label.config(
                                text="Left Arm (Blue): Returning to home", fg="blue"))
                            self.left_arm.move_to_home()
                            
                            self.root.after(0, lambda: self.left_status_label.config(
                                text="Left Arm (Blue): Returned to home", fg="blue"))
                    else: print("Left arm not connected.")
            elif tracking_mode in ["both", "left"]:
                self.root.after(0, lambda: self.left_status_label.config(
                    text="Left Arm (Blue): No blue ball detected", fg="red"))
            
            # Draw detections on images
            display_left = left_rect.copy()
            display_right = right_rect.copy()
            
            if green_detL:
                cv2.circle(display_left, green_detL[:2], green_detL[2], (0, 255, 0), 2)
                cv2.putText(display_left, "Green", (green_detL[0], green_detL[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if blue_detL:
                cv2.circle(display_left, blue_detL[:2], blue_detL[2], (0, 255, 255), 2)
                cv2.putText(display_left, "Blue", (blue_detL[0], blue_detL[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if green_detR:
                cv2.circle(display_right, green_detR[:2], green_detR[2], (0, 255, 0), 2)
                cv2.putText(display_right, "Green", (green_detR[0], green_detR[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if blue_detR:
                cv2.circle(display_right, blue_detR[:2], blue_detR[2], (0, 255, 255), 2)
                cv2.putText(display_right, "Blue", (blue_detR[0], blue_detR[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Add FPS counter
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display_left, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_right, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert to RGB for display
            left_rgb = cv2.cvtColor(display_left, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(display_right, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            left_img = Image.fromarray(left_rgb)
            right_img = Image.fromarray(right_rgb)
            
            # Resize for display
            left_img = left_img.resize((400, 300), Image.Resampling.LANCZOS)
            right_img = right_img.resize((400, 300), Image.Resampling.LANCZOS)
            
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
    
    # ============================================================================
    # TAB 3: RECORDING/PLAYBACK
    # ============================================================================
    def create_recording_tab(self):
        """Create recording/playback tab"""
        recording_frame = ttk.Frame(self.notebook)
        self.notebook.add(recording_frame, text='Recording/Playback')
        
        # Main container
        main_container = tk.Frame(recording_frame, bg=self.bg_color)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(
            main_container,
            text="Movement Recording & Playback",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(pady=(0, 20))
        
        # Two column layout
        columns_frame = tk.Frame(main_container, bg=self.bg_color)
        columns_frame.pack(fill="both", expand=True)
        columns_frame.grid_columnconfigure(0, weight=1)
        columns_frame.grid_columnconfigure(1, weight=1)
        columns_frame.grid_rowconfigure(0, weight=1)
        
        # Left column: Recording
        recording_column = tk.Frame(columns_frame, bg=self.frame_bg, relief="solid", bd=1)
        recording_column.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        tk.Label(
            recording_column,
            text="Recording",
            font=("Arial", 14, "bold"),
            bg=self.frame_bg,
            fg=self.text_color
        ).pack(pady=15)
        
        # Recording name
        name_frame = tk.Frame(recording_column, bg=self.frame_bg)
        name_frame.pack(pady=10)
        
        tk.Label(
            name_frame,
            text="Recording Name:",
            bg=self.frame_bg,
            fg=self.text_color
        ).pack(side=tk.LEFT, padx=5)
        
        self.recording_name_entry = tk.Entry(name_frame, width=25)
        self.recording_name_entry.pack(side=tk.LEFT, padx=5)
        self.recording_name_entry.insert(0, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Recording status
        self.recording_tab_status = tk.Label(
            recording_column,
            text="Status: Not Recording",
            font=("Arial", 11),
            bg=self.frame_bg,
            fg="red"
        )
        self.recording_tab_status.pack(pady=10)
        
        # Recording controls
        rec_control_frame = tk.Frame(recording_column, bg=self.frame_bg)
        rec_control_frame.pack(pady=20)
        
        self.rec_start_button = tk.Button(
            rec_control_frame,
            text="● Start Recording",
            bg=self.record_button_bg,
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.start_recording_from_tab,
            width=20
        )
        self.rec_start_button.pack(pady=5)
        
        self.rec_stop_button = tk.Button(
            rec_control_frame,
            text="◼ Stop Recording",
            bg="#F1E9E9",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.stop_recording_from_tab,
            state="disabled",
            width=20
        )
        self.rec_stop_button.pack(pady=5)
        
        self.rec_save_button = tk.Button(
            rec_control_frame,
            text="💾 Save Recording",
            bg="#2196F3",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.save_recording_from_tab,
            width=20
        )
        self.rec_save_button.pack(pady=5)
        
        # Recording info
        info_frame = tk.Frame(recording_column, bg=self.frame_bg)
        info_frame.pack(pady=20)
        
        self.rec_info_label = tk.Label(
            info_frame,
            text="Movements: 0\nDuration: 0.0s",
            font=("Arial", 10),
            bg=self.frame_bg,
            fg=self.text_color,
            justify="left"
        )
        self.rec_info_label.pack()
        
        # Right column: Playback
        playback_column = tk.Frame(columns_frame, bg=self.frame_bg, relief="solid", bd=1)
        playback_column.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        tk.Label(
            playback_column,
            text="Playback",
            font=("Arial", 14, "bold"),
            bg=self.frame_bg,
            fg=self.text_color
        ).pack(pady=15)
        
        # Load recording
        load_frame = tk.Frame(playback_column, bg=self.frame_bg)
        load_frame.pack(pady=10)
        
        tk.Button(
            load_frame,
            text="📂 Load Recording",
            bg="#FF9800",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.load_recording_for_playback,
            width=20
        ).pack(pady=5)
        
        # Loaded file info
        self.loaded_file_label = tk.Label(
            playback_column,
            text="No file loaded",
            font=("Arial", 10),
            bg=self.frame_bg,
            fg="#888888",
            wraplength=300
        )
        self.loaded_file_label.pack(pady=10)
        
        # Recording info display
        self.playback_info_text = tk.Text(
            playback_column,
            height=8,
            width=40,
            bg="#1e1e1e",
            fg="white",
            font=("Courier", 9)
        )
        self.playback_info_text.pack(pady=10, padx=10)
        self.playback_info_text.insert("1.0", "Recording info will appear here")
        self.playback_info_text.config(state="disabled")
        
        # Playback controls
        playback_control_frame = tk.Frame(playback_column, bg=self.frame_bg)
        playback_control_frame.pack(pady=20)
        
        self.play_button = tk.Button(
            playback_control_frame,
            text="▶ Play",
            bg=self.play_button_bg,
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.play_recording,
            state="disabled",
            width=10
        )
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = tk.Button(
            playback_control_frame,
            text="⏸ Pause",
            bg="#FFC107",
            fg="black",
            font=("Arial", 12, "bold"),
            command=self.pause_playback,
            state="disabled",
            width=10
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(
            playback_control_frame,
            text="⏹ Stop",
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.stop_playback,
            state="disabled",
            width=10
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Playback status
        self.playback_status = tk.Label(
            playback_column,
            text="Status: Ready",
            font=("Arial", 11),
            bg=self.frame_bg,
            fg="white"
        )
        self.playback_status.pack(pady=10)
        
        # Playback speed control
        speed_frame = tk.Frame(playback_column, bg=self.frame_bg)
        speed_frame.pack(pady=10)
        
        tk.Label(
            speed_frame,
            text="Playback Speed:",
            bg=self.frame_bg,
            fg=self.text_color
        ).pack(side=tk.LEFT)
        
        self.playback_speed = tk.DoubleVar(value=1.0)
        speed_slider = tk.Scale(
            speed_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.playback_speed,
            bg=self.frame_bg,
            fg=self.text_color,
            length=150
        )
        speed_slider.pack(side=tk.LEFT, padx=10)
        
        # Add mode switch buttons at bottom
        switch_frame = tk.Frame(main_container, bg=self.bg_color)
        switch_frame.pack(pady=20)
        
        tk.Button(
            switch_frame,
            text="Switch to Manual Mode",
            bg="#2196F3",
            fg="white",
            font=("Arial", 12, "bold"),
            command=lambda: self.notebook.select(0),
            width=20
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            switch_frame,
            text="Switch to Auto Mode",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            command=lambda: self.notebook.select(1),
            width=20
        ).pack(side=tk.LEFT, padx=10)
    
    def start_recording_from_tab(self):
        """Start recording from recording tab"""
        name = self.recording_name_entry.get()
        if not name:
            name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.recorder.start_recording(name)
        self.recording_tab_status.config(text="Status: Recording", fg="green")
        self.rec_start_button.config(state="disabled", bg="#757575")
        self.rec_stop_button.config(state="normal", bg=self.record_button_bg)
        
        # Start update timer
        self.update_recording_info()
    
    def stop_recording_from_tab(self):
        """Stop recording from recording tab"""
        self.recorder.stop_recording()
        self.recording_tab_status.config(text="Status: Stopped", fg="red")
        self.rec_start_button.config(state="normal", bg=self.record_button_bg)
        self.rec_stop_button.config(state="disabled", bg="#757575")
    
    def save_recording_from_tab(self):
        """Save recording from recording tab"""
        if not self.recorder.recorded_movements:
            messagebox.showwarning("No Recording", "No movements to save. Start recording first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=self.recorder.current_recording_name + ".json"
        )
        
        if filename:
            if self.recorder.save_recording(filename):
                messagebox.showinfo("Success", f"Recording saved to:\n{filename}")
            else:
                messagebox.showerror("Error", "Failed to save recording")
    
    def update_recording_info(self):
        """Update recording information display"""
        if self.recorder.is_recording:
            count = len(self.recorder.recorded_movements)
            duration = time.time() - self.recorder.start_time
            
            info_text = f"Movements: {count}\nDuration: {duration:.1f}s"
            self.rec_info_label.config(text=info_text)
            
            # Schedule next update
            self.root.after(1000, self.update_recording_info)
    
    def load_recording_for_playback(self):
        """Load a recording file for playback"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            data = self.recorder.load_recording(filename)
            if data:
                self.loaded_file_label.config(
                    text=f"Loaded: {os.path.basename(filename)}\n"
                         f"Movements: {len(data['movements'])}\n"
                         f"Date: {data.get('timestamp', 'Unknown')}",
                    fg="white"
                )
                
                # Display recording info
                info_text = f"Recording: {data['name']}\n"
                info_text += f"Movements: {len(data['movements'])}\n"
                info_text += f"Date: {data.get('timestamp', 'Unknown')}\n"
                info_text += f"\nArm Sides: {', '.join(self.recorder.get_unique_arm_sides())}\n"
                
                # Group by time
                time_groups = self.recorder.get_movements_by_time()
                info_text += f"Time Points: {len(time_groups)}\n"
                
                self.playback_info_text.config(state="normal")
                self.playback_info_text.delete("1.0", tk.END)
                self.playback_info_text.insert("1.0", info_text)
                self.playback_info_text.config(state="disabled")
                
                # Enable playback buttons
                self.play_button.config(state="normal")
                self.playback_status.config(text="Status: Loaded", fg="green")
                
                return True
        
        return False
    
    def play_recording(self):
        """Play back the loaded recording"""
        if not self.recorder.recorded_movements:
            messagebox.showwarning("No Recording", "Please load a recording first.")
            return
        
        self.is_playing = True
        self.playback_status.config(text="Status: Playing", fg="green")
        self.play_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.stop_button.config(state="normal")
        
        # Start playback in separate thread
        self.playback_thread = threading.Thread(target=self._execute_playback, daemon=True)
        self.playback_thread.start()
    
    def pause_playback(self):
        """Pause playback"""
        self.is_playing = False
        self.playback_status.config(text="Status: Paused", fg="yellow")
    
    def stop_playback(self):
        """Stop playback"""
        self.is_playing = False
        self.playback_status.config(text="Status: Stopped", fg="red")
        self.play_button.config(state="normal")
        self.pause_button.config(state="disabled")
        self.stop_button.config(state="disabled")
    
    def _execute_playback(self):
        """Execute the playback of movements by controlling UI sliders"""
        # Mapping from recording format to slider objects
        slider_mapping = {
            ('left', 'J0'): self.left_base,
            ('left', 'J1'): self.left_shoulder,
            ('left', 'J2'): self.left_elbow,
            ('left', 'J3'): self.left_gripper,
            ('right', 'J0'): self.right_base,
            ('right', 'J1'): self.right_shoulder,
            ('right', 'J2'): self.right_elbow,
            ('right', 'J3'): self.right_gripper,
        }
        
        # Group movements by timestamp
        time_groups = self.recorder.get_movements_by_time()
        
        # Get sorted timestamps
        sorted_times = sorted(time_groups.keys())
        
        # Calculate total duration
        total_duration = sorted_times[-1] if sorted_times else 0
        
        # Play movements
        start_time = time.time()
        last_time = 0
        
        for t in sorted_times:
            if not self.is_playing:
                break
                
            # Apply speed factor
            speed_factor = self.playback_speed.get()
            wait_time = (t - last_time) / speed_factor
            
            # Wait for the appropriate time
            time.sleep(wait_time)
            last_time = t
            
            # Execute all movements at this timestamp
            movements = time_groups[t]
            
            # Update sliders for each movement
            for movement in movements:
                if not self.is_playing:
                    break
                    
                arm_side = movement['arm_side']
                joint_name = movement['joint_name']
                angle = movement['angle']
                
                # Get the corresponding slider
                slider_key = (arm_side, joint_name)
                if slider_key in slider_mapping:
                    slider = slider_mapping[slider_key]
                    # Update slider value on main thread
                    self.root.after(0, lambda s=slider, a=angle: s.set(a))
        
        # Reset playing flag
        self.is_playing = False
        
        # Update status
        self.root.after(0, lambda: self.playback_status.config(
            text=f"Status: Playback Complete ({total_duration:.1f}s)", fg="blue"))
        self.root.after(0, lambda: self.play_button.config(state="normal"))
        self.root.after(0, lambda: self.pause_button.config(state="disabled"))
        self.root.after(0, lambda: self.stop_button.config(state="disabled"))
    
    # ============================================================================
    # TAB 4: SETTINGS
    # ============================================================================
    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text='Settings')
        
        # Add settings content
        content_frame = tk.Frame(settings_frame, bg=self.bg_color)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(
            content_frame,
            text="Robot Settings",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(pady=20)
        
        # Camera settings
        camera_frame = tk.LabelFrame(
            content_frame,
            text="Camera Settings",
            padx=10,
            pady=10,
            bg=self.frame_bg,
            fg=self.text_color,
            font=("Arial", 11, "bold")
        )
        camera_frame.pack(fill="x", pady=10)
        
        # Camera search settings
        search_frame = tk.Frame(camera_frame, bg=self.frame_bg)
        search_frame.pack(fill="x", pady=5)
        
        tk.Label(
            search_frame,
            text="Search Range - Yaw:",
            bg=self.frame_bg,
            fg=self.text_color
        ).pack(side=tk.LEFT, padx=5)
        
        self.yaw_range_var = tk.IntVar(value=CAMERA_YAW_RANGE)
        yaw_scale = tk.Scale(
            search_frame,
            from_=10,
            to=60,
            orient="horizontal",
            variable=self.yaw_range_var,
            bg=self.frame_bg,
            fg=self.text_color,
            length=150
        )
        yaw_scale.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            search_frame,
            text="Pitch:",
            bg=self.frame_bg,
            fg=self.text_color
        ).pack(side=tk.LEFT, padx=5)
        
        self.pitch_range_var = tk.IntVar(value=CAMERA_PITCH_RANGE)
        pitch_scale = tk.Scale(
            search_frame,
            from_=5,
            to=30,
            orient="horizontal",
            variable=self.pitch_range_var,
            bg=self.frame_bg,
            fg=self.text_color,
            length=150
        )
        pitch_scale.pack(side=tk.LEFT, padx=5)
        
        # Search speed
        speed_frame = tk.Frame(camera_frame, bg=self.frame_bg)
        speed_frame.pack(fill="x", pady=5)
        
        tk.Label(
            speed_frame,
            text="Search Speed (deg/step):",
            bg=self.frame_bg,
            fg=self.text_color
        ).pack(side=tk.LEFT, padx=5)
        
        self.search_speed_var = tk.DoubleVar(value=CAMERA_SEARCH_SPEED)
        speed_scale = tk.Scale(
            speed_frame,
            from_=0.5,
            to=5.0,
            resolution=0.5,
            orient="horizontal",
            variable=self.search_speed_var,
            bg=self.frame_bg,
            fg=self.text_color,
            length=150
        )
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Apply settings button
        apply_frame = tk.Frame(content_frame, bg=self.bg_color)
        apply_frame.pack(pady=20)
        
        tk.Button(
            apply_frame,
            text="Apply Settings",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.apply_settings,
            width=20
        ).pack()
        
        # Status label
        self.settings_status = tk.Label(
            content_frame,
            text="",
            font=("Arial", 10),
            bg=self.bg_color,
            fg="green"
        )
        self.settings_status.pack(pady=10)
    
    def apply_settings(self):
        """Apply settings from settings tab"""
        # Update camera search parameters
        global CAMERA_YAW_RANGE, CAMERA_PITCH_RANGE, CAMERA_SEARCH_SPEED
        
        CAMERA_YAW_RANGE = self.yaw_range_var.get()
        CAMERA_PITCH_RANGE = self.pitch_range_var.get()
        CAMERA_SEARCH_SPEED = self.search_speed_var.get()
        
        # Update camera search object
        self.camera_search.yaw_target_left = CAMERA_HOME_POSITION["yaw"] - CAMERA_YAW_RANGE
        self.camera_search.yaw_target_right = CAMERA_HOME_POSITION["yaw"] + CAMERA_YAW_RANGE
        self.camera_search.pitch_target_up = CAMERA_HOME_POSITION["pitch"] - CAMERA_PITCH_RANGE
        self.camera_search.pitch_target_down = CAMERA_HOME_POSITION["pitch"] + CAMERA_PITCH_RANGE
        
        self.settings_status.config(
            text=f"Settings applied: Yaw ±{CAMERA_YAW_RANGE}°, Pitch ±{CAMERA_PITCH_RANGE}°, Speed {CAMERA_SEARCH_SPEED}°/step",
            fg="green"
        )
    
    # ============================================================================
    # MODE SWITCHING
    # ============================================================================
    def switch_to_manual(self):
        """Initialize in manual mode"""
        self.notebook.select(0)  # Manual tab
    
    def switch_to_auto(self):
        """Initialize in auto mode"""
        self.notebook.select(1)  # Auto tab
    
    def cleanup(self):
        """Cleanup resources with threaded architecture"""
        # Stop all flags first
        self.camera_running = False
        self.auto_tracking = False
        
        # Stop camera search
        self.camera_search.stop_search()
        
        # Stop camera threads
        self.stop_camera_threads()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close serial
        if self.ser:
            self.ser.close()
        
        # Close SPI
        if hasattr(self, 'spi_voltage') and self.spi_voltage:
            self.spi_voltage.close()
        
        if hasattr(self, 'spi_current') and self.spi_current:
            self.spi_current.close()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedRobotController(root)
    
    # Handle window closing
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()