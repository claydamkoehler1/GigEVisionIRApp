#!/usr/bin/env python3
# Modern Thermal Camera App with Unified UI
import sys
import os
import time
import sqlite3
import threading
import numpy as np
import json
from datetime import datetime, timedelta
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import PySpin
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# PLC Communication
try:
    from pycomm3 import LogixDriver
    PLC_AVAILABLE = True
except ImportError:
    print("Warning: pycomm3 not installed. PLC functionality will be disabled.")
    print("Install with: pip install pycomm3")
    PLC_AVAILABLE = False

# --- FLIR Calibration Constants from SpinView ---
R = 554118
B = 1492.15
F = 1.0
O = 48550

# --- Environmental Parameters ---
EMISSIVITY = 0.95
REFLECTED_TEMP_K = 293.15

# --- Conversion Functions ---
def raw_to_kelvin(S):
    try:
        return B / np.log((R / (S - O)) + F)
    except Exception as e:
        print(f"⚠️ Error converting raw value {S}: {e}")
        return float('nan')

def corrected_temperature_K(T_obj_K):
    try:
        return (T_obj_K - (1 - EMISSIVITY) * REFLECTED_TEMP_K) / EMISSIVITY
    except Exception as e:
        print(f"⚠️ Error applying emissivity correction: {e}")
        return T_obj_K

def kelvin_to_fahrenheit(K):
    return (K - 273.15) * 9 / 5 + 32

def smooth(data, window=7):
    return np.convolve(data, np.ones(window)/window, mode='valid') if len(data) >= window else data

# --- PLC Communication Class ---
class PLCHeatingProfile:
    """Class to handle Allen-Bradley PLC communication for heating profile data using pycomm3"""
    
    def __init__(self, ip_address="192.168.1.100"):
        """
        Initialize PLC connection
        Args:
            ip_address (str): IP address of the Allen-Bradley PLC
        """
        self.ip_address = ip_address
        self.plc = None
        self.connected = False
        self.heating_profile = None
        
    def connect(self):
        """Establish connection to PLC"""
        if not PLC_AVAILABLE:
            print("PLC library not available")
            return False
            
        try:
            self.plc = LogixDriver(self.ip_address)
            self.plc.open()
            
            # Test connection by reading a simple tag
            test_read = self.plc.read("Heat_Seq_Time_SP[1]")
            if test_read.error is None:
                self.connected = True
                print(f"✅ Connected to PLC at {self.ip_address}")
                return True
            else:
                print(f"❌ Failed to connect to PLC: {test_read.error}")
                return False
                
        except Exception as e:
            print(f"❌ PLC connection error: {e}")
            return False
    
    def disconnect(self):
        """Close PLC connection"""
        if self.plc:
            try:
                self.plc.close()
                self.connected = False
                print("PLC connection closed")
            except Exception as e:
                print(f"Error closing PLC connection: {e}")
    
    def read_heating_profile(self):
        """
        Read the complete 15-step heating profile from PLC
        Returns:
            dict: Heating profile data or None if failed
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            profile_data = {
                'steps': [],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Read data for all 15 steps
            for step in range(1, 16):  # Steps 1-15
                step_data = {}
                
                # Read time setpoint (INT)
                time_tag = f"Heat_Seq_Time_SP[{step}]"
                time_result = self.plc.read(time_tag)
                if time_result.error is None:
                    step_data['time_sp'] = time_result.value
                else:
                    print(f"Failed to read {time_tag}: {time_result.error}")
                    step_data['time_sp'] = None
                
                # Read start temperature setpoint (INT)
                start_temp_tag = f"Heat_Seq_Start_Temp_SP[{step}]"
                start_temp_result = self.plc.read(start_temp_tag)
                if start_temp_result.error is None:
                    step_data['start_temp_sp'] = start_temp_result.value
                else:
                    print(f"Failed to read {start_temp_tag}: {start_temp_result.error}")
                    step_data['start_temp_sp'] = None
                
                # Read end temperature setpoint (INT)
                end_temp_tag = f"Heat_Seq_End_Temp_SP[{step}]"
                end_temp_result = self.plc.read(end_temp_tag)
                if end_temp_result.error is None:
                    step_data['end_temp_sp'] = end_temp_result.value
                else:
                    print(f"Failed to read {end_temp_tag}: {end_temp_result.error}")
                    step_data['end_temp_sp'] = None
                
                # Read vacuum setpoint (REAL)
                vac_tag = f"Heat_Seq_Vac_SP[{step}]"
                vac_result = self.plc.read(vac_tag)
                if vac_result.error is None:
                    step_data['vac_sp'] = vac_result.value
                else:
                    print(f"Failed to read {vac_tag}: {vac_result.error}")
                    step_data['vac_sp'] = None
                
                step_data['step_number'] = step
                profile_data['steps'].append(step_data)
            
            self.heating_profile = profile_data
            print(f"✅ Successfully read heating profile with {len(profile_data['steps'])} steps")
            return profile_data
            
        except Exception as e:
            print(f"❌ Error reading heating profile: {e}")
            return None
    
    def get_heating_curve_data(self):
        """
        Convert heating profile to time-temperature curve for plotting
        Returns:
            tuple: (time_points, temp_points) or (None, None) if no data
        """
        if not self.heating_profile:
            return None, None
        
        try:
            time_points = [0]  # Start at time 0
            temp_points = []
            
            current_time = 0
            
            for step in self.heating_profile['steps']:
                if step['time_sp'] is None or step['start_temp_sp'] is None or step['end_temp_sp'] is None:
                    continue
                
                # Add start temperature at beginning of step
                if len(temp_points) == 0:
                    temp_points.append(step['start_temp_sp'])
                
                # Add end temperature at end of step
                current_time += step['time_sp']
                time_points.append(current_time)
                temp_points.append(step['end_temp_sp'])
            
            return time_points, temp_points
            
        except Exception as e:
            print(f"Error processing heating curve data: {e}")
            return None, None

class ThermalWidget(QLabel):
    """Custom widget for displaying thermal camera feed and handling ROI selection"""
    roi_updated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #444; background-color: black;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Camera feed will appear here")
        
        self.polygon_pts = []
        self.polygon_ready = False
        self.current_frame = None
        self.rotation_mode = 0
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.current_frame is not None:
            # Convert widget coordinates to image coordinates
            widget_rect = self.rect()
            if hasattr(self, 'scaled_size'):
                # Calculate the actual image position within the widget
                x_offset = (widget_rect.width() - self.scaled_size.width()) // 2
                y_offset = (widget_rect.height() - self.scaled_size.height()) // 2
                
                img_x = int((event.pos().x() - x_offset) * self.current_frame.shape[1] / self.scaled_size.width())
                img_y = int((event.pos().y() - y_offset) * self.current_frame.shape[0] / self.scaled_size.height())
                
                # Ensure coordinates are within image bounds
                img_x = max(0, min(img_x, self.current_frame.shape[1] - 1))
                img_y = max(0, min(img_y, self.current_frame.shape[0] - 1))
                
                self.polygon_pts.append((img_x, img_y))
                self.roi_updated.emit()

    def update_frame(self, frame):
        self.current_frame = frame.copy()
        
        # Apply rotation
        if self.rotation_mode == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_mode == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_mode == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Apply colormap
        vis = cv2.applyColorMap(
            cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_INFERNO)
        
        # Draw ROI polygon
        if self.polygon_pts:
            # Adjust points for rotation
            adjusted_pts = self.polygon_pts.copy()
            if self.rotation_mode != 0:
                adjusted_pts = self._rotate_points(adjusted_pts, frame.shape)
            
            pts = np.array(adjusted_pts, np.int32)
            cv2.polylines(vis, [pts], self.polygon_ready, (0, 255, 0), 2)
            for p in adjusted_pts:
                cv2.circle(vis, p, 4, (0, 255, 0), -1)
        
        # Convert to Qt format and display
        height, width, channel = vis.shape
        bytes_per_line = 3 * width
        q_image = QImage(vis.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale image to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.scaled_size = scaled_pixmap.size()
        self.setPixmap(scaled_pixmap)

    def _rotate_points(self, points, shape):
        """Rotate polygon points based on rotation mode"""
        rotated_points = []
        h, w = shape[:2]
        
        for x, y in points:
            if self.rotation_mode == 1:  # 90 degrees clockwise
                new_x, new_y = h - y - 1, x
            elif self.rotation_mode == 2:  # 180 degrees
                new_x, new_y = w - x - 1, h - y - 1
            elif self.rotation_mode == 3:  # 90 degrees counterclockwise
                new_x, new_y = y, w - x - 1
            else:
                new_x, new_y = x, y
            
            rotated_points.append((new_x, new_y))
        
        return rotated_points

    def clear_roi(self):
        self.polygon_pts.clear()
        self.polygon_ready = False
        self.roi_updated.emit()

    def finish_roi(self):
        if len(self.polygon_pts) >= 3:
            self.polygon_ready = True
            self.roi_updated.emit()
            return True
        return False

    def rotate_view(self):
        self.rotation_mode = (self.rotation_mode + 1) % 4
        return self.rotation_mode * 90

class GraphWidget(FigureCanvas):
    """Custom widget for displaying temperature graphs"""
    
    def __init__(self):
        self.figure = Figure(figsize=(10, 6), facecolor='#2b2b2b')
        super().__init__(self.figure)
        
        self.ax = self.figure.add_subplot(111, facecolor='#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        self.timestamps = []
        self.maxs = []
        self.mins = []
        self.comparison_db = None
        self.heating_profile_data = None
        
        # Adjust layout with better margins to prevent Y-axis cutoff
        self.figure.tight_layout(pad=2.0)
        self.figure.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.90)

    def update_graph(self, db_path):
        if not db_path or not os.path.exists(db_path):
            return
            
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT * FROM readings ORDER BY ROWID ASC").fetchall()
            conn.close()
            
            if not rows:
                return

            self.timestamps = [datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") for r in rows]
            self.maxs = [r[2] for r in rows]
            self.mins = [r[3] for r in rows]

            smooth_max = smooth(self.maxs)
            smooth_min = smooth(self.mins)

            valid_timestamps = self.timestamps[len(self.timestamps)-len(smooth_max):]

            self.ax.clear()
            self.ax.plot(valid_timestamps, smooth_max, label="Current Max °F", color='#ff6b6b', linewidth=2)
            self.ax.plot(valid_timestamps, smooth_min, label="Current Min °F", color='#4ecdc4', linewidth=2)
            
            # Plot PLC heating curve if available
            if self.heating_profile_data and len(valid_timestamps) > 0:
                self.plot_heating_curve(valid_timestamps[0])
            
            # Plot comparison data if available
            if self.comparison_db and os.path.exists(self.comparison_db):
                try:
                    comp_conn = sqlite3.connect(self.comparison_db)
                    comp_rows = comp_conn.execute("SELECT * FROM readings ORDER BY ROWID ASC").fetchall()
                    comp_conn.close()
                    
                    if comp_rows:
                        comp_maxs = [r[2] for r in comp_rows]
                        comp_mins = [r[3] for r in comp_rows]
                        
                        comp_smooth_max = smooth(comp_maxs)
                        comp_smooth_min = smooth(comp_mins)
                        
                        # Normalize comparison timestamps to align with current data
                        # Use the current data's time range for comparison data
                        if len(valid_timestamps) > 0 and len(comp_smooth_max) > 0:
                            # Calculate the actual time interval from the comparison data
                            comp_timestamps = [datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") for r in comp_rows]
                            comp_valid_timestamps = comp_timestamps[len(comp_timestamps)-len(comp_smooth_max):]
                            
                            if len(comp_valid_timestamps) > 1:
                                # Calculate the actual duration and time intervals from the comparison data
                                comp_original_duration = (comp_valid_timestamps[-1] - comp_valid_timestamps[0]).total_seconds()
                                comp_time_intervals = []
                                for i in range(1, len(comp_valid_timestamps)):
                                    interval = (comp_valid_timestamps[i] - comp_valid_timestamps[i-1]).total_seconds()
                                    comp_time_intervals.append(interval)
                                
                                # Create normalized timestamps starting from current data's start time
                                # but preserving the original time intervals and duration
                                start_time = valid_timestamps[0]
                                comp_normalized_timestamps = [start_time]
                                
                                # Build timestamps using original intervals
                                current_time = start_time
                                for i, interval in enumerate(comp_time_intervals):
                                    current_time = current_time + timedelta(seconds=interval)
                                    comp_normalized_timestamps.append(current_time)
                                
                                # Only plot if the comparison data doesn't exceed current timeline
                                current_duration = (valid_timestamps[-1] - valid_timestamps[0]).total_seconds()
                                if comp_original_duration <= current_duration + 60:  # Allow 1 minute buffer
                                    # Plot comparison data with preserved time scale
                                    self.ax.plot(comp_normalized_timestamps, comp_smooth_max, 
                                               label=f"Comparison Max °F ({os.path.basename(self.comparison_db)})", 
                                               color='#ff9999', linewidth=2, linestyle='--', alpha=0.7)
                                    self.ax.plot(comp_normalized_timestamps, comp_smooth_min, 
                                               label=f"Comparison Min °F ({os.path.basename(self.comparison_db)})", 
                                               color='#99d4d4', linewidth=2, linestyle='--', alpha=0.7)
                                else:
                                    # Plot only the portion that fits within current timeline
                                    truncate_points = 0
                                    running_time = 0
                                    for i, interval in enumerate(comp_time_intervals):
                                        running_time += interval
                                        if running_time > current_duration:
                                            truncate_points = i + 1
                                            break
                                    
                                    if truncate_points > 0:
                                        trunc_timestamps = comp_normalized_timestamps[:truncate_points]
                                        trunc_max = comp_smooth_max[:truncate_points]
                                        trunc_min = comp_smooth_min[:truncate_points]
                                        
                                        self.ax.plot(trunc_timestamps, trunc_max, 
                                                   label=f"Comparison Max °F ({os.path.basename(self.comparison_db)}) [truncated]", 
                                                   color='#ff9999', linewidth=2, linestyle='--', alpha=0.7)
                                        self.ax.plot(trunc_timestamps, trunc_min, 
                                                   label=f"Comparison Min °F ({os.path.basename(self.comparison_db)}) [truncated]", 
                                                   color='#99d4d4', linewidth=2, linestyle='--', alpha=0.7)
                            else:
                                # Single point comparison - plot at current time
                                self.ax.plot([valid_timestamps[0]], comp_smooth_max, 
                                           label=f"Comparison Max °F ({os.path.basename(self.comparison_db)})", 
                                           color='#ff9999', linewidth=2, linestyle='--', alpha=0.7, marker='o')
                                self.ax.plot([valid_timestamps[0]], comp_smooth_min, 
                                           label=f"Comparison Min °F ({os.path.basename(self.comparison_db)})", 
                                           color='#99d4d4', linewidth=2, linestyle='--', alpha=0.7, marker='o')
                            
                            self.ax.set_title("Temperature Monitoring - Current vs Comparison", color='white', fontsize=12, fontweight='bold')
                        else:
                            self.ax.set_title("Live ROI Temperature Monitoring", color='white', fontsize=12, fontweight='bold')
                    else:
                        self.ax.set_title("Live ROI Temperature Monitoring", color='white', fontsize=12, fontweight='bold')
                        
                except Exception as e:
                    print(f"Error loading comparison data: {e}")
                    self.ax.set_title("Live ROI Temperature Monitoring", color='white', fontsize=12, fontweight='bold')
            else:
                self.ax.set_title("Live ROI Temperature Monitoring", color='white', fontsize=12, fontweight='bold')
            
            self.ax.grid(True, alpha=0.3, color='white')
            self.ax.legend(loc="upper left", facecolor='#2b2b2b', edgecolor='white')
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            # Style the graph
            self.ax.tick_params(colors='white')
            self.ax.spines['bottom'].set_color('white')
            self.ax.spines['top'].set_color('white') 
            self.ax.spines['right'].set_color('white')
            self.ax.spines['left'].set_color('white')
            
            self.figure.autofmt_xdate()
            self.draw()
            
        except Exception as e:
            print(f"Error updating graph: {e}")

    def set_comparison_db(self, db_path):
        """Set the database to compare against"""
        self.comparison_db = db_path

    def clear_comparison(self):
        """Clear the comparison database"""
        self.comparison_db = None

    def set_heating_profile(self, profile_data):
        """Set the PLC heating profile data for display"""
        self.heating_profile_data = profile_data

    def plot_heating_curve(self, recording_start_time):
        """Plot the PLC heating curve scaled to match current recording duration"""
        if not self.heating_profile_data:
            return
            
        try:
            # First, calculate the total duration of the heating profile
            total_profile_time = 0
            valid_steps = []
            
            for step in self.heating_profile_data['steps']:
                if (step.get('time_sp') is not None and 
                    step.get('start_temp_sp') is not None and 
                    step.get('end_temp_sp') is not None):
                    valid_steps.append(step)
                    total_profile_time += step['time_sp']
            
            if not valid_steps or total_profile_time <= 0:
                return
            
            # Get current recording duration
            current_time = datetime.now()
            recording_duration_seconds = (current_time - recording_start_time).total_seconds()
            
            # If we have less than 30 seconds of recording, use a minimum window
            if recording_duration_seconds < 30:
                recording_duration_seconds = 30
            
            # Calculate scaling factor
            total_profile_seconds = total_profile_time * 60  # Convert minutes to seconds
            scale_factor = recording_duration_seconds / total_profile_seconds
            
            # Generate scaled heating curve data points
            time_points = []
            temp_points = []
            current_scaled_time = 0
            
            # Add start point
            time_points.append(recording_start_time)
            temp_points.append(float(valid_steps[0]['start_temp_sp']))
            
            for step in valid_steps:
                # Scale the step time and add end point
                step_duration = step['time_sp'] * 60 * scale_factor  # Convert to seconds and scale
                current_scaled_time += step_duration
                end_timestamp = recording_start_time + timedelta(seconds=current_scaled_time)
                time_points.append(end_timestamp)
                temp_points.append(float(step['end_temp_sp']))
            
            if len(time_points) >= 2:
                # Plot the scaled heating curve with interpolation
                self.ax.plot(time_points, temp_points, 
                           label=f"PLC Heating Profile (Scaled)", 
                           color='#ffaa00', linewidth=2.5, linestyle='-', alpha=0.8)
                
                # Add markers for step transitions
                self.ax.scatter(time_points[1:], temp_points[1:], 
                              color='#ffaa00', s=30, alpha=0.9, zorder=5)
                
                # Add text annotation showing original vs scaled duration
                total_hours = total_profile_time / 60
                scaled_hours = recording_duration_seconds / 3600
                annotation_text = f"Original: {total_hours:.1f}h → Scaled: {scaled_hours:.1f}h"
                
                # Position annotation in upper right
                self.ax.text(0.98, 0.02, annotation_text, transform=self.ax.transAxes,
                           fontsize=8, color='#ffaa00', alpha=0.8,
                           horizontalalignment='right', verticalalignment='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
        except Exception as e:
            print(f"Error plotting scaled heating curve: {e}")

class TestProfileTab(QWidget):
    """Tab for creating and managing test profiles"""
    
    def __init__(self):
        super().__init__()
        self.profiles_dir = os.path.join(os.getcwd(), "LD1_Test_Profiles")
        self.setup_ui()
        self.refresh_profiles()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Test Profile Management - Standardized Testing")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4ecdc4; margin: 10px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)
        
        # Main content layout
        content_layout = QHBoxLayout()
        
        # Left panel - Profile management
        left_panel = QVBoxLayout()
        
        # Profile actions group
        actions_group = QGroupBox("Profile Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.new_profile_btn = QPushButton("Create New Test Profile")
        self.new_profile_btn.clicked.connect(self.create_new_profile)
        self.new_profile_btn.setStyleSheet("font-size: 12px; padding: 12px;")
        actions_layout.addWidget(self.new_profile_btn)
        
        self.edit_profile_btn = QPushButton("Edit Selected Profile")
        self.edit_profile_btn.clicked.connect(self.edit_selected_profile)
        self.edit_profile_btn.setEnabled(False)
        actions_layout.addWidget(self.edit_profile_btn)
        
        self.delete_profile_btn = QPushButton("Delete Selected Profile")
        self.delete_profile_btn.clicked.connect(self.delete_selected_profile)
        self.delete_profile_btn.setEnabled(False)
        self.delete_profile_btn.setStyleSheet("background-color: #e74c3c;")
        actions_layout.addWidget(self.delete_profile_btn)
        
        left_panel.addWidget(actions_group)
        
        # Existing profiles group
        profiles_group = QGroupBox("Existing Test Profiles")
        profiles_layout = QVBoxLayout(profiles_group)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Profiles")
        refresh_btn.clicked.connect(self.refresh_profiles)
        profiles_layout.addWidget(refresh_btn)
        
        # Profiles list
        self.profiles_list = QListWidget()
        self.profiles_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #0d7377;
            }
            QListWidget::item:hover {
                background-color: #14a085;
            }
        """)
        self.profiles_list.itemSelectionChanged.connect(self.on_profile_selected)
        profiles_layout.addWidget(self.profiles_list)
        
        left_panel.addWidget(profiles_group)
        
        # Status
        self.status_label = QLabel("Ready to create or manage test profiles")
        self.status_label.setStyleSheet("color: #ffd93d; font-weight: bold; padding: 10px;")
        left_panel.addWidget(self.status_label)
        
        content_layout.addLayout(left_panel, 1)
        
        # Right panel - Profile details
        right_panel = QVBoxLayout()
        
        # Profile details group
        details_group = QGroupBox("Profile Details")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QLabel("Select a profile to view details")
        self.details_text.setStyleSheet("color: #bbb; font-size: 11px; padding: 10px;")
        self.details_text.setWordWrap(True)
        self.details_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        details_layout.addWidget(self.details_text)
        
        right_panel.addWidget(details_group)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_text = QLabel(
            "Test Profile Workflow:\n\n"
            "1. Create New Profile: Define profile name and custom variables\n"
            "2. Add Variables: Specify variable names, data types, and optional default values\n"
            "3. Save Profile: Profile becomes available for use in Live Recording\n"
            "4. Use Profile: Select profile in Live Recording tab to prompt for variable values\n\n"
            "Supported Data Types:\n"
            "• String: Text values (e.g., color, operator name)\n"
            "• Number: Numeric values (e.g., temperature, pressure)\n"
            "• Boolean: True/False values (e.g., pre-heated, cleaned)\n"
            "• Date: Date values (e.g., sample date, calibration date)"
        )
        instructions_text.setWordWrap(True)
        instructions_text.setStyleSheet("color: #bbb; font-size: 10px;")
        instructions_layout.addWidget(instructions_text)
        right_panel.addWidget(instructions_group)
        
        content_layout.addLayout(right_panel, 1)
        
        layout.addLayout(content_layout)
    
    def refresh_profiles(self):
        """Refresh the list of available test profiles"""
        self.profiles_list.clear()
        
        # Ensure profiles directory exists
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        try:
            profile_files = [f for f in os.listdir(self.profiles_dir) if f.endswith('.json')]
            
            if not profile_files:
                self.status_label.setText("No test profiles found - create your first profile!")
                return
            
            for profile_file in sorted(profile_files):
                profile_path = os.path.join(self.profiles_dir, profile_file)
                try:
                    with open(profile_path, 'r') as f:
                        profile_data = json.load(f)
                    
                    profile_name = profile_data.get('name', 'Unknown')
                    var_count = len(profile_data.get('variables', []))
                    created_date = profile_data.get('created_date', 'Unknown')
                    
                    item_text = f"{profile_name} ({var_count} variables, created: {created_date})"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, profile_path)
                    self.profiles_list.addItem(item)
                    
                except Exception as e:
                    print(f"Error reading profile {profile_file}: {e}")
            
            self.status_label.setText(f"Found {len(profile_files)} test profiles")
            
        except Exception as e:
            self.status_label.setText(f"Error loading profiles: {str(e)}")
    
    def on_profile_selected(self):
        """Handle profile selection"""
        selected_items = self.profiles_list.selectedItems()
        
        if selected_items:
            self.edit_profile_btn.setEnabled(True)
            self.delete_profile_btn.setEnabled(True)
            
            # Load and display profile details
            profile_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
            self.display_profile_details(profile_path)
        else:
            self.edit_profile_btn.setEnabled(False)
            self.delete_profile_btn.setEnabled(False)
            self.details_text.setText("Select a profile to view details")
    
    def display_profile_details(self, profile_path):
        """Display details of the selected profile"""
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            
            details = f"PROFILE: {profile_data.get('name', 'Unknown')}\n\n"
            details += f"Description: {profile_data.get('description', 'No description')}\n"
            details += f"Created: {profile_data.get('created_date', 'Unknown')}\n"
            details += f"Last Modified: {profile_data.get('modified_date', 'Unknown')}\n\n"
            
            variables = profile_data.get('variables', [])
            if variables:
                details += "CUSTOM VARIABLES:\n"
                for var in variables:
                    details += f"• {var['name']} ({var['type']})"
                    if var.get('default_value'):
                        details += f" - Default: {var['default_value']}"
                    if var.get('description'):
                        details += f"\n  Description: {var['description']}"
                    details += "\n"
            else:
                details += "No custom variables defined\n"
            
            self.details_text.setText(details)
            
        except Exception as e:
            self.details_text.setText(f"Error loading profile details: {str(e)}")
    
    def create_new_profile(self):
        """Open the profile creation wizard"""
        wizard = TestProfileWizard(self)
        if wizard.exec() == QDialog.DialogCode.Accepted:
            self.refresh_profiles()
            self.status_label.setText("New test profile created successfully!")
    
    def edit_selected_profile(self):
        """Edit the selected profile"""
        selected_items = self.profiles_list.selectedItems()
        if not selected_items:
            return
        
        profile_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        wizard = TestProfileWizard(self, profile_path)
        if wizard.exec() == QDialog.DialogCode.Accepted:
            self.refresh_profiles()
            self.status_label.setText("Test profile updated successfully!")
    
    def delete_selected_profile(self):
        """Delete the selected profile"""
        selected_items = self.profiles_list.selectedItems()
        if not selected_items:
            return
        
        profile_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        profile_name = os.path.basename(profile_path).replace('.json', '')
        
        # Confirmation dialog
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Delete Test Profile")
        msg.setText(f"Are you sure you want to delete the test profile '{profile_name}'?")
        msg.setInformativeText("This action cannot be undone.")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        msg.setStyleSheet(self.parent().parent().styleSheet())
        
        if msg.exec() == QMessageBox.StandardButton.Yes:
            try:
                os.remove(profile_path)
                self.refresh_profiles()
                self.status_label.setText(f"Test profile '{profile_name}' deleted successfully!")
            except Exception as e:
                self.status_label.setText(f"Error deleting profile: {str(e)}")

class TestProfileWizard(QDialog):
    """Wizard for creating and editing test profiles"""
    
    def __init__(self, parent=None, profile_path=None):
        super().__init__(parent)
        self.profile_path = profile_path
        self.is_editing = profile_path is not None
        self.variables = []
        
        self.setWindowTitle("Test Profile Wizard" if not self.is_editing else "Edit Test Profile")
        self.setModal(True)
        self.resize(600, 500)
        
        # Apply comprehensive dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #444;
                color: black;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
                selection-background-color: #0d7377;
                selection-color: white;
            }
            QLineEdit:focus {
                border-color: #0d7377;
                background-color: white;
                color: black;
            }
            QTextEdit {
                background-color: #444;
                color: black;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
            }
            QTextEdit:focus {
                border-color: #0d7377;
            }
            QGroupBox {
                color: white;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4ecdc4;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            QScrollArea {
                background-color: #333;
                border: 1px solid #444;
            }
            QScrollBar:vertical {
                background-color: #444;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #666;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
        """)
        
        self.setup_ui()
        
        # Load existing profile if editing
        if self.is_editing:
            self.load_existing_profile()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_text = "Edit Test Profile" if self.is_editing else "Create New Test Profile"
        header_label = QLabel(header_text)
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4ecdc4; margin: 10px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)
        
        # Basic info group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout(basic_group)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter profile name (e.g., 'Color Analysis Test')")
        basic_layout.addRow("Profile Name:", self.name_input)
        
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Enter profile description (optional)")
        self.description_input.setMaximumHeight(60)
        basic_layout.addRow("Description:", self.description_input)
        
        layout.addWidget(basic_group)
        
        # Variables group
        variables_group = QGroupBox("Custom Variables")
        variables_layout = QVBoxLayout(variables_group)
        
        # Add variable button
        add_var_btn = QPushButton("Add New Variable")
        add_var_btn.clicked.connect(self.add_variable)
        variables_layout.addWidget(add_var_btn)
        
        # Variables list
        self.variables_widget = QScrollArea()
        self.variables_widget.setWidgetResizable(True)
        self.variables_widget.setMaximumHeight(200)
        
        self.variables_container = QWidget()
        self.variables_layout = QVBoxLayout(self.variables_container)
        self.variables_widget.setWidget(self.variables_container)
        
        variables_layout.addWidget(self.variables_widget)
        
        layout.addWidget(variables_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        save_text = "Update Profile" if self.is_editing else "Create Profile"
        self.save_btn = QPushButton(save_text)
        self.save_btn.clicked.connect(self.save_profile)
        self.save_btn.setStyleSheet("background-color: #0d7377; font-weight: bold;")
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
    
    def load_existing_profile(self):
        """Load existing profile data for editing"""
        try:
            with open(self.profile_path, 'r') as f:
                profile_data = json.load(f)
            
            self.name_input.setText(profile_data.get('name', ''))
            self.description_input.setPlainText(profile_data.get('description', ''))
            
            self.variables = profile_data.get('variables', [])
            self.refresh_variables_display()
            
        except Exception as e:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText(f"Failed to load profile: {str(e)}")
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2b2b2b;
                    color: white;
                }
                QMessageBox QLabel {
                    color: white;
                }
                QMessageBox QPushButton {
                    background-color: #0d7377;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QMessageBox QPushButton:hover {
                    background-color: #14a085;
                }
            """)
            msg_box.exec()
    
    def add_variable(self):
        """Add a new variable to the profile"""
        dialog = VariableDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            var_data = dialog.get_variable_data()
            self.variables.append(var_data)
            self.refresh_variables_display()
    
    def refresh_variables_display(self):
        """Refresh the variables display"""
        # Clear existing widgets
        for i in reversed(range(self.variables_layout.count())):
            child = self.variables_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Add variable widgets
        for i, var in enumerate(self.variables):
            var_widget = self.create_variable_widget(var, i)
            self.variables_layout.addWidget(var_widget)
        
        # Add stretch to push everything to top
        self.variables_layout.addStretch()
    
    def create_variable_widget(self, var_data, index):
        """Create a widget to display a variable"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setStyleSheet("""
            QFrame { 
                border: 1px solid #444; 
                padding: 5px; 
                margin: 2px; 
                background-color: white;
            }
            QLabel {
                color: black;
                background-color: transparent;
            }
        """)
        
        layout = QHBoxLayout(widget)
        
        # Variable info
        info_text = f"{var_data['name']} ({var_data['type']})"
        if var_data.get('default_value'):
            info_text += f" - Default: {var_data['default_value']}"
        if var_data.get('end_of_cycle', False):
            info_text += " [END-OF-CYCLE]"
        
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        # Edit button
        edit_btn = QPushButton("Edit")
        edit_btn.setMaximumWidth(60)
        edit_btn.clicked.connect(lambda: self.edit_variable(index))
        layout.addWidget(edit_btn)
        
        # Delete button
        delete_btn = QPushButton("Delete")
        delete_btn.setMaximumWidth(60)
        delete_btn.setStyleSheet("background-color: #e74c3c;")
        delete_btn.clicked.connect(lambda: self.delete_variable(index))
        layout.addWidget(delete_btn)
        
        return widget
    
    def edit_variable(self, index):
        """Edit an existing variable"""
        if 0 <= index < len(self.variables):
            dialog = VariableDialog(self, self.variables[index])
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.variables[index] = dialog.get_variable_data()
                self.refresh_variables_display()
    
    def delete_variable(self, index):
        """Delete a variable"""
        if 0 <= index < len(self.variables):
            var_name = self.variables[index]['name']
            reply = QMessageBox(self)
            reply.setIcon(QMessageBox.Icon.Question)
            reply.setWindowTitle("Delete Variable")
            reply.setText(f"Are you sure you want to delete the variable '{var_name}'?")
            reply.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            reply.setDefaultButton(QMessageBox.StandardButton.No)
            reply.setStyleSheet("""
                QMessageBox {
                    background-color: #2b2b2b;
                    color: white;
                }
                QMessageBox QLabel {
                    color: white;
                }
                QMessageBox QPushButton {
                    background-color: #0d7377;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QMessageBox QPushButton:hover {
                    background-color: #14a085;
                }
            """)
            result = reply.exec()
            
            if result == QMessageBox.StandardButton.Yes:
                del self.variables[index]
                self.refresh_variables_display()
    
    def save_profile(self):
        """Save the test profile"""
        profile_name = self.name_input.text().strip()
        if not profile_name:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("Please enter a profile name.")
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2b2b2b;
                    color: white;
                }
                QMessageBox QLabel {
                    color: white;
                }
                QMessageBox QPushButton {
                    background-color: #0d7377;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QMessageBox QPushButton:hover {
                    background-color: #14a085;
                }
            """)
            msg_box.exec()
            return
        
        # Prepare profile data
        profile_data = {
            'name': profile_name,
            'description': self.description_input.toPlainText().strip(),
            'variables': self.variables,
            'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S") if not self.is_editing else None,
            'modified_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # If editing, preserve the original creation date
        if self.is_editing and self.profile_path:
            try:
                with open(self.profile_path, 'r') as f:
                    existing_data = json.load(f)
                profile_data['created_date'] = existing_data.get('created_date', profile_data['created_date'])
            except:
                pass
        
        # Save to file
        try:
            profiles_dir = os.path.join(os.getcwd(), "LD1_Test_Profiles")
            os.makedirs(profiles_dir, exist_ok=True)
            
            # Create safe filename
            safe_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            
            if self.is_editing and self.profile_path:
                file_path = self.profile_path
            else:
                file_path = os.path.join(profiles_dir, f"{safe_name}.json")
            
            # Check if file exists (for new profiles)
            if not self.is_editing and os.path.exists(file_path):
                reply = QMessageBox(self)
                reply.setIcon(QMessageBox.Icon.Question)
                reply.setWindowTitle("File Exists")
                reply.setText("A profile with this name already exists. Overwrite?")
                reply.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                reply.setDefaultButton(QMessageBox.StandardButton.No)
                reply.setStyleSheet("""
                    QMessageBox {
                        background-color: #2b2b2b;
                        color: white;
                    }
                    QMessageBox QLabel {
                        color: white;
                    }
                    QMessageBox QPushButton {
                        background-color: #0d7377;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    QMessageBox QPushButton:hover {
                        background-color: #14a085;
                    }
                """)
                result = reply.exec()
                if result != QMessageBox.StandardButton.Yes:
                    return
            
            with open(file_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            self.accept()
            
        except Exception as e:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText(f"Failed to save profile: {str(e)}")
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2b2b2b;
                    color: white;
                }
                QMessageBox QLabel {
                    color: white;
                }
                QMessageBox QPushButton {
                    background-color: #0d7377;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QMessageBox QPushButton:hover {
                    background-color: #14a085;
                }
            """)
            msg_box.exec()

class VariableDialog(QDialog):
    """Dialog for adding/editing custom variables"""
    
    def __init__(self, parent=None, variable_data=None):
        super().__init__(parent)
        self.variable_data = variable_data or {}
        self.is_editing = variable_data is not None
        
        self.setWindowTitle("Edit Variable" if self.is_editing else "Add Variable")
        self.setModal(True)
        self.resize(400, 300)
        
        # Apply comprehensive dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
                background-color: transparent;
                font-weight: bold;
                margin: 5px 0px 2px 0px;
            }
            QLineEdit {
                background-color: #444;
                color: black;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
                selection-background-color: #0d7377;
                selection-color: white;
            }
            QLineEdit:focus {
                border-color: #0d7377;
                background-color: white;
                color: black;
            }
            QTextEdit {
                background-color: #444;
                color: black;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
                selection-background-color: #0d7377;
                selection-color: white;
            }
            QTextEdit:focus {
                border-color: #0d7377;
                background-color: white;
                color: black;
            }
            QComboBox {
                background-color: #444;
                color: white;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
                min-width: 100px;
            }
            QComboBox:focus {
                border-color: #0d7377;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #555;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #444;
                color: white;
                selection-background-color: #0d7377;
                border: 1px solid #666;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
        """)
        
        self.setup_ui()
        
        if self.is_editing:
            self.load_variable_data()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Variable name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter variable name (e.g., 'Color', 'Operator')")
        layout.addWidget(QLabel("Variable Name:"))
        layout.addWidget(self.name_input)
        
        # Data type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["String", "Number", "Boolean", "Date"])
        layout.addWidget(QLabel("Data Type:"))
        layout.addWidget(self.type_combo)
        
        # Default value
        self.default_input = QLineEdit()
        self.default_input.setPlaceholderText("Optional default value")
        layout.addWidget(QLabel("Default Value (Optional):"))
        layout.addWidget(self.default_input)
        
        # Description
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Optional description or instructions")
        self.description_input.setMaximumHeight(80)
        layout.addWidget(QLabel("Description (Optional):"))
        layout.addWidget(self.description_input)
        
        # End of cycle checkbox
        self.end_of_cycle_checkbox = QCheckBox("End-of-Cycle Variable")
        self.end_of_cycle_checkbox.setToolTip("Check this if the variable should be entered at the END of the recording cycle (e.g., final quality, completion status)")
        layout.addWidget(self.end_of_cycle_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        
        save_btn = QPushButton("Save Variable")
        save_btn.clicked.connect(self.save_variable)
        save_btn.setStyleSheet("background-color: #0d7377; font-weight: bold;")
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
    
    def load_variable_data(self):
        """Load existing variable data for editing"""
        self.name_input.setText(self.variable_data.get('name', ''))
        
        var_type = self.variable_data.get('type', 'String')
        index = self.type_combo.findText(var_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)
        
        self.default_input.setText(str(self.variable_data.get('default_value', '')))
        self.description_input.setPlainText(self.variable_data.get('description', ''))
        
        # Load end-of-cycle setting
        self.end_of_cycle_checkbox.setChecked(self.variable_data.get('end_of_cycle', False))
    
    def save_variable(self):
        """Save the variable"""
        name = self.name_input.text().strip()
        if not name:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("Please enter a variable name.")
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2b2b2b;
                    color: white;
                }
                QMessageBox QLabel {
                    color: white;
                }
                QMessageBox QPushButton {
                    background-color: #0d7377;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QMessageBox QPushButton:hover {
                    background-color: #14a085;
                }
            """)
            msg_box.exec()
            return
        
        self.accept()
    
    def get_variable_data(self):
        """Get the variable data"""
        return {
            'name': self.name_input.text().strip(),
            'type': self.type_combo.currentText(),
            'default_value': self.default_input.text().strip(),
            'description': self.description_input.toPlainText().strip(),
            'end_of_cycle': self.end_of_cycle_checkbox.isChecked()
        }

class EndOfCycleDialog(QDialog):
    """Dialog for collecting end-of-cycle variables when recording stops"""
    
    def __init__(self, parent, end_of_cycle_variables):
        super().__init__(parent)
        self.end_of_cycle_variables = end_of_cycle_variables
        self.variable_inputs = {}
        
        self.setWindowTitle("End-of-Cycle Variables")
        self.setModal(True)
        self.resize(500, 400)
        
        # Apply comprehensive dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #444;
                color: black;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
                selection-background-color: #0d7377;
                selection-color: white;
            }
            QLineEdit:focus {
                border-color: #0d7377;
                background-color: white;
                color: black;
            }
            QCheckBox {
                color: white;
                font-size: 11px;
            }
            QCheckBox::indicator {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 3px;
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:checked {
                background-color: #0d7377;
                border-color: #0d7377;
            }
            QCheckBox::indicator:hover {
                border-color: #14a085;
            }
            QGroupBox {
                color: white;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4ecdc4;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            QScrollArea {
                background-color: #333;
                border: 1px solid #444;
            }
            QScrollBar:vertical {
                background-color: #444;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #666;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
        """)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Recording Complete - Enter End-of-Cycle Variables")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4ecdc4; margin: 10px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)
        
        # Instructions
        instructions = QLabel("Please fill in the following variables that are measured at the end of the test cycle:")
        instructions.setStyleSheet("color: #bbb; font-size: 11px; margin: 10px; padding: 10px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Variables group
        variables_group = QGroupBox("End-of-Cycle Variables")
        variables_layout = QVBoxLayout(variables_group)
        
        # Create input fields for each end-of-cycle variable
        for var in self.end_of_cycle_variables:
            var_name = var['name']
            var_type = var['type']
            default_value = var.get('default_value', '')
            description = var.get('description', '')
            
            # Create input widget based on type
            if var_type == "Boolean":
                input_widget = QCheckBox()
                if default_value.lower() in ['true', '1', 'yes']:
                    input_widget.setChecked(True)
            elif var_type == "Date":
                input_widget = QLineEdit()
                input_widget.setPlaceholderText("YYYY-MM-DD or leave blank for today")
                if default_value:
                    input_widget.setText(default_value)
            elif var_type == "Number":
                input_widget = QLineEdit()
                input_widget.setPlaceholderText("Enter numeric value")
                if default_value:
                    input_widget.setText(str(default_value))
            else:  # String
                input_widget = QLineEdit()
                input_widget.setPlaceholderText("Enter text value")
                if default_value:
                    input_widget.setText(str(default_value))
            
            # Add label and input
            label_text = f"{var_name} ({var_type}):"
            if description:
                label_text += f"\n{description}"
            
            label = QLabel(label_text)
            label.setWordWrap(True)
            variables_layout.addWidget(label)
            variables_layout.addWidget(input_widget)
            
            self.variable_inputs[var_name] = {
                'widget': input_widget,
                'type': var_type
            }
        
        layout.addWidget(variables_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        skip_btn = QPushButton("Skip (No Values)")
        skip_btn.clicked.connect(self.reject)
        skip_btn.setStyleSheet("background-color: #666;")
        button_layout.addWidget(skip_btn)
        
        button_layout.addStretch()
        
        save_btn = QPushButton("Save Values")
        save_btn.clicked.connect(self.accept)
        save_btn.setStyleSheet("background-color: #0d7377; font-weight: bold;")
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
    
    def get_variable_values(self):
        """Get the values entered for end-of-cycle variables"""
        values = {}
        
        for var_name, input_info in self.variable_inputs.items():
            widget = input_info['widget']
            var_type = input_info['type']
            
            if var_type == "Boolean":
                values[var_name] = widget.isChecked()
            elif var_type == "Date":
                date_text = widget.text().strip()
                if not date_text:
                    values[var_name] = datetime.now().strftime("%Y-%m-%d")
                else:
                    values[var_name] = date_text
            elif var_type == "Number":
                try:
                    num_text = widget.text().strip()
                    values[var_name] = float(num_text) if num_text else 0.0
                except ValueError:
                    values[var_name] = 0.0
            else:  # String
                values[var_name] = widget.text().strip()
        
        return values

class DataAnalysisTab(QWidget):
    """Tab for analyzing all database files"""
    
    def __init__(self):
        super().__init__()
        self.db_dir = os.path.join(os.getcwd(), "LD1_Cycle_Database")
        self.selected_databases = []
        self.setup_ui()
        self.refresh_database_list()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Data Analysis - Compare Multiple Runs")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4ecdc4; margin: 10px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)
        
        # Main content layout
        content_layout = QHBoxLayout()
        
        # Left panel - Database selection
        left_panel = QVBoxLayout()
        
        # Database list group
        db_group = QGroupBox("Available Database Files")
        db_layout = QVBoxLayout(db_group)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Database List")
        refresh_btn.clicked.connect(self.refresh_database_list)
        db_layout.addWidget(refresh_btn)
        
        # Database list
        self.db_list = QListWidget()
        self.db_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #0d7377;
            }
            QListWidget::item:hover {
                background-color: #14a085;
            }
        """)
        self.db_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        db_layout.addWidget(self.db_list)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_databases)
        btn_layout.addWidget(self.select_all_btn)
        
        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        btn_layout.addWidget(self.clear_selection_btn)
        
        db_layout.addLayout(btn_layout)
        
        # Analysis buttons
        analysis_layout = QVBoxLayout()
        
        self.compare_btn = QPushButton("Analyze Selected Runs")
        self.compare_btn.clicked.connect(self.compare_selected_runs)
        analysis_layout.addWidget(self.compare_btn)
        
        self.statistics_btn = QPushButton("Generate Statistics Report")
        self.statistics_btn.clicked.connect(self.generate_statistics)
        analysis_layout.addWidget(self.statistics_btn)
        
        self.export_combined_btn = QPushButton("Export Combined CSV")
        self.export_combined_btn.clicked.connect(self.export_combined_csv)
        analysis_layout.addWidget(self.export_combined_btn)
        
        db_layout.addLayout(analysis_layout)
        
        left_panel.addWidget(db_group)
        
        # Status and info
        self.info_label = QLabel("Select database files to analyze")
        self.info_label.setStyleSheet("color: #ffd93d; font-weight: bold; padding: 10px;")
        left_panel.addWidget(self.info_label)
        
        content_layout.addLayout(left_panel, 1)
        
        # Right panel - Analysis results
        right_panel = QVBoxLayout()
        
        # Analysis graph
        analysis_group = QGroupBox("Analysis Results")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analysis_graph = GraphWidget()
        analysis_layout.addWidget(self.analysis_graph)
        
        right_panel.addWidget(analysis_group)
        
        # Statistics display
        stats_group = QGroupBox("Statistics Summary")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QLabel("No analysis performed yet")
        self.stats_text.setStyleSheet("color: #bbb; font-size: 10px; padding: 10px;")
        self.stats_text.setWordWrap(True)
        self.stats_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        stats_layout.addWidget(self.stats_text)
        
        right_panel.addWidget(stats_group)
        
        content_layout.addLayout(right_panel, 2)
        
        layout.addLayout(content_layout)
    
    def refresh_database_list(self):
        """Refresh the list of available database files from all profile subdirectories"""
        self.db_list.clear()
        
        if not os.path.exists(self.db_dir):
            self.info_label.setText("Database directory not found")
            return
        
        try:
            total_db_files = 0
            
            # Search in base directory and all subdirectories (profile folders)
            for root, dirs, files in os.walk(self.db_dir):
                db_files = [f for f in files if f.endswith('.db')]
                
                if db_files:
                    # Determine the profile name from the directory structure
                    rel_path = os.path.relpath(root, self.db_dir)
                    if rel_path == '.':
                        profile_prefix = "[No Profile] "
                    else:
                        profile_prefix = f"[{rel_path}] "
                    
                    for db_file in sorted(db_files):
                        # Get file info
                        file_path = os.path.join(root, db_file)
                        file_size = os.path.getsize(file_path)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        # Check if database has data
                        try:
                            conn = sqlite3.connect(file_path)
                            count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
                            conn.close()
                            
                            item_text = f"{profile_prefix}{db_file} ({count} readings, {mod_time.strftime('%Y-%m-%d %H:%M')})"
                            item = QListWidgetItem(item_text)
                            item.setData(Qt.ItemDataRole.UserRole, file_path)  # Store full path
                            self.db_list.addItem(item)
                            total_db_files += 1
                            
                        except Exception as e:
                            print(f"Error reading database {db_file}: {e}")
            
            if total_db_files == 0:
                self.info_label.setText("No database files found in any profile")
            else:
                self.info_label.setText(f"Found {total_db_files} database files across all profiles")
            
        except Exception as e:
            self.info_label.setText(f"Error loading database list: {str(e)}")
    
    def select_all_databases(self):
        """Select all database files"""
        for i in range(self.db_list.count()):
            self.db_list.item(i).setSelected(True)
    
    def clear_selection(self):
        """Clear all selections"""
        self.db_list.clearSelection()
    
    def get_selected_databases(self):
        """Get list of selected database file paths"""
        selected = []
        for item in self.db_list.selectedItems():
            selected.append(item.data(Qt.ItemDataRole.UserRole))
        return selected
    
    def compare_selected_runs(self):
        """Analyze selected database runs (single run or comparison)"""
        selected_dbs = self.get_selected_databases()
        
        if not selected_dbs:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Please select at least 1 database file to analyze")
            msg.setStyleSheet(self.parent().parent().styleSheet())  # Use main app styling
            msg.exec()
            return
        
        if len(selected_dbs) == 1:
            # Single run analysis
            self.analyze_single_run(selected_dbs[0])
        else:
            # Multi-run comparison
            self.plot_comparison(selected_dbs)
    
    def analyze_single_run(self, db_path):
        """Analyze a single database run with detailed statistics"""
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT * FROM readings ORDER BY ROWID ASC").fetchall()
            conn.close()
            
            if not rows:
                self.info_label.setText("Selected database contains no data")
                return
            
            timestamps = [datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") for r in rows]
            means = [r[1] for r in rows]
            maxs = [r[2] for r in rows]
            mins = [r[3] for r in rows]
            
            # Convert to minutes from start
            start_time = timestamps[0]
            time_minutes = [(t - start_time).total_seconds() / 60 for t in timestamps]
            
            # Apply smoothing
            smooth_means = smooth(means)
            smooth_maxs = smooth(maxs)
            smooth_mins = smooth(mins)
            
            # Adjust time for smoothed data
            adj_times = time_minutes[len(time_minutes)-len(smooth_means):]
            
            # Clear and plot
            self.analysis_graph.ax.clear()
            
            db_name = os.path.basename(db_path).replace('.db', '')
            
            # Plot all three temperature lines
            self.analysis_graph.ax.plot(adj_times, smooth_maxs, 
                                      label="Maximum Temperature", 
                                      color='#ff6b6b', linewidth=2.5)
            self.analysis_graph.ax.plot(adj_times, smooth_means, 
                                      label="Average Temperature", 
                                      color='#ffd93d', linewidth=2.5)
            self.analysis_graph.ax.plot(adj_times, smooth_mins, 
                                      label="Minimum Temperature", 
                                      color='#4ecdc4', linewidth=2.5)
            
            # Add trend lines if data is sufficient
            if len(adj_times) > 10:
                # Calculate trend for max temperatures
                z_max = np.polyfit(adj_times, smooth_maxs, 1)
                p_max = np.poly1d(z_max)
                self.analysis_graph.ax.plot(adj_times, p_max(adj_times), 
                                          "r--", alpha=0.5, linewidth=1, 
                                          label=f"Max Trend ({z_max[0]:.2f}°F/min)")
                
                # Calculate trend for min temperatures  
                z_min = np.polyfit(adj_times, smooth_mins, 1)
                p_min = np.poly1d(z_min)
                self.analysis_graph.ax.plot(adj_times, p_min(adj_times), 
                                          "c--", alpha=0.5, linewidth=1,
                                          label=f"Min Trend ({z_min[0]:.2f}°F/min)")
            
            # Styling
            self.analysis_graph.ax.set_xlabel("Time (minutes)", color='white')
            self.analysis_graph.ax.set_ylabel("Temperature (°F)", color='white')
            self.analysis_graph.ax.set_title(f"Single Run Analysis: {db_name}", 
                                           color='white', fontsize=14, fontweight='bold')
            self.analysis_graph.ax.grid(True, alpha=0.3, color='white')
            self.analysis_graph.ax.legend(loc="upper left", facecolor='#2b2b2b', edgecolor='white')
            
            # Style the graph
            self.analysis_graph.ax.tick_params(colors='white')
            for spine in self.analysis_graph.ax.spines.values():
                spine.set_color('white')
            
            self.analysis_graph.draw()
            
            # Generate detailed single-run statistics
            self.generate_single_run_stats(db_name, timestamps, means, maxs, mins, 
                                         smooth_means, smooth_maxs, smooth_mins, adj_times)
            
            self.info_label.setText(f"Analyzing single run: {db_name}")
            
        except Exception as e:
            print(f"Error analyzing single run {db_path}: {e}")
            self.info_label.setText(f"Error analyzing run: {str(e)}")
    
    def generate_single_run_stats(self, db_name, timestamps, means, maxs, mins, 
                                smooth_means, smooth_maxs, smooth_mins, adj_times):
        """Generate detailed statistics for a single run"""
        
        duration = (timestamps[-1] - timestamps[0]).total_seconds() / 60  # minutes
        
        # Temperature statistics
        overall_max = max(maxs)
        overall_min = min(mins)
        avg_max = np.mean(maxs)
        avg_min = np.mean(mins)
        avg_mean = np.mean(means)
        
        # Temperature ranges and variability
        temp_range = overall_max - overall_min
        max_std = np.std(maxs)
        min_std = np.std(mins)
        mean_std = np.std(means)
        
        # Rate of change analysis (using smoothed data)
        if len(adj_times) > 1:
            max_rate_changes = []
            min_rate_changes = []
            for i in range(1, len(smooth_maxs)):
                time_diff = adj_times[i] - adj_times[i-1]
                if time_diff > 0:
                    max_rate_changes.append((smooth_maxs[i] - smooth_maxs[i-1]) / time_diff)
                    min_rate_changes.append((smooth_mins[i] - smooth_mins[i-1]) / time_diff)
            
            max_heating_rate = max(max_rate_changes) if max_rate_changes else 0
            max_cooling_rate = min(max_rate_changes) if max_rate_changes else 0
            avg_max_rate = np.mean(max_rate_changes) if max_rate_changes else 0
            avg_min_rate = np.mean(min_rate_changes) if min_rate_changes else 0
        else:
            max_heating_rate = max_cooling_rate = avg_max_rate = avg_min_rate = 0
        
        # Time-based analysis
        peak_max_time = adj_times[smooth_maxs.index(max(smooth_maxs))] if smooth_maxs else 0
        peak_min_time = adj_times[smooth_mins.index(max(smooth_mins))] if smooth_mins else 0
        
        # Stability analysis (coefficient of variation)
        max_cv = (max_std / avg_max * 100) if avg_max > 0 else 0
        min_cv = (min_std / avg_min * 100) if avg_min > 0 else 0
        
        stats_text = f"SINGLE RUN ANALYSIS: {db_name}\n\n"
        
        stats_text += "=== BASIC STATISTICS ===\n"
        stats_text += f"Duration: {duration:.1f} minutes\n"
        stats_text += f"Total Readings: {len(timestamps)}\n"
        stats_text += f"Recording Interval: ~{duration*60/len(timestamps):.1f} seconds\n\n"
        
        stats_text += "=== TEMPERATURE SUMMARY ===\n"
        stats_text += f"Peak Maximum: {overall_max:.1f}°F\n"
        stats_text += f"Lowest Minimum: {overall_min:.1f}°F\n"
        stats_text += f"Overall Range: {temp_range:.1f}°F\n"
        stats_text += f"Average Maximum: {avg_max:.1f}°F\n"
        stats_text += f"Average Minimum: {avg_min:.1f}°F\n"
        stats_text += f"Average Mean: {avg_mean:.1f}°F\n\n"
        
        stats_text += "=== VARIABILITY ===\n"
        stats_text += f"Max Temp Std Dev: {max_std:.2f}°F\n"
        stats_text += f"Min Temp Std Dev: {min_std:.2f}°F\n"
        stats_text += f"Mean Temp Std Dev: {mean_std:.2f}°F\n"
        stats_text += f"Max Stability (CV): {max_cv:.1f}%\n"
        stats_text += f"Min Stability (CV): {min_cv:.1f}%\n\n"
        
        stats_text += "=== RATE OF CHANGE ===\n"
        stats_text += f"Max Heating Rate: {max_heating_rate:.2f}°F/min\n"
        stats_text += f"Max Cooling Rate: {max_cooling_rate:.2f}°F/min\n"
        stats_text += f"Avg Max Change Rate: {avg_max_rate:.2f}°F/min\n"
        stats_text += f"Avg Min Change Rate: {avg_min_rate:.2f}°F/min\n\n"
        
        stats_text += "=== TIMING ===\n"
        stats_text += f"Start Time: {timestamps[0].strftime('%Y-%m-%d %H:%M:%S')}\n"
        stats_text += f"End Time: {timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
        stats_text += f"Peak Max at: {peak_max_time:.1f} minutes\n"
        stats_text += f"Peak Min at: {peak_min_time:.1f} minutes\n\n"
        
        # Performance indicators
        if max_heating_rate > 5:
            stats_text += "⚠️ Rapid heating detected\n"
        if max_cooling_rate < -5:
            stats_text += "⚠️ Rapid cooling detected\n"
        if max_cv > 10:
            stats_text += "⚠️ High temperature variability\n"
        if temp_range > 100:
            stats_text += "📊 Wide temperature range\n"
        
        self.stats_text.setText(stats_text)
    
    def plot_comparison(self, db_paths):
        """Plot comparison of multiple database files"""
        self.analysis_graph.ax.clear()
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8', '#f7dc6f']
        
        all_data = []
        
        for i, db_path in enumerate(db_paths):
            try:
                conn = sqlite3.connect(db_path)
                rows = conn.execute("SELECT * FROM readings ORDER BY ROWID ASC").fetchall()
                conn.close()
                
                if not rows:
                    continue
                
                timestamps = [datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") for r in rows]
                maxs = [r[2] for r in rows]
                mins = [r[3] for r in rows]
                
                # Normalize timestamps to start from 0
                if timestamps:
                    start_time = timestamps[0]
                    normalized_times = [(t - start_time).total_seconds() / 60 for t in timestamps]  # Convert to minutes
                    
                    smooth_max = smooth(maxs)
                    smooth_min = smooth(mins)
                    
                    # Adjust timestamps for smoothed data
                    adj_times = normalized_times[len(normalized_times)-len(smooth_max):]
                    
                    db_name = os.path.basename(db_path).replace('.db', '')
                    color = colors[i % len(colors)]
                    
                    self.analysis_graph.ax.plot(adj_times, smooth_max, 
                                              label=f"{db_name} (Max)", 
                                              color=color, linewidth=2)
                    self.analysis_graph.ax.plot(adj_times, smooth_min, 
                                              label=f"{db_name} (Min)", 
                                              color=color, linewidth=2, 
                                              linestyle='--', alpha=0.7)
                    
                    # Store data for statistics
                    all_data.append({
                        'name': db_name,
                        'max_temps': smooth_max,
                        'min_temps': smooth_min,
                        'times': adj_times,
                        'duration': max(adj_times) if adj_times else 0
                    })
                    
            except Exception as e:
                print(f"Error processing {db_path}: {e}")
        
        if all_data:
            self.analysis_graph.ax.set_xlabel("Time (minutes)", color='white')
            self.analysis_graph.ax.set_ylabel("Temperature (°F)", color='white')
            self.analysis_graph.ax.set_title("Multi-Run Temperature Comparison", color='white', fontsize=14, fontweight='bold')
            self.analysis_graph.ax.grid(True, alpha=0.3, color='white')
            self.analysis_graph.ax.legend(loc="upper left", facecolor='#2b2b2b', edgecolor='white')
            
            # Style the graph
            self.analysis_graph.ax.tick_params(colors='white')
            for spine in self.analysis_graph.ax.spines.values():
                spine.set_color('white')
            
            self.analysis_graph.draw()
            
            # Update statistics for multi-run comparison
            self.update_comparison_statistics(all_data)
            self.info_label.setText(f"Comparing {len(all_data)} runs")
    
    def update_comparison_statistics(self, data):
        """Update statistics display for multi-run comparison"""
        if not data:
            return
        
        stats_text = "MULTI-RUN COMPARISON STATISTICS:\n\n"
        
        for run in data:
            stats_text += f"Run: {run['name']}\n"
            stats_text += f"  Duration: {run['duration']:.1f} minutes\n"
            stats_text += f"  Max Temp: {max(run['max_temps']):.1f}°F\n"
            stats_text += f"  Min Temp: {min(run['min_temps']):.1f}°F\n"
            stats_text += f"  Avg Max: {np.mean(run['max_temps']):.1f}°F\n"
            stats_text += f"  Avg Min: {np.mean(run['min_temps']):.1f}°F\n\n"
        
        # Overall statistics
        all_maxs = [temp for run in data for temp in run['max_temps']]
        all_mins = [temp for run in data for temp in run['min_temps']]
        
        stats_text += "OVERALL COMPARISON:\n"
        stats_text += f"  Highest Temperature: {max(all_maxs):.1f}°F\n"
        stats_text += f"  Lowest Temperature: {min(all_mins):.1f}°F\n"
        stats_text += f"  Average Maximum: {np.mean(all_maxs):.1f}°F\n"
        stats_text += f"  Average Minimum: {np.mean(all_mins):.1f}°F\n"
        stats_text += f"  Temperature Range: {max(all_maxs) - min(all_mins):.1f}°F\n\n"
        
        # Run-to-run comparison insights
        max_temps_by_run = [max(run['max_temps']) for run in data]
        min_temps_by_run = [min(run['min_temps']) for run in data]
        durations = [run['duration'] for run in data]
        
        stats_text += "COMPARISON INSIGHTS:\n"
        stats_text += f"  Most consistent max temps: {min(max_temps_by_run):.1f}°F - {max(max_temps_by_run):.1f}°F\n"
        stats_text += f"  Duration range: {min(durations):.1f} - {max(durations):.1f} minutes\n"
        stats_text += f"  Longest run: {data[durations.index(max(durations))]['name']}\n"
        stats_text += f"  Shortest run: {data[durations.index(min(durations))]['name']}\n"
        
        if max_temps_by_run:
            hottest_run_idx = max_temps_by_run.index(max(max_temps_by_run))
            coolest_run_idx = min_temps_by_run.index(min(min_temps_by_run))
            stats_text += f"  Hottest run: {data[hottest_run_idx]['name']} ({max(max_temps_by_run):.1f}°F)\n"
            stats_text += f"  Coolest run: {data[coolest_run_idx]['name']} ({min(min_temps_by_run):.1f}°F)\n"
        
        self.stats_text.setText(stats_text)
    
    def generate_statistics(self):
        """Generate detailed statistics report"""
        selected_dbs = self.get_selected_databases()
        
        if not selected_dbs:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Please select at least 1 database file to analyze")
            msg.setStyleSheet(self.parent().parent().styleSheet())
            msg.exec()
            return
        
        try:
            import pandas as pd
            
            # Create comprehensive statistics report
            report_data = []
            
            for db_path in selected_dbs:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query("SELECT * FROM readings", conn)
                conn.close()
                
                if df.empty:
                    continue
                
                db_name = os.path.basename(db_path).replace('.db', '')
                
                # Calculate statistics
                stats = {
                    'Run_Name': db_name,
                    'Total_Readings': len(df),
                    'Duration_Minutes': len(df) * 10 / 60,  # Assuming 10-second intervals
                    'Max_Temperature': df['temp_f_max'].max(),
                    'Min_Temperature': df['temp_f_min'].min(),
                    'Avg_Max_Temperature': df['temp_f_max'].mean(),
                    'Avg_Min_Temperature': df['temp_f_min'].mean(),
                    'Avg_Mean_Temperature': df['temp_f_mean'].mean(),
                    'Temperature_Range': df['temp_f_max'].max() - df['temp_f_min'].min(),
                    'Max_Temp_StdDev': df['temp_f_max'].std(),
                    'Min_Temp_StdDev': df['temp_f_min'].std(),
                    'Start_Time': df['timestamp'].iloc[0],
                    'End_Time': df['timestamp'].iloc[-1]
                }
                
                report_data.append(stats)
            
            if report_data:
                # Create DataFrame and save to CSV
                report_df = pd.DataFrame(report_data)
                
                # Save report
                csv_dir = os.path.join(os.getcwd(), "LD1_Cycle_CSV")
                os.makedirs(csv_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                report_path = os.path.join(csv_dir, f"Analysis_Report_{timestamp}.csv")
                
                report_df.to_csv(report_path, index=False)
                
                # Show success message
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("Success")
                msg.setText(f"Statistics report generated successfully!\n\nSaved to: {report_path}")
                msg.setStyleSheet(self.parent().parent().styleSheet())
                msg.exec()
                
                self.info_label.setText(f"Report saved: {os.path.basename(report_path)}")
            
        except Exception as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to generate report: {str(e)}")
            msg.setStyleSheet(self.parent().parent().styleSheet())
            msg.exec()
    
    def export_combined_csv(self):
        """Export all selected databases to a single combined CSV file"""
        selected_dbs = self.get_selected_databases()
        
        if not selected_dbs:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Please select at least 1 database file to export")
            msg.setStyleSheet(self.parent().parent().styleSheet())
            msg.exec()
            return
        
        try:
            import pandas as pd
            
            combined_data = []
            
            for db_path in selected_dbs:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query("SELECT * FROM readings", conn)
                conn.close()
                
                if df.empty:
                    continue
                
                # Add run identifier and profile information
                db_name = os.path.basename(db_path).replace('.db', '')
                df['run_name'] = db_name
                df['database_file'] = os.path.basename(db_path)
                
                # Determine profile name from path
                base_db_dir = os.path.join(os.getcwd(), "LD1_Cycle_Database")
                rel_path = os.path.relpath(os.path.dirname(db_path), base_db_dir)
                if rel_path == '.':
                    profile_name = "No_Profile"
                else:
                    profile_name = rel_path
                df['test_profile'] = profile_name
                
                combined_data.append(df)
            
            if combined_data:
                # Combine all data
                combined_df = pd.concat(combined_data, ignore_index=True)
                
                # Save combined CSV to a special subdirectory for combined analysis
                base_csv_dir = os.path.join(os.getcwd(), "LD1_Cycle_CSV")
                csv_dir = os.path.join(base_csv_dir, "Combined_Analysis")
                os.makedirs(csv_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                combined_path = os.path.join(csv_dir, f"Combined_Analysis_{timestamp}.csv")
                
                combined_df.to_csv(combined_path, index=False)
                
                # Show success message
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("Success")
                msg.setText(f"Combined CSV exported successfully!\n\nSaved to: {combined_path}\n\nTotal records: {len(combined_df)}")
                msg.setStyleSheet(self.parent().parent().styleSheet())
                msg.exec()
                
                self.info_label.setText(f"Combined export saved: {os.path.basename(combined_path)}")
            
        except Exception as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to export combined CSV: {str(e)}")
            msg.setStyleSheet(self.parent().parent().styleSheet())
            msg.exec()

class ModernThermalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLIR Thermal Camera - Modern Interface")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #aaa;
            }
            QLabel {
                color: white;
                font-size: 11px;
            }
            QLineEdit {
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #444;
                padding: 8px;
                border-radius: 4px;
                font-size: 11px;
                selection-background-color: #0d7377;
                selection-color: white;
            }
            QLineEdit:focus {
                border-color: #0d7377;
            }
            QLineEdit:hover {
                border-color: #14a085;
            }
            QComboBox {
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #444;
                padding: 8px;
                border-radius: 4px;
                font-size: 11px;
                selection-background-color: #0d7377;
                selection-color: white;
            }
            QComboBox:hover {
                border-color: #0d7377;
            }
            QComboBox:focus {
                border-color: #0d7377;
            }
            QComboBox::drop-down {
                background-color: #444;
                border: none;
                border-radius: 3px;
            }
            QComboBox::down-arrow {
                color: white;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #444;
                selection-background-color: #0d7377;
                selection-color: white;
            }
            QComboBox QAbstractItemView::item {
                color: white;
                background-color: transparent;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0d7377;
                color: white;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #14a085;
                color: white;
            }
            QCheckBox {
                color: white;
                font-size: 11px;
            }
            QCheckBox::indicator {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 3px;
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:checked {
                background-color: #0d7377;
                border-color: #0d7377;
            }
            QCheckBox::indicator:hover {
                border-color: #14a085;
            }
            QCheckBox:focus {
                outline: none;
            }
            QTextEdit {
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #444;
                padding: 8px;
                border-radius: 4px;
                font-size: 11px;
                selection-background-color: #0d7377;
                selection-color: white;
            }
            QTextEdit:focus {
                border-color: #0d7377;
            }
            QTextEdit:hover {
                border-color: #14a085;
            }
            QGroupBox {
                color: white;
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QTabWidget::pane {
                border: 2px solid #444;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                color: white;
                padding: 10px 20px;
                margin-right: 2px;
                border-top: 2px solid #444;
                border-left: 2px solid #444;
                border-right: 2px solid #444;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
                border-top: 2px solid #0d7377;
            }
            QTabBar::tab:hover {
                background-color: #14a085;
            }
        """)
        
        # Initialize variables
        self.camera = None
        self.camera_thread = None
        self.is_running = False
        self.recording = False
        self.paused = False
        self.current_db = None
        self.last_log_time = 0
        self.recording_started = False
        self.current_temp_data = None
        self.current_test_profile = None
        self.profile_variable_values = {}
        self.recording_profile_variables = {}  # Store profile variables for current recording session
        
        # Initialize PLC connection
        self.plc_heating = PLCHeatingProfile("192.168.1.100")  # Template IP
        self.heating_profile_data = None
        
        self.setup_ui()
        self.setup_camera()
        
        # Initialize test profiles
        self.refresh_test_profiles()
        
        # Initialize last log display
        self.update_last_log_display()
        
        # Timer for graph updates
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graph)
        self.graph_timer.start(1000)  # Update every second
        
        # Timer for PLC updates (check for changes every 30 seconds during recording)
        self.plc_timer = QTimer()
        self.plc_timer.timeout.connect(self.periodic_plc_update)
        self.plc_timer.start(30000)  # Update every 30 seconds
        
        # Add keyboard shortcut for quick exit
        exit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        exit_shortcut.activated.connect(self.close)

    def get_safe_profile_name(self, profile_name):
        """Convert profile name to safe folder name"""
        if not profile_name:
            return "No_Profile"
        
        # Create safe folder name
        safe_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        return safe_name if safe_name else "Unknown_Profile"

    def get_last_log_name(self, profile_name=None):
        """Get the most recent log name used for the current or specified profile"""
        try:
            base_db_dir = os.path.join(os.getcwd(), "LD1_Cycle_Database")
            
            if profile_name:
                safe_profile_name = self.get_safe_profile_name(profile_name)
                db_dir = os.path.join(base_db_dir, safe_profile_name)
            elif self.current_test_profile:
                profile_name = self.current_test_profile.get('name', 'Unknown_Profile')
                safe_profile_name = self.get_safe_profile_name(profile_name)
                db_dir = os.path.join(base_db_dir, safe_profile_name)
            else:
                db_dir = os.path.join(base_db_dir, "No_Profile")
            
            if not os.path.exists(db_dir):
                return None
            
            # Get all .db files in the profile directory
            db_files = [f for f in os.listdir(db_dir) if f.endswith('.db')]
            if not db_files:
                return None
            
            # Sort by modification time (most recent first)
            db_files.sort(key=lambda f: os.path.getmtime(os.path.join(db_dir, f)), reverse=True)
            
            # Get the most recent file name without extension
            most_recent = db_files[0].replace('.db', '')
            
            # If it starts with the default timestamp format, return "Auto-generated"
            if most_recent.startswith('ir_log_tempf_'):
                return "Auto-generated timestamp"
            else:
                return most_recent
                
        except Exception as e:
            print(f"Error getting last log name: {e}")
            return None

    def update_last_log_display(self):
        """Update the last log name display"""
        last_log = self.get_last_log_name()
        if last_log:
            self.last_log_label.setText(f"Last log: {last_log}")
        else:
            self.last_log_label.setText("Last log: None")

    def create_styled_message_box(self, icon, title, text, informative_text="", buttons=None, default_button=None):
        """Create a consistently styled message box for the dark theme"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        if informative_text:
            msg_box.setInformativeText(informative_text)
        
        # Apply dark theme styling
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2b2b2b;
                color: white;
                font-size: 11px;
            }
            QMessageBox QLabel {
                color: white;
                background-color: transparent;
            }
            QMessageBox QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #14a085;
            }
            QMessageBox QPushButton:pressed {
                background-color: #0a5d61;
            }
        """)
        
        if buttons:
            msg_box.setStandardButtons(buttons)
        if default_button:
            msg_box.setDefaultButton(default_button)
            
        return msg_box

    def show_critical(self, title, text, informative_text=""):
        """Show a styled critical message box"""
        msg_box = self.create_styled_message_box(
            QMessageBox.Icon.Critical, title, text, informative_text,
            QMessageBox.StandardButton.Ok
        )
        return msg_box.exec()

    def show_warning(self, title, text, informative_text=""):
        """Show a styled warning message box"""
        msg_box = self.create_styled_message_box(
            QMessageBox.Icon.Warning, title, text, informative_text,
            QMessageBox.StandardButton.Ok
        )
        return msg_box.exec()

    def show_information(self, title, text, informative_text=""):
        """Show a styled information message box"""
        msg_box = self.create_styled_message_box(
            QMessageBox.Icon.Information, title, text, informative_text,
            QMessageBox.StandardButton.Ok
        )
        return msg_box.exec()

    def show_question(self, title, text, informative_text="", buttons=None):
        """Show a styled question message box"""
        if buttons is None:
            buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        msg_box = self.create_styled_message_box(
            QMessageBox.Icon.Question, title, text, informative_text,
            buttons, QMessageBox.StandardButton.No
        )
        return msg_box.exec()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(self.tab_widget)
        
        # Create Live Recording tab
        self.setup_live_recording_tab()
        
        # Create Test Profile tab
        self.setup_test_profile_tab()
        
        # Create Data Analysis tab
        self.setup_data_analysis_tab()
    
    def setup_live_recording_tab(self):
        """Setup the live recording tab"""
        live_tab = QWidget()
        self.tab_widget.addTab(live_tab, "Live Recording")
        
        # Main layout for live tab
        main_layout = QHBoxLayout(live_tab)
        
        # Left panel for camera and controls
        left_panel = QVBoxLayout()
        
        # Camera widget
        self.thermal_widget = ThermalWidget()
        self.thermal_widget.roi_updated.connect(self.on_roi_updated)
        left_panel.addWidget(self.thermal_widget)
        
        # Temperature display
        temp_group = QGroupBox("Current Temperature")
        temp_layout = QVBoxLayout(temp_group)
        self.temp_label = QLabel("ROI Temperature: -- °F")
        self.temp_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4ecdc4;")
        self.temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        temp_layout.addWidget(self.temp_label)
        left_panel.addWidget(temp_group)
        
        # Control panel
        control_group = QGroupBox("Camera Controls")
        control_layout = QGridLayout(control_group)
        
        # ROI controls
        self.clear_roi_btn = QPushButton("Clear ROI")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        control_layout.addWidget(self.clear_roi_btn, 0, 0)
        
        self.finish_roi_btn = QPushButton("Finish ROI")
        self.finish_roi_btn.clicked.connect(self.finish_roi)
        control_layout.addWidget(self.finish_roi_btn, 0, 1)
        
        # Rotation control
        self.rotate_btn = QPushButton("Rotate 90°")
        self.rotate_btn.clicked.connect(self.rotate_view)
        control_layout.addWidget(self.rotate_btn, 1, 0)
        
        # Recording controls
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        control_layout.addWidget(self.record_btn, 1, 1)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn, 2, 0)
        
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn, 2, 1)
        
        # Comparison controls
        self.compare_btn = QPushButton("Compare Run")
        self.compare_btn.clicked.connect(self.select_comparison_db)
        control_layout.addWidget(self.compare_btn, 3, 0)
        
        self.clear_compare_btn = QPushButton("Clear Comparison")
        self.clear_compare_btn.clicked.connect(self.clear_comparison)
        self.clear_compare_btn.setEnabled(False)
        control_layout.addWidget(self.clear_compare_btn, 3, 1)
        
        left_panel.addWidget(control_group)
        
        # PLC Controls
        plc_group = QGroupBox("PLC Heating Profile")
        plc_layout = QGridLayout(plc_group)
        
        # PLC IP Address input
        plc_ip_layout = QHBoxLayout()
        plc_ip_layout.addWidget(QLabel("PLC IP:"))
        self.plc_ip_input = QLineEdit()
        self.plc_ip_input.setText("192.168.1.100")
        self.plc_ip_input.setPlaceholderText("Enter PLC IP address")
        plc_ip_layout.addWidget(self.plc_ip_input)
        plc_layout.addLayout(plc_ip_layout, 0, 0, 1, 2)
        
        # PLC connection controls
        self.plc_connect_btn = QPushButton("Connect PLC")
        self.plc_connect_btn.clicked.connect(self.connect_plc)
        plc_layout.addWidget(self.plc_connect_btn, 1, 0)
        
        self.plc_read_btn = QPushButton("Read Profile")
        self.plc_read_btn.clicked.connect(self.read_plc_profile)
        self.plc_read_btn.setEnabled(False)
        plc_layout.addWidget(self.plc_read_btn, 1, 1)
        
        # PLC status
        self.plc_status_label = QLabel("PLC Status: Disconnected")
        self.plc_status_label.setStyleSheet("color: #ffd93d; font-size: 10px;")
        plc_layout.addWidget(self.plc_status_label, 2, 0, 1, 2)
        
        left_panel.addWidget(plc_group)
        
        # Log naming section and test profile selection
        log_group = QGroupBox("Recording Settings")
        log_layout = QVBoxLayout(log_group)
        
        # Test profile selection
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Test Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("No Profile Selected", None)
        self.profile_combo.currentTextChanged.connect(self.on_test_profile_changed)
        profile_layout.addWidget(self.profile_combo)
        
        refresh_profiles_btn = QPushButton("↻")
        refresh_profiles_btn.setMaximumWidth(30)
        refresh_profiles_btn.setToolTip("Refresh test profiles")
        refresh_profiles_btn.clicked.connect(self.refresh_test_profiles)
        profile_layout.addWidget(refresh_profiles_btn)
        
        log_layout.addLayout(profile_layout)
        
        # Log name input
        log_name_layout = QHBoxLayout()
        log_name_layout.addWidget(QLabel("Log Name:"))
        self.log_name_input = QLineEdit()
        self.log_name_input.setPlaceholderText("Enter custom log name (optional)")
        log_name_layout.addWidget(self.log_name_input)
        log_layout.addLayout(log_name_layout)
        
        # Last log name display
        self.last_log_label = QLabel("Last log: None")
        self.last_log_label.setStyleSheet("color: #bbb; font-size: 10px; font-style: italic; margin-left: 5px;")
        self.last_log_label.setWordWrap(True)
        log_layout.addWidget(self.last_log_label)
        
        left_panel.addWidget(log_group)
        
        # Status label
        self.status_label = QLabel("Status: Camera initializing...")
        self.status_label.setStyleSheet("color: #ffd93d; font-weight: bold;")
        left_panel.addWidget(self.status_label)
        
        # Right panel for graph
        right_panel = QVBoxLayout()
        
        graph_group = QGroupBox("Temperature History")
        graph_layout = QVBoxLayout(graph_group)
        self.graph_widget = GraphWidget()
        graph_layout.addWidget(self.graph_widget)
        right_panel.addWidget(graph_group)
        
        # Profile variables container (moved from left panel)
        self.profile_variables_group = QGroupBox("Test Variables")
        self.profile_variables_layout = QVBoxLayout(self.profile_variables_group)
        self.profile_variables_group.setVisible(False)
        self.profile_variable_inputs = {}
        right_panel.addWidget(self.profile_variables_group)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_text = QLabel(
            "1. Click on camera image to add ROI vertices\n"
            "2. Click 'Finish ROI' when you have at least 3 points\n"
            "3. Select test profile and fill variables (if using)\n"
            "4. Enter custom log name (optional)\n"
            "5. Click 'Start Recording' to begin data logging\n"
            "6. Use 'Pause' to temporarily stop logging\n"
            "7. Click 'Export CSV' to save recorded data"
        )
        instructions_text.setWordWrap(True)
        instructions_text.setStyleSheet("color: #bbb; font-size: 10px;")
        instructions_layout.addWidget(instructions_text)
        right_panel.addWidget(instructions_group)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)  # 2/3 width for left panel
        main_layout.addLayout(right_panel, 1)  # 1/3 width for right panel
    
    def setup_test_profile_tab(self):
        """Setup the test profile management tab"""
        self.test_profile_tab = TestProfileTab()
        self.tab_widget.addTab(self.test_profile_tab, "Test Profiles")
    
    def setup_data_analysis_tab(self):
        """Setup the data analysis tab"""
        self.analysis_tab = DataAnalysisTab()
        self.tab_widget.addTab(self.analysis_tab, "Data Analysis")

    def refresh_test_profiles(self):
        """Refresh the test profile dropdown"""
        self.profile_combo.clear()
        self.profile_combo.addItem("No Profile Selected", None)
        
        profiles_dir = os.path.join(os.getcwd(), "LD1_Test_Profiles")
        if not os.path.exists(profiles_dir):
            return
        
        try:
            profile_files = [f for f in os.listdir(profiles_dir) if f.endswith('.json')]
            for profile_file in sorted(profile_files):
                profile_path = os.path.join(profiles_dir, profile_file)
                try:
                    with open(profile_path, 'r') as f:
                        profile_data = json.load(f)
                    
                    profile_name = profile_data.get('name', 'Unknown')
                    self.profile_combo.addItem(profile_name, profile_path)
                    
                except Exception as e:
                    print(f"Error reading profile {profile_file}: {e}")
                    
        except Exception as e:
            print(f"Error loading test profiles: {e}")
    
    def on_test_profile_changed(self):
        """Handle test profile selection change"""
        current_data = self.profile_combo.currentData()
        
        if current_data is None:
            # No profile selected
            self.current_test_profile = None
            self.profile_variables_group.setVisible(False)
            self.profile_variable_inputs.clear()
            self.update_last_log_display()
            return
        
        # Load selected profile
        try:
            with open(current_data, 'r') as f:
                profile_data = json.load(f)
            
            self.current_test_profile = profile_data
            self.setup_profile_variables()
            self.update_last_log_display()
            
        except Exception as e:
            print(f"Error loading test profile: {e}")
            self.current_test_profile = None
            self.profile_variables_group.setVisible(False)
            self.update_last_log_display()
    
    def setup_profile_variables(self):
        """Setup input fields for profile variables"""
        # Clear existing inputs
        for i in reversed(range(self.profile_variables_layout.count())):
            child = self.profile_variables_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.profile_variable_inputs.clear()
        
        if not self.current_test_profile or not self.current_test_profile.get('variables'):
            self.profile_variables_group.setVisible(False)
            return
        
        # Create input fields for each variable
        variables = self.current_test_profile.get('variables', [])
        
        for var in variables:
            var_name = var['name']
            var_type = var['type']
            default_value = var.get('default_value', '')
            description = var.get('description', '')
            is_end_of_cycle = var.get('end_of_cycle', False)
            
            # Skip end-of-cycle variables - they'll be prompted for at the end
            if is_end_of_cycle:
                continue
            
            # Create input widget based on type
            if var_type == "Boolean":
                input_widget = QCheckBox()
                if default_value.lower() in ['true', '1', 'yes']:
                    input_widget.setChecked(True)
            elif var_type == "Date":
                input_widget = QLineEdit()
                input_widget.setPlaceholderText("YYYY-MM-DD or leave blank for today")
                if default_value:
                    input_widget.setText(default_value)
            elif var_type == "Number":
                input_widget = QLineEdit()
                input_widget.setPlaceholderText("Enter numeric value")
                if default_value:
                    input_widget.setText(str(default_value))
            else:  # String
                input_widget = QLineEdit()
                input_widget.setPlaceholderText("Enter text value")
                if default_value:
                    input_widget.setText(str(default_value))
            
            # Add label and input to layout
            label_text = f"{var_name} ({var_type}):"
            if description:
                label_text += f"\n{description}"
            
            label = QLabel(label_text)
            label.setWordWrap(True)
            self.profile_variables_layout.addWidget(label)
            self.profile_variables_layout.addWidget(input_widget)
            
            self.profile_variable_inputs[var_name] = {
                'widget': input_widget,
                'type': var_type
            }
        
        self.profile_variables_group.setVisible(True)
    
    def get_profile_variable_values(self):
        """Get current values from profile variable inputs"""
        values = {}
        
        for var_name, input_info in self.profile_variable_inputs.items():
            widget = input_info['widget']
            var_type = input_info['type']
            
            if var_type == "Boolean":
                values[var_name] = widget.isChecked()
            elif var_type == "Date":
                date_text = widget.text().strip()
                if not date_text:
                    values[var_name] = datetime.now().strftime("%Y-%m-%d")
                else:
                    values[var_name] = date_text
            elif var_type == "Number":
                try:
                    num_text = widget.text().strip()
                    values[var_name] = float(num_text) if num_text else 0.0
                except ValueError:
                    values[var_name] = 0.0
            else:  # String
                values[var_name] = widget.text().strip()
        
        return values

    def setup_camera(self):
        """Initialize camera in a separate thread"""
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        
        # Connect signals
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self.update_camera_frame)
        self.camera_worker.status_update.connect(self.update_status)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        
        self.camera_thread.start()

    def update_camera_frame(self, frame):
        """Update camera display with new frame"""
        self.thermal_widget.update_frame(frame)
        
        # Calculate temperature if ROI is ready
        if self.thermal_widget.polygon_ready and len(self.thermal_widget.polygon_pts) >= 3:
            self.calculate_roi_temperature(frame)

    def calculate_roi_temperature(self, frame):
        """Calculate temperature statistics for the current ROI"""
        try:
            # Create mask for ROI
            mask = np.zeros(frame.shape, dtype=np.uint8)
            pts = np.array(self.thermal_widget.polygon_pts, np.int32)
            cv2.fillPoly(mask, [pts], 1)
            
            # Get ROI values
            roi_vals = frame[mask == 1]
            if roi_vals.size > 0:
                mean_raw = np.mean(roi_vals)
                max_raw = np.max(roi_vals)
                min_raw = np.min(roi_vals)
                
                # Convert to Fahrenheit
                T_F_mean = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(mean_raw)))
                T_F_max = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(max_raw)))
                T_F_min = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(min_raw)))
                
                # Update display
                self.temp_label.setText(f"ROI Temperature: {T_F_mean:.1f} °F")
                
                # Store current data
                self.current_temp_data = (T_F_mean, T_F_max, T_F_min)
                
                # Log to database if recording
                if self.recording and not self.paused and time.time() - self.last_log_time >= 10:
                    self.log_temperature_data()
                    
        except Exception as e:
            print(f"Error calculating ROI temperature: {e}")

    def log_temperature_data(self):
        """Log current temperature data to database"""
        if not self.current_db or not self.current_temp_data:
            return
            
        try:
            conn = sqlite3.connect(self.current_db)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Only store temperature data in readings table
            sql = "INSERT INTO readings (timestamp, temp_f_mean, temp_f_max, temp_f_min) VALUES (?, ?, ?, ?)"
            values = [ts, self.current_temp_data[0], self.current_temp_data[1], self.current_temp_data[2]]
            
            conn.execute(sql, values)
            conn.commit()
            conn.close()
            self.last_log_time = time.time()
            self.recording_started = True
        except Exception as e:
            print(f"Error logging data: {e}")

    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(f"Status: {message}")

    def update_graph(self):
        """Update temperature graph"""
        if self.recording and self.current_db:
            self.graph_widget.update_graph(self.current_db)

    def on_roi_updated(self):
        """Handle ROI updates"""
        if self.thermal_widget.polygon_ready:
            self.record_btn.setEnabled(True)
            self.status_label.setText("Status: ROI ready - you can start recording")
            self.status_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
        else:
            self.record_btn.setEnabled(False)
            if len(self.thermal_widget.polygon_pts) > 0:
                self.status_label.setText(f"Status: ROI in progress ({len(self.thermal_widget.polygon_pts)} points)")
                self.status_label.setStyleSheet("color: #ffd93d; font-weight: bold;")

    def clear_roi(self):
        """Clear current ROI"""
        self.thermal_widget.clear_roi()
        self.temp_label.setText("ROI Temperature: -- °F")
        self.record_btn.setEnabled(False)
        self.status_label.setText("Status: ROI cleared - click on image to start new ROI")
        self.status_label.setStyleSheet("color: #ffd93d; font-weight: bold;")

    def finish_roi(self):
        """Finish current ROI"""
        if self.thermal_widget.finish_roi():
            self.status_label.setText("Status: ROI completed - ready to record")
            self.status_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
            self.record_btn.setEnabled(True)
        else:
            msg_box = self.create_styled_message_box(
                QMessageBox.Icon.Warning,
                "Warning",
                "Need at least 3 points to finish ROI"
            )
            msg_box.exec()

    def rotate_view(self):
        """Rotate camera view"""
        angle = self.thermal_widget.rotate_view()
        self.status_label.setText(f"Status: View rotated to {angle}°")

    def connect_plc(self):
        """Connect/disconnect to PLC with user-specified IP address"""
        if self.plc_heating.connected:
            # Disconnect
            self.plc_heating.disconnect()
            self.plc_status_label.setText("PLC Status: Disconnected")
            self.plc_status_label.setStyleSheet("color: #ffd93d; font-size: 10px;")
            self.plc_connect_btn.setText("Connect PLC")
            self.plc_read_btn.setEnabled(False)
            return
            
        ip_address = self.plc_ip_input.text().strip()
        if not ip_address:
            self.show_warning("Warning", "Please enter a PLC IP address")
            return
            
        # Update PLC IP address
        self.plc_heating.ip_address = ip_address
        
        # Attempt connection
        if self.plc_heating.connect():
            self.plc_status_label.setText("PLC Status: Connected")
            self.plc_status_label.setStyleSheet("color: #4ecdc4; font-size: 10px;")
            self.plc_connect_btn.setText("Disconnect PLC")
            self.plc_read_btn.setEnabled(True)
            self.show_information("Success", f"Connected to PLC at {ip_address}")
        else:
            self.plc_status_label.setText("PLC Status: Connection Failed")
            self.plc_status_label.setStyleSheet("color: #e74c3c; font-size: 10px;")
            self.show_warning("Connection Failed", 
                            f"Could not connect to PLC at {ip_address}.\n"
                            "Recording will continue without PLC data.")

    def read_plc_profile(self):
        """Manually read PLC heating profile"""
        profile_data = self.plc_heating.read_heating_profile()
        
        if profile_data:
            self.heating_profile_data = profile_data
            steps_count = len([s for s in profile_data['steps'] 
                             if s.get('time_sp') is not None])
            self.show_information("Success", 
                                f"Read heating profile with {steps_count} valid steps")
            
            # Update graph if available
            if hasattr(self, 'graph_widget'):
                self.graph_widget.set_heating_profile(profile_data)
        else:
            self.show_warning("Read Failed", 
                            "Could not read heating profile from PLC.\n"
                            "Check PLC connection and tag names.")

    def toggle_recording(self):
        """Start/stop recording"""
        if not self.recording:
            # Start recording
            # Determine base directory and profile-specific subdirectory
            base_db_dir = os.path.join(os.getcwd(), "LD1_Cycle_Database")
            
            if self.current_test_profile:
                # Use test profile name for subdirectory
                profile_name = self.current_test_profile.get('name', 'Unknown_Profile')
                safe_profile_name = self.get_safe_profile_name(profile_name)
                db_dir = os.path.join(base_db_dir, safe_profile_name)
            else:
                # Use "No_Profile" subdirectory for runs without profiles
                db_dir = os.path.join(base_db_dir, "No_Profile")
            
            # Ensure directory exists
            os.makedirs(db_dir, exist_ok=True)
            
            log_name = self.log_name_input.text().strip()
            if log_name:
                # Use custom name
                proposed_db = os.path.abspath(os.path.join(db_dir, f"{log_name}.db"))
            else:
                # Use timestamp
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                proposed_db = os.path.abspath(os.path.join(db_dir, f"ir_log_tempf_{ts}.db"))
            
            # Check if database already exists
            if os.path.exists(proposed_db):
                reply = QMessageBox.question(
                    self, 
                    "Database Already Exists", 
                    f"A database file named '{os.path.basename(proposed_db)}' already exists.\n\n"
                    f"What would you like to do?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel
                )
                
                # Set custom button text
                yes_button = self.sender().parent().findChild(QMessageBox).button(QMessageBox.StandardButton.Yes)
                no_button = self.sender().parent().findChild(QMessageBox).button(QMessageBox.StandardButton.No)
                cancel_button = self.sender().parent().findChild(QMessageBox).button(QMessageBox.StandardButton.Cancel)
                
                # Create custom message box for better control
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Icon.Warning)
                msg_box.setWindowTitle("Database Already Exists")
                msg_box.setText(f"A database file named '{os.path.basename(proposed_db)}' already exists.")
                msg_box.setInformativeText("What would you like to do?")
                
                # Apply dark theme styling to the message box
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #2b2b2b;
                        color: white;
                        font-size: 11px;
                    }
                    QMessageBox QLabel {
                        color: white;
                        background-color: transparent;
                    }
                    QMessageBox QPushButton {
                        background-color: #0d7377;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                        font-weight: bold;
                        font-size: 10px;
                        min-width: 80px;
                    }
                    QMessageBox QPushButton:hover {
                        background-color: #14a085;
                    }
                    QMessageBox QPushButton:pressed {
                        background-color: #0a5d61;
                    }
                """)
                
                overwrite_btn = msg_box.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
                rename_btn = msg_box.addButton("Auto-rename", QMessageBox.ButtonRole.ActionRole)
                cancel_btn = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                
                msg_box.setDefaultButton(cancel_btn)
                msg_box.exec()
                
                if msg_box.clickedButton() == overwrite_btn:
                    # User chose to overwrite
                    self.current_db = proposed_db
                elif msg_box.clickedButton() == rename_btn:
                    # Auto-rename with timestamp suffix
                    base_name = os.path.splitext(proposed_db)[0]
                    timestamp_suffix = datetime.now().strftime("_%H%M%S")
                    self.current_db = f"{base_name}{timestamp_suffix}.db"
                    
                    self.status_label.setText(f"Status: Auto-renamed to {os.path.basename(self.current_db)}")
                    self.status_label.setStyleSheet("color: #ffd93d; font-weight: bold;")
                else:
                    # User cancelled
                    return
            else:
                # Database doesn't exist, proceed normally
                self.current_db = proposed_db
            
            # Validate profile variables if using a test profile
            if self.current_test_profile:
                # Check that all required profile variables are filled
                if hasattr(self, 'profile_variable_inputs'):
                    # Use the existing function to get all profile variable values
                    profile_vars = self.get_profile_variable_values()
                    
                    # Check for missing required variables (excluding end-of-cycle variables)
                    missing_vars = []
                    for var in self.current_test_profile.get('variables', []):
                        var_name = var['name']
                        is_end_of_cycle = var.get('end_of_cycle', False)
                        
                        # Skip end-of-cycle variables - they'll be prompted for at the end
                        if is_end_of_cycle:
                            continue
                            
                        if var_name not in profile_vars or (isinstance(profile_vars[var_name], str) and not profile_vars[var_name].strip()):
                            missing_vars.append(var_name)
                    
                    if missing_vars:
                        msg_box = self.create_styled_message_box(
                            QMessageBox.Icon.Warning,
                            "Missing Profile Variables",
                            f"Please fill in all required profile variables:\n• {chr(10).join(missing_vars)}"
                        )
                        msg_box.exec()
                        return
                    
                    # Store profile variables for this recording session
                    self.recording_profile_variables = profile_vars
                    print(f"DEBUG: Stored profile variables: {self.recording_profile_variables}")
                else:
                    msg_box = self.create_styled_message_box(
                        QMessageBox.Icon.Warning,
                        "Profile Variables Not Ready",
                        "Profile variable inputs are not ready. Please try selecting the profile again."
                    )
                    msg_box.exec()
                    return
            else:
                # Clear recording profile variables if no profile selected
                self.recording_profile_variables = {}
            
            self.init_database()
            self.recording = True
            self.paused = False
            self.last_log_time = 0
            
            # Update the last log display since we just created a new database file
            self.update_last_log_display()
            
            self.record_btn.setText("Stop Recording")
            self.record_btn.setStyleSheet("background-color: #e74c3c;")
            self.pause_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Show recording status with profile variables if any
            if self.recording_profile_variables:
                var_summary = ", ".join([f"{k}={v}" for k, v in self.recording_profile_variables.items()])
                self.status_label.setText(f"Status: Recording to {os.path.basename(self.current_db)} ({var_summary})")
            else:
                self.status_label.setText(f"Status: Recording to {os.path.basename(self.current_db)}")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        else:
            # Stop recording
            self.recording = False
            self.paused = False
            
            # Check for end-of-cycle variables in the current profile
            if self.current_test_profile:
                # Find end-of-cycle variables
                end_of_cycle_vars = []
                if 'variables' in self.current_test_profile:
                    for var in self.current_test_profile['variables']:
                        if var.get('end_of_cycle', False):
                            end_of_cycle_vars.append(var)
                
                # Show end-of-cycle dialog if there are variables to collect
                if end_of_cycle_vars:
                    dialog = EndOfCycleDialog(self, end_of_cycle_vars)
                    if dialog.exec() == QDialog.DialogCode.Accepted:
                        # Get the collected values
                        end_of_cycle_values = dialog.get_variable_values()
                        
                        # Save to database - add to session_info
                        conn = sqlite3.connect(self.current_db)
                        cursor = conn.cursor()
                        for var_name, value in end_of_cycle_values.items():
                            cursor.execute('''
                                INSERT OR REPLACE INTO session_info (key, value)
                                VALUES (?, ?)
                            ''', (f'end_of_cycle_{var_name}', value))
                        conn.commit()
                        conn.close()
                        
                        QMessageBox.information(self, "Success", 
                                              f"End-of-cycle variables saved successfully.")
                    else:
                        # User cancelled - ask if they want to continue without saving
                        reply = QMessageBox.question(self, "Confirm", 
                                                   "Recording stopped without saving end-of-cycle variables. Continue?",
                                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        if reply == QMessageBox.StandardButton.No:
                            # Resume recording
                            self.recording = True
                            self.record_btn.setText("Stop Recording")
                            self.record_btn.setStyleSheet("background-color: #e74c3c;")
                            self.pause_btn.setEnabled(True)
                            self.status_label.setText("Status: Recording resumed")
                            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                            return
            
            # Clear stored profile variables for this session
            self.recording_profile_variables = {}
            
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #0d7377;")
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("Pause")
            
            self.status_label.setText("Status: Recording stopped")
            self.status_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")

    def toggle_pause(self):
        """Pause/resume recording"""
        if self.recording:
            self.paused = not self.paused
            if self.paused:
                self.pause_btn.setText("Resume")
                self.status_label.setText("Status: Recording paused")
                self.status_label.setStyleSheet("color: #ffd93d; font-weight: bold;")
            else:
                self.pause_btn.setText("Pause")
                self.status_label.setText("Status: Recording resumed")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def init_database(self):
        """Initialize database for logging"""
        conn = sqlite3.connect(self.current_db)
        
        # Create base readings table with PLC data columns
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                timestamp TEXT,
                temp_f_mean REAL,
                temp_f_max REAL,
                temp_f_min REAL
            )
        """)
        
        # Create PLC heating profile table for the complete heating cycle
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plc_heating_profile (
                step_number INTEGER,
                time_sp INTEGER,
                start_temp_sp INTEGER,
                end_temp_sp INTEGER,
                vac_sp REAL,
                timestamp TEXT,
                PRIMARY KEY (step_number, timestamp)
            )
        """)
        
        # Create session_info table for profile variables (stored once per recording session)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Read and store initial PLC heating profile
        self.read_and_store_plc_data()
        
        # Store profile information and variables once
        if self.current_test_profile:
            # Store the profile definition
            profile_json = json.dumps(self.current_test_profile)
            conn.execute("INSERT OR REPLACE INTO session_info VALUES ('test_profile', ?)", (profile_json,))
            
            # Store the current variable values for this session
            for var_name, var_value in self.recording_profile_variables.items():
                conn.execute("INSERT OR REPLACE INTO session_info VALUES (?, ?)", (f"var_{var_name}", str(var_value)))
        
        # Store session start time
        session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT OR REPLACE INTO session_info VALUES ('session_start', ?)", (session_start,))
        
        conn.commit()
        conn.close()

    def read_and_store_plc_data(self):
        """Read PLC heating profile and store in database"""
        if not self.current_db:
            return
            
        try:
            # Read heating profile from PLC
            profile_data = self.plc_heating.read_heating_profile()
            
            if profile_data is None:
                print("⚠️ Warning: Could not read PLC heating profile")
                return
                
            # Store the profile data
            self.heating_profile_data = profile_data
            
            # Store in database
            conn = sqlite3.connect(self.current_db)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for step_data in profile_data['steps']:
                # Only store steps that have valid data
                if (step_data.get('time_sp') is not None and 
                    step_data.get('start_temp_sp') is not None and 
                    step_data.get('end_temp_sp') is not None):
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO plc_heating_profile 
                        (step_number, time_sp, start_temp_sp, end_temp_sp, vac_sp, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        step_data['step_number'],
                        step_data['time_sp'],
                        step_data['start_temp_sp'],
                        step_data['end_temp_sp'],
                        step_data.get('vac_sp', 0.0),
                        timestamp
                    ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ PLC heating profile stored successfully ({len(profile_data['steps'])} steps)")
            
            # Update graph with heating curve
            if hasattr(self, 'graph_widget'):
                self.graph_widget.set_heating_profile(profile_data)
                
        except Exception as e:
            print(f"⚠️ Warning: PLC data read/store failed: {e}")

    def periodic_plc_update(self):
        """Periodically read PLC data during recording to catch any changes"""
        if not self.recording or self.paused:
            return
            
        try:
            # Read current heating profile
            current_profile = self.plc_heating.read_heating_profile()
            
            if current_profile is None:
                return
                
            # Compare with stored profile to see if anything changed
            if self.heating_profile_data:
                changes_detected = False
                for i, step in enumerate(current_profile['steps']):
                    if i < len(self.heating_profile_data['steps']):
                        old_step = self.heating_profile_data['steps'][i]
                        
                        # Check if key parameters changed
                        if (step.get('time_sp') != old_step.get('time_sp') or
                            step.get('start_temp_sp') != old_step.get('start_temp_sp') or
                            step.get('end_temp_sp') != old_step.get('end_temp_sp') or
                            step.get('vac_sp') != old_step.get('vac_sp')):
                            changes_detected = True
                            break
                
                if changes_detected:
                    print("🔄 PLC heating profile changes detected, updating database...")
                    self.heating_profile_data = current_profile
                    
                    # Update database with new profile
                    conn = sqlite3.connect(self.current_db)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    for step_data in current_profile['steps']:
                        if (step_data.get('time_sp') is not None and 
                            step_data.get('start_temp_sp') is not None and 
                            step_data.get('end_temp_sp') is not None):
                            
                            conn.execute("""
                                INSERT OR REPLACE INTO plc_heating_profile 
                                (step_number, time_sp, start_temp_sp, end_temp_sp, vac_sp, timestamp)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                step_data['step_number'],
                                step_data['time_sp'],
                                step_data['start_temp_sp'],
                                step_data['end_temp_sp'],
                                step_data.get('vac_sp', 0.0),
                                timestamp
                            ))
                    
                    conn.commit()
                    conn.close()
                    
                    # Update graph
                    if hasattr(self, 'graph_widget'):
                        self.graph_widget.set_heating_profile(current_profile)
                        
        except Exception as e:
            print(f"⚠️ Warning: Periodic PLC update failed: {e}")

    def export_csv(self):
        """Export recorded data to CSV"""
        if not self.current_db or not self.recording_started:
            msg_box = self.create_styled_message_box(
                QMessageBox.Icon.Warning,
                "Warning", 
                "No recording data to export"
            )
            msg_box.exec()
            return
            
        try:
            import pandas as pd
            
            # Determine CSV directory based on test profile
            base_csv_dir = os.path.join(os.getcwd(), "LD1_Cycle_CSV")
            
            if self.current_test_profile:
                # Use test profile name for subdirectory
                profile_name = self.current_test_profile.get('name', 'Unknown_Profile')
                safe_profile_name = self.get_safe_profile_name(profile_name)
                csv_dir = os.path.join(base_csv_dir, safe_profile_name)
            else:
                # Use "No_Profile" subdirectory for runs without profiles
                csv_dir = os.path.join(base_csv_dir, "No_Profile")
            
            # Ensure directory exists
            os.makedirs(csv_dir, exist_ok=True)
            
            # Generate filename based on database name
            base_name = os.path.basename(self.current_db).replace(".db", ".csv")
            file_path = os.path.join(csv_dir, base_name)
            
            # Read temperature data
            conn = sqlite3.connect(self.current_db)
            df = pd.read_sql_query("SELECT * FROM readings", conn)
            
            # Read PLC heating profile data
            plc_df = None
            try:
                plc_df = pd.read_sql_query("SELECT * FROM plc_heating_profile ORDER BY step_number", conn)
            except:
                pass  # Table might not exist if no PLC data was recorded
            
            # Get session info (profile variables stored once)
            session_info = {}
            
            # Get start-of-cycle variables (var_ prefix)
            cursor = conn.execute("SELECT key, value FROM session_info WHERE key LIKE 'var_%'")
            for key, value in cursor.fetchall():
                var_name = key[4:]  # Remove 'var_' prefix
                session_info[var_name] = value
            
            # Get end-of-cycle variables (end_of_cycle_ prefix)
            cursor = conn.execute("SELECT key, value FROM session_info WHERE key LIKE 'end_of_cycle_%'")
            for key, value in cursor.fetchall():
                var_name = key[13:]  # Remove 'end_of_cycle_' prefix
                session_info[f"{var_name} (End-of-Cycle)"] = value
            
            conn.close()
            
            # Create CSV with profile variables as header comments, not repeated columns
            with open(file_path, 'w', newline='') as csvfile:
                # Write profile variables as comments at the top
                if session_info:
                    csvfile.write("# Profile Variables (constants for this recording session):\n")
                    for var_name, var_value in session_info.items():
                        csvfile.write(f"# {var_name}: {var_value}\n")
                    csvfile.write("#\n")
                
                # Write PLC heating profile data as comments
                if plc_df is not None and not plc_df.empty:
                    csvfile.write("# PLC Heating Profile:\n")
                    csvfile.write("# Step, Time(min), Start Temp, End Temp, Vacuum\n")
                    for _, row in plc_df.iterrows():
                        csvfile.write(f"# {row['step_number']}, {row['time_sp']}, {row['start_temp_sp']}, {row['end_temp_sp']}, {row['vac_sp']}\n")
                    csvfile.write("#\n")
                
                csvfile.write("# Temperature Readings:\n")
                
                # Write the temperature data only
                df.to_csv(csvfile, index=False)
            
            # Show success message
            msg_box = self.create_styled_message_box(
                QMessageBox.Icon.Information,
                "Success",
                f"Data exported to {file_path}"
            )
            msg_box.exec()
            self.status_label.setText(f"Status: Data exported to {os.path.basename(file_path)}")
                
        except Exception as e:
            msg_box = self.create_styled_message_box(
                QMessageBox.Icon.Critical,
                "Error",
                f"Export failed: {str(e)}"
            )
            msg_box.exec()

    def select_comparison_db(self):
        """Select a database file to compare against"""
        base_db_dir = os.path.join(os.getcwd(), "LD1_Cycle_Database")
        
        if not os.path.exists(base_db_dir):
            msg_box = self.create_styled_message_box(
                QMessageBox.Icon.Warning,
                "Warning",
                "No database directory found. Record some data first."
            )
            msg_box.exec()
            return
        
        # Start from the current profile's directory if available
        if self.current_test_profile:
            profile_name = self.current_test_profile.get('name', 'Unknown_Profile')
            safe_profile_name = self.get_safe_profile_name(profile_name)
            profile_db_dir = os.path.join(base_db_dir, safe_profile_name)
            
            # Use profile directory if it exists, otherwise use base directory
            start_dir = profile_db_dir if os.path.exists(profile_db_dir) else base_db_dir
        else:
            start_dir = base_db_dir
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Database to Compare", 
            start_dir,
            "Database Files (*.db)"
        )
        
        if file_path:
            # Verify the database has the correct structure
            try:
                conn = sqlite3.connect(file_path)
                rows = conn.execute("SELECT * FROM readings LIMIT 1").fetchall()
                conn.close()
                
                if rows:
                    self.graph_widget.set_comparison_db(file_path)
                    self.clear_compare_btn.setEnabled(True)
                    
                    # Update the graph immediately if we're currently recording
                    if self.recording and self.current_db:
                        self.graph_widget.update_graph(self.current_db)
                    
                    filename = os.path.basename(file_path)
                    msg_box = self.create_styled_message_box(
                        QMessageBox.Icon.Information,
                        "Success",
                        f"Comparison set to: {filename}"
                    )
                    msg_box.exec()
                    self.status_label.setText(f"Status: Comparing with {filename}")
                else:
                    msg_box = self.create_styled_message_box(
                        QMessageBox.Icon.Warning,
                        "Warning",
                        "Selected database contains no data"
                    )
                    msg_box.exec()
                    
            except Exception as e:
                msg_box = self.create_styled_message_box(
                    QMessageBox.Icon.Critical,
                    "Error",
                    f"Failed to load comparison database: {str(e)}"
                )
                msg_box.exec()

    def clear_comparison(self):
        """Clear the current comparison"""
        self.graph_widget.clear_comparison()
        self.clear_compare_btn.setEnabled(False)
        
        # Update the graph immediately if we're currently recording
        if self.recording and self.current_db:
            self.graph_widget.update_graph(self.current_db)
        
        msg_box = self.create_styled_message_box(
            QMessageBox.Icon.Information,
            "Success",
            "Comparison cleared"
        )
        msg_box.exec()
        self.status_label.setText("Status: Comparison cleared")

    def closeEvent(self, event):
        """Handle application close"""
        print("Closing application...")
        
        # Stop the graph update timer first
        if hasattr(self, 'graph_timer'):
            self.graph_timer.stop()
            print("Graph timer stopped")
            
        # Stop the PLC update timer
        if hasattr(self, 'plc_timer'):
            self.plc_timer.stop()
            print("PLC timer stopped")
        
        # Stop recording if active
        if self.recording:
            self.recording = False
            self.paused = False
            print("Recording stopped")
            
        # Disconnect PLC
        if hasattr(self, 'plc_heating') and self.plc_heating:
            self.plc_heating.disconnect()
            print("PLC disconnected")
        
        # Stop the camera worker cleanly
        if hasattr(self, 'camera_worker') and self.camera_worker:
            print("Stopping camera worker...")
            self.camera_worker.stop()
            
        # Wait for camera thread to finish with timeout
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.isRunning():
            print("Waiting for camera thread to finish...")
            self.camera_thread.quit()
            if not self.camera_thread.wait(3000):  # 3 second timeout
                print("Camera thread did not finish cleanly, terminating...")
                self.camera_thread.terminate()
                self.camera_thread.wait(1000)  # Wait 1 more second
            print("Camera thread finished")
        
        # Close any remaining database connections
        if hasattr(self, 'current_db') and self.current_db:
            try:
                # Force close any lingering connections
                import gc
                gc.collect()
                print("Database connections cleaned up")
            except Exception as e:
                print(f"Error cleaning up database: {e}")
        
        print("Application cleanup complete")
        event.accept()

class CameraWorker(QObject):
    """Worker class for camera operations"""
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.is_running = False

    def run(self):
        """Main camera loop"""
        self.is_running = True
        system = None
        cam = None
        
        try:
            system = PySpin.System.GetInstance()
            cams = system.GetCameras()
            
            if cams.GetSize() == 0:
                self.status_update.emit("No cameras found")
                self.finished.emit()
                return
            
            cam = cams[0]
            cam.Init()
            cam.BeginAcquisition()
            
            self.status_update.emit("Camera connected and streaming")
            
            while self.is_running:
                try:
                    # Use shorter timeout for more responsive shutdown
                    img = cam.GetNextImage(500)  # 0.5 second timeout
                    if img.IsIncomplete():
                        img.Release()
                        continue
                    
                    frame = img.GetNDArray()
                    self.frame_ready.emit(frame)
                    img.Release()
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.03)  # ~30 FPS
                    
                except Exception as e:
                    if self.is_running:  # Only print errors if we're still supposed to be running
                        print(f"Frame capture error: {e}")
                    time.sleep(0.1)
            
        except Exception as e:
            if self.is_running:
                self.status_update.emit(f"Camera error: {str(e)}")
        
        finally:
            # Cleanup camera resources
            try:
                if cam:
                    cam.EndAcquisition()
                    cam.DeInit()
                    del cam
                if system:
                    cams = system.GetCameras()
                    cams.Clear()
                    system.ReleaseInstance()
                print("Camera resources cleaned up successfully")
            except Exception as e:
                print(f"Error during camera cleanup: {e}")
        
        self.finished.emit()

    def stop(self):
        """Stop camera worker"""
        print("Camera worker stop requested")
        self.is_running = False

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application icon and info
    app.setApplicationName("FLIR Thermal Camera")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Thermal Imaging Solutions")
    
    # Handle application quit more gracefully
    app.setQuitOnLastWindowClosed(True)
    
    try:
        window = ModernThermalApp()
        window.show()
        
        exit_code = app.exec()
        print("Application exited with code:", exit_code)
        
    except Exception as e:
        print(f"Application error: {e}")
        exit_code = 1
    
    finally:
        # Final cleanup
        try:
            app.quit()
        except:
            pass
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
