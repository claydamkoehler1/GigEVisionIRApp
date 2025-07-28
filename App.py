#!/usr/bin/env python3
# Modern Thermal Camera App with Unified UI
import sys
import os
import time
import sqlite3
import threading
import numpy as np
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
        
        self.setup_ui()
        self.setup_camera()
        
        # Timer for graph updates
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graph)
        self.graph_timer.start(1000)  # Update every second
        
        # Add keyboard shortcut for quick exit
        exit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        exit_shortcut.activated.connect(self.close)

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

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
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
        
        # Log naming section
        log_group = QGroupBox("Recording Settings")
        log_layout = QVBoxLayout(log_group)
        
        log_name_layout = QHBoxLayout()
        log_name_layout.addWidget(QLabel("Log Name:"))
        self.log_name_input = QLineEdit()
        self.log_name_input.setPlaceholderText("Enter custom log name (optional)")
        log_name_layout.addWidget(self.log_name_input)
        log_layout.addLayout(log_name_layout)
        
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
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_text = QLabel(
            "1. Click on the camera image to add ROI vertices\n"
            "2. Click 'Finish ROI' when you have at least 3 points\n"
            "3. Enter a custom log name (optional)\n"
            "4. Click 'Start Recording' to begin data logging\n"
            "5. Use 'Pause' to temporarily stop logging\n"
            "6. Click 'Export CSV' to save recorded data"
        )
        instructions_text.setWordWrap(True)
        instructions_text.setStyleSheet("color: #bbb; font-size: 10px;")
        instructions_layout.addWidget(instructions_text)
        right_panel.addWidget(instructions_group)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)  # 2/3 width for left panel
        main_layout.addLayout(right_panel, 1)  # 1/3 width for right panel

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
            conn.execute("INSERT INTO readings VALUES (?, ?, ?, ?)", 
                        (ts, self.current_temp_data[0], self.current_temp_data[1], self.current_temp_data[2]))
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

    def toggle_recording(self):
        """Start/stop recording"""
        if not self.recording:
            # Start recording
            # Ensure database directory exists
            db_dir = r"P:\Plant Engineering\Lab\Lab Dryer 1\LD1 Cycle Database"
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
            
            self.init_database()
            self.recording = True
            self.paused = False
            self.last_log_time = 0
            
            self.record_btn.setText("Stop Recording")
            self.record_btn.setStyleSheet("background-color: #e74c3c;")
            self.pause_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            self.status_label.setText(f"Status: Recording to {os.path.basename(self.current_db)}")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        else:
            # Stop recording
            self.recording = False
            self.paused = False
            
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                timestamp TEXT,
                temp_f_mean REAL,
                temp_f_max REAL,
                temp_f_min REAL
            )
        """)
        conn.commit()
        conn.close()

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
            
            # Set target directory and ensure it exists
            csv_dir = r"P:\Plant Engineering\Lab\Lab Dryer 1\LD1 Cycle CSV(s)"
            os.makedirs(csv_dir, exist_ok=True)
            
            # Generate filename based on database name
            base_name = os.path.basename(self.current_db).replace(".db", ".csv")
            file_path = os.path.join(csv_dir, base_name)
            
            # Export to CSV
            conn = sqlite3.connect(self.current_db)
            df = pd.read_sql_query("SELECT * FROM readings", conn)
            conn.close()
            df.to_csv(file_path, index=False)
            
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
        db_dir = r"P:\Plant Engineering\Lab\Lab Dryer 1\LD1 Cycle Database"
        if not os.path.exists(db_dir):
            msg_box = self.create_styled_message_box(
                QMessageBox.Icon.Warning,
                "Warning",
                "No database directory found. Record some data first."
            )
            msg_box.exec()
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Database to Compare", 
            db_dir,
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
        
        # Stop recording if active
        if self.recording:
            self.recording = False
            self.paused = False
            print("Recording stopped")
        
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
