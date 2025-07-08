#!/usr/bin/env python3
# pip install PyQt5 matplotlib
import sys
import threading
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTabWidget, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import PySpin
import time
import os
import sqlite3
from datetime import datetime
import signal
import sys
import math

# --- FLIR Calibration Constants from SpinView ---
R = 554118
B = 1597.67
F = 1.0
O = 49750

# --- Environmental Parameters ---
EMISSIVITY = 0.95
REFLECTED_TEMP_K = 293.15

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

def handle_interrupt(sig, frame):
    print("\n⚠️ Interrupted by user. Exiting ...")
    sys.exit(0)
signal.signal(signal.SIGINT, handle_interrupt)

def get_new_db_filename(prefix):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.abspath(f"{prefix}_ir_log_tempf_{ts}.db")

def init_db(db_file):
    conn = sqlite3.connect(db_file)
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

def log_to_db(db_file, ts, mean_f, max_f, min_f):
    conn = sqlite3.connect(db_file)
    conn.execute("INSERT INTO readings VALUES (?, ?, ?, ?)", (ts, mean_f, max_f, min_f))
    conn.commit()
    conn.close()

def export_db_to_csv(db_file):
    try:
        import pandas as pd
        out_dir = os.path.expanduser("~/Desktop/IR Data")
        os.makedirs(out_dir, exist_ok=True)
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT * FROM readings", conn)
        conn.close()
        csv_path = os.path.join(out_dir, os.path.basename(db_file).replace(".db", ".csv"))
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV exported to {csv_path}")
    except Exception as e:
        print(f"❌ CSV export failed: {e}")

def smooth(data, window=7):
    return np.convolve(data, np.ones(window)/window, mode='valid') if len(data) >= window else data

class CameraState:
    def __init__(self, window_name, db_prefix):
        self.polygon_pts = []
        self.polygon_ready = False
        self.recording = False
        self.paused = False
        self.current_db = None
        self.last_log_time = 0
        self.recording_started = False
        self.rotation_mode = 0
        self.window_name = window_name
        self.db_prefix = db_prefix
        self.timestamps = []
        self.maxs = []
        self.mins = []
        self.lock = threading.Lock()

    def draw_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon_pts.append((x, y))

    def reset_roi(self):
        self.polygon_pts.clear()
        self.polygon_ready = False

class ROILabel(QLabel):
    roi_point_added = pyqtSignal(int, int)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.roi_point_added.emit(event.x(), event.y())

def camera_loop(cam, state: CameraState, is_running):
    cam.Init()
    cam.BeginAcquisition()
    try:
        while is_running():
            img = cam.GetNextImage(5000)
            if img.IsIncomplete():
                img.Release()
                continue
            frame = img.GetNDArray()
            
            # -----------------------------------------
            if state.rotation_mode == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif state.rotation_mode == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif state.rotation_mode == 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = np.max(frame) - frame
            norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            vis = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
            # Draw ROI overlay for GUI
            if state.polygon_pts:
                pts = np.array(state.polygon_pts, np.int32)
                cv2.polylines(vis, [pts], state.polygon_ready, (0, 255, 0), 1)
                for p in state.polygon_pts:
                    cv2.circle(vis, p, 3, (0, 255, 0), -1)
            if state.polygon_ready:
                mask = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(state.polygon_pts, np.int32)], 1)
                roi_vals = frame[mask == 1]
                if roi_vals.size:
                    mean_raw = np.mean(roi_vals)
                    max_raw = np.max(roi_vals)
                    min_raw = np.min(roi_vals)
                    T_F_mean = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(mean_raw)))
                    T_F_max  = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(max_raw)))
                    T_F_min  = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(min_raw)))
                    txt = f"ROI Temp: {T_F_mean:.1f} F"
                    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if state.recording and not state.paused and time.time() - state.last_log_time >= 10:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_to_db(state.current_db, ts, T_F_mean, T_F_max, T_F_min)
                        state.last_log_time = time.time()
                        with state.lock:
                            state.timestamps.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
                            state.maxs.append(T_F_max)
                            state.mins.append(T_F_min)
            # Always update the GUI image
            state.latest_vis = vis  # Use the color-mapped, annotated image
            img.Release()
    finally:
        cam.EndAcquisition()
        cam.DeInit()
        del cam

def plot_loop(states):
    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    while True:
        for i, state in enumerate(states):
            ax = axes[i]
            with state.lock:
                ax.clear()
                if state.timestamps and state.maxs and state.mins:
                    smooth_max = smooth(state.maxs)
                    smooth_min = smooth(state.mins)
                    valid_timestamps = state.timestamps[len(state.timestamps)-len(smooth_max):]
                    ax.plot(valid_timestamps, smooth_max, label="Max °F", color='red')
                    ax.plot(valid_timestamps, smooth_min, label="Min °F", color='blue')
                    ax.set_title(f"Live ROI Calibrated Max/Min Temperature (°F) - {state.window_name}")
                    ax.grid(True)
                    ax.legend(loc="upper left")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        plt.pause(0.1)
        time.sleep(1)

class CameraWidget(QWidget):
    def __init__(self, camera_state, quit_callback):
        super().__init__()
        self.state = camera_state
        self.quit_callback = quit_callback
        self.image_label = ROILabel("Camera Feed")
        self.image_label.roi_point_added.connect(self.add_roi_point)
        self.image_label.setAlignment(Qt.AlignCenter)
        # --- FIX: Set a fixed size and size policy ---
        self.image_label.setMinimumSize(640, 480)  # or your camera's resolution
        self.image_label.setMaximumSize(1280, 960) # or your max expected size
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # ---------------------------------------------
        self.plot_canvas = FigureCanvas(Figure(figsize=(4,2)))
        self.ax = self.plot_canvas.figure.subplots()
        self.plot_canvas.figure.subplots_adjust(bottom=0.22)  # Add this line (adjust value as needed)
        self.btn_record = QPushButton("Start Recording")
        self.btn_pause = QPushButton("Pause")
        self.btn_finish_roi = QPushButton("Finish ROI")
        self.btn_clear_roi = QPushButton("Clear ROI")  # <-- new button
        self.btn_export = QPushButton("Export CSV")
        self.btn_quit = QPushButton("Quit")
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_finish_roi.clicked.connect(self.finish_roi)
        self.btn_clear_roi.clicked.connect(self.clear_roi)  # <-- new connection
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_quit.clicked.connect(self.quit_callback)

        # --- Layout changes for consistent sizing ---
        layout = QVBoxLayout()
        layout.addWidget(self.image_label, stretch=4)  # Camera feed gets more space
        self.plot_canvas.setFixedHeight(150)           # Plot gets a fixed height
        layout.addWidget(self.plot_canvas, stretch=0)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_record)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_finish_roi)
        btn_layout.addWidget(self.btn_clear_roi)  # <-- new button in layout
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_quit)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        # --------------------------------------------

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.refresh_image)
        self.image_timer.start(50)

    def update_image(self, frame):
        # frame is already color-mapped and annotated (BGR)
        display_img = frame.copy()
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def refresh_image(self):
        if hasattr(self.state, "latest_vis"):
            self.update_image(self.state.latest_vis)

    def update_plot(self):
        with self.state.lock:
            self.ax.clear()
            n = min(len(self.state.timestamps), len(self.state.maxs), len(self.state.mins))
            if n > 0:
                self.ax.plot(self.state.timestamps[-n:], self.state.maxs[-n:], label="Max °F", color='red')
                self.ax.plot(self.state.timestamps[-n:], self.state.mins[-n:], label="Min °F", color='blue')
                self.ax.legend()
            self.plot_canvas.draw()

    def toggle_record(self):
        if not self.state.polygon_ready:
            print("⚠️ Finish ROI first.")
            return
        if not self.state.recording:
            self.state.current_db = get_new_db_filename(self.state.db_prefix)
            init_db(self.state.current_db)
            self.state.recording = True
            self.state.paused = False
            self.state.last_log_time = 0
            self.btn_record.setText("Stop Recording")
            print(f"▶️ Recording → {self.state.current_db}")
        else:
            self.state.recording = False
            self.btn_record.setText("Start Recording")
            print("⏹️ Recording stopped.")

    def toggle_pause(self):
        if self.state.recording:
            self.state.paused = not self.state.paused
            if self.state.paused:
                self.btn_pause.setText("Resume")
                print("⏸️ Paused")
            else:
                self.btn_pause.setText("Pause")
                print("▶️ Resumed")
        else:
            print("⚠️ Not recording.")

    def finish_roi(self):
        if len(self.state.polygon_pts) >= 3:
            self.state.polygon_ready = True
            print("✅ ROI finished.")
        else:
            print("⚠️ Need at least 3 points for ROI.")

    def export_csv(self):
        if self.state.current_db:
            export_db_to_csv(self.state.current_db)
        else:
            print("⚠️ No recording to export.")

    def clear_roi(self):
        self.state.reset_roi()
        self.refresh_image()

    def add_roi_point(self, x, y):
        # Map from label coordinates to image coordinates
        if hasattr(self.state, "latest_vis"):
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            img = self.state.latest_vis
            img_h, img_w = img.shape[:2]
            # Compute scale and offset for aspect-ratio preserving fit
            scale = min(label_w / img_w, label_h / img_h)
            x_offset = (label_w - img_w * scale) / 2
            y_offset = (label_h - img_h * scale) / 2
            # Only accept clicks inside the image area
            if (x_offset <= x <= label_w - x_offset) and (y_offset <= y <= label_h - y_offset):
                img_x = int((x - x_offset) / scale)
                img_y = int((y - y_offset) / scale)
                self.state.polygon_pts.append((img_x, img_y))
                self.state.polygon_ready = False
                self.refresh_image()

class MainWindow(QWidget):
    def __init__(self, state0, state1, quit_callback):
        super().__init__()
        self.setWindowTitle("Modern IR Camera Logger")
        tabs = QTabWidget()
        self.cam0_widget = CameraWidget(state0, quit_callback)
        self.cam1_widget = CameraWidget(state1, quit_callback)
        tabs.addTab(self.cam0_widget, "Camera 0")
        tabs.addTab(self.cam1_widget, "Camera 1")
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

def main():
    app = QApplication(sys.argv)

    # ---- DARK THEME STYLESHEET ----
    # dark_stylesheet = """
    # QWidget {
    #     background-color: #232629;
    #     color: #f0f0f0;
    #     font-size: 12pt;
    # }
    # QTabWidget::pane {
    #     border: 1px solid #444;
    #     background: #232629;
    # }
    # QTabBar::tab {
    #     background: #2d2f31;
    #     color: #f0f0f0;
    #     border: 1px solid #444;
    #     padding: 8px;
    #     min-width: 100px;
    # }
    # QTabBar::tab:selected {
    #     background: #393c3f;
    #     color: #fff;
    # }
    # QLabel {
    #     color: #f0f0f0;
    # }
    # QPushButton {
    #     background-color: #393c3f;
    #     color: #f0f0f0;
    #     border: 1px solid #555;
    #     border-radius: 4px;
    #     padding: 6px 12px;
    # }
    # QPushButton:hover {
    #     background-color: #505357;
    # }
    # QPushButton:pressed {
    #     background-color: #232629;
    # }
    # QLineEdit, QComboBox, QSpinBox {
    #     background-color: #2d2f31;
    #     color: #f0f0f0;
    #     border: 1px solid #555;
    #     border-radius: 4px;
    # }
    # """
    # app.setStyleSheet(dark_stylesheet)
    # ---- END DARK THEME ----

    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    if cams.GetSize() < 2:
        print("❌ Less than two cameras found.")
        cams.Clear()
        system.ReleaseInstance()
        return

    cam0 = cams[0]
    cam1 = cams[1]
    state0 = CameraState("FLIR Thermal Feed 0", "cam0")
    state1 = CameraState("FLIR Thermal Feed 1", "cam1")

    running = True
    def quit_all():
        nonlocal running
        running = False
        app.quit()

    t0 = threading.Thread(target=camera_loop, args=(cam0, state0, lambda: running), daemon=True)
    t1 = threading.Thread(target=camera_loop, args=(cam1, state1, lambda: running), daemon=True)
    t0.start()
    t1.start()

    window = MainWindow(state0, state1, quit_all)
    window.show()
    app.exec_()

    t0.join()
    t1.join()
    del cam0
    del cam1
    cams.Clear()
    system.ReleaseInstance()

if __name__ == "__main__":
    main()
