#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

import PySpin
import numpy as np
import cv2
import time
import os
import sqlite3
import threading
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

def camera_loop(cam, state: CameraState):
    cam.Init()
    cam.BeginAcquisition()
    cv2.namedWindow(state.window_name)
    cv2.setMouseCallback(state.window_name, state.draw_roi)
    print(f"Instructions for {state.window_name}:\n"
          " • Left-click = add vertex   • 'd' = finish ROI\n"
          " • 'c' = clear ROI           • 'r' = start recording   • 'p' = pause\n"
          " • 'e' = export to CSV       • 'f' = rotate 90°        • 'q' = quit\n")

    try:
        while True:
            img = cam.GetNextImage(5000)
            if img.IsIncomplete():
                img.Release()
                continue

            frame = img.GetNDArray()

            if state.rotation_mode == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif state.rotation_mode == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif state.rotation_mode == 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            vis = cv2.applyColorMap(
                cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_INFERNO)

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

            cv2.imshow(state.window_name, vis)
            img.Release()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                state.rotation_mode = (state.rotation_mode + 1) % 4
                print(f"🔄 Rotated to {state.rotation_mode * 90}°")
            elif key == ord("d") and len(state.polygon_pts) >= 3:
                state.polygon_ready = True
                print("✅ ROI finished.")
            elif key == ord("c"):
                state.reset_roi()
            elif key == ord("r"):
                if state.polygon_ready:
                    state.current_db = get_new_db_filename(state.db_prefix)
                    init_db(state.current_db)
                    state.recording, state.paused = True, False
                    state.last_log_time = 0
                    print(f"▶️ Recording → {state.current_db}")
                else:
                    print("⚠️ Finish ROI first.")
            elif key == ord("p") and state.recording:
                state.paused = not state.paused
                print("⏸️ Paused" if state.paused else "▶️ Resumed")
            elif key == ord("e") and state.current_db:
                export_db_to_csv(state.current_db)

    finally:
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cv2.destroyWindow(state.window_name)

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

def main():
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

    t0 = threading.Thread(target=camera_loop, args=(cam0, state0), daemon=True)
    t1 = threading.Thread(target=camera_loop, args=(cam1, state1), daemon=True)

    t0.start()
    t1.start()

    plot_loop([state0, state1])  # This blocks until you close the plot window

    t0.join()
    t1.join()

    cams.Clear()
    system.ReleaseInstance()

if __name__ == "__main__":
    main()
