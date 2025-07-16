#!/usr/bin/env python3
#Single Cam Version
# FLIR GigE Vision IR Camera Application
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

# --- Globals ---
# Support for 2 ROIs
roi1_pts = []
roi2_pts = []
roi1_ready = False
roi2_ready = False
current_roi = 1  # Which ROI we're currently drawing (1 or 2)
recording = False
paused = False
current_db = None
last_log_time = 0
recording_started = False
rotation_mode = 0

def handle_interrupt(sig, frame):
    print("\n⚠️ Interrupted by user. Exiting ...")
    sys.exit(0)
signal.signal(signal.SIGINT, handle_interrupt)

def export_db_to_csv():
    global current_db, recording_started
    if not current_db or not recording_started:
        print("⚠️ No recording to export.")
        return
    try:
        import pandas as pd
        out_dir = "/home/claydamkoehler/Desktop/IR Data"
        os.makedirs(out_dir, exist_ok=True)
        conn = sqlite3.connect(current_db)
        df = pd.read_sql_query("SELECT * FROM readings", conn)
        conn.close()
        csv_path = os.path.join(out_dir, os.path.basename(current_db).replace(".db", ".csv"))
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV exported to {csv_path}")
    except Exception as e:
        print(f"❌ CSV export failed: {e}")

def get_new_db_filename():
    global current_db
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_db = os.path.abspath(f"ir_log_tempf_{ts}.db")
    return current_db

def init_db(db_file):
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            timestamp TEXT,
            roi1_temp_f_mean REAL,
            roi1_temp_f_max REAL,
            roi1_temp_f_min REAL,
            roi2_temp_f_mean REAL,
            roi2_temp_f_max REAL,
            roi2_temp_f_min REAL
        )
    """)
    conn.commit()
    conn.close()

def log_to_db(ts, roi1_mean_f, roi1_max_f, roi1_min_f, roi2_mean_f, roi2_max_f, roi2_min_f):
    global recording_started
    if not current_db:
        return
    conn = sqlite3.connect(current_db)
    conn.execute("INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?)", 
                (ts, roi1_mean_f, roi1_max_f, roi1_min_f, roi2_mean_f, roi2_max_f, roi2_min_f))
    conn.commit()
    conn.close()
    recording_started = True

def draw_roi(event, x, y, flags, param):
    global roi1_pts, roi2_pts, current_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_roi == 1:
            roi1_pts.append((x, y))
        else:
            roi2_pts.append((x, y))

fig, ax = plt.subplots(figsize=(8, 4))
timestamps, means, maxs, mins = [], [], [], []

def smooth(data, window=7):
    return np.convolve(data, np.ones(window)/window, mode='valid') if len(data) >= window else data

def update_graph(_):
    if not recording or not current_db:
        return
    conn = sqlite3.connect(current_db)
    rows = conn.execute("SELECT * FROM readings ORDER BY ROWID ASC").fetchall()
    conn.close()
    if not rows:
        return

    timestamps[:] = [datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") for r in rows]
    # ROI 1 data (columns 1-3)
    roi1_means = [r[1] for r in rows]
    roi1_maxs = [r[2] for r in rows]
    roi1_mins = [r[3] for r in rows]
    # ROI 2 data (columns 4-6)
    roi2_means = [r[4] for r in rows]
    roi2_maxs = [r[5] for r in rows]
    roi2_mins = [r[6] for r in rows]

    smooth_roi1_max = smooth(roi1_maxs)
    smooth_roi1_min = smooth(roi1_mins)
    smooth_roi2_max = smooth(roi2_maxs)
    smooth_roi2_min = smooth(roi2_mins)

    valid_timestamps = timestamps[len(timestamps)-len(smooth_roi1_max):]

    ax.clear()
    ax.plot(valid_timestamps, smooth_roi1_max, label="ROI1 Max °F", color='red', linestyle='-')
    ax.plot(valid_timestamps, smooth_roi1_min, label="ROI1 Min °F", color='blue', linestyle='-')
    ax.plot(valid_timestamps, smooth_roi2_max, label="ROI2 Max °F", color='orange', linestyle='--')
    ax.plot(valid_timestamps, smooth_roi2_min, label="ROI2 Min °F", color='cyan', linestyle='--')
    ax.set_title("Live ROI Calibrated Max/Min Temperature (°F)")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()


anim = FuncAnimation(fig, update_graph, interval=1000, cache_frame_data=False)

def camera_loop():
    global roi1_pts, roi2_pts, roi1_ready, roi2_ready, current_roi
    global recording, paused, current_db, last_log_time, rotation_mode

    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    if cams.GetSize() == 0:
        print("❌ No cameras found.")
        system.ReleaseInstance()
        return

    cam = cams[0]
    cam.Init()
    cam.BeginAcquisition()
    cv2.namedWindow("FLIR Thermal Feed")
    cv2.setMouseCallback("FLIR Thermal Feed", draw_roi)
    print("Instructions:\n"
          " • Left-click = add vertex   • 'd' = finish current ROI\n"
          " • 'c' = clear current ROI   • '1'/'2' = switch to ROI 1/2\n"
          " • 'r' = start recording     • 'p' = pause\n"
          " • 'e' = export to CSV       • 'f' = rotate 90°        • 'q' = quit\n")

    try:
        while True:
            img = cam.GetNextImage(5000)
            if img.IsIncomplete():
                img.Release()
                continue

            frame = img.GetNDArray()

            if rotation_mode == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_mode == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_mode == 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            vis = cv2.applyColorMap(
                cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_INFERNO)

            # Draw ROI 1 in green
            if roi1_pts:
                pts = np.array(roi1_pts, np.int32)
                cv2.polylines(vis, [pts], roi1_ready, (0, 255, 0), 2)
                for p in roi1_pts:
                    cv2.circle(vis, p, 3, (0, 255, 0), -1)
                if roi1_ready:
                    cv2.putText(vis, "ROI1", (roi1_pts[0][0], roi1_pts[0][1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw ROI 2 in cyan
            if roi2_pts:
                pts = np.array(roi2_pts, np.int32)
                cv2.polylines(vis, [pts], roi2_ready, (255, 255, 0), 2)
                for p in roi2_pts:
                    cv2.circle(vis, p, 3, (255, 255, 0), -1)
                if roi2_ready:
                    cv2.putText(vis, "ROI2", (roi2_pts[0][0], roi2_pts[0][1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Calculate temperatures for both ROIs
            roi1_temp_data = [None, None, None]  # mean, max, min
            roi2_temp_data = [None, None, None]  # mean, max, min

            if roi1_ready:
                mask1 = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillPoly(mask1, [np.array(roi1_pts, np.int32)], 1)
                roi1_vals = frame[mask1 == 1]
                if roi1_vals.size:
                    mean_raw = np.mean(roi1_vals)
                    max_raw = np.max(roi1_vals)
                    min_raw = np.min(roi1_vals)

                    T_F_mean = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(mean_raw)))
                    T_F_max  = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(max_raw)))
                    T_F_min  = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(min_raw)))
                    
                    roi1_temp_data = [T_F_mean, T_F_max, T_F_min]
                    txt = f"ROI1: {T_F_mean:.1f} F"
                    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if roi2_ready:
                mask2 = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillPoly(mask2, [np.array(roi2_pts, np.int32)], 1)
                roi2_vals = frame[mask2 == 1]
                if roi2_vals.size:
                    mean_raw = np.mean(roi2_vals)
                    max_raw = np.max(roi2_vals)
                    min_raw = np.min(roi2_vals)

                    T_F_mean = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(mean_raw)))
                    T_F_max  = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(max_raw)))
                    T_F_min  = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(min_raw)))
                    
                    roi2_temp_data = [T_F_mean, T_F_max, T_F_min]
                    txt = f"ROI2: {T_F_mean:.1f} F"
                    cv2.putText(vis, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show current ROI being drawn
            cv2.putText(vis, f"Current ROI: {current_roi}", (10, vis.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Log data if recording
            if recording and not paused and time.time() - last_log_time >= 10:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_to_db(ts, 
                         roi1_temp_data[0] or 0, roi1_temp_data[1] or 0, roi1_temp_data[2] or 0,
                         roi2_temp_data[0] or 0, roi2_temp_data[1] or 0, roi2_temp_data[2] or 0)
                last_log_time = time.time()

            cv2.imshow("FLIR Thermal Feed", vis)
            img.Release()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                rotation_mode = (rotation_mode + 1) % 4
                print(f"🔄 Rotated to {rotation_mode * 90}°")
            elif key == ord("1"):
                current_roi = 1
                print("📍 Switched to ROI 1")
            elif key == ord("2"):
                current_roi = 2
                print("📍 Switched to ROI 2")
            elif key == ord("d"):
                if current_roi == 1 and len(roi1_pts) >= 3:
                    roi1_ready = True
                    print("✅ ROI 1 finished.")
                elif current_roi == 2 and len(roi2_pts) >= 3:
                    roi2_ready = True
                    print("✅ ROI 2 finished.")
                else:
                    print(f"⚠️ Need at least 3 points for ROI {current_roi}")
            elif key == ord("c"):
                if current_roi == 1:
                    roi1_pts.clear()
                    roi1_ready = False
                    print("🗑️ ROI 1 cleared.")
                else:
                    roi2_pts.clear()
                    roi2_ready = False
                    print("🗑️ ROI 2 cleared.")
            elif key == ord("r"):
                if roi1_ready or roi2_ready:
                    current_db = get_new_db_filename()
                    init_db(current_db)
                    recording, paused = True, False
                    last_log_time = 0
                    print(f"▶️ Recording → {current_db}")
                else:
                    print("⚠️ Finish at least one ROI first.")
            elif key == ord("p") and recording:
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord("e"):
                export_db_to_csv()

    finally:
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cams.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()

def main():
    threading.Thread(target=camera_loop, daemon=True).start()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
