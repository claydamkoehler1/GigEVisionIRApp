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
O = 49500

# --- Environmental Parameters ---
EMISSIVITY = 0.95
REFLECTED_TEMP_K = 293.15  # ~20°C in Kelvin

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

def kelvin_to_celsius(K):
    return K - 273.15

def kelvin_to_fahrenheit(K):
    return (K - 273.15) * 9 / 5 + 32

# --------------- Globals -------------------------
polygon_pts        = []
polygon_ready      = False
recording          = False
paused             = False
current_db         = None
last_log_time      = 0
recording_started  = False
rotation_mode      = 0

# --------------- Signal Handling -----------------
def handle_interrupt(sig, frame):
    print("\n⚠️ Interrupted by user. Exiting ...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

# --------------- Export CSV ----------------------
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

# --------------- File & DB Logic -----------------
def get_new_db_filename():
    global current_db
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_db = os.path.abspath(f"ir_log_raw_{ts}.db")
    return current_db

def init_db(db_file):
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            timestamp TEXT,
            mean_raw  REAL,
            max_raw   REAL,
            min_raw   REAL
        )
    """)
    conn.commit()
    conn.close()

def log_to_db(ts, mean_raw, max_raw, min_raw):
    global recording_started
    if not current_db:
        return
    # Convert all to calibrated °F
    mean_T = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(mean_raw)))
    max_T  = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(max_raw)))
    min_T  = kelvin_to_fahrenheit(corrected_temperature_K(raw_to_kelvin(min_raw)))

    conn = sqlite3.connect(current_db)
    conn.execute("INSERT INTO readings VALUES (?,?,?,?)", (ts, mean_T, max_T, min_T))
    conn.commit()
    conn.close()
    recording_started = True

# --------------- ROI Drawing ---------------------
def draw_roi(event, x, y, flags, param):
    global polygon_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_pts.append((x, y))

# --------------- Live Plot -----------------------
fig, ax = plt.subplots(figsize=(8, 4))
timestamps, means, maxs, mins = [], [], [], []

def update_graph(_):
    if not recording or not current_db:
        return
    conn = sqlite3.connect(current_db)
    rows = conn.execute("SELECT * FROM readings ORDER BY ROWID DESC LIMIT 60").fetchall()
    conn.close()
    if not rows:
        return

    rows.reverse()
    timestamps[:] = [datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") for r in rows]
    means[:]      = [r[1] for r in rows]
    maxs[:]       = [r[2] for r in rows]
    mins[:]       = [r[3] for r in rows]

    ax.clear()
    ax.plot(timestamps, means, label="Mean °F")
    ax.plot(timestamps, maxs,  label="Max °F")
    ax.plot(timestamps, mins,  label="Min °F")
    ax.set_title("Live ROI Calibrated Temperature (°F)")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

anim = FuncAnimation(fig, update_graph, interval=1000, cache_frame_data=False)

# --------------- Camera Thread -------------------
def camera_loop():
    global polygon_pts, polygon_ready
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

            if rotation_mode == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_mode == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_mode == 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            vis = cv2.applyColorMap(
                cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_INFERNO)

            if polygon_pts:
                pts = np.array(polygon_pts, np.int32)
                cv2.polylines(vis, [pts], polygon_ready, (0, 255, 0), 1)
                for p in polygon_pts:
                    cv2.circle(vis, p, 3, (0, 255, 0), -1)

            if polygon_ready:
                mask = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(polygon_pts, np.int32)], 1)
                roi_vals = frame[mask == 1]
                if roi_vals.size:
                    raw_mean = np.mean(roi_vals)
                    raw_max = np.max(roi_vals)
                    raw_min = np.min(roi_vals)

                    T_K = raw_to_kelvin(raw_mean)
                    T_K_corr = corrected_temperature_K(T_K)
                    T_C = kelvin_to_celsius(T_K_corr)
                    T_F = kelvin_to_fahrenheit(T_K_corr)

                    txt = f"ROI Temp: {T_F:.1f}°F / {T_C:.1f}°C"
                    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if recording and not paused and time.time() - last_log_time >= 10:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_to_db(ts, raw_mean, raw_max, raw_min)
                        last_log_time = time.time()

            cv2.imshow("FLIR Thermal Feed", vis)
            img.Release()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                rotation_mode = (rotation_mode + 1) % 4
                print(f"🔄 Rotated to {rotation_mode * 90}°")
            elif key == ord("d") and len(polygon_pts) >= 3:
                polygon_ready = True
                print("✅ ROI finished.")
            elif key == ord("c"):
                polygon_pts.clear()
                polygon_ready = False
            elif key == ord("r"):
                if polygon_ready:
                    current_db = get_new_db_filename()
                    init_db(current_db)
                    recording, paused = True, False
                    last_log_time = 0
                    print(f"▶️ Recording → {current_db}")
                else:
                    print("⚠️ Finish ROI first.")
            elif key == ord("p") and recording:
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord("e"):
                export_db_to_csv()

    finally:
        cam.EndAcquisition()
        cam.DeInit()
        cams.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()

# --------------- Main ----------------------------
def main():
    threading.Thread(target=camera_loop, daemon=True).start()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
