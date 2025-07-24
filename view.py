import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QGridLayout, QPushButton, QInputDialog
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import PySpin

# --- Fixed Constants ---
R = 554118
F = 1.0
EMISSIVITY = 0.95
REFLECTED_TEMP_K = 293.15

# --- Conversion Functions ---
def raw_to_kelvin(S, R, B, F, O):
    with np.errstate(divide='ignore', invalid='ignore'):
        return B / np.log((R / (S - O)) + F)

def corrected_temperature_K(T_obj_K, EMISSIVITY, REFLECTED_TEMP_K):
    return (T_obj_K - (1 - EMISSIVITY) * REFLECTED_TEMP_K) / EMISSIVITY

def kelvin_to_fahrenheit(K):
    return (K - 273.15) * 9 / 5 + 32

def fahrenheit_to_kelvin(F):
    return (F - 32) * 5 / 9 + 273.15

def solve_two_point_calibration(S1, T1_K, S2, T2_K):
    # Solve B and O from two equations:
    # T1 = B / ln(R / (S1 - O) + F)
    # T2 = B / ln(R / (S2 - O) + F)
    def equations(p):
        B, O = p
        lhs1 = T1_K
        lhs2 = T2_K
        rhs1 = B / np.log((R / (S1 - O)) + F)
        rhs2 = B / np.log((R / (S2 - O)) + F)
        return [lhs1 - rhs1, lhs2 - rhs2]

    from scipy.optimize import fsolve
    (B_sol, O_sol), _, ier, _ = fsolve(equations, (1500, 49000), full_output=True)
    if ier == 1:
        return B_sol, O_sol
    else:
        raise ValueError("Calibration fit failed")

class CalibrationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLIR Calibration Curve Viewer with 2-Point Calibration")
        self.resize(1000, 600)

        self.layout = QHBoxLayout(self)

        # --- Plotting Area ---
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.layout.addWidget(self.canvas)

        # --- Control Panel ---
        controls = QWidget()
        control_layout = QGridLayout(controls)

        self.params = {
            "R":    {"val": R,        "min": 1000,   "max": 1_000_000, "scale": 1},
            "B":    {"val": 1597.67, "min": 500,    "max": 5000,      "scale": 100},
            "F":    {"val": F,        "min": 1,      "max": 1000,      "scale": 100},
            "O":    {"val": 49800,   "min": 0,      "max": 100000,    "scale": 1},
        }

        self.sliders = {}
        self.labels = {}

        for i, (label, cfg) in enumerate(self.params.items()):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(cfg["min"])
            slider.setMaximum(cfg["max"])
            slider.setValue(int(cfg["val"] * cfg["scale"]))
            slider.valueChanged.connect(self.update_plot)

            val_label = QLabel(f"{cfg['val']:.4f}")
            self.sliders[label] = slider
            self.labels[label] = val_label

            control_layout.addWidget(QLabel(label), i, 0)
            control_layout.addWidget(slider, i, 1)
            control_layout.addWidget(val_label, i, 2)

        # --- Calibrate Button ---
        self.cal_btn = QPushButton("Run Two-Point Calibration")
        self.cal_btn.clicked.connect(self.run_two_point_calibration)
        control_layout.addWidget(self.cal_btn, len(self.params), 0, 1, 3)

        self.layout.addWidget(controls)
        self.update_plot()

    def get_param(self, key):
        slider = self.sliders[key]
        scale = self.params[key]["scale"]
        return slider.value() / scale

    def update_plot(self):
        R = self.get_param("R")
        B = self.get_param("B")
        F = self.get_param("F")
        O = self.get_param("O")

        for k in self.params:
            self.labels[k].setText(f"{self.get_param(k):.4f}")

        raw_vals = np.linspace(O + 1, O + 50000, 1000)
        kelvin_vals = raw_to_kelvin(raw_vals, R, B, F, O)
        corrected_K = corrected_temperature_K(kelvin_vals, EMISSIVITY, REFLECTED_TEMP_K)
        fahrenheit_vals = kelvin_to_fahrenheit(corrected_K)

        self.ax.clear()
        self.ax.plot(raw_vals, fahrenheit_vals, label="Raw → °F")
        self.ax.set_xlabel("Raw Sensor Value")
        self.ax.set_ylabel("Temperature (°F)")
        self.ax.set_title("Calibration Curve")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def run_two_point_calibration(self):
        print("Opening camera feed for calibration...")

        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        if cam_list.GetSize() == 0:
            print("No FLIR camera found.")
            return

        cam = cam_list[0]
        cam.Init()
        cam.BeginAcquisition()

        cv2.namedWindow("Calibration View")

        rois = []
        raw_vals = []
        temps = []

        def get_rect_roi():
            ret, frame = cam.GetNextImage().GetNDArray(), None
            vis = cv2.normalize(ret, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
            r = cv2.selectROI("Calibration View", vis, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("ROI selector")
            if r[2] == 0 or r[3] == 0:
                return None, None
            raw = ret[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            return np.mean(raw), r

        try:
            for i in range(2):
                print(f"Select ROI for Point {i+1}...")
                while True:
                    img = cam.GetNextImage()
                    raw = img.GetNDArray()
                    vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
                    cv2.imshow("Calibration View", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # space to capture and draw ROI
                        # Freeze one frame
                        frozen_img = raw.copy()
                        vis_img = cv2.applyColorMap(cv2.normalize(frozen_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_INFERNO)
                        roi = cv2.selectROI("Select ROI", vis_img, fromCenter=False)
                        cv2.destroyWindow("Select ROI")
                        if roi[2] > 0 and roi[3] > 0:
                            roi_data = frozen_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                            mean_raw = np.mean(roi_data)
                            temp, ok = QInputDialog.getDouble(self, f"Point {i+1}", "Enter actual temperature (°F):", 100.0)
                            if ok:
                                raw_vals.append(mean_raw)
                                temps.append(fahrenheit_to_kelvin(temp))
                                break
                    elif key == ord('q'):
                        print("Calibration cancelled.")
                        return

            B_new, O_new = solve_two_point_calibration(raw_vals[0], temps[0], raw_vals[1], temps[1])
            print(f"✅ New B: {B_new:.2f}, New O: {O_new:.2f}")
            self.sliders["B"].setValue(int(B_new * self.params["B"]["scale"]))
            self.sliders["O"].setValue(int(O_new * self.params["O"]["scale"]))
            self.update_plot()

        finally:
            cam.EndAcquisition()
            cam.DeInit()
            del cam
            cam_list.Clear()
            system.ReleaseInstance()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationGUI()
    window.show()
    sys.exit(app.exec())
