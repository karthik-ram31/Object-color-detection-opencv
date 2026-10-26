import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os

# HSV ranges are starting points; lighting and camera settings affect results.
COLOR_RANGES = {
    "Red": [
        ((0, 100, 70), (10, 255, 255)),
        ((170, 100, 70), (179, 255, 255)),
    ],
    "Orange": [((11, 100, 80), (24, 255, 255))],
    "Yellow": [((25, 90, 90), (35, 255, 255))],
    "Green": [((36, 65, 55), (85, 255, 255))],
    "Blue": [((90, 70, 50), (130, 255, 255))],
    "Purple": [((131, 55, 45), (169, 255, 255))],
}

BOX_COLORS = {
    "Red": (0, 0, 255),
    "Orange": (0, 140, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 200, 0),
    "Blue": (255, 100, 0),
    "Purple": (200, 0, 200),
}


class ObjectColorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Color Detection | OpenCV Project")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)
        self.root.configure(bg="#eef3f8")

        self.cap = None
        self.running = False
        self.after_id = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.latest_counts = Counter()
        self.latest_areas = Counter()
        self.latest_annotated_frame = None
        self.min_area = tk.IntVar(value=900)
        self.show_masks = tk.BooleanVar(value=False)
        self.selected_colors = {
            name: tk.BooleanVar(value=True)
            for name in COLOR_RANGES
        }

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#12233f", padx=18, pady=14)
        header.pack(fill="x")
        tk.Label(
            header, text="OBJECT COLOR DETECTION",
            font=("Segoe UI", 19, "bold"), fg="white", bg="#12233f"
        ).pack(anchor="w")
        tk.Label(
            header, text="Real-time computer vision using Python, OpenCV and HSV segmentation",
            font=("Segoe UI", 10), fg="#cbd8ec", bg="#12233f"
        ).pack(anchor="w", pady=(3, 0))

        main = tk.Frame(self.root, bg="#eef3f8", padx=14, pady=14)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg="#eef3f8")
        left.pack(side="left", fill="both", expand=True)

        video_card = tk.Frame(left, bg="#101820", highlightbackground="#cbd5e1", highlightthickness=1)
        video_card.pack(fill="both", expand=True)
        self.video_label = tk.Label(
            video_card, text="Camera is stopped\n\nClick  START CAMERA  to begin",
            font=("Segoe UI", 15), fg="#dbeafe", bg="#101820", justify="center"
        )
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        controls = tk.Frame(left, bg="#eef3f8", pady=10)
        controls.pack(fill="x")
        self.start_btn = ttk.Button(controls, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side="left", padx=(0, 7))
        self.stop_btn = ttk.Button(controls, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(side="left", padx=7)
        ttk.Button(controls, text="Save Screenshot", command=self.save_screenshot).pack(side="left", padx=7)
        ttk.Button(controls, text="Export Statistics", command=self.export_stats).pack(side="left", padx=7)

        right = tk.Frame(main, bg="white", width=290, padx=14, pady=12,
                         highlightbackground="#d6deea", highlightthickness=1)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)

        tk.Label(right, text="DETECTION SETTINGS", font=("Segoe UI", 11, "bold"),
                 bg="white", fg="#12233f").pack(anchor="w")
        tk.Label(right, text="Select colors", font=("Segoe UI", 9, "bold"),
                 bg="white", fg="#475569").pack(anchor="w", pady=(12, 4))
        for name, var in self.selected_colors.items():
            row = tk.Frame(right, bg="white")
            row.pack(fill="x", pady=1)
            swatch = tk.Canvas(row, width=15, height=15, bg="white", highlightthickness=0)
            swatch.pack(side="left", padx=(0, 7))
            bgr = BOX_COLORS[name]
            swatch.create_rectangle(1, 1, 14, 14, fill="#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0]), outline="")
            ttk.Checkbutton(row, text=name, variable=var).pack(side="left")

        tk.Label(right, text="Minimum object area (noise filter)", font=("Segoe UI", 9, "bold"),
                 bg="white", fg="#475569").pack(anchor="w", pady=(14, 2))
        self.area_value = tk.Label(right, text="900 px²", font=("Segoe UI", 9),
                                   bg="white", fg="#12233f")
        self.area_value.pack(anchor="w")
        ttk.Scale(right, from_=200, to=5000, variable=self.min_area,
                  command=self._update_area_label).pack(fill="x", pady=(2, 10))

        tk.Label(right, text="LIVE STATISTICS", font=("Segoe UI", 11, "bold"),
                 bg="white", fg="#12233f").pack(anchor="w", pady=(8, 8))
        self.total_label = tk.Label(right, text="Objects detected: 0", font=("Segoe UI", 10, "bold"),
                                    bg="white", fg="#0f766e")
        self.total_label.pack(anchor="w", pady=2)
        self.fps_label = tk.Label(right, text="FPS: 0.0", font=("Segoe UI", 9),
                                  bg="white", fg="#475569")
        self.fps_label.pack(anchor="w", pady=2)

        self.stats_labels = {}
        for name in COLOR_RANGES:
            label = tk.Label(right, text=f"{name}: 0", font=("Segoe UI", 9),
                             bg="white", fg="#334155")
            label.pack(anchor="w", pady=1)
            self.stats_labels[name] = label

        tk.Label(right, text="COLOR COUNT CHART", font=("Segoe UI", 10, "bold"),
                 bg="white", fg="#12233f").pack(anchor="w", pady=(14, 2))
        self.fig, self.ax = plt.subplots(figsize=(3.0, 2.1), dpi=85)
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.chart_canvas.get_tk_widget().pack(fill="x", pady=(0, 2))
        self._update_chart(Counter())

        footer = tk.Label(
            self.root,
            text="Tip: use matte, brightly colored objects and steady lighting. Press Q in the video window or Stop Camera to stop.",
            font=("Segoe UI", 9), bg="#eef3f8", fg="#64748b", pady=5
        )
        footer.pack(fill="x")

    def _update_area_label(self, _=None):
        self.area_value.config(text=f"{int(self.min_area.get())} px²")

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            messagebox.showerror(
                "Camera error",
                "Could not open the webcam. Close other apps using the camera and check Windows camera permissions."
            )
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.last_fps_time = time.time()
        self.frame_count = 0
        self._process_frame()

    def stop_camera(self):
        self.running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.video_label.config(
            image="", text="Camera is stopped\n\nClick  START CAMERA  to begin",
            fg="#dbeafe", bg="#101820"
        )

    def _process_frame(self):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            messagebox.showerror("Camera error", "The webcam stopped sending frames.")
            return

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        counts = Counter()
        areas = Counter()
        min_area = int(self.min_area.get())

        for color_name, ranges in COLOR_RANGES.items():
            if not self.selected_colors[color_name].get():
                continue

            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

            # Reduce small specks and smooth edges.
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLORS[color_name], 2)
                text = f"{color_name} | {int(area)} px"
                text_y = y - 10 if y > 25 else y + 22
                cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, BOX_COLORS[color_name], 2, cv2.LINE_AA)
                counts[color_name] += 1
                areas[color_name] += int(area)

        self.latest_counts = counts
        self.latest_areas = areas
        self.latest_annotated_frame = frame.copy()
        self._update_stats(counts)

        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((850, 570), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
        self.after_id = self.root.after(15, self._process_frame)

    def _update_stats(self, counts):
        total = sum(counts.values())
        self.total_label.config(text=f"Objects detected: {total}")
        for name, label in self.stats_labels.items():
            label.config(text=f"{name}: {counts.get(name, 0)}")
        self._update_chart(counts)

    def _update_chart(self, counts):
        self.ax.clear()
        names = list(COLOR_RANGES.keys())
        values = [counts.get(name, 0) for name in names]
        colors = ["red", "darkorange", "gold", "green", "royalblue", "purple"]
        self.ax.bar(names, values, color=colors)
        self.ax.set_ylim(0, max(3, max(values, default=0) + 1))
        self.ax.set_ylabel("Count", fontsize=8)
        self.ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        self.ax.tick_params(axis="y", labelsize=7)
        self.ax.grid(axis="y", alpha=0.2)
        self.fig.tight_layout()
        self.chart_canvas.draw_idle()

    def save_screenshot(self):
        if self.cap is None or not self.running:
            messagebox.showinfo("Camera stopped", "Start the camera before saving a screenshot.")
            return
        if self.latest_annotated_frame is None:
            messagebox.showinfo("No frame yet", "Wait for the first camera frame, then save the screenshot.")
            return
        filename = time.strftime("object_color_detection_%Y%m%d_%H%M%S.png")
        ok = cv2.imwrite(filename, self.latest_annotated_frame)
        if ok:
            messagebox.showinfo("Screenshot saved", f"Saved to the project folder:\n{os.path.abspath(filename)}")
        else:
            messagebox.showerror("Screenshot error", "Could not save the image.")

    def export_stats(self):
        filename = time.strftime("color_statistics_%Y%m%d_%H%M%S.csv")
        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write("Color,Object Count,Total Detected Area (pixels)\n")
                for name in COLOR_RANGES:
                    file.write(f"{name},{self.latest_counts.get(name, 0)},{self.latest_areas.get(name, 0)}\n")
                file.write(f"TOTAL,{sum(self.latest_counts.values())},{sum(self.latest_areas.values())}\n")
            messagebox.showinfo("Statistics exported", f"Saved to:\n{os.path.abspath(filename)}")
        except OSError as exc:
            messagebox.showerror("Export error", str(exc))

    def close_app(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectColorDetectionApp(root)
    root.mainloop()
