import threading
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from torchvision import ops
from ultralytics import YOLO


# ------------------- NMS Function -------------------
def apply_classwise_nms(boxes, scores, classes, iou_threshold=0.5):
    # boxes: Nx4 numpy, scores: N, classes: N
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    final_boxes, final_scores, final_classes = [], [], []
    for cls in np.unique(classes):
        cls_indices = np.where(classes == cls)[0]
        if len(cls_indices) == 0:
            continue
        cls_boxes = torch.tensor(boxes[cls_indices], dtype=torch.float32)
        cls_scores = torch.tensor(scores[cls_indices], dtype=torch.float32)
        try:
            keep = ops.nms(cls_boxes, cls_scores, iou_threshold)
        except Exception:
            # fallback: keep all
            keep = torch.arange(len(cls_indices))
        if len(keep) == 0:
            continue
        final_boxes.append(cls_boxes[keep].numpy())
        final_scores.append(cls_scores[keep].numpy())
        final_classes.append(np.full(len(keep), cls))
    if len(final_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    return (
        np.concatenate(final_boxes, axis=0),
        np.concatenate(final_scores, axis=0),
        np.concatenate(final_classes, axis=0),
    )


# ------------------- Model Setup -------------------
# Adjust these paths if your weights are at different locations
PERSON_MODEL_PATH = "yolov8l.pt"  # change to yolov8n.pt or yolov8s.pt if you need speed
PPE_MODEL_PATH = "yolov8 70 epoch/weights/best.pt"  # your trained PPE weights

# Load models once (may take a while on first run)
try:
    person_model = YOLO(PERSON_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed loading person model ({PERSON_MODEL_PATH}): {e}")

try:
    ppe_model = YOLO(PPE_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed loading PPE model ({PPE_MODEL_PATH}): {e}")

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------- Helper -------------------
def is_inside(cx, cy, box):
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


# ------------------- Processing (single image/frame) -------------------
def process_frame(frame_bgr):
    """Input: BGR numpy image (from cv2) Returns: person_img, ppe_img, final_img, safe_img (all BGR numpy arrays).
    """
    image = frame_bgr.copy()
    _h, _w = image.shape[:2]

    # Person detection
    try:
        person_result = person_model(image)[0]
        # YOLO's boxes: person_result.boxes.xyxy, .cls, .conf
        if hasattr(person_result, "boxes") and len(person_result.boxes) > 0:
            cls_arr = person_result.boxes.cls.cpu().numpy()
            mask = cls_arr == 0  # class 0 = person (common)
            if mask.sum() > 0:
                person_boxes = person_result.boxes.xyxy[mask].cpu().numpy()
            else:
                person_boxes = np.zeros((0, 4))
        else:
            person_boxes = np.zeros((0, 4))
    except Exception:
        person_boxes = np.zeros((0, 4))

    person_img = image.copy()
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(person_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # PPE detection
    try:
        ppe_result = ppe_model(image)[0]
        if hasattr(ppe_result, "boxes") and len(ppe_result.boxes) > 0:
            ppe_boxes_all = ppe_result.boxes.xyxy.cpu().numpy()
            ppe_classes_all = ppe_result.boxes.cls.cpu().numpy().astype(np.int32)
            ppe_scores_all = ppe_result.boxes.conf.cpu().numpy()
        else:
            ppe_boxes_all = np.zeros((0, 4))
            ppe_classes_all = np.zeros((0,), dtype=int)
            ppe_scores_all = np.zeros((0,))
    except Exception:
        ppe_boxes_all = np.zeros((0, 4))
        ppe_classes_all = np.zeros((0,), dtype=int)
        ppe_scores_all = np.zeros((0,))

    # class-wise NMS
    ppe_boxes, _ppe_scores, ppe_classes = apply_classwise_nms(ppe_boxes_all, ppe_scores_all, ppe_classes_all)

    ppe_img = image.copy()
    for i, box in enumerate(ppe_boxes):
        x1, y1, x2, y2 = map(int, box)
        cls = int(ppe_classes[i]) if len(ppe_classes) > 0 else -1
        label = "Helmet" if cls == 0 else ("Vest" if cls == 1 else f"cls{cls}")
        color = (0, 255, 255) if cls == 0 else (255, 255, 0)
        cv2.rectangle(ppe_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(ppe_img, label, (x1, max(15, y1 - 5)), FONT, 0.6, color, 2)

    # Assign PPE to each person
    final_img = image.copy()
    safe_img = image.copy()
    person_labels = [["NH", "NV"] for _ in range(len(person_boxes))]  # [Helmet, Vest] defaults no

    for j, ppe_box in enumerate(ppe_boxes):
        cx = (ppe_box[0] + ppe_box[2]) / 2.0
        cy = (ppe_box[1] + ppe_box[3]) / 2.0
        cls = int(ppe_classes[j]) if len(ppe_classes) > 0 else -1
        for i, person_box in enumerate(person_boxes):
            if is_inside(cx, cy, person_box):
                if cls == 0:
                    person_labels[i][0] = "H"
                elif cls == 1:
                    person_labels[i][1] = "V"

    # Draw results
    for i, person_box in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, person_box)
        label = f"{person_labels[i][0]} {person_labels[i][1]}"

        # Map colors for labels
        color_map = {
            "H V": (0, 255, 0),  # Green
            "H NV": (0, 255, 255),  # Yellow
            "NH V": (255, 0, 0),  # Blue
            "NH NV": (0, 0, 255),  # Red
        }
        color = color_map.get(label, (255, 255, 255))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(final_img, label, (x1, max(20, y1 - 10)), FONT, 0.7, color, 2)

        safety_label = "SAFE" if label == "H V" else "UNSAFE"
        safety_color = (0, 255, 0) if label == "H V" else (0, 0, 255)
        cv2.rectangle(safe_img, (x1, y1), (x2, y2), safety_color, 2)
        cv2.putText(safe_img, safety_label, (x1, max(35, y1 - 10)), FONT, 0.7, safety_color, 2)

    return person_img, ppe_img, final_img, safe_img


# ------------------- GUI Setup & Utilities -------------------
def display_images_in_labels(images):
    """Update the four GUI image labels with BGR numpy images."""
    for i, img in enumerate(images):
        if img is None:
            continue
        # convert BGR -> RGB -> PIL -> ImageTk
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((380, 280))  # keep size reasonable
        img_tk = ImageTk.PhotoImage(pil)
        image_labels[i].config(image=img_tk)
        image_labels[i].image = img_tk


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
    if not file_path:
        return
    frame = cv2.imread(file_path)
    if frame is None:
        return
    person_img, ppe_img, final_img, safe_img = process_frame(frame)
    display_images_in_labels([person_img, ppe_img, final_img, safe_img])


# ------------------- Camera Thread & Controls -------------------
camera_running = False
camera_thread = None
cap = None


def camera_loop(camera_index=0, target_fps=10):
    global camera_running, cap
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        status_var.set("Camera open failed.")
        camera_running = False
        return
    status_var.set("Camera running...")
    prev = 0
    period = 1.0 / target_fps
    try:
        while camera_running:
            t = time.time()
            if t - prev < period:
                time.sleep(max(0, period - (t - prev)))
            prev = time.time()
            ret, frame = cap.read()
            if not ret:
                continue
            # Resize for faster inference (optional)
            # Keep original aspect ratio but limit size
            H, W = frame.shape[:2]
            max_dim = 800
            if max(H, W) > max_dim:
                scale = max_dim / float(max(H, W))
                frame = cv2.resize(frame, (int(W * scale), int(H * scale)))

            person_img, ppe_img, final_img, safe_img = process_frame(frame)
            # update GUI (must be done in main thread via after)
            root.after(0, display_images_in_labels, [person_img, ppe_img, final_img, safe_img])
    finally:
        if cap is not None:
            cap.release()
        root.after(0, lambda: status_var.set("Camera stopped."))


def start_camera():
    global camera_running, camera_thread
    if camera_running:
        return
    camera_running = True
    camera_thread = threading.Thread(target=camera_loop, args=(0, 6), daemon=True)  # target fps ~6
    camera_thread.start()
    status_var.set("Starting camera...")


def stop_camera():
    global camera_running
    camera_running = False
    status_var.set("Stopping camera...")


# ------------------- Dark Theme GUI -------------------
root = tk.Tk()
root.title("PPE Detection and Safety Classification (Live Camera)")
root.configure(bg="#161616")

title = tk.Label(
    root, text="PPE Detection and Safety Classification", bg="#161616", fg="#00ffcc", font=("Segoe UI", 18, "bold")
)
title.pack(pady=8)

btn_frame = tk.Frame(root, bg="#161616")
btn_frame.pack(pady=6)

upload_btn = tk.Button(
    btn_frame,
    text="Upload Image",
    command=upload_image,
    bg="#00bfa5",
    fg="black",
    font=("Segoe UI", 11, "bold"),
    width=16,
)
upload_btn.grid(row=0, column=0, padx=8)

start_cam_btn = tk.Button(
    btn_frame,
    text="Start Camera (Live)",
    command=start_camera,
    bg="#00bfa5",
    fg="black",
    font=("Segoe UI", 11, "bold"),
    width=16,
)
start_cam_btn.grid(row=0, column=1, padx=8)

stop_cam_btn = tk.Button(
    btn_frame,
    text="Stop Camera",
    command=stop_camera,
    bg="#ff6b6b",
    fg="black",
    font=("Segoe UI", 11, "bold"),
    width=16,
)
stop_cam_btn.grid(row=0, column=2, padx=8)

# status label
status_var = tk.StringVar(value="Idle")
status_label = tk.Label(root, textvariable=status_var, bg="#161616", fg="#ffffff", font=("Segoe UI", 10))
status_label.pack(pady=4)

# image display frame
frame = tk.Frame(root, bg="#161616")
frame.pack(pady=10)

image_labels = []
captions = ["Person Detection", "PPE Detection", "Labeled Output", "Safe/Unsafe"]
for i in range(4):
    sub = tk.Frame(frame, bg="#1f1f1f", bd=0)
    sub.grid(row=(i // 2) * 2, column=i % 2, padx=12, pady=6)
    lbl = tk.Label(sub, bg="#1f1f1f")
    lbl.pack()
    caption = tk.Label(frame, text=captions[i], bg="#161616", fg="#00ffcc", font=("Segoe UI", 10, "bold"))
    caption.grid(row=(i // 2) * 2 + 1, column=i % 2, pady=(2, 12))
    image_labels.append(lbl)

# footer / credits
credit = tk.Label(
    root,
    text="Model inference: YOLO (person + PPE). Live camera mode available.",
    bg="#161616",
    fg="#8fbfb3",
    font=("Segoe UI", 9),
)
credit.pack(pady=8)


# Graceful exit: ensure camera stops
def on_closing():
    global camera_running
    camera_running = False
    time.sleep(0.1)
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
