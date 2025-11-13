import os
import cv2
import torch
import uuid
from datetime import timedelta
from ultralytics import YOLO


def _secs_to_hhmmss(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    h, m = divmod(total_seconds, 3600)
    m, s = divmod(m, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def run_object_detection(
    video_path: str,
    outdir: str,
    model_name: str = "yolov8n.pt",
    conf_thresh: float = 0.35,
    frame_step_ms: int = 1000
):
    """
    Run YOLOv8 object detection on a video.
    Flags electronic gadgets like phones or laptops, and saves screenshots + crops.
    Returns (logs, total_flags)
    """

    print("[INFO] Loading YOLOv8 model...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps
    frame_step = int((frame_step_ms / 1000.0) * fps)

    base_dir = os.path.join(outdir, "gadget_detection")
    fullshot_dir = os.path.join(base_dir, "screenshots")
    crop_dir = os.path.join(base_dir, "crops")
    os.makedirs(fullshot_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    logs = []
    flag_count = 0
    frame_idx = 0

    print("[STEP] Scanning frames for electronic gadgets...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp_s = frame_idx / fps
        timestamp_h = _secs_to_hhmmss(timestamp_s)

        results = model(frame, verbose=False, conf=conf_thresh)
        detections = results[0].boxes if len(results) > 0 else None

        if detections is not None and len(detections) > 0:
            for det in detections:
                cls = int(det.cls)
                conf = float(det.conf)
                label = model.names.get(cls, "unknown").lower()

                # Restrict to gadgets of interest
                if any(word in label for word in ["cell", "phone", "laptop", "mobile", "tablet"]):
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    timestamp_s_end = timestamp_s + (frame_step_ms / 1000.0)
                    duration_s = round(timestamp_s_end - timestamp_s, 2)
                    if duration_s <= 0:
                        duration_s = frame_step_ms / 1000.0  # fallback

                    flag_count += 1
                    timestamp_safe = timestamp_h.replace(":", "-").replace(".", "_")

                    # Save full frame screenshot
                    shot_name = f"flag_{flag_count}_{label}_{timestamp_safe}.jpg"
                    shot_path = os.path.join(fullshot_dir, shot_name)
                    cv2.imwrite(shot_path, frame)

                    # Save cropped detection
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size == 0:
                        crop_img = frame  # fallback to full frame

                    crop_name = f"crop_{flag_count}_{label}_{timestamp_safe}.jpg"
                    crop_path = os.path.join(crop_dir, crop_name)
                    cv2.imwrite(crop_path, crop_img)

                    logs.append({
                        "start": timestamp_h,
                        "end": _secs_to_hhmmss(timestamp_s_end),
                        "duration": duration_s,
                        "type": label,
                        "confidence": round(conf, 2),
                        "screenshot": shot_path,
                        "crop": crop_path
                    })

        frame_idx += 1

    cap.release()

    print(f"[INFO] Object detection done: {flag_count} gadget frames flagged.")
    print(f"[INFO] Proofs saved to: {base_dir}")

    return logs, flag_count


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Run YOLO gadget detection")
    parser.add_argument("video", help="path to video file")
    parser.add_argument("--outdir", default="reports/test_gadgets", help="output directory")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--frame-step-ms", type=int, default=1000, help="frame step in ms")
    args = parser.parse_args()

    logs, count = run_object_detection(
        args.video, args.outdir,
        model_name=args.model,
        conf_thresh=args.conf,
        frame_step_ms=args.frame_step_ms
    )

    print(json.dumps(logs, indent=2))
    print(f"Total flagged gadget frames: {count}")
