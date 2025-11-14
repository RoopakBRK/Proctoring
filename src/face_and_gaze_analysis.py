import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from datetime import timedelta
from typing import List, Dict, Any
from collections import deque

# Landmark indices used for gaze (MediaPipe FaceMesh)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]


def _avg_x(landmarks, indices):
    return sum(landmarks[i].x for i in indices) / len(indices)


def _avg_y(landmarks, indices):
    return sum(landmarks[i].y for i in indices) / len(indices)


def _secs_to_hhmmss_ms(s: float) -> str:
    td = timedelta(seconds=s)
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _is_gaze_centered(landmarks, horiz_thresh=0.02, vert_thresh=0.04) -> bool:
    """Return True if gaze appears roughly centered."""
    lx = _avg_x(landmarks, LEFT_IRIS)
    rx = _avg_x(landmarks, RIGHT_IRIS)
    ly = _avg_y(landmarks, LEFT_IRIS)
    ry = _avg_y(landmarks, RIGHT_IRIS)

    left_mid_x = (_avg_x(landmarks, [LEFT_EYE_CORNERS[0]]) + _avg_x(landmarks, [LEFT_EYE_CORNERS[1]])) / 2.0
    right_mid_x = (_avg_x(landmarks, [RIGHT_EYE_CORNERS[0]]) + _avg_x(landmarks, [RIGHT_EYE_CORNERS[1]])) / 2.0
    left_mid_y = (_avg_y(landmarks, [LEFT_EYE_CORNERS[0]]) + _avg_y(landmarks, [LEFT_EYE_CORNERS[1]])) / 2.0
    
    # --- THIS IS THE CORRECTED LINE ---
    right_mid_y = (_avg_y(landmarks, [RIGHT_EYE_CORNERS[0]]) + _avg_y(landmarks, [RIGHT_EYE_CORNERS[1]])) / 2.0
    # --- END CORRECTION ---

    left_eye_w = abs(_avg_x(landmarks, [LEFT_EYE_CORNERS[0]]) - _avg_x(landmarks, [LEFT_EYE_CORNERS[1]]))
    right_eye_w = abs(_avg_x(landmarks, [RIGHT_EYE_CORNERS[0]]) - _avg_x(landmarks, [RIGHT_EYE_CORNERS[1]]))
    eye_w = max(1e-6, (left_eye_w + right_eye_w) / 2.0)

    horiz_off = max(abs(lx - left_mid_x), abs(rx - right_mid_x)) / eye_w
    vert_off = max(abs(ly - left_mid_y), abs(ry - right_mid_y)) / eye_w

    return (horiz_off < horiz_thresh) and (vert_off < vert_thresh)


def analyze_face_and_gaze(video_path: str,
                         gaze_frame_step: int, # <-- Renamed for clarity
                         horiz_threshold: float = 0.02,
                         vert_threshold: float = 0.04,
                         consecutive_guard: int = 2,
                         outdir: str = "reports/tmp") -> Dict[str, Any]:
    """
    Analyze face presence and gaze in a video.
    Gaze is analyzed every 'gaze_frame_step' frames.
    Proof images are saved at most once per second.
    """

    os.makedirs(outdir, exist_ok=True)
    proof_dir = os.path.join(outdir, "face_proofs")
    os.makedirs(proof_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # --- New: Define screenshot interval ---
    screenshot_interval_s = 1.0  # Once per second
    last_screenshot_time_s = -1.0 # Tracks time of last saved proof image

    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    no_face_frames = 0
    multiple_faces_frames = 0
    single_face_frames = 0
    looking_away_frames = 0
    face_flag_logs = []

    away_buffer = 0
    processed_samples = 0
    gaze_buffer = deque(maxlen=3)

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        static_image_mode=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detector:

        frame_idx = 0
        estimated_samples = max(1, total_frames // max(1, gaze_frame_step))
        pbar = tqdm(total=estimated_samples, desc="Face+Gaze", unit="samples")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- Main loop is controlled by gaze_frame_step ---
            if frame_idx % gaze_frame_step != 0:
                frame_idx += 1
                continue

            processed_samples += 1
            pbar.update(1)
            timestamp_s = frame_idx / fps
            timestamp_str = _secs_to_hhmmss_ms(timestamp_s)

            # --- New: Check if it's time to save a screenshot ---
            time_since_last_shot = timestamp_s - last_screenshot_time_s
            is_screenshot_time = (time_since_last_shot >= screenshot_interval_s)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det_res = face_detector.process(rgb)

            if det_res.detections:
                detections = [
                    d for d in det_res.detections
                    if d.score[0] > 0.55
                ]
            else:
                detections = []
            
            # --- No face ---
            if not detections:
                # Always count for accuracy (Req 3)
                no_face_frames += 1
                
                # Only save proof image once per second (Req 2)
                if is_screenshot_time:
                    last_screenshot_time_s = timestamp_s
                    proof_path = os.path.join(
                        proof_dir, f"noface_{timestamp_str.replace(':','-').replace('.','_')}.jpg"
                    )
                    cv2.imwrite(proof_path, frame)
                    face_flag_logs.append({
                        "timestamp": timestamp_str,
                        "reason": "No Face",
                        "proof_image": proof_path
                    })
                
                # Always reset gaze buffers (Req 3)
                away_buffer = 0
                gaze_buffer.clear() 
                frame_idx += 1
                continue

            face_count = len(detections)

            # --- Multiple faces ---
            if face_count > 1:
                # Always count for accuracy (Req 3)
                multiple_faces_frames += 1
                
                # Only save proof image once per second (Req 2)
                if is_screenshot_time:
                    last_screenshot_time_s = timestamp_s
                    proof_path = os.path.join(
                        proof_dir, f"multiface_{timestamp_str.replace(':','-').replace('.','_')}.jpg"
                    )
                    cv2.imwrite(proof_path, frame)
                    face_flag_logs.append({
                        "timestamp": timestamp_str,
                        "reason": f"Multiple Faces ({face_count})",
                        "proof_image": proof_path
                    })
                
                # Always reset gaze buffers (Req 3)
                away_buffer = 0
                gaze_buffer.clear()
                frame_idx += 1
                continue

            # --- Single face gaze tracking (Req 1 & 3) ---
            # (This section runs on every gaze_frame_step)
            single_face_frames += 1
            mesh = face_mesh.process(rgb)
            multi = getattr(mesh, "multi_face_landmarks", None)
            if not multi:
                away_buffer = 0
                gaze_buffer.clear()
                frame_idx += 1
                continue

            landmarks = multi[0].landmark
            centered = _is_gaze_centered(landmarks, horiz_thresh=horiz_threshold, vert_thresh=vert_threshold)

            gaze_buffer.append(centered)

            if sum(gaze_buffer) <= 1:
                away_buffer += 1
            else:
                away_buffer = 0

            if away_buffer >= consecutive_guard:
                looking_away_frames += away_buffer
                away_buffer = 0

            frame_idx += 1

        pbar.close()
        cap.release()

    # ---- Gaze accuracy ----
    gaze_accuracy = 0.0
    if processed_samples > 0:
        gaze_accuracy = round(100 * (1 - (looking_away_frames / processed_samples)), 2)
    else:
        gaze_accuracy = 0.0

    summary = {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "gaze_frame_step": gaze_frame_step,
        "no_face_frames": no_face_frames,
        "multiple_faces_frames": multiple_faces_frames,
        "single_face_frames": single_face_frames,
        "looking_away_frames": looking_away_frames,
        "gaze_accuracy": f"{gaze_accuracy:.1f}%",
        "face_flag_logs": face_flag_logs,
        "processed_samples": processed_samples
    }

    return summary


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Test face & gaze analysis")
    parser.add_argument("video", help="path to video file")
    # --- Updated argument name for clarity ---
    parser.add_argument("--gaze-frame-step", type=int, default=2, help="Sample every Nth frame for gaze analysis")
    args = parser.parse_args()

    res = analyze_face_and_gaze(args.video, gaze_frame_step=args.gaze_frame_step)
    print(json.dumps(res, indent=2))