# src/voice_diarization.py
"""
Speaker Diarization + Audio Snippet Extraction (Fixed Version)
--------------------------------------------------------------
- Identifies non-primary speakers (suspicious voices)
- Saves proof .wav segments with unique filenames
- Adds robust error handling and detailed logs
"""

import os
import json
import uuid
import torch
from datetime import timedelta
from pydub import AudioSegment
from pyannote.audio import Pipeline


# -------------------------------------------------------------
# Extract audio snippet safely
# -------------------------------------------------------------
def extract_audio_segment(input_audio_path, start_s, end_s, output_path):
    """Extracts an audio segment from a WAV file using pydub."""
    try:
        audio = AudioSegment.from_wav(input_audio_path)
        start_ms = int(round(start_s * 1000))
        end_ms = int(round(end_s * 1000))

        if end_ms <= start_ms:
            print(f"[WARN] Invalid segment times: start={start_s:.2f}, end={end_s:.2f}")
            return False

        segment = audio[start_ms:end_ms]
        if len(segment) == 0:
            print(f"[WARN] Empty audio segment for {output_path}")
            return False

        segment.export(output_path, format="wav")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            print(f"[WARN] Segment export failed for {output_path}")
            return False

    except Exception as e:
        print(f"[WARN] Could not extract segment {start_s:.2f}-{end_s:.2f}s: {e}")
        return False


# -------------------------------------------------------------
# Main diarization pipeline
# -------------------------------------------------------------
def run_diarization_and_extract_snippets(
    audio_path: str,
    outdir: str,
    hf_token: str = None,
    min_flag_duration: float = 1.5
):
    """
    Performs Pyannote speaker diarization.
    Flags all non-primary speakers with duration >= min_flag_duration.
    Exports proof .wav segments for each flagged segment.
    Returns: (logs, total_flags)
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("[INFO] Loading Pyannote diarization pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
    except Exception as e:
        print(f"[WARN] Could not load Pyannote pipeline: {e}")
        print("[WARN] Skipping voice diarization.")
        return [], 0

    print("[STEP] Running Voice Diarization...")
    diarization = pipeline(audio_path)

    # Get total audio duration
    audio = AudioSegment.from_wav(audio_path)
    audio_duration = len(audio) / 1000.0  # ms → s

    # Group by speaker
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = float(turn.end - turn.start)
        speaker_segments.setdefault(speaker, []).append({
            "start": float(turn.start),
            "end": float(turn.end),
            "duration": duration
        })

    if not speaker_segments:
        print("[WARN] No speaker activity detected.")
        return [], 0

    # Determine primary speaker (most total speaking time)
    speaker_durations = {
        spk: sum(seg["duration"] for seg in segs)
        for spk, segs in speaker_segments.items()
    }
    primary_speaker = max(speaker_durations, key=speaker_durations.get)

    # Prepare proof output folder
    proof_dir = os.path.join(outdir, "voice_proofs")
    os.makedirs(proof_dir, exist_ok=True)

    logs = []
    total_flags = 0

    print(f"[INFO] Primary speaker identified: {primary_speaker}")

    # ---------------------------------------------------------
    # Process each speaker segment
    # ---------------------------------------------------------
    for speaker, segments in speaker_segments.items():
        for seg in segments:
            start_s = seg["start"]
            end_s = seg["end"]
            duration = seg["duration"]
            start_hhmmss = str(timedelta(seconds=start_s))[:-3]
            end_hhmmss = str(timedelta(seconds=end_s))[:-3]

            is_flagged = (speaker != primary_speaker and duration >= min_flag_duration)
            proof_path = None

            if is_flagged:
                safe_start = f"{start_s:.3f}".replace(".", "_")
                safe_end = f"{end_s:.3f}".replace(".", "_")
                uid = uuid.uuid4().hex[:6]
                proof_filename = f"{speaker}_from_{safe_start}s_to_{safe_end}s_{uid}.wav"
                proof_path = os.path.join(proof_dir, proof_filename)

                saved = extract_audio_segment(audio_path, start_s, end_s, proof_path)
                if saved:
                    print(f"[INFO] Saved proof: {proof_filename} ({duration:.2f}s)")
                    total_flags += 1
                else:
                    print(f"[WARN]  Failed to save proof for {speaker} ({start_s:.2f}-{end_s:.2f}s)")
                    proof_path = None

            # only store flagged segments (non-primary + long enough)
            if is_flagged:
                logs.append({
                    "speaker": speaker,
                    "start": start_hhmmss,
                    "end": end_hhmmss,
                    "duration": round(duration, 2),
                    "flagged": True,
                    "proof_audio": proof_path
                })


    # ---------------------------------------------------------
    # Save JSON log
    # ---------------------------------------------------------
    json_path = os.path.join(outdir, "voice_segments.json")
    with open(json_path, "w") as f:
        json.dump({
            "primary_speaker": primary_speaker,
            "audio_duration": round(audio_duration, 2),
            "segments": logs,
            "min_flag_duration": min_flag_duration
        }, f, indent=2)

    print(f"[INFO] Voice diarization found {len(speaker_segments)} unique speakers.")
    print(f"[INFO] Flagged {total_flags} suspicious segments (>= {min_flag_duration}s).")
    print(f"[INFO] Voice log saved: {json_path}")
    if total_flags > 0:
        print(f"[INFO] Proof audio clips saved under: {proof_dir}")

    return logs, total_flags


# -------------------------------------------------------------
# CLI for standalone testing
# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run voice diarization with proof export")
    parser.add_argument("audio", help="Path to input audio file (.wav)")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token")
    parser.add_argument("--outdir", default="reports/test", help="Output directory")
    args = parser.parse_args()

    logs, total_flags = run_diarization_and_extract_snippets(
        audio_path=args.audio,
        outdir=args.outdir,
        hf_token=args.hf_token
    )
    print(f" Total suspicious voices flagged: {total_flags}")


# Face DTtection

import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from datetime import timedelta
from typing import List, Dict, Any

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
    right_mid_y = (_avg_y(landmarks, [RIGHT_EYE_CORNERS[0]]) + _avg_y(landmarks, [RIGHT_EYE_CORNERS[1]])) / 2.0

    left_eye_w = abs(_avg_x(landmarks, [LEFT_EYE_CORNERS[0]]) - _avg_x(landmarks, [LEFT_EYE_CORNERS[1]]))
    right_eye_w = abs(_avg_x(landmarks, [RIGHT_EYE_CORNERS[0]]) - _avg_x(landmarks, [RIGHT_EYE_CORNERS[1]]))
    eye_w = max(1e-6, (left_eye_w + right_eye_w) / 2.0)

    horiz_off = max(abs(lx - left_mid_x), abs(rx - right_mid_x)) / eye_w
    vert_off = max(abs(ly - left_mid_y), abs(ry - right_mid_y)) / eye_w

    return (horiz_off < horiz_thresh) and (vert_off < vert_thresh)


def analyze_face_and_gaze(video_path: str,
                         frame_step,
                         horiz_threshold: float = 0.02,
                         vert_threshold: float = 0.04,
                         consecutive_guard: int = 2,
                         outdir: str = "reports/tmp") -> Dict[str, Any]:
    """
    Analyze face presence and gaze in a video.
    Also saves proof images for 'No Face' and 'Multiple Faces' events.
    """

    os.makedirs(outdir, exist_ok=True)
    proof_dir = os.path.join(outdir, "face_proofs")
    os.makedirs(proof_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = int(fps)

    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    no_face_frames = 0
    multiple_faces_frames = 0
    single_face_frames = 0
    looking_away_frames = 0
    face_flag_logs = []

    away_buffer = 0
    processed_samples = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detector:

        frame_idx = 0
        estimated_samples = max(1, total_frames // max(1, frame_step))
        pbar = tqdm(total=estimated_samples, desc="Face+Gaze", unit="samples")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            processed_samples += 1
            pbar.update(1)
            timestamp_s = frame_idx / fps
            timestamp_str = _secs_to_hhmmss_ms(timestamp_s)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det_res = face_detector.process(rgb)
            detections = getattr(det_res, "detections", None)

            # --- No face ---
            if not detections:
                no_face_frames += 1
                proof_path = os.path.join(
                    proof_dir, f"noface_{timestamp_str.replace(':','-').replace('.','_')}.jpg"
                )
                cv2.imwrite(proof_path, frame)
                face_flag_logs.append({
                    "timestamp": timestamp_str,
                    "reason": "No Face",
                    "proof_image": proof_path
                })
                away_buffer = 0
                frame_idx += 1
                continue

            face_count = len(detections)

            # --- Multiple faces ---
            if face_count > 1:
                multiple_faces_frames += 1
                proof_path = os.path.join(
                    proof_dir, f"multiface_{timestamp_str.replace(':','-').replace('.','_')}.jpg"
                )
                cv2.imwrite(proof_path, frame)
                face_flag_logs.append({
                    "timestamp": timestamp_str,
                    "reason": f"Multiple Faces ({face_count})",
                    "proof_image": proof_path
                })
                away_buffer = 0
                frame_idx += 1
                continue

            # --- Single face gaze tracking ---
            single_face_frames += 1
            mesh = face_mesh.process(rgb)
            multi = getattr(mesh, "multi_face_landmarks", None)
            if not multi:
                away_buffer = 0
                frame_idx += 1
                continue

            landmarks = multi[0].landmark
            centered = _is_gaze_centered(landmarks, horiz_thresh=horiz_threshold, vert_thresh=vert_threshold)
            if centered:
                away_buffer = 0
            else:
                away_buffer += 1

            if away_buffer >= consecutive_guard:
                looking_away_frames += away_buffer
                away_buffer = 0

            frame_idx += 1

        pbar.close()
        cap.release()

    # ---- Gaze accuracy ----
    gaze_accuracy = 0.0
    if total_frames > 0:
        # processed_samples = how many frames you actually processed after subsampling
        effective_total = processed_samples

        if effective_total > 0:
            gaze_accuracy = round(100 * (1 - (looking_away_frames / effective_total)), 2)
        else:
            gaze_accuracy = "N/A"


    summary = {
        "total_frames": total_frames,
        "fps": round(fps, 2),
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
    parser.add_argument("--frame-step", type=int, default=2, help="sample every Nth frame")
    args = parser.parse_args()

    res = analyze_face_and_gaze(args.video, frame_step=args.frame_step)
    print(json.dumps(res, indent=2))
