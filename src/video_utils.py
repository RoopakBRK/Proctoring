import os
import sys
import subprocess
from pathlib import Path
import cv2
from .utils import ensure_dir


def download_video(vimeo_url: str, out_dir: str) -> str:
    """Download Vimeo video using yt-dlp."""
    ensure_dir(out_dir)
    out_template = os.path.join(out_dir, "video.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "best",
        "-o",
        out_template,
        vimeo_url,
    ]

    print("[INFO] Downloading video...")
    subprocess.check_call(cmd)

    for ext in ["mp4", "mkv", "webm", "mov", "m4v"]:
        candidate = os.path.join(out_dir, f"video.{ext}")
        if os.path.exists(candidate):
            print(f"[INFO] Video downloaded: {candidate}")
            return candidate

    files = sorted(Path(out_dir).glob("video.*"))
    if files:
        return str(files[-1])
    raise FileNotFoundError("No video file found after download.")


def extract_audio(video_path: str, out_wav: str) -> str:
    """Extract mono 16kHz audio from video."""
    print("[INFO] Extracting audio...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        out_wav,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[INFO] Audio saved to: {out_wav}")
    return out_wav


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count / fps if fps else 0.0
