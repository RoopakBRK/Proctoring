import os
import json
import base64
from typing import List
from jinja2 import Template
from dataclasses import asdict
from pathlib import Path

from .utils import secs_to_hhmmss, ensure_dir


def build_report(
    out_dir: str,
    video_path: str,
    duration: float,
    faces: List,
    segments: List,
    flags: List,
    report_name: str = "report.html",
):
    """Builds HTML + JSON reports summarizing detections."""
    ensure_dir(out_dir)

    # --- Embed faces as base64 ---
    face_entries = []
    for rec in faces:
        img_path = os.path.join(out_dir, rec.image_fname)
        if not os.path.exists(img_path):
            continue
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        face_entries.append(
            {
                "timestamp": rec.timestamp,
                "hhmmss": rec.hhmmss,
                "bbox": rec.bbox,
                "image_b64": b64,
                "image_fname": rec.image_fname,
            }
        )

    # --- Determine primary speaker ---
    primary = None
    if segments:
        dur_by_speaker = {}
        for seg in segments:
            dur_by_speaker[seg.speaker] = dur_by_speaker.get(seg.speaker, 0.0) + (seg.end - seg.start)
        primary = max(dur_by_speaker.items(), key=lambda kv: kv[1])[0]

    # --- Load HTML template ---
    template_path = os.path.join("templates", "report_template.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            html_template = f.read()
    else:
        html_template = """<!doctype html><html><body><h1>Report</h1><p>No template found.</p></body></html>"""

    # --- Render template ---
    template = Template(html_template)
    html = template.render(
        video_filename=os.path.basename(video_path),
        duration_hhmmss=secs_to_hhmmss(duration),
        face_count=len(face_entries),
        faces=face_entries,
        primary_speaker=primary or "N/A",
        total_speaker_segments=len(segments),
        segments=[asdict(s) for s in segments],
        flags=[asdict(f) for f in flags],
    )

    # --- Write report.html ---
    report_path = os.path.join(out_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # --- JSON Summary ---
    summary = {
        "video": os.path.basename(video_path),
        "duration": duration,
        "face_count": len(face_entries),
        "faces": [
            {
                "timestamp": f["timestamp"],
                "hhmmss": f["hhmmss"],
                "bbox": f["bbox"],
                "image": f["image_fname"],
            }
            for f in face_entries
        ],
        "primary_speaker": primary,
        "total_speaker_segments": len(segments),
        "speaker_segments": [asdict(s) for s in segments],
        "voice_flags": [asdict(f) for f in flags],
    }

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Report generated at {report_path}")
    print(f"[INFO] Summary JSON at {json_path}")
