#!/usr/bin/env python3
"""
AI Proctoring Report (Simplified + Proofs)
------------------------------------------
Integrates:
 - Face + Gaze analysis (proofs for no/multiple faces)
 - Voice diarization (proofs for flagged non-primary speakers)
 - Object detection (YOLOv8 for gadget detection)
Generates PDF + JSON reports.
"""

import os
import json
import uuid
import shutil
import argparse
import subprocess
from datetime import datetime
from fpdf import FPDF

from src.face_and_gaze_analysis import analyze_face_and_gaze
from src.voice_diarization import run_diarization_and_extract_snippets
from src.object_detection import run_object_detection
from src.utils import merge_continuous_flags, merge_voice_segments


# ---------------- Utility ---------------- #
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_audio(video_path: str, outdir: str) -> str:
    """Extract audio (.wav) from video using ffmpeg."""
    ensure_dir(outdir)
    audio_path = os.path.join(outdir, "audio.wav")
    print("[INFO] Extracting audio from video...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(audio_path):
        raise RuntimeError("❌ Audio extraction failed. Check ffmpeg installation.")
    print(f"[INFO] Audio saved at: {audio_path}")
    return audio_path


# ---------------- PDF Builder ---------------- #
class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "AI Proctoring Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(5)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def log_row(self, cols, widths, bold=False):
        self.set_font("Helvetica", "B" if bold else "", 10)
        for text, width in zip(cols, widths):
            self.cell(width, 8, str(text), border=1, align="C")
        self.ln(8)


# ---------------- PDF Report ---------------- #
def build_pdf_report(outdir, session_id, duration, face_flags, voice_segments, gaze_summary, gadget_flags):
    pdf = ReportPDF()
    pdf.add_page()

    # --- Header Info ---
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Session ID: {session_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Duration: {duration}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # --- FACE FLAGS ---
    pdf.section_title("Face Detection Flags")
    pdf.log_row(["Timestamp", "Type", "Proof"], [50, 70, 70], bold=True)
    if not face_flags:
        pdf.log_row(["-", "No issues detected", "-"], [50, 70, 70])
    else:
        for f in face_flags:
            pdf.log_row([
                f.get("timestamp", "-"),
                f.get("reason", "-"),
                os.path.basename(f.get("proof_image", "-"))
            ], [50, 70, 70])
    pdf.ln(8)

    # --- GAZE SUMMARY ---
    pdf.section_title("Gaze Analysis Summary")
    pdf.set_font("Helvetica", "", 11)
    if not gaze_summary:
        pdf.cell(0, 8, "No gaze data available.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 8, f"Total Frames: {gaze_summary.get('total_frames', 0)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"No Face Frames: {gaze_summary.get('no_face_frames', 0)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Multiple Face Frames: {gaze_summary.get('multiple_faces_frames', 0)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Looking Away Frames: {gaze_summary.get('looking_away_frames', 0)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Gaze Accuracy (on-screen): {gaze_summary.get('gaze_accuracy', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # --- VOICE DETECTION ---
    pdf.section_title("Voice Detection")
    pdf.log_row(["Start", "End", "Duration(s)", "Speaker", "Proof"], [35, 35, 35, 45, 50], bold=True)
    if not voice_segments:
        pdf.log_row(["-", "-", "-", "-", "No suspicious voices"], [35, 35, 35, 45, 50])
    else:
        for seg in voice_segments:
            proof_audio = seg.get("proof_audio")
            pdf.log_row([
                seg.get("start", "N/A"),
                seg.get("end", "N/A"),
                round(seg.get("duration", 0), 2),
                seg.get("speaker", "Unknown"),
                os.path.basename(proof_audio) if proof_audio else "-"
            ], [35, 35, 35, 45, 50])
    pdf.ln(8)

    # --- GADGET DETECTION ---
    pdf.section_title("Electronic Gadget Detection")
    pdf.log_row(["Start", "End", "Duration(s)", "Type"], [40, 40, 40, 70], bold=True)
    if not gadget_flags:
        pdf.log_row(["-", "-", "-", "No gadgets detected"], [40, 40, 40, 70])
    else:
        for g in gadget_flags:
            pdf.log_row([
                g.get("start", "-"),
                g.get("end", "-"),
                round(g.get("duration", 0), 2),
                g.get("type", "-")
            ], [40, 40, 40, 70])
    pdf.ln(8)

    # --- SAVE FILES ---
    ensure_dir(outdir)
    pdf_path = os.path.join(outdir, "report.pdf")
    pdf.output(pdf_path)
    print(f"[INFO] PDF saved: {pdf_path}")

    json_path = os.path.join(outdir, "summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "session_id": session_id,
            "duration": duration,
            "face_flags": face_flags,
            "gaze_summary": gaze_summary,
            "voice_segments": voice_segments,
            "gadget_flags": gadget_flags
        }, f, indent=2)
    print(f"[INFO] JSON summary saved: {json_path}")


# ---------------- MAIN ---------------- #
def main():
    parser = argparse.ArgumentParser(description="AI Proctoring Analyzer (Simplified)")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--outdir", default="reports/session_test", help="Output directory")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token for diarization")
    parser.add_argument("--frame-step-ms", type=int, default=1000, help="Frame sampling step (ms)")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    tmp_dir = os.path.join(args.outdir, "tmp")
    ensure_dir(tmp_dir)
    session_id = str(uuid.uuid4())[:8]

    print(f"[INFO] Starting analysis for {args.video}")

    try:
        # --- 1. FACE + GAZE ANALYSIS ---
        print("[STEP] Running Face + Gaze Analysis...")
        fg_result = analyze_face_and_gaze(args.video, frame_step=2, outdir=args.outdir)
        fps = fg_result.get("fps", 30)
        duration = f"{round(fg_result.get('total_frames', 0) / max(fps, 1), 2)}s"

        face_flags = fg_result.get("face_flag_logs", [])
        gaze_summary = {
            "total_frames": fg_result.get("total_frames", 0),
            "no_face_frames": fg_result.get("no_face_frames", 0),
            "multiple_faces_frames": fg_result.get("multiple_faces_frames", 0),
            "looking_away_frames": fg_result.get("looking_away_frames", 0),
            "gaze_accuracy": fg_result.get("gaze_accuracy", "N/A")
        }

        # --- 2. AUDIO EXTRACTION ---
        audio_path = extract_audio(args.video, tmp_dir)

        # --- 3. VOICE DIARIZATION ---
        print("[STEP] Running Voice Diarization...")
        voice_raw, _ = run_diarization_and_extract_snippets(audio_path, args.outdir, hf_token=args.hf_token)
        voice_segments = [v for v in voice_raw if v.get("flagged", False)]

        # --- 4. GADGET DETECTION ---
        print("[STEP] Running Gadget Detection (YOLOv8)...")
        gadget_raw, _ = run_object_detection(args.video, args.outdir,
                                             model_name="yolov8n.pt",
                                             conf_thresh=0.35,
                                             frame_step_ms=1000)
        gadget_flags = merge_continuous_flags(gadget_raw, fps)

        # --- 5. BUILD REPORT ---
        build_pdf_report(args.outdir, session_id, duration, face_flags, voice_segments, gaze_summary, gadget_flags)

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n✅ Proctoring Analysis Complete!\n")


if __name__ == "__main__":
    main()
