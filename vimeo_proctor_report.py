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
import time
from datetime import datetime
from fpdf import FPDF
import cv2

# --- Import all sub-modules from 'src' folder ---
try:
    from src.face_and_gaze_analysis import analyze_face_and_gaze
    from src.voice_diarization import run_diarization_and_extract_snippets
    from src.object_detection import run_object_detection
    # --- UPDATED IMPORT ---
    from src.utils import merge_voice_segments, merge_gadget_logs
except ImportError as e:
    print(f"Error: Could not import a required module from 'src/'. {e}")
    print("Please ensure all files (face_and_gaze_analysis.py, voice_diarization.py, object_detection.py, utils.py) exist in the 'src' folder.")
    exit(1)


# ---------------- TIMER HELPERS ---------------- #
def start_timer():
    """Starts a high-performance timer."""
    return time.perf_counter()

def end_timer(t0, label=""):
    """Ends the timer and prints the duration."""
    dt = round(time.perf_counter() - t0, 4)
    print(f"[TIMER] {label} took: {dt} seconds")
    return dt


# ---------------- Utility ---------------- #
def ensure_dir(path: str):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def extract_audio(video_path: str, outdir: str) -> str:
    """
    Extract audio (.wav) from video using ffmpeg.
    Forces 16kHz, 16-bit, mono PCM for compatibility.
    """
    ensure_dir(outdir)
    audio_path = os.path.join(outdir, "audio.wav")
    print("[INFO] Extracting audio from video...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("❌ Audio extraction failed. Is ffmpeg installed and in your system's PATH?")

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("❌ Audio extraction failed. No output file was created.")

    print(f"[INFO] Audio saved at: {audio_path}")
    return audio_path


# ---------------- PDF Builder ---------------- #
class ReportPDF(FPDF):
    """Custom PDF class to build the report with headers and sections."""
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
        
    def add_summary_line(self, key, value):
        self.set_font("Helvetica", "B", 11)
        self.cell(60, 8, str(key))
        self.set_font("Helvetica", "", 11)
        self.cell(0, 8, str(value), new_x="LMARGIN", new_y="NEXT")


# ---------------- PDF Report ---------------- #
def build_pdf_report(outdir, session_id, duration, face_flags, voice_segments, gaze_summary, gadget_flags):
    """Generates the final PDF and JSON reports."""
    
    print("[INFO] Building PDF and JSON reports...")
    pdf = ReportPDF()
    pdf.add_page()

    # --- Header Info ---
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Session ID: {session_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Video Duration: {duration}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # --- GAZE SUMMARY ---
    pdf.section_title("Gaze Analysis Summary")
    pdf.add_summary_line("Gaze Accuracy (on-screen):", gaze_summary.get('gaze_accuracy', 'N/A'))
    pdf.add_summary_line("Total Frames:", f"{gaze_summary.get('total_frames', 0)}")
    pdf.add_summary_line("Gaze Sample Rate:", f"1 frame per {gaze_summary.get('gaze_frame_step', 'N/A')} video frames")
    pdf.add_summary_line("Frames Processed for Gaze:", gaze_summary.get('processed_samples', 0))
    pdf.add_summary_line("Looking Away Frames:", gaze_summary.get('looking_away_frames', 0))
    pdf.add_summary_line("No Face Frames:", gaze_summary.get('no_face_frames', 0))
    pdf.add_summary_line("Multiple Face Frames:", gaze_summary.get('multiple_faces_frames', 0))
    pdf.ln(8)
    
    # --- FACE FLAGS (Proofs) ---
    pdf.section_title("Face Detection Flags (Proofs)")
    pdf.log_row(["Timestamp", "Type", "Proof"], [50, 70, 70], bold=True)
    if not face_flags:
        pdf.log_row(["-", "No issues detected", "-"], [50, 70, 70])
    else:
        for f in face_flags[:10]: # Limit to 10 proofs in PDF
            pdf.log_row([
                f.get("timestamp", "-"),
                f.get("reason", "-"),
                os.path.basename(f.get("proof_image", "-"))
            ], [50, 70, 70])
    pdf.ln(8)

    # --- VOICE DETECTION ---
    pdf.section_title("Suspicious Voice Detection")
    pdf.log_row(["Start", "End", "Duration(s)", "Speaker", "Proof"], [35, 35, 35, 45, 50], bold=True)
    if not voice_segments:
        pdf.log_row(["-", "-", "-", "-", "No suspicious voices"], [35, 35, 35, 45, 50])
    else:
        for seg in voice_segments:
            proof = os.path.basename(seg.get("proof_audio")) if seg.get("proof_audio") else "-"
            pdf.log_row([
                seg.get("start", "N/A"),
                seg.get("end", "N/A"),
                round(seg.get("duration", 0), 2),
                seg.get("speaker", "Unknown"),
                proof
            ], [35, 35, 35, 45, 50])
    pdf.ln(8)

    # --- GADGET DETECTION ---
    pdf.section_title("Electronic Gadget Detection")
    pdf.log_row(["Start", "End", "Duration(s)", "Type"], [40, 40, 40, 70], bold=True)
    if not gadget_flags:
        pdf.log_row(["-", "-", "-", "No gadgets detected"], [40, 40, 40, 70])
    else:
        # This will now show the MERGED logs
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
            "gaze_summary": gaze_summary,
            "face_flags": face_flags,
            "voice_segments": voice_segments,
            "gadget_flags": gadget_flags
        }, f, indent=2)

    print(f"[INFO] JSON summary saved: {json_path}")


# ---------------- MAIN ---------------- #
def main():
    """Main function to run the entire proctoring pipeline."""
    
    t0_total = start_timer()

    parser = argparse.ArgumentParser(description="AI Proctoring Analyzer")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--outdir", default="reports/session_test", help="Output directory")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token for diarization")
    
    parser.add_argument("--object-sample-ms", type=int, default=1000, help="Frame sampling step (ms) for Object Detection")
    parser.add_argument("--gaze-frame-step", type=int, default=2, help="Sample every Nth frame for gaze analysis (e.g., 2, 5). Lower is more accurate but slower.")
    
    args = parser.parse_args()

    # --- Setup Directories ---
    ensure_dir(args.outdir)
    tmp_dir = os.path.join(args.outdir, "tmp")
    ensure_dir(tmp_dir)
    session_id = str(uuid.uuid4())[:8]

    print(f"[INFO] Starting analysis for {args.video}")
    print(f"[INFO] Session ID: {session_id}")
    print(f"[INFO] Output will be saved to: {args.outdir}")

    try:
        # --- 1. GET VIDEO FPS ---
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        
        if total_frames == 0:
            raise RuntimeError("Video file has 0 frames or is unreadable.")
            
        duration = f"{round(total_frames / max(fps, 1), 2)}s"

        # --- 2. FACE + GAZE ANALYSIS ---
        t0_face = start_timer()
        print("[STEP] Running Face + Gaze Analysis...")
        fg_result = analyze_face_and_gaze(
            args.video, 
            gaze_frame_step=args.gaze_frame_step, 
            outdir=args.outdir
        )
        end_timer(t0_face, "Face + Gaze Analysis")

        face_flags = fg_result.get("face_flag_logs", [])
        gaze_summary = {
            "total_frames": fg_result.get("total_frames", 0),
            "processed_samples": fg_result.get("processed_samples", 0),
            "gaze_frame_step": fg_result.get("gaze_frame_step", args.gaze_frame_step),
            "no_face_frames": fg_result.get("no_face_frames", 0),
            "multiple_faces_frames": fg_result.get("multiple_faces_frames", 0),
            "looking_away_frames": fg_result.get("looking_away_frames", 0),
            "gaze_accuracy": fg_result.get("gaze_accuracy", "N/A")
        }

        # --- 3. AUDIO EXTRACTION ---
        t0_audio = start_timer()
        audio_path = extract_audio(args.video, tmp_dir)
        end_timer(t0_audio, "Audio Extraction")

        # --- 4. VOICE DIARIZATION ---
        t0_voice = start_timer()
        print("[STEP] Running Voice Diarization...")
        voice_raw, _ = run_diarization_and_extract_snippets(audio_path, args.outdir, hf_token=args.hf_token)
        voice_segments = [v for v in voice_raw if v.get("flagged", False)]
        end_timer(t0_voice, "Voice Diarization")

        # --- 5. GADGET DETECTION (MODIFIED) ---
        t0_obj = start_timer()
        print("[STEP] Running Gadget Detection (YOLOv8)...")
        
        # Step 5a: Get the *unmerged* logs from your script
        # We also tune the model for better accuracy
        unmerged_gadget_logs, _ = run_object_detection(
            args.video, 
            args.outdir,
            model_name="yolov8s.pt",  # "small" model for better accuracy
            conf_thresh=0.25,        # Lower confidence for more detections
            frame_step_ms=args.object_sample_ms
        )
        
        # --- NEW MERGING STEP ---
        # Step 5b: Merge the consecutive 1-second logs
        print(f"[INFO] Merging {len(unmerged_gadget_logs)} gadget detections into continuous events...")
        gadget_flags = merge_gadget_logs(unmerged_gadget_logs)
        
        end_timer(t0_obj, "Object Detection")

        # --- 6. BUILD REPORT ---
        build_pdf_report(
            args.outdir, 
            session_id, 
            duration, 
            face_flags, 
            voice_segments, 
            gaze_summary, 
            gadget_flags
        )

    except Exception as e:
        print(f"\n[FATAL ERROR] The pipeline failed: {e}")
    finally:
        # --- Cleanup ---
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                print(f"[INFO] Cleaned up temporary directory: {tmp_dir}")
            except Exception as e:
                print(f"[WARN] Could not remove temporary directory: {e}")

    end_timer(t0_total, "TOTAL Pipeline Runtime")
    print(f"\n✅ Proctoring Analysis Complete! Report saved in: {args.outdir}\n")


if __name__ == "__main__":
    main()