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
