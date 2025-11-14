"""
Speaker Diarization + Audio Snippet Extraction (Tier 1 Optimized)
--------------------------------------------------------------
- Identifies non-primary speakers (suspicious voices)
- Saves proof .wav segments with unique filenames
- Adds robust error handling and detailed logs
- OPTIMIZATIONS APPLIED (per request):
  - 1: Pipeline loaded globally (Opt #1)
  - 2: MPS/CUDA/CPU auto-detection (Opt #2)
  - 3: Noise segments < 0.2s filtered early (Opt #4)
  - 4: Pydub replaced with 'wave' module for 10x+ clip speed (Opt #5)
"""

import os
import json
import uuid
import torch
from datetime import timedelta
import wave                      # <-- OPTIMIZATION 5: Replaced pydub
import contextlib                # <-- OPTIMIZATION 5: Used for 'wave'
from pyannote.audio import Pipeline


# <-- OPTIMIZATION 1: Global pipeline cache
PIPELINE = None

def load_pipeline(hf_token):
    """Loads the diarization pipeline, caching it globally."""
    global PIPELINE
    if PIPELINE is None:
        print("[INFO] Loading Pyannote pipeline for the first time...")
        try:
            PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # <-- OPTIMIZATION 2: Auto-select best device (MPS, CUDA, CPU)
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
            print(f"[INFO] Sending pipeline to device: {device}")
            PIPELINE.to(device)

        except Exception as e:
            print(f"[FATAL] Could not load Pyannote pipeline: {e}")
            PIPELINE = None # Ensure it's still None so we don't try to use a failed load
    return PIPELINE
# <-- END OPTIMIZATION 1 & 2


# -------------------------------------------------------------
# Extract audio snippet safely
# -------------------------------------------------------------
def extract_audio_segment(
    input_audio_path: str, # <-- OPTIMIZATION 5: Signature changed (no pydub obj)
    start_s: float,
    end_s: float,
    output_path: str
) -> bool:
    """Extracts an audio segment using the fast 'wave' module."""
    try:
        # <-- OPTIMIZATION 5: Use 'wave' module instead of pydub
        with contextlib.closing(wave.open(input_audio_path, 'rb')) as wav_in:
            frame_rate = wav_in.getframerate()
            n_channels = wav_in.getnchannels()
            samp_width = wav_in.getsampwidth()
            
            start_frame = int(round(start_s * frame_rate))
            end_frame = int(round(end_s * frame_rate))
            
            if end_frame <= start_frame:
                print(f"[WARN] Invalid segment times (wave): start={start_s:.2f}, end={end_s:.2f}")
                return False

            wav_in.setpos(start_frame)
            frames_to_read = end_frame - start_frame
            frames = wav_in.readframes(frames_to_read)
            
            if not frames:
                print(f"[WARN] Empty audio segment (wave) for {output_path}")
                return False

        with contextlib.closing(wave.open(output_path, 'wb')) as wav_out:
            wav_out.setnchannels(n_channels)
            wav_out.setsampwidth(samp_width)
            wav_out.setframerate(frame_rate)
            wav_out.writeframes(frames)
        # <-- END OPTIMIZATION 5

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            print(f"[WARN] Segment export failed (wave) for {output_path}")
            return False
    
    except wave.Error as e:
        print(f"[WARN] 'wave' module failed for {input_audio_path}: {e}")
        print("         (Is the file a valid PCM WAV? pydub was more flexible.)")
        return False
    except Exception as e:
        print(f"[WARN] Could not extract segment (wave) {start_s:.2f}-{end_s:.2f}s: {e}")
        return False


# -------------------------------------------------------------
# Main diarization pipeline
# -------------------------------------------------------------
def run_diarization_and_extract_snippets(
    audio_path: str,
    outdir: str,
    hf_token: str = None,
    min_flag_duration: float = 1.5,
    min_noise_duration: float = 0.2  # <-- OPTIMIZATION 4: Changed default
):
    """
    Performs Pyannote speaker diarization.
    Flags all non-primary speakers with duration >= min_flag_duration.
    Exports proof .wav segments for each flagged segment.
    Returns: (logs, total_flags)
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # <-- OPTIMIZATION 1: Load pipeline (cached)
    pipeline = load_pipeline(hf_token)
    if pipeline is None:
        print("[WARN] Pipeline loading failed. Skipping voice diarization.")
        return [], 0
    # <-- END OPTIMIZATION 1

    # Removing the batch size optimization as it's not in the requested list
    # and can be version-specific.

    print("[STEP] Running Voice Diarization...")
    diarization = pipeline(audio_path)

    # <-- OPTIMIZATION 5: Get duration with 'wave' module (fast)
    print("[INFO] Getting audio duration...")
    try:
        with contextlib.closing(wave.open(audio_path, 'rb')) as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            audio_duration = frames / float(rate)
    except Exception as e:
        print(f"[WARN] Failed to get audio duration with 'wave': {e}")
        # Fallback or error
        audio_duration = 0.0
        print("[WARN] Could not determine audio duration.")
    # <-- END OPTIMIZATION 5

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
        
    # <-- OPTIMIZATION 4: Filter out tiny noise segments BEFORE calculating totals
    print(f"[INFO] Filtering out noise segments < {min_noise_duration}s...")
    
    filtered_speaker_segments = {}
    for spk, segs in speaker_segments.items():
        # Keep only segments that are >= the noise threshold
        valid_segs = [
            seg for seg in segs if seg["duration"] >= min_noise_duration
        ]
        
        # Only add the speaker back if they still have valid segments
        if valid_segs:
            filtered_speaker_segments[spk] = valid_segs
    
    # Use this newly filtered dictionary for all subsequent processing
    speaker_segments = filtered_speaker_segments 
    
    if not speaker_segments:
        print(f"[WARN] No speaker activity remaining after noise filtering (<{min_noise_duration}s).")
        return [], 0
    # <-- END OPTIMIZATION 4

    # Determine primary speaker (most total speaking time)
    speaker_durations = {
        spk: sum(seg["duration"] for seg in segs)
        for spk, segs in speaker_segments.items() # This now uses the filtered segments
    }
    
    if not speaker_durations:
        print("[WARN] No speaker durations left after noise filtering.")
        return [], 0
        
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

                # <-- OPTIMIZATION 5: Call 'wave' extractor
                # Pass the audio_path, not the (now removed) pydub object
                saved = extract_audio_segment(
                    audio_path, start_s, end_s, proof_path
                )
                # <-- END OPTIMIZATION 5
                
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
            "min_noise_duration": min_noise_duration,
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
    
    parser.add_argument(
        "--min-noise",
        type=float,
        default=0.2, # <-- OPTIMIZATION 4: Changed default
        help="Min duration (s) to consider a segment valid (filters noise)."
    )
    parser.add_argument(
        "--min-flag",
        type=float,
        default=1.5,
        help="Min duration (s) to flag a non-primary segment as suspicious."
    )
    
    args = parser.parse_args()

    logs, total_flags = run_diarization_and_extract_snippets(
        audio_path=args.audio,
        outdir=args.outdir,
        hf_token=args.hf_token,
        min_flag_duration=args.min_flag,
        min_noise_duration=args.min_noise
    )
    print(f" Total suspicious voices flagged: {total_flags}")