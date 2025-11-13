# src/utils.py
from datetime import timedelta

# -------------------------------------------------------------
# Timestamp parser
# -------------------------------------------------------------
def _parse_timestamp(ts):
    """Convert HH:MM:SS(.mmm) or float/int → seconds (float)."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        parts = ts.split(":")
        try:
            if len(parts) == 3:
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return float(m) * 60 + float(s)
            return float(parts[0])
        except Exception:
            return 0.0
    return 0.0


def _fmt(seconds: float) -> str:
    """Return HH:MM:SS.mmm formatted string."""
    return str(timedelta(seconds=round(seconds, 3)))[:-3]


# -------------------------------------------------------------
# Merge continuous detection flags (face/gadget)
# -------------------------------------------------------------
def merge_continuous_flags(flag_logs, fps=30, min_duration_sec=0.5, gap_threshold=1.5):
    """
    Merge temporally close detections that share the same type/reason/label.
    Handles both single 'timestamp' and 'start'/'end' logs.
    """

    if not flag_logs:
        return []

    # Normalize timestamps
    for log in flag_logs:
        if "start" in log and "end" in log:
            log["start"] = _parse_timestamp(log["start"])
            log["end"] = _parse_timestamp(log["end"])
        elif "timestamp" in log:
            t = _parse_timestamp(log["timestamp"])
            log["start"], log["end"] = t, t
        else:
            log["start"], log["end"] = 0.0, 0.0

    flag_logs = sorted(flag_logs, key=lambda x: x["start"])

    def _get_type(entry):
        return entry.get("type") or entry.get("reason") or entry.get("label") or "Unknown"

    merged = []
    current = {
        "start": flag_logs[0]["start"],
        "end": flag_logs[0]["end"],
        "type": _get_type(flag_logs[0])
    }

    for log in flag_logs[1:]:
        r = _get_type(log)
        if r == current["type"] and log["start"] - current["end"] <= gap_threshold:
            current["end"] = log["end"]
        else:
            dur = current["end"] - current["start"]
            if dur >= min_duration_sec:
                merged.append({
                    "start": _fmt(current["start"]),
                    "end": _fmt(current["end"]),
                    "duration": round(dur, 2),
                    "type": current["type"]
                })
            current = {"start": log["start"], "end": log["end"], "type": r}

    # finalize last
    dur = current["end"] - current["start"]
    if dur >= min_duration_sec:
        merged.append({
            "start": _fmt(current["start"]),
            "end": _fmt(current["end"]),
            "duration": round(dur, 2),
            "type": current["type"]
        })

    return merged


# -------------------------------------------------------------
# Merge voice diarization segments
# -------------------------------------------------------------
def merge_voice_segments(segments, gap_threshold=1.0, min_duration=1.5):
    """
    Merge consecutive voice segments from the same speaker if close in time.
    """

    if not segments:
        return []

    # Normalize start/end
    for s in segments:
        s["start"] = _parse_timestamp(s.get("start", 0.0))
        s["end"] = _parse_timestamp(s.get("end", 0.0))
        s["flagged"] = s.get("flagged", True)

    segments.sort(key=lambda s: (s.get("speaker", ""), s["start"]))
    merged = []
    cur = segments[0].copy()

    for seg in segments[1:]:
        same_speaker = seg.get("speaker") == cur.get("speaker")
        close_enough = seg["start"] - cur["end"] <= gap_threshold

        if same_speaker and close_enough:
            cur["end"] = max(cur["end"], seg["end"])
            cur["flagged"] = cur["flagged"] or seg.get("flagged", False)
        else:
            dur = cur["end"] - cur["start"]
            if dur >= min_duration:
                merged.append({
                    "speaker": cur.get("speaker", "Unknown"),
                    "start": _fmt(cur["start"]),
                    "end": _fmt(cur["end"]),
                    "duration": round(dur, 2),
                    "flagged": cur.get("flagged", True)
                })
            cur = seg.copy()

    dur = cur["end"] - cur["start"]
    if dur >= min_duration:
        merged.append({
            "speaker": cur.get("speaker", "Unknown"),
            "start": _fmt(cur["start"]),
            "end": _fmt(cur["end"]),
            "duration": round(dur, 2),
            "flagged": cur.get("flagged", True)
        })

    return merged
