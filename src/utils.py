import os
from datetime import timedelta

def _secs_to_hhmmss(s: float) -> str:
    """
    Converts seconds to H:MM:SS.sss format.
    (e.g., 90.5 -> "0:01:30.500")
    """
    if s < 0:
        s = 0
    # [:-3] cuts "0:01:30.500000" to "0:01:30.500"
    return str(timedelta(seconds=s))[:-3]


def merge_voice_segments(segments: list, max_gap_s: float = 2.0) -> list:
    """
    Merges consecutive voice segments from the same speaker.
    """
    if not segments:
        return []

    segments.sort(key=lambda x: (x["start"], x["speaker"]))
    merged = []
    current_segment = segments[0].copy()

    for next_segment in segments[1:]:
        try:
            def parse_time(t_str):
                parts = t_str.split(':')
                sec_ms = parts[2].split('.')
                return timedelta(hours=int(parts[0]), 
                                 minutes=int(parts[1]), 
                                 seconds=int(sec_ms[0]),
                                 milliseconds=int(sec_ms[1])).total_seconds()

            current_end_s = parse_time(current_segment["end"])
            next_start_s = parse_time(next_segment["start"])
            gap = next_start_s - current_end_s
        except Exception as e:
            print(f"Warning: Could not parse time for voice merging: {e}")
            gap = max_gap_s + 1

        if next_segment["speaker"] == current_segment["speaker"] and 0 <= gap <= max_gap_s:
            current_segment["end"] = next_segment["end"]
            current_start_s = parse_time(current_segment["start"])
            current_end_s_new = parse_time(current_segment["end"])
            current_segment["duration"] = current_end_s_new - current_start_s
        else:
            merged.append(current_segment)
            current_segment = next_segment.copy()

    merged.append(current_segment)
    return merged


# --- NEW FUNCTION TO FIX YOUR REPORT ---
def merge_gadget_logs(logs: list, max_gap_s: float = 0.5) -> list:
    """
    Merges consecutive gadget logs from your object_detection.py script.
    
    It expects logs like:
    [{"start": "0:00:25.000", "end": "0:00:26.000", "type": "cell phone"},
     {"start": "0:00:26.000", "end": "0:00:27.000", "type": "cell phone"}]
    
    And merges them into:
    [{"start": "0:00:25.000", "end": "0:00:27.000", "duration": 2.0, "type": "cell phone"}]
    """
    if not logs:
        return []

    merged = []
    
    # Helper to parse time string 'H:MM:SS.ms' to seconds
    def to_seconds(t_str):
        try:
            parts = t_str.split(':')
            sec_ms = parts[2].split('.')
            return timedelta(hours=int(parts[0]), 
                             minutes=int(parts[1]), 
                             seconds=int(sec_ms[0]),
                             milliseconds=int(sec_ms[1])).total_seconds()
        except Exception:
            # Fallback for 0:00 format
            return 0.0

    current_segment = logs[0].copy()
    
    for next_log in logs[1:]:
        current_end_s = to_seconds(current_segment['end'])
        next_start_s = to_seconds(next_log['start'])
        gap = next_start_s - current_end_s
        
        # Check if same type and the gap is very small (i.e., consecutive)
        if next_log['type'] == current_segment['type'] and gap <= max_gap_s:
            # Merge: update the end time
            current_segment['end'] = next_log['end']
            
            # Recalculate duration
            current_start_s = to_seconds(current_segment['start'])
            current_end_s_new = to_seconds(current_segment['end'])
            current_segment['duration'] = current_end_s_new - current_start_s
            
            # Keep the proof/confidence from the *start* of the event
        else:
            # Gap is too large or type is different, save the old segment
            merged.append(current_segment)
            # Start a new segment
            current_segment = next_log.copy()
    
    # Add the very last segment
    merged.append(current_segment)
    
    return merged