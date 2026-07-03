#!/usr/bin/env python3
import time
_SCRIPT_START = time.perf_counter()

import argparse
import csv
import json
import string
from pathlib import Path
from typing import Any

from audio_utils_board import format_ts, write_json

_IMPORTS_DONE = time.perf_counter()


def write_profile_md(path: Path, profile: dict[str, Any]) -> None:
    timings = profile.get("timings", {})
    total = float(timings.get("total_script_sec") or 0.0)
    lines = [
        "# Alignment Internal Profile",
        "",
        f"- ASR segments: {profile.get('asr_segment_count')}",
        f"- diarization segments: {profile.get('diarization_segment_count')}",
        f"- audio duration sec: {profile.get('audio_duration_sec')}",
        f"- total RTF: {profile.get('total_rtf')}",
        "",
        "| stage | sec | percent |",
        "| --- | ---: | ---: |",
    ]
    for key, value in timings.items():
        if key.endswith("_sec") and isinstance(value, (int, float)):
            pct = (float(value) / total * 100.0) if total > 0 else 0.0
            lines.append(f"| {key} | {float(value):.6f} | {pct:.2f}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def speaker_label(seg: dict[str, Any]) -> str:
    return str(seg.get("speaker_label") or f"speaker_{int(seg['speaker']):02d}")


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def unknown_label(index: int) -> str:
    letters = string.ascii_uppercase
    return f"说话人{letters[index]}" if index < len(letters) else f"说话人{index + 1}"


def build_registered_map(identification: dict[str, Any], threshold: float) -> dict[str, dict[str, Any]]:
    out = {}
    for label, item in identification.items():
        top1_score = item.get("top1_score")
        matched = item.get("matched_speaker")
        scores = item.get("scores") or []
        verified = bool(matched) and top1_score is not None and float(top1_score) >= threshold
        if verified and scores:
            verified = any(s.get("speaker") == matched and s.get("verified") for s in scores)
        out[label] = {
            "registered_name": matched if verified else None,
            "verified": verified,
            "top1_speaker": item.get("top1_speaker"),
            "top1_score": top1_score,
            "threshold": threshold,
        }
    return out


def find_gap(asr_start: float, asr_end: float, diar: list[dict], gap_fill_sec: float):
    prev_seg = None
    next_seg = None
    for seg in diar:
        if float(seg["end"]) <= asr_start:
            prev_seg = seg
        if float(seg["start"]) >= asr_end and next_seg is None:
            next_seg = seg
            break
    if prev_seg and next_seg:
        prev_label = speaker_label(prev_seg)
        next_label = speaker_label(next_seg)
        prev_dist = asr_start - float(prev_seg["end"])
        next_dist = float(next_seg["start"]) - asr_end
        if prev_label == next_label and max(prev_dist, next_dist) <= gap_fill_sec:
            return prev_label, "gap_fill_same_neighbor", max(prev_dist, next_dist)
        if min(prev_dist, next_dist) <= gap_fill_sec:
            return (prev_label if prev_dist <= next_dist else next_label), "gap_fill_nearest", min(prev_dist, next_dist)
    return None, None, None


def assign(asr: dict, diar: list[dict], front_fill_sec: float, gap_fill_sec: float, min_overlap_sec: float, min_overlap_ratio: float):
    a0 = float(asr["start"])
    a1 = float(asr["end"])
    dur = max(0.0, a1 - a0)
    by_speaker = {}
    for seg in diar:
        ov = overlap(a0, a1, float(seg["start"]), float(seg["end"]))
        if ov > 0:
            label = speaker_label(seg)
            by_speaker[label] = by_speaker.get(label, 0.0) + ov
    max_ov = 0.0
    ratio = 0.0
    if by_speaker:
        best, max_ov = max(by_speaker.items(), key=lambda kv: (kv[1], kv[0]))
        ratio = max_ov / dur if dur > 0 else 0.0
        if max_ov >= min_overlap_sec or ratio >= min_overlap_ratio:
            return best, "direct_overlap", max_ov, ratio, by_speaker

    if diar and a1 <= float(diar[0]["start"]):
        dist = float(diar[0]["start"]) - a1
        if dist <= front_fill_sec:
            return speaker_label(diar[0]), "front_fill_first_speaker", 0.0, 0.0, by_speaker

    label, rule, _ = find_gap(a0, a1, diar, gap_fill_sec)
    if label:
        return label, rule, max_ov, ratio, by_speaker
    return "speaker_unknown", "unassigned", max_ov, ratio, by_speaker


def display_map(assignments: list[str], registered_map: dict[str, dict[str, Any]], mode: str):
    out = {}
    identified = []
    unknown = []
    next_unknown = 0 if mode == "anonymous" else 1
    for label in assignments:
        if label == "speaker_unknown" or label in out:
            continue
        reg = registered_map.get(label, {})
        if mode == "registered_unknown" and reg.get("verified") and reg.get("registered_name"):
            out[label] = str(reg["registered_name"])
            identified.append(str(reg["registered_name"]))
        else:
            out[label] = unknown_label(next_unknown)
            unknown.append(label)
            next_unknown += 1
    out["speaker_unknown"] = "说话人未知"
    return out, sorted(set(identified)), unknown


def write_md(path: Path, aligned: list[dict], asr_source: str, threshold: float) -> None:
    lines = [
        "# Final ASR with sherpa speaker labels",
        "",
        "Source:",
        f"- ASR: {asr_source}",
        "- Diarization: board sherpa-onnx",
        "- Identification: board sherpa Ye enrollment",
        "- ASR text rewritten: False",
        f"- Speaker threshold: {threshold:.3f}",
        "",
    ]
    for seg in aligned:
        lines.append(f"[{format_ts(seg['start'])}-{format_ts(seg['end'])}] {seg['display_label']}:")
        lines.append(str(seg["text"]))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, aligned: list[dict]) -> None:
    fields = ["segment_id", "start", "end", "duration", "text", "assigned_sherpa_speaker", "display_label", "assignment_rule", "max_overlap_sec", "overlap_ratio", "speaker_overlaps_json"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seg in aligned:
            writer.writerow({
                "segment_id": seg["segment_id"],
                "start": f"{seg['start']:.6f}",
                "end": f"{seg['end']:.6f}",
                "duration": f"{seg['duration']:.6f}",
                "text": seg["text"],
                "assigned_sherpa_speaker": seg["assigned_sherpa_speaker"],
                "display_label": seg["display_label"],
                "assignment_rule": seg["assignment_rule"],
                "max_overlap_sec": f"{seg['max_overlap_sec']:.6f}",
                "overlap_ratio": f"{seg['overlap_ratio']:.6f}",
                "speaker_overlaps_json": json.dumps(seg["speaker_overlaps"], ensure_ascii=False, sort_keys=True),
            })


def main() -> int:
    timings = {
        "python_imports_sec": _IMPORTS_DONE - _SCRIPT_START,
    }
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-segments", type=Path, required=True)
    parser.add_argument("--diarization", type=Path, required=True)
    parser.add_argument("--identification", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speaker-threshold", type=float, default=0.6)
    parser.add_argument("--display-mode", choices=["registered_unknown", "anonymous"], default="registered_unknown")
    parser.add_argument("--front-fill-sec", type=float, default=3.0)
    parser.add_argument("--gap-fill-sec", type=float, default=1.0)
    parser.add_argument("--min-overlap-sec", type=float, default=0.20)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.10)
    args = parser.parse_args()
    timings["argument_parse_sec"] = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    asr = json.loads(args.asr_segments.read_text(encoding="utf-8"))
    timings["load_asr_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    diar = sorted(json.loads(args.diarization.read_text(encoding="utf-8")), key=lambda x: float(x["start"]))
    timings["load_diarization_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    identification = json.loads(args.identification.read_text(encoding="utf-8"))
    timings["load_identification_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    registered = build_registered_map(identification, args.speaker_threshold)
    timings["registered_map_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows = []
    for item in asr:
        assigned, rule, max_ov, ratio, overlaps = assign(item, diar, args.front_fill_sec, args.gap_fill_sec, args.min_overlap_sec, args.min_overlap_ratio)
        rows.append({
            "segment_id": int(item["segment_id"]),
            "start": float(item["start"]),
            "end": float(item["end"]),
            "duration": max(0.0, float(item["end"]) - float(item["start"])),
            "text": str(item["text"]),
            "source": item.get("source"),
            "assigned_sherpa_speaker": assigned,
            "assignment_rule": rule,
            "max_overlap_sec": max_ov,
            "overlap_ratio": ratio,
            "speaker_overlaps": overlaps,
            "text_unchanged": True,
        })
    dmap, identified, unknown = display_map([r["assigned_sherpa_speaker"] for r in rows], registered, args.display_mode)
    for row in rows:
        row["display_label"] = dmap.get(row["assigned_sherpa_speaker"], "说话人未知")
        row["speaker_identification"] = registered.get(row["assigned_sherpa_speaker"])
    timings["overlap_assignment_sec"] = time.perf_counter() - t0

    audio_duration = max([float(s["end"]) for s in asr], default=0.0)
    unassigned = sum(1 for r in rows if r["assigned_sherpa_speaker"] == "speaker_unknown")
    metrics = {
        "elapsed_sec": 0.0,
        "audio_duration_sec": audio_duration,
        "rtf": None,
        "asr_segment_count": len(asr),
        "diarization_segment_count": len(diar),
        "diarization_speaker_count": len({speaker_label(s) for s in diar}),
        "assigned_segment_count": len(rows) - unassigned,
        "unassigned_segment_count": unassigned,
        "direct_overlap_count": sum(1 for r in rows if r["assignment_rule"] == "direct_overlap"),
        "front_fill_count": sum(1 for r in rows if r["assignment_rule"] == "front_fill_first_speaker"),
        "gap_fill_count": sum(1 for r in rows if r["assignment_rule"].startswith("gap_fill")),
        "text_unchanged": True,
        "identified_speakers": identified,
        "unknown_speakers": unknown,
        "speaker_display_map": dmap,
        "registered_speaker_map": registered,
    }
    t0 = time.perf_counter()
    write_json(args.out_dir / "final_asr_with_sherpa_speakers.json", rows)
    write_md(args.out_dir / "final_asr_with_sherpa_speakers.md", rows, str(asr[0].get("source", args.asr_segments)) if asr else str(args.asr_segments), args.speaker_threshold)
    write_csv(args.out_dir / "alignment_debug.csv", rows)
    timings["write_outputs_sec"] = time.perf_counter() - t0
    timings["total_script_sec"] = time.perf_counter() - _SCRIPT_START
    elapsed = timings["total_script_sec"]
    metrics["elapsed_sec"] = elapsed
    metrics["rtf"] = elapsed / audio_duration if audio_duration > 0 else None
    metrics["timing_breakdown"] = timings
    write_json(args.out_dir / "metrics.json", metrics)
    profile = {
        "asr_segments": str(args.asr_segments),
        "diarization": str(args.diarization),
        "identification": str(args.identification),
        "audio_duration_sec": audio_duration,
        "total_rtf": metrics["rtf"],
        "asr_segment_count": len(asr),
        "diarization_segment_count": len(diar),
        "timings": timings,
    }
    write_json(args.out_dir / "profile_internal.json", profile)
    write_profile_md(args.out_dir / "profile_internal.md", profile)
    print(f"alignment assigned={metrics['assigned_segment_count']} unassigned={unassigned} rtf={metrics['rtf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
