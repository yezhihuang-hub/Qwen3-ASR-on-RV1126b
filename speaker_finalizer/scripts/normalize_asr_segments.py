#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any


BLOCK_RE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}(?:\.\d+)?)-(?P<end>\d{2}:\d{2}(?:\.\d+)?)\](?:\s*(?P<label>.*?):)?\s*$"
)


def parse_ts(text: str) -> float:
    minutes, seconds = text.split(":", 1)
    return int(minutes) * 60.0 + float(seconds)


def normalize_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("segments", "asr_segments", "committed_segments"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"JSON source has no segment list: {path}")

    out = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"segment {idx} is not an object")
        start = item.get("start", item.get("start_sec"))
        end = item.get("end", item.get("end_sec"))
        text = item.get("text", item.get("transcript", ""))
        if start is None or end is None:
            raise ValueError(f"segment {idx} missing start/end")
        out.append(
            {
                "segment_id": idx,
                "start": float(start),
                "end": float(end),
                "text": str(text),
                "source": str(path),
            }
        )
    return out


def normalize_markdown(path: Path) -> list[dict[str, Any]]:
    out = []
    current = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal current, lines
        if current is None:
            return
        out.append(
            {
                "segment_id": len(out),
                "start": current["start"],
                "end": current["end"],
                "text": "\n".join(line for line in lines if line.strip()).strip(),
                "source": str(path),
            }
        )
        current = None
        lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = BLOCK_RE.match(line.strip())
        if match:
            flush()
            current = {
                "start": parse_ts(match.group("start")),
                "end": parse_ts(match.group("end")),
            }
            continue
        if current is not None:
            lines.append(line)
    flush()
    if not out:
        raise ValueError(f"No timestamped transcript blocks found: {path}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    segments = normalize_json(args.source) if args.source.suffix.lower() == ".json" else normalize_markdown(args.source)
    for idx, seg in enumerate(segments):
        seg["segment_id"] = idx
        if seg["end"] < seg["start"]:
            raise ValueError(f"segment {idx} end before start")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"normalized {len(segments)} ASR segments from {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
