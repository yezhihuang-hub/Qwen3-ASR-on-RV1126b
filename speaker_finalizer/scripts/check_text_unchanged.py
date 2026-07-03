#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-segments", type=Path, required=True)
    parser.add_argument("--aligned-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    src = json.loads(args.asr_segments.read_text(encoding="utf-8"))
    dst = json.loads(args.aligned_json.read_text(encoding="utf-8"))
    mismatches = []
    for idx, (a, b) in enumerate(zip(src, dst)):
        if a.get("text") != b.get("text"):
            mismatches.append({"segment_id": idx, "input_text": a.get("text"), "output_text": b.get("text")})
    if len(src) != len(dst):
        mismatches.append({"segment_id": None, "input_count": len(src), "output_count": len(dst), "error": "segment_count_mismatch"})
    result = {
        "text_unchanged": not mismatches,
        "input_segment_count": len(src),
        "output_segment_count": len(dst),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["text_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
