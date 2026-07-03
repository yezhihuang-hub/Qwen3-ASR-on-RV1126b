#!/usr/bin/env bash
set -euo pipefail

EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility
RUN_DIR=/data/qwen3asr_rv1126b/outputs/real_two_person_board_mic_capture
TAG=thread_case
NUM_SPEAKERS=2
NUM_SPEAKERS_SET=false
CLUSTER_THRESHOLD=""
CLUSTER_THRESHOLD_SET=false
SEG_THREADS=1
EMB_THREADS=1
ID_THREADS=1
SEG_PROVIDER=cpu
EMB_PROVIDER=cpu
ID_PROVIDER=cpu
EMBEDDING_MODEL="$EXP/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
EMBEDDING_NAME="3dspeaker_eres2net"
SEGMENTATION_MODEL="$EXP/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
USE_CACHE=false
CACHE_DIR="$EXP/data/enroll_cache"

usage() {
  cat <<'EOF'
Usage:
  run_finalizer_thread_case.sh --tag <name> [--seg-threads N] [--emb-threads N] [--id-threads N]
                              [--embedding-model PATH] [--embedding-name NAME]
                              [--segmentation-model PATH] [--use-cache true|false]
                              [--num-speakers N | --cluster-threshold X]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --num-speakers) NUM_SPEAKERS="$2"; NUM_SPEAKERS_SET=true; shift 2 ;;
    --cluster-threshold) CLUSTER_THRESHOLD="$2"; CLUSTER_THRESHOLD_SET=true; shift 2 ;;
    --seg-threads) SEG_THREADS="$2"; shift 2 ;;
    --emb-threads) EMB_THREADS="$2"; shift 2 ;;
    --id-threads) ID_THREADS="$2"; shift 2 ;;
    --seg-provider) SEG_PROVIDER="$2"; shift 2 ;;
    --emb-provider) EMB_PROVIDER="$2"; shift 2 ;;
    --id-provider) ID_PROVIDER="$2"; shift 2 ;;
    --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
    --embedding-name) EMBEDDING_NAME="$2"; shift 2 ;;
    --segmentation-model) SEGMENTATION_MODEL="$2"; shift 2 ;;
    --use-cache) USE_CACHE="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$NUM_SPEAKERS_SET" == "true" && "$CLUSTER_THRESHOLD_SET" == "true" ]]; then
  echo "--num-speakers and --cluster-threshold are mutually exclusive" >&2
  exit 2
fi
if [[ "$CLUSTER_THRESHOLD_SET" == "true" ]]; then
  "$EXP/venv/bin/python" - "$CLUSTER_THRESHOLD" <<'PY'
import sys
value = float(sys.argv[1])
if value <= 0:
    raise SystemExit(2)
PY
  NUM_SPEAKERS=-1
  SPEAKER_COUNT_MODE="auto_cluster_threshold"
else
  if [[ ! "$NUM_SPEAKERS" =~ ^-?[0-9]+$ ]]; then
    echo "--num-speakers must be an integer; got: $NUM_SPEAKERS" >&2
    exit 2
  fi
  SPEAKER_COUNT_MODE="known_num_speakers"
fi

PY="$EXP/venv/bin/python"
RAW_WAV="$RUN_DIR/raw_mic_16k.wav"
if [[ ! -x "$PY" ]]; then echo "Missing python: $PY" >&2; exit 1; fi
if [[ ! -f "$RAW_WAV" ]]; then echo "Missing wav: $RAW_WAV" >&2; exit 1; fi
if [[ ! -f "$SEGMENTATION_MODEL" ]]; then echo "Missing segmentation model: $SEGMENTATION_MODEL" >&2; exit 1; fi
if [[ ! -f "$EMBEDDING_MODEL" ]]; then echo "Missing embedding model: $EMBEDDING_MODEL" >&2; exit 1; fi

ASR_SOURCE=""
for name in asr_committed_segments.json asr_transcript.md final_transcript.md asr_display_paragraphs.md; do
  if [[ -f "$RUN_DIR/$name" ]]; then
    ASR_SOURCE="$RUN_DIR/$name"
    break
  fi
done
if [[ -z "$ASR_SOURCE" ]]; then echo "ASR source not found in $RUN_DIR" >&2; exit 1; fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$EXP/outputs/optimization_${TAG}_${TS}"
mkdir -p "$OUT"/{inputs,diarization,identification,final_alignment,logs}
exec > >(tee "$OUT/logs/run.log") 2>&1

now_sec() { "$PY" -c 'import time; print(f"{time.perf_counter():.9f}")'; }

START="$(now_sec)"
echo "[CASE] tag=$TAG out=$OUT"
echo "[CASE] speaker_count_mode=$SPEAKER_COUNT_MODE"
echo "[CASE] num_speakers=$NUM_SPEAKERS"
if [[ "$CLUSTER_THRESHOLD_SET" == "true" ]]; then
  echo "[CASE] cluster_threshold=$CLUSTER_THRESHOLD"
fi
echo "[CASE] seg_threads=$SEG_THREADS emb_threads=$EMB_THREADS id_threads=$ID_THREADS"
echo "[CASE] segmentation_model=$SEGMENTATION_MODEL"
echo "[CASE] embedding_name=$EMBEDDING_NAME"
echo "[CASE] embedding_model=$EMBEDDING_MODEL"

NORM_S="$(now_sec)"
"$PY" "$EXP/scripts/normalize_asr_segments.py" --source "$ASR_SOURCE" --out "$OUT/inputs/asr_segments_normalized.json"
NORM_E="$(now_sec)"

DIA_S="$(now_sec)"
DIARIZATION_ARGS=(
  --wav "$RAW_WAV"
  --segmentation-model "$SEGMENTATION_MODEL"
  --embedding-model "$EMBEDDING_MODEL"
  --out-dir "$OUT/diarization"
  --num-speakers "$NUM_SPEAKERS"
  --segmentation-num-threads "$SEG_THREADS"
  --embedding-num-threads "$EMB_THREADS"
  --segmentation-provider "$SEG_PROVIDER"
  --embedding-provider "$EMB_PROVIDER"
)
if [[ "$CLUSTER_THRESHOLD_SET" == "true" ]]; then
  DIARIZATION_ARGS+=(--cluster-threshold "$CLUSTER_THRESHOLD")
fi
"$PY" "$EXP/scripts/run_sherpa_diarization_thread_case.py" \
  "${DIARIZATION_ARGS[@]}"
DIA_E="$(now_sec)"

ID_S="$(now_sec)"
ID_ARGS=()
if [[ "$USE_CACHE" == "true" ]]; then
  ID_ARGS+=(--use-cache --cache-dir "$CACHE_DIR")
fi
"$PY" "$EXP/scripts/run_sherpa_diarize_then_identify_thread_board.py" \
  --wav "$RAW_WAV" \
  --diarization "$OUT/diarization/diarization_segments.json" \
  --enroll-manifest "$EXP/data/enroll/enrollment_manifest.json" \
  --embedding-model "$EMBEDDING_MODEL" \
  --out-dir "$OUT/identification" \
  --embedding-num-threads "$ID_THREADS" \
  --embedding-provider "$ID_PROVIDER" \
  "${ID_ARGS[@]}"
ID_E="$(now_sec)"

ALIGN_S="$(now_sec)"
"$PY" "$EXP/scripts/align_sherpa_to_asr.py" \
  --asr-segments "$OUT/inputs/asr_segments_normalized.json" \
  --diarization "$OUT/diarization/diarization_segments.json" \
  --identification "$OUT/identification/board_mic_cluster_identification_scores.json" \
  --out-dir "$OUT/final_alignment" \
  --speaker-threshold 0.6 \
  --display-mode registered_unknown \
  --front-fill-sec 3.0 \
  --gap-fill-sec 1.0 \
  --min-overlap-sec 0.20 \
  --min-overlap-ratio 0.10
ALIGN_E="$(now_sec)"

TEXT_S="$(now_sec)"
"$PY" "$EXP/scripts/check_text_unchanged.py" \
  --asr-segments "$OUT/inputs/asr_segments_normalized.json" \
  --aligned-json "$OUT/final_alignment/final_asr_with_sherpa_speakers.json" \
  --out "$OUT/final_alignment/text_unchanged_check.json"
TEXT_E="$(now_sec)"

"$PY" - "$OUT" "$TAG" "$RUN_DIR" "$RAW_WAV" "$SEG_THREADS" "$EMB_THREADS" "$ID_THREADS" "$START" "$NORM_S" "$NORM_E" "$DIA_S" "$DIA_E" "$ID_S" "$ID_E" "$ALIGN_S" "$ALIGN_E" "$TEXT_S" "$TEXT_E" "$USE_CACHE" "$EMBEDDING_NAME" "$EMBEDDING_MODEL" "$SPEAKER_COUNT_MODE" "$NUM_SPEAKERS" "${CLUSTER_THRESHOLD:-}" <<'PY'
import json
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
tag = sys.argv[2]
run_dir = sys.argv[3]
raw_wav = sys.argv[4]
seg_threads = int(sys.argv[5])
emb_threads = int(sys.argv[6])
id_threads = int(sys.argv[7])
start, norm_s, norm_e, dia_s, dia_e, id_s, id_e, align_s, align_e, text_s, text_e = [float(x) for x in sys.argv[8:19]]
use_cache = sys.argv[19].lower() == "true"
embedding_name = sys.argv[20]
embedding_model = sys.argv[21]
speaker_count_mode = sys.argv[22]
num_speakers_arg = int(sys.argv[23])
cluster_threshold = sys.argv[24] or None
end = time.perf_counter()

def load(p, default):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else default

dia = load(out / "diarization" / "metrics.json", {})
ident = load(out / "identification" / "metrics.json", {})
align = load(out / "final_alignment" / "metrics.json", {})
text = load(out / "final_alignment" / "text_unchanged_check.json", {})
aligned = load(out / "final_alignment" / "final_asr_with_sherpa_speakers.json", [])
audio = float(dia.get("audio_duration_sec") or 0.0)
total = max(0.0, end - start)
stage = {
    "normalize_sec": max(0.0, norm_e - norm_s),
    "diarization_process_sec": max(0.0, dia_e - dia_s),
    "identification_process_sec": max(0.0, id_e - id_s),
    "alignment_process_sec": max(0.0, align_e - align_s),
    "text_check_sec": max(0.0, text_e - text_s),
}
labels = [x.get("display_label") for x in aligned]
summary = {
    "tag": tag,
    "out_dir": str(out),
    "run_dir": run_dir,
    "raw_wav": raw_wav,
    "segmentation_num_threads": seg_threads,
    "embedding_num_threads": emb_threads,
    "identification_num_threads": id_threads,
    "embedding_name": embedding_name,
    "embedding_model": embedding_model,
    "speaker_count_mode": speaker_count_mode,
    "num_speakers_arg": num_speakers_arg,
    "cluster_threshold": float(cluster_threshold) if cluster_threshold is not None else None,
    "use_cache": use_cache,
    "audio_duration_sec": audio,
    "total_finalizer_sec": total,
    "total_finalizer_rtf": total / audio if audio > 0 else None,
    "diarization_process_sec_shell": stage["diarization_process_sec"],
    "diarization_inference_sec": dia.get("elapsed_sec"),
    "diarization_rtf": dia.get("rtf"),
    "diarization_total_rtf": dia.get("total_rtf"),
    "diarization_speaker_count": dia.get("predicted_speaker_count"),
    "diarization_segment_count": dia.get("number_of_segments"),
    "identification_sec": ident.get("elapsed_sec"),
    "identification_rtf": ident.get("rtf"),
    "identification_cache_hits": ident.get("cache_hits"),
    "identification_cache_misses": ident.get("cache_misses"),
    "alignment_sec": align.get("elapsed_sec"),
    "alignment_rtf": align.get("rtf"),
    "text_unchanged": text.get("text_unchanged"),
    "display_labels": labels,
    "stage_seconds": stage,
    "final_transcript_path": str(out / "final_alignment" / "final_asr_with_sherpa_speakers.md"),
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = [
    "# Optimization finalizer case summary",
    "",
    f"- tag: {tag}",
    f"- speaker_count_mode: {speaker_count_mode}",
    f"- num_speakers_arg: {num_speakers_arg}",
    f"- cluster_threshold: {cluster_threshold}",
    f"- embedding_name: {embedding_name}",
    f"- embedding_model: {embedding_model}",
    f"- total_finalizer_sec: {summary['total_finalizer_sec']}",
    f"- total_finalizer_rtf: {summary['total_finalizer_rtf']}",
    f"- diarization_rtf: {summary['diarization_rtf']}",
    f"- identification_rtf: {summary['identification_rtf']}",
    f"- text_unchanged: {summary['text_unchanged']}",
    f"- display_labels: {labels}",
]
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "========== FINAL SPEAKER-LABELED TRANSCRIPT =========="
cat "$OUT/final_alignment/final_asr_with_sherpa_speakers.md"
echo "========== SUMMARY =========="
cat "$OUT/summary.md"
echo "[DONE] out=$OUT"
