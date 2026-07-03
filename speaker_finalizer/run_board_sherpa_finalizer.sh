#!/usr/bin/env bash
set -euo pipefail

EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility
RUN_DIR=""
NUM_SPEAKERS=1
NUM_SPEAKERS_SET=false
CLUSTER_THRESHOLD=""
CLUSTER_THRESHOLD_SET=false
TAG="run"
PROFILE_SYSTEM=false
EMBEDDING_MODEL="$EXP/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
EMBEDDING_NAME="3dspeaker_eres2net"

usage() {
  cat <<'EOF'
Usage:
  run_board_sherpa_finalizer.sh --run-dir <board-output-dir> (--num-speakers <N> | --cluster-threshold <X>) --tag <name> [--profile-system true|false]
                                [--embedding-model <path>] [--embedding-name <name>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --num-speakers) NUM_SPEAKERS="$2"; NUM_SPEAKERS_SET=true; shift 2 ;;
    --cluster-threshold) CLUSTER_THRESHOLD="$2"; CLUSTER_THRESHOLD_SET=true; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --profile-system) PROFILE_SYSTEM="$2"; shift 2 ;;
    --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
    --embedding-name) EMBEDDING_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "--run-dir is required" >&2
  exit 2
fi
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
  SPEAKER_COUNT_MODE="known_num_speakers"
fi
if [[ ! "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Tag must contain only letters, numbers, underscore, dot, or dash" >&2
  exit 2
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "run-dir does not exist: $RUN_DIR" >&2
  exit 1
fi
RAW_WAV="$RUN_DIR/raw_mic_16k.wav"
if [[ ! -f "$RAW_WAV" ]]; then
  echo "raw_mic_16k.wav not found: $RAW_WAV" >&2
  exit 1
fi
if [[ ! -f "$EMBEDDING_MODEL" ]]; then
  echo "embedding model not found: $EMBEDDING_MODEL" >&2
  exit 1
fi

ASR_SOURCE=""
for name in asr_committed_segments.json asr_transcript.md final_transcript.md asr_display_paragraphs.md; do
  if [[ -f "$RUN_DIR/$name" ]]; then
    ASR_SOURCE="$RUN_DIR/$name"
    break
  fi
done

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$EXP/outputs/${TAG}_${TS}"
mkdir -p "$OUT"/{inputs,diarization,identification,final_alignment,logs}
exec > >(tee "$OUT/logs/run.log") 2>&1

if [[ -z "$ASR_SOURCE" ]]; then
  {
    echo "# ASR_SOURCE_NOT_FOUND"
    echo
    echo "run_dir: $RUN_DIR"
    echo
    find "$RUN_DIR" -maxdepth 1 -type f | sort
  } > "$OUT/ASR_SOURCE_NOT_FOUND.md"
  exit 1
fi

PY="$EXP/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing isolated venv python: $PY" >&2
  exit 1
fi

now_sec() {
  "$PY" -c 'import time; print(f"{time.perf_counter():.9f}")'
}

MONITOR_PID=""
stop_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}
trap stop_monitor EXIT

START_TS="$(now_sec)"
echo "[INFO] run_dir=$RUN_DIR"
echo "[INFO] raw_wav=$RAW_WAV"
echo "[INFO] asr_source=$ASR_SOURCE"
echo "[INFO] out=$OUT"
echo "[INFO] speaker_count_mode=$SPEAKER_COUNT_MODE"
echo "[INFO] num_speakers=$NUM_SPEAKERS"
if [[ "$CLUSTER_THRESHOLD_SET" == "true" ]]; then
  echo "[INFO] cluster_threshold=$CLUSTER_THRESHOLD"
fi
echo "[INFO] profile_system=$PROFILE_SYSTEM"
echo "[INFO] embedding_name=$EMBEDDING_NAME"
echo "[INFO] embedding_model=$EMBEDDING_MODEL"

if [[ "$PROFILE_SYSTEM" == "true" ]]; then
  "$EXP/scripts/monitor_freq_temp.sh" "$OUT/logs/freq_temp_trace.csv" &
  MONITOR_PID="$!"
  echo "[INFO] system profile monitor pid=$MONITOR_PID trace=$OUT/logs/freq_temp_trace.csv"
fi

NORMALIZE_START="$(now_sec)"
"$PY" "$EXP/scripts/normalize_asr_segments.py" \
  --source "$ASR_SOURCE" \
  --out "$OUT/inputs/asr_segments_normalized.json"
NORMALIZE_END="$(now_sec)"

DIA_START="$(now_sec)"
DIARIZATION_ARGS=(
  --wav "$RAW_WAV"
  --segmentation-model "$EXP/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
  --embedding-model "$EMBEDDING_MODEL"
  --out-dir "$OUT/diarization"
  --num-speakers "$NUM_SPEAKERS"
)
if [[ "$CLUSTER_THRESHOLD_SET" == "true" ]]; then
  DIARIZATION_ARGS+=(--cluster-threshold "$CLUSTER_THRESHOLD")
fi
"$PY" "$EXP/scripts/run_sherpa_diarization_board.py" \
  "${DIARIZATION_ARGS[@]}"
DIA_END="$(now_sec)"

ID_START="$(now_sec)"
"$PY" "$EXP/scripts/run_sherpa_diarize_then_identify_board.py" \
  --wav "$RAW_WAV" \
  --diarization "$OUT/diarization/diarization_segments.json" \
  --enroll-manifest "$EXP/data/enroll/enrollment_manifest.json" \
  --embedding-model "$EMBEDDING_MODEL" \
  --out-dir "$OUT/identification"
ID_END="$(now_sec)"

ALIGN_START="$(now_sec)"
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
ALIGN_END="$(now_sec)"

TEXT_CHECK_START="$(now_sec)"
"$PY" "$EXP/scripts/check_text_unchanged.py" \
  --asr-segments "$OUT/inputs/asr_segments_normalized.json" \
  --aligned-json "$OUT/final_alignment/final_asr_with_sherpa_speakers.json" \
  --out "$OUT/final_alignment/text_unchanged_check.json"
TEXT_CHECK_END="$(now_sec)"

SUMMARY_START="$(now_sec)"
"$PY" - "$OUT" "$RUN_DIR" "$RAW_WAV" "$NUM_SPEAKERS" "$START_TS" "$NORMALIZE_START" "$NORMALIZE_END" "$DIA_START" "$DIA_END" "$ID_START" "$ID_END" "$ALIGN_START" "$ALIGN_END" "$TEXT_CHECK_START" "$TEXT_CHECK_END" "$SUMMARY_START" "$PROFILE_SYSTEM" "$EMBEDDING_NAME" "$EMBEDDING_MODEL" "$SPEAKER_COUNT_MODE" "${CLUSTER_THRESHOLD:-}" <<'PY'
import json
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
run_dir = sys.argv[2]
raw_wav = sys.argv[3]
num_speakers = int(sys.argv[4])
start_ts = float(sys.argv[5])
norm_s = float(sys.argv[6])
norm_e = float(sys.argv[7])
dia_s = float(sys.argv[8])
dia_e = float(sys.argv[9])
id_s = float(sys.argv[10])
id_e = float(sys.argv[11])
align_s = float(sys.argv[12])
align_e = float(sys.argv[13])
text_s = float(sys.argv[14])
text_e = float(sys.argv[15])
summary_s = float(sys.argv[16])
profile_system = sys.argv[17].lower() == "true"
embedding_name = sys.argv[18]
embedding_model = sys.argv[19]
speaker_count_mode = sys.argv[20]
cluster_threshold = sys.argv[21] or None

def read_json(p, default):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else default

dia = read_json(out / "diarization" / "metrics.json", {})
ident = read_json(out / "identification" / "board_mic_cluster_identification_scores.json", {})
id_metrics = read_json(out / "identification" / "metrics.json", {})
align = read_json(out / "final_alignment" / "metrics.json", {})
text = read_json(out / "final_alignment" / "text_unchanged_check.json", {})
audio = float(dia.get("audio_duration_sec") or 0.0)
summary_e = time.perf_counter()
stage_seconds = {
    "normalize_asr_sec": max(0.0, norm_e - norm_s),
    "diarization_process_sec": max(0.0, dia_e - dia_s),
    "identification_process_sec": max(0.0, id_e - id_s),
    "alignment_process_sec": max(0.0, align_e - align_s),
    "text_check_sec": max(0.0, text_e - text_s),
    "summary_generation_sec": max(0.0, summary_e - summary_s),
}
total = max(0.0, summary_e - start_ts)
stage_percent = {
    key.replace("_sec", "_percent"): (value / total * 100.0 if total > 0 else 0.0)
    for key, value in stage_seconds.items()
}
profile_shell = {
    "run_dir": run_dir,
    "raw_wav": raw_wav,
    "out_dir": str(out),
    "audio_duration_sec": audio,
    "total_finalizer_sec": total,
    "total_finalizer_rtf": total / audio if audio > 0 else None,
    "stage_seconds": stage_seconds,
    "stage_percent": stage_percent,
    "profile_system_enabled": profile_system,
    "embedding_name": embedding_name,
    "embedding_model": embedding_model,
    "speaker_count_mode": speaker_count_mode,
    "cluster_threshold": float(cluster_threshold) if cluster_threshold is not None else None,
    "freq_temp_trace": str(out / "logs" / "freq_temp_trace.csv") if profile_system else None,
    "diarization_profile": str(out / "diarization" / "profile_internal.json"),
    "identification_profile": str(out / "identification" / "profile_internal.json"),
    "alignment_profile": str(out / "final_alignment" / "profile_internal.json"),
}
(out / "profile_shell.json").write_text(json.dumps(profile_shell, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
profile_lines = [
    "# Shell Stage Profile",
    "",
    f"- run_dir: `{run_dir}`",
    f"- raw_wav: `{raw_wav}`",
    f"- audio duration sec: {audio}",
    f"- total finalizer sec: {total}",
    f"- total finalizer RTF: {profile_shell['total_finalizer_rtf']}",
    f"- system profile enabled: {profile_system}",
    "",
    "| stage | sec | percent |",
    "| --- | ---: | ---: |",
]
for key, value in stage_seconds.items():
    pct = value / total * 100.0 if total > 0 else 0.0
    profile_lines.append(f"| {key} | {value:.6f} | {pct:.2f}% |")
(out / "profile_shell.md").write_text("\n".join(profile_lines) + "\n", encoding="utf-8")
summary = {
    "run_dir": run_dir,
    "raw_wav": raw_wav,
    "num_speakers": num_speakers,
    "speaker_count_mode": speaker_count_mode,
    "cluster_threshold": float(cluster_threshold) if cluster_threshold is not None else None,
    "embedding_name": embedding_name,
    "embedding_model": embedding_model,
    "audio_duration_sec": audio,
    "diarization_speaker_count": dia.get("predicted_speaker_count"),
    "diarization_rtf": dia.get("rtf"),
    "identification_rtf": id_metrics.get("rtf"),
    "alignment_rtf": align.get("rtf"),
    "diarization_wall_sec_shell": max(0.0, dia_e - dia_s),
    "identification_wall_sec_shell": max(0.0, id_e - id_s),
    "alignment_wall_sec_shell": max(0.0, align_e - align_s),
    "profile_shell": profile_shell,
    "total_finalizer_elapsed_sec": total,
    "total_finalizer_rtf": total / audio if audio > 0 else None,
    "text_unchanged": text.get("text_unchanged"),
    "assigned_segment_count": align.get("assigned_segment_count"),
    "unassigned_segment_count": align.get("unassigned_segment_count"),
    "identification": ident,
    "final_transcript_path": str(out / "final_alignment" / "final_asr_with_sherpa_speakers.md"),
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
score_lines = []
for label, item in sorted(ident.items()):
    score_lines.append(f"- {label} -> top1={item.get('top1_speaker')} score={item.get('top1_score')} matched={item.get('matched_speaker')} assigned={item.get('assigned_label')}")
lines = [
    "# Board sherpa finalizer summary",
    "",
    f"- run_dir: `{run_dir}`",
    f"- raw_wav: `{raw_wav}`",
    f"- num_speakers: {num_speakers}",
    f"- speaker_count_mode: {speaker_count_mode}",
    f"- cluster_threshold: {cluster_threshold}",
    f"- audio duration: {audio}",
    f"- diarization speaker count: {dia.get('predicted_speaker_count')}",
    f"- diarization RTF: {dia.get('rtf')}",
    f"- identification RTF: {id_metrics.get('rtf')}",
    f"- alignment RTF: {align.get('rtf')}",
    f"- total finalizer elapsed sec: {total}",
    f"- total finalizer RTF: {summary['total_finalizer_rtf']}",
    f"- profile shell: `{out / 'profile_shell.md'}`",
    f"- ASR text unchanged: {text.get('text_unchanged')}",
    f"- final transcript path: `{summary['final_transcript_path']}`",
    "",
    "## Identification",
    "",
    *(score_lines or ["- No identification scores"]),
]
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "========== FINAL SPEAKER-LABELED TRANSCRIPT =========="
cat "$OUT/final_alignment/final_asr_with_sherpa_speakers.md"
stop_monitor
echo "========== SUMMARY =========="
cat "$OUT/summary.md"
echo "[DONE] out=$OUT"
