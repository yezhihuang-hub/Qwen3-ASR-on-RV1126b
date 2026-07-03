#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/qwen3asr_rv1126b
EXP="$ROOT/experiments/sherpa_finalizer_feasibility"
WRAP="$EXP/asr_first_post_record_finalizer"

ASR_SCRIPT="$ROOT/scripts/run_board_single_user_verify.sh"
FINALIZER_SCRIPT="$EXP/scripts/run_finalizer_thread_case.sh"
PY="$EXP/venv/bin/python"

SEGMENTATION_MODEL="$EXP/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDING_MODEL="$EXP/models/embedding_backends/nemo_en_speakerverification_speakernet.onnx"
EMBEDDING_NAME="nemo_speakernet"
CACHE_DIR="$EXP/data/enroll_cache_nemo_speakernet"
CACHE_FILE="$CACHE_DIR/Ye.embedding.json"
ENROLL_MANIFEST="$EXP/data/enroll/enrollment_manifest.json"
ENROLL_WAV="$EXP/data/enroll/Ye.wav"

TAG=""
DURATION=""
NUM_SPEAKERS=""
CLUSTER_THRESHOLD=""
LANGUAGE="Chinese"
MODE="mic"

usage() {
  cat <<'EOF'
Usage:
  run_asr_first_post_record_finalizer.sh --tag <tag> --duration <seconds> (--num-speakers <N> | --cluster-threshold <X>) [--language Chinese] [--mode mic]

Notes:
  --num-speakers accepts any positive integer and is passed through to sherpa clustering.
  Current V1 acceptance tests cover N=1 and N=2 only; N>2 is allowed but experimental.
  --cluster-threshold enables sherpa unknown speaker-count clustering mode and is experimental in V1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --num-speakers) NUM_SPEAKERS="$2"; shift 2 ;;
    --cluster-threshold) CLUSTER_THRESHOLD="$2"; shift 2 ;;
    --language) LANGUAGE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$TAG" || -z "$DURATION" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Tag must contain only letters, numbers, underscore, dot, or dash" >&2
  exit 2
fi
if [[ -n "$NUM_SPEAKERS" && -n "$CLUSTER_THRESHOLD" ]]; then
  echo "--num-speakers and --cluster-threshold are mutually exclusive" >&2
  exit 2
fi
if [[ -z "$NUM_SPEAKERS" && -z "$CLUSTER_THRESHOLD" ]]; then
  echo "Choose exactly one speaker count mode: --num-speakers <N> or --cluster-threshold <X>" >&2
  usage >&2
  exit 2
fi
if [[ "$MODE" != "mic" ]]; then
  echo "Only --mode mic is supported by this V1 wrapper. File smoke is intentionally skipped." >&2
  exit 2
fi
if [[ -n "$NUM_SPEAKERS" ]]; then
  if [[ ! "$NUM_SPEAKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-speakers must be a positive integer; got: $NUM_SPEAKERS" >&2
    exit 2
  fi
  SPEAKER_COUNT_MODE="known_num_speakers"
  NUM_SPEAKERS_FOR_FINALIZER="$NUM_SPEAKERS"
  NUM_SPEAKERS_JSON="$NUM_SPEAKERS"
  CLUSTER_THRESHOLD_JSON="null"
  if (( NUM_SPEAKERS > 2 )); then
    SPEAKER_COUNT_NOTE="N=$NUM_SPEAKERS is allowed and passed through to sherpa, but 3+ speaker product recordings have not been validated yet."
  else
    SPEAKER_COUNT_NOTE="N=$NUM_SPEAKERS is within the current V1 acceptance scenarios."
  fi
else
  "$PY" - "$CLUSTER_THRESHOLD" <<'PY'
import sys
value = float(sys.argv[1])
if value <= 0:
    raise SystemExit(2)
PY
  SPEAKER_COUNT_MODE="auto_cluster_threshold"
  NUM_SPEAKERS_FOR_FINALIZER=""
  NUM_SPEAKERS_JSON="null"
  CLUSTER_THRESHOLD_JSON="$CLUSTER_THRESHOLD"
  SPEAKER_COUNT_NOTE="cluster_threshold=$CLUSTER_THRESHOLD uses sherpa unknown speaker-count clustering; V1 treats this mode as experimental."
fi

mkdir -p "$WRAP"/{scripts,outputs,reports,logs}
mkdir -p "$ROOT/docs"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$WRAP/outputs/${TAG}_${TS}"
mkdir -p "$OUT"/{logs,asr_run,finalizer_refs}

exec > >(tee -a "$OUT/logs/wrapper.log") 2>&1

if [[ "$SPEAKER_COUNT_MODE" == "known_num_speakers" ]]; then
  WRAPPER_CMD="$0 --tag $TAG --duration $DURATION --num-speakers $NUM_SPEAKERS --language $LANGUAGE --mode $MODE"
else
  WRAPPER_CMD="$0 --tag $TAG --duration $DURATION --cluster-threshold $CLUSTER_THRESHOLD --language $LANGUAGE --mode $MODE"
fi
ASR_RUN_DIR="$OUT/asr_run"
ASR_CMD=(
  "$ASR_SCRIPT"
  --input-source mic
  --duration "$DURATION"
  --wait-for-enter
  --language "$LANGUAGE"
  --speaker-mode off
  --out-dir "$ASR_RUN_DIR"
)
FINALIZER_TAG="${TAG}_post_record_sherpa"
FINALIZER_CMD=(
  "$FINALIZER_SCRIPT"
  --run-dir "$ASR_RUN_DIR"
  --tag "$FINALIZER_TAG"
  --seg-threads 4
  --emb-threads 4
  --id-threads 4
  --segmentation-model "$SEGMENTATION_MODEL"
  --embedding-model "$EMBEDDING_MODEL"
  --embedding-name "$EMBEDDING_NAME"
  --use-cache true
  --cache-dir "$CACHE_DIR"
)
if [[ "$SPEAKER_COUNT_MODE" == "known_num_speakers" ]]; then
  FINALIZER_CMD+=(--num-speakers "$NUM_SPEAKERS")
else
  FINALIZER_CMD+=(--cluster-threshold "$CLUSTER_THRESHOLD")
fi

quote_cmd() {
  printf '%q ' "$@"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing $label: $path" >&2
    return 1
  fi
}

require_exec() {
  local path="$1"
  local label="$2"
  if [[ ! -x "$path" ]]; then
    echo "[ERROR] missing executable $label: $path" >&2
    return 1
  fi
}

now_sec() {
  "$PY" -c 'import time; print(f"{time.perf_counter():.9f}")'
}

json_value() {
  local path="$1"
  local expr="$2"
  "$PY" - "$path" "$expr" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expr = sys.argv[2].split(".")
if not path.is_file():
    print("")
    raise SystemExit(0)
data = json.loads(path.read_text(encoding="utf-8"))
cur = data
for part in expr:
    if isinstance(cur, dict):
        cur = cur.get(part)
    else:
        cur = None
        break
if cur is None:
    print("")
elif isinstance(cur, bool):
    print("true" if cur else "false")
else:
    print(cur)
PY
}

count_asr_segments() {
  local path="$1"
  "$PY" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("")
    raise SystemExit(0)
data = json.loads(path.read_text(encoding="utf-8"))
if isinstance(data, list):
    print(len(data))
elif isinstance(data, dict):
    for key in ("segments", "committed_segments", "asr_committed_segments"):
        if isinstance(data.get(key), list):
            print(len(data[key]))
            break
    else:
        print("")
else:
    print("")
PY
}

audio_duration() {
  local path="$1"
  "$PY" - "$path" <<'PY'
import sys
import wave
from pathlib import Path

path = Path(sys.argv[1])
try:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        print(frames / float(rate) if rate else "")
except Exception:
    print("")
PY
}

PRELOAD_STATUS="OK"
SHERPA_IMPORT_STATUS="FAILED"
SHERPA_IMPORT_DETAIL=""
if "$PY" -c 'import sherpa_onnx; print(getattr(sherpa_onnx, "__version__", "unknown"))' > "$OUT/logs/sherpa_import.txt" 2>&1; then
  SHERPA_IMPORT_STATUS="OK"
  SHERPA_IMPORT_DETAIL="$(tr '\n' ' ' < "$OUT/logs/sherpa_import.txt" | sed 's/[[:space:]]*$//')"
else
  SHERPA_IMPORT_DETAIL="$(tr '\n' ' ' < "$OUT/logs/sherpa_import.txt" | sed 's/[[:space:]]*$//')"
  PRELOAD_STATUS="FAILED"
fi

for item in \
  "$PY:isolated python" \
  "$ASR_SCRIPT:ASR script" \
  "$FINALIZER_SCRIPT:finalizer script" \
  "$SEGMENTATION_MODEL:pyannote fp32 segmentation model" \
  "$EMBEDDING_MODEL:NeMo SpeakerNet embedding model" \
  "$ENROLL_MANIFEST:Ye enrollment manifest" \
  "$ENROLL_WAV:Ye enrollment wav"
do
  path="${item%%:*}"
  label="${item#*:}"
  if [[ "$label" == "isolated python" || "$label" == "ASR script" || "$label" == "finalizer script" ]]; then
    require_exec "$path" "$label" || PRELOAD_STATUS="FAILED"
  else
    require_file "$path" "$label" || PRELOAD_STATUS="FAILED"
  fi
done

if [[ -f "$CACHE_FILE" ]]; then
  CACHE_STATUS="exists: $CACHE_FILE"
else
  CACHE_STATUS="missing: $CACHE_FILE (finalizer can create it during identification if needed)"
fi

{
  echo "========== PRELOADING SPEAKER FINALIZER =========="
  echo "sherpa_onnx import: $SHERPA_IMPORT_STATUS ${SHERPA_IMPORT_DETAIL:+($SHERPA_IMPORT_DETAIL)}"
  echo "segmentation model: $SEGMENTATION_MODEL"
  echo "embedding backend: NeMo SpeakerNet"
  echo "embedding model: $EMBEDDING_MODEL"
  echo "Ye enrollment/cache: manifest=$ENROLL_MANIFEST wav=$ENROLL_WAV cache=$CACHE_STATUS"
  echo "finalizer script: $FINALIZER_SCRIPT"
  echo "preload/check status: $PRELOAD_STATUS"
} | tee "$OUT/preload_check.txt"

if [[ "$PRELOAD_STATUS" != "OK" ]]; then
  echo "[ERROR] preload/check failed; aborting before ASR" >&2
  exit 1
fi

echo
echo "========== REALTIME ASR =========="
echo "command: $(quote_cmd "${ASR_CMD[@]}")"
echo "speaker label during recording: disabled; live label is pending only"
echo
echo "========== RECORDING NOTICE =========="
echo "ASR engine will preload first. When the ASR script prints '[REC] start recording', recording has started."
echo "Please speak clearly near the board microphone during the requested duration."
echo "Speaker labels will not be printed during recording."
echo

ASR_START="$(now_sec)"
set +e
printf '\n' | "${ASR_CMD[@]}" 2>&1 | tee "$OUT/logs/realtime_asr.log"
ASR_STATUS=${PIPESTATUS[1]}
set -e
ASR_END="$(now_sec)"
if [[ "$ASR_STATUS" -ne 0 ]]; then
  echo "[ERROR] realtime ASR failed with status $ASR_STATUS" >&2
  exit "$ASR_STATUS"
fi

RAW_WAV="$ASR_RUN_DIR/raw_mic_16k.wav"
ASR_SEGMENTS="$ASR_RUN_DIR/asr_committed_segments.json"
ASR_TRANSCRIPT="$ASR_RUN_DIR/final_transcript.md"
ASR_METRICS="$ASR_RUN_DIR/metrics.json"

require_file "$RAW_WAV" "raw_mic_16k.wav"
require_file "$ASR_SEGMENTS" "asr_committed_segments.json"
if [[ ! -f "$ASR_TRANSCRIPT" ]]; then
  ASR_TRANSCRIPT=""
  for candidate in "$ASR_RUN_DIR/asr_transcript.md" "$ASR_RUN_DIR/asr_display_paragraphs.md"; do
    if [[ -f "$candidate" ]]; then
      ASR_TRANSCRIPT="$candidate"
      break
    fi
  done
fi
if [[ -z "$ASR_TRANSCRIPT" ]]; then
  echo "[ERROR] ASR transcript not found in $ASR_RUN_DIR" >&2
  exit 1
fi

echo
echo "========== REALTIME ASR DONE =========="
echo "Recording has ended. The wrapper is now starting post-record speaker labeling."
echo "raw_mic_16k.wav: $RAW_WAV"
echo "asr_committed_segments.json: $ASR_SEGMENTS"
echo "ASR transcript: $ASR_TRANSCRIPT"
echo

FINALIZER_START="$(now_sec)"
echo "========== RUNNING POST-RECORD SHERPA SPEAKER FINALIZER =========="
echo "chain: pyannote fp32 segmentation + NeMo SpeakerNet embedding + sherpa diarization/clustering + Ye identification + ASR alignment"
echo "command: $(quote_cmd "${FINALIZER_CMD[@]}")"
set +e
"${FINALIZER_CMD[@]}" 2>&1 | tee "$OUT/logs/finalizer.log"
FINALIZER_STATUS=${PIPESTATUS[0]}
set -e
FINALIZER_END="$(now_sec)"
if [[ "$FINALIZER_STATUS" -ne 0 ]]; then
  echo "[ERROR] post-record finalizer failed with status $FINALIZER_STATUS" >&2
  exit "$FINALIZER_STATUS"
fi

FINALIZER_OUT="$(grep -o '/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/outputs/[^[:space:]]*' "$OUT/logs/finalizer.log" | tail -1 || true)"
if [[ -z "$FINALIZER_OUT" || ! -d "$FINALIZER_OUT" ]]; then
  FINALIZER_OUT="$(find "$EXP/outputs" -maxdepth 1 -type d -name "optimization_${FINALIZER_TAG}_*" | sort | tail -1)"
fi
if [[ -z "$FINALIZER_OUT" || ! -d "$FINALIZER_OUT" ]]; then
  echo "[ERROR] could not locate finalizer output directory" >&2
  exit 1
fi

FINAL_SPEAKER_TRANSCRIPT="$FINALIZER_OUT/final_alignment/final_asr_with_sherpa_speakers.md"
FINAL_ALIGNMENT_JSON="$FINALIZER_OUT/final_alignment/final_asr_with_sherpa_speakers.json"
TEXT_CHECK_JSON="$FINALIZER_OUT/final_alignment/text_unchanged_check.json"
FINALIZER_SUMMARY_JSON="$FINALIZER_OUT/summary.json"
FINALIZER_SUMMARY_MD="$FINALIZER_OUT/summary.md"
require_file "$FINAL_SPEAKER_TRANSCRIPT" "final speaker-labeled transcript"
require_file "$TEXT_CHECK_JSON" "text unchanged check"

cp "$FINAL_SPEAKER_TRANSCRIPT" "$OUT/final_speaker_labeled_transcript.md"
cp "$FINALIZER_SUMMARY_JSON" "$OUT/finalizer_summary.json" 2>/dev/null || true
cp "$FINALIZER_SUMMARY_MD" "$OUT/finalizer_summary.md" 2>/dev/null || true
echo "$FINALIZER_OUT" > "$OUT/finalizer_output_dir.txt"

RAW_DURATION="$(audio_duration "$RAW_WAV")"
ASR_SEGMENT_COUNT="$(count_asr_segments "$ASR_SEGMENTS")"
ASR_STREAMED="$(json_value "$ASR_METRICS" "asr_streamed_during_recording")"
FINALIZER_ELAPSED="$(json_value "$FINALIZER_SUMMARY_JSON" "total_finalizer_sec")"
FINALIZER_RTF="$(json_value "$FINALIZER_SUMMARY_JSON" "total_finalizer_rtf")"
TEXT_UNCHANGED="$(json_value "$TEXT_CHECK_JSON" "text_unchanged")"
DIARIZATION_RTF="$(json_value "$FINALIZER_SUMMARY_JSON" "diarization_rtf")"
IDENTIFICATION_RTF="$(json_value "$FINALIZER_SUMMARY_JSON" "identification_rtf")"
PREDICTED_SPEAKER_COUNT="$(json_value "$FINALIZER_SUMMARY_JSON" "diarization_speaker_count")"

if [[ -z "$FINALIZER_ELAPSED" ]]; then
  FINALIZER_ELAPSED="$("$PY" - "$FINALIZER_START" "$FINALIZER_END" <<'PY'
import sys
print(float(sys.argv[2]) - float(sys.argv[1]))
PY
)"
fi
if [[ -z "$FINALIZER_RTF" && -n "$RAW_DURATION" ]]; then
  FINALIZER_RTF="$("$PY" - "$FINALIZER_ELAPSED" "$RAW_DURATION" <<'PY'
import sys
elapsed = float(sys.argv[1])
duration = float(sys.argv[2])
print(elapsed / duration if duration > 0 else "")
PY
)"
fi

cat > "$OUT/run_summary.json" <<EOF
{
  "mode": "$MODE",
  "tag": "$TAG",
  "timestamp": "$TS",
  "duration_requested_sec": $DURATION,
  "speaker_count_mode": "$SPEAKER_COUNT_MODE",
  "num_speakers": $NUM_SPEAKERS_JSON,
  "cluster_threshold": $CLUSTER_THRESHOLD_JSON,
  "language": "$LANGUAGE",
  "wrapper_output_dir": "$OUT",
  "asr_output_dir": "$ASR_RUN_DIR",
  "raw_wav": "$RAW_WAV",
  "raw_wav_duration_sec": "${RAW_DURATION}",
  "asr_committed_segments": "$ASR_SEGMENTS",
  "asr_committed_segment_count": "${ASR_SEGMENT_COUNT}",
  "asr_transcript": "$ASR_TRANSCRIPT",
  "asr_streamed_during_recording": "${ASR_STREAMED}",
  "finalizer_script": "$FINALIZER_SCRIPT",
  "finalizer_output_dir": "$FINALIZER_OUT",
  "finalizer_elapsed_sec": "${FINALIZER_ELAPSED}",
  "finalizer_rtf": "${FINALIZER_RTF}",
  "predicted_speaker_count": "${PREDICTED_SPEAKER_COUNT}",
  "diarization_rtf": "${DIARIZATION_RTF}",
  "identification_rtf": "${IDENTIFICATION_RTF}",
  "text_unchanged": "${TEXT_UNCHANGED}",
  "final_speaker_labeled_transcript": "$FINAL_SPEAKER_TRANSCRIPT"
}
EOF

echo
echo "========== FINAL SPEAKER-LABELED TRANSCRIPT =========="
cat "$FINAL_SPEAKER_TRANSCRIPT"

echo
echo "========== SUMMARY =========="
echo "mode: $MODE"
echo "duration: $DURATION"
echo "speaker_count_mode: $SPEAKER_COUNT_MODE"
if [[ "$SPEAKER_COUNT_MODE" == "known_num_speakers" ]]; then
  echo "num_speakers: $NUM_SPEAKERS"
else
  echo "cluster_threshold: $CLUSTER_THRESHOLD"
fi
echo "speaker_count_note: $SPEAKER_COUNT_NOTE"
echo "raw_mic_duration_sec: $RAW_DURATION"
echo "asr_streamed_during_recording: $ASR_STREAMED"
echo "asr_segment_count: $ASR_SEGMENT_COUNT"
echo "finalizer_elapsed_sec: $FINALIZER_ELAPSED"
echo "finalizer_rtf: $FINALIZER_RTF"
echo "predicted_speaker_count: $PREDICTED_SPEAKER_COUNT"
echo "diarization_rtf: $DIARIZATION_RTF"
echo "identification_rtf: $IDENTIFICATION_RTF"
echo "text_unchanged: $TEXT_UNCHANGED"
echo "final_transcript_path: $FINAL_SPEAKER_TRANSCRIPT"
echo "wrapper_output_dir: $OUT"

REPORT="$WRAP/reports/asr_first_post_record_finalizer_report.md"
DOC_REPORT="$ROOT/docs/asr_first_post_record_finalizer_report.md"
{
  echo "# ASR First, Post-Record Sherpa Finalizer V1 Report"
  echo
  echo "## 1. 目标"
  echo
  echo "本轮在 RV1126B 板端做最小 V1 产品式 wrapper：录音期间只跑真实 realtime mic ASR，speaker label 关闭；录音结束后，对同一次录音的 raw_mic_16k.wav 运行完整 sherpa speaker finalizer，再按时间 overlap 贴回同一次 asr_committed_segments.json。"
  echo
  echo "## 2. 板端路径"
  echo
  echo "- wrapper root: \`$WRAP\`"
  echo "- wrapper output: \`$OUT\`"
  echo "- ASR output: \`$ASR_RUN_DIR\`"
  echo "- finalizer output: \`$FINALIZER_OUT\`"
  echo
  echo "## 3. Wrapper 完整命令"
  echo
  echo "\`\`\`bash"
  echo "$WRAPPER_CMD"
  echo "\`\`\`"
  echo
  echo "## 4. 复用的 ASR 脚本路径和命令"
  echo
  echo "- ASR script: \`$ASR_SCRIPT\`"
  echo "- speaker mode during recording: \`off\`"
  echo
  echo "\`\`\`bash"
  quote_cmd "${ASR_CMD[@]}"
  echo
  echo "\`\`\`"
  echo
  echo "## 5. 复用的完整 sherpa finalizer 脚本路径和命令"
  echo
  echo "- finalizer script: \`$FINALIZER_SCRIPT\`"
  echo "- 说明：这里使用已有 thread/cached finalizer case 脚本，以匹配当前推荐配置 4/4/4 threads + Ye enrollment cache。"
  echo
  echo "\`\`\`bash"
  quote_cmd "${FINALIZER_CMD[@]}"
  echo
  echo "\`\`\`"
  echo
  echo "## 6. 完整 speaker finalizer 链路"
  echo
  echo "\`raw_mic_16k.wav -> pyannote fp32 segmentation ONNX -> NeMo SpeakerNet embedding backend -> sherpa diarization/clustering -> Ye identification -> ASR overlap alignment -> final speaker-labeled transcript\`"
  echo
  echo "NeMo SpeakerNet 只是 speaker embedding backend，替换的是之前的 3D-Speaker ERes2Net embedding backend；它不是完整 finalizer，不替代 pyannote segmentation，不替代 sherpa diarization/clustering，也不替代最终 ASR alignment。"
  echo
  echo "Diarization 和 identification 是两个独立阶段：sherpa diarization 先输出 \`speaker_00/speaker_01/...\` 时间线；Ye identification 只在 enrollment score 通过阈值时把某个匿名 cluster 映射为 \`Ye\`。其他未登记 speaker 继续显示为 \`说话人B/说话人C/...\`。当前只验证了 Ye identity recognition，没有新增或复用主工程 speaker_db，也没有配置多登记人数据库。"
  echo
  echo "## 7. PRELOADING / CHECK 输出"
  echo
  echo "\`\`\`text"
  cat "$OUT/preload_check.txt"
  echo "\`\`\`"
  echo
  echo "说明：当前没有改成常驻 daemon/service。录音前完成 import、路径和 cache 检查；模型对象仍由 finalizer 子进程在录音结束后加载。"
  echo
  echo "## 8. File / offline smoke test"
  echo
  echo "本轮跳过。V1 wrapper 只支持 \`--mode mic\`，没有为了 file mode 修改 ASR core。"
  echo
  echo "## 9. Realtime mic 测试"
  echo
  echo "- command: \`$WRAPPER_CMD\`"
  echo "- requested duration: $DURATION"
  echo "- speaker_count_mode: $SPEAKER_COUNT_MODE"
  if [[ "$SPEAKER_COUNT_MODE" == "known_num_speakers" ]]; then
    echo "- num_speakers: $NUM_SPEAKERS"
  else
    echo "- cluster_threshold: $CLUSTER_THRESHOLD"
  fi
  echo "- speaker-count note: $SPEAKER_COUNT_NOTE"
  echo "- raw_mic_16k.wav: \`$RAW_WAV\`"
  echo "- raw_mic duration sec: $RAW_DURATION"
  echo "- asr_committed_segments.json: \`$ASR_SEGMENTS\`"
  echo "- ASR segment count: $ASR_SEGMENT_COUNT"
  echo "- ASR streamed during recording: $ASR_STREAMED"
  echo "- finalizer elapsed sec: $FINALIZER_ELAPSED"
  echo "- finalizer RTF: $FINALIZER_RTF"
  echo "- predicted speaker count: $PREDICTED_SPEAKER_COUNT"
  echo "- diarization RTF: $DIARIZATION_RTF"
  echo "- identification RTF: $IDENTIFICATION_RTF"
  echo "- text_unchanged: $TEXT_UNCHANGED"
  echo
  echo "### Realtime ASR transcript"
  echo
  echo "\`\`\`text"
  cat "$ASR_TRANSCRIPT"
  echo "\`\`\`"
  echo
  echo "### Final speaker-labeled transcript"
  echo
  echo "\`\`\`text"
  cat "$FINAL_SPEAKER_TRANSCRIPT"
  echo "\`\`\`"
  echo
  echo "## 10. 录音中输出行为"
  echo
  echo "- 录音中 ASR 命令使用 \`--speaker-mode off\`。"
  echo "- 录音中不输出最终 speaker label；speaker label 只在 post-record sherpa finalizer 结束后输出。"
  echo "- wrapper 会在录音前提示 ASR engine 正在预加载，并说明看到 \`[REC] start recording\` 后即为录音开始；ASR 返回后会提示录音结束并开始 post-record speaker labeling。"
  echo
  echo "## 11. Unknown speaker count / cluster-threshold mode"
  echo
  echo "### 已知人数模式命令"
  echo
  echo "\`\`\`bash"
  echo "$0 --tag mic_known2 --duration 60 --num-speakers 2 --language Chinese --mode mic"
  echo "\`\`\`"
  echo
  echo "### 未知人数模式命令"
  echo
  echo "\`\`\`bash"
  echo "$0 --tag mic_auto_threshold --duration 60 --cluster-threshold 0.5 --language Chinese --mode mic"
  echo "\`\`\`"
  echo
  echo "- \`cluster-threshold\` 只是透传给 sherpa \`FastClusteringConfig.threshold\`，不是本 wrapper 自己实现 clustering。"
  echo "- threshold 模式下，外层 finalizer shell 调用底层 diarization Python 时使用 \`--num-speakers -1 --cluster-threshold X\`。"
  echo "- 当前 run 的 speaker_count_mode: \`$SPEAKER_COUNT_MODE\`。"
  if [[ "$SPEAKER_COUNT_MODE" == "known_num_speakers" ]]; then
    echo "- 当前 run 的 num_speakers: \`$NUM_SPEAKERS\`。"
  else
    echo "- 当前 run 的 cluster_threshold: \`$CLUSTER_THRESHOLD\`。"
  fi
  echo "- 当前 run 的 predicted speaker count: \`$PREDICTED_SPEAKER_COUNT\`。"
  echo "- 当前 run 的 text_unchanged: \`$TEXT_UNCHANGED\`。"
  echo "- no-overlap 导致的 \`说话人未知\` 问题本次不处理；cluster-threshold 只解决未知人数聚类配置。"
  echo
  echo "### 当前 run final transcript"
  echo
  echo "\`\`\`text"
  cat "$FINAL_SPEAKER_TRANSCRIPT"
  echo "\`\`\`"
  echo
  echo "## 12. 当前 V1 是否可用"
  if [[ "$TEXT_UNCHANGED" == "true" && "$ASR_STREAMED" == "true" ]]; then
    echo
    echo "当前 V1 可用：真实 mic ASR 已在录音中运行，录音后完整 sherpa speaker finalizer 成功输出带 speaker label 的 transcript，且 ASR 文本未被改写。"
  else
    echo
    echo "当前 V1 需要复查：asr_streamed_during_recording=$ASR_STREAMED，text_unchanged=$TEXT_UNCHANGED。"
  fi
  echo
  echo "## 13. 局限"
  echo
  echo "- 当前是 post-record full-file finalizer，不是 streaming diarization。"
  echo "- 当前没有实现后台 chunk diarization 或 chunk merge。"
  echo "- 当前 wrapper 不做 benchmark/profiling，只记录一次产品式 run 的必要指标。"
  echo "- 模型对象没有在 ASR 前常驻到 finalizer 进程中，finalizer 仍按现有脚本在录音结束后启动并加载模型。"
  echo "- \`--num-speakers\` 支持任意正整数并透传给 sherpa clustering；当前只验证 N=1/N=2，N>2 产品录音仍是 experimental。"
  echo "- \`--cluster-threshold\` 已支持透传到底层 sherpa clustering，但 threshold 取值和 3+ speaker 产品录音仍需后续验证。"
  echo
  echo "## 14. 后续建议"
  echo
  echo "- 继续做产品式中文 A/B 录音验证。"
  echo "- 如果 post-stop wait 仍过长，再考虑后台 chunk diarization，但不要在 V1 wrapper 里提前引入复杂服务。"
  echo
} > "$REPORT"
cp "$REPORT" "$DOC_REPORT"

echo
echo "[REPORT] $REPORT"
echo "[REPORT] $DOC_REPORT"
