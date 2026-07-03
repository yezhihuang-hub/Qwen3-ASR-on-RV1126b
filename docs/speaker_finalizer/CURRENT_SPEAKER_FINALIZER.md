# Current Speaker Finalizer

## Current backend

- Segmentation: sherpa-onnx pyannote fp32 segmentation
- Embedding: NeMo SpeakerNet
- Threads: segmentation 4, embedding 4, identification 4
- Enrollment: Ye enrollment wav + per-backend cache
- Alignment: ASR committed segments are aligned to sherpa speaker timeline by time overlap
- ASR text rewrite: false

## Current best board result

- Source run: `/data/qwen3asr_rv1126b/outputs/real_two_person_board_mic_capture`
- Output kept: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/outputs/optimization_embedding_nemo_speakernet_20251130_034712`
- Total RTF: `0.776`
- Diarization RTF: `0.616`
- Identification RTF: `0.088`
- Labels: `Ye, Ye, 说话人B, Ye, 说话人B, Ye, 说话人B, Ye, 说话人B`
- Text unchanged: `true`

## Old baseline

- Backend: 3D-Speaker ERes2Net
- Best optimized RTF: `2.592`
- Status: replaced for the current experiment chain

## Rejected variants

- pyannote int8 segmentation: slightly faster, but caused the known third-segment mislabel.
- WeSpeaker CAM++: label errors on the controlled A/B sample.
- WeSpeaker CAM++ large-margin: label errors on the controlled A/B sample.
- Main project `speaker_model.onnx`: not sherpa-compatible as a direct embedding extractor.

## Current recommendation

Keep this chain inside the isolated experiment directory and run the next public diarization benchmark before product integration.

## Exact current command

```bash
EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility

$EXP/scripts/run_finalizer_thread_case.sh \
  --run-dir /data/qwen3asr_rv1126b/outputs/real_two_person_board_mic_capture \
  --tag cleanup_smoke_speakernet \
  --num-speakers 2 \
  --seg-threads 4 \
  --emb-threads 4 \
  --id-threads 4 \
  --segmentation-model $EXP/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --embedding-model $EXP/models/embedding_backends/nemo_en_speakerverification_speakernet.onnx \
  --embedding-name nemo_speakernet \
  --use-cache true \
  --cache-dir $EXP/data/enroll_cache_nemo_speakernet
```

## Kept model paths

- `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`
- `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/nemo_en_speakerverification_speakernet.onnx`
- `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll/Ye.wav`
- `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll_cache_nemo_speakernet`

## Kept report paths

- `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/reports/board_sherpa_embedding_backend_replacement.md`
- `/data/qwen3asr_rv1126b/docs/board_sherpa_embedding_backend_replacement.md`
- `/home/ye/Projects/qwen3asr_reports/board_sherpa_finalizer_feasibility/embedding_replacement_results/board_sherpa_embedding_backend_replacement.md`
