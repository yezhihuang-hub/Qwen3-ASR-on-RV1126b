# ASR First, Post-Record Sherpa Finalizer V1 Report

## 1. 目标

本轮在 RV1126B 板端做最小 V1 产品式 wrapper：录音期间只跑真实 realtime mic ASR，speaker label 关闭；录音结束后，对同一次录音的 raw_mic_16k.wav 运行完整 sherpa speaker finalizer，再按时间 overlap 贴回同一次 asr_committed_segments.json。

## 2. 板端路径

- wrapper root: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer`
- wrapper output: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/outputs/mic_auto_threshold_05_20251130_151122`
- ASR output: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/outputs/mic_auto_threshold_05_20251130_151122/asr_run`
- finalizer output: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/outputs/optimization_mic_auto_threshold_05_post_record_sherpa_20251130_151229`

## 3. Wrapper 完整命令

```bash
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh --tag mic_auto_threshold_05 --duration 45 --cluster-threshold 0.5 --language Chinese --mode mic
```

## 4. 复用的 ASR 脚本路径和命令

- ASR script: `/data/qwen3asr_rv1126b/scripts/run_board_single_user_verify.sh`
- speaker mode during recording: `off`

```bash
/data/qwen3asr_rv1126b/scripts/run_board_single_user_verify.sh --input-source mic --duration 45 --wait-for-enter --language Chinese --speaker-mode off --out-dir /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/outputs/mic_auto_threshold_05_20251130_151122/asr_run
```

## 5. 复用的完整 sherpa finalizer 脚本路径和命令

- finalizer script: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/scripts/run_finalizer_thread_case.sh`
- 说明：这里使用已有 thread/cached finalizer case 脚本，以匹配当前推荐配置 4/4/4 threads + Ye enrollment cache。

```bash
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/scripts/run_finalizer_thread_case.sh --run-dir /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/outputs/mic_auto_threshold_05_20251130_151122/asr_run --tag mic_auto_threshold_05_post_record_sherpa --seg-threads 4 --emb-threads 4 --id-threads 4 --segmentation-model /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx --embedding-model /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/nemo_en_speakerverification_speakernet.onnx --embedding-name nemo_speakernet --use-cache true --cache-dir /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll_cache_nemo_speakernet --cluster-threshold 0.5
```

## 6. 完整 speaker finalizer 链路

`raw_mic_16k.wav -> pyannote fp32 segmentation ONNX -> NeMo SpeakerNet embedding backend -> sherpa diarization/clustering -> Ye identification -> ASR overlap alignment -> final speaker-labeled transcript`

NeMo SpeakerNet 只是 speaker embedding backend，替换的是之前的 3D-Speaker ERes2Net embedding backend；它不是完整 finalizer，不替代 pyannote segmentation，不替代 sherpa diarization/clustering，也不替代最终 ASR alignment。

Diarization 和 identification 是两个独立阶段：sherpa diarization 先输出 `speaker_00/speaker_01/...` 时间线；Ye identification 只在 enrollment score 通过阈值时把某个匿名 cluster 映射为 `Ye`。其他未登记 speaker 继续显示为 `说话人B/说话人C/...`。当前只验证了 Ye identity recognition，没有新增或复用主工程 speaker_db，也没有配置多登记人数据库。

## 7. PRELOADING / CHECK 输出

```text
========== PRELOADING SPEAKER FINALIZER ==========
sherpa_onnx import: OK (1.13.3)
segmentation model: /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx
embedding backend: NeMo SpeakerNet
embedding model: /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/nemo_en_speakerverification_speakernet.onnx
Ye enrollment/cache: manifest=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll/enrollment_manifest.json wav=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll/Ye.wav cache=exists: /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll_cache_nemo_speakernet/Ye.embedding.json
finalizer script: /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/scripts/run_finalizer_thread_case.sh
preload/check status: OK
```

说明：当前没有改成常驻 daemon/service。录音前完成 import、路径和 cache 检查；模型对象仍由 finalizer 子进程在录音结束后加载。

## 8. File / offline smoke test

本轮跳过。V1 wrapper 只支持 `--mode mic`，没有为了 file mode 修改 ASR core。

## 9. Realtime mic 测试

- command: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh --tag mic_auto_threshold_05 --duration 45 --cluster-threshold 0.5 --language Chinese --mode mic`
- requested duration: 45
- speaker_count_mode: auto_cluster_threshold
- cluster_threshold: 0.5
- speaker-count note: cluster_threshold=0.5 uses sherpa unknown speaker-count clustering; V1 treats this mode as experimental.
- raw_mic_16k.wav: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/outputs/mic_auto_threshold_05_20251130_151122/asr_run/raw_mic_16k.wav`
- raw_mic duration sec: 45.0
- asr_committed_segments.json: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/outputs/mic_auto_threshold_05_20251130_151122/asr_run/asr_committed_segments.json`
- ASR segment count: 1
- ASR streamed during recording: true
- finalizer elapsed sec: 19.19823766600166
- finalizer RTF: 0.42662750368892577
- predicted speaker count: 1
- diarization RTF: 0.33009289059991714
- identification RTF: 0.025582381533361818
- text_unchanged: true

### Realtime ASR transcript

```text
# Single User Speaker Verification Transcript

[00:00.00-00:45.00] 说话人待确认:
目前进行未知人数模式录入，请现在开始说
```

### Final speaker-labeled transcript

```text
# Final ASR with sherpa speaker labels

Source:
- ASR: /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/outputs/mic_auto_threshold_05_20251130_151122/asr_run/asr_committed_segments.json
- Diarization: board sherpa-onnx
- Identification: board sherpa Ye enrollment
- ASR text rewritten: False
- Speaker threshold: 0.600

[00:00.00-00:45.00] Ye:
目前进行未知人数模式录入，请现在开始说
```

## 10. 录音中输出行为

- 录音中 ASR 命令使用 `--speaker-mode off`。
- 录音中不输出最终 speaker label；speaker label 只在 post-record sherpa finalizer 结束后输出。
- wrapper 会在录音前提示 ASR engine 正在预加载，并说明看到 `[REC] start recording` 后即为录音开始；ASR 返回后会提示录音结束并开始 post-record speaker labeling。

## 11. Unknown speaker count / cluster-threshold mode

### 已知人数模式命令

```bash
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh --tag mic_known2 --duration 60 --num-speakers 2 --language Chinese --mode mic
```

### 未知人数模式命令

```bash
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh --tag mic_auto_threshold --duration 60 --cluster-threshold 0.5 --language Chinese --mode mic
```

- `cluster-threshold` 只是透传给 sherpa `FastClusteringConfig.threshold`，不是本 wrapper 自己实现 clustering。
- threshold 模式下，外层 finalizer shell 调用底层 diarization Python 时使用 `--num-speakers -1 --cluster-threshold X`。
- 当前 run 的 speaker_count_mode: `auto_cluster_threshold`。
- 当前 run 的 cluster_threshold: `0.5`。
- 当前 run 的 predicted speaker count: `1`。
- 当前 run 的 text_unchanged: `true`。
- no-overlap 导致的 `说话人未知` 问题本次不处理；cluster-threshold 只解决未知人数聚类配置。

### 当前 run final transcript

```text
# Final ASR with sherpa speaker labels

Source:
- ASR: /data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/outputs/mic_auto_threshold_05_20251130_151122/asr_run/asr_committed_segments.json
- Diarization: board sherpa-onnx
- Identification: board sherpa Ye enrollment
- ASR text rewritten: False
- Speaker threshold: 0.600

[00:00.00-00:45.00] Ye:
目前进行未知人数模式录入，请现在开始说
```

## 12. 当前 V1 是否可用

当前 V1 可用：真实 mic ASR 已在录音中运行，录音后完整 sherpa speaker finalizer 成功输出带 speaker label 的 transcript，且 ASR 文本未被改写。

## 13. 局限

- 当前是 post-record full-file finalizer，不是 streaming diarization。
- 当前没有实现后台 chunk diarization 或 chunk merge。
- 当前 wrapper 不做 benchmark/profiling，只记录一次产品式 run 的必要指标。
- 模型对象没有在 ASR 前常驻到 finalizer 进程中，finalizer 仍按现有脚本在录音结束后启动并加载模型。
- `--num-speakers` 支持任意正整数并透传给 sherpa clustering；当前只验证 N=1/N=2，N>2 产品录音仍是 experimental。
- `--cluster-threshold` 已支持透传到底层 sherpa clustering，但 threshold 取值和 3+ speaker 产品录音仍需后续验证。

## 14. 后续建议

- 继续做产品式中文 A/B 录音验证。
- 如果 post-stop wait 仍过长，再考虑后台 chunk diarization，但不要在 V1 wrapper 里提前引入复杂服务。


## 15. Known / threshold regression results

### Known mode regression

Command:

```bash
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh \
  --tag mic_known1_regression \
  --duration 30 \
  --num-speakers 1 \
  --language Chinese \
  --mode mic
```

Result:

- speaker_count_mode: `known_num_speakers`
- num_speakers: `1`
- predicted speaker count: `1`
- finalizer elapsed sec: `14.377740429998084`
- finalizer RTF: `0.4792580143332695`
- diarization RTF: `0.3336575612665911`
- identification RTF: `0.03920409570006692`
- text_unchanged: `true`
- finalizer output: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/outputs/optimization_mic_known1_regression_post_record_sherpa_20251130_151056`

Final transcript:

```text
# Final ASR with sherpa speaker labels

[00:00.00-00:24.38] Ye:
目前在进行实时链路测试，不知道最后一段是否会被录入。

[00:24.38-00:30.00] 说话人未知:
我讲叶之晃。
```

### Unknown cluster-threshold mode test

Command:

```bash
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh \
  --tag mic_auto_threshold_05 \
  --duration 45 \
  --cluster-threshold 0.5 \
  --language Chinese \
  --mode mic
```

Result:

- speaker_count_mode: `auto_cluster_threshold`
- cluster_threshold: `0.5`
- bottom diarization call: `--num-speakers -1 --cluster-threshold 0.5`
- predicted speaker count: `1`
- finalizer elapsed sec: `19.19823766600166`
- finalizer RTF: `0.42662750368892577`
- diarization RTF: `0.33009289059991714`
- identification RTF: `0.025582381533361818`
- text_unchanged: `true`
- finalizer output: `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/outputs/optimization_mic_auto_threshold_05_post_record_sherpa_20251130_151229`

Final transcript:

```text
# Final ASR with sherpa speaker labels

[00:00.00-00:45.00] Ye:
目前进行未知人数模式录入，请现在开始说
```

### Scope note

- 本次只做参数透传：`--cluster-threshold` 进入 sherpa clustering，不自己实现 clustering。
- 完整链路仍然是：pyannote fp32 segmentation + NeMo SpeakerNet embedding backend + sherpa diarization/clustering + Ye identification + ASR overlap alignment。
- no-overlap 导致的 `说话人未知` 不是 cluster-threshold 要解决的问题，本次没有处理。
