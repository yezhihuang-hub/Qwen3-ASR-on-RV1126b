# Speaker Finalizer V1 中文 README

本目录收录 RV1126B 上当前可用的 speaker finalizer V1 后处理链路。它属于 Qwen3-ASR-on-RV1126B 的录音后处理模块，不是独立项目。

## 1. 当前完整链路

产品式 V1 流程是：

```text
录音期间：
RV1126B mic realtime ASR
→ speaker-mode off
→ 终端只显示实时/提交 ASR 文本
→ 保存 raw_mic_16k.wav
→ 保存 asr_committed_segments.json

录音结束后：
raw_mic_16k.wav
→ pyannote fp32 segmentation ONNX
→ NeMo SpeakerNet embedding ONNX
→ sherpa-onnx diarization / clustering
→ Ye enrollment identification
→ overlap align 到同一次 ASR committed segments
→ 输出 final speaker-labeled transcript
```

注意：NeMo SpeakerNet 只是 speaker embedding backend。它不替代 pyannote segmentation，不替代 sherpa diarization/clustering，不替代 ASR，也不生成 transcript。

## 2. 目录内容

```text
speaker_finalizer/
├── README.md
├── run_board_sherpa_finalizer.sh
├── asr_first_post_record_finalizer/
│   └── scripts/
│       └── run_asr_first_post_record_finalizer.sh
└── scripts/
    ├── audio_utils_board.py
    ├── normalize_asr_segments.py
    ├── run_sherpa_diarization_board.py
    ├── run_sherpa_diarization_thread_case.py
    ├── run_sherpa_diarize_then_identify_board.py
    ├── run_sherpa_diarize_then_identify_thread_board.py
    ├── align_sherpa_to_asr.py
    ├── check_text_unchanged.py
    └── run_finalizer_thread_case.sh
```

这些脚本来自板端隔离实验目录：

```text
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility
```

当前脚本保留板端实验目录的绝对路径约定。实际部署到板端时，建议继续放在：

```text
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility
```

从仓库同步到板端隔离实验目录时，目标结构应是：

```text
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/
├── run_board_sherpa_finalizer.sh
├── scripts/
└── asr_first_post_record_finalizer/scripts/
```

示例：

```bash
EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility

adb shell "mkdir -p $EXP/scripts $EXP/asr_first_post_record_finalizer/scripts"
adb push speaker_finalizer/run_board_sherpa_finalizer.sh $EXP/run_board_sherpa_finalizer.sh
adb push speaker_finalizer/scripts/. $EXP/scripts/
adb push speaker_finalizer/asr_first_post_record_finalizer/scripts/. $EXP/asr_first_post_record_finalizer/scripts/
```

模型、venv、wheelhouse、Ye 登记音频和 cache 不在本仓库提交范围内，需要在板端按实验目录准备。

## 3. 当前推荐配置

```text
segmentation: pyannote fp32 segmentation ONNX
embedding: NeMo SpeakerNet ONNX
threads: segmentation 4 / embedding 4 / identification 4
Ye enrollment cache: enabled
speaker threshold: 0.60
ASR text rewrite: false
```

当前保留模型路径：

```text
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/nemo_en_speakerverification_speakernet.onnx
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll/Ye.wav
/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll_cache_nemo_speakernet
```

本仓库本次只提交脚本和文档，不提交 speaker ONNX 模型、venv、wheelhouse 或登记音频。

## 4. 怎么运行产品式 V1 wrapper

已知人数模式：

```bash
EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility

$EXP/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh \
  --tag mic_known2 \
  --duration 60 \
  --num-speakers 2 \
  --language Chinese \
  --mode mic
```

未知人数实验模式：

```bash
EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility

$EXP/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh \
  --tag mic_auto_threshold \
  --duration 60 \
  --cluster-threshold 0.5 \
  --language Chinese \
  --mode mic
```

`--num-speakers` 和 `--cluster-threshold` 互斥。V1 默认建议用已知人数模式；threshold 模式已经透传到底层 sherpa clustering，但在公开中文会议 ablation 中仍有明显过拆风险。

## 5. 只对已有录音目录做 finalizer

如果已经有某次 ASR 录音目录：

```text
/data/qwen3asr_rv1126b/outputs/<run_dir>/
├── raw_mic_16k.wav
└── asr_committed_segments.json
```

可以直接运行：

```bash
EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility

$EXP/scripts/run_finalizer_thread_case.sh \
  --run-dir /data/qwen3asr_rv1126b/outputs/<run_dir> \
  --tag post_record_speaker_finalizer \
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

## 6. 登记人和陌生人显示规则

当前实际登记库只有 Ye：

```json
[
  {"speaker": "Ye", "wav": "/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll/Ye.wav"}
]
```

显示规则：

```text
sherpa diarization 先输出 speaker_00 / speaker_01 / ...
如果某个 cluster 和 Ye 登记 embedding 分数 >= 0.60 且 verify 通过 -> 显示 Ye
否则未知 cluster 按首次出现顺序显示 说话人B / 说话人C / 说话人D ...
如果 ASR 段没有足够 diarization overlap -> 显示 说话人未知
```

底层 `enrollment_manifest.json` 支持多条登记人记录，identification 脚本会遍历 manifest 并打分。但当前产品化只验证了 Ye，没有做正式“新增登记人”命令、UI、质量检查、冲突处理或主工程 speaker_db 集成。

## 7. 关键报告

- 技术总报告：`docs/speaker_finalizer/speaker_finalizer_v1_technical_report.md`
- V1 wrapper 板端测试：`docs/speaker_finalizer/asr_first_post_record_finalizer_report.md`
- 当前链路摘要：`docs/speaker_finalizer/CURRENT_SPEAKER_FINALIZER.md`
- 板端 embedding backend 替换报告：`docs/speaker_finalizer/benchmarks/board_sherpa_embedding_backend_replacement.md`
- 英文 VoxConverse benchmark：`docs/speaker_finalizer/benchmarks/voxconverse_public_diarization_benchmark_report.md`
- 中文公开会议 benchmark：`docs/speaker_finalizer/benchmarks/chinese_public_diarization_benchmark_report.md`
