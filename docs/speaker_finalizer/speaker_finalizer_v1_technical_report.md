# RV1126B Qwen3-ASR Speaker Finalizer V1 中文技术报告

## 1. 目标

Speaker Finalizer V1 的目标是把 speaker attribution 做成 Qwen3-ASR-on-RV1126B 的录音后处理链路：

```text
板端 realtime mic ASR
→ 录音中 speaker 关闭，只显示纯 ASR
→ 录音结束后对同一次 raw_mic_16k.wav 做 sherpa speaker diarization
→ 用登记人声纹做 identity recognition
→ 把 speaker label 按时间 overlap 贴回同一次 asr_committed_segments.json
→ 输出最终带说话人标签 transcript
```

本模块不复活旧 CAM++ 多人线，不接 realtime diarization，不改 ASR 文本，不改主工程 `speaker_db`。

## 2. 当前完整链路

```text
raw_mic_16k.wav
→ pyannote fp32 segmentation ONNX
→ NeMo SpeakerNet embedding ONNX
→ sherpa-onnx offline speaker diarization / clustering
→ diarization timeline: speaker_00 / speaker_01 / ...
→ Ye enrollment identification
→ cluster identity scores
→ ASR committed segments overlap alignment
→ final_asr_with_sherpa_speakers.md/json
```

分工必须明确：

| 阶段 | 职责 |
| --- | --- |
| pyannote segmentation | 找 speech activity / speaker change 的时间结构 |
| NeMo SpeakerNet | 只负责 speaker embedding |
| sherpa diarization/clustering | 把时间段聚成 `speaker_00/01/...` |
| Ye identification | 把匿名 cluster 和登记人 embedding 比对 |
| ASR alignment | 按时间 overlap 给 ASR committed segments 贴 label |

NeMo SpeakerNet 不是完整 speaker finalizer，它只是 embedding backend。

## 3. 当前仓库新增内容

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

docs/speaker_finalizer/
├── speaker_finalizer_v1_technical_report.md
├── CURRENT_SPEAKER_FINALIZER.md
├── asr_first_post_record_finalizer_report.md
└── benchmarks/
    ├── board_sherpa_embedding_backend_replacement.md
    ├── public_subset_board_rtf.md
    ├── voxconverse_public_diarization_benchmark_report.md
    ├── voxconverse_backend_summary.csv
    ├── chinese_public_diarization_benchmark_report.md
    ├── chinese_public_diarization_benchmark_report_summary.md
    ├── chinese_backend_summary.csv
    └── chinese_threshold_ablation_summary.csv
```

## 4. 运行方式

### 4.1 产品式 realtime ASR + post-record speaker labeling

```bash
EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility

$EXP/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh \
  --tag mic_known2 \
  --duration 60 \
  --num-speakers 2 \
  --language Chinese \
  --mode mic
```

录音期间：

```text
speaker-mode off
终端只输出 ASR 文本
speaker label 不提前显示
```

录音结束后：

```text
调用完整 sherpa finalizer
打印 FINAL SPEAKER-LABELED TRANSCRIPT
检查 text_unchanged
输出 summary / metrics / run.log
```

### 4.2 未知人数实验模式

```bash
EXP=/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility

$EXP/asr_first_post_record_finalizer/scripts/run_asr_first_post_record_finalizer.sh \
  --tag mic_auto_threshold \
  --duration 60 \
  --cluster-threshold 0.5 \
  --language Chinese \
  --mode mic
```

`--cluster-threshold` 只是透传给 sherpa clustering threshold，不是本仓库自己实现 clustering。当前公开中文会议 ablation 显示 threshold 模式容易过拆，V1 默认仍建议 known speaker count。

## 5. 登记人与陌生人

当前板端实际登记 manifest：

```json
[
  {
    "speaker": "Ye",
    "wav": "/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/data/enroll/Ye.wav"
  }
]
```

识别和显示规则：

| 情况 | 输出 |
| --- | --- |
| cluster 与 Ye score >= 0.60 且 verify 通过 | `Ye` |
| cluster 未通过任何登记人阈值 | `说话人B` / `说话人C` / `说话人D` |
| ASR segment 没有足够 speaker timeline overlap | `说话人未知` |

底层 identification 脚本已经按 manifest 列表遍历登记人，技术上可以支持多条登记人记录。但当前只验证 Ye，一个正式“登记入库”功能还没有产品化。后续需要补：

- 采集登记音频的命令或 UI
- 登记音频质量检查
- manifest 安全更新
- embedding cache 生成和失效管理
- 多登记人相似度冲突处理
- 删除/重命名登记人
- 是否接入主工程 `speaker_db` 的决策

## 6. 当前已验证结果

### 6.1 板端受控 A/B

当前推荐链路：

```text
pyannote fp32 segmentation
+ NeMo SpeakerNet embedding
+ 4/4/4 threads
+ Ye enrollment cache
```

板端旧 A/B 样本结果：

| 指标 | 结果 |
| --- | --- |
| total RTF | 0.776 |
| diarization RTF | 0.616 |
| identification RTF | 0.088 |
| labels | `Ye, Ye, 说话人B, Ye, 说话人B, Ye, 说话人B, Ye, 说话人B` |
| text_unchanged | true |

旧 3D-Speaker ERes2Net backend 板端 RTF 约 2.592，已被当前 SpeakerNet fast chain 替换为默认实验链路。

### 6.2 英文 VoxConverse benchmark

Host full VoxConverse dev+test：

| backend | files | mean RTF | mean relaxed DER |
| --- | ---: | ---: | ---: |
| NeMo SpeakerNet | 448 | 0.0405 | 0.1229 |
| NeMo TitaNet small | 448 | 0.0641 | 0.1309 |
| 3D-Speaker ERes2Net | 448 | 0.1506 | 0.1826 |

英文公开集上，SpeakerNet 同时速度和 DER 最好。

### 6.3 中文公开会议 benchmark

Host AliMeeting Eval/Test + AISHELL-4 Test：

| backend | files | hours | mean RTF | mean relaxed DER |
| --- | ---: | ---: | ---: | ---: |
| NeMo SpeakerNet | 48 | 27.71 | 0.0641 | 0.2825 |
| NeMo TitaNet small | 48 | 27.71 | 0.0948 | 0.3682 |
| 3D-Speaker ERes2Net | 48 | 27.71 | 0.2738 | 0.2187 |

中文会议场景和英文 VoxConverse 结论不同：SpeakerNet 仍最快，但 3D-Speaker ERes2Net DER 更低。建议保留：

```text
SpeakerNet = fast/default backend
3D-Speaker ERes2Net = 中文会议 quality mode 候选
```

### 6.4 Unknown speaker-count threshold ablation

中文公开会议 threshold ablation 使用 SpeakerNet：

| dataset | threshold | mean relaxed DER | speaker-count 结论 |
| --- | ---: | ---: | --- |
| AliMeeting Eval | 0.5 | 0.6687 | 严重过拆 |
| AliMeeting Eval | 0.7 | 0.4753 | 明显过拆 |
| AliMeeting Eval | 0.9 | 0.2386 | 最好但仍过拆 |
| AISHELL-4 Test | 0.5 | 0.4291 | 严重过拆 |
| AISHELL-4 Test | 0.7 | 0.2072 | 明显过拆 |
| AISHELL-4 Test | 0.9 | 0.1630 | 最好但仍过拆 |

结论：`--cluster-threshold` 已可用，但仍是 experimental。可用于未知人数探索，不应替代当前已知人数主路径。

## 7. 和旧 CAM++ 多人线的区别

旧路线：

```text
手写 CAM++ 多 speaker replay
手写 smoothing / display compaction / registered-first fixes
容易出现匿名 label 跳变、混合 label、早期 B 句误标
```

当前路线：

```text
sherpa 官方 diarization/clustering
cluster-first
再做登记人 identification
最后按时间 overlap 贴回 ASR
不改 ASR 文本
不接旧 speaker_db
```

本次入仓不是继续修旧 CAM++ 路线，而是把当前 sherpa finalizer V1 固化成主仓库后处理模块。

## 8. 当前限制

- 当前是 post-record full-file finalizer，不是 streaming diarization。
- 没有后台 chunk diarization，也没有 chunk merge。
- 模型对象没有常驻服务化，finalizer 仍在录音后启动子进程加载模型。
- 只验证 Ye identity recognition；多登记人还没有产品化。
- `--num-speakers N` 支持任意正整数，但产品验证重点仍是 1 人和 2 人。
- `--cluster-threshold` 可用，但 unknown speaker count 产品效果仍需继续验证。
- 本仓库不提交 sherpa ONNX 模型和板端 venv，需要按文档准备。

## 9. 建议下一步

1. 在主仓库里保持 `speaker_finalizer/` 为隔离后处理模块，先不改 ASR core。
2. 增加正式 enrollment 命令：录音、质量检查、manifest 更新、cache 生成。
3. 多做中文产品式 A/B/三人录音验证。
4. 根据产品场景选择 backend：
   - 默认速度优先：NeMo SpeakerNet
   - 中文会议精度优先：评估 3D-Speaker quality mode
5. 如果录音结束后等待仍不可接受，再考虑后台 chunk diarization 或 NPU/RKNN 路线。
