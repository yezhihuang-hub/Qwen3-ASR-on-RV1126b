# Speaker Finalizer 文档索引

本目录是 Qwen3-ASR-on-RV1126B speaker finalizer V1 的中文文档入口。

## 推荐阅读顺序

1. [技术总报告](speaker_finalizer_v1_technical_report.md)
2. [当前链路摘要](CURRENT_SPEAKER_FINALIZER.md)
3. [ASR first + post-record finalizer 板端测试报告](asr_first_post_record_finalizer_report.md)
4. [板端 embedding backend 替换报告](benchmarks/board_sherpa_embedding_backend_replacement.md)
5. [英文 VoxConverse 公开 benchmark](benchmarks/voxconverse_public_diarization_benchmark_report.md)
6. [中文公开会议 benchmark](benchmarks/chinese_public_diarization_benchmark_report.md)

## 当前推荐链路

```text
realtime mic ASR
→ raw_mic_16k.wav / asr_committed_segments.json
→ pyannote fp32 segmentation
→ NeMo SpeakerNet embedding backend
→ sherpa diarization/clustering
→ Ye identification
→ ASR overlap alignment
→ final speaker-labeled transcript
```

## 重要边界

- 不复活旧 CAM++ 多人线。
- 不修改 ASR 文本。
- 不接主工程 speaker_db。
- 不做 realtime diarization。
- 当前只验证 Ye identity recognition；其他 speaker 以 `说话人B/C/D` 匿名显示。
