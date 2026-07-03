# 板端 sherpa embedding backend 替换矩阵报告

## 1. Goal

本轮只替换 sherpa finalizer 使用的 speaker embedding model，不替换 ASR model，也不替换默认 fp32 pyannote segmentation model。目标是验证 NeMo SpeakerNet、NeMo TitaNet、WeSpeaker CAM++、CAM++ large-margin 是否能替代 3D-Speaker ERes2Net，在保持 A/B 标签正确的同时降低 RTF。

## 2. Baseline recap

- 当前最佳 CPU 配置：4/4/4 threads + Ye enrollment cache。
- baseline embedding：3D-Speaker ERes2Net，RTF 2.592，total 114.068s。
- baseline 标签正确，`text_unchanged=true`。

## 3. Candidate model inventory

| name | family | size MB | benchmark | board path |
| --- | --- | ---: | --- | --- |
| 3dspeaker_eres2net_baseline | 3D-Speaker ERes2Net | 37.8 | False | `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` |
| nemo_speakernet | NeMo SpeakerNet | 22.3 | True | `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/nemo_en_speakerverification_speakernet.onnx` |
| nemo_titanet_small | NeMo TitaNet small | 38.4 | True | `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/nemo_en_titanet_small.onnx` |
| wespeaker_campp | WeSpeaker CAM++ | 27.9 | True | `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/wespeaker_en_voxceleb_CAM++.onnx` |
| wespeaker_campp_lm | WeSpeaker CAM++ large-margin | 27.9 | True | `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/wespeaker_en_voxceleb_CAM++_LM.onnx` |
| nemo_titanet_large_extra | NeMo TitaNet large extra | 96.7 | False | `/data/qwen3asr_rv1126b/experiments/sherpa_finalizer_feasibility/models/embedding_backends/nemo_en_titanet_large.onnx` |
| main_project_speaker_model_readonly | main project speaker_model.onnx read-only probe | 27.0 | False | `/data/qwen3asr_rv1126b/models/speaker/speaker_model.onnx` |

## 4. Sherpa compatibility smoke test

| name | init | dim | compatible | cosine Ye/sample | error |
| --- | --- | ---: | --- | ---: | --- |
| 3dspeaker_eres2net_baseline | ok | 512 | yes | 0.705154 |  |
| nemo_speakernet | ok | 256 | yes | 0.702234 |  |
| nemo_titanet_small | ok | 192 | yes | 0.761520 |  |
| wespeaker_campp | ok | 512 | yes | 0.541219 |  |
| wespeaker_campp_lm | ok | 512 | yes | 0.435637 |  |
| nemo_titanet_large_extra | ok | 192 | yes | 0.795223 |  |
| main_project_speaker_model_readonly | fail |  | no |  | single candidate process did not write JSON |

## 5. Full A/B benchmark table

| name | total RTF | diar RTF | id RTF | cluster emb sec | speaker_00 Ye score | speaker_01 Ye score | speakers | segments | labels correct | accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| nemo_speakernet | 0.776 | 0.616 | 0.088 | 2.105 | 0.702 | 0.355 | 2 | 8 | True | True |
| nemo_titanet_small | 0.965 | 0.754 | 0.126 | 2.943 | 0.762 | 0.352 | 2 | 8 | True | True |
| wespeaker_campp | 1.562 | 1.049 | 0.300 | 4.433 | 0.630 | 0.589 | 2 | 8 | False | False |
| wespeaker_campp_lm | 1.562 | 1.047 | 0.301 | 4.456 | 0.744 | 0.465 | 2 | 7 | False | False |

## 6. Label quality comparison

- 质量判定使用上一轮正确输出的 A/B 序列：`Ye, Ye, 说话人B, Ye, 说话人B, Ye, 说话人B, Ye, 说话人B`。
- 还要求 `text_unchanged=true`，并且 diarization speaker count 为 2。
- nemo_speakernet: labels=['Ye', 'Ye', '说话人B', 'Ye', '说话人B', 'Ye', '说话人B', 'Ye', '说话人B'], text_unchanged=True, accepted=True
- nemo_titanet_small: labels=['Ye', 'Ye', '说话人B', 'Ye', '说话人B', 'Ye', '说话人B', 'Ye', '说话人B'], text_unchanged=True, accepted=True
- wespeaker_campp: labels=['Ye', 'Ye', 'Ye', 'Ye', '说话人B', 'Ye', '说话人B', '说话人B', 'Ye'], text_unchanged=True, accepted=False
- wespeaker_campp_lm: labels=['说话人B', '说话人B', '说话人B', '说话人B', '说话人B', '说话人B', '说话人B', 'Ye', 'Ye'], text_unchanged=True, accepted=False

## 7. Best speed model

- raw fastest：nemo_speakernet，RTF 0.776，accepted=True。

## 8. Best quality model

- best accepted：nemo_speakernet，RTF 0.776。

## 9. Recommended replacement

建议将 EXP 实验默认 embedding backend 改为 `nemo_speakernet` 做后续验证。它相对 baseline RTF 2.592 降低 70.1%，且本轮 A/B 标签正确。

## 10. Whether this makes board CPU finalizer product-usable

best accepted RTF 已小于等于 1，板端 CPU finalizer 有进入产品化验证的可能。

## 11. Segmentation cross-test

| name | total RTF | diar RTF | id RTF | cluster emb sec | speaker_00 Ye score | speaker_01 Ye score | speakers | segments | labels correct | accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| nemo_speakernet | 0.776 | 0.616 | 0.088 | 2.105 | 0.702 | 0.355 | 2 | 8 | True | True |
| nemo_speakernet | 0.763 | 0.598 | 0.091 | 2.226 | 0.691 | 0.358 | 2 | 9 | False | False |

如果 int8 segmentation 继续出现已知第三段误贴问题，则不应选择 int8 segmentation。

## 12. Next step

由于 best accepted 已低于实时，下一步应先录多组新的 A/B 和单人样本验证稳定性，再决定是否把 EXP 默认后端切到 SpeakerNet。如果新样本上 RTF 又高于实时或标签不稳定，再考虑 RKNN/NPU 或保留 host/dock finalizer。CAM++/TitaNet/SpeakerNet 在本轮只作为 embedding backend 替换，不等于替代完整 diarization；segmentation、embedding、clustering、speaker attribution、ASR 对齐仍是独立职责。

## 13. Best transcript excerpt

```text
# Final ASR with sherpa speaker labels

Source:
- ASR: /data/qwen3asr_rv1126b/outputs/real_two_person_board_mic_capture/asr_committed_segments.json
- Diarization: board sherpa-onnx
- Identification: board sherpa Ye enrollment
- ASR text rewritten: False
- Speaker threshold: 0.600

[00:00.00-00:02.11] Ye:
我是第一个说话人。

[00:02.11-00:05.85] Ye:
现在开始测试版仔麦克风双人录音。

[00:05.85-00:09.60] 说话人B:
我是第二个说话人，现在接着说话。

[00:09.60-00:15.68] Ye:
今天我们测试是RV幺幺二六B上的离线多人说话人区分。

[00:15.68-00:21.77] 说话人B:
系统应该保持原始转写文本不变，只在前面加说话人标签。

[00:21.77-00:28.55] Ye:
如果标签能在两个人之间合理切换，说明离线多人流程初步可用。

[00:28.55-00:33.00] 说话人B:
如果出现很多说话人，说明聚类还是过拆。

[00:33.00-00:39.09] Ye:
现在我再多说一句：测试同一个说话人，能不能保持一致？

[00:39.09-00:44.00] 说话人B:
```

## 14. Evidence files

- `$EXP/outputs/embedding_backend_smoke_tests.md` / `.json`
- `$EXP/outputs/embedding_backend_benchmark_summary.md` / `.json`
- `$EXP/outputs/embedding_backend_segmentation_crosstest_summary.md` / `.json`
