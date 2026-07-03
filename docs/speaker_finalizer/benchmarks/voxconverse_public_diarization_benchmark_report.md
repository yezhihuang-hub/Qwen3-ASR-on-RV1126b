# Public Diarization Benchmark Report

## 1. Goal

在 host PC 上使用 VoxConverse dev + test 全量可用 split，对当前 sherpa-onnx speaker diarization 后端做公开数据集 DER/JER/RTF 对比，并准备小 subset 给 RV1126B 只做 RTF/stability 确认。

## 2. Dataset and split

- Dataset: Hugging Face `diarizers-community/voxconverse`.
- Export method: `datasets` streaming + `Audio(decode=False)`，用 `soundfile` 从 WAV bytes 解码并导出 16 kHz mono WAV。
- `Audio(decode=False)` 避免了 torchcodec；本轮没有因为数据集解码安装或使用 PyTorch/torchcodec。sherpa inference 使用 sherpa-onnx CPU。
- dev: 216 recordings, 20.30 hours.
- test: 232 recordings, 43.54 hours.
- combined: 448 recordings, 63.83 hours.

## 3. Full benchmark size

- dev and test were both attempted and completed.
- Core backends completed 448/448 files each.
- Full VoxConverse was not pushed to RV1126B; only the 12-file subset was pushed.

## 4. Metric definitions

- Strict DER: collar=0.0, skip_overlap=False.
- Relaxed DER: collar=0.25, skip_overlap=True.
- JER: `pyannote.metrics.diarization.JaccardErrorRate`, collar=0.0, skip_overlap=False.
- UEM: full audio duration `[0, duration]` for each file.
- Speaker-count error: `abs(pred_num_speakers - ref_num_speakers)`.

## 5. Backend configurations

- Shared segmentation model: pyannote fp32 segmentation ONNX.
- Known speaker count was used: `num_speakers = reference speaker count`.
- CPU calibration selected `num_threads=4`, `workers=2`, `max_total_threads=8` by fastest calibration wall-clock throughput.
- Benchmarked core backends: NeMo SpeakerNet, NeMo TitaNet small, 3D-Speaker ERes2Net.
- Optional WeSpeaker CAM++ / CAM++ LM were not run in the full public benchmark because the required core three-backend benchmark already took several hours and previous board A/B label quality rejected them.

CPU calibration:

| num_threads | workers | max_total_threads | wall_sec | status | successful_files | mean_rtf | median_rtf | sum_sample_elapsed_sec | results_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 8 | 35.7560 | success | 6 | 0.0380 | 0.0390 | 50.9955 | results/voxconverse_full/dev/calibration/threads_4_workers_2 |
| 8 | 1 | 8 | 43.6458 | success | 6 | 0.0307 | 0.0322 | 43.0276 | results/voxconverse_full/dev/calibration/threads_8_workers_1 |
| 16 | 1 | 8 |  | skipped_invalid_thread_budget | 0 |  |  | 0.0000 |  |

## 6. Full host speed table

Combined dev+test:

| backend | successful_files | failed_files | mean_RTF | median_RTF | p95_RTF |
| --- | --- | --- | --- | --- | --- |
| 3dspeaker_eres2net | 448.0000 | 0.0000 | 0.1506 | 0.1539 | 0.1754 |
| nemo_speakernet | 448.0000 | 0.0000 | 0.0405 | 0.0413 | 0.0441 |
| nemo_titanet_small | 448.0000 | 0.0000 | 0.0641 | 0.0655 | 0.0706 |

dev:

| backend | successful_files | failed_files | mean_RTF | median_RTF | p95_RTF |
| --- | --- | --- | --- | --- | --- |
| 3dspeaker_eres2net | 216.0000 | 0.0000 | 0.1538 | 0.1568 | 0.1800 |
| nemo_speakernet | 216.0000 | 0.0000 | 0.0405 | 0.0414 | 0.0441 |
| nemo_titanet_small | 216.0000 | 0.0000 | 0.0641 | 0.0659 | 0.0706 |

test:

| backend | successful_files | failed_files | mean_RTF | median_RTF | p95_RTF |
| --- | --- | --- | --- | --- | --- |
| 3dspeaker_eres2net | 232.0000 | 0.0000 | 0.1476 | 0.1512 | 0.1710 |
| nemo_speakernet | 232.0000 | 0.0000 | 0.0405 | 0.0412 | 0.0440 |
| nemo_titanet_small | 232.0000 | 0.0000 | 0.0641 | 0.0654 | 0.0705 |

## 7. Full host accuracy table

Combined dev+test:

| backend | mean_DER_strict | weighted_DER_strict_by_duration | mean_DER_relaxed | weighted_DER_relaxed_by_duration | mean_JER | mean_abs_speaker_count_error |
| --- | --- | --- | --- | --- | --- | --- |
| 3dspeaker_eres2net | 0.2130 | 0.2365 | 0.1826 | 0.2048 | 0.4158 | 0.4531 |
| nemo_speakernet | 0.1576 | 0.1691 | 0.1229 | 0.1341 | 0.3387 | 0.3683 |
| nemo_titanet_small | 0.1624 | 0.1743 | 0.1309 | 0.1416 | 0.3493 | 0.4241 |

dev:

| backend | mean_DER_strict | weighted_DER_strict_by_duration | mean_DER_relaxed | weighted_DER_relaxed_by_duration | mean_JER | mean_abs_speaker_count_error |
| --- | --- | --- | --- | --- | --- | --- |
| 3dspeaker_eres2net | 0.1892 | 0.2019 | 0.1651 | 0.1754 | 0.3479 | 0.3796 |
| nemo_speakernet | 0.1353 | 0.1236 | 0.1055 | 0.0908 | 0.2661 | 0.3380 |
| nemo_titanet_small | 0.1458 | 0.1471 | 0.1190 | 0.1182 | 0.2800 | 0.3843 |

test:

| backend | mean_DER_strict | weighted_DER_strict_by_duration | mean_DER_relaxed | weighted_DER_relaxed_by_duration | mean_JER | mean_abs_speaker_count_error |
| --- | --- | --- | --- | --- | --- | --- |
| 3dspeaker_eres2net | 0.2351 | 0.2527 | 0.1989 | 0.2185 | 0.4790 | 0.5216 |
| nemo_speakernet | 0.1784 | 0.1903 | 0.1392 | 0.1543 | 0.4062 | 0.3966 |
| nemo_titanet_small | 0.1778 | 0.1870 | 0.1420 | 0.1525 | 0.4138 | 0.4612 |

## 8. Worst-case analysis

See detailed files:

- `results/voxconverse_full/dev/worst_cases.md`
- `results/voxconverse_full/test/worst_cases.md`
- `results/voxconverse_full/combined/worst_cases.md`

High-level observation: NeMo SpeakerNet has the best combined relaxed DER and best speed. 3D-Speaker ERes2Net is much slower and has worse DER/JER on VoxConverse in this setup.

## 9. Speaker count analysis

Mean absolute speaker-count error on combined split:

| backend | mean_abs_speaker_count_error |
| --- | --- |
| 3dspeaker_eres2net | 0.4531 |
| nemo_speakernet | 0.3683 |
| nemo_titanet_small | 0.4241 |

SpeakerNet also has the lowest mean absolute speaker-count error among the three core backends.

## 10. NeMo SpeakerNet vs 3D-Speaker ERes2Net

- SpeakerNet combined mean RTF: 0.0405.
- 3D-Speaker combined mean RTF: 0.1506.
- SpeakerNet combined weighted relaxed DER: 0.1341.
- 3D-Speaker combined weighted relaxed DER: 0.2048.

Conclusion: on this VoxConverse benchmark, SpeakerNet is both faster and more accurate than 3D-Speaker ERes2Net.

## 11. NeMo SpeakerNet vs NeMo TitaNet small

- TitaNet small is slower than SpeakerNet on host.
- TitaNet small is close but worse than SpeakerNet on combined weighted relaxed DER and JER.
- SpeakerNet remains the better default backend from this public benchmark.

## 12. Whether NeMo SpeakerNet remains recommended

Yes. In this benchmark it is the best speed backend and also the best accuracy backend among the three core candidates. This supports keeping pyannote fp32 segmentation + NeMo SpeakerNet as the current default experimental backend.

## 13. Board subset selection

Selected 12 clips covering short, medium, longer, high-DER/harder cases, and 1/2/3/4-speaker examples:

| board_sample_id | duration_sec | num_ref_speakers | DER_relaxed_nemo_speakernet | overlap_ratio |
| --- | --- | --- | --- | --- |
| dev_hqyok | 21.9951 | 1.0000 | 0.0168 | 0.0000 |
| test_fuzfh | 26.0480 | 3.0000 | 0.1203 | 0.0216 |
| dev_tucrg | 26.2080 | 3.0000 | 4.4187 | 0.0000 |
| dev_tfvyr | 27.9772 | 1.0000 | 0.0057 | 0.0000 |
| dev_whmpa | 53.8560 | 2.0000 | 0.0354 | 0.0268 |
| dev_rtvuw | 56.8560 | 3.0000 | 0.1163 | 0.1651 |
| dev_usbgm | 58.3920 | 1.0000 | 0.0000 | 0.0000 |
| test_wdvva | 65.8560 | 2.0000 | 0.0303 | 0.2261 |
| test_vylyk | 161.0880 | 3.0000 | 0.5331 | 0.0080 |
| dev_rcxzg | 178.7821 | 4.0000 | 0.2179 | 0.0309 |
| test_iabca | 179.3920 | 3.0000 | 0.3429 | 0.0830 |
| test_bjruf | 360.5120 | 2.0000 | 0.5554 | 0.1335 |

## 14. Board subset RTF results

- Board subset successful files: 12/12.
- Board total audio duration: 1216.962s.
- Board mean RTF: 0.6143.
- Board median RTF: 0.6237.
- Board P95 RTF: 0.7667.
- Board full summary: `results/board_subset_rtf/public_subset_board_rtf.md`.

## 15. Comparison with RV1126B local A/B result

- Previous RV1126B local A/B product-like chain: NeMo SpeakerNet finalizer RTF about 0.776, labels correct, `text_unchanged=true`.
- Board public subset diarization-only mean RTF: 0.6143, P95 RTF: 0.7667.
- Host benchmark measures generic diarization DER/JER and speed.
- Board local A/B measures product-like Ye identification chain.
- Board public subset measures deployment speed/stability only, not Ye identity recognition.

## 16. Limitations

- VoxConverse is not Ye identification.
- VoxConverse is mostly English/web video, not Chinese product meeting audio.
- Known `num_speakers` was used for fair public diarization comparison; product use may need unknown-speaker or estimated-speaker mode.
- Public DER/JER and product A/B label quality are complementary, not identical.

## 17. Recommendation

- Keep NeMo SpeakerNet as the current best backend.
- Run AliMeeting Eval or a small AISHELL-4 subset next if Chinese DER benchmark is needed.
- For product integration, collect more product-like Chinese A/B and multi-speaker recordings before promoting this out of the experiment directory.
- Push only tiny public or product subsets to RV1126B for RTF confirmation; do not push full datasets to the board.

## Disk usage

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda3        98G   74G   19G  80% /

6.9G	/home/ye/Projects/qwen3asr_reports/public_diarization_benchmark/data
39M	/home/ye/Projects/qwen3asr_reports/public_diarization_benchmark/results
12K	/home/ye/Projects/qwen3asr_reports/public_diarization_benchmark/reports
```

## Key output files

- `results/voxconverse_full/dev/backend_summary.csv`
- `results/voxconverse_full/dev/per_file_metrics.csv`
- `results/voxconverse_full/test/backend_summary.csv`
- `results/voxconverse_full/test/per_file_metrics.csv`
- `results/voxconverse_full/combined/backend_summary.csv`
- `results/voxconverse_full/combined/per_file_metrics.csv`
- `results/voxconverse_full/board_subset_manifest.csv`
- `results/board_subset_rtf/public_subset_board_rtf.md`
