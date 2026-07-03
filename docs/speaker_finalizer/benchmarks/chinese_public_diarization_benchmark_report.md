# 中文公开会议 Speaker Diarization Benchmark

## 1. 摘要结论

- 本报告是 host-side offline benchmark；没有连接 RV1126B，没有运行 ASR，没有 Ye identification，没有产品 wrapper。
- 主 benchmark 覆盖 48 个 scored session，约 27.71 小时音频。
- 主评测使用 known speaker count：`num_speakers = reference unique speaker count`。
- 主 DER 使用 relaxed 设置：collar = 0.25s，skip overlap = True；附带 strict DER：collar = 0.0s，skip overlap = False。
- Speed ranking: nemo_speakernet (0.0641) > nemo_titanet_small (0.0948) > 3dspeaker_eres2net (0.2738)
- DER ranking: 3dspeaker_eres2net (0.2187) > nemo_speakernet (0.2825) > nemo_titanet_small (0.3682)
- 最快 backend: nemo_speakernet (mean_RTF=0.0641)。
- 最低 DER backend: 3dspeaker_eres2net (mean_DER_relaxed=0.2187)。

## 2. 数据集

- AliMeeting: official OpenSLR 119, far-field single channel ch1。Source: https://www.openslr.org/119/
- AISHELL-4: official OpenSLR 111, single channel ch1 when prepared。Source: https://www.openslr.org/111/
- 本轮 AliMeeting 完整跑 Eval + Test；AliMeeting Train far-field archive 约 73.24G，未下载。
- 本轮 AISHELL-4 完整跑 official test；AISHELL-4 train_L/train_M/train_S 合计约 46G，当前磁盘不足，未下载。
- 未下载的 Train split 不伪造 reference、不计入 scored result；主报告只统计实际准备并有 reference 的 scored split。

### Manifest Summary

| manifest              | dataset    | split   |   sessions |    hours |   min_duration_sec |   median_duration_sec |   max_duration_sec | speaker_count_distribution   | channel   | source               |
|:----------------------|:-----------|:--------|-----------:|---------:|-------------------:|----------------------:|-------------------:|:-----------------------------|:----------|:---------------------|
| aishell4_test.jsonl   | aishell4   | test    |         20 | 12.7253  |            2195.43 |               2299.4  |            2393.93 | {'5': 8, '6': 8, '7': 4}     | ch1       | official_openslr_111 |
| alimeeting_eval.jsonl | alimeeting | eval    |          8 |  4.20506 |            1573.85 |               1910.44 |            2239.47 | {'2': 3, '3': 1, '4': 4}     | ch1       | official_openslr_119 |
| alimeeting_test.jsonl | alimeeting | test    |         20 | 10.7765  |            1776.99 |               1960.82 |            2093.01 | {'2': 8, '3': 4, '4': 8}     | ch1       | official_openslr_119 |

## 3. 实验设置

### System

```text
Linux ye-virtual-machine 6.8.0-124-generic #124~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 21:05:19 UTC  x86_64 x86_64 x86_64 GNU/Linux

Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           45 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  8
On-line CPU(s) list:                     0-7
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen 9 7945HX with Radeon Graphics
CPU family:                              25
Model:                                   97
Thread(s) per core:                      1
Core(s) per socket:                      4
Socket(s):                               2
Stepping:                                2
BogoMIPS:                                4990.52
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ibpb vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_bf16 clzero arat avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor fsrm
Hypervisor vendor:                       VMware
Virtualization type:                     full
L1d cache:                               256 KiB (8 instances)
L1i cache:                               256 KiB (8 instances)
L2 cache:                                8 MiB (8 instances)
L3 cache:                                64 MiB (2 instances)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-7
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Vulnerable: Safe RET, no microcode
Vulnerability Spec store bypass:         Vulnerable
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Retpolines; IBPB conditional; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Vulnerable: Clear CPU buffers attempted, no microcode
Vulnerability Tsx async abort:           Not affected

Filesystem      Size  Used Avail Use% Mounted on
/dev/sda3        98G   75G   19G  81% /
```

### Python / Packages

```text
python_executable=/home/ye/Projects/qwen3asr_reports/chinese_diarization_benchmark/venv/bin/python
python_version=3.13.12 | packaged by conda-forge | (main, Feb  5 2026, 05:53:46) [GCC 14.3.0]
platform=Linux-6.8.0-124-generic-x86_64-with-glibc2.35
machine=x86_64
cpu_count=8
sherpa_onnx=1.13.3
numpy=2.5.0
scipy=1.18.0
pandas=3.0.3
pyannote_metrics=4.1
```

### Models

# Model Inventory

| purpose | backend | path | size | source |
| --- | --- | --- | ---: | --- |
| segmentation | pyannote fp32 segmentation | `models/segmentation/pyannote_segmentation_3_0_fp32.onnx` | 5992913 | local copy from sherpa_onnx_speaker_fresh |
| embedding | 3D-Speaker ERes2Net | `models/embedding_backends/3dspeaker_eres2net.onnx` | 39593761 | local copy from sherpa_onnx_speaker_fresh |
| embedding | NeMo SpeakerNet | `models/embedding_backends/nemo_speakernet.onnx` | 23411863 | local copy from embedding_replacement_assets |
| embedding | NeMo TitaNet small | `models/embedding_backends/nemo_titanet_small.onnx` | 40257283 | local copy from embedding_replacement_assets |
| embedding optional smoke | WeSpeaker CAM++ | `models/embedding_backends/wespeaker_campp.onnx` | 28901840 | local copy from embedding_replacement_assets |
| embedding optional smoke | WeSpeaker CAM++ LM | `models/embedding_backends/wespeaker_campp_lm.onnx` | 28901840 | local copy from embedding_replacement_assets |

## sha256

```text
220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079  models/segmentation/pyannote_segmentation_3_0_fp32.onnx
1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b  models/embedding_backends/3dspeaker_eres2net.onnx
d204dc8aac0014b8543f05fc8e310510c7022bc65b6452c203ec205ef7a66b23  models/embedding_backends/nemo_speakernet.onnx
ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e  models/embedding_backends/nemo_titanet_small.onnx
```


### Optional CAM++ Smoke

CAM++ / CAM++ LM 只做 AliMeeting Eval 前 3 条 smoke，不进入 full benchmark。原因：smoke DER 明显偏高，且此前产品 A/B 侧也出现过标签质量问题；为保证核心三 backend 完整性，本轮不做 CAM++ full run。

| backend            |   successful_files |   failed_files |   mean_RTF |   mean_DER_relaxed |   mean_JER |
|:-------------------|-------------------:|---------------:|-----------:|-------------------:|-----------:|
| wespeaker_campp    |                  3 |              0 |     0.0737 |             0.6181 |     0.7126 |
| wespeaker_campp_lm |                  3 |              0 |     0.0736 |             0.6181 |     0.7499 |


## 4. Calibration

| config            |   workers |   num_threads |   wall_sec |   wall_rtf |   mean_file_rtf | selected   |
|:------------------|----------:|--------------:|-----------:|-----------:|----------------:|:-----------|
| threads4_workers2 |         2 |             4 |     258.45 |      0.023 |          0.0449 | False      |
| threads8_workers1 |         1 |             8 |     405.36 |      0.036 |          0.0359 | False      |
| threads2_workers4 |         4 |             2 |     224.55 |      0.02  |          0.0612 | True       |


## 5. Main Speed Table

### aishell4/test

| backend            |   successful_files |   failed_files |   total_audio_hours |   mean_RTF |   median_RTF |   p95_RTF |
|:-------------------|-------------------:|---------------:|--------------------:|-----------:|-------------:|----------:|
| 3dspeaker_eres2net |                 20 |              0 |             12.7253 |     0.3052 |       0.307  |    0.3205 |
| nemo_speakernet    |                 20 |              0 |             12.7253 |     0.0632 |       0.0634 |    0.0661 |
| nemo_titanet_small |                 20 |              0 |             12.7253 |     0.0989 |       0.0988 |    0.1045 |


### alimeeting/eval

| backend            |   successful_files |   failed_files |   total_audio_hours |   mean_RTF |   median_RTF |   p95_RTF |
|:-------------------|-------------------:|---------------:|--------------------:|-----------:|-------------:|----------:|
| 3dspeaker_eres2net |                  8 |              0 |              4.2051 |     0.2666 |       0.2774 |    0.3216 |
| nemo_speakernet    |                  8 |              0 |              4.2051 |     0.0648 |       0.0651 |    0.0701 |
| nemo_titanet_small |                  8 |              0 |              4.2051 |     0.0966 |       0.0971 |    0.105  |


### alimeeting/test

| backend            |   successful_files |   failed_files |   total_audio_hours |   mean_RTF |   median_RTF |   p95_RTF |
|:-------------------|-------------------:|---------------:|--------------------:|-----------:|-------------:|----------:|
| 3dspeaker_eres2net |                 20 |              0 |             10.7765 |     0.2453 |       0.2554 |    0.2964 |
| nemo_speakernet    |                 20 |              0 |             10.7765 |     0.0648 |       0.0651 |    0.0722 |
| nemo_titanet_small |                 20 |              0 |             10.7765 |     0.0898 |       0.0917 |    0.1014 |


### aishell4/combined

| backend            |   successful_files |   failed_files |   total_audio_hours |   mean_RTF |   median_RTF |   p95_RTF |
|:-------------------|-------------------:|---------------:|--------------------:|-----------:|-------------:|----------:|
| 3dspeaker_eres2net |                 20 |              0 |             12.7253 |     0.3052 |       0.307  |    0.3205 |
| nemo_speakernet    |                 20 |              0 |             12.7253 |     0.0632 |       0.0634 |    0.0661 |
| nemo_titanet_small |                 20 |              0 |             12.7253 |     0.0989 |       0.0988 |    0.1045 |


### alimeeting/combined

| backend            |   successful_files |   failed_files |   total_audio_hours |   mean_RTF |   median_RTF |   p95_RTF |
|:-------------------|-------------------:|---------------:|--------------------:|-----------:|-------------:|----------:|
| 3dspeaker_eres2net |                 28 |              0 |             14.9816 |     0.2514 |       0.2683 |    0.3083 |
| nemo_speakernet    |                 28 |              0 |             14.9816 |     0.0648 |       0.0651 |    0.0719 |
| nemo_titanet_small |                 28 |              0 |             14.9816 |     0.0918 |       0.096  |    0.1021 |


### combined

| backend            |   successful_files |   failed_files |   total_audio_hours |   mean_RTF |   median_RTF |   p95_RTF |
|:-------------------|-------------------:|---------------:|--------------------:|-----------:|-------------:|----------:|
| 3dspeaker_eres2net |                 48 |              0 |             27.7069 |     0.2738 |       0.2891 |    0.3202 |
| nemo_speakernet    |                 48 |              0 |             27.7069 |     0.0641 |       0.064  |    0.0713 |
| nemo_titanet_small |                 48 |              0 |             27.7069 |     0.0948 |       0.0972 |    0.1042 |


## 6. Main DER Table

### aishell4/test

| backend            |   mean_DER_relaxed |   median_DER_relaxed |   p95_DER_relaxed |   mean_miss |   mean_fa |   mean_confusion |   mean_JER |
|:-------------------|-------------------:|---------------------:|------------------:|------------:|----------:|-----------------:|-----------:|
| 3dspeaker_eres2net |             0.175  |               0.1485 |            0.3538 |      0.0113 |    0.0287 |           0.1351 |     0.4558 |
| nemo_speakernet    |             0.2664 |               0.2855 |            0.4013 |      0.0113 |    0.0291 |           0.2259 |     0.5464 |
| nemo_titanet_small |             0.3616 |               0.3744 |            0.5798 |      0.0112 |    0.0287 |           0.3217 |     0.6435 |


### alimeeting/eval

| backend            |   mean_DER_relaxed |   median_DER_relaxed |   p95_DER_relaxed |   mean_miss |   mean_fa |   mean_confusion |   mean_JER |
|:-------------------|-------------------:|---------------------:|------------------:|------------:|----------:|-----------------:|-----------:|
| 3dspeaker_eres2net |             0.192  |               0.1754 |            0.4202 |      0.0042 |    0.041  |           0.1468 |     0.3585 |
| nemo_speakernet    |             0.2583 |               0.3512 |            0.4153 |      0.0041 |    0.0422 |           0.2119 |     0.4385 |
| nemo_titanet_small |             0.3687 |               0.3688 |            0.6018 |      0.004  |    0.0411 |           0.3236 |     0.568  |


### alimeeting/test

| backend            |   mean_DER_relaxed |   median_DER_relaxed |   p95_DER_relaxed |   mean_miss |   mean_fa |   mean_confusion |   mean_JER |
|:-------------------|-------------------:|---------------------:|------------------:|------------:|----------:|-----------------:|-----------:|
| 3dspeaker_eres2net |             0.2732 |               0.3108 |            0.4994 |      0.0073 |    0.0566 |           0.2093 |     0.4477 |
| nemo_speakernet    |             0.3084 |               0.3206 |            0.5877 |      0.0073 |    0.0557 |           0.2454 |     0.5035 |
| nemo_titanet_small |             0.3746 |               0.4287 |            0.5891 |      0.0071 |    0.0556 |           0.312  |     0.5971 |


### aishell4/combined

| backend            |   mean_DER_relaxed |   median_DER_relaxed |   p95_DER_relaxed |   mean_miss |   mean_fa |   mean_confusion |   mean_JER |
|:-------------------|-------------------:|---------------------:|------------------:|------------:|----------:|-----------------:|-----------:|
| 3dspeaker_eres2net |             0.175  |               0.1485 |            0.3538 |      0.0113 |    0.0287 |           0.1351 |     0.4558 |
| nemo_speakernet    |             0.2664 |               0.2855 |            0.4013 |      0.0113 |    0.0291 |           0.2259 |     0.5464 |
| nemo_titanet_small |             0.3616 |               0.3744 |            0.5798 |      0.0112 |    0.0287 |           0.3217 |     0.6435 |


### alimeeting/combined

| backend            |   mean_DER_relaxed |   median_DER_relaxed |   p95_DER_relaxed |   mean_miss |   mean_fa |   mean_confusion |   mean_JER |
|:-------------------|-------------------:|---------------------:|------------------:|------------:|----------:|-----------------:|-----------:|
| 3dspeaker_eres2net |             0.25   |               0.2553 |            0.4968 |      0.0064 |    0.0521 |           0.1914 |     0.4222 |
| nemo_speakernet    |             0.2941 |               0.3339 |            0.5434 |      0.0064 |    0.0518 |           0.2359 |     0.4849 |
| nemo_titanet_small |             0.3729 |               0.4094 |            0.593  |      0.0062 |    0.0514 |           0.3153 |     0.5887 |


### combined

| backend            |   mean_DER_relaxed |   median_DER_relaxed |   p95_DER_relaxed |   mean_miss |   mean_fa |   mean_confusion |   mean_JER |
|:-------------------|-------------------:|---------------------:|------------------:|------------:|----------:|-----------------:|-----------:|
| 3dspeaker_eres2net |             0.2187 |               0.173  |            0.4849 |      0.0084 |    0.0424 |           0.168  |     0.4362 |
| nemo_speakernet    |             0.2825 |               0.3042 |            0.4629 |      0.0084 |    0.0424 |           0.2317 |     0.5105 |
| nemo_titanet_small |             0.3682 |               0.3796 |            0.593  |      0.0083 |    0.042  |           0.318  |     0.6116 |


## 7. Ranking

- Speed ranking: nemo_speakernet (0.0641) > nemo_titanet_small (0.0948) > 3dspeaker_eres2net (0.2738)
- DER ranking: 3dspeaker_eres2net (0.2187) > nemo_speakernet (0.2825) > nemo_titanet_small (0.3682)
- 英文 VoxConverse 结论只作为背景对照；中文会议场景以本报告 DER/RTF 为准。

## 8. Threshold Ablation

| dataset    | split   | backend         |   threshold |   successful_files |   failed_files |   total_audio_hours |   mean_RTF |   median_RTF |   mean_DER_relaxed |   median_DER_relaxed |   p95_DER_relaxed |   exact_speaker_count_match_rate |   under_split_rate |   over_split_rate |   mean_abs_speaker_count_error |   median_abs_speaker_count_error |   mean_pred_num_speakers |   mean_ref_num_speakers |
|:-----------|:--------|:----------------|------------:|-------------------:|---------------:|--------------------:|-----------:|-------------:|-------------------:|---------------------:|------------------:|---------------------------------:|-------------------:|------------------:|-------------------------------:|---------------------------------:|-------------------------:|------------------------:|
| aishell4   | test    | nemo_speakernet |         0.5 |                 20 |              0 |             12.7253 |     0.0635 |       0.0636 |             0.4291 |               0.4553 |            0.7248 |                             0    |               0    |               1   |                         90.3   |                             83.5 |                   96.1   |                   5.8   |
| aishell4   | test    | nemo_speakernet |         0.7 |                 20 |              0 |             12.7253 |     0.0633 |       0.0632 |             0.2072 |               0.1764 |            0.4285 |                             0    |               0    |               1   |                         39.25  |                             40   |                   45.05  |                   5.8   |
| aishell4   | test    | nemo_speakernet |         0.9 |                 20 |              0 |             12.7253 |     0.0633 |       0.0632 |             0.163  |               0.1684 |            0.2862 |                             0.05 |               0.05 |               0.9 |                          8.4   |                              9.5 |                   14.1   |                   5.8   |
| alimeeting | eval    | nemo_speakernet |         0.5 |                  8 |              0 |              4.2051 |     0.0609 |       0.0613 |             0.6687 |               0.704  |            0.8847 |                             0    |               0    |               1   |                        153.625 |                            153.5 |                  156.75  |                   3.125 |
| alimeeting | eval    | nemo_speakernet |         0.7 |                  8 |              0 |              4.2051 |     0.0592 |       0.0603 |             0.4753 |               0.4357 |            0.7608 |                             0    |               0    |               1   |                         61.375 |                             62   |                   64.5   |                   3.125 |
| alimeeting | eval    | nemo_speakernet |         0.9 |                  8 |              0 |              4.2051 |     0.059  |       0.06   |             0.2386 |               0.2766 |            0.408  |                             0    |               0    |               1   |                         15.5   |                             16   |                   18.625 |                   3.125 |


## 9. Case Study

### Best DER samples

| dataset    | split   | sample_id   | backend            |   DER_relaxed |    rtf |   ref_num_speakers |   pred_num_speakers |
|:-----------|:--------|:------------|:-------------------|--------------:|-------:|-------------------:|--------------------:|
| alimeeting | eval    | R8009_M8020 | nemo_speakernet    |        0.0256 | 0.0649 |                  2 |                   2 |
| alimeeting | eval    | R8009_M8020 | 3dspeaker_eres2net |        0.0261 | 0.2872 |                  2 |                   2 |
| alimeeting | eval    | R8009_M8020 | nemo_titanet_small |        0.0272 | 0.0957 |                  2 |                   2 |
| alimeeting | test    | R8009_M8025 | 3dspeaker_eres2net |        0.0319 | 0.3108 |                  2 |                   2 |
| alimeeting | test    | R8009_M8028 | 3dspeaker_eres2net |        0.0349 | 0.2739 |                  2 |                   2 |
| alimeeting | test    | R8009_M8028 | nemo_speakernet    |        0.0363 | 0.0616 |                  2 |                   2 |
| alimeeting | test    | R8009_M8021 | nemo_titanet_small |        0.039  | 0.0895 |                  2 |                   2 |
| alimeeting | eval    | R8009_M8019 | 3dspeaker_eres2net |        0.0415 | 0.2926 |                  2 |                   2 |
| alimeeting | test    | R8009_M8021 | nemo_speakernet    |        0.0442 | 0.0651 |                  2 |                   2 |
| alimeeting | test    | R8009_M8022 | 3dspeaker_eres2net |        0.0478 | 0.287  |                  2 |                   2 |


### Worst DER samples

| dataset    | split   | sample_id    | backend            |   DER_relaxed |    rtf |   ref_num_speakers |   pred_num_speakers |
|:-----------|:--------|:-------------|:-------------------|--------------:|-------:|-------------------:|--------------------:|
| alimeeting | test    | R8006_M8012  | nemo_speakernet    |        0.6406 | 0.0667 |                  4 |                   4 |
| alimeeting | eval    | R8008_M8013  | nemo_titanet_small |        0.6348 | 0.0978 |                  3 |                   3 |
| aishell4   | test    | M_R003S05C01 | nemo_titanet_small |        0.6024 | 0.0984 |                  6 |                   6 |
| alimeeting | test    | R8008_M8017  | nemo_titanet_small |        0.5953 | 0.0964 |                  3 |                   3 |
| alimeeting | test    | R8008_M8016  | nemo_titanet_small |        0.5888 | 0.0963 |                  3 |                   3 |
| alimeeting | test    | R8008_M8017  | nemo_speakernet    |        0.5849 | 0.0719 |                  3 |                   3 |
| aishell4   | test    | L_R004S06C01 | nemo_titanet_small |        0.5786 | 0.1025 |                  7 |                   5 |
| aishell4   | test    | S_R004S04C01 | nemo_titanet_small |        0.5757 | 0.0976 |                  5 |                   4 |
| alimeeting | test    | R8005_M8009  | nemo_titanet_small |        0.5434 | 0.0648 |                  4 |                   4 |
| alimeeting | eval    | R8001_M8004  | nemo_titanet_small |        0.5405 | 0.0965 |                  4 |                   4 |


### Slowest samples

| dataset    | split   | sample_id    | backend            |    rtf |   DER_relaxed |
|:-----------|:--------|:-------------|:-------------------|-------:|--------------:|
| alimeeting | eval    | R8009_M8018  | 3dspeaker_eres2net | 0.3313 |        0.0507 |
| aishell4   | test    | L_R004S01C01 | 3dspeaker_eres2net | 0.3212 |        0.0872 |
| aishell4   | test    | S_R004S04C01 | 3dspeaker_eres2net | 0.3205 |        0.1185 |
| aishell4   | test    | L_R004S06C01 | 3dspeaker_eres2net | 0.3196 |        0.3103 |
| aishell4   | test    | S_R004S02C01 | 3dspeaker_eres2net | 0.3192 |        0.0818 |
| aishell4   | test    | L_R004S03C01 | 3dspeaker_eres2net | 0.317  |        0.1448 |
| aishell4   | test    | S_R004S03C01 | 3dspeaker_eres2net | 0.315  |        0.0947 |
| aishell4   | test    | L_R004S02C01 | 3dspeaker_eres2net | 0.3143 |        0.1113 |
| aishell4   | test    | S_R003S01C01 | 3dspeaker_eres2net | 0.3111 |        0.2589 |
| alimeeting | test    | R8009_M8025  | 3dspeaker_eres2net | 0.3108 |        0.0319 |


### Speaker-count error samples

| dataset   | split   | sample_id    | backend            |   ref_num_speakers |   pred_num_speakers |   speaker_count_error |   DER_relaxed |
|:----------|:--------|:-------------|:-------------------|-------------------:|--------------------:|----------------------:|--------------:|
| aishell4  | test    | L_R004S06C01 | nemo_titanet_small |                  7 |                   5 |                    -2 |        0.5786 |
| aishell4  | test    | S_R004S03C01 | nemo_speakernet    |                  5 |                   4 |                    -1 |        0.2751 |
| aishell4  | test    | L_R004S02C01 | nemo_titanet_small |                  7 |                   6 |                    -1 |        0.3834 |
| aishell4  | test    | S_R004S01C01 | nemo_titanet_small |                  5 |                   4 |                    -1 |        0.2273 |
| aishell4  | test    | S_R004S02C01 | nemo_titanet_small |                  5 |                   4 |                    -1 |        0.4137 |
| aishell4  | test    | S_R004S03C01 | nemo_titanet_small |                  5 |                   4 |                    -1 |        0.419  |
| aishell4  | test    | S_R004S04C01 | nemo_titanet_small |                  5 |                   4 |                    -1 |        0.5757 |
| aishell4  | test    | L_R004S06C01 | 3dspeaker_eres2net |                  7 |                   6 |                    -1 |        0.3103 |
| aishell4  | test    | S_R004S03C01 | 3dspeaker_eres2net |                  5 |                   4 |                    -1 |        0.0947 |


## 10. 结论

- 中文会议 known speaker count 主结果中，速度最优是 nemo_speakernet (mean_RTF=0.0641)。
- 中文会议 known speaker count 主结果中，DER 最优是 3dspeaker_eres2net (mean_DER_relaxed=0.2187)。
- 本轮中文会议结果与英文 VoxConverse 背景结论不同：SpeakerNet 仍然最快，但 3D-Speaker ERes2Net 在 DER 上明显更优。
- 建议：SpeakerNet 适合作为 fast/default backend；如果中文会议 diarization 精度优先，保留 3D-Speaker ERes2Net 作为 quality mode 或后续重点优化对象。

## 11. 局限

- 这是 host benchmark，不代表 RV1126B 板端速度。
- 主结果使用 single-channel，不代表多通道最优。
- 主结果使用 known speaker count，不等同于 unknown speaker-count 产品模式。
- 没有 ASR alignment，没有 Ye identification，没有实时链路。
- segmentation model 未针对中文会议重新训练，可能存在域差异。

## 12. 文件清单

- manifests: `data/manifests/`
- results: `results/`
- per-file CSV: `results/*/*/metrics/per_file_metrics.csv` and `results/combined/per_file_metrics.csv`
- aggregate JSON/CSV: `results/*/*/metrics/backend_summary.*` and `results/combined/backend_summary.*`
- logs: `logs/`
