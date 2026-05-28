# RV1126B 语音识别音频实验总结



## 1. 总体范围

本总结覆盖 RV1126B 板端 Qwen3-ASR 实时语音识别链路中，与音频输入、ASR 推理、VAD、实时性、功耗和系统负载相关的实验结果。

当前已验证链路：

```text
RV1126B 麦克风
-> arecord 采集 48kHz stereo S16_LE raw PCM
-> ffmpeg 连续流式重采样为 16kHz mono PCM
-> Python 每 0.5s 读取 PCM
-> Mel 特征提取
-> RKNN encoder
-> RKLLM W4A16 decoder
-> 实时文本输出
```

当前也整理了一个 pre-roll 录音原型，用于保留触发前音频：

```text
RV1126B 麦克风
-> arecord + ffmpeg 连续采集 16kHz mono PCM
-> Python 环形缓冲保留最近 N 秒音频
-> 触发后保存 pre-roll + post-roll WAV
-> 可选调用 transcribe.py 做事件音频转写
```

当前运行配置：

| 项目 | 当前配置 |
|---|---|
| SoC / 平台 | RV1126B |
| ASR 模型 | Qwen3-ASR-0.6B |
| Encoder | RKNN fp16, fixed 5s single-input encoder |
| Encoder 输入 | `input_features: [1, 128, 500]` |
| Encoder 输出 | `audio_embeds: [1, 65, 1024]` |
| Decoder | RKLLM W4A16 |
| VAD | Silero VAD ONNX，可选 |
| 麦克风设备 | `plughw:CARD=rockchiprv1126b,DEV=0` |
| 实时采集源格式 | 48kHz, stereo, S16_LE |
| ASR 输入格式 | 16kHz, mono, PCM |

## 2. 音频链路说明

RV1126B 当前麦克风采集按 48kHz stereo S16_LE 处理，ASR 前端实际需要 16kHz mono PCM。因此实时链路中使用 `ffmpeg` 对 `arecord` 输出做连续流式重采样。

使用连续 pipe 的原因是实时 ASR 对音频块边界比较敏感。如果在 Python 内对每个小块单独重采样，块与块之间可能出现边界不连续，带来漏字、重复或错接。当前 `arecord + ffmpeg pipe` 的方式可以保证 Python 侧持续拿到稳定的 16kHz mono PCM。

麦克风增益配置：

```bash
amixer cset name='ACodec_LP ADC Switch' 1
amixer cset name='ACodec_LP Digital Gain Volume' 127
amixer cset name='ACodec_LP PGA Gain Volume' 31
```

录音与转换验证命令：

```bash
arecord -D plughw:CARD=rockchiprv1126b,DEV=0 \
  -d 10 -f S16_LE -r 48000 -c 2 \
  /data/mic_48k.wav

ffmpeg -y -i /data/mic_48k.wav \
  -ac 1 -ar 16000 \
  /data/mic_16k.wav
```

## 3. 实时 ASR 参数

本轮 ASR 实验主要使用以下参数：

| 参数 | 当前值 |
|---|---|
| `--model-dir` | `/data/qwen3asr_rv1126b/models` |
| `--platform` | `rv1126b` |
| `--device` | `plughw:CARD=rockchiprv1126b,DEV=0` |
| `--chunk-size` | 5 |
| `--memory-num` | 2 |
| `--decoder-quant` | `w4a16` |
| `--max-new-tokens` | 128 |
| `--rollback-tokens` | 2 |
| `--cpus` | 4 |
| `--max-seconds` | 60 |
| 无 VAD | `--no-vad` |
| 有 VAD | `--vad-threshold 0.35 --vad-min-silence 1.2` |

## 4. 功耗结果



| Workload | 记录电流/功耗 | 阶段结论 |
|---|---:|---|
| Board idle | 约 114.3 mA，约 1.38 W | 板端空闲基线 |
| Audio capture + ffmpeg only | 约 115 mA，约 1.40 W | 与 idle 很接近，音频采集/重采样本身功耗增量很小 |
| Chinese ASR no VAD | 约 2.3-2.4 W | 完整 ASR 推理后功耗明显升高 |
| Chinese ASR with VAD | 约 2.7 W | 带 VAD 场景功耗高于普通话 no VAD |

阶段结论：

- idle 和 audio-only 的功耗非常接近，说明麦克风采集和 `ffmpeg` 重采样不是主要功耗来源。
- 完整 ASR 推理会把系统功耗从约 1.4 W 拉升到约 2.3-2.7 W。
- 当前已保存数据中，普通话带 VAD 功耗高于普通话 no VAD，与监控日志中 VAD 版本 CPU/NPU 活动更高的趋势一致。
- 表中其他语言/方言的功耗字段尚未形成可直接引用的完整保存记录，因此本总结不展开跨语种功耗排序。

## 5. Audio-only 负载

实验目的：确认 `arecord + ffmpeg` 音频链路本身是否构成主要系统负载。

测试命令：

```bash
arecord -D plughw:CARD=rockchiprv1126b,DEV=0 \
  -f S16_LE -r 48000 -c 2 -t raw \
| ffmpeg -hide_banner -loglevel error \
  -f s16le -ar 48000 -ac 2 -i pipe:0 \
  -f s16le -ar 16000 -ac 1 pipe:1 \
> /dev/null
```

监控结果：

| Workload | CPU avg | CPU peak | Mem avg | Temp avg | NPU avg |
|---|---:|---:|---:|---:|---:|
| Board idle | 4.0% | 4.7% | 235.0 MB | 34.8 C | 0.0% |
| Python env idle | 4.0% | 4.5% | 236.3 MB | 35.4 C | 0.0% |
| Audio capture + ffmpeg | 5.6% | 8.7% | 245.4 MB | 36.3 C | 0.0% |

阶段结论：

- audio-only 相比 idle 平均 CPU 只增加约 1.6 个百分点。
- audio-only 平均内存约 245.4 MB，相比 idle 增加约 10 MB。
- audio-only NPU load 为 0，符合预期。
- 结合功耗和负载两组数据，音频采集与重采样可以认为是轻负载环节，主要压力来自后续 ASR 推理。

## 6. 实时 ASR 系统负载

本地监控日志覆盖了 idle、audio-only、普通话、中英混说和粤语 ASR workload。板端日志中的 `hwmon_power_info` 为 `NA`，因此系统负载统计和外部功耗测量分开看。

| Workload | CPU avg | CPU peak | Mem avg | Mem peak | Min available | Temp avg | Temp peak | NPU avg | NPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 普通话 ASR, no VAD | 34.4% | 76.2% | 1885.7 MB | 2434.3 MB | 1524.9 MB | 40.0 C | 43.8 C | 13.6% | 53% |
| 普通话 ASR, VAD | 42.2% | 76.2% | 2007.3 MB | 2462.2 MB | 1497.0 MB | 42.6 C | 45.9 C | 23.5% | 53% |
| 中英混说 ASR, VAD | 43.8% | 75.7% | 2124.0 MB | 2467.0 MB | 1492.3 MB | 44.0 C | 46.9 C | 26.2% | 54% |
| 中英混说 ASR, no VAD | 37.7% | 76.3% | 1970.3 MB | 2433.0 MB | 1526.2 MB | 43.8 C | 47.2 C | 13.6% | 54% |
| 粤语 ASR | 37.4% | 75.4% | 2111.1 MB | 2431.9 MB | 1527.3 MB | 44.2 C | 47.4 C | 15.5% | 53% |

阶段结论：

- ASR 推理是主要系统负载，明显高于 audio-only。
- 当前 4GB 内存环境下，ASR 峰值内存约 2.43 GB 到 2.47 GB，最低可用内存约 1.49 GB 到 1.53 GB。
- CPU 平均使用率约 34% 到 44%，峰值约 75% 到 76%。
- CPU 频率会从 idle 的 594 MHz 升到 1608 MHz。
- ASR 阶段出现非零 NPU load，峰值约 54%，说明 encoder/decoder 推理链路触发 NPU 侧活动。
- 短时测试温度峰值约 47.4 C。

## 7. 60 秒多语言/方言实时性

该实验使用约 60 秒音频，均为实时麦克风链路、无 VAD、5 秒 chunk。

| 测试类型 | Language | Audio | Encoder | Decoder | RTF | 观察 |
|---|---|---:|---:|---:|---:|---|
| 中英混说 | Chinese | 60.0s | 2494 ms | 27518 ms | 0.500 | 实时性最好之一 |
| 普通话 | Chinese | 60.0s | 2507 ms | 27899 ms | 0.507 | 输出较完整 |
| 四川话/川渝口音 | Sichuan | 60.0s | 2501 ms | 30104 ms | 0.543 | 能保留部分口音词 |
| 英文 | English | 60.0s | 2524 ms | 30578 ms | 0.552 | 新闻类英文较稳定 |
| 粤语 | Cantonese | 60.0s | 2493 ms | 34411 ms | 0.615 | 有粤语表达，存在错词/异常字符 |
| 法语 | French | 60.0s | 2511 ms | 37082 ms | 0.660 | 可输出连续法语，有拼写/词形错误 |
| 吴语/上海话 | Wu language | 60.0s | 2502 ms | 37764 ms | 0.671 | 普通话化和错词较明显 |
| 西班牙语 | Spanish | 60.0s | 2510 ms | 38165 ms | 0.678 | 连续输出可用，decoder 最慢 |

平均结果：

| 指标 | 结果 |
|---|---:|
| 平均 Encoder 耗时 | 约 2505 ms / 60s |
| 平均 Decoder 耗时 | 约 32940 ms / 60s |
| 平均 RTF | 约 0.591 |
| 最低 RTF | 0.500，中英混说 |
| 最高 RTF | 0.678，西班牙语 |

阶段结论：

- 当前 8 组 60 秒测试 RTF 均小于 1，说明在当前参数和样本下，RV1126B 可以做到快于实时处理。
- Encoder 耗时稳定，基本由 5 秒固定输入结构和音频长度决定。
- Decoder 是主要耗时来源，且受语言、输出 token 数、文本复杂度影响更明显。
- 这些 60 秒音频不是同一语义内容的多语言标准集，因此不作为严格跨语言准确率 benchmark。

## 8. 短句准确性与 VAD 对比

该实验使用固定预期文本，比较无 VAD 与有 VAD 的最终结果。中文类以 CER 估算，西班牙语/法语以 WER 估算。标点、空格、大小写不作为主要错误。

| 测试项 | 无 VAD RTF / 错误率 | 有 VAD RTF / 错误率 | 当前判断 |
|---|---:|---:|---|
| 普通话 | 0.512 / 0.0% CER | 0.729 / 0.0% CER | 稳定 |
| 中英混说 | 0.477 / 约 16.7% CER | 0.757 / 约 1.7% CER | VAD 版本更完整，但速度更慢 |
| 粤语 | 0.516 / 约 0.0% 到 1.7% CER | 0.766 / 约 3.3% CER | 语义基本完整，输出倾向繁体 |
| 四川话/川渝口音 | 0.486 / 0.0% CER | 0.705 / 0.0% CER | 本轮文本偏普通话，不能代表强方言测试 |
| 吴语/上海话 | 0.487 / 约 18.3% CER | 0.717 / 约 18.3% CER | 当前稳定性弱 |
| 西班牙语 | 0.554 / 约 9.6% WER | 0.815 / 约 23.1% WER | 无 VAD 更好，有 VAD 分段影响明显 |
| 法语 | 0.585 / 0.0% WER | 0.828 / 约 3.0% WER | 整体稳定 |

总体结果：

| 指标 | 无 VAD | 有 VAD |
|---|---:|---:|
| 平均 RTF | 0.517 | 0.760 |
| 更适合的输入形态 | 连续朗读、长段口播、准确性评估 | 交互式短句、有停顿输入 |
| 主要风险 | 长静音仍会按 chunk 处理 | 增加延迟，可能产生分段误差 |

阶段结论：

- 无 VAD 更快，更适合连续口播和当前准确性评估。
- 有 VAD 对真实交互更自然，可以跳过静音，但会增加 VAD 检测和分段处理成本。
- VAD 对准确性的影响不是单向的：中英混说最后一次有 VAD 更好，西班牙语无 VAD 更好。

## 9. Pre-roll 录音原型

`mic_preroll.py` 用来验证“触发前音频保留”这一类交互场景。它不直接改变 ASR 模型链路，而是在 ASR 前面增加一个轻量的音频事件缓存。

当前原型机制：

| 项目 | 当前实现 |
|---|---|
| 音频输入 | `arecord` 48kHz stereo raw PCM |
| 格式转换 | `ffmpeg` 转 16kHz mono S16_LE raw PCM |
| 缓冲方式 | Python 内存环形缓冲 |
| 默认 pre-roll | 60s |
| 默认 post-roll | 20s |
| 读取粒度 | 0.5s |
| 触发方式 | 终端 Enter 或创建 `/tmp/asr_trigger` |
| 输出 | `pre-roll + post-roll` 合并保存为 WAV |
| 后处理 | 可选调用 `transcribe.py` 做离线事件转写 |

按 16kHz、mono、16-bit PCM 计算，音频数据约为 32KB/s。默认 60s pre-roll 只需要约 1.92MB 原始 PCM 缓冲，80s 事件音频约 2.56MB，因此内存压力很小。

这个原型解决的是“触发瞬间之前说过的话不要丢”。例如用户按键、GPIO、唤醒词或其他事件到来时，系统可以把触发前 60s 和触发后 20s 合成一个事件音频，再交给 ASR 做转写。

同时它也暴露了后续低功耗设计的原因：如果 pre-roll 完全由 RV1126B 软件侧完成，板端必须长期保持 `arecord + ffmpeg` 音频链路运行。虽然 audio-only 负载和功耗接近 idle，约 1.40W，但这仍然不是极低功耗待机。

因此后续考虑外置一个低功耗音频检测模块，专门做音频大小/能量/幅度检测。这个模块常开监听音频能量变化，只在超过阈值或满足触发条件时唤醒 RV1126B 或产生触发信号。这样可以把“长期监听”从主 SoC 上移走，RV1126B 只在需要录音或识别时进入完整音频链路和 ASR 推理状态。

从当前实验数据看，这个方向的动机比较明确：

- audio-only 已经比完整 ASR 轻很多，但仍需要主 SoC 维持采集和重采样。
- 完整 ASR 功耗约 2.3-2.7W，不适合长期无条件常开。
- pre-roll 需要保存触发前音频，软件方案可以验证体验，但低功耗产品形态更适合用外置低功耗模块先做音频能量检测。

## 10. 模型与运行资产

当前主要模型/运行资产大小约：

| 文件 | 大小 |
|---|---:|
| `decoder_qwen3.w4a16.rv1126b.rkllm` | 702.1 MB |
| `qwen3_asr_encoder_merged.fp16.5s.rv1126b.rknn` | 378.7 MB |
| `embed_tokens.npy` | 311.2 MB |
| `librkllmrt.so` | 7.5 MB |
| `silero_vad.onnx` | 2.3 MB |
| `mel_filters.npy` | 0.1 MB |

运行时涉及：

| 类型 | 内容 |
|---|---|
| 系统工具 | `arecord`, `aplay`, `amixer`, `ffmpeg` |
| Python 依赖 | `numpy`, `soundfile`, `tokenizers`, `rknnlite`, `onnxruntime`，VAD 相关为 `sherpa_onnx` |
| Runtime | RKNN runtime, RKLLM runtime, `librkllmrt.so` |
| 模型目录 | `models/encoder/rv1126b`, `models/decoder`, `models/vad`, `tokenizer`, `embed_tokens.npy`, `mel_filters.npy` |

## 11. 当前实验边界

本轮总结没有覆盖：

1. 整机声学设计指标。
2. 唤醒词、AEC、NS、AGC、波束形成。
3. 远场、强噪声、回声、多人同时说话、强混响场景。
4. 四 MIC、两路独立 MIC、蓝牙 PCM 与本地 MIC 同时工作。
5. 长时间热稳定性和 24h 运行稳定性。
6. 所有语种/方言的完整功耗排序。
7. 大规模标准数据集 CER/WER。

## 12. 总结结论

- 当前 RV1126B 样机已跑通从板端麦克风采集到实时文本输出的完整 ASR 链路。
- 当前音频采集采用 ALSA `arecord` 48kHz stereo 输入，经 `ffmpeg` 连续重采样为 16kHz mono 后送入 ASR。
- pre-roll 原型通过软件环形缓冲保留触发前音频，默认保存触发前 60s 和触发后 20s，可用于事件录音后再转写。
- 功耗上，idle 约 1.38 W，audio-only 约 1.40 W，普通话 ASR no VAD 约 2.3-2.4 W，普通话 ASR with VAD 约 2.7 W。
- audio-only 的 CPU、内存、功耗都接近 idle，说明音频采集和重采样不是主要负载来源。
- 软件 pre-roll 需要主 SoC 长期开启音频采集链路；后续外置低功耗音频大小/能量检测模块的动机，是把常开监听从 RV1126B 主链路中剥离出来，只在检测到有效音频事件后再唤醒录音或 ASR。
- 完整 ASR 推理是主要系统负载，峰值内存约 2.47 GB，CPU 峰值约 76%，NPU debug load 峰值约 54%。
- 60 秒多语言/方言实时测试中，当前样本 RTF 均小于 1，平均约 0.591，具备快于实时处理能力。
- Encoder 耗时稳定，Decoder 是当前实时性主要瓶颈，语言和输出长度会影响 decoder 耗时。
- 无 VAD 平均 RTF 约 0.517，有 VAD 平均 RTF 约 0.760；无 VAD 更快，有 VAD 更接近交互输入形态但会增加延迟和分段误差风险。
- 当前测试中普通话、法语、四川话文本链路较稳，粤语和中英混说可用，吴语/上海话和西班牙语仍需要更多样本支撑。

## 13. 材料来源

- `yezhihuang-hub/Qwen3-ASR-on-RV1126b`: README、部署代码、模型文件指针、测试总结。
- `rv1126b_power_test_command_manual (1).md`: 本地功耗/负载测试命令。
- `rv1126b_workload_monitor_analysis_cn.md`: 本地负载监控分析。
- `rv1126b_power_workload_summary (1).xlsx`: 本地功耗测试记录。
- `mic_preroll.py`: pre-roll 音频环形缓冲和事件录音原型。
- `monitor_logs/`: 本地 CSV 与 process log。
- 荣品 RV1126B 文档：硬件介绍、接口使用、蓝牙 PCM 配置说明。
