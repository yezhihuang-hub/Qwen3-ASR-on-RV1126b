 # RV1126B 实时麦克风语音识别 README

## 1. 功能概述

本文档说明 RV1126B 平台上实时麦克风语音识别系统的整体流程、实现方式、运行命令和交付文件结构。

当前系统已经实现：

```text
麦克风实时采集
→ 音频格式转换
→ 流式音频分块
→ Mel 特征提取
→ RKNN encoder 编码
→ RKLLM decoder 解码
→ 实时文本输出
```

系统支持：

```text
1. 离线音频文件转写
2. 实时麦克风识别
3. 可选 VAD 语音活动检测
4. 多语言/方言识别测试
5. 运行耗时与 RTF 统计
6. 录音结束后的 sherpa speaker finalizer 说话人贴标签
```

新增 speaker finalizer V1 后处理链路：

```text
realtime mic ASR
→ speaker-mode off，录音中只输出纯 ASR
→ raw_mic_16k.wav + asr_committed_segments.json
→ pyannote fp32 segmentation ONNX
→ NeMo SpeakerNet embedding ONNX
→ sherpa diarization / clustering
→ Ye enrollment identification
→ ASR overlap alignment
→ final speaker-labeled transcript
```

入口文档：

- `speaker_finalizer/README.md`
- `docs/speaker_finalizer/speaker_finalizer_v1_technical_report.md`
- `docs/speaker_finalizer/asr_first_post_record_finalizer_report.md`

当前主链路采用：

```text
Qwen3-ASR-0.6B
RKNN audio encoder
RKLLM W4A16 decoder
RV1126B 平台
5 秒固定输入 encoder
```

---

## 2. 整体工作流程

实时麦克风识别系统分为两部分：

```text
A. 音频采集与预处理 pipeline
B. ASR 推理 pipeline
```

### 2.1 音频采集与预处理 pipeline

RV1126B 板端麦克风采集到的原始音频通常为：

```text
48kHz stereo S16_LE PCM
```

语音识别前端需要：

```text
16kHz mono PCM
```

因此实时音频链路设计为：

```text
RV1126B 麦克风
→ arecord 采集 48kHz stereo raw PCM
→ ffmpeg 连续流式转换为 16kHz mono raw PCM
→ Python 每 0.5 秒读取一段 16kHz mono PCM
→ 送入流式 ASR session
```

这样做的原因是：  
实时识别要求音频流连续。如果在 Python 中对每个小音频块单独重采样，块与块之间可能出现边界不连续，导致漏字、重复或错接。使用 `ffmpeg` 做连续流式重采样后，Python 侧只读取稳定的 16kHz mono PCM，实时识别结果更接近离线文件转写效果。

### 2.2 ASR 推理 pipeline

音频进入 ASR 后的处理流程为：

```text
16kHz mono PCM
→ Mel 特征提取
→ RKNN audio encoder
→ audio embedding
→ text prompt embedding + audio embedding 拼接
→ RKLLM W4A16 decoder
→ token 输出
→ 文本解码
```

其中：

```text
encoder:
    将 Mel 特征转换为 audio embedding。

decoder:
    接收 text embedding 和 audio embedding，生成识别文本。

embed_tokens.npy:
    用于将 prompt token 转换为 text embedding。

mel_filters.npy:
    用于音频前处理阶段生成 Mel 特征。
```

---

## 3. Encoder 与 Decoder 连接方式

当前系统不是一个单模型直接输入音频输出文本，而是拆成：

```text
audio encoder + text decoder
```

### 3.1 Encoder

当前 encoder 是 5 秒固定输入版本：

```text
输入:
input_features: [1, 128, 500]

输出:
audio_embeds: [1, 65, 1024]
```

含义：

```text
5 秒音频
→ 500 帧 Mel feature
→ 5 个 chunk
→ 每个 chunk 13 个 audio token
→ 共 65 个 audio token
→ 每个 token 1024 维
```

### 3.2 Decoder

Decoder 使用 embedding 输入方式。输入不是单纯 token id，而是拼接后的 embedding 序列：

```text
prefix text embedding
+ audio embedding
+ suffix text embedding
→ RKLLM decoder
→ output tokens
→ text
```

这样可以将语音 encoder 的输出直接接入文本 decoder。

---

## 4. 目录结构

推荐板端目录：

```text
/data/qwen3asr_rv1126b/
```

推荐结构：

```text
qwen3asr_rv1126b/
├── README_realtime_asr.md
├── models/
│   ├── mel_filters.npy
│   ├── embed_tokens.npy
│   ├── tokenizer.json 或 tokenizer/
│   ├── encoder/
│   │   └── rv1126b/
│   │       └── qwen3_asr_encoder_merged.fp16.5s.rv1126b.rknn
│   ├── decoder/
│   │   └── decoder_qwen3.w4a16.rv1126b.rkllm
│   └── vad/
│       └── silero_vad.onnx
├── python/
│   ├── mic_stream.py
│   ├── transcribe.py
│   └── qwen3asr/
│       ├── __init__.py
│       ├── engine.py
│       ├── stream.py
│       ├── encoder.py
│       ├── decoder.py
│       ├── mel.py
│       └── vad.py
└── lib/
    └── RKNN/RKLLM 运行所需动态库
```

---

## 5. 需要包含的文件

### 5.1 模型文件

```text
models/mel_filters.npy
models/embed_tokens.npy
models/tokenizer.json 或 models/tokenizer/

models/encoder/rv1126b/qwen3_asr_encoder_merged.fp16.5s.rv1126b.rknn
models/decoder/decoder_qwen3.w4a16.rv1126b.rkllm
models/vad/silero_vad.onnx
```

### 5.2 Python 运行文件

```text
python/mic_stream.py
python/transcribe.py
python/qwen3asr/
```

### 5.3 运行库

```text
lib/librkllmrt.so
其他 RKNN/RKLLM runtime 需要的动态库
```

### 5.4 不需要包含

```text
__pycache__/
*.bak
*.old
大量测试 wav
完整 Python 虚拟环境
无关历史 demo
临时日志大文件
```

---

## 6. 环境准备

进入运行目录：

```bash
cd /data/qwen3asr_rv1126b/python
source /data/whisper_env/bin/activate
export LD_LIBRARY_PATH=/data/qwen3asr_rv1126b/lib:$LD_LIBRARY_PATH
```

检查 Python 依赖：

```bash
python - <<'PY'
import numpy
print("numpy ok")

import soundfile
print("soundfile ok")

from tokenizers import Tokenizer
print("tokenizers ok")

from rknnlite.api import RKNNLite
print("rknnlite ok")

import onnxruntime
print("onnxruntime ok")
PY
```

检查系统工具：

```bash
which arecord
which ffmpeg
which amixer
```

---

## 7. 麦克风检查

查看声卡：

```bash
arecord -l
arecord -L | head -80
```

当前 RV1126B 设备示例：

```text
plughw:CARD=rockchiprv1126b,DEV=0
```

开启麦克风增益：

```bash
amixer cset name='ACodec_LP ADC Switch' 1
amixer cset name='ACodec_LP Digital Gain Volume' 127
amixer cset name='ACodec_LP PGA Gain Volume' 31
```

录音测试：

```bash
arecord -D plughw:CARD=rockchiprv1126b,DEV=0 \
  -d 10 -f S16_LE -r 48000 -c 2 \
  /data/mic_48k.wav
```

转换成 16kHz mono：

```bash
ffmpeg -y -i /data/mic_48k.wav \
  -ac 1 -ar 16000 \
  /data/mic_16k.wav
```

---

## 8. 离线音频文件转写

离线转写用于验证模型主链路是否正常。

### 8.1 录音

```bash
arecord -D plughw:CARD=rockchiprv1126b,DEV=0 \
  -d 30 -f S16_LE -r 48000 -c 2 \
  /data/1_48k.wav
```

### 8.2 转换格式

```bash
ffmpeg -y -i /data/1_48k.wav \
  -ac 1 -ar 16000 \
  /data/1.wav
```

### 8.3 运行转写

```bash
cd /data/qwen3asr_rv1126b/python
source /data/whisper_env/bin/activate
export LD_LIBRARY_PATH=/data/qwen3asr_rv1126b/lib:$LD_LIBRARY_PATH

python transcribe.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --audio /data/1.wav \
  --language Chinese \
  --chunk-size 5 \
  --memory-num 2 \
  --rollback-tokens 5 \
  --max-new-tokens 128 \
  --cpus 4 \
  --repeat-penalty 1.15
```

---

## 9. 实时麦克风识别

### 9.1 无 VAD 版本

适合连续朗读、新闻、长段口播。

```bash
cd /data/qwen3asr_rv1126b/python
source /data/whisper_env/bin/activate
export LD_LIBRARY_PATH=/data/qwen3asr_rv1126b/lib:$LD_LIBRARY_PATH

python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language Chinese \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --no-vad \
  --max-seconds 300
```

### 9.2 带 VAD 版本

适合说一句停一会儿的交互式语音场景。

```bash
cd /data/qwen3asr_rv1126b/python
source /data/whisper_env/bin/activate
export LD_LIBRARY_PATH=/data/qwen3asr_rv1126b/lib:$LD_LIBRARY_PATH

python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language Chinese \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --vad-threshold 0.5 \
  --vad-min-silence 0.8 \
  --max-seconds 300
```

如果 VAD 容易切掉轻声或句尾，可使用：

```bash
--vad-threshold 0.35 \
--vad-min-silence 1.2
```

---

## 10. 参数说明

```text
--model-dir
    模型目录。

--platform
    当前平台，RV1126B 使用 rv1126b。

--device
    ALSA 麦克风设备。

--language
    识别语言，例如 Chinese、English、Cantonese、Wu language。

--chunk-size
    每个识别 chunk 的长度。当前 encoder 是 5 秒固定输入，所以设为 5。

--memory-num
    保留最近几个 chunk 的上下文，用于提升跨段连续性。

--decoder-quant
    decoder 量化类型，当前使用 w4a16。

--max-new-tokens
    每个 chunk 最多生成 token 数。连续口播建议 128。

--rollback-tokens
    chunk 边界回退 token 数，用于减少边界重复或断裂。

--cpus
    允许 RKLLM runtime 使用的 CPU 核数。encoder 仍由 RKNN/NPU 执行。

--no-vad
    关闭 VAD，按固定 chunk 识别。

--vad-threshold
    VAD 判断人声的阈值。

--vad-min-silence
    VAD 判断一句话结束所需的静音时长。

--max-seconds
    实时识别最长运行时间。
```

---

## 11. mic_stream.py 实现逻辑

核心逻辑如下：

```text
1. 初始化 Qwen3-ASR engine
2. 初始化 VAD，若使用 --no-vad 则跳过
3. 启动 arecord 采集 48kHz stereo raw PCM
4. 启动 ffmpeg，将 48kHz stereo 转为 16kHz mono
5. Python 每 0.5 秒读取一段 16kHz mono PCM
6. 将 PCM 转为 float32 [-1, 1]
7. 调用 stream.feed_audio(pcm)
8. 若达到 chunk-size 或 VAD 判断一句话结束，则执行 encoder + decoder
9. 实时打印局部识别结果
10. Ctrl+C 或 max-seconds 到达后调用 stream.finish()
11. 输出最终结果和耗时统计
```

音频 pipeline 关键结构：

```python
arecord_cmd = [
    "arecord",
    "-D", args.device,
    "-f", "S16_LE",
    "-r", "48000",
    "-c", "2",
    "-t", "raw",
    "--buffer-size=16384",
]

ffmpeg_cmd = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel", "error",
    "-f", "s16le",
    "-ar", "48000",
    "-ac", "2",
    "-i", "pipe:0",
    "-f", "s16le",
    "-ar", "16000",
    "-ac", "1",
    "pipe:1",
]
```

读取音频并送入 stream：

```python
read_samples = int(16000 * 0.5)
read_bytes = read_samples * 2

raw = proc.stdout.read(read_bytes)
pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

result = stream.feed_audio(pcm)
```

---

## 12. 性能统计

当前实时麦克风测试中，典型 60 秒音频结果为：

```text
Audio processed: 60.0s
Encoder: 约 2500ms
Decoder: 约 27500ms - 38100ms
VAD: 0ms 或按实际启用情况统计
RTF: 约 0.50 - 0.68
```

多语言/方言 60 秒测试平均：

```text
平均 RTF ≈ 0.591
```

说明系统在当前测试中可快于实时运行。

观察结论：

```text
1. Encoder 耗时较稳定。
2. Decoder 是主要耗时来源。
3. 输出 token 数、语言类型和上下文长度会影响 decoder 耗时。
4. 无 VAD 更适合连续口播。
5. 带 VAD 更适合真实交互式语音输入。
```

---

## 13. 常见问题

### 13.1 为什么要使用 ffmpeg？

因为麦克风原始采集格式和模型输入格式不同。  
`ffmpeg` 用于把实时采集到的 48kHz stereo PCM 连续转换为 16kHz mono PCM，避免 Python 分块重采样导致的边界不连续。

### 13.2 为什么 `--cpus 4` 不代表没用 NPU？

当前系统是混合运行：

```text
arecord / ffmpeg：CPU
Mel 特征提取：CPU
RKNN encoder：NPU
RKLLM decoder：NPU + CPU runtime 协同
tokenizer / embedding 拼接 / 文本处理：CPU
```

`--cpus 4` 表示允许 RKLLM runtime 使用 4 个 CPU 核辅助运行和调度。

### 13.3 为什么无 VAD 有时比有 VAD 更完整？

连续朗读时，VAD 可能把轻声、句尾或短暂停顿误切开，影响上下文连续性。  
无 VAD 按固定 5 秒 chunk 识别，更适合连续口播。

### 13.4 为什么离线转写更稳定？

离线转写可以基于完整音频进行处理，实时识别需要边录边输出，chunk 边界处更容易出现重复、断句或漏字。因此实时识别适合实时预览，最终文本可以在录音结束后再用离线转写生成。

---

## 14. 推荐测试顺序

```text
1. 检查模型文件和运行库是否完整
2. 检查 Python 依赖
3. 检查 arecord / ffmpeg
4. 录制 10 秒 wav
5. 转成 16kHz mono
6. 跑 transcribe.py 离线转写
7. 跑 mic_stream.py --no-vad 实时识别
8. 跑 mic_stream.py 带 VAD 实时识别
9. 测试 60 秒连续口播
10. 测试不同语言/方言
```

---



## 15. 总结

当前实时麦克风语音识别系统已经完成从板端麦克风采集到文本输出的完整链路。系统通过 `arecord + ffmpeg pipe` 保证实时音频输入连续性，通过 RKNN encoder 提取 audio embedding，再通过 RKLLM W4A16 decoder 生成识别文本。

在当前 5 秒固定输入 encoder 设置下，系统可以支持离线转写和实时麦克风识别。60 秒多语言/方言实时测试中，整体 RTF 小于 1，说明系统具备快于实时的运行能力。后续可继续围绕 VAD 参数、chunk 策略、上下文长度、输出 token 数和不同语种/方言进行性能与准确性分析。
