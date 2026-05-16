# RV1126B → RK3588 迁移模板（基于当前 single-input 5s encoder 流程）

## 1. 迁移目标

将当前 RV1126B 版本的 Qwen3-ASR 端侧运行包迁移到 RK3588 平台。

当前流程保持不变：

```text
Qwen3-ASR 原始模型
→ 导出 5s fixed single-input encoder ONNX
→ 转换 RK3588 RKNN encoder
→ 导出 decoder_hf
→ 转换 RK3588 RKLLM decoder
→ 导出 embed_tokens.npy / mel_filters.npy
→ 整理 deploy 运行包
→ RK3588 板端验证离线转写和实时麦克风识别
```

本文档默认沿用当前已跑通的 **single-input merged encoder** 方案。

---

## 2. 当前 encoder 格式说明

当前 encoder 是通过 `FixedAudioTowerWrapper` 导出的 **单输入固定 5 秒 encoder**。

输入：

```text
input_features: [1, 128, 500]
```

输出：

```text
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

因此 RKNN 转换时只需要一个输入：

```python
inputs=["input_features"]
input_size_list=[[1, 128, 500]]
```

---

## 3. 建议目录结构

在本地转换环境中新建 RK3588 目录，避免覆盖 RV1126B 文件：

```text
asr/
├── Qwen3-ASR-0.6B/
├── build_rk3588/
├── qwen3asr_rk3588_run/
│   └── models/
└── convert/
    └── python/
        ├── export_encoder_5s_onnx.py
        ├── convert_encoder_5s_rk3588.py
        ├── export_decoder_hf_fp16.py
        ├── convert_decoder_w4a16_only_rk3588.py
        ├── export_embed_tokens.py
        └── export_mel_filters.py
```

创建目录：

```bash
cd ~/Projects/asr

mkdir -p build_rk3588
mkdir -p qwen3asr_rk3588_run/models/encoder/rk3588
mkdir -p qwen3asr_rk3588_run/models/decoder
mkdir -p qwen3asr_rk3588_run/models/vad
```

---

## 4. Step 1：导出 RK3588 encoder ONNX

复制当前脚本：

```bash
cd ~/Projects/asr/convert/python

cp export_encoder_5s_onnx.py export_encoder_5s_onnx_rk3588.py
```

修改输出路径：

```python
onnx_path = f"build_rk3588/qwen3_asr_encoder_merged.fp16.{SECONDS}s.rk3588.onnx"
```

保持以下内容不变：

```python
SECONDS = 5
FRAMES = SECONDS * 100
CHUNKS = FRAMES // 100
TOKENS = CHUNKS * 13
```

保持 forward 只有一个输入：

```python
def forward(self, input_features):
```

运行：

```bash
cd ~/Projects/asr

python convert/python/export_encoder_5s_onnx_rk3588.py
```

检查：

```bash
ls -lh build_rk3588/*.onnx
```

期望得到：

```text
build_rk3588/qwen3_asr_encoder_merged.fp16.5s.rk3588.onnx
```

---

## 5. Step 2：转换 RK3588 encoder RKNN

复制当前 RV1126B 转换脚本：

```bash
cd ~/Projects/asr/convert/python

cp convert_encoder_5s_rv1126b.py convert_encoder_5s_rk3588.py
```

修改为：

```python
from rknn.api import RKNN

onnx_model = "build_rk3588/qwen3_asr_encoder_merged.fp16.5s.rk3588.onnx"
out_model = "build_rk3588/qwen3_asr_encoder_merged.fp16.5s.rk3588.rknn"

rknn = RKNN(verbose=True)

print("[1/4] config")
ret = rknn.config(
    target_platform="rk3588",
    optimization_level=3,
)
if ret != 0:
    print("config failed", ret)
    exit(ret)

print("[2/4] load onnx")
ret = rknn.load_onnx(
    model=onnx_model,
    inputs=["input_features"],
    input_size_list=[[1, 128, 500]],
)
if ret != 0:
    print("load_onnx failed", ret)
    exit(ret)

print("[3/4] build")
ret = rknn.build(do_quantization=False)
if ret != 0:
    print("build failed", ret)
    exit(ret)

print("[4/4] export")
ret = rknn.export_rknn(out_model)
if ret != 0:
    print("export failed", ret)
    exit(ret)

rknn.release()
print("DONE:", out_model)
```

运行：

```bash
cd ~/Projects/asr

python convert/python/convert_encoder_5s_rk3588.py
```

检查：

```bash
ls -lh build_rk3588/*.rknn
```

期望得到：

```text
build_rk3588/qwen3_asr_encoder_merged.fp16.5s.rk3588.rknn
```

复制到 RK3588 运行包：

```bash
cp build_rk3588/qwen3_asr_encoder_merged.fp16.5s.rk3588.rknn \
  qwen3asr_rk3588_run/models/encoder/rk3588/
```

---

## 6. Step 3：导出 decoder_hf

复制脚本：

```bash
cd ~/Projects/asr/convert/python

cp export_decoder_hf_fp16.py export_decoder_hf_fp16_rk3588.py
```

修改路径：

```python
src = "./Qwen3-ASR-0.6B"
dst = "./qwen3asr_rk3588_run/models/decoder_hf"
```

保留 decoder config 修改逻辑：

```python
text_cfg["model_type"] = "qwen3"
text_cfg["architectures"] = ["Qwen3ForCausalLM"]
text_cfg["tie_word_embeddings"] = True
text_cfg["torch_dtype"] = "float16"
text_cfg["rope_scaling"] = {
    "rope_type": "default",
    "type": "default"
}
```

如脚本里有旧路径，例如：

```python
qz_tok_dir = "./qwen3asr_rk_code/models/decoder_hf"
```

建议改成原始模型目录优先：

```python
tok_dir = src
```

或直接从 `src` 复制 tokenizer 相关文件。

运行：

```bash
cd ~/Projects/asr

python convert/python/export_decoder_hf_fp16_rk3588.py
```

检查：

```bash
ls -lh qwen3asr_rk3588_run/models/decoder_hf
```

应至少包含：

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
generation_config.json
```

---

## 7. Step 4：转换 RK3588 decoder RKLLM

复制脚本：

```bash
cd ~/Projects/asr/convert/python

cp convert_decoder_w4a16_only_rv1126b.py convert_decoder_w4a16_only_rk3588.py
```

修改核心配置：

```python
SRC_MODEL = Path("./qwen3asr_rk3588_run/models/decoder_hf").resolve()
OUT_DIR = Path("./qwen3asr_rk3588_run/models/decoder").resolve()
DATASET = Path("./qwen3asr_rk3588_run/models/data_quant.json").resolve()

TARGET_PLATFORM = "rk3588"
NUM_NPU_CORE = 1
MAX_CONTEXT = 4096
OPT_LEVEL = 1

QUANT_DTYPE = "w4a16"
QUANT_ALGO = "grq"
```

临时目录也改为：

```python
tmpdir = Path(tempfile.mkdtemp(
    prefix="decoder_as_qwen3_rk3588_",
    dir="./build_rk3588",
)).resolve()
```

输出文件名保持自动生成：

```python
save_path = OUT_DIR / f"decoder_qwen3.{QUANT_DTYPE}.{TARGET_PLATFORM}.rkllm"
```

运行：

```bash
cd ~/Projects/asr

python convert/python/convert_decoder_w4a16_only_rk3588.py
```

检查：

```bash
ls -lh qwen3asr_rk3588_run/models/decoder
```

期望得到：

```text
decoder_qwen3.w4a16.rk3588.rkllm
```

> 说明：先用 `NUM_NPU_CORE = 1` 跑通功能。RK3588 功能验证通过后，可以再尝试 `NUM_NPU_CORE = 3` 做性能测试。

---

## 8. Step 5：导出 embed_tokens.npy

复制脚本：

```bash
cd ~/Projects/asr/convert/python

cp export_embed_tokens.py export_embed_tokens_rk3588.py
```

修改输出路径：

```python
out = "./qwen3asr_rk3588_run/models/embed_tokens.npy"
```

运行：

```bash
cd ~/Projects/asr

python convert/python/export_embed_tokens_rk3588.py
```

检查：

```bash
ls -lh qwen3asr_rk3588_run/models/embed_tokens.npy
```

---

## 9. Step 6：生成 mel_filters.npy

复制脚本：

```bash
cd ~/Projects/asr/convert/python

cp export_mel_filters.py export_mel_filters_rk3588.py
```

修改输出路径：

```python
out_path = Path("./qwen3asr_rk3588_run/models/mel_filters.npy")
```

保持参数不变：

```python
sr = 16000
n_fft = 400
n_mels = 128
fmax = 8000
```

运行：

```bash
cd ~/Projects/asr

python convert/python/export_mel_filters_rk3588.py
```

检查：

```bash
ls -lh qwen3asr_rk3588_run/models/mel_filters.npy
```

---

## 10. Step 7：整理 RK3588 deploy 包

最终 RK3588 deploy 目录建议为：

```text
qwen3asr_rk3588/
├── models/
│   ├── mel_filters.npy
│   ├── embed_tokens.npy
│   ├── tokenizer.json 或 tokenizer/
│   ├── encoder/
│   │   └── rk3588/
│   │       └── qwen3_asr_encoder_merged.fp16.5s.rk3588.rknn
│   ├── decoder/
│   │   └── decoder_qwen3.w4a16.rk3588.rkllm
│   └── vad/
│       └── silero_vad.onnx
├── python/
│   ├── transcribe.py
│   ├── mic_stream.py
│   └── qwen3asr/
└── lib/
    └── librkllmrt.so
```

复制文件：

```bash
mkdir -p qwen3asr_rk3588/models
mkdir -p qwen3asr_rk3588/models/encoder/rk3588
mkdir -p qwen3asr_rk3588/models/decoder
mkdir -p qwen3asr_rk3588/models/vad
mkdir -p qwen3asr_rk3588/python
mkdir -p qwen3asr_rk3588/lib

cp qwen3asr_rk3588_run/models/mel_filters.npy qwen3asr_rk3588/models/
cp qwen3asr_rk3588_run/models/embed_tokens.npy qwen3asr_rk3588/models/

cp build_rk3588/qwen3_asr_encoder_merged.fp16.5s.rk3588.rknn \
  qwen3asr_rk3588/models/encoder/rk3588/

cp qwen3asr_rk3588_run/models/decoder/decoder_qwen3.w4a16.rk3588.rkllm \
  qwen3asr_rk3588/models/decoder/

cp -r qwen3asr_rk3588_run/models/decoder_hf/tokenizer* qwen3asr_rk3588/models/ 2>/dev/null || true
cp -r qwen3asr_rk3588_run/models/decoder_hf/special_tokens_map.json qwen3asr_rk3588/models/ 2>/dev/null || true
cp -r qwen3asr_rk3588_run/models/decoder_hf/generation_config.json qwen3asr_rk3588/models/ 2>/dev/null || true
```

从当前 RV1126B deploy 包中复制 runtime：

```bash
cp -r qwen3asr_rv1126b/python/* qwen3asr_rk3588/python/
cp -r qwen3asr_rv1126b/lib/* qwen3asr_rk3588/lib/
cp qwen3asr_rv1126b/models/vad/silero_vad.onnx qwen3asr_rk3588/models/vad/
```

---

## 11. Step 8：检查 runtime 代码平台路径

在 deploy 包中检查是否写死了 `rv1126b`：

```bash
cd qwen3asr_rk3588

grep -R "rv1126b" python/*.py python/qwen3asr
```

需要确认以下逻辑支持：

```text
--platform rk3588
models/encoder/rk3588/
decoder_qwen3.w4a16.rk3588.rkllm
```

如果代码中只查找 RV1126B 文件名，需要改成根据 `platform` 拼路径，例如：

```python
encoder_path = f"{model_dir}/encoder/{platform}/qwen3_asr_encoder_merged.fp16.5s.{platform}.rknn"
decoder_path = f"{model_dir}/decoder/decoder_qwen3.{decoder_quant}.{platform}.rkllm"
```

---

## 12. Step 9：RK3588 板端离线测试

将 `qwen3asr_rk3588/` 传到 RK3588：

```bash
adb push qwen3asr_rk3588 /data/
```

进入板端：

```bash
cd /data/qwen3asr_rk3588/python
source /data/your_env/bin/activate
export LD_LIBRARY_PATH=/data/qwen3asr_rk3588/lib:$LD_LIBRARY_PATH
```

准备 16kHz mono wav：

```bash
ffmpeg -y -i /data/test.wav -ac 1 -ar 16000 /data/test_16k.wav
```

运行离线转写：

```bash
python transcribe.py \
  --model-dir /data/qwen3asr_rk3588/models \
  --platform rk3588 \
  --audio /data/test_16k.wav \
  --language Chinese \
  --chunk-size 5 \
  --memory-num 2 \
  --rollback-tokens 5 \
  --max-new-tokens 128 \
  --cpus 4 \
  --decoder-quant w4a16 \
  --repeat-penalty 1.15
```

---

## 13. Step 10：RK3588 实时麦克风测试

查看声卡：

```bash
arecord -l
arecord -L | head -80
```

根据实际设备修改 `--device`。

如果麦克风是 USB：

```bash
--device plughw:CARD=USB,DEV=0
```

如果是第 2 张卡第 0 个设备：

```bash
--device plughw:2,0
```

运行实时识别：

```bash
cd /data/qwen3asr_rk3588/python
source /data/your_env/bin/activate
export LD_LIBRARY_PATH=/data/qwen3asr_rk3588/lib:$LD_LIBRARY_PATH

python mic_stream.py \
  --model-dir /data/qwen3asr_rk3588/models \
  --platform rk3588 \
  --device plughw:2,0 \
  --language Chinese \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --no-vad \
  --max-seconds 60
```

如果 RK3588 麦克风可以直接输出 16kHz mono，可以后续简化音频链路；迁移初期建议先保留当前 `arecord + ffmpeg pipe` 方案。

---

## 14. 必查日志

运行时确认：

```text
RKNN Runtime Information
target platform: rk3588
```

确认 RKLLM：

```text
platform: RK3588
model_dtype: W4A16
```

确认输出不是只生成 1 到 3 个 token。若只生成极短文本，优先检查：

```text
1. decoder 文件名和 --decoder-quant 是否匹配
2. decoder 是否为 qwen3 config
3. encoder 输出 shape 是否为 [1, 65, 1024]
4. runtime 是否按 single-input encoder 调用
5. audio embedding + text embedding 拼接是否正常
```

---

## 15. 最小修改清单

```text
[ ] build_rv1126b → build_rk3588
[ ] qwen3asr_rv1126b_run → qwen3asr_rk3588_run
[ ] qwen3asr_rv1126b → qwen3asr_rk3588
[ ] target_platform="rv1126b" → "rk3588"
[ ] TARGET_PLATFORM="rv1126b" → "rk3588"
[ ] 文件名 *.rv1126b.rknn → *.rk3588.rknn
[ ] 文件名 *.rv1126b.rkllm → *.rk3588.rkllm
[ ] encoder 仍保持 single input: input_features [1,128,500]
[ ] decoder 先用 NUM_NPU_CORE=1 验证
[ ] RK3588 跑通后可测试 NUM_NPU_CORE=3
[ ] 运行命令 --platform rk3588
[ ] 麦克风 --device 根据 arecord -l 修改
```

---

## 16. 推荐验证顺序

```text
1. 导出 encoder ONNX
2. 转换 encoder RKNN
3. 导出 decoder_hf
4. 转换 decoder RKLLM
5. 生成 embed_tokens.npy / mel_filters.npy
6. 整理 deploy 目录
7. RK3588 离线 transcribe.py 测试
8. RK3588 实时 mic_stream.py 测试
9. 再测试 VAD、多语言和长音频
```
