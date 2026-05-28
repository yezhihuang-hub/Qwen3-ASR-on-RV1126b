

进入板端后运行：

```bash
cd /data/qwen3asr_rv1126b/python
source /data/whisper_env/bin/activate
export LD_LIBRARY_PATH=/data/qwen3asr_rv1126b/lib:$LD_LIBRARY_PATH
```

检查当前目录：

```bash
pwd
ls
```

检查模型目录：

```bash
find /data/qwen3asr_rv1126b/models -maxdepth 3 -type f | sort
```

检查运行库：

```bash
ls -lh /data/qwen3asr_rv1126b/lib
```

---

## 4. 麦克风检查命令

查看声卡：

```bash
arecord -l
arecord -L | head -80
```

当前默认设备：

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

转成 16kHz mono：

```bash
ffmpeg -y -i /data/mic_48k.wav \
  -ac 1 -ar 16000 \
  /data/mic_16k.wav
```

---


```text
P0 Board idle
P1 Python env idle
P2 Audio capture + ffmpeg only
P3 Chinese ASR no VAD
P4 Chinese ASR with VAD
P5 Mixed Chinese-English no VAD
P6 Mixed Chinese-English with VAD
P7 Cantonese no VAD
P8 Cantonese with VAD
P9 Sichuan no VAD
P10 Sichuan with VAD
P11 Wu language no VAD
P12 Wu language with VAD
P13 Spanish no VAD
P14 Spanish with VAD
P15 French no VAD
P16 French with VAD
P17 RAG embedding/search optional
```

每组需要记录：

```text
Voltage
Idle Current
Average Current
Peak Current
RTF
Encoder time
Decoder time
VAD time
Audio processed
Accuracy / output note
```

---

## 6. P0：Board idle

什么都不跑，系统空闲状态下观察电流 30–60 秒。

可以开一个 top 观察 CPU：

```bash
top
```

记录到 Excel：

```text
Workload: Board idle
Command Ref: CMD-00
```

---

## 7. P1：Python env idle

进入环境但不跑任务：

```bash
cd /data/qwen3asr_rv1126b/python
source /data/whisper_env/bin/activate
export LD_LIBRARY_PATH=/data/qwen3asr_rv1126b/lib:$LD_LIBRARY_PATH
```

保持 30–60 秒，记录电流。

记录到 Excel：

```text
Workload: Python env idle
Command Ref: CMD-01
```

---

## 8. P2：Audio capture + ffmpeg only

只测麦克风采集和 ffmpeg 实时转码，不跑 ASR 模型：

```bash
arecord -D plughw:CARD=rockchiprv1126b,DEV=0 \
  -f S16_LE -r 48000 -c 2 -t raw \
| ffmpeg -hide_banner -loglevel error \
  -f s16le -ar 48000 -ac 2 -i pipe:0 \
  -f s16le -ar 16000 -ac 1 pipe:1 \
> /dev/null
```

测 30–60 秒后按：

```text
Ctrl+C
```

记录到 Excel：

```text
Workload: Audio capture + ffmpeg only
Command Ref: CMD-02
```

---

## 9. ASR 实时识别通用参数

下面所有 ASR 命令都使用：

```text
--model-dir /data/qwen3asr_rv1126b/models
--platform rv1126b
--device plughw:CARD=rockchiprv1126b,DEV=0
--chunk-size 5
--memory-num 2
--decoder-quant w4a16
--max-new-tokens 128
--rollback-tokens 2
--cpus 4
--max-seconds 60
```

无 VAD 使用：

```text
--no-vad
```

带 VAD 使用：

```text
--vad-threshold 0.35
--vad-min-silence 1.2
```

---

## 10. P3：普通话，无 VAD

```bash
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
  --max-seconds 60
```

测试文本：

```text
昨天下午，我坐公交去图书馆。路上忽然下雨，大家都急着躲雨。一个男孩却停下来，把迷路的小猫抱到屋檐下。雨停后，我觉得这一天很普通，也很温柔。
```

---

## 11. P4：普通话，带 VAD

```bash
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
  --vad-threshold 0.35 \
  --vad-min-silence 1.2 \
  --max-seconds 60
```

---

## 12. P5：中英混说，无 VAD

```bash
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
  --max-seconds 60
```

测试文本：

```text
昨天下午，我坐 bus 去图书馆。路上忽然下雨，everyone 都急着躲雨。一个 boy 停下来，把 lost cat 抱到屋檐下。雨停后，我觉得这一天 ordinary but warm。
```

---

## 13. P6：中英混说，带 VAD

```bash
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
  --vad-threshold 0.35 \
  --vad-min-silence 1.2 \
  --max-seconds 60
```

---

## 14. P7：粤语，无 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language Cantonese \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --no-vad \
  --max-seconds 60
```

---

## 15. P8：粤语，带 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language Cantonese \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --vad-threshold 0.35 \
  --vad-min-silence 1.2 \
  --max-seconds 60
```

---

## 16. P9：四川话/川渝口音，无 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language Sichuan \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --no-vad \
  --max-seconds 60
```

---

## 17. P10：四川话/川渝口音，带 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language Sichuan \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --vad-threshold 0.35 \
  --vad-min-silence 1.2 \
  --max-seconds 60
```

---

## 18. P11：吴语/上海话，无 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language "Wu language" \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --no-vad \
  --max-seconds 60
```

---

## 19. P12：吴语/上海话，带 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language "Wu language" \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --vad-threshold 0.35 \
  --vad-min-silence 1.2 \
  --max-seconds 60
```

---

## 20. P13：西班牙语，无 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language Spanish \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --no-vad \
  --max-seconds 60
```

测试文本：

```text
Ayer por la tarde tomé el autobús para ir a la biblioteca. De repente empezó a llover y todos se apresuraron a buscar refugio. Un niño se detuvo y llevó a un gatito perdido bajo el alero. Cuando dejó de llover, sentí que aquel día era muy normal, pero también muy tierno.
```

---

## 21. P14：西班牙语，带 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language Spanish \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --vad-threshold 0.35 \
  --vad-min-silence 1.2 \
  --max-seconds 60
```

---

## 22. P15：法语，无 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language French \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --no-vad \
  --max-seconds 60
```

测试文本：

```text
Hier après midi, j'ai pris le bus pour aller à la bibliothèque. Il pleuvait, un garçon a mis un chat perdu à l'abri. Après la pluie, cette journée m'a semblé simple et douce.
```

---

## 23. P16：法语，带 VAD

```bash
python mic_stream.py \
  --model-dir /data/qwen3asr_rv1126b/models \
  --platform rv1126b \
  --device plughw:CARD=rockchiprv1126b,DEV=0 \
  --language French \
  --chunk-size 5 \
  --memory-num 2 \
  --decoder-quant w4a16 \
  --max-new-tokens 128 \
  --rollback-tokens 2 \
  --cpus 4 \
  --vad-threshold 0.35 \
  --vad-min-silence 1.2 \
  --max-seconds 60
```

---





chmod +x /data/monitor_rv1126b.sh
bash /data/monitor_rv1126b.sh -i 2 -t chinese_no_vad

---                                                      |
| -------- | ------------------------------------------------------------ |
| 平均电流     | 电流表/功率计读数                                                    |
| 电压       | 电源/功率计/万用表读数                                                 |
| 平均功耗     | V × I                                                        |
| CPU 使用率  | `top`                                                        |
| 内存使用     | `free -h`                                                    |
| CPU 频率   | `/sys/devices/system/cpu/cpufreq/.../scaling_cur_freq`       |
| NPU load | `/sys/kernel/debug/rknpu/load` 或 `/sys/class/devfreq/*/load` |
| NPU 频率   | `/sys/class/devfreq/*/cur_freq`                              |
| 温度       | `/sys/class/thermal/thermal_zone*/temp`                      |
| RTF      | mic_stream.py 最后输出                                           |


