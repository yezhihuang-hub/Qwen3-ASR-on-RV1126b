# RV1126B 实时麦克风多语言/方言准确性测试统计

## 1. 测试说明

本报告整理 RV1126B 平台上 Qwen3-ASR 实时麦克风识别链路的准确性测试结果。测试覆盖普通话、中英混说、粤语、四川话/川渝口音、吴语/上海话、西班牙语和法语，并分别比较无 VAD 与有 VAD 两种模式。

本报告只保留最终用于统计的测试组；如果同一测试项进行了多次测试，则以最后一次结果为准。

本轮测试使用固定预期文本进行短句准确性评估；中文、粤语、四川话、吴语和中英混说围绕同一段中文语义展开，西班牙语和法语使用该文本对应的外语表达进行对比。后续如果要进一步做更严格 benchmark，可以在此基础上扩展更多样本、固定录音源，并统一用人工标注文本计算 CER/WER。

## 2. 预期文本

### 2.1 中文基准文本

> 昨天下午，我坐公交去图书馆。路上忽然下雨，大家都急着躲雨。一个男孩却停下来，把迷路的小猫抱到屋檐下。雨停后，我觉得这一天很普通，也很温柔。

### 2.2 中英混说参考文本

> 昨天下午，我坐 bus 去图书馆。路上忽然下雨，everyone 都急着躲雨。一个 boy 停下来，把 lost cat 抱到屋檐下。雨停后，我觉得这一天 ordinary but warm。

### 2.3 西班牙语参考大意

> Ayer por la tarde tomé el autobús para ir a la biblioteca. De repente empezó a llover y todos se apresuraron a buscar refugio. Un niño se detuvo y llevó a un gatito perdido bajo el alero. Cuando dejó de llover, sentí que aquel día era muy normal, pero también muy tierno.

### 2.4 法语参考大意

> Hier après midi, j'ai pris le bus pour aller à la bibliothèque. Il pleuvait, un garçon a mis un chat perdu à l'abri. Après la pluie, cette journée m'a semblé simple et douce.

## 3. 测试配置

| 项目 | 设置 |
|---|---|
| 平台 | RV1126B |
| 输入方式 | 实时麦克风 |
| 音频链路 | arecord 采集 + ffmpeg pipe 转 16kHz mono PCM |
| Encoder | RKNN fp16, 5s fixed single-input encoder |
| Decoder | RKLLM W4A16 |
| chunk-size | 5s |
| memory-num | 2 |
| max-new-tokens | 128 |
| rollback-tokens | 2 |
| CPUs | 4 |
| VAD 参数 | threshold=0.35, min_silence=1.2s |

## 4. 统计方式

- 中文类测试使用 CER 作为主要参考。
- 粤语输出中的繁体字按语义转换为简体后评估。
- 中英混说将 `bus / everyone / boy / lost cat / ordinary but warm` 按参考文本进行对齐。
- 西班牙语、法语使用对应外语参考表达估算 WER。
- 标点、空格、大小写不作为主要错误。
- 由于测试来自实时麦克风输入，结果仍会受到说话速度、停顿、麦克风音量和停止时机影响。

## 5. 最终结果总表

| 测试项 | 模式 | Language | RTF | 指标 | 错误率 | 评价 |
|---|---|---|---:|---|---:|---|
| 普通话 | 无 VAD | Chinese | 0.512 | CER | 0.0% | 优秀 |
| 普通话 | 有 VAD | Chinese | 0.729 | CER | 0.0% | 优秀 |
| 中英混说 | 无 VAD | Chinese | 0.477 | CER | 约 16.7% | 需优化 |
| 中英混说 | 有 VAD | Chinese | 0.757 | CER | 约 1.7% | 较好 |
| 粤语 | 无 VAD | Cantonese | 0.516 | CER | 约 0.0%～1.7% | 较好 |
| 粤语 | 有 VAD | Cantonese | 0.766 | CER | 约 3.3% | 较好 |
| 四川话/川渝口音 | 无 VAD | Sichuan | 0.486 | CER | 0.0% | 优秀 |
| 四川话/川渝口音 | 有 VAD | Sichuan | 0.705 | CER | 0.0% | 优秀 |
| 吴语/上海话 | 无 VAD | Wu language | 0.487 | CER | 约 18.3% | 需优化 |
| 吴语/上海话 | 有 VAD | Wu language | 0.717 | CER | 约 18.3% | 需优化 |
| 西班牙语 | 无 VAD | Spanish | 0.554 | WER | 约 9.6% | 可用 |
| 西班牙语 | 有 VAD | Spanish | 0.815 | WER | 约 23.1% | 需优化 |
| 法语 | 无 VAD | French | 0.585 | WER | 0.0% | 优秀 |
| 法语 | 有 VAD | French | 0.828 | WER | 约 3.0% | 较好 |

### 5.1 VAD 与无 VAD 总体对比

- 无 VAD 平均 RTF：**0.517**。
- 有 VAD 平均 RTF：**0.760**。
- 无 VAD 速度整体更快，更适合连续朗读和准确性评估。
- 有 VAD 更适合真实交互式场景，但会引入额外 VAD 耗时和分段误差。
- 本轮中，中英混说有 VAD 的最后一次结果明显更完整；西班牙语无 VAD 更好；普通话、四川话、法语差异较小。

## 6. 最终保留测试组明细

### 6.1 普通话｜无 VAD｜Language=Chinese

**识别结果：**

```text
昨天下午，我坐公交去图书馆。路上忽然下雨，大家都急着躲雨。一个男孩却停下来，把迷路的小猫抱到屋檐下。雨停后，我觉得这一天很普通也很温柔。
```

**翻译 / 语义说明：**

```text
中文输出，与预期文本语义一致：昨天下午坐公交去图书馆，途中下雨，大家躲雨，一个男孩把迷路小猫抱到屋檐下，雨停后觉得这一天普通但温柔。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 18.6s | 835ms | 8705ms | 0ms | 0.512 | 0 |

**准确性说明：**

- 指标：CER
- 错误率：0.0%
- 评价：优秀
- 说明：与预期文本完全一致；仅标点略有差异，不计入错误。

### 6.2 普通话｜有 VAD｜Language=Chinese

**识别结果：**

```text
昨天下午，我坐公交去图书馆。路上忽然下雨，大家都急着躲雨。一个男孩却停下来，把迷路的小猫抱到屋檐下。雨停后，我觉得这一天很普通。也很温柔。
```

**翻译 / 语义说明：**

```text
中文输出，与预期文本语义一致：雨天路上，一个男孩帮助迷路小猫，最后表达这一天普通而温柔。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 14.5s | 3245ms | 7327ms | 2300ms | 0.729 | 1 |

**准确性说明：**

- 指标：CER
- 错误率：0.0%
- 评价：优秀
- 说明：与预期文本完全一致；VAD 额外增加检测耗时，RTF 高于无 VAD。

### 6.3 中英混说｜无 VAD｜Language=Chinese

**识别结果：**

```text
昨天下午，我坐 bus 去图书馆。路上忽然下雨， everyone 都急着躲雨。一个 boy 停了下来，把 lost cat 摇到屋檐下。雨停后，的这一天 ordinary but
```

**翻译 / 语义说明：**

```text
大意：昨天下午坐 bus 去图书馆，everyone 躲雨，一个 boy 停下来，把 lost cat 到屋檐下。末尾缺少 warm，并且“我觉得这一天”不完整。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 24.0s | 1053ms | 10391ms | 0ms | 0.477 | 0 |

**准确性说明：**

- 指标：CER
- 错误率：约 16.7%
- 评价：需优化
- 说明：语义主干保留，但句尾截断，且“抱”识别为“摇”。

### 6.4 中英混说｜有 VAD｜Language=Chinese

**识别结果：**

```text
昨天下午，我坐 bus 去图书馆。路上忽然下雨。 Everyone 都急着躲雨。一个 boy 停下来，把 lost cat 抱到屋檐下。雨停后，我觉得这一天 ordinary but warm。
```

**翻译 / 语义说明：**

```text
大意：昨天下午坐 bus 去图书馆，路上下雨，Everyone 躲雨，boy 把 lost cat 抱到屋檐下，雨停后觉得这一天 ordinary but warm。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 14.0s | 3052ms | 7542ms | 2440ms | 0.757 | 1 |

**准确性说明：**

- 指标：CER
- 错误率：约 1.7%
- 评价：较好
- 说明：中英混合词保留较好；相比无 VAD，准确性更好但实时性更差。

### 6.5 粤语｜无 VAD｜Language=Cantonese

**识别结果：**

```text
昨天下午，我坐公交去圖書館。路上忽然下雨，大家都急著躲雨。一個男孩卻停下來，把迷路的小貓抱到屋檐下。於停後，我覺得這一天很普通，也很溫柔。
```

**翻译 / 语义说明：**

```text
简体大意：昨天下午坐公交去图书馆，路上下雨，大家躲雨，一个男孩停下来把迷路小猫抱到屋檐下。雨停后，我觉得这一天很普通，也很温柔。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 22.7s | 1036ms | 10644ms | 0ms | 0.516 | 0 |

**准确性说明：**

- 指标：CER
- 错误率：约 0.0%～1.7%
- 评价：较好
- 说明：繁简转换后几乎完全一致；“雨停后”被写成“於停後”，按语义基本可接受。

### 6.6 粤语｜有 VAD｜Language=Cantonese

**识别结果：**

```text
昨天下午，我坐公交去圖書館。路上忽然下雨，大家都急著躲雨。一個男孩卻停下來，把迷路的小貓抱到屋檐下。雨停後，都覺得這一天很普通也很溫馨。
```

**翻译 / 语义说明：**

```text
简体大意：昨天下午坐公交去图书馆，路上下雨，大家躲雨，一个男孩把迷路小猫抱到屋檐下。雨停后，觉得这一天普通也温馨。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 16.5s | 3599ms | 9042ms | 2365ms | 0.766 | 1 |

**准确性说明：**

- 指标：CER
- 错误率：约 3.3%
- 评价：较好
- 说明：语义基本完整，但“我觉得”变成“都觉得”，“温柔”变成“温馨”。

### 6.7 四川话/川渝口音｜无 VAD｜Language=Sichuan

**识别结果：**

```text
昨天下午，我坐公交去图书馆。路上忽然下雨，大家都急着躲雨。一个男孩却停下来，把迷路的小猫抱到屋檐下。雨停后，我觉得这一天很普通也很温柔。
```

**翻译 / 语义说明：**

```text
中文输出，语义完整：坐公交去图书馆，路上下雨，一个男孩帮助迷路小猫，最后觉得普通而温柔。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 23.1s | 1054ms | 10191ms | 0ms | 0.486 | 0 |

**准确性说明：**

- 指标：CER
- 错误率：0.0%
- 评价：优秀
- 说明：与预期文本一致；由于测试文本本身偏普通话，该项主要验证 Sichuan prompt 下链路是否稳定。

### 6.8 四川话/川渝口音｜有 VAD｜Language=Sichuan

**识别结果：**

```text
昨天下午，我坐公交去图书馆。路上忽然下雨，大家都急着躲雨。一个男孩却停下来，把迷路的小猫抱到屋檐下。雨停后，我觉得这一天很普通，也很温柔。
```

**翻译 / 语义说明：**

```text
中文输出，语义完整：坐公交、下雨、男孩帮助小猫、最后觉得这一天普通且温柔。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 17.5s | 3608ms | 8724ms | 2578ms | 0.705 | 1 |

**准确性说明：**

- 指标：CER
- 错误率：0.0%
- 评价：优秀
- 说明：与预期文本一致；VAD 版本同样稳定，但耗时增加。

### 6.9 吴语/上海话｜无 VAD｜Language=Wu language

**识别结果：**

```text
昨天下午，我坐公交去图书馆。路上风沙大雨，大家都及时躲雨。一个男孩却停下来，把路小猫，跑到搞性下。雨停后，我觉得这天很普通，也很温柔。
```

**翻译 / 语义说明：**

```text
大意：模型识别出“昨天下午坐公交去图书馆、大家躲雨、男孩停下来、雨停后觉得普通温柔”的框架，但中间“路上风沙大雨”“把路小猫，跑到搞性下”等明显偏离原意。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 20.0s | 826ms | 8918ms | 0ms | 0.487 | 0 |

**准确性说明：**

- 指标：CER
- 错误率：约 18.3%
- 评价：需优化
- 说明：语义主线部分保留，但关键词错误较多，是本轮中文方言中最不稳定的一组。

### 6.10 吴语/上海话｜有 VAD｜Language=Wu language

**识别结果：**

```text
昨天下午，我坐公交去图书馆。路上风沙大雨，大家都及时躲雨。一个男孩却停下来，把迷路小猫抱到屋檐。雨停后，我觉得这一天很普通。
```

**翻译 / 语义说明：**

```text
大意：坐公交去图书馆，路上大雨，大家躲雨，一个男孩把迷路小猫抱到屋檐下，雨停后觉得这一天普通。缺少“也很温柔”。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 14.5s | 3257ms | 7143ms | 2477ms | 0.717 | 1 |

**准确性说明：**

- 指标：CER
- 错误率：约 18.3%
- 评价：需优化
- 说明：比无 VAD 版本的小猫相关内容更好，但仍有“风沙大雨/及时躲雨”错误，且句尾缺失。

### 6.11 西班牙语｜无 VAD｜Language=Spanish

**识别结果：**

```text
Ayer por la tarde tomé el autobús para ir a la biblioteca. De repente empezó a llover y todos se apressuraron a buscar refugio. Pn niño se detuvo y llevó a un gatito perdido bajo el alero, cuando dejó sentir que aquel día era muy normal, pero también muy tierno.
```

**翻译 / 语义说明：**

```text
中文大意：昨天下午我坐公交去图书馆。突然开始下雨，大家赶紧找地方避雨。一个孩子停下来，把一只迷路小猫带到屋檐下。雨停后，我觉得那一天很普通，但也很温柔。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 25.0s | 1047ms | 12801ms | 0ms | 0.554 | 0 |

**准确性说明：**

- 指标：WER
- 错误率：约 9.6%
- 评价：可用
- 说明：语义完整度较高，但有拼写错误 apressuraron，应为 apresuraron；“Un niño”误成“Pn niño”；“cuando dejó de llover, sentí...”中间漏词。

### 6.12 西班牙语｜有 VAD｜Language=Spanish

**识别结果：**

```text
Ayer por la tarde tomé el autobús para ir a la biblioteca. De repente empezó a llover y todos se apressuraron abuscar refugio, pero un niño se detuvo y a un gatito perdido bajo el alaero. Cuando dejó de llover, sentí que aquí mal pero también muy terno.
```

**翻译 / 语义说明：**

```text
中文大意：昨天下午坐公交去图书馆，突然下雨，大家赶紧找地方避雨，一个孩子停下来和一只迷路小猫在屋檐下。雨停后，觉得……但也很温柔。后半句存在明显错误。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 21.0s | 4736ms | 12376ms | 2891ms | 0.815 | 1 |

**准确性说明：**

- 指标：WER
- 错误率：约 23.1%
- 评价：需优化
- 说明：语义主线存在，但错误多于无 VAD：缺少 llevó，alero 误成 alaero，normal/tierno 部分识别较差。

### 6.13 法语｜无 VAD｜Language=French

**识别结果：**

```text
Hier après midi, j'ai pris le bus pour aller à la bibliothèque. Il pleuvait, un garçon a mis un chat perdu à l'abri. Après la pluie, cette journée m'a semblé simple et douce.
```

**翻译 / 语义说明：**

```text
中文大意：昨天下午，我坐公交去图书馆。下雨了，一个男孩把一只迷路的猫安置到避雨处。雨后，这一天让我觉得简单而温柔。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 15.0s | 647ms | 8130ms | 0ms | 0.585 | 0 |

**准确性说明：**

- 指标：WER
- 错误率：0.0%
- 评价：优秀
- 说明：法语输出完整、自然，准确性最好的一组之一。

### 6.14 法语｜有 VAD｜Language=French

**识别结果：**

```text
Hier après midi, j'ai pris le bus pour aller à la bibliothèque. Il pleuvait, un garçon a mis un chat perdu à l'abri. Après la pluissié, cette journée m'a semblé simple et douce.
```

**翻译 / 语义说明：**

```text
中文大意：昨天下午，我坐公交去图书馆。下雨了，一个男孩把一只迷路的猫放到避雨处。雨后，这一天让我觉得简单而温柔。
```

**性能统计：**

| Audio processed | Encoder | Decoder | VAD | RTF | Utterances |
|---:|---:|---:|---:|---:|---:|
| 13.0s | 2662ms | 8100ms | 1926ms | 0.828 | 1 |

**准确性说明：**

- 指标：WER
- 错误率：约 3.0%
- 评价：较好
- 说明：整体语义完整，但 Après la pluie 被识别为 Après la pluissié。

## 7. 分语言观察

### 7.1 普通话
普通话无 VAD 和有 VAD 均与预期文本完全一致，主要差异仅为标点。无 VAD RTF 更低，更适合连续朗读准确性测试。

### 7.2 中英混说
中英混说无 VAD 出现句尾截断和局部错词；有 VAD 的最后一次结果更完整，能够保留 bus、Everyone、boy、lost cat、ordinary but warm 等中英文混合内容。

### 7.3 粤语
粤语输出倾向繁体中文，整体语义完整。主要问题是个别词替换，例如“於停後”“溫馨”等。

### 7.4 四川话/川渝口音
本轮输入文本实际较接近普通话，Sichuan prompt 下仍能稳定输出完整句子。后续如果要验证真正方言能力，需要换成更典型的川渝口音材料。

### 7.5 吴语/上海话
吴语/上海话测试中出现明显错词，例如“风沙大雨”“搞性下”等，说明该场景目前稳定性弱于普通话、粤语和四川话。VAD 版本对“小猫抱到屋檐”部分稍有改善，但句尾缺失。

### 7.6 西班牙语
西班牙语无 VAD 版本整体语义较完整，但存在拼写和漏词；VAD 版本错误更多，尤其是动词缺失和句尾语义错误。本轮建议西班牙语准确性测试优先使用无 VAD。

### 7.7 法语
法语无 VAD 输出最稳定，几乎完整。VAD 最终版本也基本完整，只是 Après la pluie 被识别为 Après la pluissié。

## 8. 总结

本轮实时麦克风测试表明，当前 RV1126B 上的 Qwen3-ASR 实时识别链路可以稳定覆盖普通话、中英混说、粤语、四川话/川渝口音、吴语/上海话、西班牙语和法语。基于固定预期文本对比，普通话、四川话和法语表现最好；粤语整体较好；中英混说在有 VAD 的最终测试中较好；吴语/上海话和西班牙语仍存在明显错词和漏词，需要后续优化。

性能上，无 VAD 的平均 RTF 低于有 VAD，说明无 VAD 更适合连续朗读和准确性评估；有 VAD 更适合真实交互式场景，但会带来额外延迟和分段误差风险。后续可在当前固定文本测试基础上扩展更多语种、更多说话人和更长音频，并统一使用人工标注文本计算 CER/WER。