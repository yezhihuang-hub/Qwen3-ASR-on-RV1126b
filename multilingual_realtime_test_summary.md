# Qwen3-ASR-0.6B RV1126B 实时麦克风多语言/方言测试统计

## 1. 测试目的

本轮测试用于验证 Qwen3-ASR-0.6B 在 RV1126B 平台上的实时麦克风识别能力，重点观察不同语言和中文方言场景下的实时性、稳定性和输出效果。

由于暂时没有找到“同一段语义内容对应不同语言/方言版本”的标准音频，因此本轮测试不作为严格的跨语言准确率 benchmark。当前测试采用不同语种/方言下语速相近、时长均约为 60 秒的音频，通过实时麦克风链路分别进行识别，用于评估系统在多语言输入下的端侧运行表现。

本轮测试均为：

```text
实时麦克风输入
无 VAD
60 秒音频
chunk-size = 5s
memory-num = 2
max-new-tokens = 128
rollback-tokens = 2
decoder-quant = w4a16
platform = rv1126b
```

---

## 2. 实时麦克风识别链路

当前实时麦克风链路如下：

```text
RV1126B 麦克风
→ arecord 采集 48kHz stereo raw PCM
→ ffmpeg 连续转换为 16kHz mono PCM
→ mic_stream.py 读取连续 PCM 音频流
→ Qwen3-ASR RKNN encoder
→ audio embedding
→ RKLLM W4A16 decoder
→ 输出识别文本
```

本轮测试未启用 VAD，因此所有音频均按固定 5 秒 chunk 进入 ASR 流式识别流程。

---

## 3. 测试结果汇总

| 序号 | 测试类型 | Language 参数 | 音频时长 | Encoder 耗时 | Decoder 耗时 | VAD 耗时 | RTF | 简要观察 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | 普通话 | Chinese | 60.0s | 2507ms | 27899ms | 0ms | 0.507 | 普通话新闻类语音，输出较完整，实时性最好之一 |
| 2 | 粤语 | Cantonese | 60.0s | 2493ms | 34411ms | 0ms | 0.615 | 能输出粤语风格文本，夹杂部分繁体/粤语词，decoder 耗时增加 |
| 3 | 吴语/上海话 | Wu language | 60.0s | 2502ms | 37764ms | 0ms | 0.671 | 能识别出部分上海话/吴语表达，但存在较多普通话化和错词 |
| 4 | 中英混说 | Chinese | 60.0s | 2494ms | 27518ms | 0ms | 0.500 | 中英文混合输出较自然，RTF 最低，实时性较好 |
| 5 | 四川话/川渝口音 | Sichuan | 60.0s | 2501ms | 30104ms | 0ms | 0.543 | 能保留部分川渝口音词汇，但语义细节需人工核对 |
| 6 | 西班牙语 | Spanish | 60.0s | 2510ms | 38165ms | 0ms | 0.678 | 可输出连续西班牙语文本，decoder 耗时最高 |
| 7 | 法语 | French | 60.0s | 2511ms | 37082ms | 0ms | 0.660 | 可输出连续法语文本，但存在部分拼写/词形错误 |
| 8 | 英文 | English | 60.0s | 2524ms | 30578ms | 0ms | 0.552 | 英文新闻类音频识别较稳定，实时性较好 |

---

## 4. 平均性能统计

| 指标 | 数值 |
|---|---:|
| 平均音频时长 | 60.0s |
| 平均 Encoder 耗时 | 约 2505ms |
| 平均 Decoder 耗时 | 约 32940ms |
| 平均 RTF | 约 0.591 |
| 最低 RTF | 0.500，中英混说 |
| 最高 RTF | 0.678，西班牙语 |

### 结论

在当前 8 组 60 秒实时麦克风测试中，所有测试的 RTF 均小于 1，说明系统在 RV1126B 上能够实现快于实时的语音识别。

Encoder 阶段在不同语言下耗时基本稳定，均约为 2.5 秒左右，说明 RKNN encoder 的运行耗时主要由输入音频长度和固定 5s encoder 结构决定，与语种关系不大。

Decoder 阶段耗时差异更明显，不同语言/方言的输出长度、token 分布和文本复杂度会影响 RKLLM decoder 的生成时间。因此当前系统的实时性主要受 decoder 生成阶段影响。

---

## 5. 各测试输出摘要

### 5.1 普通话 Chinese

测试内容为新闻类普通话音频。

输出片段：

```text
五月十五日上午，国家主席习近平在中南海同美国总统特朗普举行小范围会晤。春末夏初的中南海，绿茵遍，浓。草木葱茏。特朗普抵达时，习近平热情迎接。两国人手边走，边谈。不时驻足，观赏园中苍劲挺拔的古树和各色月季。习近平指出：“特朗普总统此访是一次历史性、标志性的访问。”我们共同确定了中美建设性战略稳定关系的新定位，就保持经贸关系稳定、拓展各领域务实合作、妥善解决彼此关切、达成重要共识，同意就国际和地区问题加强沟通和协调。
```

观察：

```text
普通话识别整体较稳定，文本连续性较好。
个别词存在错识别或断句不自然，例如“绿茵遍，浓”。
RTF = 0.507，实时性较好。
```

---

### 5.2 粤语 Cantonese

测试内容为粤语新闻类音频。

输出片段：

```text
欢迎收听六点半新闻。行政长官李家超用咗三个钟头，发表施政报告，比上一份长�咗半个钟。一齐睇下，有啲乜嘮重点？李家超第四份施政报告，主题系：深法改革、新是民优势同创未来。六色嘅封面，寓意活力同埋政策。延续发展经济同改善民生，系今年施政报告嘅两大主做。经济嘅新引擎就係北部都会区。李家超会親自領導一個委員會，拆牆、松榜，檢法、行政程序定立專屬法律。而喺民生方面，李家超話：公屋嘅重合輪候時間，目標係下年到。而為咗協助公屋職，業居屋嘅錄表比例會增加到。另外都會新推出居屋，長者扭換樓計劃。免補地價大單位，換細單位。

```

观察：

```text
模型能够输出粤语表达，例如“用咗”“一齐睇下”等。
输出中存在部分乱码或异常字符，例如“长�咗”。
部分词汇存在错识别，但整体能反映粤语内容。
RTF = 0.615。
```

---

### 5.3 吴语 / 上海话 Wu language

测试内容为上海话/吴语类音频。

输出片段：

```text
周日新闻》：《坊》，爱我？上海听。我讲：“观众朋友们大家好。”我是王老师。“讲：‘大家好’是刘言。”谢谢您。《王老师》，您好。哎，王老师。“哎，我要先来考考你。”。嗯，“三”？“三五二，就会得志。嗯，跟人民警察有关系。对的。同时呢？要告诉大家：从今天开始啊！每年一月上有一个新个纪实。啥纪事呢？叫中国人民警察节。携个节日的设立，对人民警察长期以来，对当中人、对人民服务得好的一个奖励。四百人，鼓励伊拉。继续个前进。格么可能？对幺幺零这个话题，侬看哪能？好，王老师。讲到幺幺零啊！我第一趟出题就是对幺幺零大家了。哦，携这个点我倒不晓得。出题啊？嗯后头哪？王老师。王老师，不是、不是迭个出题。迭个题啊是电视剧的题。其实搚辰光我老小辰光有趟。
```

观察：

```text
模型能够识别出部分上海话/吴语语境，并保留部分方言表达。
但输出明显存在普通话化、错词和语义不稳定。

RTF = 0.671。
```

---

### 5.4 中英混说 Chinese

测试内容为中英混合儿童节目类音频。

输出片段：

```text
我是佩奇。I'm Peppa Pig. 这是我的弟弟乔治。This is my little brother George. 这是我的妈妈。This is my big pig. 这是我的爸爸。And this is daddy pig. 小猪佩泡泡，bubbles。佩奇和乔治在喝橙汁。Peppa and George are drinking orange juice.你们这两个小家伙可真吵。George，你看这些小小的泡泡。Look at the tiny bubbles. 我可以弄出更大的泡泡！I can make bigger bubbles. 奧斯·佩奇在橙汁儿里吹出了好多氣泡。
```

观察：

```text
中英混合识别效果较好，能在中文和英文之间切换。
英文句子基本能保留原文形式。
存在少量错词，例如 “This is my big pig” 可能应为 “This is Mummy Pig”。
RTF = 0.500，是本轮测试中最低 RTF。
```

---

### 5.5 四川话 / 川渝口音 Sichuan

测试内容为四川话/川渝口音类音频。

输出片段：

```text
男到中年不如狗？没房，没车，没存款。好交鸡！一年努力的优秀教师，平生开始了。平上了就可以当官，职！机会来了，想什么？求上全。求平生一次优秀教师嘛！人我残缺的人生。嗯，我为圆满一点是。好嘛？好嘛！你先看一看吧。哎，第一条。要融入学生之中。嗯。啷个入赘？你们看《儒易》出来了没？出来了。我也想到，那个马六德孙公哥天天跟你耍朋友。哦，原来如此。哎呀，哎呀！哎呀。各位哥哥直向。我给你请安下一条是要对学生充满关爱，把学生当成自己的娃儿。
```

观察：

```text
模型能识别出部分川渝口音表达，例如“啷个”“好嘛”等。
部分内容语义不够稳定，存在错词和断句问题。
RTF = 0.543。
```

---

### 5.6 西班牙语 Spanish

测试内容为西班牙语新闻类音频。

输出片段：

```text
¡Espero que hayan disfrutado con su familia, con sus amigos. Les saluda como siempre Nicol Suárez con mucho gusto. Vamos a la información y esta noticia de última hora en el que envía un mensaje, redes anunciando una pausa migratoria permanente para países del tercer mundo y afirma que expulsarán a cualquier inmigrante que no sea un activo UU, pondrán fin a todos los beneficios féderales para quienes no son ciudadanos estaduíndos. La ciudadanía de aquellos que lo acaben. La paz y que deportará extranjeros, que sean un a riesgo a la seguridad nacional evidentemente, esto es algo que está generando muchas preguntas también preocupación. Definitivamente Nicol y sí. Así es, este es el más reciente endurecimiento de políticas migratorias en respuesta al tiroteo en Washington contra miembros del agu. Perpetuoado por un hombre afgano que trabajó con la sía y entró al país en 2. La salida de tropas en Kabul, este incidente ha dado pie al anuncio de medidas drácticas que hicieron en Estados Unidos, particularmente.
```

大意翻译：

```text
希望大家和家人、朋友度过了愉快的时光。我是 Nicol Suárez，很高兴再次向大家问好。接下来进入新闻内容：一则突发消息提到，美国方面宣布对第三世界国家实施永久性移民暂停，并表示将驱逐不被认为有价值的移民，同时终止非美国公民的联邦福利。这一政策引发了许多疑问和担忧。
```

观察：

```text
模型能够输出较长的连续西班牙语文本。
部分词汇存在拼写或识别错误，例如 “estaduíndos”等。
整体内容能反映新闻语义，但准确率需要进一步评估。
RTF = 0.678，是本轮测试中最高 RTF。
```

---

### 5.7 法语 French

测试内容为法语演讲/新闻类音频。

输出片段：

```text
Français, français mes chers compatriotes. Je m'adresse à vous ce soir en raison de la situation internationale et de ses conséquences pour notre pays et pour l'Europe, et cela après plusieurs clexions diplomatiques. Vous êtes en effet, je le sais légitimement inquié devant les évènements historiques en cours qui bouleversent l'ordre mondial. La guerre qui a entraîné près d'un million de morts et de blessés continue avec la même intensité. Notre allié, ont changé leur position sur cette guerre, soutient moins l'Ukraine et laisse planer le doute sur la suite. Dans le même temps, les mêmes États européens entendent imposer des tarifs douaniers aux produits venant d lEurope. Enfune continue d'etre sans cesse plus brutale, et la menace terroriste ne faitible au total. Notre prospérité et notre sécurité sont devenus putes. Il faut bien le dire. Nous rentrons dans une nouvelle ère. La guerre en Ukraine dure maintenant.
```

大意翻译：

```text
法国人民，亲爱的同胞们，今晚我因为国际局势及其对我们国家和欧洲的影响向你们讲话。当前历史性事件正在改变世界秩序，你们对此感到担忧是可以理解的。战争仍在持续，造成了大量伤亡；同时，盟友在相关战争上的立场发生变化，欧洲也面临关税、安全和恐怖主义等多重压力。我们正在进入一个新的时代。
```

观察：

```text
模型能够输出连续法语文本，并保持基本语义。
存在部分拼写或词形错误，例如 “clexions”“d lEurope”“Enfune”等。
RTF = 0.660。
```

---

### 5.8 英文 English

测试内容为 BBC 新闻类英文音频。

输出片段：

```text
This is the global news podcast from the BBC World Service. I'm Will Churcann in the early hours of Saturday, the sixteenth of May. These are our main stories. Bill says it's killed one of the Hamas architects behind the October 7th attacks. City services in Gaza city say the Israeli strikes killed at least seven people. The Bolivian government says it has reached a deal with protesting miners after violence on Thursday. Also in this podcast: the hunt for the next James Bond has begun. I don't think they have to be British; it could be anyone who can portray those sensibilities. Bond is our last sort of world-wide cultural hero, but who will accept the mission? There is officially a ceasefire between Israel and Hamas, but what practically that means is less clear. Israel says it has
```

观察：

```text
英文识别整体较稳定，新闻类内容连续性较好。
存在少量人名或专有名词错识别，例如 “Will Churcann”。
RTF = 0.552。
```

---

## 6. 结果分析

### 6.1 实时性

本轮所有测试的 RTF 均小于 1，说明当前 RV1126B 部署链路可以满足实时识别要求。

RTF 排序如下：

| 测试类型 | RTF |
|---|---:|
| 中英混说 | 0.500 |
| 普通话 | 0.507 |
| 四川话/川渝口音 | 0.543 |
| 英文 | 0.552 |
| 粤语 | 0.615 |
| 法语 | 0.660 |
| 吴语/上海话 | 0.671 |
| 西班牙语 | 0.678 |

可以看到，普通话、中英混说、英文和四川话测试的 RTF 相对较低；粤语、吴语、法语和西班牙语的 decoder 耗时更高。

### 6.2 Encoder 与 Decoder 耗时

Encoder 耗时在不同测试中基本稳定：

```text
约 2.49s - 2.52s / 60s 音频
```

说明 encoder 侧主要受音频长度和模型固定输入结构影响。

Decoder 耗时变化较大：

```text
约 27.5s - 38.2s / 60s 音频
```

说明 decoder 侧受语言类型、输出 token 数、文本复杂度和模型生成行为影响更明显。

当前链路的主要性能瓶颈仍然在 RKLLM decoder 生成阶段。

### 6.3 多语言与方言能力

从定性结果看：

```text
普通话：较稳定，可作为主任务 baseline。
英文：识别稳定，适合技术场景测试。
中英混说：表现较好，适合项目 demo。
粤语：能保留粤语表达，但存在错词和异常字符。
四川话：能识别部分方言表达，但需人工参考文本评估。
吴语/上海话：可识别部分语义，但普通话化和错词较明显。
西班牙语/法语：能输出连续文本，但存在拼写和词形错误。
```

---

## 7. 当前测试限制

本轮测试存在以下限制：

```text
1. 不同语言/方言使用的是不同音频内容，不是同一语义文本的多语言版本，因此不能严格比较准确率。

2. 各音频虽然时长均约为 60 秒，语速相近，但说话人、录音质量、内容复杂度和语言结构不同，会影响 decoder 生成耗时和识别效果。

3. 当前测试未接入 VAD，适合连续口播场景；对于真实交互场景，还需要单独测试带 VAD 的表现。

4. 当前统计主要基于 RTF、encoder/decoder 耗时和文本可读性，尚未计算 CER/WER 等标准准确率指标。

5. 方言类音频缺少人工标注文本，因此目前只能进行定性评估。
```

---

## 8. 后续计划

后续可以继续开展以下测试：

```text
1. 为每种语言/方言准备人工标注文本，计算 CER/WER。

2. 尽量寻找同一语义内容的多语言/多方言音频，减少测试内容差异带来的影响。

3. 对普通话、中英混说、粤语和上海话进行实时麦克风 demo 测试。

4. 分别比较无 VAD 与有 VAD 的识别结果，评估 VAD 对静音场景和连续口播场景的影响。

5. 统计每组测试的输出 token 数，进一步分析 decoder 耗时与 token 数之间的关系。

6. 对不同 chunk-size、memory-num、rollback-tokens、max-new-tokens 参数进行消融测试，分析端侧实时性和识别完整性的权衡。
```

---

## 9. 总结

本轮测试表明，当前 Qwen3-ASR-0.6B 的 RV1126B 端侧部署链路能够支持普通话、英文、中英混说、粤语、四川话、吴语/上海话、西班牙语和法语等多种语言/方言场景下的实时麦克风识别。所有 60 秒测试的 RTF 均小于 1，平均 RTF 约为 0.591，说明系统具备快于实时的处理能力。

从性能拆分看，RKNN encoder 耗时稳定，主要性能差异来自 RKLLM decoder 阶段。
