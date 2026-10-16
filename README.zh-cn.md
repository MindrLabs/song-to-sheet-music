[English](README.md) · [한국어](README.ko.md) · [中文](README.zh-cn.md)

# song-to-sheet-music

这个服务把一首歌变成乐谱：为它听到的每件乐器各写一份分谱——鼓、贝斯、吉他（音符谱和节奏谱）、钢琴，以及带歌词的演唱旋律——所有分谱共用同一套小节网格、调性和和弦进行，输出 MusicXML、MIDI 和 PDF。给它一个 YouTube 链接或磁盘上的录音即可。

我们用 model-compose 串起七个开源模型——一个六轨分离器、一个鼓件分离器、一个节拍跟踪器、三个转录器和一个和弦识别器——并把整条流水线跑在 NVIDIA DGX Spark、RTX 4090 和 8 GB 的 MacBook Air M1 上。我们问的是：它能否在每台机器上运行，一首歌要多久，分谱与出版乐谱有多接近。为此，我们把 34 首流行与摇滚歌曲的分谱与其商业出版的分谱逐小节比对打分，没有人工判定；那些出版乐谱是凭耳朵记下的，自身也有错误，所以每个数字都应读作一致率，而不是准确率。一位鼓手和一位吉他手读了分谱的片段，告诉我们演奏者不会接受什么；他们的意见塑造了第 3 节的记谱规则。

![在网页 UI 中填入歌曲文件和标题，点 Run 后每份分谱随其第一页返回](docs/images/shared/demo.gif)

演示：在 RTX 4090 上（歌词也用 GPU）通过网页 UI（gradio）处理一段磁盘上的录音，每份分谱的第一页随之返回。这首歌用的 45 秒已剪掉，其余为实际速度。

## 目录

- [模型](#模型)
- [快速开始](#快速开始)
- [摘要](#摘要)
- [1 实验环境](#1-实验环境)
- [2 能在我的设备上运行吗](#2-能在我的设备上运行吗)
- [3 各分谱结果](#3-各分谱结果)
  - [3.1 节拍、小节、调性与和弦](#31-节拍小节调性与和弦)
  - [3.2 鼓](#32-鼓)
  - [3.3 贝斯](#33-贝斯)
  - [3.4 吉他](#34-吉他)
  - [3.5 钢琴](#35-钢琴)
  - [3.6 人声与歌词](#36-人声与歌词)
- [4 建议](#4-建议)
- [5 局限与未测量的内容](#5-局限与未测量的内容)
- [参考](#参考)
- [变更](#变更)
- [许可证](#许可证)

## 模型

| 阶段 | 模型 | 权重 | 运行于 |
|---|---|---|---|
| 六轨（贝斯、鼓、其他、人声、吉他、钢琴） | BS-Roformer-SW，jarredou 训练的 band-split RoFormer | 700 MB | GPU 或 CPU |
| 鼓件（底鼓、军鼓、通鼓、踩镝、叮叮镝、吊镝） | MDX23C DrumSep，aufr33 与 jarredou | 438 MB | GPU 或 CPU |
| 节拍与小节 | [Beat This!](https://github.com/CPJKU/beat_this)（`final0`） | 77 MB | GPU 或 CPU |
| 钢琴音符 | [piano-transcription](https://github.com/qiuqiangkong/piano_transcription_inference)（Kong 等） | 165 MB | GPU 或 CPU |
| 贝斯与吉他音符 | [basic-pitch](https://github.com/spotify/basic-pitch)（Spotify），包内的 ONNX 模型 | 随包提供 | CPU |
| 演唱音符 | [GAME](https://github.com/openvpi/GAME) small，歌声转录器（openvpi），ONNX | 46 MB | CPU |
| 和弦（含七和弦） | [BTC](https://github.com/jayg996/BTC-ISMIR19)，和弦 Transformer（Park 等，ISMIR 2019） | 34 MB | GPU 或 CPU |
| 歌词 | Whisper large-v3-turbo 的 CTranslate2 转换版，经 faster-whisper 运行 | 1.6 GB | CPU（x86 Linux 上可用 GPU） |

调性从混音中估计，分谱用 music21 记谱、用 Verovio 渲染。所有权重在首次运行时下载，本仓库不包含任何权重。

## 快速开始

需要在 Python 3.10+ 的虚拟环境（我们用的是 3.13）中安装 [model-compose](https://github.com/hanyeol/model-compose) 0.4.117 或更高版本，另加 FFmpeg、供 yt-dlp 使用的 JavaScript 运行时（Node.js、Deno 或 Bun——没有它，部分 YouTube 视频会以 HTTP 403 失败），以及生成 PNG 页面所需的 Cairo 库（macOS 用 `brew install cairo`，Debian 和 Ubuntu 用 `sudo apt install libcairo2`）。强烈建议使用 GPU：一首 3 分钟的歌在 RTX 4090 上约一分钟，在 8 GB 的 M1 上要 25 分钟（[第 2 节](#2-能在我的设备上运行吗)）。

在你将要运行的虚拟环境里用 [uv](https://docs.astral.sh/uv/) 或 pip 安装——有几个步骤会运行 `PATH` 中的 `python`，所以它必须是该环境的：

```bash
uv pip install model-compose      # 或：pip install model-compose
```

克隆本仓库：

```bash
git clone https://github.com/MindrLabs/song-to-sheet-music
cd song-to-sheet-music
```

运行：

```bash
DEVICE=cuda model-compose up        # 没有 NVIDIA GPU 的机器上省略 DEVICE
```

首次启动会安装每个组件的包——PyTorch、bs-roformer、music21、Verovio 等，视 CUDA 库而定为 2 到 8 GB——耗时一分钟到几分钟。各组件同时安装，在新环境里首次启动可能因一个组件导入了另一个组件正在替换的包而以 `ImportError` 或 pip 错误停下。再运行一次 `model-compose up`；此时安装已完成，几秒内就能启动。我们的三台机器都没能一次启动成功（表 3）。如果每次启动都出现同一个 `No module named …`，请看故障排除。

网页 UI 在 `http://localhost:8081`，HTTP API 在 `http://localhost:8080/api`；把 `.env.sample` 复制为 `.env` 可修改 `PORT`、`SERVER_PORT` 或 `DEVICE`。在 "Song to Sheet Music" 中粘贴 YouTube 链接，给歌曲起个标题，然后运行。用 Ctrl+C 停止服务，或在本目录的另一个 shell 里运行 `model-compose down`。

请求示例：

```bash
curl -X POST http://localhost:8080/api/workflows/runs \
  -H "Content-Type: application/json" \
  -d '{"workflow_id": "full", "input": {"url": "https://youtu.be/<video-id>", "title": "My Song"}}'
```

用 `"audio": "/path/to/song.wav"` 代替 `url` 可接受磁盘上的录音，FFmpeg 能读的任何格式都行。`"parts": "drums,bass"` 只写这些分谱并跳过其余分谱的阶段——在 CPU 上，这是 25 分钟与几分钟的差别。`up` 装好包之后，同样的输入也可以在命令行使用：`model-compose run full --input '{...}'`。

| 首次运行时 | 发生什么 |
| :---: | --- |
| 权重 | 约 3.1 GB 下载到 Hugging Face 和 PyTorch 的常规缓存：Whisper 1.6 GB、BS-Roformer-SW 700 MB、DrumSep 438 MB、piano-transcription 165 MB（到 `~/.cache/models`）、Beat This! 77 MB、GAME 46 MB（到 `~/.cache/game-onnx`）、BTC 34 MB（到 `~/.cache/btc-chords`） |
| 输出 | 全部落在 `./output`：歌曲的 FLAC、各音轨、`tempo.json`、`key.json`、`chords.json`，以及每份分谱的 MusicXML、MIDI、PDF 和页面图片。一次一首歌：下一次运行会覆盖 |

MusicXML 在 MuseScore 中按所写的乐器打开——贝斯分谱像贝斯谱一样比记写低一个八度发声，六线谱轨道作为六线谱打开，鼓分谱作为 MuseScore 的鼓组打开，每次击打各有其音色。

请只处理你有权使用的录音。

## 摘要

下面的每个时间都是每台设备上一首歌跑两次、取第二次；每个一致率都是该分谱可比对的歌曲的中位数。

1. 一首 3 分钟的歌在 RTX 4090 上（歌词也用 GPU）要 57 秒（默认用 CPU 处理歌词时为 1 分 48 秒），在 DGX Spark 上 2.5 分钟，在 8 GB 的 M1 MacBook Air 上 24 分钟，大部分是音轨分离（[第 2 节](#2-能在我的设备上运行吗)）。
2. 三台机器写出了相同的小节网格、调性、和弦和鼓击打；有音高的分谱在机器之间相差几个音，钢琴约二十个音差一个，M1 的歌词约有三分之一的词不同（[2](#2-能在我的设备上运行吗)）。
3. 与 34 首歌的商业出版分谱相比，和弦根音在每首歌时长的 0.80 上一致，鼓的底鼓和军鼓 F 值为 0.94 和 0.84，贝斯音符召回率 0.76、精确率 0.78，演唱旋律的音符一致 0.71，歌词的音节一致 0.48（[表 2](#摘要)）。
4. 吉他分谱最弱：在扫弦、失真的吉他上，音符谱对出版乐谱的节奏斜线 F 值为 0.41，而节奏谱为 0.93——这种情况读节奏谱（[3.4](#34-吉他)）。
5. 钢琴分谱写出的音符是出版键盘谱的两到三倍，因为它写下听到的一切；对那些乐谱的精确音符召回率右手 0.42、左手 0.55（[3.5](#35-钢琴)）。
6. 六条规则把原始 MIDI 变成了演奏者接受的分谱：所有分谱共用一套小节网格、在歌曲段落处带排练记号的双声部鼓分谱、用音轨自身频谱校验的贝斯八度、在音轨电平跳升处拆开的重复音、可演奏的六线谱，以及旋律上方和弦、下方歌词的领奏谱（[3](#3-各分谱结果)）。
7. 安装是比较粗糙的部分：首次 `model-compose up` 会一次安装数 GB 的包，可能因安装错误停下，在此之前还有几个一次性步骤；[快速开始](#快速开始)列出了这些步骤以及启动失败时该怎么做。

表 1：按设备的摘要（一首 3 分钟的歌，两次中的第二次）

| 设备 | 能运行？ | 整首歌 | 音轨分离 | 峰值内存 |
|---|---|---|---|---|
| MacBook Air M1，8 GB | 能，在 CPU 上 | 24 分钟 | 15.3 分钟 | 3.9 GB |
| RTX 4090（24 GB） | 能 | 歌词也用 GPU 时 57 秒（默认用 CPU 时 1 分 48 秒） | 21 秒 | GPU 显存 6.3 GB |
| DGX Spark（GB10，120 GB 统一内存） | 能 | 2 分 31 秒 | 51 秒 | 6.1 GB |

表 2：与商业出版分谱的一致率（34 首流行与摇滚歌曲；中位数）

| 分谱 | 指标 | 一致率 |
|---|---|---|
| 和弦（29 首） | 根音 / 大小 / 七和弦，占歌曲时长的比例 | 0.80 / 0.76 / 0.61 |
| 贝斯（25） | 起音与音高相同的音符——召回率 / 精确率 | 0.76 / 0.78 |
| 鼓（20） | 底鼓 / 军鼓 / 踩镝 / 吊镝 / 通鼓，F 值 | 0.94 / 0.84 / 0.83 / 0.72 / 0.56 |
| 鼓（20） | 写在与乐谱相同的镝（踩镝或叮叮镝）上的击打 | 0.87 |
| 吉他（3） | 乐谱以节奏斜线记写的小节中的扫弦——节奏谱 / 音符谱，F | 0.93 / 0.41 |
| 钢琴（右手 15，左手 14） | 对出版键盘谱的精确音符召回率 | 0.42 / 0.55 |
| 人声（27） | 起音与音高相同的演唱音符，忽略八度 | 0.71 |
| 歌词（27） | 出版乐谱的音节落在乐谱所配的那个音符上 | 0.48 |
| 曲式（29） | 落在乐谱段落线上的排练记号，F | 0.41（每 8 小节一个记号：0.24） |

## 1 实验环境

表 3：设备与计时方法

| 设备 | OS，Python | model-compose，torch | 条件 | 首次启动 |
|---|---|---|---|---|
| MacBook Air M1，8 GB 统一内存 | macOS 27，3.13 | 0.4.117，torch 2.14（CPU） | 接电源，未重启；开着编辑器、运行测量的会话和一个空闲的 model-compose 服务；运行期间平均可用 1.8 GB | 两个组件同时安装 torch 时因 pip 错误停了两次，第三次尝试 61 秒启动 |
| RTX 4090 工作站（24 GB；61 GB 内存，2 块 GPU，用 1 块） | Ubuntu 22.04，3.13 | 0.4.117，torch 2.13 cu126（model-compose 为 575 驱动选择） | 共用机器；另一块 GPU 上跑着其他用户的任务，可用内存最少只剩 2.5 GB | 停了六次：第一次是在下载 torch 时，之后每次都在几秒内失败，因为各组件同时解压同一个缓存的 torch wheel。单独安装 torch 之后（见故障排除），45 秒启动 |
| DGX Spark（GB10，120 GB 统一内存） | Ubuntu 24.04 aarch64，3.13 | 0.4.117，torch 2.13 cu130 | 共用机器，空闲 | 因 pip 错误停了两次，第三次尝试 4.5 分钟启动（包括从 Zenodo 下载 165 MB 的钢琴检查点） |

- 每次运行都从全新的虚拟环境开始，按上面的快速开始操作；歌曲是 Tom McKenzie 的 [Karma](https://zenodo.org/records/19683001)（3:19，来自 AlbumDB 多轨集，CC BY 4.0），以本地文件给入。每台设备跑两次，报告第二次；RTX 4090 另外用 `LYRICS_DEVICE=cuda` 又跑了两次。峰值内存是整个进程树的常驻大小，每秒采样一次，RTX 4090 再加上 GPU 的已用显存；Spark 的统一内存无法按进程读取。
- 一致率在 34 首韩国与西方的流行与摇滚歌曲（每首 3 到 6 分钟）上测量，对照其商业出版的分谱，由脚本读取出版 PDF、按和弦进行将其小节与我们的对齐，再逐音符比较。出版分谱与歌曲均不包含在此。34 首歌在 DGX Spark 上通过流水线；第 2 节的三设备比较在每台设备上运行同样的代码。
- 权重：来自 `Politrees/UVR_resources` 的 BS-Roformer-SW `model_BandSplit-Roformer_SW_by-jarredou.ckpt` 与 MDX23C DrumSep `MDX23C-DrumSep-aufr33-jarredou.ckpt`；Beat This! `final0`；piano-transcription `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth`（Zenodo 4034264）；basic-pitch 0.4.0 的 ONNX 模型；GAME 1.0.3 small ONNX；BTC 提交 `2682317`；`dropbox-dash/faster-whisper-large-v3-turbo`，int8。
- 判定：表中没有人工判定。一位鼓手和一位吉他手读了鼓和吉他分谱的片段，回答他们会演奏哪些写下的音；他们的回答改变了记谱规则，而不是数字。

## 2 能在我的设备上运行吗

表 4：各阶段完成时刻，自开始起的秒数（3 分钟的歌，第二次运行）

| 阶段 | RTX 4090（歌词用 GPU） | DGX Spark | M1，8 GB（CPU） |
|---|---|---|---|
| 节拍与小节 | 6 | 5 | 15 |
| 调性、和弦 | 17 | 17 | 29 |
| 六轨 | 24 | 53 | 921 |
| 鼓件 | 35 | 73 | 1,438 |
| 贝斯分谱 | 39 | 86 | 987 |
| 吉他分谱、节奏谱 | 34 | 75 | 992 |
| 钢琴分谱 | 41 | 89 | 1,268 |
| 带歌词的人声分谱 | 55 | 149 | 1,224 |
| 完成 | 57 | 151 | 1,449 |

- **分离是代价所在。** 在 M1 上，仅六轨分离器就要 15 分钟，鼓件再要 9 分钟；其他分谱的转录和记谱与鼓件分离并行进行。在 RTX 4090 上，整首歌所用的时间与 Spark 的分离相当。
- **各分谱并行运行。** 音轨保存后，五个分谱同时开始；鼓件和歌词最后完成。`parts` 会连同阶段一起跳过整个分谱：在 Spark 上，只做鼓和贝斯用了 72 秒，全部分谱用了 149 秒。
- **歌词在 CPU 上运行**，除非设置 `LYRICS_DEVICE=cuda`：运行 Whisper 的 CTranslate2 没有 ARM Linux（Spark）的 CUDA 构建，也没有 Mac 的。在 Spark 上这占了 2.5 分钟中的约 1 分钟。在 RTX 4090 上这是 1 分 48 秒与 57 秒的差别，所以在那里请设置它。
- **内存。** M1 开着编辑器时 8 GB 中峰值 3.9 GB；如果还开着浏览器、已用 9 GB 交换，同一首歌曾用了 40 分钟，所以能关的请关掉。RTX 4090 峰值用了 6.3 GB 显存和 7.1 GB 内存；Spark 6.1 GB。
- **首次启动。** model-compose 在第一次启动时同时安装每个组件的包，并为你的 CUDA 驱动选择 torch 构建（RTX 4090 的 575 驱动选 cu126，Spark 选 cu130）。安装会互相冲突，三台机器都没能一次启动。M1 和 Spark 在第三次启动成功。RTX 4090 的第一次启动在下载 torch 时死掉，之后每次都因多个组件同时解压同一个缓存的 wheel 而在几秒内失败；单独安装一次 torch 后才解决。虚拟环境最终在 Mac 上为 1.7 GB，带 CUDA 时为 6 到 7.5 GB。

表 5：同一首歌在三台设备上——结果的一致程度

| 输出 | 同一设备跑两次 | 设备之间 |
|---|---|---|
| 节拍、小节、调性、和弦 | 相同 | 三台全部相同 |
| 鼓分谱 | 相同 | 三台均为 2,137 次击打；两块 GPU 逐字节相同 |
| 吉他音符 | 相同 | F 0.998–0.999（1,929–1,932 个音） |
| 贝斯音符 | 相同 | F 0.96–0.97（423–425 个音） |
| 演唱音符 | 相同 | 两块 GPU 相同；M1 对任一 GPU F 0.93 |
| 钢琴音符 | 相同 | F 0.95–0.96（197–207 个音） |
| 歌词 | 相同 | GPU 之间 90% 的词相同（两者都在 CPU 上运行 Whisper 时为 99%）；M1 与任一 GPU 之间 67–70% |

分离器的运算在 GPU 与 CPU 之间末位数字不同，转录器的阈值把它变成这里那里的一个音——钢琴最多，因为它的音轨最弱。Whisper 只听有演唱音符的地方，所以 M1 略有不同的音符让它听到不同的片段，歌词的差异因此比音符更大。记谱规则是确定性的：给定相同的音符，就得到相同的分谱。

## 3 各分谱结果

### 3.1 节拍、小节、调性与和弦

每份分谱都写在同一套小节网格上，网格取自节拍跟踪器在完整混音上找到的强拍，所以各分谱的小节号一致；6/8 的歌写成 6/8。跟踪器在某一段以歌曲脉动两倍的速度跟拍，或把同一拍找到两次时，这些拍在铺网格前被减半或删除——仅这一项修正就把和弦根音从 0.77 提到 0.80，并使写出的速度与出版乐谱相符。

调性对整首歌估计一次，每份有音高的分谱从中取调号；`key` 可覆盖它。和弦来自和弦 Transformer BTC，在小节网格上：根音与出版乐谱在歌曲时长的 0.80 上一致，大小调 0.76，七和弦 0.61（29 首）。我们最先尝试的基于色度的估计器在 BTC 得到 0.72 和 0.56 的地方只有 0.67 和 0.50。和弦没有转位，在逐音勾勒和弦的 riff 上，乐谱写一个和弦的地方它们可能每拍都变。贝斯、吉他和人声分谱印出和弦符号；鼓分谱从中读出歌曲的段落。

### 3.2 鼓

MDX23C DrumSep 把鼓音轨拆成底鼓、军鼓、通鼓、踩镝、叮叮镝和吊镝，每件的起音成为击打。对多轨录音自身的底鼓和军鼓轨道，它找回了 87% 的底鼓和 100% 的军鼓起音，无论喂入真实鼓混音还是分离出的音轨。对出版鼓谱（20 首），底鼓、军鼓和踩镝 F 值为 0.94、0.84 和 0.83，吊镝 0.72，通鼓 0.56。

鼓手要求的，以及我们写的：双声部分谱，手部符干向上、底鼓符干向下，音长和休止在拍点处拆分；在加花之后接吊镝的地方放排练记号，用字母标注，使演奏相同和弦的段落共享一个字母，记号读起来就是歌曲的曲式（A B A B C …）——它们落在出版乐谱段落线上的 F 值为 0.41，而每八小节一个记号为 0.24；踩镝还是叮叮镝，由哪件声音更响每几小节决定一次（0.87 的击打与乐谱在同一镝上）；以及一个击打级的响度门限，因为每件鼓都带着其他鼓件的串音，起音检测器会在串音里找到"击打"。门限也会丢掉真实的击打：比军鼓安静得多的通鼓和吊镝被当作串音丢弃，所以通鼓混得很低的歌会丢一些。

![鼓分谱的开头几小节：休止之后是踩镲和军鼓，接着是下声部的底鼓，上方有方框排练记号 A](docs/images/shared/drum-part.png)

图 1：为 Karma（Tom McKenzie，CC BY 4.0）写出的鼓分谱开头——双声部，手部符干向上、底鼓符干向下，带字母的排练记号。

### 3.3 贝斯

basic-pitch 在 30–400 Hz 内转录贝斯音轨；分谱随后修正转录器在分离音轨上系统性出错的地方。八度用音轨自身的频谱校验，因为模型在某些音上会选高一个八度的谐波。被转录器合并成一个长音的重复音，在音轨电平跳升处拆开。音轨音高滑向高一个八度的地方画上滑音——向下的滑音在演奏者的判断中错多于对，所以只写向上的。对出版贝斯谱（25 首），音符一致率为召回率 0.76、精确率 0.78。

六线谱放在音符下方：四根弦，旋律低于 E 时用第五根（低 B），`tab: false` 可去掉。指法是标准调弦下可演奏的一种，优先低把位，不是唱片的指法。

### 3.4 吉他

吉他音轨产出两份分谱。音符谱是 basic-pitch 在 70–1,400 Hz 内、带与贝斯相同的八度与重复音修正的初稿：在干净的分解线上可用，在扫弦、失真的吉他上几乎找不到正确的音——对出版乐谱以节奏斜线记写的扫弦，F 值为 0.41（3 首）。节奏谱把和弦符号写在从吉他起音读出的节奏斜线上方，正如出版乐谱记写扫弦段落的方式，在同样的小节上达到 0.93；音符谱密集的地方请读节奏谱。一个音轨装着歌里的所有吉他，所以两份分谱都分不出节奏吉他和主音吉他。

六线谱是一只手能按的指法，如吉他手所要求：扫弦和弦用歌曲和弦的标准指型，在该把位手能够到的跨度内（品位越往上越窄），优先低把位。它不是唱片的指法：当它为同一个音选了与出版 TAB 不同的弦时（6 首歌 2,242 个音中的 59%），多半是在更高的把位上弹。低于低 E 的音——降 D 的歌——不写入六线谱，降半音的歌品位会低一品。

### 3.5 钢琴

piano-transcription 把钢琴音轨写在大谱表上。它听到的比出版键盘谱写的多：两到三倍的音符，因为出版乐谱会编配而转录器不会。对出版键盘谱的精确音符召回率右手 0.42、左手 0.55（15 首和 14 首）。低于 `min_pitch`（默认 MIDI 36）的音作为漏进钢琴音轨的贝斯丢弃。六个音轨中，分离器最抓不住的是钢琴：对多轨录音的真实钢琴轨，BS-Roformer-SW 把 41% 的钢琴放进了钢琴音轨，而 Demucs htdemucs_6s 为 5%。

### 3.6 人声与歌词

model-compose 的转录器是为乐器设计的；在人声音轨上，basic-pitch 与出版旋律约有一半音符一致，而歌声转录器 GAME 为 0.69（6 首），所以由 GAME 写演唱音符。人声分谱是一份领奏谱——旋律上方是和弦符号，下方是歌词，一音一个音节，低音（男声）旋律写在带 8 的高音谱号上。对出版旋律（27 首），忽略八度后音符一致 0.71。和声人声共用同一音轨，可能混进旋律；喊唱或粗糙的唱法转录效果差。

歌词来自 Whisper，它只听有演唱音符的地方：听整条音轨时它会在乐队独奏处编造文字，其自带的语音检测器还会丢掉摇滚人声，而只听演唱段落时它听到了印刷歌词的 0.74（7 首，最差 0.60），整条音轨则为 0.61。然后每个音节归到最近的音符：出版乐谱 0.48 的音节落在乐谱所配的音符上，旋律音符偏差处更少——听错的词、早或晚一个音的音节，或在找到的音少于唱出的音时一个音上挤两三个音节。`lyrics_language` 告诉 Whisper 语言；不给则由它自己判断。

![领奏谱的前几行：旋律上方是和弦符号，下方是歌词，一音一个音节](docs/images/shared/lead-sheet.png)

图 2：同一首歌的领奏谱——演唱旋律上方是和弦符号，下方是歌词；男声旋律写在带 8 的高音谱号上。

## 4 建议

- **硬件：** RTX 4090 在歌词也用 GPU 时约一分钟完成一首歌，显存峰值 6.3 GB，所以 8 GB 以上的 GPU 应该装得下；我们只测了 4090 和 DGX Spark（两分半）。8 GB 的 Mac 在 CPU 上约 25 分钟完成；低于 8 GB 内存我们没有尝试。
- **在 CPU 上，只要你需要的分谱：** `"parts": "bass"` 跳过鼓件分离、钢琴和人声转录器以及歌词，几分钟完成，而全部分谱要 25 分钟。
- **歌词用 GPU：** 在带 NVIDIA GPU 的 x86 Linux 上设置 `LYRICS_DEVICE=cuda`——在 RTX 4090 上用时减半；在 Spark 或 Mac 上不要设，CTranslate2 在那里没有 CUDA 构建。
- **磁盘上的录音**以 `audio` 代替 `url` 传入：更快，不需要 JavaScript 运行时，也是不在线的歌曲唯一的入口。
- **先读鼓、贝斯和人声分谱；** 它们与出版乐谱最一致。扫弦的歌选吉他节奏谱而非吉他音符谱，并预期钢琴分谱需要删减。
- **指定调性**，当歌曲介于两个调之间时：`key.json` 会写 `low_confidence`，而错误的调号会让正确的分谱难读。
- **密集的歌**用 `"grid": 2` 更好读，它以八分音符而非十六分音符记写节奏。
- **在 MuseScore 里编辑。** MusicXML 以正确的乐器打开，带六线谱和鼓组；在那里修少数错音，而不是指望转录器来修。
- **用 Ctrl+C 停止。** `model-compose run` 不安装任何东西；先运行一次 `model-compose up`，如果首次启动死掉就再运行一次。

## 5 局限与未测量的内容

- **每首歌一个拍号。** 识别 4/4、3/4 和 6/8（12/8 按同一规则，未测试）；歌曲中途的拍号变化不识别，跟踪器读错节拍的歌到处都是错的网格。
- **出版乐谱是参考，不是真相。** 它们由出版方凭耳朵转录，有些地方与唱片不符；我们与它们不一致之处，我们没有去听来判定谁对，除了乐手读过的片段。
- **吉他数字只有三首歌**，因为很少有出版乐谱把扫弦写成斜线；其他吉他数字基于 6 首。歌词在 27 首有英文或韩文歌词的歌上打分。
- **只有节奏组。** 弦乐、管乐、合成器和第二把吉他进入"其他"音轨，不被记写。没有钢琴或贝斯的歌得到 `piano.skipped.txt` 而不是分谱。
- **鼓：** 开放踩镝写得和闭合的一样（约每十四次开放击打有一次写成叮叮镝）；鼓组的两只吊装通鼓共用一条线；其他乐器串进鼓件的击打有时被写出，安静的通鼓有时被丢掉。
- **六线谱**是可演奏的指法，不是录音的指法，也不知道降 D 或降半音的调弦。
- **跨机器的可重复性**如表 5：有音高的分谱在机器之间可差几个音，钢琴二十个音差一个，歌词最多差三分之一的词。
- **未测量：** 流行与摇滚以外的体裁、器乐曲、超过六分钟的歌、小于 24 GB 的 GPU、Apple 的 GPU（MPS；Mac 上分离器跑在 CPU 上）、慢速网络下的首次下载时间，以及第 3 节的两项比较之外分离器和转录器与替代方案的对比。

## 参考

### `full` 的输入

| 输入 | 类型 | 必需 | 默认值 | 说明 |
|---|---|---|---|---|
| `url` | string | `url`、`audio` 之一 | - | YouTube（或任何 yt-dlp 支持的）URL |
| `audio` | string | `url`、`audio` 之一 | - | 磁盘上录音的路径，FFmpeg 能读的任何格式 |
| `parts` | string | 否 | `all` | 要写的分谱：`all`，或 `drums`、`bass`、`guitar`、`piano`、`vocals` 的逗号分隔列表；未列出的分谱连同其阶段一起跳过 |
| `title` | string | 否 | - | 作为每份分谱的标题（"My Song — Bass"） |
| `key` | string | 否 | 估计 | 调性，如 `"C minor"`、`"Ab"` |
| `grid` | integer | 否 | `4` | 节奏对齐的每四分音符细分数；`4` = 十六分，`2` = 八分 |
| `tab` | boolean | 否 | `true` | 吉他和贝斯分谱下方的六线谱 |
| `language` | string | 否 | - | 演唱的语言，供 GAME 使用：`en`、`ja`、`yue` 或 `zh`。在英文歌上没有可测量的差别；其他语言不设也可以 |
| `vocal_model` | string | 否 | `small` | GAME 大小，`small`、`medium` 或 `large`。更大的与出版旋律一致率并未更高，且慢几倍 |
| `lyrics_language` | string | 否 | - | 歌词的语言，供 Whisper 使用（`ko`、`en`、…）；不设则由它自己判断 |
| `bpm` | string | 否 | `auto` | 钢琴分谱：`auto` 跟随小节网格；数字固定为一个速度 |
| `min_pitch` | integer | 否 | `36` | 钢琴分谱：低于此 MIDI 音高的音作为贝斯漏音丢弃 |

结果包含 `key`、`chords`，以及每份分谱的 `score`（MusicXML 路径、速度、拍号、小节数）和 `render`（PDF 和页面图片）——歌中没有该乐器的分谱为 `skipped` 及原因，未请求的分谱为 `null`；`vocals` 还带有 `notes` 和 `lyrics`。

### 输出文件夹

```
output/
├── 01-source.flac                 歌曲：下载的，或本地文件转为 FLAC
├── stems/                         bass, drums, other, vocals, guitar, piano (.wav)
│   └── drums/                     kick, snare, toms, hh, ride, crash (.wav)
├── tempo.json                     每一拍，含强拍
├── key.json                       歌曲的调性（含备选与置信度）
├── chords.json                    小节网格上的和弦：小节、偏移、Harte 标记、符号
├── drums.musicxml  drums.mid  drums.pdf  drums-p1.png ...
├── bass.musicxml   bass.mid   bass.pdf   bass-p1.png ...   音符，下方带六线谱
├── guitar.musicxml guitar.mid guitar.pdf guitar-p1.png ...   音符，下方带六线谱
├── guitar-rhythm.musicxml guitar-rhythm.pdf guitar-rhythm-p1.png ...   节奏谱
├── piano.musicxml  piano.mid  piano.pdf  piano-p1.png ...
├── vocals.musicxml vocals.mid vocals.pdf vocals-p1.png ...   （以及 vocals.notes.json，以秒计的演唱音符；vocals.lyrics.json，歌词）
└── <part>.skipped.txt             代替歌中没有该乐器的分谱，并说明原因
```

### 工作流

`full` 运行全曲阶段，然后每个分谱一个工作流。每个阶段本身也是一个工作流，读写同一个 `./output`，所以可以不重跑前面的步骤而只重跑一步——比如改个标题然后重跑 `engrave-bass`。

| 工作流 | 做什么 |
|---|---|
| `full` | 全部：来源、节拍、调性、和弦、音轨，然后是所请求的分谱 |
| `part-drums`、`part-bass`、`part-guitar`、`part-piano`、`part-vocals` | 从已保存的音轨出发，端到端做一份分谱 |
| `download` | 把 URL 的音频取到 `./output` |
| `beats`、`key`、`chords` | 从已保存的歌曲得到 `tempo.json`、`key.json`、`chords.json` |
| `separate`、`split-drums` | 六个音轨；鼓件 |
| `transcribe`、`transcribe-bass`、`transcribe-guitar`、`transcribe-vocals`、`transcribe-lyrics` | 一个音轨的音符；人声音符上的歌词 |
| `engrave`、`engrave-drums`、`engrave-bass`、`engrave-guitar`、`engrave-guitar-rhythm`、`engrave-vocals` | 从已保存的音符生成分谱，渲染为 PDF 和 PNG |

### 设置

环境变量，或 `model-compose.yml` 旁的 `.env`（见 `.env.sample`）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEVICE` | `cpu` | NVIDIA GPU 上用 `cuda`，供分离器和转录器使用 |
| `LYRICS_DEVICE` | `cpu` | `cuda` 在 CTranslate2 有 CUDA 构建的地方（x86 Linux）把 Whisper 跑在 GPU 上 |
| `LYRICS_MODEL` | `dropbox-dash/faster-whisper-large-v3-turbo` | Hugging Face Hub 上的另一个 faster-whisper 模型 |
| `BEAT_THIS_MODEL` | `final0` | `small0` 使用 8 MB 的节拍检查点代替 77 MB 的 |
| `PORT`、`SERVER_PORT` | `8080`、`8081` | HTTP API 和网页 UI |

### 故障排除

- **`FileNotFoundError: 'python'`** —— 启动 `model-compose up` 时虚拟环境未激活。
- **`model-compose run` 时 `No module named 'torch'`** —— 先运行一次 `model-compose up`；`run` 不安装包。
- **首次 `up` 时 `ImportError` 或 pip 错误** —— 再运行一次 `model-compose up`；各组件当时在同时安装包。如果每次启动都出现同一个 `No module named …`，说明安装冲突留下了一个只装了一半的包，而 pip 认为它已安装：单独重装该包，例如 `pip install --force-reinstall --no-deps numba`。
- **每次启动都在几秒内停在 `pip install … torch …`** —— 各组件在同时解压同一个缓存的 torch wheel。自己运行一次错误中给出的 pip 命令，然后再启动。
- **`/tmp/gradio` 下的 `PermissionError`** —— 在共用机器上，该文件夹属于其他用户的网页 UI；把 `GRADIO_TEMP_DIR` 设为你自己的文件夹。
- **下载时 HTTP 403** —— 安装 Node.js（或 Deno、Bun）；yt-dlp 会在 `PATH` 中找到它。
- **标题显示为方框** —— 安装包含标题字符（韩文、日文、中文）的字体。
- **有钢琴的歌出现 `piano.skipped.txt`** —— 音轨太弱无法转录；分离器保留的钢琴不到一半。

## 变更

| 日期 | 版本 | 变更内容 |
|---|---|---|
| 2026-10-16 | v1 | 首次发布，基于 model-compose 0.4.117 |

## 许可证

| 对象 | 许可证 | 商业使用 |
|---|---|---|
| 本仓库（工作流、`tools/` 中的脚本） | [MIT](LICENSE) | ✓ |
| BS-Roformer-SW 权重（jarredou） | 未声明¹ | 未声明¹ |
| MDX23C DrumSep 权重（aufr33 与 jarredou） | 未声明¹ | 未声明¹ |
| Beat This! 代码与 `final0` 权重（CPJKU） | [MIT](https://github.com/CPJKU/beat_this)——部分训练数据有其自身条款 | ✓ |
| piano-transcription 代码 / 检查点（Kong） | MIT / [CC BY 4.0](https://zenodo.org/records/4034264) | ✓，需署名 |
| basic-pitch（Spotify） | [Apache 2.0](https://github.com/spotify/basic-pitch) | ✓ |
| GAME 代码与 1.0.3 ONNX 权重（openvpi） | [MIT](https://github.com/openvpi/GAME)；其 README 禁止未经同意生成任何人的歌声 | ✓ |
| BTC 代码与权重（Park） | [MIT](https://github.com/jayg996/BTC-ISMIR19) | ✓ |
| Whisper large-v3-turbo，CTranslate2 转换版 | [MIT](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo) | ✓ |
| music21 / Verovio / faster-whisper / yt-dlp | BSD-3 / LGPL-3.0 / MIT / Unlicense | ✓ |
| Karma，Tom McKenzie（[AlbumDB](https://zenodo.org/records/19683001)）——计时用歌曲，以及图 1、图 2 | CC BY 4.0。不包含录音；两张图是从中转录的分谱节选 | ✓，需署名 |
| 34 首歌及其出版分谱 | 商业作品。仅用于测量，不包含 | — |

¹ **BS-Roformer-SW 与 MDX23C DrumSep。** 其作者（jarredou；aufr33 与 jarredou）发布这些检查点时未附许可证，原始发布页面——jarredou 名下的一个 GitHub 仓库和一个 Hugging Face 账户——此后已被删除。本服务下载的副本来自社区镜像 `Politrees/UVR_resources`，其 `license: mit` 标签描述的是镜像仓库，而非权重。我们没有找到作者关于商业使用的任何声明，所以该列写*未声明*。同样的检查点由 Ultimate Vocal Remover、Music-Source-Separation-Training 和 MVSep 分发和使用。权重不是本仓库的一部分；它们在首次运行时下载。如果你需要明确的许可证，可把分离器换成 `htdemucs_6s`（MIT；它保留的钢琴少得多，见 3.5 节）。截至 2026-10-13。

- 本项目以 MIT 许可证发布。
- 请只处理你有权使用的录音。
