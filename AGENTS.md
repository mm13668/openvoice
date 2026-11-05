# AGENTS.md — OpenVoice 项目代码逻辑梳理

本文件面向新加入的开发者（及 AI 编码助手），说明本仓库的代码结构、核心调用链与关键约定。

## 1. 项目简介

OpenVoice 是 MyShell 开源的**即时音色克隆（Instant Voice Cloning）**库（MIT 协议）。核心思想：

- **不直接克隆整个声音**，只克隆"音色"（tone color）。语言、口音、情绪、语速由"基础 TTS 模型"（Base Speaker TTS / MeloTTS）控制。
- 通过一个 **ToneColorConverter** 把任意源语音的音色换成目标说话人的音色，实现零样本跨语言克隆。
- V1（`checkpoints/`）支持英/中两个 base speaker；V2（`checkpoints_v2/`）音质更好，配合 [MeloTTS](https://github.com/myshell-ai/MeloTTS) 原生支持 EN/ES/FR/ZH/JP/KR。

**本仓库是一个 Python 库 + Gradio Demo + FastAPI HTTP 服务**。`openvoice/openvoice_app.py` 是 V1 的网页 Demo；`openvoice_server.py` 是面向生产/程序化调用的 HTTP 接口（推荐部署方式），对外通过 `POST /v1/voice/clone` 提供音色克隆能力。

## 2. 目录结构

```
OpenVoice/
├── openvoice/               # 核心 Python 包
│   ├── api.py               # 对外 API：BaseSpeakerTTS、ToneColorConverter
│   ├── se_extractor.py      # 参考音频处理 + 说话人音色嵌入(SE)提取
│   ├── openvoice_app.py     # Gradio Web Demo（V1）
│   ├── models.py            # VITS 类模型定义（SynthesizerTrn 等）
│   ├── modules.py / attentions.py / commons.py / transforms.py  # 网络基础组件
│   ├── mel_processing.py    # 梅尔频谱计算
│   ├── utils.py             # hp 配置加载、水印 bits 转换、句子切分等
│   └── text/                # 文本前端：cleaners、english、mandarin、symbols
├── demo_part1.ipynb         # V1：灵活语音风格控制
├── demo_part2.ipynb         # V1：跨语言克隆
├── demo_part3.ipynb         # V2：MeloTTS 多语言克隆（推荐入口）
├── openvoice/openvoice_app.py # Gradio 网页 Demo 入口（V1）
├── openvoice_server.py      # FastAPI HTTP 服务（对外 API，推荐部署用）
├── requirements-server.txt  # 服务端依赖（fastapi/uvicorn/python-multipart）
├── setup.py / requirements.txt
├── resources/               # 示例参考音频（example_reference.mp3）、logo 图片等
├── docs/                    # USAGE.md、QA.md、DEPLOYMENT.md（部署）、API.md（接口对接）
└── checkpoints(_v2)/        # 运行时需手动下载的权重（不在仓库内）
```

## 3. 核心模块与调用链

### 3.1 `openvoice/api.py` — 对外 API

两个类，均继承 `OpenVoiceBaseClass`（负责按 `config.json` 构建 `SynthesizerTrn` 并加载 `checkpoint.pth`，见 `api.py:14-39`）：

- **`BaseSpeakerTTS`**（V1 用）：文本 → 语音。`tts(text, output_path, speaker, language, speed)`：
  按语言加标记 `[EN]...[EN]` → `text_to_sequence` → `model.infer()` 合成音频（`api.py:73-98`）。
- **`ToneColorConverter`**：音色转换核心。
  - `extract_se(ref_wav_list, se_save_path)`：加载参考音频 → 梅尔频谱 → `model.ref_enc()` 得到音色嵌入 SE（多条取平均），可缓存为 `.pth`（`api.py:114-139`）。
  - `convert(audio_src_path, src_se, tgt_se, output_path, tau=0.3, message=...)`：源音频频谱 + `model.voice_conversion(spec, sid_src=src_se, sid_tgt=tgt_se, tau)` → 输出音频，并默认嵌入 `wavmark` 音频水印（`api.py:141-184`）。
  - `detect_watermark()`：检测/解码水印。
  - `enable_watermark=False` 可关闭水印；`tau` 越小音色克隆越强但音质风险越大。

### 3.2 `openvoice/se_extractor.py` — 参考音频预处理

`get_se(audio_path, vc_model, vad=True)` 是标准入口（`se_extractor.py:129-152`）：

1. 对音频内容做 sha256+base64 哈希生成缓存目录名（`processed/<name>/`）。
2. 切分音频：
   - `vad=True` → `split_audio_vad`：用 `whisper_timestamped` 的 Silero VAD 检测语音段，按 ~10s 切分。**首次运行会从 GitHub 下载 silero-vad 模型到 `~/.cache/torch/hub/`，离线环境会失败**。
   - `vad=False` → `split_audio_whisper`：用 `faster-whisper`（medium）逐句转写并按词级时间戳切分（需要 GPU + float16 硬编码在 `se_extractor.py:22`）。
3. 对所有片段调用 `ToneColorConverter.extract_se` 得到目标说话人 SE。

### 3.3 `openvoice/models.py` — 模型结构（VITS 变体）

`SynthesizerTrn`（`models.py:399`）由 TextEncoder、StochasticDurationPredictor、PosteriorEncoder、Generator(HiFi-GAN 类)、ResidualCouplingBlock(flow) 和 **ReferenceEncoder**（`models.py:301`，提取音色 SE）组成。两个推理入口：

- `infer(...)`：TTS 合成。
- `voice_conversion(y, y_lengths, sid_src, sid_tgt, tau)`：换音色（`models.py:492`）。

### 3.4 `openvoice/text/` — 文本前端

`text_to_sequence` + cleaners（英文 `english.py` 含 g2p_en/eng_to_ipa，中文 `mandarin.py` 含 pypinyin/jieba/cn2an），符号表在 `symbols.py`。

### 3.5 `openvoice/openvoice_app.py` — Gradio Demo（V1）

入口：`python -m openvoice.openvoice_app --share`。启动时加载 EN/ZH 两个 `BaseSpeakerTTS` + `ToneColorConverter` 及预置 SE（`openvoice_app.py:22-32`），用 `langid` 检测输入语言，最后 `demo.queue(); demo.launch(debug=True, show_api=True, share=args.share)`（`openvoice_app.py:274-275`）。注意 launch 未传 server_name/port，默认监听 127.0.0.1:7860，服务器部署需改代码或用反向代理。

### 3.6 `openvoice_server.py` — FastAPI HTTP 服务（推荐部署）

基于 FastAPI 的程序化调用入口，复用 V2 推理链路（MeloTTS 合成 → ToneColorConverter 换音色）：

- `GET /health`：返回设备、已加载语言、权重就绪状态。
- `POST /v1/voice/clone`（别名 `/tts`）：`multipart/form-data` 上传 `reference_audio` + `text`，可选 `language`(`EN_NEWEST/EN/ES/FR/ZH/JP/KR`)、`speaker`、`speed`、`tau`，返回 wav 二进制。
- 性能与并发：MeloTTS 模型按语言懒加载并缓存；参考音频按二进制 sha256 缓存音色嵌入（SE），重复调用更快；单进程推理用 `threading.Lock` 串行化，扩吞吐需每卡起一个进程 + Nginx 负载均衡。
- 配置可用环境变量覆盖：`OPENVOICE_DEVICE` / `OPENVOICE_V2_CKPT` / `OPENVOICE_OUTPUT_DIR` / `OPENVOICE_SPEED` / `OPENVOICE_TAU`。
- 接口契约见 [docs/API.md](docs/API.md)，部署见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

## 4. 典型推理流程（V2，对应 demo_part3.ipynb）

```
参考音频(resources/example_reference.mp3)
        │  se_extractor.get_se(vad=True)            → target_se（目标音色嵌入）
        ▼
文本 ──► MeloTTS: TTS(language, device).tts_to_file()  → tmp.wav（源语音）
        │  （源 SE 直接读 checkpoints_v2/base_speakers/ses/<spk>.pth）
        ▼
ToneColorConverter.convert(src=tmp.wav, src_se, tgt_se, message="@MyShell")
        ▼
outputs_v2/output_v2_<speaker>.wav（带水印的目标音色语音）
```

## 5. 权重与外部模型

| 资源 | 来源 | 落地位置 |
|---|---|---|
| V1 checkpoints | `https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip` | `./checkpoints/` |
| V2 checkpoints | `https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip` | `./checkpoints_v2/` |
| MeloTTS 模型 | pip 装 MeloTTS 后自动从 HuggingFace 下载 | `~/.cache/`（MeloTTS 自己管理） |
| Silero VAD | torch hub 从 GitHub 下载 | `~/.cache/torch/hub/` |
| Whisper (faster-whisper, 仅 vad=False 路径) | HuggingFace 下载 | `~/.cache/huggingface/` |
| wavmark 水印模型 | pip 安装自带 | — |

## 6. 关键依赖与约定

- Python `>=3.9`（`setup.py:27`）。`requirements.txt` 锁定 `numpy==1.22.0`、`librosa==0.9.1`、`gradio==3.48.0` 等，**建议 Python 3.9/3.10 + 独立虚拟环境**；torch 需另行按 CUDA 版本安装。
- V2 需要额外安装 `melo` 包（MeloTTS 仓库）并执行 `python -m unidic download`。
- `pydub` 依赖系统 **ffmpeg**；Linux 服务器还需 `libsndfile1`（soundfile）。
- 设备选择：`"cuda:0" if torch.cuda.is_available() else "cpu"`（demo_part3 写法），CPU 可跑但慢；注意 `api.py:18-19` 对显式指定 cuda 的配置会断言。
- 部署相关（本地/Ubuntu/Docker）见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)；HTTP 接口对接见 [docs/API.md](docs/API.md)。
