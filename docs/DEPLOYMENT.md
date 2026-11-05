# OpenVoice 部署指南

本文说明两种部署方式：本地部署（Windows / Linux 均适用）与 Ubuntu 服务器部署。
安装原理相同，区别主要在系统依赖与对外服务方式。

## 目录

- [准备：下载模型权重](#准备下载模型权重)
- [一、本地部署](#一本地部署)
  - [Windows](#windows)
  - [macOS](#macos)
- [二、Ubuntu 服务器部署](#二ubuntu-服务器部署)
  - [CPU 版](#方案-a-cpu-版推荐先跑通)
  - [GPU 版](#方案-b-gpu-版)
  - [对外提供服务](#对外提供服务)
  - [systemd 常驻服务](#systemd-常驻服务可选)
  - [Docker 部署](#docker-部署可选)
- [常见问题](#常见问题)

---

## 准备：下载模型权重

无论哪种平台，先在仓库根目录下载并解压权重（V1/V2 至少下其一，推荐 V2）：

```bash
# V1（base_speakers + converter）
curl -L -o checkpoints_1226.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip
unzip checkpoints_1226.zip          # 得到 ./checkpoints/

# V2（base_speakers/ses + converter）
curl -L -o checkpoints_v2_0417.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
unzip checkpoints_v2_0417.zip       # 得到 ./checkpoints_v2/
```

解压后目录结构应包含：

```
checkpoints_v2/
├── base_speakers/ses/*.pth      # MeloTTS 各说话人的源 SE
└── converter/{config.json, checkpoint.pth}
```

> 注意：必须解压到项目根目录下的 `checkpoints` / `checkpoints_v2`，demo 里的路径是相对路径。

---

## 一、本地部署

### 环境要求

- Python **3.9**（强依赖 `numpy==1.22.0`、`librosa==0.9.1`，3.11+ 会安装失败）
- NVIDIA GPU + CUDA（可选，CPU 可跑但慢数倍~数十倍）
- ffmpeg（`pydub` 处理 mp3 等格式必需）

### Windows

```powershell
# 1. 安装 ffmpeg（二选一）
winget install Gyan.FFmpeg
# 或 choco install ffmpeg

# 2. 创建虚拟环境（conda 或 venv）
conda create -n openvoice python=3.9 -y
conda activate openvoice

# 3. 安装 PyTorch（按需二选一）
# GPU（CUDA 12.1 示例）：
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU：
pip install torch torchaudio

# 4. 安装 OpenVoice 本体
cd d:\wk\OpenVoice
pip install -e .

# 5. V2 需要额外安装 MeloTTS（V1 跳过此步）
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download

# 6. 运行
#    - 交互式体验：打开 demo_part1/2/3.ipynb 逐格运行
#    - Gradio 网页 Demo（V1 权重）：
python -m openvoice.openvoice_app          # 本地访问 http://127.0.0.1:7860
python -m openvoice.openvoice_app --share  # 生成临时公网链接
```

### macOS

```bash
conda create -n openvoice python=3.9 -y && conda activate openvoice
pip install torch torchaudio    # Apple Silicon 自动装 MPS 版
pip install -e .
# V2 同样需要: pip install git+https://github.com/myshell-ai/MeloTTS.git && python -m unidic download
brew install ffmpeg
```

### 验证安装成功

```bash
python -c "from openvoice.api import ToneColorConverter; print('ok')"
# 再跑 demo_part3.ipynb，能在 outputs_v2/ 生成 wav 即成功
```

---

## 二、Ubuntu 服务器部署

### 通用系统依赖

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 build-essential git
# Python 3.9：Ubuntu 22.04 自带 3.10 也可（numpy 1.22 支持 ≤3.10）；
# 24.04（Python 3.12）需通过 Miniconda/pyenv 安装 3.9 或 3.10
```

### 方案 A：CPU 版（推荐先跑通）

```bash
git clone https://github.com/myshell-ai/OpenVoice.git && cd OpenVoice
python3.10 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio          # CPU 版 torch
pip install -e .
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
pip install -r requirements-server.txt      # HTTP API 服务依赖（FastAPI/uvicorn）
# 下载权重（见上文"准备"部分）
curl -L -o checkpoints_v2_0417.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
unzip checkpoints_v2_0417.zip
```

先在服务器上跑一遍 notebook 或一段最小脚本验证，无 GPU 时把 `device` 设为 `"cpu"`。

### 方案 B：GPU 版

1. 安装 NVIDIA 驱动：

   ```bash
   sudo apt install -y nvidia-driver-535   # 或官网 .run 包
   nvidia-smi                              # 确认驱动 OK
   ```

2. 装对应 CUDA 的 PyTorch（无需单独安装 CUDA Toolkit，pip 包自带运行时）：

   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   python -c "import torch; print(torch.cuda.is_available())"   # 应输出 True
   ```

3. 其余步骤与 CPU 版相同。

### 对外提供服务

`openvoice_app.py` 的 launch 默认只监听 `127.0.0.1:7860`，两种做法：

**做法 1：改监听地址**（简单直接），把 `openvoice/openvoice_app.py:275` 改为：

```python
demo.launch(server_name="0.0.0.0", server_port=7860, debug=False, show_api=True)
```

然后用防火墙/安全组放行 7860 端口，访问 `http://<服务器IP>:7860`。

**做法 2：Nginx 反向代理**（推荐，可加 HTTPS）：

```nginx
server {
    listen 80;
    server_name your.domain.com;
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;     # Gradio 需要 WebSocket
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;
        client_max_body_size 50m;                    # 允许上传参考音频
    }
}
```

> 也可以用 `--share` 让 Gradio 生成临时公网隧道链接，仅适合临时演示。

### systemd 常驻服务（可选）

`/etc/systemd/system/openvoice.service`：

```ini
[Unit]
Description=OpenVoice Gradio App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/OpenVoice
ExecStart=/home/ubuntu/OpenVoice/venv/bin/python -m openvoice.openvoice_app
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now openvoice
journalctl -u openvoice -f      # 看日志
```

### HTTP API 服务（程序化调用，推荐）

如果你需要通过接口对接业务系统（而不是网页 Demo），本仓库提供了 [FastAPI 服务](openvoice_server.py)，
对外暴露 `POST /v1/voice/clone`：上传参考音频 + 文本，返回克隆音色的 wav。接口契约见 [docs/API.md](API.md)。

1. 安装服务端依赖（在已装好 OpenVoice + MeloTTS 的虚拟环境内）：

   ```bash
   pip install -r requirements-server.txt
   ```

2. 启动（默认 8000 端口，监听 0.0.0.0）：

   ```bash
   python openvoice_server.py --host 0.0.0.0 --port 8000
   # 或 uvicorn（workers 保持 1，单设备推理串行）：
   uvicorn openvoice_server:app --host 0.0.0.0 --port 8000 --workers 1
   ```

3. 验证：

   ```bash
   curl http://127.0.0.1:8000/health
   curl -F "text=Hello" -F "reference_audio=@resources/example_reference.mp3" \
        -F "language=EN" http://127.0.0.1:8000/v1/voice/clone -o out.wav
   ```

4. Nginx 反向代理到 8000（推荐加 HTTPS）：

   ```nginx
   server {
       listen 80;
       server_name your.domain.com;
       client_max_body_size 50m;                 # 允许上传参考音频
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_read_timeout 300s;              # 首包加载/长文本可能慢
       }
   }
   ```

   > 加 API Key 鉴权：在 Nginx 用 `auth_request`，或在 `openvoice_server.py` 里加一个依赖校验请求头（生产环境强烈建议，否则任何人都能调用你的 GPU）。

5. systemd 常驻（API 版）`/etc/systemd/system/openvoice-api.service`：

   ```ini
   [Unit]
   Description=OpenVoice HTTP API
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/OpenVoice
   Environment=OPENVOICE_DEVICE=cuda:0
   Environment=OPENVOICE_V2_CKPT=/home/ubuntu/OpenVoice/checkpoints_v2
   Environment=PYTHONUNBUFFERED=1
   ExecStart=/home/ubuntu/OpenVoice/venv/bin/python openvoice_server.py --host 127.0.0.1 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   sudo systemctl daemon-reload && sudo systemctl enable --now openvoice-api
   journalctl -u openvoice-api -f
   ```

6. 高并发：单进程推理串行（`threading.Lock`），要扩吞吐就**每卡起一个进程**，前面 Nginx 做负载均衡；不要单纯调大 workers（共享同一 GPU 反而竞争）。

### Docker 部署（可选）

社区有现成指南：https://github.com/StevenJSCF/OpenVoice/blob/update-docs/docs/DF_USAGE.md 。
自建镜像的要点：基础镜像 `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`（GPU）或 `python:3.10-slim`（CPU），
`apt install ffmpeg libsndfile1`，`pip install torch ... -e . + MeloTTS`，再 `pip install -r requirements-server.txt`，
把 `checkpoints*` 目录 COPY 或挂载进容器，`EXPOSE 8000` 并以 `python openvoice_server.py --host 0.0.0.0 --port 8000` 启动；GPU 场景加 `--gpus all`。

---

## 常见问题

1. **numpy/librosa 安装冲突**：坚持用 Python 3.9/3.10，先装 torch 再 `pip install -e .`。
2. **首次运行卡在下载**：`se_extractor.py` 的 VAD 路径会从 GitHub 下载 silero-vad 到 `~/.cache/torch/hub/`，MeloTTS 会从 HuggingFace 下载模型。国内服务器建议设置镜像：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
   silero 离线方案见 [QA.md](QA.md)。
3. **mp3 读取报错**：缺少 ffmpeg / libsndfile1。
4. **显存不足**：Whisper medium 需要较多显存（仅 `vad=False` 路径用），用 `vad=True` 可避开；纯转换推理约占 2-4GB。
5. **服务器无公网下载权限**：提前在本机下载好权重、`~/.cache/torch/hub`（silero）、HuggingFace 缓存后整包拷贝到服务器。
6. **声音不像/音质差**：参考音频需干净、单人、无长静音，且改名后删除 `processed/` 缓存目录，详见 [QA.md](QA.md)。
