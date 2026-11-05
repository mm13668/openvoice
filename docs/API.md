# OpenVoice HTTP API 对接文档

基于 FastAPI 的服务端 `openvoice_server.py`，对外提供**音色克隆**能力：上传一段参考音频 + 文本，返回克隆了参考音色、用指定语言朗读文本的 wav 音频。

> 服务端要求：已按 `docs/DEPLOYMENT.md` 装好 OpenVoice（V2 权重）+ MeloTTS，
> 并已 `pip install -r requirements-server.txt`。

## 1. 启动服务

```bash
# 方式一：自带入口
python openvoice_server.py --host 0.0.0.0 --port 8000

# 方式二：uvicorn（workers 保持 1，推理在单设备上串行）
uvicorn openvoice_server:app --host 0.0.0.0 --port 8000 --workers 1

# 环境变量（可选）：
#   OPENVOICE_DEVICE     cuda:0 / cpu，默认自动检测
#   OPENVOICE_V2_CKPT    权重目录，默认 checkpoints_v2
#   OPENVOICE_OUTPUT_DIR 输出目录，默认 outputs_server
#   OPENVOICE_SPEED / OPENVOICE_TAU  默认语速/音色强度
```

启动后：
- Swagger 调试界面：`http://<host>:8000/docs`
- ReDoc：`http://<host>:8000/redoc`

---

## 2. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查，返回设备、已加载语言、权重就绪状态 |
| POST | `/v1/voice/clone` | **核心接口**：参考音频 + 文本 → 克隆音色音频(wav) |
| POST | `/tts` | 兼容别名，参数同 `/v1/voice/clone` |

---

## 3. `POST /v1/voice/clone`

**Content-Type**：`multipart/form-data`

### 请求字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | string | 是 | 要合成的文本（支持目标语言自然语言，可中英混排） |
| `reference_audio` | file | 是 | 参考音频文件（任意语言，建议干净、单人、5~20s），支持 wav/mp3 等 |
| `language` | string | 否 | 输出语言，默认 `EN`。可选：`EN_NEWEST` `EN` `ES` `FR` `ZH` `JP` `KR` |
| `speaker` | string | 否 | 该语言下的基础说话人键名；留空用默认第一个 |
| `speed` | float | 否 | 语速，默认 1.0 |
| `tau` | float | 否 | 音色强度 0~1，默认 0.3（越小克隆越强，但音质风险略增） |

### 响应

- 成功：`200`，`Content-Type: audio/wav`，body 为 wav 二进制（文件名 `cloned_voice.wav`）。
- 失败：`4xx/5xx`，JSON：`{ "detail": "错误信息" }`

### 调用示例

**curl**
```bash
curl -F "text=Hello, this is a cloned voice test." \
     -F "reference_audio=@reference.mp3" \
     -F "language=EN" \
     -F "speed=1.0" \
     -F "tau=0.3" \
     http://localhost:8000/v1/voice/clone -o output.wav
```

**Python（requests）**
```python
import requests

with open("reference.mp3", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/voice/clone",
        data={"text": "你好，这是音色克隆测试。", "language": "ZH", "speed": 1.0, "tau": 0.3},
        files={"reference_audio": f},
        timeout=120,
    )

if resp.status_code == 200:
    with open("output.wav", "wb") as out:
        out.write(resp.content)
else:
    print("Error:", resp.status_code, resp.json())
```

**JavaScript / 浏览器（fetch + FormData）**
```javascript
const form = new FormData();
form.append("text", "Bonjour, ceci est un test.");
form.append("language", "FR");
form.append("reference_audio", fileInput.files[0]); // <input type="file">

const res = await fetch("http://localhost:8000/v1/voice/clone", { method: "POST", body: form });
if (res.ok) {
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
}
```

---

## 4. `GET /health`

```bash
curl http://localhost:8000/health
```

返回示例：
```json
{
  "status": "ok",
  "device": "cuda:0",
  "torch_cuda_available": true,
  "converter_ready": true,
  "loaded_tts_languages": ["EN"],
  "supported_languages": ["EN_NEWEST","EN","ES","FR","ZH","JP","KR"]
}
```

---

## 5. 对接注意事项

1. **首次调用较慢**：首次用到某语言会懒加载 MeloTTS 模型并下载权重；之后走进程内缓存。
2. **参考音频缓存**：相同参考音频（按二进制 sha256）只做一次音色嵌入提取，重复调用更快。若更换参考音频请直接重新上传。
3. **并发**：单进程内推理串行（`threading.Lock`），高并发请多起几个进程（每张 GPU 一个），前面用 Nginx 做负载均衡（见 `docs/DEPLOYMENT.md`）。
4. **超时**：长文本 + 首包加载可能数秒到数十秒，客户端/网关超时建议设 ≥ 60s。
5. **音频格式**：返回固定为 16-bit/24kHz wav（OpenVoice 采样率）。前端如需 mp3 自行转码。
6. **水印**：输出音频含 `wavmark` 不可听水印（`@MyShell`），属官方防伪机制，不影响播放。
7. **生产安全**：服务不做鉴权，外网暴露务必放在内网 / 加 API Key / Nginx 限流，详见部署文档。
