#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenVoice HTTP 服务（基于 FastAPI）

对外暴露音色克隆（voice cloning）接口，内部复用 V2 的推理链路：
    MeloTTS 合成基础语音 -> ToneColorConverter 换成目标音色 -> 输出 wav

启动：
    pip install -r requirements-server.txt
    python openvoice_server.py --host 0.0.0.0 --port 8000

也可用 uvicorn：
    uvicorn openvoice_server:app --host 0.0.0.0 --port 8000 --workers 1
（注意：推理在单设备上进行，workers 不要超过 1，或用多进程部署多张卡）
"""

import os
import io
import threading
import hashlib
import tempfile
from pathlib import Path

import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from openvoice import api as openvoice_api

# ----------------------------------------------------------------------------
# 配置（可用环境变量覆盖）
# ----------------------------------------------------------------------------
DEVICE = os.environ.get(
    "OPENVOICE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
)
V2_CKPT = os.environ.get("OPENVOICE_V2_CKPT", "checkpoints_v2")
OUTPUT_DIR = os.environ.get("OPENVOICE_OUTPUT_DIR", "outputs_server")
SPEED_DEFAULT = float(os.environ.get("OPENVOICE_SPEED", "1.0"))
TAU_DEFAULT = float(os.environ.get("OPENVOICE_TAU", "0.3"))
# 允许的语言（与 MeloTTS 一致）
SUPPORTED_LANGUAGES = ["EN_NEWEST", "EN", "ES", "FR", "ZH", "JP", "KR"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="OpenVoice API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# 模型状态（懒加载，进程内缓存）
# ----------------------------------------------------------------------------
_tone_color_converter = None
_tts_models = {}          # language -> melo TTS 实例
_source_se_cache = {}     # f"{language}|{speaker}" -> se tensor
_target_se_cache = {}     # audio_sha256 -> se tensor
_lock = threading.Lock()  # 串行化 GPU/CPU 推理，避免并发冲突

CONVERTER_CKPT = Path(V2_CKPT) / "converter"


def _load_converter() -> ToneColorConverter:
    global _tone_color_converter
    if _tone_color_converter is None:
        if not (CONVERTER_CKPT / "config.json").exists():
            raise RuntimeError(
                f"未找到 ToneColorConverter 权重：{CONVERTER_CKPT}。"
                f"请先下载 checkpoints_v2 并解压到 '{V2_CKPT}'（见 docs/DEPLOYMENT.md）。"
            )
        _tone_color_converter = ToneColorConverter(
            f"{CONVERTER_CKPT}/config.json", device=DEVICE
        )
        _tone_color_converter.load_ckpt(f"{CONVERTER_CKPT}/checkpoint.pth")
    return _tone_color_converter


def _load_tts(language: str):
    if language in _tts_models:
        return _tts_models[language]
    try:
        from melo.api import TTS
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="未安装 MeloTTS，请先执行：pip install git+https://github.com/myshell-ai/MeloTTS.git "
                   "&& python -m unidic download",
        ) from e
    model = TTS(language=language, device=DEVICE)
    _tts_models[language] = model
    return model


def _resolve_source_se_path(language: str, speaker: str):
    """定位基础说话人的音色嵌入(.pth)。

    V2 权重里 ses 文件名形如 en-newest / en-us / zh / fr / kr / es 等，
    与 MeloTTS 的 spk2id 键名并不总是一一对应（例如没有 en-default.pth）。
    这里做多重回退，尽量按语言自动选到存在的文件。
    """
    ses_dir = Path(V2_CKPT) / "base_speakers" / "ses"
    lang_lower = language.lower()
    # 1) 显式指定 speaker（支持 melo 键名，也支持直接给文件名 stem）
    if speaker:
        for cand in (speaker.lower().replace("_", "-"), speaker):
            p = ses_dir / f"{cand}.pth"
            if p.exists():
                return p
    # 2) 按语言推导：优先 <lang>-newest，再 <lang>，再任意以 <lang> 开头
    for cand in (f"{lang_lower}-newest", lang_lower):
        p = ses_dir / f"{cand}.pth"
        if p.exists():
            return p
    matches = sorted(ses_dir.glob(f"{lang_lower}*.pth"))
    if matches:
        return matches[0]
    # 3) 兜底：整个目录第一个 pth
    any_se = sorted(ses_dir.glob("*.pth"))
    if any_se:
        return any_se[0]
    raise HTTPException(
        status_code=400,
        detail=f"找不到任何基础说话人的音色嵌入，请检查 {ses_dir} 下是否存在 V2 权重",
    )


def _get_source_se(language: str, speaker: str, tts):
    key = f"{language}|{speaker}"
    if key in _source_se_cache:
        return _source_se_cache[key]
    se_path = _resolve_source_se_path(language, speaker)
    se = torch.load(str(se_path), map_location=DEVICE)
    _source_se_cache[key] = se
    return se


def _get_target_se(audio_bytes: bytes, ext: str):
    """用参考音频内容哈希做缓存，相同参考音频只提取一次。"""
    h = hashlib.sha256(audio_bytes).hexdigest()
    if h in _target_se_cache:
        return _target_se_cache[h]

    suffix = ext if ext.startswith(".") else f".{ext}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        ref_path = tmp.name

    converter = _load_converter()
    try:
        target_se, _ = se_extractor.get_se(ref_path, converter, vad=True)
    finally:
        os.remove(ref_path)

    target_se = target_se.to(DEVICE)
    _target_se_cache[h] = target_se
    return target_se


def _do_clone(text, audio_bytes, ext, language, speaker, speed, tau):
    """核心推理（在锁内同步执行）。返回 wav 文件路径。"""
    with _lock:
        converter = _load_converter()
        tts = _load_tts(language)
        speaker_ids = tts.hps.data.spk2id
        if speaker is None:
            speaker = list(speaker_ids.keys())[0]
        if speaker not in speaker_ids:
            raise HTTPException(
                status_code=400,
                detail=f"说话人 '{speaker}' 不存在，可选：{list(speaker_ids.keys())}",
            )
        speaker_id = speaker_ids[speaker]

        source_se = _get_source_se(language, speaker, tts)
        target_se = _get_target_se(audio_bytes, ext)

        # 1) MeloTTS 合成基础语音
        src_path = tempfile.mktemp(suffix=".wav")
        tts.tts_to_file(text, speaker_id, src_path, speed=speed)

        # 2) ToneColorConverter 换成目标音色（带 wavmark 水印）
        out_path = tempfile.mktemp(suffix=".wav")
        converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            tau=tau,
            message="@MyShell",
        )
        os.remove(src_path)
    return out_path


# ----------------------------------------------------------------------------
# 接口
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        converter_ready = (CONVERTER_CKPT / "config.json").exists()
    except Exception:
        converter_ready = False
    return {
        "status": "ok",
        "device": DEVICE,
        "torch_cuda_available": torch.cuda.is_available(),
        "converter_ready": converter_ready,
        "loaded_tts_languages": list(_tts_models.keys()),
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.post("/v1/voice/clone")
def clone_voice(
    text: str = Form(..., description="要合成的文本"),
    reference_audio: UploadFile = File(..., description="参考音频文件（任意语言，建议干净单人）"),
    language: str = Form("EN", description="输出语言：EN_NEWEST/EN/ES/FR/ZH/JP/KR"),
    speaker: str = Form(None, description="说话人键名，留空用该语言默认第一个"),
    speed: float = Form(SPEED_DEFAULT, description="语速，1.0 为原始速度"),
    tau: float = Form(TAU_DEFAULT, description="音色强度，越小克隆越强（0~1，推荐 0.3）"),
):
    """上传参考音频 + 文本，返回克隆音色后的 wav 音频。"""
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的语言 '{language}'，可选：{SUPPORTED_LANGUAGES}",
        )
    audio_bytes = reference_audio.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="参考音频为空")
    ext = Path(reference_audio.filename or "ref.wav").suffix or ".wav"

    try:
        out_path = _do_clone(text, audio_bytes, ext, language, speaker, speed, tau)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"推理失败：{e}") from e

    def _cleanup(path: str):
        try:
            os.remove(path)
        except OSError:
            pass

    bg = BackgroundTasks()
    bg.add_task(_cleanup, out_path)

    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename="cloned_voice.wav",
        background=bg,
    )


# 兼容 alias（部分老客户端可能用 /tts）
@app.post("/tts")
def tts_alias(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    language: str = Form("EN"),
    speaker: str = Form(None),
    speed: float = Form(SPEED_DEFAULT),
    tau: float = Form(TAU_DEFAULT),
):
    return clone_voice(text, reference_audio, language, speaker, speed, tau)


# ----------------------------------------------------------------------------
# 命令行启动
# ----------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenVoice HTTP API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn_run = __import__("uvicorn").run
    uvicorn_run(
        "openvoice_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
