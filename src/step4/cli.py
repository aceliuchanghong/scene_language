"""Step 4 · Video Composer — Kokoro-82M TTS + FFmpeg 合成教学视频。

视频结构(1080x1920, 25fps):
    阶段1 中文层    : 逐词播中文(zm_yunxi)
    阶段2 双语层    : 逐词播英文(bm_george)
    阶段3 聚光帧    : 每词一帧,播英文(慢速)
    阶段4 词汇表格  : 快速跟读英文

单独运行:
    uv run python -m src.step4.cli --image input_pics/生活场景/carriage.png
    uv run python -m src.step4.cli --image ... --voice bm_george --speed 1.0
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np

from src import config
from src.models import SceneData, load_scene_data

FPS = 25
LEAD, TAIL = 0.30, 0.55  # 每段音频前后留白(秒)
MIN_SEG = 1.6  # 单段最短时长
GAP_TABLE = 0.25  # 表格阶段词间隔

# 音色前缀 -> espeak 语言代码(Kokoro 多语言音素化)
_VOICE_LANG = {
    "af": "en-us", "am": "en-us", "bf": "en-gb", "bm": "en-gb",
    "ef": "es", "em": "es", "ff": "fr-fr", "fm": "fr-fr",
    "hf": "hi", "hm": "hi", "if": "it", "im": "it",
    "jf": "ja", "jm": "ja", "pf": "pt", "pm": "pt",
    "zf": "cmn", "zm": "cmn",
}


def _build_voices_npz() -> str:
    """把 voices/ 下单体 .bin 合并成一个 npz(带 512 行补齐)。"""
    import os

    if config.KOKORO_VOICES_NPZ.exists():
        return str(config.KOKORO_VOICES_NPZ)
    config.KOKORO_VOICES_NPZ.parent.mkdir(parents=True, exist_ok=True)
    tables = {}
    for bin_file in sorted(glob.glob(str(config.KOKORO_VOICES_DIR / "*.bin"))):
        name = Path(bin_file).stem
        table = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 1, 256)
        if table.shape[0] < 512:
            pad = np.repeat(table[-1:], 512 - table.shape[0], axis=0)
            table = np.concatenate([table, pad], axis=0)
        tables[name] = table
    np.savez(config.KOKORO_VOICES_NPZ, **tables)
    return str(config.KOKORO_VOICES_NPZ)


class TTS:
    def __init__(self):
        from kokoro_onnx import Kokoro

        self.kokoro = Kokoro(
            model_path=str(config.KOKORO_ONNX), voices_path=_build_voices_npz()
        )
        self.cache_dir = config.AUDIO_DIR / ".tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def synth(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        import hashlib

        key = f"{text}|{voice}|{speed}"
        cache = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.npy"
        if cache.exists():
            return np.load(cache)
        lang = _VOICE_LANG.get(voice[:2], "en-us")
        phonemes = self.kokoro.tokenizer.phonemize(text, lang)
        voice_style = self.kokoro.get_voice_style(voice)
        parts = []
        # Kokoro 上下文 512,首尾各 1 个 pad,每段 token <= 505
        cur, cur_len, chunks = [], 0, []
        for w in phonemes.split():
            if cur_len + len(w) + 1 > 505 and cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            cur.append(w)
            cur_len += len(w) + 1
        if cur:
            chunks.append(" ".join(cur))
        for chunk in chunks or [phonemes]:
            tokens = self.kokoro.tokenizer.tokenize(chunk)
            style = np.asarray(voice_style[len(tokens)], dtype=np.float32)
            ids = np.array([[0, *tokens, 0]], dtype=np.int64)
            audio = self.kokoro.sess.run(
                None,
                {"input_ids": ids, "style": style, "speed": np.array([speed], dtype=np.float32)},
            )[0]
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        out = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        np.save(cache, out)
        return out


def _pad(audio: np.ndarray, lead: float, tail: float, min_total: float) -> np.ndarray:
    sr = config.TTS_SAMPLE_RATE
    pre = int(lead * sr)
    post = max(int(tail * sr), int(min_total * sr) - pre - len(audio))
    return np.concatenate([np.zeros(pre), audio, np.zeros(post)])


def _write_wav(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, config.TTS_SAMPLE_RATE)


def _run_ffmpeg(seg_dir: Path, idx: int, image: Path, wav: Path, dur: float) -> Path:
    out = seg_dir / f"seg_{idx:03d}.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(image), "-i", str(wav),
        "-t", f"{dur:.3f}", "-r", str(FPS),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
        "-shortest", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def compose_video(
    json_path: Path,
    voice: str = config.DEFAULT_VOICE,
    speed: float = 1.0,
    zh_voice: str = config.DEFAULT_ZH_VOICE,
) -> Path:
    data: SceneData = load_scene_data(json_path)
    stem = Path(data.image).stem
    layers = {
        "source": config.SOURCE_LANG_DIR / f"{stem}.png",
        "target": config.TARGET_LANG_DIR / f"{stem}.png",
        "table": config.TABLE_DIR / f"{stem}.png",
    }
    frames = sorted((config.FRAMES_DIR / stem).glob("*.png"))
    missing = [str(p) for p in [*layers.values(), *frames] if not p.exists()]
    if missing:
        raise SystemExit(f"[step4] 缺少渲染产物 {missing},先运行 step3")

    print("[step4] 加载 Kokoro TTS 模型…")
    tts = TTS()

    def segment(image: Path, audio: np.ndarray, min_total: float = MIN_SEG) -> tuple[Path, np.ndarray, float]:
        padded = _pad(audio, LEAD, TAIL, min_total)
        return image, padded, len(padded) / config.TTS_SAMPLE_RATE

    segs: list[tuple[Path, np.ndarray, float]] = []
    # 阶段1:中文逐词
    for w in data.words:
        segs.append(segment(layers["source"], tts.synth(w.zh, zh_voice, 1.0)))
    # 阶段2:双语层英文逐词
    for w in data.words:
        segs.append(segment(layers["target"], tts.synth(w.en, voice, speed)))
    # 阶段3:聚光帧 + 慢速英文
    for img, w in zip(frames, data.words):
        segs.append(segment(img, tts.synth(w.en, voice, speed * 0.9), min_total=2.0))
    # 阶段4:表格快速跟读
    for w in data.words:
        segs.append(segment(layers["table"], tts.synth(w.en, voice, speed), min_total=1.2))

    work = config.VIDEO_DIR / f".work_{stem}"
    work.mkdir(parents=True, exist_ok=True)
    final = config.VIDEO_DIR / f"{stem}.mp4"
    parts = []
    total = sum(d for _, _, d in segs)
    print(f"[step4] 合成 {len(segs)} 段,总时长 {total / 60:.1f} 分钟…")
    for i, (img, audio, dur) in enumerate(segs, 1):
        wav = work / f"a_{i:03d}.wav"
        _write_wav(wav, audio)
        parts.append(_run_ffmpeg(work, i, img, wav, dur))
        print(f"       段 {i}/{len(segs)} ({dur:.1f}s)")

    concat_file = work / "list.txt"
    concat_file.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_file), "-c", "copy", str(final)],
        check=True,
    )
    print(f"[step4] 视频完成 -> {final}")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Step4 · TTS + 视频合成")
    parser.add_argument("--image", required=True, help="输入图片(定位 JSON 与渲染产物)")
    parser.add_argument("--json", default=None, help="直接指定 JSON 路径")
    parser.add_argument("--voice", default=config.DEFAULT_VOICE, help=f"英文音色(默认 {config.DEFAULT_VOICE})")
    parser.add_argument("--zh-voice", default=config.DEFAULT_ZH_VOICE, help=f"中文音色(默认 {config.DEFAULT_ZH_VOICE})")
    parser.add_argument("--speed", type=float, default=1.0, help="英文语速 0.5~2.0")
    args = parser.parse_args()
    config.ensure_dirs()
    json_path = Path(args.json) if args.json else config.JSON_DIR / f"{Path(args.image).stem}.json"
    if not json_path.exists():
        raise SystemExit(f"找不到 {json_path},先运行 step1/step2")
    compose_video(json_path, voice=args.voice, speed=args.speed, zh_voice=args.zh_voice)


if __name__ == "__main__":
    main()
