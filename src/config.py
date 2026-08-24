"""全局配置:读取 .env,集中管理路径与字体。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input_pics"
OUTPUT_DIR = ROOT / "output"

# .env 中的 LLM / VLM
BASE_URL = os.getenv("BASE_URL", "")
MODEL = os.getenv("MODEL", "")
API_KEY = os.getenv("API_KEY", "")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", BASE_URL)
VLM_MODEL = os.getenv("VLM_MODEL", "")
VLM_API_KEY = os.getenv("VLM_API_KEY", API_KEY)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "12800"))
TIMEOUT = float(os.getenv("TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# 本地 Kokoro-82M ONNX TTS(可用 .env 中 KOKORO_MODEL_DIR 覆盖)
KOKORO_MODEL_DIR = Path(
    os.getenv(
        "KOKORO_MODEL_DIR",
        r"C:\Users\lawrence\PycharmProjects\luoci_log\z_using_file\tools\models\Kokoro-82M-v1.0-ONNX",
    )
)
KOKORO_ONNX = KOKORO_MODEL_DIR / "onnx" / "model.onnx"
KOKORO_VOICES_DIR = KOKORO_MODEL_DIR / "voices"
KOKORO_VOICES_NPZ = OUTPUT_DIR / ".kokoro" / "voices.npz"
TTS_SAMPLE_RATE = 24000
DEFAULT_VOICE = "bm_george"  # 英式男声
DEFAULT_ZH_VOICE = "zm_yunxi"  # 中文男声

# 各步骤产物目录
JSON_DIR = OUTPUT_DIR / "json"
SOURCE_LANG_DIR = OUTPUT_DIR / "source_language"
TARGET_LANG_DIR = OUTPUT_DIR / "target_language"
PRON_DIR = OUTPUT_DIR / "pronunciation"
FRAMES_DIR = OUTPUT_DIR / "frames"
AUDIO_DIR = OUTPUT_DIR / "audios"
VIDEO_DIR = OUTPUT_DIR / "videos"
MUSIC_DIR = ROOT / "src" / "music"
DEFAULT_BGM = MUSIC_DIR / "booty.wav"

# Windows 字体
FONT_ZH = r"C:\Windows\Fonts\msyh.ttc"  # 微软雅黑(中文+拉丁)
FONT_EN = r"C:\Windows\Fonts\segoeui.ttf"  # Segoe UI(英文+IPA)
FONT_ZH_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"  # 微软雅黑 粗体
FONT_EN_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"  # Segoe UI 粗体


def ensure_dirs() -> None:
    for d in (
        JSON_DIR,
        SOURCE_LANG_DIR,
        TARGET_LANG_DIR,
        PRON_DIR,
        FRAMES_DIR,
        AUDIO_DIR,
        VIDEO_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
