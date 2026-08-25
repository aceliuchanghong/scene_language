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

# 支持的语言
SUPPORTED_LANGUAGES = ["en", "ja", "ko"]
DEFAULT_LANG = "en"

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
DEFAULT_VOICE = "bm_george"  # 默认英文音色(英式男声)
DEFAULT_ZH_VOICE = "zm_yunxi"  # 中文男声

# 本地 OmniVoice 多语言 Zero-Shot TTS(用于日韩等语言)
OMNIVOICE_DIR = Path(
    os.getenv(
        "OMNIVOICE_DIR",
        r"C:\Users\lawrence\PycharmProjects\luoci_log\z_using_file\tools\OmniVoice",
    )
)
OMNIVOICE_PYTHON = OMNIVOICE_DIR / "OmniVoice" / ".venv" / "Scripts" / "python.exe"
OMNIVOICE_CLI = OMNIVOICE_DIR / "cli.py"
OMNIVOICE_MODEL_DIR = Path(
    os.getenv(
        "OMNIVOICE_MODEL_DIR",
        r"C:\Users\lawrence\PycharmProjects\luoci_log\z_using_file\tools\models\OmniVoice",
    )
)

# 各语言默认音色与随机种子
DEFAULT_VOICES = {
    "en": "bm_george",
    "ja": "female, middle-aged, moderate pitch",
    "ko": "male, middle-aged, low pitch",
}
DEFAULT_SEED = 42
DEFAULT_SPEECH_SPEEDS = {
    "en": 1.2,
    "ja": 1.0,
    "ko": 1.0,
}

# 预提取的专属音色 Prompt 文件 (.pt)
VOICES_DIR = ROOT / "src" / "voices"
DEFAULT_VOICE_PTS = {
    "ja": VOICES_DIR / "ja_f2.pt",
    "ko": VOICES_DIR / "ko_f.pt",
}



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
FONT_ZH = r"C:\Windows\Fonts\msyh.ttc"  # 微软雅黑(中文)
FONT_ZH_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"  # 微软雅黑 粗体
FONT_EN = r"C:\Windows\Fonts\segoeui.ttf"  # Segoe UI(英文+IPA)
FONT_EN_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"  # Segoe UI 粗体
FONT_JA = r"C:\Windows\Fonts\YuGothM.ttc"  # 游黑体 中等(日文)
FONT_JA_BOLD = r"C:\Windows\Fonts\YuGothB.ttc"  # 游黑体 粗体(日文)
FONT_KO = r"C:\Windows\Fonts\malgun.ttf"  # Malgun Gothic(韩文)
FONT_KO_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"  # Malgun Gothic 粗体(韩文)


def get_lang_fonts(lang: str = "en") -> dict[str, str]:
    """返回指定语言的字体映射: regular, bold, zh, zh_bold。"""
    lang = lang.lower()
    if lang == "ja":
        return {
            "regular": FONT_JA if Path(FONT_JA).exists() else FONT_ZH,
            "bold": FONT_JA_BOLD if Path(FONT_JA_BOLD).exists() else FONT_ZH_BOLD,
            "zh": FONT_ZH,
            "zh_bold": FONT_ZH_BOLD,
        }
    if lang == "ko":
        return {
            "regular": FONT_KO if Path(FONT_KO).exists() else FONT_ZH,
            "bold": FONT_KO_BOLD if Path(FONT_KO_BOLD).exists() else FONT_ZH_BOLD,
            "zh": FONT_ZH,
            "zh_bold": FONT_ZH_BOLD,
        }
    return {
        "regular": FONT_EN,
        "bold": FONT_EN_BOLD,
        "zh": FONT_ZH,
        "zh_bold": FONT_ZH_BOLD,
    }


def get_json_path(stem: str, lang: str = "en") -> Path:
    """获取指定语言的 JSON 产物路径。en 保持 <stem>.json 以完全向后兼容。"""
    if lang.lower() == "en":
        return JSON_DIR / f"{stem}.json"
    return JSON_DIR / f"{stem}_{lang.lower()}.json"


def get_video_path(stem: str, lang: str = "en") -> Path:
    """获取指定语言的视频产物路径。en 保持 <stem>.mp4 以完全向后兼容。"""
    if lang.lower() == "en":
        return VIDEO_DIR / f"{stem}.mp4"
    return VIDEO_DIR / f"{stem}_{lang.lower()}.mp4"


def get_layer_paths(stem: str, lang: str = "en") -> dict[str, Path]:
    """获取指定语言的 3 层标注图路径。"""
    suffix = "" if lang.lower() == "en" else f"_{lang.lower()}"
    return {
        "source": SOURCE_LANG_DIR / f"{stem}.png",
        "target": TARGET_LANG_DIR / f"{stem}{suffix}.png",
        "pronunciation": PRON_DIR / f"{stem}{suffix}.png",
    }


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
