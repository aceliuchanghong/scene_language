"""Step 4 · Video Composer — 基于 Visual Cue 视觉动效引导 (Ken Burns + 聚光灯) 的沉浸式场景视频合成。

视频结构 (1080x1920, 25fps):
    1. 片头 (~3s): 全景图展示与场景引导
    2. 正文 (遍历词汇): 镜头平滑推近至目标物体 (x, y) + 聚光灯高亮 + 悬浮大字卡片 + 单词发音与例句朗读
    3. 片尾 (~3s): 回显图C (音标层) 静止停留后淡出至黑场收束

单独运行:
    uv run python -m src.step4.cli --image input_pics/生活场景/carriage.png
    uv run python -m src.step4.cli --image ... --voice bm_george --speed 1.1 --zoom 1.7
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src import config
from src.models import SceneData, WordItem, load_scene_data

# ==============================================================================
# ==============================================================================
# 全局时间、动效、UI 与音频配置 (集中管理，方便调优与修改)
# ==============================================================================

# 1. 基础画幅与帧率
W, H = 1080, 1920
FPS = 25
CANVAS_MARGIN = 12  # 原图在竖屏画幅中的边距 (px)
AMBIENT_BLUR_RADIUS = 32  # 竖屏环境背景高斯模糊半径
AMBIENT_DARKNESS = 0.40  # 竖屏环境背景压暗比例 (保留 40% 亮度)

# 2. 悬浮教学卡片 (HUD Focus Card) 布局与视觉 (集中管理)
HUD_CARD_TOP_RATIO = 0.20  # 卡片顶部距离屏幕顶部的比例 (0.2 即 20% 高度处)
HUD_CARD_WIDTH = 1040  # 卡片宽度 (px)
HUD_CARD_HEIGHT = 420  # 卡片高度 (px)
HUD_CARD_RADIUS = 33  # 卡片圆角半径
HUD_CARD_BG_COLOR = (12, 16, 24)  # 卡片底板深色玻璃颜色
HUD_CARD_BG_ALPHA = 235  # 卡片底板不透明度 (0~255)
HUD_CARD_BORDER_COLOR = (60, 72, 90)  # 卡片边框颜色
HUD_CARD_BORDER_WIDTH = 3  # 卡片边框线宽
HUD_CARD_ACCENT_BAR_HEIGHT = 8  # 顶部强调彩色条高度
HUD_CARD_DIVIDER_COLOR = (38, 48, 62)  # 分割线颜色
HUD_CARD_TEXT_MAIN_COLOR = (255, 255, 255)  # 英文单词白色
HUD_CARD_IPA_COLOR = (100, 200, 255)  # 音标浅蓝
HUD_CARD_ZH_COLOR = (255, 214, 102)  # 中文释义暖黄
HUD_CARD_EX_EN_COLOR = (240, 246, 255)  # 英文例句颜色
HUD_CARD_EX_ZH_COLOR = (160, 175, 195)  # 中文例句翻译浅灰

# 3. 视觉引导聚光灯 (Spotlight & Visual Cue)
SPOTLIGHT_CORE_RADIUS = 110  # 聚光灯核心全亮区半径 (px)
SPOTLIGHT_OUTER_RADIUS = 380  # 聚光灯外围过渡区半径 (px)
SPOTLIGHT_MAX_DARKNESS = 0.38  # 聚光灯外围最暗系数 (0.38 表示压暗至 38%)
SPOTLIGHT_CORE_BOOST = 1.05  # 聚光灯核心区亮度增益 (1.05)

# 4. 目标准星指示器 (Reticle & Glow)
RETICLE_INNER_RADIUS = 20  # 中心准星内圈半径 (px)
RETICLE_PULSE_BASE = 24  # 呼吸光环基准半径 (px)
RETICLE_PULSE_AMP = 10  # 呼吸光环扩散振幅 (px)
RETICLE_PULSE_FREQ = 6.0  # 呼吸闪烁频率
RETICLE_TICK_LEN = 8  # 十字准星标尺刻线长度

# 5. 逐词正文运镜动效 (Visual Cue Glide Pan)
DEFAULT_ZOOM = 1.7  # 聚焦目标物体时的放大倍率
DEFAULT_TRANS_FIRST = 0.50  # 首个词汇推镜时长 (秒, 全景 -> 特写 1.7x)
DEFAULT_TRANS_GLIDE = 0.35  # 后续词汇平移漫游时长 (秒, 从上一词滑行至当前词)
BREATHING_AMP = 0.008  # 漫游停留时的微幅呼吸浮动
BREATHING_FREQ = 2.0  # 呼吸浮动频率

# 6. 片头 3 张全景图展示与过渡
DEFAULT_LAYER_DURS = [
    1.5,
    4.0,
    4.0,
]  # 片头 3 张图展示时长(中文1.5s, 双语4.0s, 音标4.0s)
DEFAULT_XFADE_DUR = 0.30  # 片头图片之间交叉淡化时长(秒)

# 7. TTS 语音朗读与音频留白
DEFAULT_SPEED = 1.2  # 默认英文例句 TTS 朗读语速 (1.0 为原速, 1.2 为加速 20%)
DEFAULT_AUDIO_PRE_PAD = 0.06  # 朗读前静音留白时长(秒)
DEFAULT_AUDIO_POST_PAD = 0.50  # 朗读后静音留白时长(秒)
DEFAULT_MIN_SEG_DUR = 1.20  # 单个词汇片段保底最小时长(秒)

# 8. 片尾收束
DEFAULT_OUTRO_HOLD_DUR = 4.0  # 片尾回显第 3 张图(图C)静止停留时长(秒)
OUTRO_FADE_DUR = 0.50  # 片尾结束前淡出至黑场的时长(秒)

# 9. 背景音乐 (BGM) 智能压音与混音
BGM_VOLUME_INTRO = 0.80  # 片头静止图展示阶段 BGM 音量 (0.0~1.0, 较大声)
BGM_VOLUME_DUCKED = 0.10  # 正文 TTS 朗读阶段 BGM 压低音量 (0.0~1.0, 轻柔旋律)
BGM_DUCK_LEAD = 0.40  # BGM 在片头结束前提早开始压音的时间(秒)
BGM_DUCK_TAIL = 0.30  # BGM 压音过渡到最低音量的时间点(片头结束后秒数)
BGM_FADE_OUT_DUR = 0.50  # 视频结尾 BGM 平滑淡出时长(秒)

# 10. 输出画面整体缩放 (内容等比缩小后居中，四周补黑)
OUTPUT_SHRINK_FRACTION = (
    1 / 6
)  # 缩小量：内容保留原尺寸的 5/6，画布仍保持 1080x1920 黑底

# 11. 并发与加速渲染
MAX_RENDER_WORKERS = min(4, os.cpu_count() or 4)  # 并行渲染词汇片段数 (建议 2~4)

# ==============================================================================

# 调色盘
PALETTE = [
    (240, 180, 41),  # 暖金
    (64, 196, 255),  # 亮青
    (239, 83, 80),  # 珊瑚红
    (102, 187, 106),  # 翡翠绿
    (171, 71, 188),  # 紫罗兰
    (255, 138, 101),  # 暖橙
    (38, 166, 154),  # 薄荷青
    (141, 110, 99),  # 暖褐
    (92, 107, 192),  # 靛蓝
    (236, 64, 122),  # 玫粉
    (0, 172, 193),  # 孔雀蓝
    (212, 225, 87),  # 青柠
]

# 音色前缀 -> espeak 语言代码 (Kokoro 多语言音素化)
_VOICE_LANG = {
    "af": "en-us",
    "am": "en-us",
    "bf": "en-gb",
    "bm": "en-gb",
    "ef": "es",
    "em": "es",
    "ff": "fr-fr",
    "fm": "fr-fr",
    "hf": "hi",
    "hm": "hi",
    "if": "it",
    "im": "it",
    "jf": "ja",
    "jm": "ja",
    "pf": "pt",
    "pm": "pt",
    "zf": "cmn",
    "zm": "cmn",
}


def _build_voices_npz() -> str:
    """把 voices/ 下单体 .bin 合并成一个 npz (带 512 行补齐)。"""
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
    def __init__(self, no_cache: bool = False):
        from kokoro_onnx import Kokoro

        self.kokoro = Kokoro(
            model_path=str(config.KOKORO_ONNX), voices_path=_build_voices_npz()
        )
        self.no_cache = no_cache
        self.cache_dir = config.AUDIO_DIR / ".tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def synth(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        import hashlib

        text = text.strip()
        if not text:
            return np.zeros(1, dtype=np.float32)
        key = f"{text}|{voice}|{speed:.2f}"
        cache = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.npy"
        if cache.exists() and not self.no_cache:
            return np.load(cache)
        lang = _VOICE_LANG.get(voice[:2], "en-us")
        phonemes = self.kokoro.tokenizer.phonemize(text, lang)
        voice_style = self.kokoro.get_voice_style(voice)
        parts = []
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
                {
                    "input_ids": ids,
                    "style": style,
                    "speed": np.array([speed], dtype=np.float32),
                },
            )[0]
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        out = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        np.save(cache, out)
        return out


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """文本自动折行。"""
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    lines: list[str] = []
    cur = ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if d.textlength(cur + ch, font=font) <= max_w or not cur:
            cur += ch
        else:
            if " " in cur and ch != " ":
                head, tail = cur.rsplit(" ", 1)
                lines.append(head)
                cur = tail + ch
            else:
                lines.append(cur)
                cur = ch
    if cur:
        lines.append(cur)
    return lines


class VisualCueAnimator:
    """负责单帧渲染：Ken Burns 运镜、聚光灯遮罩、目标锚点指示器与悬浮教学卡片。"""

    def __init__(self, image_path: Path):
        self.src_img = Image.open(image_path).convert("RGB")
        self.src_w, self.src_h = self.src_img.size

        # 字体加载 (教学卡片字体统一放大 1.5 倍)
        self.f_sub = _font(config.FONT_ZH, 24)
        self.f_word_en = _font(config.FONT_EN_BOLD, 69)
        self.f_word_ipa = _font(config.FONT_EN, 42)
        self.f_word_zh = _font(config.FONT_ZH_BOLD, 48)
        self.f_ex_en = _font(config.FONT_EN, 42)
        self.f_ex_zh = _font(config.FONT_ZH, 36)
        self.f_badge = _font(config.FONT_EN_BOLD, 33)
        self.f_reticle = _font(config.FONT_EN_BOLD, 18)

        # 预先生成环境背景 (高斯模糊 + 压暗)
        bg = self.src_img.resize((W, H), Image.BILINEAR).filter(
            ImageFilter.GaussianBlur(AMBIENT_BLUR_RADIUS)
        )
        self.ambient_bg = Image.eval(bg, lambda p: int(p * AMBIENT_DARKNESS)).convert("RGB")

        # 原图在竖屏画布未缩放时的基准显示矩形
        margin = CANVAS_MARGIN
        max_w, max_h = W - 2 * margin, H - 2 * margin
        ratio = min(max_w / self.src_w, max_h / self.src_h)
        self.base_w = int(self.src_w * ratio)
        self.base_h = int(self.src_h * ratio)
        self.base_rx = (W - self.base_w) // 2
        self.base_ry = (H - self.base_h) // 2

        # 预先计算聚光灯径向加权模板 (大幅提升逐帧渲染速度)
        r_core = SPOTLIGHT_CORE_RADIUS
        r_outer = SPOTLIGHT_OUTER_RADIUS
        yy, xx = np.ogrid[-r_outer : r_outer + 1, -r_outer : r_outer + 1]
        dist = np.sqrt(xx**2 + yy**2)
        template = np.zeros((2 * r_outer + 1, 2 * r_outer + 1), dtype=np.float32)
        template[dist <= r_core] = 1.0
        trans = (dist > r_core) & (dist < r_outer)
        ratio_t = (dist[trans] - r_core) / (r_outer - r_core)
        template[trans] = 0.5 * (1.0 + np.cos(np.pi * ratio_t))
        self.spot_template = template

        # 预先构建 100 级亮度查找表 (C 语言级快速 LUT 点运算)
        self.lut_cache = {
            i: [int(p * (i / 100.0)) for p in range(256)] * 3 for i in range(101)
        }

        # 当前词汇缓存
        self.cached_card: Image.Image | None = None
        self.card_pos = (0, 0)
        self.cur_theme_color = (255, 255, 255)

    def prepare_word(
        self,
        word: WordItem,
        word_idx: int,
        total_words: int,
    ) -> None:
        """为当前词预渲染静态 HUD 教学卡片（每词仅执行一次，避免每帧重复排版计算）。"""
        theme_color = PALETTE[(word_idx - 1) % len(PALETTE)]
        self.cur_theme_color = theme_color

        card_w = HUD_CARD_WIDTH
        card_h = HUD_CARD_HEIGHT
        card_x0 = (W - card_w) // 2
        card_y0 = int(H * HUD_CARD_TOP_RATIO)
        self.card_pos = (card_x0, card_y0)

        # 创建带透明通道的整张卡片
        card_img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(card_img)

        # 卡片底板 (深色玻璃质感)
        d.rounded_rectangle(
            [(0, 0), (card_w, card_h)],
            radius=HUD_CARD_RADIUS,
            fill=(*HUD_CARD_BG_COLOR, HUD_CARD_BG_ALPHA),
        )

        # 卡片边框与顶部强调条
        d.rounded_rectangle(
            [(0, 0), (card_w, card_h)],
            radius=HUD_CARD_RADIUS,
            outline=HUD_CARD_BORDER_COLOR,
            width=HUD_CARD_BORDER_WIDTH,
        )
        d.rounded_rectangle(
            [(30, 0), (240, HUD_CARD_ACCENT_BAR_HEIGHT)],
            radius=3,
            fill=(*theme_color, 255),
        )

        # 卡片内容：
        # 第 1 行：序号角标 + 英文大字 + 音标 + 中文释义
        tag_text = f"FOCUS {word_idx:02d}"
        d.rounded_rectangle(
            [(42, 36), (216, 90)],
            radius=9,
            fill=(*theme_color, 255),
        )
        d.text(
            (129, 63),
            tag_text,
            font=self.f_badge,
            fill=(15, 20, 30),
            anchor="mm",
        )

        # 单词英文
        en_x = 246
        d.text(
            (en_x, 63),
            word.en,
            font=self.f_word_en,
            fill=HUD_CARD_TEXT_MAIN_COLOR,
            anchor="lm",
        )
        en_len = int(d.textlength(word.en, font=self.f_word_en))

        # 音标
        ipa_x = en_x + en_len + 27
        d.text(
            (ipa_x, 66),
            word.ipa,
            font=self.f_word_ipa,
            fill=HUD_CARD_IPA_COLOR,
            anchor="lm",
        )
        ipa_len = int(d.textlength(word.ipa, font=self.f_word_ipa))

        # 中文含义
        zh_x = ipa_x + ipa_len + 30
        if zh_x < card_w - 270:
            d.text(
                (zh_x, 63),
                f"· {word.zh}",
                font=self.f_word_zh,
                fill=HUD_CARD_ZH_COLOR,
                anchor="lm",
            )
        else:
            d.text(
                (card_w - 48, 63),
                word.zh,
                font=self.f_word_zh,
                fill=HUD_CARD_ZH_COLOR,
                anchor="rm",
            )

        # 分割线
        div_y = 126
        d.line(
            [(42, div_y), (card_w - 42, div_y)],
            fill=HUD_CARD_DIVIDER_COLOR,
            width=2,
        )

        # 第 2 行：英文例句 (自动换行)
        ex_en_lines = _wrap_text(word.example_en, self.f_ex_en, card_w - 90)[:3]
        line_y = div_y + 24
        for line in ex_en_lines:
            d.text(
                (48, line_y),
                line,
                font=self.f_ex_en,
                fill=HUD_CARD_EX_EN_COLOR,
                anchor="la",
            )
            line_y += self.f_ex_en.size + 12

        # 第 3 行：中文翻译 (自动换行)
        line_y += 6
        ex_zh_lines = _wrap_text(word.example_zh, self.f_ex_zh, card_w - 90)[:2]
        for line in ex_zh_lines:
            d.text(
                (48, line_y),
                line,
                font=self.f_ex_zh,
                fill=HUD_CARD_EX_ZH_COLOR,
                anchor="la",
            )
            line_y += self.f_ex_zh.size + 9

        self.cached_card = card_img

    def render_cue_frame(
        self,
        word: WordItem,
        word_idx: int,
        total_words: int,
        t: float,
        duration: float,
        zoom_target: float = DEFAULT_ZOOM,
        prev_word: WordItem | None = None,
    ) -> Image.Image:
        """渲染某一时刻 t 的高品质教学帧。"""
        canvas = self.ambient_bg.copy()

        # 1. 运镜平滑过渡参数计算
        t_trans = DEFAULT_TRANS_FIRST if prev_word is None else DEFAULT_TRANS_GLIDE
        if t < t_trans:
            tau = t / t_trans
            alpha = 3 * (tau**2) - 2 * (tau**3)
        else:
            tau = 1.0
            alpha = 1.0 + BREATHING_AMP * math.sin((t - t_trans) * BREATHING_FREQ)

        if prev_word is None:
            current_zoom = 1.0 + (zoom_target - 1.0) * min(alpha, 1.0)
            start_x = self.src_w / 2
            start_y = self.src_h / 2
        else:
            current_zoom = zoom_target + (
                BREATHING_AMP * math.sin((t - t_trans) * BREATHING_FREQ)
                if t >= t_trans
                else 0.0
            )
            start_x = prev_word.x * self.src_w
            start_y = prev_word.y * self.src_h

        # 2. 原图裁剪与视口计算
        cw = self.src_w / current_zoom
        ch = self.src_h / current_zoom
        target_src_x = word.x * self.src_w
        target_src_y = word.y * self.src_h

        src_center_x = start_x + (target_src_x - start_x) * min(alpha, 1.0)
        src_center_y = start_y + (target_src_y - start_y) * min(alpha, 1.0)

        crop_x0 = max(0.0, min(src_center_x - cw / 2, self.src_w - cw))
        crop_y0 = max(0.0, min(src_center_y - ch / 2, self.src_h - ch))
        crop_x1 = crop_x0 + cw
        crop_y1 = crop_y0 + ch

        cropped = self.src_img.crop(
            (int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1))
        )
        fg = cropped.resize((self.base_w, self.base_h), Image.BILINEAR)

        # 3. 计算聚光灯与准星在原图中的平滑坐标
        spot_src_x = start_x + (target_src_x - start_x) * min(alpha, 1.0)
        spot_src_y = start_y + (target_src_y - start_y) * min(alpha, 1.0)
        u = (spot_src_x - crop_x0) / cw
        v = (spot_src_y - crop_y0) / ch
        px = int(self.base_rx + u * self.base_w)
        py = int(self.base_ry + v * self.base_h)

        # 4. 高性能聚光灯局部贴合 (LUT 查表压暗 + 局部切片计算，速度提升 20 倍)
        rel_px = px - self.base_rx
        rel_py = py - self.base_ry

        spotlight_strength = min(alpha, 1.0) if prev_word is None else 1.0
        base_darkness = 1.0 - (1.0 - SPOTLIGHT_MAX_DARKNESS) * spotlight_strength

        # 使用快速 LUT 查表对背景进行压暗 (在 C 语言层 0.5ms 完成)
        lut_idx = int(round(base_darkness * 100))
        fg_dimmed = fg.point(self.lut_cache[lut_idx])

        # 仅对 760x760 聚光灯影响区域做切片叠加计算
        r_outer = SPOTLIGHT_OUTER_RADIUS
        y0 = max(0, rel_py - r_outer)
        y1 = min(self.base_h, rel_py + r_outer + 1)
        x0 = max(0, rel_px - r_outer)
        x1 = min(self.base_w, rel_px + r_outer + 1)

        if y1 > y0 and x1 > x0:
            ty0 = y0 - (rel_py - r_outer)
            ty1 = ty0 + (y1 - y0)
            tx0 = x0 - (rel_px - r_outer)
            tx1 = tx0 + (x1 - x0)

            patch_crop = fg.crop((x0, y0, x1, y1))
            patch_arr = np.array(patch_crop, dtype=np.float32)
            kernel_slice = self.spot_template[ty0:ty1, tx0:tx1]
            boost_factor = SPOTLIGHT_CORE_BOOST - base_darkness
            patch_mult = base_darkness + boost_factor * kernel_slice
            patch_res = np.clip(patch_arr * patch_mult[:, :, None], 0, 255).astype(np.uint8)
            patch_img = Image.fromarray(patch_res)
            fg_dimmed.paste(patch_img, (x0, y0))

        canvas.paste(fg_dimmed, (self.base_rx, self.base_ry))

        # 5. 目标锚点指示器 (Reticle & Pulse Glow - 高性能局部微图合成)
        theme_color = self.cur_theme_color
        pulse = (math.sin(t * RETICLE_PULSE_FREQ) + 1.0) / 2.0
        r_inner = RETICLE_INNER_RADIUS
        r_pulse = int(RETICLE_PULSE_BASE + RETICLE_PULSE_AMP * pulse)
        pulse_alpha = int(180 * (1.0 - pulse))

        # 小区域绘制呼吸外圈光环
        pad = r_pulse + 4
        glow_patch = Image.new("RGBA", (2 * pad, 2 * pad), (0, 0, 0, 0))
        glow_d = ImageDraw.Draw(glow_patch)
        glow_d.ellipse(
            [(pad - r_pulse, pad - r_pulse), (pad + r_pulse, pad + r_pulse)],
            outline=(*theme_color, pulse_alpha),
            width=2,
        )
        canvas.paste(glow_patch, (px - pad, py - pad), glow_patch)

        # 直接在 canvas 上绘制实心准星与刻线 (零内存复制开销)
        d = ImageDraw.Draw(canvas)
        d.ellipse(
            [(px - r_inner, py - r_inner), (px + r_inner, py + r_inner)],
            outline=theme_color,
            width=3,
        )
        d.ellipse([(px - 5, py - 5), (px + 5, py + 5)], fill=(255, 255, 255))

        tick = RETICLE_TICK_LEN
        d.line([(px - r_inner - tick, py), (px - r_inner + 2, py)], fill=theme_color, width=2)
        d.line([(px + r_inner - 2, py), (px + r_inner + tick, py)], fill=theme_color, width=2)
        d.line([(px, py - r_inner - tick), (px, py - r_inner + 2)], fill=theme_color, width=2)
        d.line([(px, py + r_inner - 2), (px, py + r_inner + tick)], fill=theme_color, width=2)

        # 6. 贴合预渲染的 HUD 悬浮教学卡片
        if self.cached_card is not None:
            canvas.paste(self.cached_card, self.card_pos, self.cached_card)

        return canvas


def _write_wav(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, config.TTS_SAMPLE_RATE)


def _render_word_segment_video(
    animator: VisualCueAnimator,
    word: WordItem,
    word_idx: int,
    total_words: int,
    audio_wav: Path,
    duration: float,
    out_mp4: Path,
    zoom_target: float = DEFAULT_ZOOM,
    prev_word: WordItem | None = None,
) -> Path:
    """将单词的连续帧流式灌入 FFmpeg 并合成为 MP4 视频片段。"""
    animator.prepare_word(word=word, word_idx=word_idx, total_words=total_words)
    num_frames = int(round(duration * FPS))

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-i",
        str(audio_wav),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
        str(out_mp4),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_idx in range(num_frames):
        t = frame_idx / FPS
        frame_img = animator.render_cue_frame(
            word=word,
            word_idx=word_idx,
            total_words=total_words,
            t=t,
            duration=duration,
            zoom_target=zoom_target,
            prev_word=prev_word,
        )
        proc.stdin.write(frame_img.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 渲染片段失败: {out_mp4}")
    return out_mp4


def _intro_layers_clip(
    stills: list[Path],
    out_mp4: Path,
    layer_durs: list[float] | None = None,
    xfade_dur: float = DEFAULT_XFADE_DUR,
) -> float:
    """片头 3 层全景图依次展示(中文层 1.5s -> 双语层 3.0s -> 音标层 3.0s)，带 0.3s 平滑交叉淡化与静音轨。返回总时长。"""
    if layer_durs is None:
        layer_durs = DEFAULT_LAYER_DURS
    if len(layer_durs) < len(stills):
        layer_durs = list(layer_durs) + [layer_durs[-1]] * (
            len(stills) - len(layer_durs)
        )

    inputs: list[str] = []
    for p, dur in zip(stills, layer_durs):
        inputs += ["-loop", "1", "-t", f"{dur:.2f}", "-i", str(p)]
    total = sum(layer_durs[: len(stills)]) - (len(stills) - 1) * xfade_dur

    fc, prev = [], "[0:v]"
    cur_offset = 0.0
    for k in range(1, len(stills)):
        cur_offset += layer_durs[k - 1] - xfade_dur
        lab = f"[v{k}]"
        fc.append(
            f"{prev}[{k}:v]xfade=transition=fade:duration={xfade_dur:.2f}:offset={cur_offset:.2f}{lab}"
        )
        prev = lab

    audio_input = [
        "-f",
        "lavfi",
        "-t",
        f"{total:.2f}",
        "-i",
        "anullsrc=r=44100:cl=stereo",
    ]
    filter_str = ";".join(fc) + f";{prev}null[vout]"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        *inputs,
        *audio_input,
        "-filter_complex",
        filter_str,
        "-map",
        "[vout]",
        "-map",
        f"{len(stills)}:a",
        "-t",
        f"{total:.2f}",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    return total


def _outro_hold_clip(
    still: Path,
    out_mp4: Path,
    hold_dur: float = DEFAULT_OUTRO_HOLD_DUR,
) -> float:
    """片尾：回显图C静止停留 hold_dur 秒，结尾淡出至黑场后收束。返回总时长。"""
    fade_st = max(0.0, hold_dur - OUTRO_FADE_DUR)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-t",
        f"{hold_dur:.2f}",
        "-i",
        str(still),
        "-f",
        "lavfi",
        "-t",
        f"{hold_dur:.2f}",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-vf",
        f"fade=t=out:st={fade_st:.2f}:d={OUTRO_FADE_DUR:.2f}",
        "-t",
        f"{hold_dur:.2f}",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    return hold_dur


def _apply_shrink_pad(src: Path, dst: Path) -> None:
    """把视频内容整体缩小 1/6 (保留 5/6 尺寸) 并居中放到 1080x1920 黑色画布上 (保持 9:16)。"""
    sw = round(W * (1 - OUTPUT_SHRINK_FRACTION))
    sh = round(H * (1 - OUTPUT_SHRINK_FRACTION))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            f"scale={sw}:{sh},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "copy",
            str(dst),
        ],
        check=True,
    )


def compose_video(
    json_path: Path,
    voice: str = config.DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    zoom_target: float = DEFAULT_ZOOM,
    layer_durs: list[float] | None = None,
    xfade_dur: float = DEFAULT_XFADE_DUR,
    bgm_path: Path | None = config.DEFAULT_BGM,
    no_cache: bool = False,
) -> Path:
    if layer_durs is None:
        layer_durs = DEFAULT_LAYER_DURS

    t_step4_start = time.perf_counter()
    data: SceneData = load_scene_data(json_path)
    stem = Path(data.image).stem
    src_image_path = Path(data.image)
    if not src_image_path.exists():
        src_image_path = config.INPUT_DIR / data.image

    stills = [
        config.SOURCE_LANG_DIR / f"{stem}.png",
        config.TARGET_LANG_DIR / f"{stem}.png",
        config.PRON_DIR / f"{stem}.png",
    ]
    missing = [str(p) for p in stills if not p.exists()]
    if missing:
        raise SystemExit(f"[step4] 缺少渲染产物 {missing}, 请先执行 step3")

    print("[step4] 初始化 Kokoro TTS 模型…")
    tts = TTS(no_cache=no_cache)

    work_dir = config.VIDEO_DIR / f".work_{stem}"
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_concat = work_dir / "raw_concat.mp4"
    final_video = config.VIDEO_DIR / f"{stem}.mp4"

    animator = VisualCueAnimator(src_image_path)
    sr = config.TTS_SAMPLE_RATE

    parts: list[Path] = []

    # 1. 片头 3 层全景图展示 (中文层 1.5s -> 双语层 3.0s -> 音标层 3.0s，0.3s淡化)
    t_intro_start = time.perf_counter()
    intro_mp4 = work_dir / "00_intro.mp4"
    total_intro_t = _intro_layers_clip(
        stills,
        intro_mp4,
        layer_durs=layer_durs,
        xfade_dur=xfade_dur,
    )
    t_intro_elapsed = time.perf_counter() - t_intro_start
    print(f"[step4] 片头 3 层全景图过渡展示生成完毕 (总时长 {total_intro_t:.1f}s, 耗时 {t_intro_elapsed:.1f}s)")
    parts.append(intro_mp4)

    # 2. 逐词视觉动效引导 (Visual Cue Main Flow - 紧凑高能版，并行加速渲染)
    total_words = len(data.words)
    print(f"[step4] 开始生成 {total_words} 个词汇的 Visual Cue 视觉动效漫游片段…")
    t_words_start = time.perf_counter()

    word_durations: list[float] = [0.0] * total_words
    word_tasks = []

    for idx, word in enumerate(data.words, 1):
        prev_word = data.words[idx - 2] if idx > 1 else None
        # 仅朗读地道英文例句 (极短前/后留白 + 地道语速)
        a_ex = tts.synth(word.example_en, voice=voice, speed=speed)

        pre_pad = np.zeros(int(DEFAULT_AUDIO_PRE_PAD * sr), dtype=np.float32)
        post_pad = np.zeros(int(DEFAULT_AUDIO_POST_PAD * sr), dtype=np.float32)

        combined_audio = np.concatenate([pre_pad, a_ex, post_pad])
        seg_dur = max(
            len(combined_audio) / sr, DEFAULT_MIN_SEG_DUR
        )  # 单个词汇片段保底最小时长
        word_durations[idx - 1] = seg_dur

        wav_path = work_dir / f"seg_{idx:02d}.wav"
        _write_wav(wav_path, combined_audio)

        seg_mp4 = work_dir / f"seg_{idx:02d}.mp4"
        word_tasks.append((
            src_image_path,
            word,
            idx,
            total_words,
            wav_path,
            seg_dur,
            seg_mp4,
            zoom_target,
            prev_word,
        ))

    def _render_task(task_args):
        (
            src_img_p,
            w,
            w_idx,
            tot_w,
            w_path,
            s_dur,
            s_mp4,
            z_target,
            p_word,
        ) = task_args
        anim = VisualCueAnimator(src_img_p)
        t0 = time.perf_counter()
        _render_word_segment_video(
            animator=anim,
            word=w,
            word_idx=w_idx,
            total_words=tot_w,
            audio_wav=w_path,
            duration=s_dur,
            out_mp4=s_mp4,
            zoom_target=z_target,
            prev_word=p_word,
        )
        return w_idx, w.zh, w.en, s_dur, time.perf_counter() - t0

    if MAX_RENDER_WORKERS > 1 and total_words > 1:
        print(f"[step4] 启用 {MAX_RENDER_WORKERS} 并发加速渲染…")
        with ThreadPoolExecutor(max_workers=MAX_RENDER_WORKERS) as pool:
            futures = [pool.submit(_render_task, t) for t in word_tasks]
            for fut in as_completed(futures):
                w_idx, zh, en, dur, el = fut.result()
                print(
                    f"       [{w_idx:02d}/{total_words:02d}] 聚焦: {zh} ({en}) 完成 "
                    f"(片段 {dur:.1f}s, 耗时 {el:.1f}s)"
                )
    else:
        for t in word_tasks:
            w_idx, zh, en, dur, el = _render_task(t)
            print(
                f"       [{w_idx:02d}/{total_words:02d}] 聚焦: {zh} ({en}) 完成 "
                f"(片段 {dur:.1f}s, 耗时 {el:.1f}s)"
            )

    for idx in range(1, total_words + 1):
        parts.append(work_dir / f"seg_{idx:02d}.mp4")

    t_words_elapsed = time.perf_counter() - t_words_start
    print(
        f"[step4] {total_words} 个词汇漫游片段全部生成完毕 "
        f"(总耗时 {t_words_elapsed:.1f}s, 平均 {t_words_elapsed / max(1, total_words):.1f}s/词)"
    )

    # 3. 片尾收束: 回显图C静止 3s 后淡出关闭
    t_outro_start = time.perf_counter()
    outro_mp4 = work_dir / "99_outro.mp4"
    total_outro_t = _outro_hold_clip(stills[2], outro_mp4)
    t_outro_elapsed = time.perf_counter() - t_outro_start
    print(f"[step4] 片尾图C回显 {total_outro_t:.1f}s (结尾淡出收束) 生成完毕 (耗时 {t_outro_elapsed:.1f}s)")
    parts.append(outro_mp4)

    # 4. 视频初步无缝拼接 (3图全景 + 逐词漫游 TTS 视频 + 片尾回显)
    concat_list = work_dir / "concat_list.txt"
    concat_list.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in parts),
        encoding="utf-8",
    )

    print(f"[step4] 正在拼接基础音视频序列 -> {raw_concat}")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(raw_concat),
        ],
        check=True,
    )

    # 5. 全局 BGM 动态智能压音混音与画面缩放 (单次编解码高效完成)
    sw = round(W * (1 - OUTPUT_SHRINK_FRACTION))
    sh = round(H * (1 - OUTPUT_SHRINK_FRACTION))
    shrink_filter = f"scale={sw}:{sh},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black"

    total_video_t = total_intro_t + sum(word_durations) + total_outro_t
    if bgm_path and Path(bgm_path).exists():
        print(
            f"[step4] 正在注入全篇智能压音 BGM ({Path(bgm_path).name}) 并缩放居中 -> {final_video}"
        )
        t1 = max(0.0, total_intro_t - BGM_DUCK_LEAD)
        t2 = total_intro_t + BGM_DUCK_TAIL
        t_duck = max(0.1, t2 - t1)
        t_fade = max(0.0, total_video_t - BGM_FADE_OUT_DUR)

        audio_filter = (
            f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{total_video_t:.2f},asetpts=PTS-STARTPTS,"
            f"volume='if(lte(t,{t1:.2f}),{BGM_VOLUME_INTRO:.2f},if(lte(t,{t2:.2f}),{BGM_VOLUME_INTRO:.2f}-(t-{t1:.2f})/{t_duck:.2f}*({BGM_VOLUME_INTRO:.2f}-{BGM_VOLUME_DUCKED:.2f}),{BGM_VOLUME_DUCKED:.2f}))':eval=frame,"
            f"afade=t=out:st={t_fade:.2f}:d={BGM_FADE_OUT_DUR:.2f},"
            f"aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[bgm];"
            f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[voice];"
            f"[voice][bgm]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1[aout]"
        )

        filter_complex = f"[0:v]{shrink_filter}[vout];{audio_filter}"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_concat),
                "-i",
                str(bgm_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(final_video),
            ],
            check=True,
        )
    else:
        print(
            f"[step4] 输出画面整体缩小 {OUTPUT_SHRINK_FRACTION:.2f} 并居中补黑 -> {final_video}"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_concat),
                "-vf",
                shrink_filter,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(final_video),
            ],
            check=True,
        )

    t_step4_total = time.perf_counter() - t_step4_start
    print(
        f"[step4] 视频合成完毕! 输出: {final_video} "
        f"(视频总长 {total_video_t:.1f}s, Step4 总耗时 {t_step4_total:.1f}s)"
    )
    return final_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Step4 · Visual Cue 场景视频合成")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--json", default=None, help="指定 JSON 路径")
    parser.add_argument(
        "--voice",
        default=config.DEFAULT_VOICE,
        help=f"英文音色 (默认 {config.DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"朗读语速 (默认 {DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=DEFAULT_ZOOM,
        help=f"推镜缩放倍率 (默认 {DEFAULT_ZOOM})",
    )
    parser.add_argument(
        "--layer-durs",
        type=float,
        nargs="+",
        default=DEFAULT_LAYER_DURS,
        help=f"片头每张分层图展示时长秒 (默认 {DEFAULT_LAYER_DURS})",
    )
    parser.add_argument(
        "--bgm",
        default=str(config.DEFAULT_BGM),
        help="背景音乐文件路径 (默认 src/music/booty.wav)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="忽略 TTS 语音缓存重新合成"
    )
    args = parser.parse_args()
    config.ensure_dirs()
    json_path = (
        Path(args.json)
        if args.json
        else config.JSON_DIR / f"{Path(args.image).stem}.json"
    )
    if not json_path.exists():
        raise SystemExit(f"找不到 {json_path}, 请先运行 step1 / step2 / step3")
    bgm_path = Path(args.bgm) if args.bgm else None
    compose_video(
        json_path,
        voice=args.voice,
        speed=args.speed,
        zoom_target=args.zoom,
        layer_durs=args.layer_durs,
        bgm_path=bgm_path,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
