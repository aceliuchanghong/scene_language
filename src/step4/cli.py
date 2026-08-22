"""Step 4 · Video Composer — 基于 Visual Cue 视觉动效引导 (Ken Burns + 聚光灯) 的沉浸式场景视频合成。

视频结构 (1080x1920, 25fps):
    1. 片头 (~3s): 全景图展示与场景引导
    2. 正文 (遍历词汇): 镜头平滑推近至目标物体 (x, y) + 聚光灯高亮 + 悬浮大字卡片 + 单词发音与例句朗读
    3. 片尾 (~4s): 完整词汇汇总表格打卡页 (提示截图保存)

单独运行:
    uv run python -m src.step4.cli --image input_pics/生活场景/carriage.png
    uv run python -m src.step4.cli --image ... --voice bm_george --speed 1.1 --zoom 1.7
"""

from __future__ import annotations

import argparse
import glob
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src import config
from src.models import SceneData, WordItem, load_scene_data

# ==============================================================================
# 全局时间、动效与音频配置 (集中管理，方便调优与修改)
# ==============================================================================

# 1. 基础画幅与帧率
W, H = 1080, 1920
FPS = 25

# 2. 片头 3 张全景图展示与过渡
DEFAULT_LAYER_DURS = [
    1.5,
    3.0,
    3.0,
]  # 片头 3 张图展示时长(中文1.5s, 双语3.0s, 音标3.0s)
DEFAULT_XFADE_DUR = 0.30  # 片头图片之间交叉淡化时长(秒)

# 3. 逐词正文运镜动效 (Visual Cue Glide Pan)
DEFAULT_ZOOM = 1.7  # 聚焦目标物体时的放大倍率
DEFAULT_TRANS_FIRST = 0.50  # 首个词汇推镜时长(秒, 全景 -> 特写 1.7x)
DEFAULT_TRANS_GLIDE = 0.35  # 后续词汇平移漫游时长(秒, 从上一词滑行至当前词)

# 4. TTS 语音朗读与音频留白
DEFAULT_SPEED = 1.2  # 默认英文例句 TTS 朗读语速 (1.0 为原速, 1.2 为加速 20%)
DEFAULT_AUDIO_PRE_PAD = 0.06  # 朗读前静音留白时长(秒)
DEFAULT_AUDIO_POST_PAD = 0.08  # 朗读后静音留白时长(秒)
DEFAULT_MIN_SEG_DUR = 1.20  # 单个词汇片段保底最小时长(秒)

# 5. 背景音乐 (BGM) 智能压音与混音
BGM_VOLUME_INTRO = 0.70  # 片头静止图展示阶段 BGM 音量 (0.0~1.0, 较大声)
BGM_VOLUME_DUCKED = 0.15  # 正文 TTS 朗读阶段 BGM 压低音量 (0.0~1.0, 轻柔旋律)
BGM_DUCK_LEAD = 0.40  # BGM 在片头结束前提早开始压音的时间(秒)
BGM_DUCK_TAIL = 0.30  # BGM 压音过渡到最低音量的时间点(片头结束后秒数)
BGM_FADE_OUT_DUR = 0.50  # 视频结尾 BGM 平滑淡出时长(秒)

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

        # 字体加载
        self.f_title = _font(config.FONT_ZH_BOLD, 32)
        self.f_sub = _font(config.FONT_ZH, 24)
        self.f_word_en = _font(config.FONT_EN_BOLD, 46)
        self.f_word_ipa = _font(config.FONT_EN, 28)
        self.f_word_zh = _font(config.FONT_ZH_BOLD, 32)
        self.f_ex_en = _font(config.FONT_EN, 28)
        self.f_ex_zh = _font(config.FONT_ZH, 24)
        self.f_badge = _font(config.FONT_EN_BOLD, 22)
        self.f_reticle = _font(config.FONT_EN_BOLD, 18)

        # 预先生成环境背景 (高斯模糊 + 压暗)
        bg = self.src_img.resize((W, H), Image.LANCZOS).filter(
            ImageFilter.GaussianBlur(32)
        )
        self.ambient_bg = Image.eval(bg, lambda p: int(p * 0.40)).convert("RGB")

        # 原图在竖屏画布未缩放时的基准显示矩形
        margin = 12
        max_w, max_h = W - 2 * margin, H - 2 * margin
        ratio = min(max_w / self.src_w, max_h / self.src_h)
        self.base_w = int(self.src_w * ratio)
        self.base_h = int(self.src_h * ratio)
        self.base_rx = (W - self.base_w) // 2
        self.base_ry = (H - self.base_h) // 2

    def render_cue_frame(
        self,
        word: WordItem,
        word_idx: int,
        total_words: int,
        t: float,
        duration: float,
        scene_title: str,
        zoom_target: float = DEFAULT_ZOOM,
        prev_word: WordItem | None = None,
    ) -> Image.Image:
        """渲染某一时刻 t 的高品质教学帧。"""
        canvas = self.ambient_bg.copy()

        # 1. 运镜平滑过渡参数计算 (首词推镜 DEFAULT_TRANS_FIRST, 平移漫游 DEFAULT_TRANS_GLIDE 极速丝滑)
        t_trans = DEFAULT_TRANS_FIRST if prev_word is None else DEFAULT_TRANS_GLIDE
        if t < t_trans:
            tau = t / t_trans
            # Smoothstep 缓动
            alpha = 3 * (tau**2) - 2 * (tau**3)
        else:
            tau = 1.0
            alpha = 1.0 + 0.008 * math.sin((t - t_trans) * 2.0)

        if prev_word is None:
            current_zoom = 1.0 + (zoom_target - 1.0) * min(alpha, 1.0)
            start_x = self.src_w / 2
            start_y = self.src_h / 2
        else:
            current_zoom = zoom_target + (
                0.008 * math.sin((t - t_trans) * 2.0) if t >= t_trans else 0.0
            )
            start_x = prev_word.x * self.src_w
            start_y = prev_word.y * self.src_h

        # 2. 原图裁剪与视口计算
        cw = self.src_w / current_zoom
        ch = self.src_h / current_zoom
        target_src_x = word.x * self.src_w
        target_src_y = word.y * self.src_h

        # 中心平滑插值 (从起始位置渐变到目标中心)
        src_center_x = start_x + (target_src_x - start_x) * min(alpha, 1.0)
        src_center_y = start_y + (target_src_y - start_y) * min(alpha, 1.0)

        # 裁剪边界约束
        crop_x0 = max(0.0, min(src_center_x - cw / 2, self.src_w - cw))
        crop_y0 = max(0.0, min(src_center_y - ch / 2, self.src_h - ch))
        crop_x1 = crop_x0 + cw
        crop_y1 = crop_y0 + ch

        cropped = self.src_img.crop(
            (int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1))
        )
        fg = cropped.resize((self.base_w, self.base_h), Image.LANCZOS)

        # 3. 计算聚光灯与准星在原图中的平滑坐标 (随运镜平移)
        spot_src_x = start_x + (target_src_x - start_x) * min(alpha, 1.0)
        spot_src_y = start_y + (target_src_y - start_y) * min(alpha, 1.0)
        u = (spot_src_x - crop_x0) / cw
        v = (spot_src_y - crop_y0) / ch
        px = int(self.base_rx + u * self.base_w)
        py = int(self.base_ry + v * self.base_h)

        # 4. 聚光灯径向遮罩处理 (使用 Numpy 快速加权)
        fg_arr = np.array(fg, dtype=np.float32)
        fg_h, fg_w = fg_arr.shape[:2]
        rel_px = px - self.base_rx
        rel_py = py - self.base_ry

        # 聚光灯强度：首个词随推镜增强；后续词在特写中始终保持聚焦
        spotlight_strength = min(alpha, 1.0) if prev_word is None else 1.0
        base_darkness = 1.0 - 0.62 * spotlight_strength  # 最暗降到 0.38

        # 构建径向距离矩阵
        yy, xx = np.ogrid[:fg_h, :fg_w]
        dist = np.sqrt((xx - rel_px) ** 2 + (yy - rel_py) ** 2)

        r_core = 110
        r_outer = 380
        mask = np.full((fg_h, fg_w), base_darkness, dtype=np.float32)

        # 核心全亮区
        core_mask = dist <= r_core
        mask[core_mask] = 1.05

        # 渐变过渡区
        trans_mask = (dist > r_core) & (dist < r_outer)
        ratio = (dist[trans_mask] - r_core) / (r_outer - r_core)
        mask[trans_mask] = base_darkness + (1.05 - base_darkness) * 0.5 * (
            1.0 + np.cos(np.pi * ratio)
        )

        fg_spotlight = np.clip(fg_arr * mask[:, :, None], 0, 255).astype(np.uint8)
        fg_img = Image.fromarray(fg_spotlight)
        canvas.paste(fg_img, (self.base_rx, self.base_ry))

        d = ImageDraw.Draw(canvas)

        # 5. 顶部场景状态条 (Top Pill Bar)
        theme_color = PALETTE[(word_idx - 1) % len(PALETTE)]
        top_title = (
            f"{scene_title or '场景外语学习'} · 词汇 {word_idx:02d}/{total_words:02d}"
        )
        top_chip_w = int(d.textlength(top_title, font=self.f_title)) + 48
        tc_x0 = (W - top_chip_w) // 2
        tc_y0 = 32
        d.rounded_rectangle(
            [(tc_x0, tc_y0), (tc_x0 + top_chip_w, tc_y0 + 52)],
            radius=26,
            fill=(14, 18, 26),
            outline=(255, 255, 255),
            width=1,
        )
        d.text(
            (W // 2, tc_y0 + 26),
            top_title,
            font=self.f_title,
            fill=(255, 255, 255),
            anchor="mm",
        )

        # 6. 目标锚点指示器 (Reticle & Pulse Glow)
        pulse = (math.sin(t * 6.0) + 1.0) / 2.0  # 0~1 呼吸
        r_inner = 20
        r_pulse = int(24 + 10 * pulse)

        # 绘制扩散光环
        pulse_alpha = int(180 * (1.0 - pulse))
        glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)
        glow_draw.ellipse(
            [(px - r_pulse, py - r_pulse), (px + r_pulse, py + r_pulse)],
            outline=(*theme_color, pulse_alpha),
            width=2,
        )
        # 中心准星
        glow_draw.ellipse(
            [(px - r_inner, py - r_inner), (px + r_inner, py + r_inner)],
            outline=(*theme_color, 240),
            width=3,
        )
        glow_draw.ellipse(
            [(px - 5, py - 5), (px + 5, py + 5)],
            fill=(255, 255, 255, 255),
        )

        # 十字准星标尺刻线
        tick = 8
        glow_draw.line(
            [(px - r_inner - tick, py), (px - r_inner + 2, py)],
            fill=(*theme_color, 220),
            width=2,
        )
        glow_draw.line(
            [(px + r_inner - 2, py), (px + r_inner + tick, py)],
            fill=(*theme_color, 220),
            width=2,
        )
        glow_draw.line(
            [(px, py - r_inner - tick), (px, py - r_inner + 2)],
            fill=(*theme_color, 220),
            width=2,
        )
        glow_draw.line(
            [(px, py + r_inner - 2), (px, py + r_inner + tick)],
            fill=(*theme_color, 220),
            width=2,
        )

        canvas = Image.alpha_composite(canvas.convert("RGBA"), glow_layer).convert(
            "RGB"
        )
        d = ImageDraw.Draw(canvas)

        # 7. 悬浮教学卡片 (HUD Focus Card)
        # 智能避让：如果目标物体靠近屏幕下方，将卡片移至上方
        card_w = 1000
        card_h = 280
        card_x0 = (W - card_w) // 2
        card_x1 = card_x0 + card_w

        if py > 1300:
            card_y0 = 100
        else:
            card_y0 = H - card_h - 50
        card_y1 = card_y0 + card_h

        # 卡片底板 (深色玻璃质感)
        card_mask = Image.new("L", (card_w, card_h), 0)
        ImageDraw.Draw(card_mask).rounded_rectangle(
            [(0, 0), (card_w, card_h)], radius=22, fill=235
        )
        card_bg = Image.new("RGB", (card_w, card_h), (12, 16, 24))
        canvas.paste(card_bg, (card_x0, card_y0), card_mask)

        # 卡片边框与顶部强调条
        d.rounded_rectangle(
            [(card_x0, card_y0), (card_x1, card_y1)],
            radius=22,
            outline=(60, 72, 90),
            width=2,
        )
        d.rounded_rectangle(
            [(card_x0 + 20, card_y0), (card_x0 + 160, card_y0 + 5)],
            radius=2,
            fill=theme_color,
        )

        # 卡片内容：
        # 第 1 行：序号角标 + 英文大字 + 音标 + 中文释义
        tag_text = f"FOCUS {word_idx:02d}"
        d.rounded_rectangle(
            [(card_x0 + 28, card_y0 + 24), (card_x0 + 144, card_y0 + 60)],
            radius=6,
            fill=(*theme_color,),
        )
        d.text(
            (card_x0 + 86, card_y0 + 42),
            tag_text,
            font=self.f_badge,
            fill=(15, 20, 30),
            anchor="mm",
        )

        # 单词英文
        en_x = card_x0 + 164
        d.text(
            (en_x, card_y0 + 42),
            word.en,
            font=self.f_word_en,
            fill=(255, 255, 255),
            anchor="lm",
        )
        en_len = int(d.textlength(word.en, font=self.f_word_en))

        # 音标
        ipa_x = en_x + en_len + 18
        d.text(
            (ipa_x, card_y0 + 44),
            word.ipa,
            font=self.f_word_ipa,
            fill=(100, 200, 255),
            anchor="lm",
        )
        ipa_len = int(d.textlength(word.ipa, font=self.f_word_ipa))

        # 中文含义
        zh_x = ipa_x + ipa_len + 20
        if zh_x < card_x1 - 180:
            d.text(
                (zh_x, card_y0 + 42),
                f"· {word.zh}",
                font=self.f_word_zh,
                fill=(255, 214, 102),
                anchor="lm",
            )
        else:
            # 若一行排不下，中文在右上角
            d.text(
                (card_x1 - 32, card_y0 + 42),
                word.zh,
                font=self.f_word_zh,
                fill=(255, 214, 102),
                anchor="rm",
            )

        # 分割线
        div_y = card_y0 + 84
        d.line(
            [(card_x0 + 28, div_y), (card_x1 - 28, div_y)], fill=(38, 48, 62), width=1
        )

        # 第 2 行：英文例句 (自动换行)
        ex_en_lines = _wrap_text(word.example_en, self.f_ex_en, card_w - 60)[:2]
        line_y = div_y + 18
        for line in ex_en_lines:
            d.text(
                (card_x0 + 32, line_y),
                line,
                font=self.f_ex_en,
                fill=(240, 246, 255),
                anchor="la",
            )
            line_y += self.f_ex_en.size + 8

        # 第 3 行：中文翻译 (自动换行)
        line_y += 4
        ex_zh_lines = _wrap_text(word.example_zh, self.f_ex_zh, card_w - 60)[:2]
        for line in ex_zh_lines:
            d.text(
                (card_x0 + 32, line_y),
                line,
                font=self.f_ex_zh,
                fill=(160, 175, 195),
                anchor="la",
            )
            line_y += self.f_ex_zh.size + 6

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
    scene_title: str,
    out_mp4: Path,
    zoom_target: float = DEFAULT_ZOOM,
    prev_word: WordItem | None = None,
) -> Path:
    """将单词的连续帧流式灌入 FFmpeg 并合成为 MP4 视频片段。"""
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
        "fast",
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
            scene_title=scene_title,
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
    intro_mp4 = work_dir / "00_intro.mp4"
    total_intro_t = _intro_layers_clip(
        stills,
        intro_mp4,
        layer_durs=layer_durs,
        xfade_dur=xfade_dur,
    )
    print(f"[step4] 片头 3 层全景图过渡展示生成完毕 (总时长 {total_intro_t:.1f}s)")
    parts.append(intro_mp4)

    # 2. 逐词视觉动效引导 (Visual Cue Main Flow - 紧凑高能版，仅朗读例句)
    total_words = len(data.words)
    print(f"[step4] 开始生成 {total_words} 个词汇的 Visual Cue 视觉动效漫游片段…")

    word_durations: list[float] = []

    for idx, word in enumerate(data.words, 1):
        prev_word = data.words[idx - 2] if idx > 1 else None
        print(f"       [{idx:02d}/{total_words:02d}] 聚焦: {word.zh} ({word.en}) …")
        # 仅朗读地道英文例句 (极短前/后留白 + 地道语速)
        a_ex = tts.synth(word.example_en, voice=voice, speed=speed)

        pre_pad = np.zeros(int(DEFAULT_AUDIO_PRE_PAD * sr), dtype=np.float32)
        post_pad = np.zeros(int(DEFAULT_AUDIO_POST_PAD * sr), dtype=np.float32)

        combined_audio = np.concatenate([pre_pad, a_ex, post_pad])
        seg_dur = max(
            len(combined_audio) / sr, DEFAULT_MIN_SEG_DUR
        )  # 单个词汇片段保底最小时长
        word_durations.append(seg_dur)

        wav_path = work_dir / f"seg_{idx:02d}.wav"
        _write_wav(wav_path, combined_audio)

        seg_mp4 = work_dir / f"seg_{idx:02d}.mp4"
        _render_word_segment_video(
            animator=animator,
            word=word,
            word_idx=idx,
            total_words=total_words,
            audio_wav=wav_path,
            duration=seg_dur,
            scene_title=data.scene,
            out_mp4=seg_mp4,
            zoom_target=zoom_target,
            prev_word=prev_word,
        )
        parts.append(seg_mp4)

    # 3. 视频初步无缝拼接 (3图全景 + 逐词漫游 TTS 视频)
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

    # 4. 全局 BGM 动态智能压音混音 (片头静止图较大声，TTS朗读时自动压低轻柔旋律)
    total_video_t = total_intro_t + sum(word_durations)
    if bgm_path and Path(bgm_path).exists():
        print(
            f"[step4] 正在注入全篇智能压音 BGM ({Path(bgm_path).name}) -> {final_video}"
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
                audio_filter,
                "-map",
                "0:v",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(final_video),
            ],
            check=True,
        )
    else:
        import shutil

        shutil.copyfile(raw_concat, final_video)

    print(f"[step4] 视频合成完毕! 输出: {final_video}")
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
