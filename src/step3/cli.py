"""Step 3 · Visual Renderer — 基于 Pillow 的 4 层标注图 + 聚光高亮帧。

输出(统一 1080x1920 竖版,竖屏短视频友好):
    output/source_language/<stem>.png   中文层
    output/target_language/<stem>.png   中英双语层
    output/pronunciation/<stem>.png     双语+音标层
    output/table/<stem>.png             词汇总览表格
    output/frames/<stem>/NN.png         每个词的聚光高亮帧

单独运行:
    uv run python -m src.step3.cli --image input_pics/生活场景/carriage.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src import config
from src.models import SceneData, load_scene_data

W, H = 1080, 1920
MARGIN = 40
TITLE_H = 170
FOOTER_H = 150
IMG_TOP = TITLE_H
IMG_BOTTOM = H - FOOTER_H

PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (231, 111, 81),
    (94, 96, 206),
    (188, 71, 73),
    (27, 124, 129),
    (119, 47, 26),
    (76, 149, 106),
    (150, 100, 20),
    (73, 79, 130),
    (170, 62, 94),
]


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


class Renderer:
    def __init__(self, image_path: Path):
        self.src = Image.open(image_path).convert("RGB")
        self.f_zh = _font(config.FONT_ZH, 34)
        self.f_zh_big = _font(config.FONT_ZH, 58)
        self.f_en = _font(config.FONT_EN, 36)
        self.f_en_big = _font(config.FONT_EN, 72)
        self.f_ipa = _font(config.FONT_EN, 30)
        self.f_ipa_big = _font(config.FONT_EN, 40)
        self.f_title = _font(config.FONT_ZH, 44)
        self.f_small = _font(config.FONT_ZH, 26)
        self.f_badge = _font(config.FONT_EN, 26)
        # 标注小卡字体:英文大、中文中、音标小灰
        self.f_label_en = _font(config.FONT_EN_BOLD, 34)
        self.f_label_zh = _font(config.FONT_ZH_BOLD, 28)
        self.f_label_ipa = _font(config.FONT_EN, 23)
        self.f_index = _font(config.FONT_EN_BOLD, 18)

    # ---------- 基础画布:原图尽量占满,仅留极窄边距,无装饰 ----------

    def canvas(self, stage_title: str = "", scene: str = "", extra: str = "") -> tuple[Image.Image, tuple]:
        """竖版画布:模糊放大的原图做背景,前景图几乎占满(边距 12px)。"""
        bg = self.src.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(24))
        bg = Image.eval(bg, lambda p: int(p * 0.55))
        canvas = bg.convert("RGB")

        margin = 12
        max_w, max_h = W - 2 * margin, H - 2 * margin
        ratio = min(max_w / self.src.width, max_h / self.src.height)
        fw, fh = int(self.src.width * ratio), int(self.src.height * ratio)
        fg = self.src.resize((fw, fh), Image.LANCZOS)
        rx = (W - fw) // 2
        ry = (H - fh) // 2
        canvas.paste(fg, (rx, ry))
        rect = (rx, ry, rx + fw, ry + fh)

        if stage_title:
            d = ImageDraw.Draw(canvas)
            chip_w = int(d.textlength(stage_title, font=self.f_small)) + 44
            cx0, cy0 = (W - chip_w) // 2, 16
            d.rounded_rectangle(
                [(cx0, cy0), (cx0 + chip_w, cy0 + 46)], radius=23,
                fill=(16, 20, 28), outline=(255, 255, 255), width=1,
            )
            d.text((W // 2, cy0 + 23), stage_title, font=self.f_small, fill=(255, 255, 255), anchor="mm")
        return canvas, rect

    # ---------- 发音层:与中英标注统一的深色玻璃卡 ----------

    def render_pronunciation(self, data: SceneData) -> Path:
        canvas, rect = self.canvas("发音音标", data.scene)
        lines_per_word = [
            [(w.en, self.f_label_en), (w.zh, self.f_label_zh), (w.ipa, self.f_label_ipa)]
            for w in data.words
        ]
        sizes = [self._card_size(lines) for lines in lines_per_word]
        anchors = [
            (rect[0] + w.x * (rect[2] - rect[0]), rect[1] + w.y * (rect[3] - rect[1]))
            for w in data.words
        ]
        boxes = self._layout(anchors, sizes, rect)
        for i, (box, anchor, lines) in enumerate(zip(boxes, anchors, lines_per_word)):
            self._draw_card(canvas, box, anchor, i + 1, lines, PALETTE[i % len(PALETTE)])

        out = config.PRON_DIR / f"{Path(data.image).stem}.png"
        config.PRON_DIR.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
        print(f"[step3] 发音音标 -> {out}")
        return out

    # ---------- 标签排版(抗重叠) ----------

    def _card_size(self, lines: list[tuple[str, ImageFont.FreeTypeFont]]) -> tuple[int, int]:
        d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        pad, gap = 13, 2
        w = max((d.textlength(t, font=f) for t, f in lines), default=0)
        h = sum(f.size + gap for _, f in lines)
        return int(w) + pad * 2, h + pad * 2

    def _layout(self, anchors: list[tuple[float, float]], sizes: list[tuple[int, int]], rect):
        """为每个标签在锚点附近选一个不与已放置标签重叠、且不出画面区域的位置。"""
        rx0, ry0, rx1, ry1 = rect
        pad = 6
        placed: list[tuple[int, int, int, int]] = []

        def overlap(a, b) -> int:
            ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
            return ix * iy

        result = []
        step = 30
        for (ax, ay), (w, h) in zip(anchors, sizes):
            candidates = []
            for gap in (10, 30, 70, 130, 200, 280):
                candidates += [
                    (int(ax) + gap, int(ay) - h - gap),  # 右上
                    (int(ax) - w - gap, int(ay) - h - gap),  # 左上
                    (int(ax) + gap, int(ay) + gap),  # 右下
                    (int(ax) - w - gap, int(ay) + gap),  # 左下
                    (int(ax) - w // 2, int(ay) - h - gap - 18),  # 正上
                    (int(ax) - w // 2, int(ay) + gap + 18),  # 正下
                    (int(ax) + gap, int(ay) - h // 2),  # 右侧
                    (int(ax) - w - gap, int(ay) - h // 2),  # 左侧
                ]
            # 锚点附近找不到空位时,全区域网格扫描兜底
            for gy in range(ry0 + pad, ry1 - h - pad + 1, step):
                for gx in range(rx0 + pad, rx1 - w - pad + 1, step):
                    candidates.append((gx, gy))
            best, best_cost = None, None
            n_radial = 6 * 8  # 径向候选固定在前 48 个
            for idx, (cx, cy) in enumerate(candidates):
                cx = min(max(cx, rx0 + pad), rx1 - w - pad)
                cy = min(max(cy, ry0 + pad), ry1 - h - pad)
                box = (cx, cy, w, h)
                ov = sum(overlap(box, p) for p in placed)
                # 主目标零重叠,次目标离锚点近
                dist = (cx + w // 2 - ax) ** 2 + (cy + h // 2 - ay) ** 2
                cost = (ov, dist)
                if best is None or cost < best_cost:
                    best, best_cost = box, cost
                if ov == 0 and idx < n_radial:
                    break
            placed.append(best)
            result.append(best)
        return result

    def _draw_connector(self, canvas, box, anchor, color) -> None:
        """用克制的引线保留标签与实物的空间关系。"""
        x, y, w, h = box
        ax, ay = anchor
        px = min(max(ax, x), x + w)
        py = min(max(ay, y), y + h)
        d = ImageDraw.Draw(canvas)
        d.line([(ax, ay), (px, py)], fill=(245, 245, 242), width=3)
        d.ellipse([(ax - 7, ay - 7), (ax + 7, ay + 7)], fill=color, outline=(255, 255, 255), width=2)

    def _draw_card(
        self,
        canvas: Image.Image,
        box: tuple[int, int, int, int],
        anchor: tuple[float, float],
        idx: int,
        lines: list[tuple[str, ImageFont.FreeTypeFont]],
        color: tuple,
    ) -> None:
        x, y, w, h = box
        self._draw_connector(canvas, box, anchor, color)
        # 深色玻璃小卡，信息紧凑但仍保持足够对比度。
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([(0, 0), (w, h)], radius=12, fill=220)
        canvas.paste(Image.new("RGB", (w, h), (16, 19, 24)), (x, y), mask)
        d = ImageDraw.Draw(canvas)
        # 文本:首行白色(英文/中文),其后浅灰
        d.rounded_rectangle([(x, y), (x + 5, y + h)], radius=2, fill=color)
        ty = y + 12
        for i, (text, font) in enumerate(lines):
            if not text:
                continue
            fill = (255, 255, 255) if i == 0 else (190, 198, 208)
            d.text((x + 17, ty + font.size // 2), text, font=font, fill=fill, anchor="lm")
            ty += font.size + 2

    # ---------- 三层标注图 ----------

    def render_layers(self, data: SceneData) -> list[Path]:
        layers = [
            ("source_language", "中文场景", [lambda w: (w.zh, self.f_label_zh)]),
            ("target_language", "中英对照", [lambda w: (w.en, self.f_label_en), lambda w: (w.zh, self.f_label_zh)]),
        ]
        rect = None
        outputs = []
        for dirname, stage, getters in layers:
            canvas, rect = self.canvas(stage, data.scene, extra=f"共 {len(data.words)} 个词汇")
            lines_per_word = [[g(w) for g in getters] for w in data.words]
            sizes = [self._card_size(ls) for ls in lines_per_word]
            anchors = [
                (rect[0] + w.x * (rect[2] - rect[0]), rect[1] + w.y * (rect[3] - rect[1]))
                for w in data.words
            ]
            boxes = self._layout(anchors, sizes, rect)
            for i, (w, box, anchor, lines) in enumerate(zip(data.words, boxes, anchors, lines_per_word), 1):
                self._draw_card(canvas, box, anchor, i, lines, PALETTE[(i - 1) % len(PALETTE)])
            out_dir = getattr(config, {"source_language": "SOURCE_LANG_DIR", "target_language": "TARGET_LANG_DIR", "pronunciation": "PRON_DIR"}[dirname])
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{Path(data.image).stem}.png"
            canvas.save(out)
            outputs.append(out)
            print(f"[step3] {stage} -> {out}")
        return outputs

    # ---------- 聚光高亮帧 ----------

    def render_spotlight_frames(self, data: SceneData) -> list[Path]:
        out_dir = config.FRAMES_DIR / Path(data.image).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        n = len(data.words)
        for i, w in enumerate(data.words, 1):
            canvas, rect = self.canvas(f"聚焦学习 {i} / {n}", data.scene)
            px = rect[0] + w.x * (rect[2] - rect[0])
            py = rect[1] + w.y * (rect[3] - rect[1])
            r = int(min(rect[2] - rect[0], rect[3] - rect[1]) * 0.22)

            # 暗幕 + 径向透光
            fg_zone = canvas.crop(rect)
            dark = Image.eval(fg_zone, lambda p: int(p * 0.32))
            mask = Image.new("L", fg_zone.size, 255)
            md = ImageDraw.Draw(mask)
            md.ellipse([(px - rect[0] - r, py - rect[1] - r), (px - rect[0] + r, py - rect[1] + r)], fill=0)
            mask = mask.filter(ImageFilter.GaussianBlur(r // 5))
            fg_zone.paste(dark, (0, 0), mask)
            canvas.paste(fg_zone, (rect[0], rect[1]))

            d = ImageDraw.Draw(canvas)
            color = PALETTE[(i - 1) % len(PALETTE)]
            d.ellipse([(px - r, py - r), (px + r, py + r)], outline=(255, 216, 106), width=6)
            d.ellipse([(px - r + 12, py - r + 12), (px + r - 12, py + r - 12)], outline=color, width=2)

            # 底部居中大词卡(白底细黑描边,与标注卡同风格)
            card_w, card_h = W - 2 * MARGIN - 80, 190
            cx, cy = MARGIN + 40, H - card_h - 50
            d.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=14, fill=(255, 255, 255), outline=(15, 18, 26), width=2)
            d.text((cx + card_w // 2, cy + 48), w.en, font=self.f_en_big, fill=(17, 20, 28), anchor="mm")
            d.text((cx + card_w // 2, cy + 108), w.zh, font=self.f_zh, fill=(60, 66, 78), anchor="mm")
            d.text((cx + card_w // 2, cy + 155), w.ipa, font=self.f_ipa, fill=(105, 112, 125), anchor="mm")
            out = out_dir / f"{i:02d}.png"
            canvas.save(out)
            outputs.append(out)
        print(f"[step3] 聚光帧 x{n} -> {out_dir}/")
        return outputs

    # ---------- 词汇总览表格 ----------

    def render_table(self, data: SceneData) -> Path:
        n = len(data.words)
        cols = 1 if n <= 6 else 2
        rows = (n + cols - 1) // cols
        top, bottom = 250, H - 92
        gap_x = 22
        gap_y = 18 if rows <= 6 else 10
        cell_w = (W - 2 * MARGIN - gap_x) // cols
        cell_h = (bottom - top - gap_y * (rows - 1)) // rows
        thumb = max(72, min(cell_h - 24, 150 if cols == 2 else 190))
        if cell_h >= 220:
            card_en, card_zh, card_ipa = self.f_en, self.f_zh, self.f_ipa
        elif cell_h >= 165:
            card_en = _font(config.FONT_EN, 32)
            card_zh = _font(config.FONT_ZH, 29)
            card_ipa = _font(config.FONT_EN, 24)
        else:
            card_en = _font(config.FONT_EN, 27)
            card_zh = _font(config.FONT_ZH, 24)
            card_ipa = _font(config.FONT_EN, 20)

        bg = self.src.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(40))
        canvas = Image.eval(bg, lambda p: int(p * 0.22)).convert("RGB")
        overlay = Image.new("RGBA", (W, H), (8, 12, 18, 185))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(canvas)
        d.text((MARGIN, 58), "SCENE VOCABULARY", font=self.f_badge, fill=(236, 184, 72), anchor="la")
        d.text((MARGIN, 108), "场景词汇", font=self.f_title, fill=(255, 255, 255), anchor="la")
        scene = data.scene or "生活场景"
        d.text((MARGIN, 176), scene, font=self.f_small, fill=(178, 187, 200), anchor="la")
        count_text = f"{n:02d} WORDS"
        d.text((W - MARGIN, 120), count_text, font=self.f_badge, fill=(178, 187, 200), anchor="ra")
        d.line([(MARGIN, 216), (W - MARGIN, 216)], fill=(255, 255, 255), width=1)

        sw, sh = self.src.size
        for i, w in enumerate(data.words):
            r, c = divmod(i, cols)
            x = MARGIN + c * (cell_w + gap_x)
            y = top + r * (cell_h + gap_y)
            color = PALETTE[i % len(PALETTE)]
            d.rounded_rectangle([(x, y), (x + cell_w, y + cell_h)], radius=18, fill=(247, 247, 244))
            d.rounded_rectangle([(x, y), (x + 6, y + cell_h)], radius=3, fill=color)
            # 缩略图:以词坐标为中心裁剪正方形
            cx_px, cy_px = int(w.x * sw), int(w.y * sh)
            side = max(int(min(sw, sh) * 0.30), 60)
            x0, y0 = max(0, min(sw - side, cx_px - side // 2)), max(0, min(sh - side, cy_px - side // 2))
            crop = self.src.crop((x0, y0, x0 + side, y0 + side)).resize((thumb, thumb), Image.LANCZOS)
            iy = y + (cell_h - thumb) // 2
            canvas.paste(crop, (x + 18, iy))
            tx = x + 18 + thumb + 18
            content_h = card_en.size + card_zh.size + card_ipa.size + 10
            content_y = y + (cell_h - content_h) // 2
            d.text((tx, y + 16), f"{i + 1:02d}", font=self.f_index, fill=color, anchor="la")
            d.text((tx, content_y), w.en, font=card_en, fill=(19, 24, 32), anchor="la")
            d.text((tx, content_y + card_en.size + 4), w.zh, font=card_zh, fill=(65, 72, 82), anchor="la")
            d.text((tx, content_y + card_en.size + card_zh.size + 10), w.ipa, font=card_ipa, fill=(116, 123, 132), anchor="la")

        d.text((W // 2, H - 45), "LOOK · LISTEN · REMEMBER", font=self.f_badge, fill=(150, 158, 170), anchor="mm")
        out = config.TABLE_DIR / f"{Path(data.image).stem}.png"
        config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
        print(f"[step3] 词汇表格 -> {out}")
        return out


def render_all(json_path: Path) -> dict[str, list[Path] | Path]:
    data: SceneData = load_scene_data(json_path)
    if not all(w.en and w.ipa for w in data.words):
        raise SystemExit("[step3] JSON 缺少 en/ipa,先运行 step2")
    r = Renderer(Path(data.image))
    layers = r.render_layers(data)
    pronunciation = r.render_pronunciation(data)
    frames = r.render_spotlight_frames(data)
    table = r.render_table(data)
    return {"layers": layers, "pronunciation": pronunciation, "frames": frames, "table": table}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3 · 图片渲染")
    parser.add_argument("--image", required=True, help="输入图片(定位 output/json 下同名 JSON)")
    parser.add_argument("--json", default=None, help="直接指定 JSON 路径")
    args = parser.parse_args()
    config.ensure_dirs()
    json_path = Path(args.json) if args.json else config.JSON_DIR / f"{Path(args.image).stem}.json"
    if not json_path.exists():
        raise SystemExit(f"找不到 {json_path},先运行 step1/step2")
    render_all(json_path)


if __name__ == "__main__":
    main()
