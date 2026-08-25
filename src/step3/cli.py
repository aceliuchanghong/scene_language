"""Step 3 · Visual Renderer — 基于 Pillow 的 3 层标注图。

输出(统一 1080x1920 竖版,竖屏短视频友好):
    output/source_language/<stem>.png           中文层
    output/target_language/<stem>[_<lang>].png   双语对照层
    output/pronunciation/<stem>[_<lang>].png     双语+注音层

单独运行:
    uv run python -m src.step3.cli --image input_pics/生活场景/carriage.png
    uv run python -m src.step3.cli --image ... --lang ja
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src import config
from src.models import SceneData, load_scene_data

W, H = 1080, 1920

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
    def __init__(self, image_path: Path, lang: str = "en"):
        self.lang = lang.lower()
        self.src = Image.open(image_path).convert("RGB")
        fonts = config.get_lang_fonts(self.lang)

        self.f_zh = _font(fonts["zh"], 34)
        self.f_zh_big = _font(fonts["zh"], 58)
        self.f_target = _font(fonts["regular"], 36)
        self.f_target_big = _font(fonts["regular"], 72)
        self.f_pron = _font(fonts["regular"], 30)
        self.f_pron_big = _font(fonts["regular"], 40)
        self.f_title = _font(fonts["zh"], 44)
        self.f_small = _font(fonts["zh"], 26)
        self.f_badge = _font(fonts["bold"], 26)
        # 标注小卡字体:外文大、中文中、音标/注音小灰
        self.f_label_target = _font(fonts["bold"], 34)
        self.f_label_zh = _font(fonts["zh_bold"], 28)
        self.f_label_pron = _font(fonts["regular"], 23)
        self.f_index = _font(fonts["bold"], 18)

    # ---------- 基础画布:原图尽量占满,仅留极窄边距,无装饰 ----------

    def canvas(
        self, stage_title: str = "", scene: str = "", extra: str = ""
    ) -> tuple[Image.Image, tuple]:
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
                [(cx0, cy0), (cx0 + chip_w, cy0 + 46)],
                radius=23,
                fill=(16, 20, 28),
                outline=(255, 255, 255),
                width=1,
            )
            d.text(
                (W // 2, cy0 + 23),
                stage_title,
                font=self.f_small,
                fill=(255, 255, 255),
                anchor="mm",
            )
        return canvas, rect

    # ---------- 发音层:与双语标注统一的深色玻璃卡 ----------

    def render_pronunciation(self, data: SceneData) -> Path:
        title = "发音音标" if self.lang == "en" else "发音注音"
        canvas, rect = self.canvas(title, data.scene)
        lines_per_word = [
            [
                (w.target or w.en, self.f_label_target),
                (w.zh, self.f_label_zh),
                (w.pron or w.ipa, self.f_label_pron),
            ]
            for w in data.words
        ]
        sizes = [self._card_size(lines) for lines in lines_per_word]
        anchors = [
            (rect[0] + w.x * (rect[2] - rect[0]), rect[1] + w.y * (rect[3] - rect[1]))
            for w in data.words
        ]
        boxes = self._layout(anchors, sizes, rect)
        for i, (box, anchor, lines) in enumerate(zip(boxes, anchors, lines_per_word)):
            self._draw_card(
                canvas, box, anchor, i + 1, lines, PALETTE[i % len(PALETTE)]
            )

        stem = Path(data.image).stem
        layer_paths = config.get_layer_paths(stem, self.lang)
        out = layer_paths["pronunciation"]
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
        print(f"[step3] [{self.lang}] {title} -> {out}")
        return out

    # ---------- 标签排版(抗重叠) ----------

    def _card_size(
        self, lines: list[tuple[str, ImageFont.FreeTypeFont]]
    ) -> tuple[int, int]:
        d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        pad, gap = 13, 2
        w = max((d.textlength(t, font=f) for t, f in lines), default=0)
        h = sum(f.size + gap for _, f in lines)
        return int(w) + pad * 2, h + pad * 2

    def _layout(
        self, anchors: list[tuple[float, float]], sizes: list[tuple[int, int]], rect
    ):
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
        d.ellipse(
            [(ax - 7, ay - 7), (ax + 7, ay + 7)],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )

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
        # 文本:首行白色(外文/中文),其后浅灰
        d.rounded_rectangle([(x, y), (x + 5, y + h)], radius=2, fill=color)
        ty = y + 12
        for i, (text, font) in enumerate(lines):
            if not text:
                continue
            fill = (255, 255, 255) if i == 0 else (190, 198, 208)
            d.text(
                (x + 17, ty + font.size // 2), text, font=font, fill=fill, anchor="lm"
            )
            ty += font.size + 2

    # ---------- 三层标注图 ----------

    def render_layers(self, data: SceneData) -> list[Path]:
        stem = Path(data.image).stem
        layer_paths = config.get_layer_paths(stem, self.lang)
        bilingual_title = {
            "en": "中英对照",
            "ja": "中日对照",
            "ko": "中韩对照",
        }.get(self.lang, "双语对照")

        layers = [
            (
                "source",
                "中文场景",
                [lambda w: (w.zh, self.f_label_zh)],
                layer_paths["source"],
            ),
            (
                "target",
                bilingual_title,
                [
                    lambda w: (w.target or w.en, self.f_label_target),
                    lambda w: (w.zh, self.f_label_zh),
                ],
                layer_paths["target"],
            ),
        ]
        rect = None
        outputs = []
        for key, stage, getters, out_file in layers:
            canvas, rect = self.canvas(
                stage, data.scene, extra=f"共 {len(data.words)} 个词汇"
            )
            lines_per_word = [[g(w) for g in getters] for w in data.words]
            sizes = [self._card_size(ls) for ls in lines_per_word]
            anchors = [
                (
                    rect[0] + w.x * (rect[2] - rect[0]),
                    rect[1] + w.y * (rect[3] - rect[1]),
                )
                for w in data.words
            ]
            boxes = self._layout(anchors, sizes, rect)
            for i, (w, box, anchor, lines) in enumerate(
                zip(data.words, boxes, anchors, lines_per_word), 1
            ):
                self._draw_card(
                    canvas, box, anchor, i, lines, PALETTE[(i - 1) % len(PALETTE)]
                )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out_file)
            outputs.append(out_file)
            print(f"[step3] [{self.lang}] {stage} -> {out_file}")
        return outputs


def render_all(json_path: Path, lang: str = "en") -> dict:
    data: SceneData = load_scene_data(json_path)
    lang = (data.lang or lang).lower()

    if not all((w.target or w.en) and (w.pron or w.ipa) for w in data.words):
        raise SystemExit(f"[step3] JSON 缺少外语词汇或注音,先运行 step2 --lang {lang}")
    if not all((w.example_target or w.example_en) and w.example_zh for w in data.words):
        raise SystemExit(
            f"[step3] JSON 缺少例句,先运行 step2 --lang {lang}(可加 --no-cache)"
        )
    r = Renderer(Path(data.image), lang=lang)
    layers = r.render_layers(data)
    pronunciation = r.render_pronunciation(data)
    return {
        "layers": layers,
        "pronunciation": pronunciation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3 · 图片渲染")
    parser.add_argument(
        "--image", required=True, help="输入图片(定位 output/json 下对应 JSON)"
    )
    parser.add_argument(
        "--lang",
        default=config.DEFAULT_LANG,
        choices=config.SUPPORTED_LANGUAGES,
        help="目标语言 (默认 en)",
    )
    parser.add_argument("--json", default=None, help="直接指定 JSON 路径")
    args = parser.parse_args()
    config.ensure_dirs()
    json_path = (
        Path(args.json)
        if args.json
        else config.get_json_path(Path(args.image).stem, args.lang)
    )
    if not json_path.exists():
        raise SystemExit(f"找不到 {json_path},先运行 step1/step2 --lang {args.lang}")
    render_all(json_path, lang=args.lang)


if __name__ == "__main__":
    main()
