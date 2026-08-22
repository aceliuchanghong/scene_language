"""Step 3 · Visual Renderer — 基于 Pillow 的 3 层标注图 + 例句表格。

输出(统一 1080x1920 竖版,竖屏短视频友好):
    output/source_language/<stem>.png   中文层
    output/target_language/<stem>.png   中英双语层
    output/pronunciation/<stem>.png     双语+音标层
    output/table/<stem>.png             词汇例句表格(单词/例句/例句翻译)
    output/table/<stem>/NN.png          表格逐行高亮帧(供视频逐句朗读)

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

    # ---------- 发音层:与中英标注统一的深色玻璃卡 ----------

    def render_pronunciation(self, data: SceneData) -> Path:
        canvas, rect = self.canvas("发音音标", data.scene)
        lines_per_word = [
            [
                (w.en, self.f_label_en),
                (w.zh, self.f_label_zh),
                (w.ipa, self.f_label_ipa),
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

        out = config.PRON_DIR / f"{Path(data.image).stem}.png"
        config.PRON_DIR.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
        print(f"[step3] 发音音标 -> {out}")
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
        # 文本:首行白色(英文/中文),其后浅灰
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
        layers = [
            ("source_language", "中文场景", [lambda w: (w.zh, self.f_label_zh)]),
            (
                "target_language",
                "中英对照",
                [lambda w: (w.en, self.f_label_en), lambda w: (w.zh, self.f_label_zh)],
            ),
        ]
        rect = None
        outputs = []
        for dirname, stage, getters in layers:
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
            out_dir = getattr(
                config,
                {
                    "source_language": "SOURCE_LANG_DIR",
                    "target_language": "TARGET_LANG_DIR",
                    "pronunciation": "PRON_DIR",
                }[dirname],
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{Path(data.image).stem}.png"
            canvas.save(out)
            outputs.append(out)
            print(f"[step3] {stage} -> {out}")
        return outputs

    # ---------- 词汇例句表格 ----------

    def _wrap(self, text: str, font, max_w: int) -> list[str]:
        """贪心折行:拉丁按词断,中日韩按字断。"""
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
                # 拉丁词间优先在空格断行
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

    def _render_table(self, data: SceneData, highlight: int | None, path: Path) -> Path:
        n = len(data.words)
        left, right = MARGIN, W - MARGIN
        table_top, table_bottom = 252, H - 80
        header_h = 64
        row_h = (table_bottom - table_top - header_h) // n
        # 三列:单词 / 例句 / 例句翻译
        col_x = [left, left + 268, left + 268 + 396]
        col_w = [268, 396, right - col_x[2]]

        f_en = _font(config.FONT_EN_BOLD, 30)
        f_zh = _font(config.FONT_ZH, 26)
        f_ipa = _font(config.FONT_EN, 21)
        f_ex = _font(config.FONT_EN, 25)
        f_ex_zh = _font(config.FONT_ZH, 25)
        f_head = _font(config.FONT_ZH_BOLD, 30)

        bg = self.src.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(40))
        canvas = Image.eval(bg, lambda p: int(p * 0.22)).convert("RGB")
        overlay = Image.new("RGBA", (W, H), (8, 12, 18, 190))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(canvas)

        d.text(
            (MARGIN, 50),
            "SCENE VOCABULARY",
            font=self.f_badge,
            fill=(236, 184, 72),
            anchor="la",
        )
        d.text(
            (MARGIN, 96),
            "场景词汇 · 例句",
            font=self.f_title,
            fill=(255, 255, 255),
            anchor="la",
        )
        d.text(
            (W - MARGIN, 110),
            f"{n:02d} WORDS",
            font=self.f_badge,
            fill=(178, 187, 200),
            anchor="ra",
        )

        # 表头
        d.rectangle(
            [(left, table_top), (right, table_top + header_h)], fill=(34, 40, 52)
        )
        for title, cx, cw in zip(("单词", "例句", "例句翻译"), col_x, col_w):
            d.text(
                (cx + 16, table_top + header_h // 2),
                title,
                font=f_head,
                fill=(255, 255, 255),
                anchor="lm",
            )

        for i, w in enumerate(data.words):
            y0 = table_top + header_h + i * row_h
            y1 = y0 + row_h - 1
            if highlight is not None and i != highlight:
                # 高亮帧:非当前行压暗
                d.rectangle([(left, y0), (right, y1)], fill=(10, 13, 19))
            elif highlight == i:
                d.rectangle([(left, y0), (right, y1)], fill=(255, 216, 106))
                d.rectangle([(left, y0), (left + 6, y1)], fill=(200, 90, 20))
            d.line(
                [(left, y1), (right, y1)],
                fill=(255, 255, 255) if highlight is None else (70, 76, 88),
                width=1,
            )

            dark_text = highlight == i
            c_word = (17, 20, 28)
            c_sub = (74, 62, 24) if dark_text else (72, 79, 90)
            c_ipa = (60, 52, 20) if dark_text else (116, 123, 132)
            c_ex = (35, 30, 10) if dark_text else (232, 238, 246)
            if highlight is not None and i != highlight:
                c_word = c_sub = c_ipa = c_ex = (128, 135, 148)

            # 列1:单词(en/zh/ipa 纵排居中)
            wx = col_x[0] + 16
            block_h = f_en.size + f_zh.size + f_ipa.size + 12
            wy = y0 + (row_h - block_h) // 2
            d.text((wx, wy), w.en, font=f_en, fill=c_word, anchor="la")
            d.text((wx, wy + f_en.size + 3), w.zh, font=f_zh, fill=c_sub, anchor="la")
            d.text(
                (wx, wy + f_en.size + f_zh.size + 8),
                w.ipa,
                font=f_ipa,
                fill=c_ipa,
                anchor="la",
            )

            # 列2:英文例句折行居中
            lines = self._wrap(w.example_en, f_ex, col_w[1] - 32)[:4]
            lh = f_ex.size + 6
            ly = y0 + (row_h - lh * len(lines)) // 2
            for ln in lines:
                d.text((col_x[1] + 16, ly), ln, font=f_ex, fill=c_ex, anchor="lm")
                ly += lh

            # 列3:例句翻译折行居中
            lines = self._wrap(w.example_zh, f_ex_zh, col_w[2] - 32)[:4]
            ly = y0 + (row_h - lh * len(lines)) // 2
            for ln in lines:
                d.text((col_x[2] + 16, ly), ln, font=f_ex_zh, fill=c_ex, anchor="lm")
                ly += lh

        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        return path

    def render_table(self, data: SceneData) -> tuple[Path, list[Path]]:
        stem = Path(data.image).stem
        base = self._render_table(data, None, config.TABLE_DIR / f"{stem}.png")
        print(f"[step3] 词汇例句表格 -> {base}")
        out_dir = config.TABLE_DIR / stem
        frames = [
            self._render_table(data, i, out_dir / f"{i + 1:02d}.png")
            for i in range(len(data.words))
        ]
        print(f"[step3] 表格高亮帧 x{len(frames)} -> {out_dir}/")
        return base, frames


def render_all(json_path: Path) -> dict:
    data: SceneData = load_scene_data(json_path)
    if not all(w.en and w.ipa for w in data.words):
        raise SystemExit("[step3] JSON 缺少 en/ipa,先运行 step2")
    if not all(w.example_en and w.example_zh for w in data.words):
        raise SystemExit("[step3] JSON 缺少例句,先运行 step2(可加 --no-cache)")
    r = Renderer(Path(data.image))
    layers = r.render_layers(data)
    pronunciation = r.render_pronunciation(data)
    table, table_frames = r.render_table(data)
    return {
        "layers": layers,
        "pronunciation": pronunciation,
        "table": table,
        "table_frames": table_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3 · 图片渲染")
    parser.add_argument(
        "--image", required=True, help="输入图片(定位 output/json 下同名 JSON)"
    )
    parser.add_argument("--json", default=None, help="直接指定 JSON 路径")
    args = parser.parse_args()
    config.ensure_dirs()
    json_path = (
        Path(args.json)
        if args.json
        else config.JSON_DIR / f"{Path(args.image).stem}.json"
    )
    if not json_path.exists():
        raise SystemExit(f"找不到 {json_path},先运行 step1/step2")
    render_all(json_path)


if __name__ == "__main__":
    main()
