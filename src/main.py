"""流水线总入口。

    uv run python -m src.main --image input_pics/生活场景/carriage.png
    uv run python -m src.main --all
    uv run python -m src.main --image ... --step 3      # 单步调试
    uv run python -m src.main --image ... --no-cache --no-video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config
from src.step1.cli import analyze_image
from src.step2.cli import generate_language
from src.step3.cli import render_all
from src.step4.cli import compose_video

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_images(all_pics: bool, image: str | None) -> list[Path]:
    if all_pics:
        pics = sorted(
            p for p in config.INPUT_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not pics:
            raise SystemExit(f"{config.INPUT_DIR} 下没有图片")
        return pics
    p = Path(image or "")
    if not p.exists():
        raise SystemExit(f"图片不存在: {p}")
    return [p]


def run_pipeline(
    image_path: Path,
    step: int | None = None,
    no_cache: bool = False,
    no_video: bool = False,
    voice: str = config.DEFAULT_VOICE,
    speed: float = 1.0,
) -> None:
    stem = image_path.stem
    json_path = config.JSON_DIR / f"{stem}.json"

    steps = {
        1: lambda: analyze_image(image_path, no_cache=no_cache),
        2: lambda: generate_language(json_path, no_cache=no_cache),
        3: lambda: render_all(json_path),
        4: lambda: compose_video(json_path, voice=voice, speed=speed),
    }
    # 单步模式:step1 需要 image,其余依赖 json;批量入口可能跨目录重名,step1 总是先跑
    if step:
        if step != 1 and not json_path.exists():
            raise SystemExit(f"缺少 {json_path},先运行 step1")
        steps[step]()
        return

    json_path = steps[1]()
    steps[2]()
    steps[3]()
    if not no_video:
        steps[4]()
    print(f"\n全部完成: {stem}")


def main() -> None:
    parser = argparse.ArgumentParser(description="场景外语词汇视频生成流水线")
    parser.add_argument("--image", help="单张图片路径")
    parser.add_argument("--all", action="store_true", help="批处理 input_pics/ 下全部图片")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="只执行某一步")
    parser.add_argument("--no-cache", action="store_true", help="强制重新请求 VLM/LLM")
    parser.add_argument("--no-video", action="store_true", help="只渲染图片,不合成视频")
    parser.add_argument("--voice", default=config.DEFAULT_VOICE, help="英文 TTS 音色")
    parser.add_argument("--speed", type=float, default=1.0, help="英文语速")
    args = parser.parse_args()
    if not args.all and not args.image:
        parser.error("需要 --image 或 --all")

    config.ensure_dirs()
    for image_path in collect_images(args.all, args.image):
        print(f"\n===== {image_path} =====")
        try:
            run_pipeline(
                image_path,
                step=args.step,
                no_cache=args.no_cache,
                no_video=args.no_video,
                voice=args.voice,
                speed=args.speed,
            )
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[main] 处理 {image_path.name} 失败: {e}", file=sys.stderr)
            if not args.all:
                raise


if __name__ == "__main__":
    main()
