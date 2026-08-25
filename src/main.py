"""流水线总入口。

uv run python -m src.main input_pics/生活场景/carriage.png
uv run python -m src.main --all                          # 整个 input_pics/
uv run python -m src.main input_pics/02_饮食与购物 --all  # 只跑某个子目录
uv run python -m src.main input_pics/.../carriage.png --step 3   # 单步调试
uv run python -m src.main input_pics/.../carriage.png --no-cache --no-video
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import config
from src.step1.cli import analyze_image
from src.step2.cli import generate_language
from src.step3.cli import render_all
from src.step4.cli import compose_video

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 音色前缀 -> 语言说明(与 step4 的 _VOICE_LANG 对应)
_VOICE_GROUPS = {
    "af": "美式女声",
    "am": "美式男声",
    "bf": "英式女声",
    "bm": "英式男声",
    "ef": "西班牙女声",
    "em": "西班牙男声",
    "ff": "法语女声",
    "fm": "法语男声",
    "hf": "印地女声",
    "hm": "印地男声",
    "if": "意大女声",
    "im": "意大男声",
    "jf": "日语女声",
    "jm": "日语男声",
    "pf": "葡语女声",
    "pm": "葡语男声",
    "zf": "中文女声",
    "zm": "中文男声",
}


def list_voices() -> None:
    """列出本地 Kokoro 可用音色,按语言分组;英文音色标 *。"""
    import re

    bins = sorted(p.stem for p in config.KOKORO_VOICES_DIR.glob("*.bin"))
    if not bins:
        raise SystemExit(f"未找到音色文件:{config.KOKORO_VOICES_DIR}")
    names: dict[str, list[str]] = {}
    for name in bins:
        prefix = re.match(r"^[a-z]{2}", name).group(0)
        names.setdefault(prefix, []).append(name)
    print(f"默认英文音色:{config.DEFAULT_VOICE}  (--voice 指定,英文音色标 *)\n")
    for prefix in sorted(names):
        label = _VOICE_GROUPS.get(prefix, prefix)
        mark = "  *" if prefix in ("af", "am", "bf", "bm") else "   "
        print(f"{mark}{label}({prefix}_): {' '.join(names[prefix])}")


def collect_images(all_pics: bool, image: str | None) -> list[Path]:
    root = Path(image) if image else config.INPUT_DIR
    if all_pics:
        if not root.exists():
            raise SystemExit(f"目录不存在: {root}")
        pics = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not pics:
            raise SystemExit(f"{root} 下没有图片")
        return pics
    if not root.is_file():
        raise SystemExit(f"图片不存在: {root}")
    return [root]


def run_pipeline(
    image_path: Path,
    step: int | None = None,
    no_cache: bool = False,
    no_cache_vlm: bool = False,
    no_cache_llm: bool = False,
    no_cache_tts: bool = False,
    no_video: bool = False,
    voice: str = config.DEFAULT_VOICE,
    speed: float = 1.2,
    zoom: float = 1.7,
) -> None:
    stem = image_path.stem
    json_path = config.JSON_DIR / f"{stem}.json"

    vlm_no_cache = no_cache or no_cache_vlm
    llm_no_cache = no_cache or no_cache_llm
    tts_no_cache = no_cache or no_cache_tts

    steps = {
        1: lambda: analyze_image(image_path, no_cache=vlm_no_cache),
        2: lambda: generate_language(json_path, no_cache=llm_no_cache),
        3: lambda: render_all(json_path),
        4: lambda: compose_video(
            json_path, voice=voice, speed=speed, zoom_target=zoom, no_cache=tts_no_cache
        ),
    }
    # 单步模式:step1 需要 image,其余依赖 json;批量入口可能跨目录重名,step1 总是先跑
    if step:
        if step != 1 and not json_path.exists():
            raise SystemExit(f"缺少 {json_path},先运行 step1")
        steps[step]()
        return

    t_pipe_start = time.perf_counter()
    json_path = steps[1]()
    steps[2]()
    steps[3]()
    if not no_video:
        steps[4]()
    t_pipe_total = time.perf_counter() - t_pipe_start
    print(f"\n全部完成: {stem} (总耗时 {t_pipe_total:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="场景外语词汇视频生成流水线")
    parser.add_argument(
        "image",
        nargs="?",
        help="单张图片路径;配合 --all 时为要批处理的子目录 (默认整个 input_pics/)",
    )
    parser.add_argument(
        "--all", action="store_true", help="批处理指定目录(或 input_pics/)下全部图片"
    )
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="只执行某一步")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="全流程强制重新生成 (忽略所有缓存: VLM/LLM/TTS 全部重跑)",
    )
    parser.add_argument(
        "--no-cache-vlm",
        action="store_true",
        help="仅重新请求 VLM 识别场景及物体坐标 (Step 1)",
    )
    parser.add_argument(
        "--no-cache-llm",
        action="store_true",
        help="仅重新请求 LLM 翻译、音标与例句 (Step 2, 保留已有坐标)",
    )
    parser.add_argument(
        "--no-cache-tts",
        action="store_true",
        help="仅强制重新合成 TTS 语音 (Step 4, 不请求任何 VLM/LLM API)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑流水线 (批处理时不跳过已有视频; 仍可复用 VLM/LLM/TTS 缓存)",
    )
    parser.add_argument("--no-video", action="store_true", help="只渲染图片,不合成视频")
    parser.add_argument("--voice", default=config.DEFAULT_VOICE, help="英文 TTS 音色")
    parser.add_argument("--speed", type=float, default=1.2, help="英文语速 (默认 1.2)")
    parser.add_argument("--zoom", type=float, default=1.7, help="推镜缩放倍率 (默认 1.7)")
    parser.add_argument(
        "--list-voices", action="store_true", help="列出可用 TTS 音色后退出"
    )
    args = parser.parse_args()
    if args.list_voices:
        list_voices()
        return
    if not args.all and not args.image:
        parser.error("需要 --image 或 --all")

    config.ensure_dirs()
    images = collect_images(args.all, args.image)
    skip_existing = (
        args.all
        and not args.force
        and not args.no_cache
        and not args.no_cache_vlm
        and not args.no_cache_llm
        and not args.no_cache_tts
        and not args.step
        and not args.no_video
    )
    failed: list[Path] = []
    done = skipped = 0
    for image_path in images:
        print(f"\n===== {image_path} =====")
        if skip_existing and (config.VIDEO_DIR / f"{image_path.stem}.mp4").exists():
            print(f"[main] 视频已存在,跳过: {image_path.stem}.mp4")
            skipped += 1
            continue
        try:
            run_pipeline(
                image_path,
                step=args.step,
                no_cache=args.no_cache,
                no_cache_vlm=args.no_cache_vlm,
                no_cache_llm=args.no_cache_llm,
                no_cache_tts=args.no_cache_tts,
                no_video=args.no_video,
                voice=args.voice,
                speed=args.speed,
                zoom=args.zoom,
            )
            done += 1
        except SystemExit as e:  # 各步骤以 SystemExit 报错;批量模式下记录后继续
            failed.append(image_path)
            print(f"[main] 处理 {image_path.name} 失败: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failed.append(image_path)
            print(f"[main] 处理 {image_path.name} 失败: {e}", file=sys.stderr)
            if not args.all:
                raise

    if args.all:
        total = len(images)
        print(f"\n===== 批量完成: 共 {total} | 成功 {done} | 跳过 {skipped} | 失败 {len(failed)} =====")
        if failed:
            print("失败列表:")
            for p in failed:
                print(f"  - {p}")
            sys.exit(1)


if __name__ == "__main__":
    main()
