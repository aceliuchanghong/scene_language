"""Step 1 · Scene Analyzer — 调用 VLM 识别场景学习点及归一化坐标。

若 scene_catalog/<分类>/<图片同名>.json 存在,则使用其中预定义的
targets 词表让 VLM 只做定位(坐标),不再自行选词。

单独运行:
    uv run python -m src.step1.cli --image input_pics/生活场景/carriage.png
    uv run python -m src.step1.cli --image ... --no-cache
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path

from openai import OpenAI

from src import config
from src.models import SceneData, load_scene_data, parse_scene_data

PROMPT_FREE = """你是场景外语学习助手。仔细观察这张真实场景照片。

任务:
1. 用一句中文概括场景(字段 scene)。
2. 挑出 8~12 个最值得学习的高价值物品/动作/概念,给出中文名称与该物体在图片中的位置。
3. 位置用归一化坐标 (x, y),左上角为 (0,0),右下角为 (1,1),对准物体可见中心。
4. 坐标必须在图片内均匀分布,覆盖画面各个区域,不要都挤在中间或一侧。
5. 中文词优先用日常口语词(如 婴儿车 而不是 载人工具)。

只输出 JSON,格式:
{
  "scene": "一句话场景概括",
  "words": [
    {"zh": "婴儿车", "x": 0.31, "y": 0.42}
  ]
}"""

PROMPT_WITH_TARGETS = """你是场景外语学习助手。仔细观察这张真实场景照片。

这张图是按下面的词表生成的,图中应包含每一个词对应的物体。

任务:
1. 用一句中文概括场景(字段 scene)。
2. 在图中逐一定位下列每个词对应的物体,zh 字段必须原样使用词表里的词,不得增删或改写。
3. 位置用归一化坐标 (x, y),左上角为 (0,0),右下角为 (1,1),对准物体可见中心。
4. 如果某个词的物体确实不在图中,坐标给 (0.5, 0.5) 并照常输出,不要遗漏。

词表:
{targets}

只输出 JSON,格式:
{{
  "scene": "一句话场景概括",
  "words": [
    {{"zh": "词表中的词", "x": 0.31, "y": 0.42}}
  ]
}}"""


def load_catalog_targets(image_path: Path) -> list[str]:
    """按图片文件名匹配 scene_catalog/<分类>/<stem>.json,返回预定义中文词表。"""
    catalog_dir = config.ROOT / "scene_catalog"
    matches = sorted(catalog_dir.rglob(f"{image_path.stem}.json"))
    if not matches:
        return []
    try:
        data = json.loads(matches[0].read_text(encoding="utf-8"))
        return [str(t["zh"]) for t in data.get("targets", []) if t.get("zh")]
    except Exception as e:  # noqa: BLE001
        print(f"[step1] 读取词表失败({matches[0]}): {e}", file=sys.stderr)
        return []


def analyze_image(image_path: Path, no_cache: bool = False) -> Path:
    """返回场景 JSON 路径(output/json/<stem>.json)。"""
    config.ensure_dirs()
    out_path = config.JSON_DIR / f"{image_path.stem}.json"
    if out_path.exists() and not no_cache:
        print(f"[step1] 使用缓存 {out_path}")
        return out_path

    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    client = OpenAI(base_url=config.VLM_BASE_URL, api_key=config.VLM_API_KEY)

    targets = load_catalog_targets(image_path)
    if targets:
        prompt = PROMPT_WITH_TARGETS.format(
            targets="\n".join(f"- {t}" for t in targets)
        )
        print(f"[step1] 使用预定义词表 {len(targets)} 个词: {'、'.join(targets)}")
    else:
        prompt = PROMPT_FREE

    last_err: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=config.VLM_MODEL,
                max_tokens=config.MAX_TOKENS,
                timeout=config.TIMEOUT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    }
                ],
            )
            raw = resp.choices[0].message.content or ""
            data: SceneData = parse_scene_data(raw, image_path)
            import re

            m = re.search(r'"scene"\s*:\s*"([^"]+)"', raw)
            data.scene = m.group(1) if m else ""
            if targets:
                got = {w.zh for w in data.words}
                missing = [t for t in targets if t not in got]
                extra = sorted(got - set(targets))
                if missing or extra:
                    raise ValueError(
                        f"词表校验失败 缺少: {missing} 多余: {extra}"
                    )
                data.words.sort(key=lambda w: targets.index(w.zh))
            out_path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
            print(f"[step1] 识别到 {len(data.words)} 个词汇 -> {out_path}")
            for w in data.words:
                print(f"       {w.zh:<10} ({w.x:.2f}, {w.y:.2f})")
            return out_path
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[step1] 第 {attempt} 次尝试失败: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise SystemExit(f"[step1] VLM 调用失败: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step1 · VLM 场景分析")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存重新请求")
    args = parser.parse_args()
    path = Path(args.image)
    if not path.exists():
        raise SystemExit(f"图片不存在: {path}")
    json_path = analyze_image(path, no_cache=args.no_cache)
    data = load_scene_data(json_path)
    print(f"\n场景: {data.scene}")


if __name__ == "__main__":
    main()
