"""Step 2 · Language Generator — 调用 LLM 生成地道英文与英式音标。

单独运行:
    uv run python -m src.step2.cli --image input_pics/生活场景/carriage.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

from src import config
from src.models import SceneData, load_scene_data

PROMPT = """你是英式英语词汇专家。下面是一张{scene}场景照片中识别出的中文词汇列表。
为每个词给出:
- en: 最地道、最常用的英文单词或短语(英式用法优先,如 pram/baby carriage)
- ipa: 该英文表达的标准英式发音国际音标(带重音符号与长音符号,如 /ˈpræm/)

只输出 JSON 数组,顺序与输入一致:
[{{"zh": "婴儿车", "en": "pram", "ipa": "/ˈpræm/"}}]

词汇列表:
{words}"""


def generate_language(json_path: Path, no_cache: bool = False) -> Path:
    """读取 step1 的 JSON,补全 en/ipa 后写回。返回同一 JSON 路径。"""
    data: SceneData = load_scene_data(json_path)
    if all(w.en and w.ipa for w in data.words) and not no_cache:
        print(f"[step2] 已有翻译,使用缓存 {json_path}")
        return json_path

    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
    words_text = "\n".join(f"{i + 1}. {w.zh}" for i, w in enumerate(data.words))
    prompt = PROMPT.format(scene=data.scene or "生活", words=words_text)

    last_err: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=config.MODEL,
                max_tokens=config.MAX_TOKENS,
                timeout=config.TIMEOUT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (resp.choices[0].message.content or "").strip()
            start, end = raw.find("["), raw.rfind("]")
            items = json.loads(raw[start : end + 1])
            if len(items) != len(data.words):
                raise ValueError(f"返回条数 {len(items)} != {len(data.words)}")
            for w, it in zip(data.words, items):
                it = {k.lower(): v for k, v in it.items()}
                w.en = str(it.get("en", "")).strip()
                w.ipa = str(it.get("ipa", "")).strip()
                if not w.ipa.startswith("/"):
                    w.ipa = f"/{w.ipa.strip('/')}/"
            if not all(w.en and w.ipa for w in data.words):
                raise ValueError("存在缺失 en/ipa 的条目")
            json_path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
            print(f"[step2] 翻译完成 -> {json_path}")
            for w in data.words:
                print(f"       {w.zh:<10} {w.en:<18} {w.ipa}")
            return json_path
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[step2] 第 {attempt} 次尝试失败: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise SystemExit(f"[step2] LLM 调用失败: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step2 · LLM 翻译与音标")
    parser.add_argument("--image", required=True, help="输入图片(定位 output/json 下同名 JSON)")
    parser.add_argument("--json", default=None, help="直接指定 step1 输出的 JSON 路径")
    parser.add_argument("--no-cache", action="store_true", help="忽略已有翻译重新请求")
    args = parser.parse_args()
    json_path = Path(args.json) if args.json else config.JSON_DIR / f"{Path(args.image).stem}.json"
    if not json_path.exists():
        raise SystemExit(
            f"找不到 {json_path},先运行: uv run python -m src.step1.cli --image {args.image}"
        )
    generate_language(json_path, no_cache=args.no_cache)


if __name__ == "__main__":
    main()
