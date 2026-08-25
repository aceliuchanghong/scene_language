"""Step 2 · Language Generator — 调用 LLM 生成地道外语(英文/日文/韩文)、注音与例句。

单独运行:
    uv run python -m src.step2.cli --image input_pics/生活场景/carriage.png
    uv run python -m src.step2.cli --image ... --lang ja
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

PROMPT_EN = """你是英式英语词汇专家。下面是一张{scene}场景照片中识别出的中文词汇列表。
为每个词给出:
- target: 最地道、最常用的英文单词或短语(英式用法优先,如 pram)
- pron: 该英文表达的标准英式发音国际音标(带重音符号与长音符号,如 /ˈpræm/)
- example_target: 一个包含该词的日常简短英文例句(8~14 词,英式用法)
- example_zh: 该例句的中文翻译

只输出 JSON 数组,顺序与输入一致:
[{{"zh": "婴儿车", "target": "pram", "pron": "/ˈpræm/", "example_target": "She pushed the pram along the pavement.", "example_zh": "她推着婴儿车沿着人行道走。"}}]

词汇列表:
{words}"""

PROMPT_JA = """你是地道日语词汇教学专家。下面是一张{scene}场景照片中识别出的中文词汇列表。
为每个词给出:
- target: 最常用、最地道的日语单词或短语(日文汉字或假名,如 玄関、傘立て、スリッパ)
- pron: 该词的标准平假名读音与罗马音(如 げんかん / genkan)
- example_target: 一个包含该词的日常简短自然口语例句(日文)
- example_zh: 该例句的中文翻译

只输出 JSON 数组,顺序与输入一致:
[{{"zh": "前门", "target": "玄関", "pron": "げんかん / genkan", "example_target": "玄関で靴を脱いでください。", "example_zh": "请在玄关脱鞋。"}}]

词汇列表:
{words}"""

PROMPT_KO = """你是地道韩语词汇教学专家。下面是一张{scene}场景照片中识别出的中文词汇列表。
为每个词给出:
- target: 最常用、最地道的韩语单词或短语(韩文谚文,如 현관、우산꽂이、슬리퍼)
- pron: 该词的标准韩语罗马拼音(Revised Romanization,如 hyeon-gwan)
- example_target: 一个包含该词的日常简短自然口语例句(韩文)
- example_zh: 该例句的中文翻译

只输出 JSON 数组,顺序与输入一致:
[{{"zh": "前门", "target": "현관", "pron": "hyeon-gwan", "example_target": "현관에서 신발을 벗어 주세요.", "example_zh": "请在玄关脱鞋。"}}]

词汇列表:
{words}"""

PROMPTS = {
    "en": PROMPT_EN,
    "ja": PROMPT_JA,
    "ko": PROMPT_KO,
}


def generate_language(
    json_path: Path, lang: str = "en", no_cache: bool = False
) -> Path:
    """读取 step1 的 JSON,补全 target/pron/例句后写回。返回同一 JSON 路径。"""
    data: SceneData = load_scene_data(json_path)
    lang = (data.lang or lang).lower()

    if (
        all(
            (w.target or w.en)
            and (w.pron or w.ipa)
            and (w.example_target or w.example_en)
            and w.example_zh
            for w in data.words
        )
        and not no_cache
    ):
        print(f"[step2] [{lang}] 已有翻译,使用缓存 {json_path}")
        return json_path

    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
    words_text = "\n".join(f"{i + 1}. {w.zh}" for i, w in enumerate(data.words))
    prompt_template = PROMPTS.get(lang, PROMPT_EN)
    prompt = prompt_template.format(scene=data.scene or "生活", words=words_text)

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
                target = str(
                    it.get("target")
                    or it.get("en")
                    or it.get("ja")
                    or it.get("ko")
                    or it.get("word")
                    or ""
                ).strip()
                pron = str(
                    it.get("pron")
                    or it.get("ipa")
                    or it.get("kana")
                    or it.get("romaji")
                    or ""
                ).strip()
                example_target = str(
                    it.get("example_target")
                    or it.get("example_en")
                    or it.get("example")
                    or ""
                ).strip()
                example_zh = str(
                    it.get("example_zh") or it.get("example_translation") or ""
                ).strip()

                if lang == "en" and pron and not pron.startswith("/"):
                    pron = f"/{pron.strip('/')}/"

                w.target = target
                w.en = target
                w.pron = pron
                w.ipa = pron
                w.example_target = example_target
                w.example_en = example_target
                w.example_zh = example_zh

            if not all(
                w.target and w.pron and w.example_target and w.example_zh
                for w in data.words
            ):
                raise ValueError("存在缺失 目标词/注音/例句 的条目")

            data.lang = lang
            json_path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
            print(f"[step2] [{lang}] 翻译完成 -> {json_path}")
            for w in data.words:
                print(f"       {w.zh:<10} {w.target:<18} {w.pron}")
            return json_path
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[step2] 第 {attempt} 次尝试失败: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise SystemExit(f"[step2] LLM 调用失败: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step2 · LLM 翻译与音标")
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
    parser.add_argument("--no-cache", action="store_true", help="忽略已有翻译重新请求")
    args = parser.parse_args()
    json_path = (
        Path(args.json)
        if args.json
        else config.get_json_path(Path(args.image).stem, args.lang)
    )
    if not json_path.exists():
        raise SystemExit(
            f"找不到 {json_path},先运行: uv run python -m src.step1.cli --image {args.image} --lang {args.lang}"
        )
    generate_language(json_path, lang=args.lang, no_cache=args.no_cache)


if __name__ == "__main__":
    main()
