"""流水线各步骤共享的数据模型。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

_PUNCT = ".,!?;:'\"“”‘’()[]{}<>《》…—-·,。!?:;、~〜·"


def strip_punct(s: str) -> str:
    """去掉标点(保留英文撇号,避免破坏 don't 等词),压缩多余空格。"""
    s = "".join(ch for ch in s if ch not in _PUNCT or ch == "'")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


class WordItem(BaseModel):
    """场景中一个可学习词汇。x/y 为归一化坐标(0~1)。"""

    zh: str
    en: str = ""
    ipa: str = ""
    example_en: str = ""  # 英文例句
    example_zh: str = ""  # 例句中文翻译
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)

    @field_validator("example_en", "example_zh", mode="after")
    @classmethod
    def _no_punct(cls, v: str) -> str:
        return strip_punct(v)


class SceneData(BaseModel):
    image: str  # 源图片路径
    scene: str = ""  # VLM 对场景的一句话概括
    words: list[WordItem]


def _extract_json(raw: str):
    """用 json_repair 从 LLM 响应中恢复 JSON(容忍截断/围栏/尾随文本等)。"""
    try:
        from json_repair import repair_json

        obj = repair_json(raw, return_objects=True)
        if isinstance(obj, list):
            # 尾随文本可能被并成多元素数组,取第一个字典元素
            dicts = [e for e in obj if isinstance(e, dict)]
            if dicts:
                obj = dicts[0]
        if obj is not None:
            return obj
    except ImportError:
        pass
    raise ValueError("未能从响应中找到有效 JSON")


def _normalize_coords(items: list[dict], image_path: Path) -> list[dict]:
    """坐标归一化兜底:VLM 偶尔返回 0~100 或像素坐标,统一折算到 0~1。"""
    xs = [abs(float(it.get("x", 0.5))) for it in items]
    ys = [abs(float(it.get("y", 0.5))) for it in items]
    mx, my = max(xs), max(ys)
    if mx <= 1.5 and my <= 1.5:
        return items
    if mx <= 101 and my <= 101:  # 百分比刻度
        sx = sy = 100.0
    else:  # 像素刻度,按真实图片尺寸换算
        try:
            from PIL import Image

            with Image.open(image_path) as im:
                sx, sy = float(im.width), float(im.height)
        except Exception:
            sx, sy = max(mx, 1.0), max(my, 1.0)
    for it, x, y in zip(items, xs, ys):
        it["x"], it["y"] = x / sx, y / sy
    return items


def parse_scene_data(raw: str, image_path: Path) -> SceneData:
    """从 LLM/VLM 返回的文本中抠出 JSON 并解析为 SceneData。"""
    data = _extract_json(raw)
    if isinstance(data, dict):
        data = data.get("words") or data.get("items") or data
    if isinstance(data, dict):
        raise ValueError("未能从响应中找到词汇列表")
    items = []
    for it in data:
        if not isinstance(it, dict):
            continue
        it = {k.lower(): v for k, v in it.items()}
        if not str(it.get("zh") or it.get("chinese") or it.get("zh_word") or ""):
            continue
        items.append(it)
    _normalize_coords(items, image_path)
    words = [
        WordItem(
            zh=str(it.get("zh") or it.get("chinese") or it.get("zh_word") or ""),
            en=str(it.get("en") or it.get("english") or it.get("en_word") or ""),
            ipa=str(it.get("ipa") or ""),
            example_en=str(it.get("example_en") or it.get("example") or ""),
            example_zh=str(
                it.get("example_zh") or it.get("example_translation") or ""
            ),
            x=float(it.get("x", 0.5)),
            y=float(it.get("y", 0.5)),
        )
        for it in items
    ]
    if not (4 <= len(words) <= 20):
        raise ValueError(f"词汇数量异常: {len(words)}")
    # 坐标越界截断已在 Field 校验,这里再夹一次防浮点误差
    for w in words:
        w.x = min(max(w.x, 0.02), 0.98)
        w.y = min(max(w.y, 0.02), 0.98)
    return SceneData(image=str(image_path), words=words)


def load_scene_data(path: Path) -> SceneData:
    return SceneData.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "WordItem",
    "SceneData",
    "parse_scene_data",
    "load_scene_data",
    "ValidationError",
]
