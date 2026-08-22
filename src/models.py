"""流水线各步骤共享的数据模型。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class WordItem(BaseModel):
    """场景中一个可学习词汇。x/y 为归一化坐标(0~1)。"""

    zh: str
    en: str = ""
    ipa: str = ""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class SceneData(BaseModel):
    image: str  # 源图片路径
    scene: str = ""  # VLM 对场景的一句话概括
    words: list[WordItem]


def parse_scene_data(raw: str, image_path: Path) -> SceneData:
    """从 LLM/VLM 返回的文本中抠出 JSON 并解析为 SceneData。"""
    import json
    import re

    text = raw.strip()
    m = re.search(r"\[.*\]|\{.*\}", text, re.S)
    if m:
        text = m.group(0)
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("words") or data.get("items") or data
    if isinstance(data, dict):
        raise ValueError("未能从响应中找到词汇列表")
    words = []
    for it in data:
        it = {k.lower(): v for k, v in it.items()}
        words.append(
            WordItem(
                zh=str(it.get("zh") or it.get("chinese") or it.get("zh_word") or ""),
                en=str(it.get("en") or it.get("english") or it.get("en_word") or ""),
                ipa=str(it.get("ipa") or ""),
                x=float(it.get("x", 0.5)),
                y=float(it.get("y", 0.5)),
            )
        )
    words = [w for w in words if w.zh]
    if not (4 <= len(words) <= 20):
        raise ValueError(f"词汇数量异常: {len(words)}")
    # 坐标越界截断已在 Field 校验,这里再夹一次防浮点误差
    for w in words:
        w.x = min(max(w.x, 0.02), 0.98)
        w.y = min(max(w.y, 0.02), 0.98)
    return SceneData(image=str(image_path), words=words)


def load_scene_data(path: Path) -> SceneData:
    return SceneData.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = ["WordItem", "SceneData", "parse_scene_data", "load_scene_data", "ValidationError"]
