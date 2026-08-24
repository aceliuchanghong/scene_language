"""Validate and refresh the JSON-first scene catalog.

The per-scene JSON files under ``scene_catalog/<category>/`` are the source of
truth. After editing a scene's ``targets`` or scene name, run:

    python scripts/maintain_scene_catalog.py

The script rebuilds generation prompts, content signatures, category indexes,
and the root index. Use ``--check`` in CI or before committing to verify that
all derived fields are already current without writing files.

以后修改单词：

  1. 修改对应 JSON 的 targets[].zh / targets[].en
  2. 运行：

  uv run python scripts/maintain_scene_catalog.py

  脚本会自动更新生图提示词、内容签名、分类索引和总索引。

  只检查不修改：

  uv run python scripts/maintain_scene_catalog.py --check
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "scene_catalog"

EXPECTED_CATEGORY_COUNTS = {
    "H": 12,
    "F": 10,
    "T": 10,
    "W": 8,
    "S": 6,
    "L": 6,
    "R": 6,
    "D": 6,
}
EXPECTED_SCENES = 64
EXPECTED_TARGETS_PER_SCENE = 10
EXPECTED_LEARNING_SLOTS = 640
EXPECTED_UNIQUE_ENGLISH_TARGETS = 633
SCENE_ID_RE = re.compile(r"^[HFTWSLRD]\d{2}$")
STATUS_FLOW = [
    "planned",
    "generated",
    "qa_passed",
    "localized",
    "rendered",
    "published",
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def require_text(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path.relative_to(ROOT)}: {field} must be non-empty text")
    return value.strip()


def generation_prompt(
    scene_zh: str, scene_en: str, targets: list[dict[str, Any]]
) -> str:
    target_text = ", ".join(target["en"] for target in targets)
    return (
        f"Create a photorealistic vertical 9:16 educational scene of "
        f"{scene_en} ({scene_zh}).\n\n"
        f"The following ten learning targets must each appear once, clearly visible "
        f"and easy to point to: {target_text}.\n\n"
        "Arrange the targets naturally in one coherent real-life scene. Distribute "
        "them across the upper, middle and lower parts of the frame. Keep every "
        "target large enough to recognise, unobstructed, and visually separated "
        "from the others. Use realistic scale, lighting and spatial relationships. "
        "Background details may exist but should be subdued. Do not add captions, "
        "labels, subtitles, watermarks, logos, gibberish text, collages or duplicate "
        "copies of the target objects."
    )


def signature_for(scene: dict[str, Any]) -> str:
    content = {
        "id": scene["id"],
        "scene": scene["scene"],
        "targets": scene["targets"],
        "image": scene["image"],
        "generation": scene["generation"],
    }
    encoded = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scene_paths() -> list[Path]:
    paths = [path for path in CATALOG.glob("*/*.json") if path.name != "_category.json"]
    return sorted(paths)


def normalise_scene(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    scene = copy.deepcopy(raw)
    scene_id = require_text(scene.get("id"), "id", path)
    if not SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError(f"{path.relative_to(ROOT)}: invalid scene id {scene_id!r}")

    slug = require_text(scene.get("slug"), "slug", path)
    if path.stem != slug:
        raise ValueError(f"{path.relative_to(ROOT)}: filename must match slug {slug!r}")
    if not slug.startswith(f"{scene_id}_"):
        raise ValueError(f"{path.relative_to(ROOT)}: slug must start with {scene_id}_")

    category = scene.get("category")
    if not isinstance(category, dict):
        raise ValueError(f"{path.relative_to(ROOT)}: category must be an object")
    category_id = require_text(category.get("id"), "category.id", path)
    if category_id != scene_id[0]:
        raise ValueError(
            f"{path.relative_to(ROOT)}: category.id must match the scene id"
        )
    if not isinstance(category.get("order"), int):
        raise ValueError(f"{path.relative_to(ROOT)}: category.order must be an integer")
    require_text(category.get("zh"), "category.zh", path)
    require_text(category.get("en"), "category.en", path)

    batch = require_text(scene.get("batch"), "batch", path)
    if batch not in {"A", "B", "C"}:
        raise ValueError(f"{path.relative_to(ROOT)}: batch must be A, B, or C")

    scene_name = scene.get("scene")
    if not isinstance(scene_name, dict):
        raise ValueError(f"{path.relative_to(ROOT)}: scene must be an object")
    scene_zh = require_text(scene_name.get("zh"), "scene.zh", path)
    scene_en = require_text(scene_name.get("en"), "scene.en", path)

    targets = scene.get("targets")
    if not isinstance(targets, list) or len(targets) != EXPECTED_TARGETS_PER_SCENE:
        raise ValueError(
            f"{path.relative_to(ROOT)}: expected {EXPECTED_TARGETS_PER_SCENE} targets"
        )
    for expected_order, target in enumerate(targets, start=1):
        if not isinstance(target, dict):
            raise ValueError(
                f"{path.relative_to(ROOT)}: target {expected_order} must be an object"
            )
        if target.get("order") != expected_order:
            raise ValueError(
                f"{path.relative_to(ROOT)}: target order must be 1 through 10"
            )
        target["zh"] = require_text(
            target.get("zh"), f"targets[{expected_order}].zh", path
        )
        target["en"] = require_text(
            target.get("en"), f"targets[{expected_order}].en", path
        )
    english_targets = [target["en"] for target in targets]
    if len(english_targets) != len(set(english_targets)):
        raise ValueError(
            f"{path.relative_to(ROOT)}: duplicate English targets in scene"
        )

    image = scene.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"{path.relative_to(ROOT)}: image must be an object")
    expected_filename = f"{slug}.png"
    expected_image_path = f"input_pics/{path.parent.name}/{expected_filename}"
    if image.get("filename") != expected_filename:
        raise ValueError(
            f"{path.relative_to(ROOT)}: image.filename must be {expected_filename!r}"
        )
    if image.get("path") != expected_image_path:
        raise ValueError(
            f"{path.relative_to(ROOT)}: image.path must be {expected_image_path!r}"
        )
    if (
        image.get("aspect_ratio") != "9:16"
        or image.get("width") != 1080
        or image.get("height") != 1920
    ):
        raise ValueError(
            f"{path.relative_to(ROOT)}: image must be 1080x1920 with aspect_ratio 9:16"
        )

    generation = scene.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{path.relative_to(ROOT)}: generation must be an object")
    require_text(generation.get("asset_strategy"), "generation.asset_strategy", path)
    generation["prompt"] = generation_prompt(scene_zh, scene_en, targets)

    provenance = scene.get("source")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError(f"{path.relative_to(ROOT)}: source must be an object")
        provenance.pop("document", None)

    scene["content_signature"] = signature_for(scene)
    return scene


def validate_collection(scenes: list[tuple[Path, dict[str, Any]]]) -> None:
    if len(scenes) != EXPECTED_SCENES:
        raise ValueError(f"Expected {EXPECTED_SCENES} scenes, got {len(scenes)}")

    ids = [scene["id"] for _, scene in scenes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scene ids found")

    category_counts = Counter(scene["category"]["id"] for _, scene in scenes)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(
            f"Unexpected category scene counts: {dict(sorted(category_counts.items()))}"
        )

    english_targets = [
        target["en"] for _, scene in scenes for target in scene["targets"]
    ]
    if len(english_targets) != EXPECTED_LEARNING_SLOTS:
        raise ValueError(
            f"Expected {EXPECTED_LEARNING_SLOTS} learning slots, "
            f"got {len(english_targets)}"
        )
    unique_count = len(set(english_targets))
    if unique_count != EXPECTED_UNIQUE_ENGLISH_TARGETS:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_ENGLISH_TARGETS} unique English targets, "
            f"got {unique_count}"
        )


def category_documents(
    scenes: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, dict[str, Any]]]:
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, scene in scenes:
        groups[scene["category"]["id"]].append((path, scene))

    documents: list[tuple[Path, dict[str, Any]]] = []
    for category_id, members in groups.items():
        members.sort(key=lambda item: item[1]["id"])
        first_path, first = members[0]
        category = {**first["category"], "batch": first["batch"]}

        for member_path, member in members[1:]:
            comparable = {**member["category"], "batch": member["batch"]}
            if comparable != category:
                raise ValueError(
                    f"{member_path.relative_to(ROOT)}: inconsistent category metadata"
                )
            if member_path.parent != first_path.parent:
                raise ValueError(f"Category {category_id} spans multiple directories")

        document = {
            "schema_version": 1,
            "source_of_truth": "scene_json",
            "category": category,
            "summary": {
                "scene_count": len(members),
                "learning_slots": sum(len(member["targets"]) for _, member in members),
            },
            "scenes": [path.name for path, _ in members],
        }
        documents.append((first_path.parent / "_category.json", document))

    return sorted(documents, key=lambda item: item[1]["category"]["order"])


def root_index(
    scenes: list[tuple[Path, dict[str, Any]]],
    categories: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    english_targets = {
        target["en"] for _, scene in scenes for target in scene["targets"]
    }
    batches: dict[str, list[str]] = defaultdict(list)
    category_refs: list[dict[str, Any]] = []

    for path, document in categories:
        category = document["category"]
        batches[category["batch"]].append(category["id"])
        category_refs.append(
            {
                **category,
                **document["summary"],
                "directory": path.parent.name,
                "index": path.relative_to(CATALOG).as_posix(),
            }
        )

    return {
        "schema_version": 1,
        "source_of_truth": "scene_json",
        "summary": {
            "category_count": len(categories),
            "scene_count": len(scenes),
            "targets_per_scene": EXPECTED_TARGETS_PER_SCENE,
            "learning_slots": sum(len(scene["targets"]) for _, scene in scenes),
            "unique_english_targets": len(english_targets),
        },
        "status_flow": STATUS_FLOW,
        "batches": dict(sorted(batches.items())),
        "categories": category_refs,
    }


def maintain(check_only: bool = False) -> None:
    paths = scene_paths()
    raw_scenes = [(path, read_json(path)) for path in paths]
    scenes = [(path, normalise_scene(path, raw)) for path, raw in raw_scenes]
    validate_collection(scenes)

    categories = category_documents(scenes)
    index = root_index(scenes, categories)

    expected_documents = [*scenes, *categories, (CATALOG / "index.json", index)]
    stale_paths: list[Path] = []
    for path, expected in expected_documents:
        actual = read_json(path) if path.exists() else None
        if actual != expected:
            stale_paths.append(path)

    if check_only:
        if stale_paths:
            relative = ", ".join(str(path.relative_to(ROOT)) for path in stale_paths)
            raise ValueError(f"Catalog has stale derived data: {relative}")
        print(
            f"Catalog is current: {len(scenes)} scenes, "
            f"{EXPECTED_LEARNING_SLOTS} learning slots, "
            f"{EXPECTED_UNIQUE_ENGLISH_TARGETS} unique English targets"
        )
        return

    for path, document in expected_documents:
        write_json(path, document)

    print(
        f"Refreshed {len(scenes)} scenes and {len(categories)} category indexes; "
        f"{EXPECTED_LEARNING_SLOTS} learning slots, "
        f"{EXPECTED_UNIQUE_ENGLISH_TARGETS} unique English targets"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and refresh the JSON-first scene catalog"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate without writing and fail if derived data is stale",
    )
    args = parser.parse_args()
    maintain(check_only=args.check)


if __name__ == "__main__":
    main()
