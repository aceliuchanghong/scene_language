"""Run ``uv run sau ...`` lines from upload markdown, one browser at a time.

Dry-run (default) only prints the plan and checks that videos and covers exist.
Pass ``--run`` to upload. A command is recorded in ``output/upload_ledger.json``
only after exit code 0, and later runs skip it. The first failure stops the
batch so a rerun does not publish the successes twice.

    uv run python scripts/batch_upload.py
    uv run python scripts/batch_upload.py --run --limit 1
    uv run python scripts/batch_upload.py --run --platform douyin
    uv run python scripts/batch_upload.py --run --platform tencent
    uv run python scripts/batch_upload.py --run src/command/UPLOAD_08_DIGITAL_LIFE_JA.md
    uv run python scripts/batch_upload.py --run src/command/UPLOAD_08_DIGITAL_LIFE_JA.md --platform tencent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAU_DIR = ROOT.parent / "social-auto-upload"
LEDGER = ROOT / "output" / "upload_ledger.json"
DEFAULT_DOCS = [
    ROOT / "src" / "command" / "UPLOAD_07_TRAVEL_AND_OUTDOORS.md",
    ROOT / "src" / "command" / "UPLOAD_07_TRAVEL_AND_OUTDOORS_JA.md",
]
PATH_FLAGS = {"--file", "--thumbnail", "--thumbnail-portrait", "--thumbnail-landscape"}


@dataclass
class Job:
    source: Path
    line: int
    scene: str
    platform: str
    command: str
    argv: list[str]

    @property
    def key(self) -> str:
        return hashlib.sha256(self.command.encode("utf-8")).hexdigest()


def parse_docs(paths: list[Path]) -> list[Job]:
    jobs: list[Job] = []
    for path in paths:
        scene = ""
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if line.startswith("## "):
                scene = line[3:].split()[0]
            if not line.startswith("uv run sau "):
                continue
            argv = shlex.split(line)
            if len(argv) < 4 or argv[:3] != ["uv", "run", "sau"]:
                raise SystemExit(f"{path.name}:{line_no} 不是 uv run sau 命令")
            jobs.append(Job(path, line_no, scene, argv[3], line, argv))
    return jobs


def flag(argv: list[str], name: str) -> str:
    if name not in argv:
        return ""
    index = argv.index(name)
    return argv[index + 1] if index + 1 < len(argv) else ""


def missing_files(job: Job) -> list[str]:
    missing: list[str] = []
    for index, arg in enumerate(job.argv[:-1]):
        if arg not in PATH_FLAGS:
            continue
        raw = job.argv[index + 1]
        path = Path(raw)
        if not path.is_absolute():
            path = SAU_DIR / path
        if not path.is_file():
            missing.append(raw)
    return missing


def load_done() -> dict:
    if not LEDGER.is_file():
        return {}
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    return data.get("done", {})


def save_done(done: dict, job: Job) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    done[job.key] = {
        "command": job.command,
        "scene": job.scene,
        "platform": job.platform,
        "source": job.source.name,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    LEDGER.write_text(
        json.dumps({"done": done}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="按 markdown 里的 sau 命令顺序上传")
    parser.add_argument(
        "docs", nargs="*", type=Path, help="上传命令 markdown，默认这两个 07 文件"
    )
    parser.add_argument("--run", action="store_true", help="真正上传。不加则只预览")
    parser.add_argument(
        "--platform", choices=("douyin", "tencent"), help="只跑一个平台"
    )
    parser.add_argument("--limit", type=int, help="最多执行几条尚未完成的命令")
    parser.add_argument(
        "--pause", type=float, default=5, help="两条上传之间停几秒，默认 5"
    )
    args = parser.parse_args()

    docs = args.docs or DEFAULT_DOCS
    docs = [path if path.is_absolute() else ROOT / path for path in docs]
    for path in docs:
        if not path.is_file():
            raise SystemExit(f"找不到 {path}")
    if not SAU_DIR.is_dir():
        raise SystemExit(f"找不到 sau 目录 {SAU_DIR}")

    jobs = parse_docs(docs)
    if args.platform:
        jobs = [job for job in jobs if job.platform == args.platform]
    done = load_done()

    pending = 0
    blocked = False
    for index, job in enumerate(jobs, 1):
        gone = missing_files(job)
        state = "skip" if job.key in done else "missing" if gone else "pending"
        if state == "pending":
            pending += 1
        if state == "missing":
            blocked = True
        title = flag(job.argv, "--title")
        schedule = flag(job.argv, "--schedule")
        print(
            f"{index:02d} {state:7} {job.platform:7} {job.scene:4} {schedule}  {title}"
        )
        if gone:
            print(f"    缺文件: {', '.join(gone)}")

    runnable = [job for job in jobs if job.key not in done and not missing_files(job)]
    if args.limit is not None:
        runnable = runnable[: args.limit]
    print(
        f"\n共 {len(jobs)} 条，待上传 {len(runnable)} 条"
        + ("（预览，未执行）" if not args.run else "")
    )
    if not args.run:
        return 1 if blocked else 0
    if blocked:
        raise SystemExit("有文件缺失，未开始上传")
    if not runnable:
        return 0

    for index, job in enumerate(runnable, 1):
        print(
            f"\n[{index}/{len(runnable)}] {job.source.name}:{job.line} {job.scene} {job.platform}"
        )
        try:
            result = subprocess.run(job.argv, cwd=SAU_DIR)
        except KeyboardInterrupt:
            print("\n已中断。这一条未记入完成，下次会重试。", file=sys.stderr)
            return 130
        if result.returncode != 0:
            print(
                f"失败，退出码 {result.returncode}。已完成的不会重传。", file=sys.stderr
            )
            return result.returncode
        save_done(done, job)
        if index < len(runnable):
            time.sleep(args.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
