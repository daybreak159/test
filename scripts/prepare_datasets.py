#!/usr/bin/env python3
"""Prepare dataset files for MemSkill / MemSkillJittor experiments.

The original MemSkill code supports several dataset families through
``--dataset {locomo,longmemeval,hotpotqa,alfworld}``.  This helper keeps the
expected local filenames explicit and avoids baking private/local paths into
the repository.

For datasets that require an external download or license, pass either a local
source file with ``--*-source`` or a direct URL with ``--*-url``.  The script can
then copy/download the file into ``data/`` and run lightweight format checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("data")


def read_json_or_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"{path} is empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def copy_or_download(source: str | None, url: str | None, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source:
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"source file not found: {src}")
        shutil.copyfile(src, target)
        return True
    if url:
        urllib.request.urlretrieve(url, target)
        return True
    return target.exists()


def require_list(data: Any, name: str) -> list[Any]:
    if not isinstance(data, list):
        raise ValueError(f"{name} should be a JSON list, got {type(data).__name__}")
    return data


def prepare_locomo(args: argparse.Namespace) -> None:
    out_dir = Path(args.data_dir)
    raw_path = out_dir / "locomo.json"
    has_raw = copy_or_download(args.locomo_source, args.locomo_url, raw_path)

    if has_raw:
        raw = require_list(read_json_or_jsonl(raw_path), "LoCoMo")
        if len(raw) < args.locomo_subset_size:
            raise ValueError(
                f"LoCoMo source has {len(raw)} samples, "
                f"need at least {args.locomo_subset_size}"
            )
        locomo10 = raw[: args.locomo_subset_size]
        write_json(out_dir / "locomo10.json", locomo10)
        write_json(out_dir / "locomo10_one.json", locomo10[:1])
        print(f"[locomo] wrote {out_dir / 'locomo10.json'}")
        print(f"[locomo] wrote {out_dir / 'locomo10_one.json'}")
        return

    for path in (out_dir / "locomo10.json", out_dir / "locomo10_one.json"):
        if not path.exists():
            raise FileNotFoundError(
                "LoCoMo data not found. Provide --locomo-source or --locomo-url, "
                f"or place {path} manually."
            )
        require_list(read_json_or_jsonl(path), str(path))
        print(f"[locomo] found {path}")


def prepare_longmemeval(args: argparse.Namespace) -> None:
    out_dir = Path(args.data_dir)
    target = out_dir / "longmemeval_s.json"
    has_data = copy_or_download(args.longmemeval_source, args.longmemeval_url, target)

    if not has_data:
        print(
            "[longmemeval] skipped: provide --longmemeval-source or "
            "--longmemeval-url if you want to run this dataset."
        )
        return

    data = require_list(read_json_or_jsonl(target), "LongMemEval")
    print(f"[longmemeval] found {len(data)} samples at {target}")

    split_path = out_dir / "longmemeval_s_splits.json"
    if not split_path.exists():
        n = len(data)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        splits = {
            "train": list(range(0, train_end)),
            "val": list(range(train_end, val_end)),
            "test": list(range(val_end, n)),
        }
        write_json(split_path, splits)
        print(f"[longmemeval] wrote fallback split file {split_path}")
    else:
        print(f"[longmemeval] found split file {split_path}")


def prepare_hotpotqa(args: argparse.Namespace) -> None:
    out_dir = Path(args.data_dir)
    train_target = out_dir / "hotpotqa_train.json"
    eval_target = out_dir / "eval_200.json"

    train_ok = copy_or_download(args.hotpotqa_source, args.hotpotqa_url, train_target)
    eval_ok = copy_or_download(args.hotpotqa_eval_source, args.hotpotqa_eval_url, eval_target)

    if not train_ok:
        print(
            "[hotpotqa] train file skipped: provide --hotpotqa-source or "
            "--hotpotqa-url if you want to train on HotpotQA."
        )
    else:
        require_list(read_json_or_jsonl(train_target), "HotpotQA train")
        print(f"[hotpotqa] found train file {train_target}")

    if not eval_ok:
        print(
            "[hotpotqa] eval file skipped: provide --hotpotqa-eval-source or "
            "--hotpotqa-eval-url if you want HotpotQA eval."
        )
    else:
        require_list(read_json_or_jsonl(eval_target), "HotpotQA eval")
        print(f"[hotpotqa] found eval file {eval_target}")


def prepare_alfworld(args: argparse.Namespace) -> None:
    out_dir = Path(args.data_dir)
    train_target = out_dir / "alfworld_train_offline.json"
    eval_target = out_dir / "alfworld_expert_eval_in_distribution.json"

    train_ok = copy_or_download(args.alfworld_source, args.alfworld_url, train_target)
    eval_ok = copy_or_download(args.alfworld_eval_source, args.alfworld_eval_url, eval_target)

    if not train_ok:
        print(
            "[alfworld] train file skipped: provide --alfworld-source or "
            "--alfworld-url if you want ALFWorld pair training."
        )
    else:
        data = read_json_or_jsonl(train_target)
        if not isinstance(data, (dict, list)):
            raise ValueError("ALFWorld train file should be a JSON dict or list")
        print(f"[alfworld] found train file {train_target}")

    if not eval_ok:
        print(
            "[alfworld] eval file skipped: provide --alfworld-eval-source or "
            "--alfworld-eval-url if you want ALFWorld eval."
        )
    else:
        read_json_or_jsonl(eval_target)
        print(f"[alfworld] found eval file {eval_target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", "locomo", "longmemeval", "hotpotqa", "alfworld"],
        default="all",
        help="Dataset family to prepare.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))

    parser.add_argument("--locomo-source")
    parser.add_argument("--locomo-url")
    parser.add_argument("--locomo-subset-size", type=int, default=10)

    parser.add_argument("--longmemeval-source")
    parser.add_argument("--longmemeval-url")

    parser.add_argument("--hotpotqa-source")
    parser.add_argument("--hotpotqa-url")
    parser.add_argument("--hotpotqa-eval-source")
    parser.add_argument("--hotpotqa-eval-url")

    parser.add_argument("--alfworld-source")
    parser.add_argument("--alfworld-url")
    parser.add_argument("--alfworld-eval-source")
    parser.add_argument("--alfworld-eval-url")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    tasks = {
        "locomo": prepare_locomo,
        "longmemeval": prepare_longmemeval,
        "hotpotqa": prepare_hotpotqa,
        "alfworld": prepare_alfworld,
    }
    selected = tasks.keys() if args.dataset == "all" else [args.dataset]

    for name in selected:
        print(f"\n== preparing {name} ==")
        tasks[name](args)

    print("\nDataset preparation check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
