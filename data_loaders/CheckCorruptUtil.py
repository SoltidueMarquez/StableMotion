
"""
辅助工具：扫描已损坏样本文件，输出每个文件的损坏帧区间。

数据格式假定由 `data_loaders/corrupting_globsmpl_dataset.py` 生成，
其中每个 `.npz` 文件含有 `labels` 字段（或在特征 `x` 的最后一列）
表示每一帧是否包含合成的伪影+脚部滑动标签。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _load_labels(file_path: Path, label_key: str, fallback_last_channel: bool) -> np.ndarray:
    """
    对 `.npz` 文件进行载入，优先尝试读取 `label_key` 指定的字段；
    如果字段不存在且允许回退，则读取特征 `x` 的最后一列。
    """
    with np.load(file_path) as data:
        if label_key in data:
            labels = data[label_key]
        elif fallback_last_channel and "x" in data:
            labels = data["x"][..., -1]
        else:
            raise KeyError(f"未在 {file_path} 中找到字段 '{label_key}'")
    return np.asarray(labels).reshape(-1)


def _find_corrupt_intervals(
    mask: Iterable[bool],
    min_length: int,
) -> List[Tuple[int, int]]:
    """
    将布尔掩码转换为闭区间列表，结果为 [1, len] 1-based 帧索引。
    只返回长度大于指定 min_length 的连续段落。
    """
    intervals: List[Tuple[int, int]] = []
    start: int | None = None
    length = 0
    for index, flag in enumerate(mask):
        if flag:
            length += 1
            if start is None:
                start = index
        elif start is not None:
            if length >= min_length:
                intervals.append((start + 1, index))
            start = None
            length = 0

    if start is not None and length >= min_length:
        intervals.append((start + 1, len(mask)))
    return intervals


def _iter_motion_files(base_path: Path, ext: str, recursive: bool) -> Iterable[Path]:
    if base_path.is_file():
        yield base_path
        return

    glob_func = Path.rglob if recursive else Path.glob
    pattern = f"*{ext}"
    yield from sorted(glob_func(base_path, pattern))


def _format_intervals(intervals: List[Tuple[int, int]]) -> str:
    if not intervals:
        return "[]"
    return "[" + ",".join(f"[{start},{end}]" for start, end in intervals) + "]"


def scan_corrupt_frames(
    path: Path,
    ext: str,
    label_key: str,
    recursive: bool,
    threshold: float,
    min_length: int,
) -> None:
    base_dir = path if path.is_dir() else path.parent
    for motion_file in _iter_motion_files(path, ext, recursive):
        if not motion_file.exists():
            continue
        try:
            labels = _load_labels(motion_file, label_key, fallback_last_channel=True)
        except Exception as exc:
            print(f"跳过 {motion_file}: {exc}")
            continue

        mask = labels.astype(np.float32) > threshold
        intervals = _find_corrupt_intervals(mask, min_length)
        corrupt_frames = sum(end - start + 1 for start, end in intervals)
        try:
            rel_path = motion_file.relative_to(base_dir)
        except ValueError:
            rel_path = motion_file.name
        print(
            f"{rel_path}: {_format_intervals(intervals)} "
            f"(segments={len(intervals)}, frames={corrupt_frames})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="读取损坏后的样本，输出每段损坏帧的闭区间")
    parser.add_argument("path", type=Path, help="包含已损坏 npz 文件的目录或单个文件路径")
    parser.add_argument(
        "--ext",
        type=str,
        default=".npz",
        help="文件扩展名，默认 .npz",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="labels",
        help="表示损坏标签的字段名，默认 'labels'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="判定为损坏帧的阈值（标签大于该值视为异常）",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="忽略长度小于该值的连续段",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录（默认 False，只遍历指定目录）",
    )
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"路径不存在: {args.path}")

    scan_corrupt_frames(
        path=args.path,
        ext=args.ext,
        label_key=args.label_key,
        recursive=args.recursive,
        threshold=args.threshold,
        min_length=max(1, args.min_length),
    )


if __name__ == "__main__":
    main()
