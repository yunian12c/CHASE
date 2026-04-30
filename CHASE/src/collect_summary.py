#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集多次 seed 实验的最优 epoch 并做平均。

规则（按你的论文实验逻辑）：
- 每个 seed：从该 seed 的 result.csv 中，选择 phone_test_mse 最低的 epoch 作为 best epoch
- 多个 seed：对 best epoch 的各项指标做 mean/std 汇总

默认：
- exp_root = <project_root>/exp
- seeds = 3 312 712 644 867
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SeedBest:
    seed: int
    best_epoch: int
    criterion_value: float
    row: np.ndarray


def _read_result_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    # np.savetxt(header=..., comments="") -> header is written as '# ' + header
    if header_line.startswith("#"):
        header_line = header_line.lstrip("#").strip()
    header = [h.strip() for h in header_line.split(",") if h.strip()]

    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return header, data


def _col_index(header: List[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError:
        raise ValueError(f"Column '{name}' not found in header. Available: {header}") from None


def find_best_for_seed(result_csv: Path, criterion: str) -> SeedBest:
    header, data = _read_result_csv(result_csv)
    cidx = _col_index(header, criterion)

    # 过滤掉空行/未写入的 epoch：epoch 列是第 0 列
    epoch_col = data[:, 0]
    valid = np.isfinite(epoch_col)
    data = data[valid]
    if data.size == 0:
        raise ValueError(f"No valid rows in {result_csv}")

    # 选 criterion 最小的 epoch
    best_row_idx = int(np.nanargmin(data[:, cidx]))
    best_row = data[best_row_idx]

    seed = int(result_csv.parent.name.replace("seed", "")) if result_csv.parent.name.startswith("seed") else -1
    best_epoch = int(best_row[_col_index(header, "epoch")])
    crit_val = float(best_row[cidx])
    return SeedBest(seed=seed, best_epoch=best_epoch, criterion_value=crit_val, row=best_row)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--exp-root",
        type=str,
        default=str(PROJECT_ROOT / "exp"),
        help="exp 根目录（包含 seed{seed}/result.csv）",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[3, 312, 712, 644, 867],
        help="要汇总的 seeds",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="phone_test_mse",
        help="选择 best epoch 的指标列名（取最小）",
    )
    parser.add_argument(
        "--out-best",
        type=str,
        default="result_best_by_phone_test_mse.csv",
        help="每个 seed 的 best 行输出文件名（写到 exp_root 下）",
    )
    parser.add_argument(
        "--out-mean-std",
        type=str,
        default="result_best_mean_std.csv",
        help="best 行的 mean/std 输出文件名（写到 exp_root 下）",
    )
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)

    best_list: List[SeedBest] = []
    header_ref: List[str] | None = None

    missing: List[int] = []
    for seed in args.seeds:
        result_csv = exp_root / f"seed{seed}" / "result.csv"
        if not result_csv.exists():
            missing.append(seed)
            continue
        header, _ = _read_result_csv(result_csv)
        if header_ref is None:
            header_ref = header
        elif header_ref != header:
            raise ValueError(f"Header mismatch between seeds. seed={seed} has different header.")

        best = find_best_for_seed(result_csv, args.criterion)
        best_list.append(best)

    if missing:
        print(f"[collect_summary] Warning: missing result.csv for seeds: {missing}")

    if not best_list:
        raise SystemExit("[collect_summary] No valid seed results found.")

    assert header_ref is not None
    header = ["seed", "best_epoch", args.criterion] + header_ref

    rows = []
    for b in best_list:
        rows.append([b.seed, b.best_epoch, b.criterion_value] + b.row.tolist())
    rows_np = np.asarray(rows, dtype=np.float64)

    out_best = exp_root / args.out_best
    np.savetxt(out_best, rows_np, delimiter=",", header=",".join(header), comments="")
    print(f"[collect_summary] Wrote per-seed best to: {out_best}")

    # mean/std（只对数值列做，seed/best_epoch 也会算出 mean/std 但没意义；保留方便对齐）
    mean = np.mean(rows_np, axis=0)
    std = np.std(rows_np, axis=0)
    out_mean_std = exp_root / args.out_mean_std
    np.savetxt(out_mean_std, np.vstack([mean, std]), delimiter=",", header=",".join(header), comments="")
    print(f"[collect_summary] Wrote mean/std to: {out_mean_std}")

    # 简要输出：每个 seed 的 best epoch + criterion
    for b in best_list:
        print(f"[collect_summary] seed={b.seed} best_epoch={b.best_epoch} {args.criterion}={b.criterion_value:.6f}")


if __name__ == "__main__":
    main()

