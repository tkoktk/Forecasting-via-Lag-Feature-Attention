from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings
from typing import Iterator, Tuple

# This implementation follows the guidance of Lopez de. Prado 'Advances in Financial Machine Learning' Chapter 7


@dataclass
class FoldSpec:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    purge_idx: np.ndarray
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    n_train: int
    n_test: int
    n_purged: int


class PurgedWalkForwardSplit:
    def __init__(
        self,
        horizon: int,
        step: int,
        eval_start: pd.Timestamp,
        eval_end: pd.Timestamp,
        embargo: int = 0,
        min_train_size: int = 1,
    ):
        self.horizon = horizon
        self.step = step
        self.embargo = embargo
        self.eval_start = pd.Timestamp(eval_start)
        self.eval_end = pd.Timestamp(eval_end)
        self.min_train_size = min_train_size

    def split(self, dates: pd.DatetimeIndex) -> Iterator[FoldSpec]:
        dates = pd.DatetimeIndex(dates)
        n = len(dates)

        eval_mask = (dates >= self.eval_start) & (dates <= self.eval_end)
        eval_positions = np.where(eval_mask)[0]

        print(
            f"split | n_dates={n}, eval_positions={len(eval_positions)}, "
            f"eval_range=[{self.eval_start.date()}, {self.eval_end.date()}]"
        )

        if len(eval_positions) == 0:
            raise ValueError(
                f"No dates between {self.eval_start.date()} and {self.eval_end.date()}"
            )

        fold_id = 0
        for block_start_offset in range(0, len(eval_positions), self.step):
            block_positions = eval_positions[
                block_start_offset : block_start_offset + self.step
            ]
            test_start_pos = block_positions[0]
            test_end_pos = block_positions[-1]

            train_cutoff = test_start_pos - self.horizon - self.embargo
            train_idx = np.arange(0, max(train_cutoff, 0))
            test_idx = block_positions
            purge_idx = np.arange(max(train_cutoff, 0), test_start_pos)

            print(
                f"fold {fold_id} | test=[{test_start_pos}:{test_end_pos + 1}] | ({dates[test_start_pos].date()} to {dates[test_end_pos].date()}) | train_cutoff={train_cutoff} | n_train={len(train_idx)} | n_purged={len(purge_idx)}"
            )

            if len(train_idx) < self.min_train_size:
                warnings.warn(
                    f"fold {fold_id} skipped: n_train={len(train_idx)} "
                    f"< min_train_size={self.min_train_size}"
                )
                fold_id += 1
                continue

            yield FoldSpec(
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=test_idx,
                purge_idx=purge_idx,
                test_start_date=dates[test_start_pos],
                test_end_date=dates[test_end_pos],
                n_train=len(train_idx),
                n_test=len(test_idx),
                n_purged=len(purge_idx),
            )
            fold_id += 1


class PurgedTrainValSplit:
    def __init__(
        self,
        val_fraction: float,
        horizon: int,
        min_inner_train_size: int = 1,
        min_val_size: int = 1,
    ):
        self.val_fraction = val_fraction
        self.horizon = horizon
        self.min_inner_train_size = min_inner_train_size
        self.min_val_size = min_val_size

    def split(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        split_pos = int(n * (1.0 - self.val_fraction))
        inner_train_end = split_pos - self.horizon

        inner_train_idx = np.arange(0, max(inner_train_end, 0))
        val_idx = np.arange(split_pos, n)

        print(
            f"PurgedTrainValSplit | n={n} split_pos={split_pos} "
            f"inner_train_end={inner_train_end} "
            f"n_inner_train={len(inner_train_idx)} n_val={len(val_idx)} "
            f"n_purged={max(split_pos - max(inner_train_end, 0), 0)}"
        )

        if len(inner_train_idx) < self.min_inner_train_size:
            raise ValueError(
                f"Inner train size {len(inner_train_idx)} < min_inner_train_size "
                f"{self.min_inner_train_size}. Increase outer fold size or decrease val_fraction."
            )
        if len(val_idx) < self.min_val_size:
            raise ValueError(
                f"Val size {len(val_idx)} < min_val_size {self.min_val_size}. "
                f"Increase outer fold size or increase val_fraction."
            )

        return inner_train_idx, val_idx
