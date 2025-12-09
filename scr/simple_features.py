from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


class SimpleFeatureGenerator:
    """Облегчённые признаки на базе текущего состояния, дельт и коротких окон."""

    def __init__(self, state_dim: int, *, seed: int = 20240912):
        self.state_dim = int(state_dim)
        if self.state_dim <= 0:
            raise ValueError("state_dim должен быть положительным")

        rng = np.random.default_rng(seed)
        self.proj_current = self._make_projection(rng, self.state_dim, 64)
        self.proj_delta = self._make_projection(rng, self.state_dim, 32)
        self.proj_avg4 = self._make_projection(rng, self.state_dim, 16)
        self.proj_avg8 = self._make_projection(rng, self.state_dim, 16)

        self.output_dim = 64 + 32 + 16 + 16

        self._history4: Deque[np.ndarray] = deque(maxlen=4)
        self._history8: Deque[np.ndarray] = deque(maxlen=8)
        self._prev_state: np.ndarray | None = None

    @staticmethod
    def _make_projection(rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
        scale = 1.0 / np.sqrt(in_dim)
        weights = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * scale
        return weights.astype(np.float32)

    def reset(self) -> None:
        self._history4.clear()
        self._history8.clear()
        self._prev_state = None

    def update(
        self,
        state: np.ndarray,
        *,
        step: int | None = None,
        seq_ix: int | None = None,
    ) -> np.ndarray:
        state_f = state.astype(np.float32, copy=False)
        if state_f.shape[0] != self.state_dim:
            raise ValueError(
                f"Ожидается вектор состояния длиной {self.state_dim}, получено {state_f.shape[0]}"
            )

        # Проекция текущего состояния
        current_feat = state_f @ self.proj_current

        # Проекция дельты по отношению к предыдущему состоянию
        if self._prev_state is None:
            delta = np.zeros_like(state_f)
        else:
            delta = state_f - self._prev_state
        delta_feat = delta @ self.proj_delta

        # Обновляем буферы коротких окон
        self._history4.append(state_f.copy())
        self._history8.append(state_f.copy())

        avg4 = self._mean(self._history4)
        avg8 = self._mean(self._history8)

        avg4_feat = avg4 @ self.proj_avg4
        avg8_feat = avg8 @ self.proj_avg8

        self._prev_state = state_f.copy()

        features = np.concatenate((current_feat, delta_feat, avg4_feat, avg8_feat), axis=0)
        return features.astype(np.float32, copy=False)

    @staticmethod
    def _mean(buffer: Deque[np.ndarray]) -> np.ndarray:
        if not buffer:
            raise ValueError("Буфер средних не должен быть пустым")
        stacked = np.stack(buffer, axis=0)
        return stacked.mean(axis=0)
