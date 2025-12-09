from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Sequence


class _EnhancedRollingWindow:
    def __init__(self, window: int, state_dim: int):
        self.window = window
        self.buffer: deque[np.ndarray] = deque(maxlen=window)
        self.sum = np.zeros(state_dim, dtype=np.float32)
        self.sumsq = np.zeros(state_dim, dtype=np.float32)

    def reset(self) -> None:
        self.buffer.clear()
        self.sum.fill(0.0)
        self.sumsq.fill(0.0)

    def update(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        if len(self.buffer) == self.window:
            old = self.buffer[0]
            self.sum -= old
            self.sumsq -= old * old
        state_copy = state.astype(np.float32, copy=True)
        self.buffer.append(state_copy)
        self.sum += state_copy
        self.sumsq += state_copy * state_copy
        length = len(self.buffer)

        mean = self.sum / length
        var = np.maximum(self.sumsq / length - mean * mean, 1e-6)
        std = np.sqrt(var)

        stacked = np.stack(self.buffer, axis=0)
        min_vals = stacked.min(axis=0)
        max_vals = stacked.max(axis=0)

        return {
            "mean": mean,
            "std": std,
            "min": min_vals,
            "max": max_vals,
        }


class _ExponentialAverage:
    def __init__(self, alpha: float, state_dim: int):
        self.alpha = float(alpha)
        self.state = np.zeros(state_dim, dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        self.state.fill(0.0)
        self.initialized = False

    def update(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        if not self.initialized:
            self.state[:] = values
            self.initialized = True
        else:
            self.state[:] = self.alpha * values + (1.0 - self.alpha) * self.state
        return self.state


class AdvancedFeatureGenerator:
    """Расширенный генератор признаков с окнами, EMA, диапазонами и гармониками."""

    def __init__(
        self,
        state_dim: int,
        *,
        mean_windows: Optional[Sequence[int]] = None,
        std_windows: Optional[Sequence[int]] = None,
        range_windows: Optional[Sequence[int]] = None,
        ema_alphas: Optional[Sequence[float]] = None,
        reference_window: int = 32,
        momentum_window: int = 8,
        fourier_period: int = 128,
        fourier_harmonics: int = 3,
    ):
        self.state_dim = state_dim
        self.mean_windows = tuple(mean_windows or (4, 8, 16, 32, 64, 128))
        self.std_windows = tuple(std_windows or (8, 16, 24, 32, 64, 128))
        self.range_windows = tuple(range_windows or (16, 32, 64))
        self.reference_window = reference_window
        self.momentum_window = max(1, momentum_window)
        self.fourier_period = max(1, fourier_period)
        self.fourier_harmonics = max(0, fourier_harmonics)
        self.ema_alphas = tuple(ema_alphas or (0.3, 0.15, 0.05))

        windows_to_track = set(self.mean_windows) | set(self.std_windows) | set(self.range_windows)
        windows_to_track.add(self.reference_window)
        windows_to_track.add(self.momentum_window)
        self.windows: Dict[int, _EnhancedRollingWindow] = {
            window: _EnhancedRollingWindow(window, state_dim) for window in sorted(windows_to_track)
        }

        self.ema_trackers = [_ExponentialAverage(alpha, state_dim) for alpha in self.ema_alphas]

        self.prev_state = np.zeros(state_dim, dtype=np.float32)
        self.prev_diff = np.zeros(state_dim, dtype=np.float32)
        self.has_prev = False
        self.has_prev_diff = False

    def reset(self) -> None:
        for window in self.windows.values():
            window.reset()
        for tracker in self.ema_trackers:
            tracker.reset()
        self.has_prev = False
        self.has_prev_diff = False

    def update(self, state: np.ndarray, *, step: Optional[int] = None, seq_ix: Optional[int] = None) -> np.ndarray:
        state = state.astype(np.float32, copy=False)

        window_stats: Dict[int, Dict[str, np.ndarray]] = {}
        for key, tracker in self.windows.items():
            window_stats[key] = tracker.update(state)

        reference_stats = window_stats.get(self.reference_window) or next(iter(window_stats.values()))
        ref_mean = reference_stats["mean"]
        ref_std = reference_stats["std"]
        ref_std_safe = np.where(ref_std == 0.0, 1.0, ref_std)

        momentum_stats = window_stats.get(self.momentum_window, reference_stats)

        ema_outputs: List[np.ndarray] = [tracker.update(state) for tracker in self.ema_trackers]

        if self.has_prev:
            diff_prev = state - self.prev_state
        else:
            diff_prev = np.zeros_like(state)

        if self.has_prev_diff:
            accel = diff_prev - self.prev_diff
        else:
            accel = np.zeros_like(state)

        momentum = state - momentum_stats["mean"]
        momentum_std = np.where(momentum_stats["std"] == 0.0, 1.0, momentum_stats["std"])

        range_features: List[np.ndarray] = []
        for window in self.range_windows:
            stats = window_stats.get(window)
            if stats is None:
                continue
            rng = stats["max"] - stats["min"]
            rng_std = np.where(stats["std"] == 0.0, 1.0, stats["std"])
            range_features.append(rng)
            range_features.append(rng / rng_std)

        ema_diffs: List[np.ndarray] = []
        for left, right in zip(ema_outputs, ema_outputs[1:]):
            ema_diffs.append(left - right)

        ema_tail = ema_outputs[-1] if ema_outputs else ref_mean
        ema_ratio = state / np.where(np.abs(ema_tail) < 1e-4, 1.0, ema_tail)

        diff_ratio = diff_prev / momentum_std
        accel_ratio = accel / momentum_std

        features: List[np.ndarray] = [state]

        for window in self.mean_windows:
            stats = window_stats.get(window)
            if stats is not None:
                features.append(stats["mean"])

        for window in self.std_windows:
            stats = window_stats.get(window)
            if stats is not None:
                features.append(stats["std"])

        deviation = state - ref_mean
        features.append(deviation)
        features.append(deviation / ref_std_safe)

        features.extend(ema_outputs)
        features.extend(ema_diffs)
        if ema_outputs:
            features.append(ema_outputs[0] - ema_tail)
        else:
            features.append(deviation)
        features.append(ema_ratio)

        features.append(diff_prev)
        features.append(diff_ratio)
        features.append(accel)
        features.append(accel_ratio)
        features.append(np.abs(diff_prev))
        features.append(momentum)

        features.extend(range_features)

        if self.fourier_harmonics > 0:
            step_value = float(step or 0)
            base = 2.0 * np.pi * step_value / float(self.fourier_period)
            for harmonic in range(1, self.fourier_harmonics + 1):
                angle = harmonic * base
                sin_factor = np.float32(np.sin(angle))
                cos_factor = np.float32(np.cos(angle))
                features.append(state * sin_factor)
                features.append(state * cos_factor)

        features.append(state * diff_prev)
        features.append((state - ema_tail) * diff_prev)

        self.prev_state = state.copy()
        self.prev_diff = diff_prev.copy()
        self.has_prev = True
        self.has_prev_diff = True

        return np.concatenate(features, axis=0).astype(np.float32)
