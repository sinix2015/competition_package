"""
Минимальный генератор признаков для трансформера.
Вместо сложных агрегаций (EMA, окна) даёт только сырые лаги,
позволяя трансформеру самому выучить паттерны.
"""
from collections import deque
from typing import Optional

import numpy as np


class MinimalFeatureGenerator:
    """
    Генератор признаков на основе чистых лагов состояний.
    Возвращает конкатенацию последних N шагов без дополнительных преобразований.
    """

    def __init__(self, state_dim: int, n_lags: int = 4):
        """
        Args:
            state_dim: Размерность вектора состояния
            n_lags: Количество лагов для хранения (по умолчанию 4)
        """
        self.state_dim = state_dim
        self.n_lags = n_lags
        self.history: deque = deque(maxlen=n_lags)

    def reset(self) -> None:
        """Сброс истории при старте новой последовательности."""
        self.history.clear()

    def update(
        self,
        state: np.ndarray,
        step: Optional[int] = None,
        seq_ix: Optional[int] = None,
    ) -> np.ndarray:
        """
        Обновление истории и генерация признаков.

        Args:
            state: Текущее состояние (state_dim,)
            step: Номер шага (не используется)
            seq_ix: Индекс последовательности (не используется)

        Returns:
            Вектор признаков размерности (state_dim * n_lags,)
        """
        self.history.append(state.astype(np.float32, copy=False))

        # Если истории недостаточно, паддим нулями
        if len(self.history) < self.n_lags:
            padding_count = self.n_lags - len(self.history)
            padded = [np.zeros(self.state_dim, dtype=np.float32)] * padding_count
            full_history = padded + list(self.history)
        else:
            full_history = list(self.history)

        # Конкатенируем все лаги в один вектор
        features = np.concatenate(full_history, axis=0).astype(np.float32)
        return features


class LagDiffFeatureGenerator:
    """
    Расширенная версия с добавлением разностных признаков.
    Возвращает лаги + первые разности между соседними шагами.
    """

    def __init__(self, state_dim: int, n_lags: int = 4):
        self.state_dim = state_dim
        self.n_lags = n_lags
        self.history: deque = deque(maxlen=n_lags)

    def reset(self) -> None:
        self.history.clear()

    def update(
        self,
        state: np.ndarray,
        step: Optional[int] = None,
        seq_ix: Optional[int] = None,
    ) -> np.ndarray:
        self.history.append(state.astype(np.float32, copy=False))

        if len(self.history) < self.n_lags:
            padding_count = self.n_lags - len(self.history)
            padded = [np.zeros(self.state_dim, dtype=np.float32)] * padding_count
            full_history = padded + list(self.history)
        else:
            full_history = list(self.history)

        # Лаги
        lags = np.concatenate(full_history, axis=0)

        # Разности (diff между соседними лагами)
        diffs = []
        for i in range(len(full_history) - 1):
            diff = full_history[i + 1] - full_history[i]
            diffs.append(diff)

        if diffs:
            diffs_concat = np.concatenate(diffs, axis=0)
        else:
            diffs_concat = np.zeros(self.state_dim * (self.n_lags - 1), dtype=np.float32)

        # Конкатенируем лаги и разности
        features = np.concatenate([lags, diffs_concat], axis=0).astype(np.float32)
        return features
