import argparse
import importlib
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# --------------------------------------------------------------------
# Пути и импорты проекта
# --------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_prep import build_training_arrays, load_dataset
from src.models import FeatureMLP, TemporalConvGRU, TemporalDilatedCNN, TemporalTransformer, TemporalResNet, TemporalMamba, TemporalDeepGRU, TemporalTransformerV2
from src.ema_trainer import EMAWrapper

# --------------------------------------------------------------------
# Логирование и предупреждения
# --------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

# Фиксированный сид для разбиения на фолды, чтобы сплиты были стабильны
FIXED_SPLIT_SEED = 42


# --------------------------------------------------------------------
# Конфиги и статистика
# --------------------------------------------------------------------

@dataclass
class DataStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray

    def save(self, path: Path) -> None:
        np.savez(
            path,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            state_mean=self.state_mean,
            state_std=self.state_std,
        )


@dataclass
class MLPConfig:
    fold_epochs: int = 70
    fold_patience: int = 9
    fold_weight_decay: float = 3e-4
    final_epochs: int = 130
    final_patience: int = 18
    final_weight_decay: float = 1.2e-4
    dropout: float = 0.1
    batch_size: int = 1024
    val_batch_size: int = 2048
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 1e-3


@dataclass
class RidgeConfig:
    fold_alpha: float = 4.5
    final_alpha: float = 8.0


@dataclass
class TemporalConfig:
    model_type: str = "conv_gru"
    fold_epochs: int = 45
    fold_patience: int = 7
    fold_weight_decay: float = 3e-4
    final_epochs: int = 70
    final_patience: int = 12
    final_weight_decay: float = 2.2e-4
    batch_size: int = 512
    val_batch_size: int = 1024
    lr: float = 1e-3

    # ConvGRU
    conv_channels: int = 64
    hidden_dim: int = 64

    # CNN
    cnn_channels: int = 96
    cnn_layers: int = 4
    cnn_kernel_size: int = 3
    cnn_dropout: float = 0.1

    # Transformer
    transformer_d_model: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 256
    transformer_dropout: float = 0.1
    transformer_max_window: int = 128
    transformer_pe_type: str = "learned"
    warmup_epochs: int = 0
    use_ema: bool = False
    ema_decay: float = 0.995

    # ResNet
    resnet_channels: int = 128
    resnet_blocks: int = 4
    resnet_dropout: float = 0.1
    resnet_kernel_size: int = 3

    # Mamba
    mamba_d_model: int = 128
    mamba_layers: int = 2
    mamba_dropout: float = 0.1
    mamba_expand: int = 2

    # Deep GRU
    deep_gru_hidden: int = 128
    deep_gru_layers: int = 4
    deep_gru_dropout: float = 0.1

@dataclass
class TrainConfig:
    n_folds: int = 5
    final_val_ratio: float = 0.1
    sequence_window: int = 16
    ensemble_alpha: float = 12.0
    random_seed: int = 42
    mlp: MLPConfig = field(default_factory=MLPConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    feature_module: Optional[str] = None
    feature_class: str = "FeatureGenerator"
    feature_params: Dict[str, Any] = field(default_factory=dict)
    # "level" — предсказываем уровень; "delta" — предсказываем приращение (next - prev)
    target_mode: str = "level"

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TrainConfig":
        payload = dict(data)
        mlp_data = payload.pop("mlp", None)
        ridge_data = payload.pop("ridge", None)
        temporal_data = payload.pop("temporal", None)
        return cls(
            **payload,
            mlp=MLPConfig(**mlp_data) if isinstance(mlp_data, dict) else MLPConfig(),
            ridge=RidgeConfig(**ridge_data) if isinstance(ridge_data, dict) else RidgeConfig(),
            temporal=TemporalConfig(**temporal_data) if isinstance(temporal_data, dict) else TemporalConfig(),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_folds": self.n_folds,
            "final_val_ratio": self.final_val_ratio,
            "sequence_window": self.sequence_window,
            "ensemble_alpha": self.ensemble_alpha,
            "random_seed": self.random_seed,
            "mlp": asdict(self.mlp),
            "ridge": asdict(self.ridge),
            "temporal": asdict(self.temporal),
            "feature_module": self.feature_module,
            "feature_class": self.feature_class,
            "feature_params": self.feature_params,
            "target_mode": self.target_mode,
        }

    def with_overrides(self, updates: Dict[str, object]) -> "TrainConfig":
        merged = _deep_merge(self.to_dict(), updates)
        return TrainConfig.from_dict(merged)


def _deep_merge(base: Dict[str, object], new_data: Dict[str, object]) -> Dict[str, object]:
    result = dict(base)
    for key, value in new_data.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------
# Датасет для временной модели
# --------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    indices: индексы сэмплов (строк) в глобальных массивах (X, y, seqs, steps).
    steps[sample_idx] трактуется как ИНДЕКС ШАГА, состояние которого мы предсказываем.
    Контекстное окно состоит из шагов [target_step - window, ..., target_step - 1].
    """
    def __init__(
        self,
        indices: np.ndarray,
        seqs: np.ndarray,
        steps: np.ndarray,
        states_by_seq: Dict[int, np.ndarray],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        window: int,
        target_mode: str = "level",
    ):
        self.indices = indices
        self.seqs = seqs
        self.steps = steps
        self.states_by_seq = states_by_seq
        self.state_mean = state_mean
        self.state_std_safe = np.where(state_std == 0.0, 1.0, state_std)
        self.target_mean = target_mean
        self.target_std_safe = np.where(target_std == 0.0, 1.0, target_std)
        self.window = window
        self.target_mode = target_mode.lower()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        seq_ix = int(self.seqs[sample_idx])
        target_step = int(self.steps[sample_idx])
        states = self.states_by_seq[seq_ix]

        # target_step — индекс шага, состояние которого хотим предсказать.
        # Контекст берётся строго ДО этого шага: [start_idx : target_step)
        end_idx = target_step
        start_idx = max(0, end_idx - self.window)
        window_states = states[start_idx:end_idx]

        if window_states.shape[0] == 0:
            # Крайний случай, если истории нет вообще
            window_states = np.zeros((1, states.shape[1]), dtype=np.float32)

        # Padding: повторяем последний доступный элемент (edge padding по "хвосту" истории)
        if window_states.shape[0] < self.window:
            pad = np.repeat(window_states[-1:], self.window - window_states.shape[0], axis=0)
            window_states = np.concatenate([pad, window_states], axis=0)

        # Нормализация состояний
        window_states = (window_states - self.state_mean) / self.state_std_safe

        # Формирование таргета
        if self.target_mode == "delta":
            if target_step == 0:
                raw_target = np.zeros(states.shape[1], dtype=np.float32)
            else:
                raw_target = states[target_step] - states[target_step - 1]
        else:
            raw_target = states[target_step]

        target = (raw_target - self.target_mean) / self.target_std_safe

        return (
            torch.from_numpy(window_states.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


# --------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------

def create_tensor_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_predictions(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    R² считаем всегда по уровням (восстановленным значениям).
    """
    scores = {
        "mean_r2": float(r2_score(target, pred, multioutput="uniform_average")),
    }
    return scores


# --------------------------------------------------------------------
# Обучение моделей
# --------------------------------------------------------------------

def train_feature_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    stats: DataStats,
    device: torch.device,
    config: MLPConfig,
    *,
    num_epochs: int,
    patience_limit: int,
    weight_decay: float,
    save_model_path: Optional[Path] = None,
    save_history_path: Optional[Path] = None,
) -> Tuple[FeatureMLP, np.ndarray]:
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = FeatureMLP(
        input_dim,
        config.hidden_dims,
        output_dim,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = torch.nn.MSELoss()

    train_loader = create_tensor_loader(X_train, y_train, batch_size=config.batch_size, shuffle=True)
    val_loader = create_tensor_loader(X_val, y_val, batch_size=config.val_batch_size, shuffle=False)

    best_state = None
    best_loss = float("inf")
    patience = 0
    history: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        train_loss = running_loss / len(train_loader.dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        if patience >= patience_limit:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
    if save_history_path:
        with open(save_history_path, "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

    # Предсказания на валидации (в пространстве y: delta или level)
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            val_preds.append(pred)
    val_pred_array = np.concatenate(val_preds, axis=0)

    # Денормализация из стандартизованного y в исходное пространство y (delta/level)
    val_pred_array = val_pred_array * stats.target_std + stats.target_mean

    return model, val_pred_array


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    *,
    alpha: float,
    save_path: Optional[Path] = None,
) -> Tuple[Ridge, np.ndarray]:
    """
    y_train и предсказания ridge находятся в том же пространстве, что и y_train:
    - если target_mode == "level" -> уровни
    - если target_mode == "delta" -> дельты
    """
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_train, y_train)
    val_pred = ridge.predict(X_val)

    if save_path is not None:
        np.savez(
            save_path,
            coef=ridge.coef_.astype(np.float32),
            intercept=ridge.intercept_.astype(np.float32),
        )

    return ridge, val_pred


def _create_temporal_model(config: TemporalConfig, state_dim: int, max_window: int) -> torch.nn.Module:
    model_type = config.model_type.lower()
    if model_type == "conv_gru":
        return TemporalConvGRU(state_dim, config.conv_channels, config.hidden_dim)
    if model_type == "dilated_cnn":
        return TemporalDilatedCNN(
            state_dim, config.cnn_channels, config.cnn_layers, config.cnn_kernel_size, config.cnn_dropout
        )
    if model_type == "deep_gru":
        return TemporalDeepGRU(
            state_dim,
            hidden_dim=config.deep_gru_hidden,
            num_layers=config.deep_gru_layers,
            dropout=config.deep_gru_dropout
        )    
    if model_type == "transformer":
        return TemporalTransformer(
            state_dim,
            config.transformer_d_model,
            config.transformer_heads,
            config.transformer_layers,
            config.transformer_ff,
            config.transformer_dropout,
            max(config.transformer_max_window, max_window),
            config.transformer_pe_type,
        )
    if model_type == "transformer_v2":
        return TemporalTransformerV2(
            state_dim,
            d_model=config.transformer_d_model,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff,
            dropout=config.transformer_dropout,
            max_window=max(config.transformer_max_window, max_window),
        )    
    if model_type == "resnet":
        return TemporalResNet(
            state_dim, 
            channels=config.resnet_channels, 
            num_blocks=config.resnet_blocks, 
            kernel_size=config.resnet_kernel_size,
            dropout=config.resnet_dropout
        )    
    if model_type == "mamba":
        return TemporalMamba(
            state_dim,
            d_model=config.mamba_d_model,
            num_layers=config.mamba_layers,
            dropout=config.mamba_dropout,
            expand=config.mamba_expand
        )
    raise ValueError(f"Unknown temporal model type: {config.model_type}")


def _temporal_model_filename(model_type: str) -> str:
    model_type = model_type.lower()
    if model_type == "conv_gru":
        return "temporal_conv_gru.pt"
    if model_type == "dilated_cnn":
        return "temporal_dcnn.pt"
    if model_type == "transformer":
        return "temporal_transformer.pt"
    if model_type == "transformer_v2":
        return "temporal_transformer_v2.pt"    
    if model_type == "resnet":
        return "temporal_resnet.pt" 
    if model_type == "mamba":
        return "temporal_mamba.pt"   
    if model_type == "deep_gru":
        return "temporal_deep_gru.pt"    
    return "temporal_model.pt"


def train_temporal_model(
    dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    stats: DataStats,
    device: torch.device,
    config: TemporalConfig,
    *,
    num_epochs: int,
    patience_limit: int,
    weight_decay: float,
    save_model_path: Optional[Path] = None,
) -> Tuple[torch.nn.Module, np.ndarray]:
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, drop_last=False)

    model = _create_temporal_model(config, stats.state_mean.shape[0], dataset.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=weight_decay)

    warmup_epochs = max(0, int(getattr(config, "warmup_epochs", 0)))
    warmup_epochs = min(warmup_epochs, num_epochs)
    if warmup_epochs > 0:
        warmup_iters = min(warmup_epochs, num_epochs)
        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_iters,
        )
        if warmup_iters == num_epochs:
            scheduler = scheduler_warmup
        else:
            scheduler_cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_epochs - warmup_iters),
                eta_min=0.0,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_iters],
            )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs), eta_min=0.0)

    criterion = torch.nn.MSELoss()

    # Опционально включаем EMA весов
    use_ema = getattr(config, "use_ema", False)
    ema_decay = float(getattr(config, "ema_decay", 0.995))
    ema = EMAWrapper(model, decay=ema_decay) if use_ema else None

    best_state = None
    best_loss = float("inf")
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Обновляем EMA после каждого батча
            if ema is not None:
                ema.update(model)

        scheduler.step()

        model.eval()
        # Для валидации используем EMA веса
        if ema is not None:
            ema.apply_shadow(model)

        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        # Восстанавливаем оригинальные веса для продолжения обучения
        if ema is not None:
            ema.restore(model)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            # Сохраняем EMA веса как лучшие
            if ema is not None:
                ema.apply_shadow(model)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                ema.restore(model)
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= patience_limit:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_model_path:
        torch.save(model.state_dict(), save_model_path)

    # Предсказания на валидации (в пространстве y: delta или level)
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            val_preds.append(pred)

    val_pred_array = np.concatenate(val_preds, axis=0)
    # Денормализация из стандартизованного y в исходное пространство y (delta/level)
    val_pred_array = val_pred_array * stats.target_std + stats.target_mean

    return model, val_pred_array


# --------------------------------------------------------------------
# Ансамбль
# --------------------------------------------------------------------

def compute_ensemble_weights(
    base_predictions: List[np.ndarray],
    target: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    base_predictions: список массивов (n_samples, n_features) — в уровнях.
    target: (n_samples, n_features) — целевые уровни.
    """
    state_dim = target.shape[1]
    n_models = len(base_predictions)
    weights = np.zeros((state_dim, n_models), dtype=np.float32)
    bias = np.zeros(state_dim, dtype=np.float32)

    for feature_idx in range(state_dim):
        X = np.column_stack([pred[:, feature_idx] for pred in base_predictions])
        y = target[:, feature_idx]
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y)
        weights[feature_idx] = model.coef_.astype(np.float32)
        bias[feature_idx] = float(model.intercept_)

    return weights, bias


def apply_ensemble(
    base_predictions: List[np.ndarray],
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """
    base_predictions: список [ (n_samples, n_features) ] в уровнях.
    weights: (n_features, n_models)
    bias: (n_features,)
    """
    stacked = np.stack(base_predictions, axis=2)  # (n_samples, n_features, n_models)
    return np.sum(stacked * weights[None, :, :], axis=2) + bias  # (n_samples, n_features)


# --------------------------------------------------------------------
# Обучение на одном сплите (фолд или финальный)
# --------------------------------------------------------------------

def train_on_split(
    config: TrainConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_train_orig: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seqs: np.ndarray,
    steps: np.ndarray,
    states_by_seq: Dict[int, np.ndarray],
    stats: DataStats,
    device: torch.device,
    base_states_val: np.ndarray,
    is_final: bool = False,
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:

    # 1. MLP
    logger.info("Training MLP...")
    mlp_model, mlp_pred = train_feature_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        stats,
        device,
        config.mlp,
        num_epochs=config.mlp.final_epochs if is_final else config.mlp.fold_epochs,
        patience_limit=config.mlp.final_patience if is_final else config.mlp.fold_patience,
        weight_decay=config.mlp.final_weight_decay if is_final else config.mlp.fold_weight_decay,
        save_model_path=(artifacts_dir / "feature_mlp.pt") if is_final and artifacts_dir else None,
        save_history_path=(artifacts_dir / "feature_mlp_history.json") if is_final and artifacts_dir else None,
    )

    # 2. Ridge
    logger.info("Training Ridge...")
    ridge_model, ridge_pred = train_ridge(
        X_train,
        y_train_orig,
        X_val,
        alpha=config.ridge.final_alpha if is_final else config.ridge.fold_alpha,
        save_path=(artifacts_dir / "ridge_weights.npz") if is_final and artifacts_dir else None,
    )

    # 3. Temporal
    logger.info(f"Training Temporal model ({config.temporal.model_type})...")
    seq_train_ds = SequenceDataset(
        train_indices,
        seqs,
        steps,
        states_by_seq,
        stats.state_mean,
        stats.state_std,
        stats.target_mean,
        stats.target_std,
        config.sequence_window,
        config.target_mode,
    )
    seq_val_ds = SequenceDataset(
        val_indices,
        seqs,
        steps,
        states_by_seq,
        stats.state_mean,
        stats.state_std,
        stats.target_mean,
        stats.target_std,
        config.sequence_window,
        config.target_mode,
    )

    temporal_filename = _temporal_model_filename(config.temporal.model_type)
    temporal_model, temporal_pred = train_temporal_model(
        seq_train_ds,
        seq_val_ds,
        stats,
        device,
        config.temporal,
        num_epochs=config.temporal.final_epochs if is_final else config.temporal.fold_epochs,
        patience_limit=config.temporal.final_patience if is_final else config.temporal.fold_patience,
        weight_decay=config.temporal.final_weight_decay if is_final else config.temporal.fold_weight_decay,
        save_model_path=(artifacts_dir / temporal_filename) if is_final and artifacts_dir else None,
    )

    # Восстановление уровней, если режим delta (все три модели дают дельты)
    if config.target_mode == "delta":
        mlp_pred = mlp_pred + base_states_val
        ridge_pred = ridge_pred + base_states_val
        temporal_pred = temporal_pred + base_states_val

    # Возвращаем предсказания в УРОВНЯХ
    return {
        "mlp": mlp_pred,
        "ridge": ridge_pred,
        "temporal": temporal_pred,
        "models": (mlp_model, ridge_model, temporal_model),
    }


# --------------------------------------------------------------------
# Основной пайплайн обучения
# --------------------------------------------------------------------

def run_training(
    config: TrainConfig,
    dataset_path: Path,
    artifacts_dir: Path,
    *,
    save_artifacts: bool = True,
    run_final: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    if not verbose:
        logger.setLevel(logging.WARNING)

    dataset_path = Path(dataset_path)
    artifacts_dir = Path(artifacts_dir)
    if save_artifacts:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    _set_random_seeds(config.random_seed)

    logger.info(f"Loading dataset from {dataset_path}...")
    df = load_dataset(dataset_path)

    logger.info("Building training arrays...")
    feature_module = importlib.import_module(config.feature_module or "src.features")
    feature_generator_cls = getattr(feature_module, config.feature_class)

    # X: фичи, y: уровни (изначально), seqs, steps — индексы шагов, для которых предсказываем состояние
    X, y, seqs, steps = build_training_arrays(
        df,
        state_dim=df.shape[1] - 3,
        feature_generator_cls=feature_generator_cls,
        feature_kwargs=config.feature_params,
    )

    # Подготовка состояний для временных моделей
    state_columns = df.columns[3:]
    grouped = df.groupby("seq_ix", sort=False)
    states_by_seq = {
        int(seq_ix): group[state_columns].to_numpy(dtype=np.float32)
        for seq_ix, group in grouped
    }

    # Базовые состояния (предыдущий шаг) для режима delta и восстановления уровней
    base_states = np.zeros_like(y, dtype=np.float32)
    for idx in range(len(y)):
        seq_ix = int(seqs[idx])
        target_step = int(steps[idx])
        seq_states = states_by_seq[seq_ix]
        if target_step <= 0:
            base_states[idx] = seq_states[0]
        else:
            base_states[idx] = seq_states[target_step - 1]

    # Сохраняем оригинальные уровни для всех сэмплов
    y_levels_orig = y.copy()

    # Преобразование в дельты при необходимости (только для обучения моделей)
    if config.target_mode.lower() == "delta":
        y = y - base_states
    elif config.target_mode.lower() != "level":
        raise ValueError(f"Unknown target_mode: {config.target_mode}")

    # Статистика
    feature_mean = X.mean(axis=0)
    feature_std_raw = X.std(axis=0)
    feature_std = np.where(feature_std_raw == 0.0, 1.0, feature_std_raw + 1e-6)

    target_mean = y.mean(axis=0)
    target_std_raw = y.std(axis=0)
    target_std = np.where(target_std_raw == 0.0, 1.0, target_std_raw + 1e-6)

    state_mean = df[state_columns].mean(axis=0).to_numpy(dtype=np.float32)
    state_std_raw = df[state_columns].std(axis=0).to_numpy(dtype=np.float32) + 1e-6
    state_std = np.where(state_std_raw == 0.0, 1.0, state_std_raw)

    stats = DataStats(
        feature_mean.astype(np.float32),
        feature_std.astype(np.float32),
        target_mean.astype(np.float32),
        target_std.astype(np.float32),
        state_mean.astype(np.float32),
        state_std.astype(np.float32),
    )

    # Скейлинг
    X_scaled = (X - stats.feature_mean) / stats.feature_std
    y_scaled = (y - stats.target_mean) / stats.target_std

    # Разбиение на фолды по seq_ix (фиксированное)
    unique_seq = np.unique(seqs)
    split_rng = np.random.default_rng(FIXED_SPLIT_SEED)
    split_rng.shuffle(unique_seq)
    fold_splits = np.array_split(unique_seq, config.n_folds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Порядок моделей в ансамбле
    temporal_name = config.temporal.model_type
    model_names = ["mlp", "ridge", temporal_name]
    oof_predictions_dict: Dict[str, List[np.ndarray]] = {name: [] for name in model_names}
    oof_targets: List[np.ndarray] = []
    oof_indices: List[np.ndarray] = []
    fold_summaries: List[Dict[str, float]] = []

    logger.info(f"Starting K-fold training: {config.n_folds} folds (split seed: {FIXED_SPLIT_SEED})")

    for fold_idx, val_seq_array in enumerate(fold_splits):
        logger.info(f"[FOLD {fold_idx + 1}/{config.n_folds}]")

        val_seq_set = set(val_seq_array.tolist())
        train_seq_set = set(unique_seq.tolist()) - val_seq_set

        train_mask = np.isin(seqs, list(train_seq_set))
        val_mask = np.isin(seqs, list(val_seq_set))

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        results = train_on_split(
            config,
            X_scaled[train_mask],
            y_scaled[train_mask],
            X_scaled[val_mask],
            y_scaled[val_mask],
            y[train_mask],  # y в пространстве обучения (level/delta)
            train_indices,
            val_indices,
            seqs,
            steps,
            states_by_seq,
            stats,
            device,
            base_states[val_mask],  # для восстановления уровней в delta-режиме
        )

        # Предсказания уже в УРОВНЯХ
        oof_predictions_dict["mlp"].append(results["mlp"])
        oof_predictions_dict["ridge"].append(results["ridge"])
        oof_predictions_dict[temporal_name].append(results["temporal"])

        val_target_levels = y_levels_orig[val_mask]  # таргеты в уровнях
        oof_targets.append(val_target_levels)
        oof_indices.append(val_indices.astype(np.int32))

        # Метрики на фолде
        fold_metrics = {
            "fold": fold_idx,
            "mlp_r2": evaluate_predictions(results["mlp"], val_target_levels)["mean_r2"],
            "ridge_r2": evaluate_predictions(results["ridge"], val_target_levels)["mean_r2"],
            "temporal_r2": evaluate_predictions(results["temporal"], val_target_levels)["mean_r2"],
        }
        fold_summaries.append(fold_metrics)
        logger.info(f"Fold metrics: {fold_metrics}")

        # Очистка памяти
        del results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Сборка OOF
    oof_target_concat = np.concatenate(oof_targets, axis=0)
    oof_indices_concat = np.concatenate(oof_indices, axis=0)

    base_predictions_list: List[np.ndarray] = []
    oof_metrics: Dict[str, Dict[str, float]] = {}

    for name in model_names:
        pred_concat = np.concatenate(oof_predictions_dict[name], axis=0)
        base_predictions_list.append(pred_concat)
        oof_metrics[name] = evaluate_predictions(pred_concat, oof_target_concat)
        logger.info(f"OOF {name} R2: {oof_metrics[name]['mean_r2']:.4f}")

    # Ансамбль по уровням
    ensemble_weights, ensemble_bias = compute_ensemble_weights(
        base_predictions_list, oof_target_concat, alpha=config.ensemble_alpha
    )
    oof_ensemble_pred = apply_ensemble(base_predictions_list, ensemble_weights, ensemble_bias)
    oof_metrics["ensemble"] = evaluate_predictions(oof_ensemble_pred, oof_target_concat)
    logger.info(f"OOF Ensemble R2: {oof_metrics['ensemble']['mean_r2']:.4f}")

    # Итоговая информация о данных
    final_data_info: Dict[str, Optional[int]] = {
        "total_samples": int(len(X)),
        "total_sequences": int(len(unique_seq)),
        "train_samples_final": None,
        "val_samples_final": None,
        "train_sequences_final": None,
        "val_sequences_final": None,
    }

    # Финальное обучение
    final_metrics: Optional[Dict[str, Dict[str, float]]] = None

    if run_final:
        final_split_idx = max(1, int(len(unique_seq) * (1 - config.final_val_ratio)))
        final_train_seq = set(unique_seq[:final_split_idx])
        final_val_seq = set(unique_seq[final_split_idx:])

        logger.info(
            f"Final training split: {len(final_train_seq)} train seqs, {len(final_val_seq)} val seqs"
        )

        train_mask = np.isin(seqs, list(final_train_seq))
        val_mask = np.isin(seqs, list(final_val_seq))

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        results_final = train_on_split(
            config,
            X_scaled[train_mask],
            y_scaled[train_mask],
            X_scaled[val_mask],
            y_scaled[val_mask],
            y[train_mask],
            train_indices,
            val_indices,
            seqs,
            steps,
            states_by_seq,
            stats,
            device,
            base_states[val_mask],
            is_final=True,
            artifacts_dir=artifacts_dir if save_artifacts else None,
        )
        # --- FIX: Переименовываем 'temporal' в реальное имя модели ---
        results_final[temporal_name] = results_final.pop("temporal")
        # -------------------------------------------------------------

        val_target_levels = y_levels_orig[val_mask]

        # Ансамбль на финальной валидации
        final_preds_list = [results_final[name] for name in model_names]
        final_ensemble_pred = apply_ensemble(final_preds_list, ensemble_weights, ensemble_bias)

        final_metrics = {
            name: evaluate_predictions(results_final[name], val_target_levels)
            for name in model_names
        }
        final_metrics["ensemble"] = evaluate_predictions(final_ensemble_pred, val_target_levels)

        final_data_info.update(
            {
                "train_samples_final": int(train_mask.sum()),
                "val_samples_final": int(val_mask.sum()),
                "train_sequences_final": len(final_train_seq),
                "val_sequences_final": len(final_val_seq),
            }
        )

        # Сохранение артефактов
        if save_artifacts:
            stats.save(artifacts_dir / "data_stats.npz")
            np.savez(
                artifacts_dir / "ensemble_weights.npz",
                weights=ensemble_weights.astype(np.float32),
                bias=ensemble_bias.astype(np.float32),
                model_names=np.array(model_names, dtype=object),
            )

            # Метаданные временной модели + фичей
            temporal_meta = {
                "model_type": config.temporal.model_type,
                "sequence_window": config.sequence_window,
                "target_mode": config.target_mode,
                "feature_module": (config.feature_module or "src.features"),
                "feature_class": config.feature_class,
                "feature_params": config.feature_params,
            }
            temporal_meta.update(
                {k: v for k, v in asdict(config.temporal).items() if k not in ["model_type"]}
            )
            with open(artifacts_dir / "temporal_model_meta.json", "w", encoding="utf-8") as fp:
                json.dump(temporal_meta, fp, ensure_ascii=False, indent=2)

    # Суммарный отчёт
    summary = {
        "config": config.to_dict(),
        "folds": fold_summaries,
        "oof": oof_metrics,
        "final_validation": final_metrics,
        "data_info": final_data_info,
    }

    if save_artifacts:
        # Сохраняем OOF-предсказания (для возможного стэкинга/анализа)
        oof_arrays = {name: base_predictions_list[i].astype(np.float32)
                      for i, name in enumerate(model_names)}
        oof_arrays["target"] = oof_target_concat.astype(np.float32)
        oof_arrays["indices"] = oof_indices_concat.astype(np.int32)
        np.savez(artifacts_dir / "oof_predictions.npz", **oof_arrays)

        if run_final:
            with open(artifacts_dir / "training_summary.json", "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)

    return {
        "summary": summary,
        "artifacts_dir": str(artifacts_dir) if save_artifacts and run_final else None,
    }


# --------------------------------------------------------------------
# CLI и main
# --------------------------------------------------------------------

def _load_config_from_args(args: argparse.Namespace) -> TrainConfig:
    base = TrainConfig()
    if args.config_path:
        with open(args.config_path, "r", encoding="utf-8") as fp:
            base = base.with_overrides(json.load(fp))
    if args.override_json:
        base = base.with_overrides(json.loads(args.override_json))
    if args.set_seed is not None:
        base = base.with_overrides({"random_seed": args.set_seed})
    return base


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ensemble models")
    parser.add_argument("--dataset", type=str, default=str(Path("datasets") / "train.parquet"))
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--config-path", type=str, default=None, help="Path to JSON config overrides")
    parser.add_argument("--override-json", type=str, default=None, help="Inline JSON overrides")
    parser.add_argument("--set-seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--skip-final", action="store_true", help="Skip final training stage")
    parser.add_argument("--no-save", action="store_true", help="Do not save artifacts")
    parser.add_argument("--summary-out", type=str, default=None, help="Path to dump training summary JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = _load_config_from_args(args)

    result = run_training(
        config,
        Path(args.dataset),
        Path(args.artifacts_dir),
        save_artifacts=not args.no_save,
        run_final=not args.skip_final,
        verbose=not args.quiet,
    )

    summary = result["summary"]
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

    if not args.quiet and not args.summary_out:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()