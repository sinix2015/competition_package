import argparse
import importlib
import json
import logging
import math
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_prep import build_training_arrays, load_dataset
from src.models import (
    TemporalTransformer,
    TemporalTransformerV2,
    TemporalConvGRU,
    TemporalDilatedCNN,
    TemporalResNet,
    TemporalMamba,
    TemporalDeepGRU,
)
from src.ema_trainer import EMAWrapper


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

FIXED_SPLIT_SEED = 42


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
class TemporalConfig:
    model_type: str = "transformer"
    fold_epochs: int = 70
    fold_patience: int = 12
    fold_weight_decay: float = 4e-4
    final_epochs: int = 120
    final_patience: int = 20
    final_weight_decay: float = 2e-4
    batch_size: int = 256
    val_batch_size: int = 512
    lr: float = 1e-3
    transformer_d_model: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 6
    transformer_ff: int = 512
    transformer_dropout: float = 0.1
    transformer_max_window: int = 64
    transformer_pe_type: str = "learned"
    warmup_epochs: int = 5
    use_ema: bool = False
    ema_decay: float = 0.995
    conv_channels: int = 64
    hidden_dim: int = 64
    cnn_channels: int = 96
    cnn_layers: int = 4
    cnn_kernel_size: int = 3
    cnn_dropout: float = 0.1
    resnet_channels: int = 128
    resnet_blocks: int = 4
    resnet_dropout: float = 0.1
    resnet_kernel_size: int = 3
    mamba_d_model: int = 128
    mamba_layers: int = 2
    mamba_dropout: float = 0.1
    mamba_expand: int = 2
    deep_gru_hidden: int = 128
    deep_gru_layers: int = 4
    deep_gru_dropout: float = 0.1


@dataclass
class TrainConfig:
    n_folds: int = 5
    final_val_ratio: float = 0.1
    sequence_window: int = 32
    ensemble_alpha: float = 12.0
    random_seed: int = 42
    target_mode: str = "level"
    feature_module: Optional[str] = None
    feature_class: str = "FeatureGenerator"
    feature_params: Dict[str, Any] = field(default_factory=dict)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        payload = dict(data)
        temporal_data = payload.pop("temporal", None)
        payload.pop("mlp", None)
        payload.pop("ridge", None)
        payload.pop("full_refit", None)
        known: Dict[str, Any] = {}
        for key in list(payload.keys()):
            if key in cls.__dataclass_fields__:
                known[key] = payload.pop(key)
        config = cls(**known)
        if temporal_data is not None:
            config.temporal = TemporalConfig(**temporal_data)
        if payload:
            extra = dict(config.feature_params)
            extra["__ignored__"] = payload
            config.feature_params = extra
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "final_val_ratio": self.final_val_ratio,
            "sequence_window": self.sequence_window,
            "ensemble_alpha": self.ensemble_alpha,
            "random_seed": self.random_seed,
            "target_mode": self.target_mode,
            "feature_module": self.feature_module,
            "feature_class": self.feature_class,
            "feature_params": self.feature_params,
            "temporal": asdict(self.temporal),
        }

    def with_overrides(self, updates: Dict[str, Any]) -> "TrainConfig":
        merged = _deep_merge(self.to_dict(), updates)
        return TrainConfig.from_dict(merged)


def _deep_merge(base: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
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


class SequenceDataset(Dataset):
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
        target_mode: str,
    ) -> None:
        self.indices = indices.astype(np.int32)
        self.seqs = seqs
        self.steps = steps
        self.states_by_seq = states_by_seq
        self.state_mean = state_mean
        self.state_std_safe = np.where(state_std == 0.0, 1.0, state_std)
        self.target_mean = target_mean
        self.target_std_safe = np.where(target_std == 0.0, 1.0, target_std)
        self.window = int(window)
        self.target_mode = target_mode.lower()

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        seq_ix = int(self.seqs[sample_idx])
        target_step = int(self.steps[sample_idx])
        states = self.states_by_seq[seq_ix]

        end_idx = target_step
        start_idx = max(0, end_idx - self.window)
        window_states = states[start_idx:end_idx]

        if window_states.shape[0] == 0:
            window_states = np.zeros((1, states.shape[1]), dtype=np.float32)

        if window_states.shape[0] < self.window:
            pad = np.repeat(window_states[-1:], self.window - window_states.shape[0], axis=0)
            window_states = np.concatenate([pad, window_states], axis=0)

        window_states = (window_states - self.state_mean) / self.state_std_safe

        if self.target_mode == "delta":
            if target_step == 0:
                raw_target = np.zeros(states.shape[1], dtype=np.float32)
            else:
                prev_state = states[target_step - 1]
                raw_target = states[target_step] - prev_state
        else:
            raw_target = states[target_step]

        target = (raw_target - self.target_mean) / self.target_std_safe

        return (
            torch.from_numpy(window_states.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


def evaluate_predictions(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    score = float(r2_score(target, pred, multioutput="uniform_average"))
    return {"mean_r2": score}


def _temporal_model_filename(model_type: str) -> str:
    mt = model_type.lower()
    if mt == "conv_gru":
        return "temporal_conv_gru.pt"
    if mt == "dilated_cnn":
        return "temporal_dcnn.pt"
    if mt == "transformer":
        return "temporal_transformer.pt"
    if mt == "transformer_v2":
        return "temporal_transformer_v2.pt"
    if mt == "resnet":
        return "temporal_resnet.pt"
    if mt == "mamba":
        return "temporal_mamba.pt"
    if mt == "deep_gru":
        return "temporal_deep_gru.pt"
    return "temporal_model.pt"


def _create_temporal_model(config: TemporalConfig, state_dim: int, max_window: int) -> torch.nn.Module:
    """
    Creates temporal model based on config.model_type.
    Supports: transformer, transformer_v2, resnet, mamba, deep_gru, conv_gru, dilated_cnn
    """
    model_type = config.model_type.lower()
    
    if model_type == "transformer":
        return TemporalTransformer(
            state_dim=state_dim,
            d_model=config.transformer_d_model,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff,
            dropout=config.transformer_dropout,
            max_window=max(config.transformer_max_window, max_window),
            pe_type=config.transformer_pe_type,
        )
    elif model_type == "transformer_v2":
        return TemporalTransformerV2(
            state_dim=state_dim,
            d_model=config.transformer_d_model,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff,
            dropout=config.transformer_dropout,
            max_window=max(config.transformer_max_window, max_window),
        )
    elif model_type == "resnet":
        return TemporalResNet(
            state_dim=state_dim,
            channels=config.resnet_channels,
            num_blocks=config.resnet_blocks,
            kernel_size=config.resnet_kernel_size,
            dropout=config.resnet_dropout,
        )
    elif model_type == "mamba":
        return TemporalMamba(
            state_dim=state_dim,
            d_model=config.mamba_d_model,
            num_layers=config.mamba_layers,
            dropout=config.mamba_dropout,
            expand=config.mamba_expand,
        )
    elif model_type == "deep_gru":
        return TemporalDeepGRU(
            state_dim=state_dim,
            hidden_dim=config.deep_gru_hidden,
            num_layers=config.deep_gru_layers,
            dropout=config.deep_gru_dropout,
        )
    elif model_type == "conv_gru":
        return TemporalConvGRU(
            state_dim=state_dim,
            conv_channels=config.conv_channels,
            hidden_dim=config.hidden_dim,
        )
    elif model_type == "dilated_cnn":
        return TemporalDilatedCNN(
            state_dim=state_dim,
            channels=config.cnn_channels,
            num_layers=config.cnn_layers,
            kernel_size=config.cnn_kernel_size,
            dropout=config.cnn_dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


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
    log_prefix: str = "",
) -> Tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, drop_last=False)

    model = _create_temporal_model(config, stats.state_mean.shape[0], dataset.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=weight_decay)

    # --- НАСТРОЙКА WARMUP + COSINE ---
    # Берем warmup_epochs из конфига (по умолчанию 5)
    warmup_epochs = getattr(config, "warmup_epochs", 5)
    
    # 1. Фаза разогрева: линейно от 1% до 100% LR
    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # 2. Фаза затухания: косинусное затухание до конца обучения
    scheduler_cosine = CosineAnnealingLR(
        optimizer, 
        T_max=max(1, num_epochs - warmup_epochs),
        eta_min=0.0 # LR упадет до 0 в конце
    )
    
    # Объединяем их: сначала warmup, потом cosine
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs]
    )
    # ---------------------------------

    criterion = torch.nn.MSELoss()

    # Опционально включаем EMA
    use_ema = getattr(config, "use_ema", False)
    ema_decay = float(getattr(config, "ema_decay", 0.995))
    ema = EMAWrapper(model, decay=ema_decay) if use_ema else None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    patience = 0
    log_interval = max(1, num_epochs // 10)
    
    model.train()

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Обновляем EMA после каждого батча
            if ema is not None:
                ema.update(model)
        
        # Обновляем LR после эпохи
        scheduler.step()

        model.eval()
        # Валидация с EMA весами
        if ema is not None:
            ema.apply_shadow(model)

        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        # Восстанавливаем оригинальные веса
        if ema is not None:
            ema.restore(model)

        improved = val_loss < best_loss - 1e-6
        if improved:
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

        should_log = (
            epoch == 0
            or (epoch + 1) % log_interval == 0
            or improved
            or patience >= patience_limit
            or epoch + 1 == num_epochs
        )
        
        if should_log:
            # Получаем текущий LR для логов
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "%s epoch %d/%d val_loss=%.6f lr=%.6f",
                log_prefix,
                epoch + 1,
                num_epochs,
                val_loss,
                current_lr,
            )

        if patience >= patience_limit:
            logger.info("%s early stop at epoch %d", log_prefix, epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    val_preds: List[np.ndarray] = []
    val_targets: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            val_preds.append(pred)
            val_targets.append(yb.cpu().numpy())

    val_pred_array = np.concatenate(val_preds, axis=0)
    val_target_array = np.concatenate(val_targets, axis=0)

    val_pred_array = val_pred_array * stats.target_std + stats.target_mean
    val_target_array = val_target_array * stats.target_std + stats.target_mean

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    return model, val_pred_array, val_target_array


def _restore_levels(
    pred: np.ndarray,
    target: np.ndarray,
    base_states: np.ndarray,
    *,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_mode.lower() == "delta":
        return pred + base_states, target + base_states
    return pred, target


def _train_temporal_on_indices(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seqs: np.ndarray,
    steps: np.ndarray,
    states_by_seq: Dict[int, np.ndarray],
    stats: DataStats,
    device: torch.device,
    config: TrainConfig,
    *,
    num_epochs: int,
    patience: int,
    weight_decay: float,
    log_prefix: str,
    save_model_path: Optional[Path] = None,
) -> Tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    seq_train = SequenceDataset(
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
    seq_val = SequenceDataset(
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

    model, val_pred_raw, val_target_raw = train_temporal_model(
        seq_train,
        seq_val,
        stats,
        device,
        config.temporal,
        num_epochs=num_epochs,
        patience_limit=patience,
        weight_decay=weight_decay,
        save_model_path=save_model_path,
        log_prefix=log_prefix,
    )
    return model, val_pred_raw, val_target_raw


def run_training(
    config: TrainConfig,
    dataset_path: Path,
    artifacts_dir: Path,
    *,
    save_artifacts: bool = True,
    run_final: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    if not verbose:
        logger.setLevel(logging.WARNING)

    dataset_path = Path(dataset_path)
    artifacts_dir = Path(artifacts_dir)
    if save_artifacts:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    _set_random_seeds(config.random_seed)

    logger.info("Loading dataset from %s", dataset_path)
    df = load_dataset(dataset_path)

    logger.info("Building training arrays")
    feature_module_name = config.feature_module or "src.features"
    feature_module = importlib.import_module(feature_module_name)
    feature_class = getattr(feature_module, config.feature_class)
    X, y, seqs, steps = build_training_arrays(
        df,
        state_dim=df.shape[1] - 3,
        feature_generator_cls=feature_class,
        feature_kwargs=config.feature_params,
    )

    state_columns = df.columns[3:]
    grouped = df.groupby("seq_ix", sort=False)
    states_by_seq = {
        int(seq_ix): group[state_columns].to_numpy(dtype=np.float32)
        for seq_ix, group in grouped
    }

    base_states = np.zeros_like(y, dtype=np.float32)
    for idx in range(len(y)):
        seq_ix = int(seqs[idx])
        target_step = int(steps[idx])
        seq_states = states_by_seq[seq_ix]
        if target_step <= 0:
            base_states[idx] = seq_states[0]
        else:
            base_states[idx] = seq_states[target_step - 1]

    y_levels_orig = y.copy()
    target_mode = config.target_mode.lower()
    if target_mode == "delta":
        y = y - base_states
    elif target_mode != "level":
        raise ValueError(f"Unsupported target_mode: {config.target_mode}")

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

    unique_seq = np.unique(seqs)
    rng = np.random.default_rng(FIXED_SPLIT_SEED)
    shuffled_seq = unique_seq.copy()
    rng.shuffle(shuffled_seq)
    fold_splits = np.array_split(shuffled_seq, config.n_folds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    oof_predictions: List[np.ndarray] = []
    oof_targets: List[np.ndarray] = []
    oof_indices: List[np.ndarray] = []
    fold_summaries: List[Dict[str, Any]] = []

    logger.info("Starting %d-fold training", config.n_folds)
    for fold_idx, val_seq_array in enumerate(fold_splits):
        fold_label = f"[FOLD {fold_idx + 1}/{config.n_folds}]"
        logger.info("%s preparing split", fold_label)

        val_seq_set = set(val_seq_array.tolist())
        train_seq_set = set(unique_seq.tolist()) - val_seq_set

        train_mask = np.isin(seqs, list(train_seq_set))
        val_mask = np.isin(seqs, list(val_seq_set))

        train_indices = np.where(train_mask)[0].astype(np.int32)
        val_indices = np.where(val_mask)[0].astype(np.int32)

        if len(val_indices) == 0:
            raise ValueError("Validation indices are empty")
        if len(train_indices) == 0:
            raise ValueError("Training indices are empty")

        model_label = config.temporal.model_type.lower()
        model, val_pred_raw, val_target_raw = _train_temporal_on_indices(
            train_indices,
            val_indices,
            seqs,
            steps,
            states_by_seq,
            stats,
            device,
            config,
            num_epochs=config.temporal.fold_epochs,
            patience=config.temporal.fold_patience,
            weight_decay=config.temporal.fold_weight_decay,
            log_prefix=f"{fold_label} {model_label}",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_val = base_states[val_indices]
        val_pred, val_target = _restore_levels(
            val_pred_raw,
            val_target_raw,
            base_val,
            target_mode=config.target_mode,
        )
        metrics = evaluate_predictions(val_pred, val_target)
        logger.info("%s mean R2 = %.4f", fold_label, metrics["mean_r2"])

        oof_predictions.append(val_pred.astype(np.float32))
        oof_targets.append(val_target.astype(np.float32))
        oof_indices.append(val_indices.astype(np.int32))

        fold_summaries.append(
            {
                "fold": fold_idx,
                "val_sequences": len(val_seq_set),
                "val_samples": int(val_mask.sum()),
                "mean_r2": metrics["mean_r2"],
            }
        )

    oof_pred_concat = np.concatenate(oof_predictions, axis=0)
    oof_target_concat = np.concatenate(oof_targets, axis=0)
    oof_indices_concat = np.concatenate(oof_indices, axis=0)
    order = np.argsort(oof_indices_concat)
    oof_pred_sorted = oof_pred_concat[order]
    oof_target_sorted = oof_target_concat[order]
    oof_indices_sorted = oof_indices_concat[order]

    oof_metrics = evaluate_predictions(oof_pred_sorted, oof_target_sorted)
    logger.info("OOF mean R2 = %.4f", oof_metrics["mean_r2"])

    # Save OOF predictions even if skipping final training
    if save_artifacts:
        stats.save(artifacts_dir / "data_stats.npz")
        np.savez(
            artifacts_dir / "oof_predictions.npz",
            temporal=oof_pred_sorted.astype(np.float32),
            target=oof_target_sorted.astype(np.float32),
            indices=oof_indices_sorted.astype(np.int32),
        )
        logger.info("Saved OOF predictions and data stats")

    final_metrics: Optional[Dict[str, Dict[str, float]]] = None
    final_data_info: Dict[str, Optional[int]] = {
        "train_samples_final": None,
        "val_samples_final": None,
        "train_sequences_final": None,
        "val_sequences_final": None,
    }

    if run_final:
        final_split_idx = max(1, int(len(unique_seq) * (1 - config.final_val_ratio)))
        final_train_seq = set(unique_seq[:final_split_idx])
        final_val_seq = set(unique_seq[final_split_idx:])
        logger.info(
            "Final split: %d train seq, %d val seq",
            len(final_train_seq),
            len(final_val_seq),
        )

        train_mask = np.isin(seqs, list(final_train_seq))
        val_mask = np.isin(seqs, list(final_val_seq))
        train_indices = np.where(train_mask)[0].astype(np.int32)
        val_indices = np.where(val_mask)[0].astype(np.int32)

        model_label = config.temporal.model_type.lower()
        model, val_pred_raw, val_target_raw = _train_temporal_on_indices(
            train_indices,
            val_indices,
            seqs,
            steps,
            states_by_seq,
            stats,
            device,
            config,
            num_epochs=config.temporal.final_epochs,
            patience=config.temporal.final_patience,
            weight_decay=config.temporal.final_weight_decay,
            log_prefix=f"[FINAL] {model_label}",
            save_model_path=(artifacts_dir / _temporal_model_filename(config.temporal.model_type)) if save_artifacts else None,
        )

        base_val = base_states[val_indices]
        val_pred, val_target = _restore_levels(
            val_pred_raw,
            val_target_raw,
            base_val,
            target_mode=config.target_mode,
        )
        final_metric = evaluate_predictions(val_pred, val_target)
        final_metrics = {"transformer": final_metric}
        logger.info("[FINAL] mean R2 = %.4f", final_metric["mean_r2"])

        final_data_info = {
            "train_samples_final": int(train_mask.sum()),
            "val_samples_final": int(val_mask.sum()),
            "train_sequences_final": len(final_train_seq),
            "val_sequences_final": len(final_val_seq),
        }

        if save_artifacts:
            # OOF already saved above, now save temporal_meta
            temporal_meta = {
                "model_type": config.temporal.model_type,
                "sequence_window": config.sequence_window,
                "target_mode": config.target_mode,
                "feature_module": feature_module_name,
                "feature_class": config.feature_class,
                "feature_params": config.feature_params,
            }
            temporal_meta.update(
                {k: v for k, v in asdict(config.temporal).items() if k != "model_type"}
            )
            with open(artifacts_dir / "temporal_model_meta.json", "w", encoding="utf-8") as fp:
                json.dump(temporal_meta, fp, ensure_ascii=False, indent=2)

    summary = {
        "config": config.to_dict(),
        "folds": fold_summaries,
        "oof": {"transformer": oof_metrics},
        "final_validation": final_metrics,
        "data_info": {
            "total_samples": int(len(y_levels_orig)),
            "total_sequences": int(len(unique_seq)),
            "feature_dim": int(X.shape[1]),
            "sequence_window": config.sequence_window,
            **final_data_info,
        },
    }

    if save_artifacts and run_final:
        with open(artifacts_dir / "training_summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

    return {
        "summary": summary,
        "artifacts_dir": str(artifacts_dir) if save_artifacts and run_final else None,
    }


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
    parser = argparse.ArgumentParser(description="Train transformer-only temporal model")
    parser.add_argument("--dataset", type=str, default=str(Path("datasets") / "train.parquet"))
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--config-path", type=str, default=None, help="Path to JSON config")
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
