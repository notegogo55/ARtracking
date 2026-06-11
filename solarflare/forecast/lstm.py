"""LSTM flare-probability model (PyTorch) with imbalance-aware training.

Standardization stats come from TRAIN only; NaNs are mean-imputed (zero after
standardization). Class imbalance is handled with BCE pos_weight. Early
stopping monitors validation TSS (threshold re-picked on val each epoch, which
is legitimate because the final operating threshold is also chosen on val and
frozen before touching test). Each fit writes a training-curve CSV + PNG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from solarflare.eval.metrics import best_tss_threshold

log = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 7
    seed: int = 1337
    device: str = "cpu"
    clip_sigma: float = 10.0
    extra: dict = field(default_factory=dict)


class LSTMForecaster:
    """fit/predict_proba wrapper around a small LSTM classifier."""

    def __init__(self, config: LSTMConfig | None = None,
                 curves_dir: str | Path | None = None, name: str = "lstm") -> None:
        self.cfg = config or LSTMConfig()
        self.curves_dir = Path(curves_dir) if curves_dir else None
        self.name = name
        self._model = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.history_: list[dict] = []
        self.val_threshold_: float = 0.5

    # --- preprocessing -----------------------------------------------------
    def _fit_scaler(self, X: np.ndarray) -> None:
        flat = X.reshape(-1, X.shape[-1])
        self._mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        self._std = np.where(std > 0, std, 1.0)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self._mean) / self._std
        Z = np.clip(Z, -self.cfg.clip_sigma, self.cfg.clip_sigma)
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)

    # --- training ----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from solarflare.utils.seed import set_global_seed

        cfg = self.cfg
        set_global_seed(cfg.seed)
        device = torch.device(cfg.device)
        self._fit_scaler(X)
        Xt = torch.from_numpy(self._transform(X))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=cfg.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))

        n_features = X.shape[-1]
        model = nn.ModuleDict({
            "lstm": nn.LSTM(n_features, cfg.hidden_size, cfg.num_layers,
                            batch_first=True,
                            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0),
            "head": nn.Sequential(
                nn.Linear(cfg.hidden_size, 32), nn.ReLU(), nn.Dropout(cfg.dropout),
                nn.Linear(32, 1),
            ),
        }).to(device)

        n_pos = float(np.sum(y))
        pos_weight = torch.tensor([(len(y) - n_pos) / max(n_pos, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        def forward(batch: torch.Tensor) -> torch.Tensor:
            out, _ = model["lstm"](batch)
            return model["head"](out[:, -1, :]).squeeze(-1)

        best_state, best_score, best_epoch = None, -np.inf, -1
        self.history_ = []
        for epoch in range(cfg.max_epochs):
            model.train()
            losses = []
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(forward(xb.to(device)), yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            row = {"epoch": epoch, "train_loss": float(np.mean(losses))}

            if X_val is not None and len(X_val):
                self._model = model
                p_val = self.predict_proba(X_val, _already_fit=True)
                thr, val_tss = best_tss_threshold(y_val, p_val)
                row.update({"val_tss": val_tss, "val_threshold": thr})
                score = val_tss
            else:
                score = -row["train_loss"]
            self.history_.append(row)
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                self.val_threshold_ = row.get("val_threshold", 0.5)
            if epoch - best_epoch >= cfg.patience:
                log.info("%s: early stop at epoch %d (best %d)", self.name, epoch, best_epoch)
                break

        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        log.info("%s: trained %d epochs, best score %.4f @ %d",
                 self.name, len(self.history_), best_score, best_epoch)
        if self.curves_dir is not None:
            self._write_curves()
        return self

    # --- inference ---------------------------------------------------------
    def predict_proba(self, X: np.ndarray, _already_fit: bool = False) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("fit first")
        model = self._model
        model.eval()
        Z = torch.from_numpy(self._transform(X))
        probs = []
        with torch.no_grad():
            for i in range(0, len(Z), self.cfg.batch_size):
                batch = Z[i : i + self.cfg.batch_size]
                out, _ = model["lstm"](batch)
                logits = model["head"](out[:, -1, :]).squeeze(-1)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs) if probs else np.empty(0)

    # --- diagnostics ---------------------------------------------------------
    def _write_curves(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        self.curves_dir.mkdir(parents=True, exist_ok=True)
        hist = pd.DataFrame(self.history_)
        hist.to_csv(self.curves_dir / f"{self.name}_history.csv", index=False)
        fig, ax1 = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax1.plot(hist["epoch"], hist["train_loss"], label="train loss", color="tab:blue")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss", color="tab:blue")
        if "val_tss" in hist:
            ax2 = ax1.twinx()
            ax2.plot(hist["epoch"], hist["val_tss"], label="val TSS", color="tab:red")
            ax2.set_ylabel("val TSS", color="tab:red")
        fig.suptitle(f"{self.name} training curves")
        fig.savefig(self.curves_dir / f"{self.name}_curves.png", dpi=130)
        plt.close(fig)
