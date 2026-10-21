"""LSTM baseline: single shared-weights model trained across all 41 60s-resolution AIOps18
KPIs at once (fit_mode="shared"), per the architecture decision that DL models share weights
across series while classical models (SARIMA/XGBoost, baseline_sarima.py/baseline_xgboost.py)
fit one instance per KPI. Relies on the existing per-KPI z-score normalization so scale
differences across series don't confound a shared representation.

Requires the dedicated .venv-torch environment (PyTorch 2.2.2, the last version with macOS
x86_64/Intel wheels available -- the project's main .venv is Python 3.13, for which no
PyTorch wheel exists on this platform).

KPI identity is passed to the model via an embedding, used to initialize the LSTM's hidden
state (rather than concatenated at every timestep) -- cheaper, and lets the model start from
a series-specific "prior" before seeing any of that window's actual values.

Direct multi-step forecasting (one linear head predicting all horizon_steps at once from the
final hidden state), not recursive -- avoids the per-step error compounding that hurt
XGBoost's recursive strategy in Phase 1.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model_interface import ForecastModel, TrainData


class _LSTMNet(nn.Module):
    def __init__(self, n_kpis: int, hidden_size: int = 64, embed_dim: int = 16, horizon_steps: int = 15):
        super().__init__()
        self.kpi_embed = nn.Embedding(n_kpis, embed_dim)
        self.embed_to_hidden = nn.Linear(embed_dim, hidden_size)
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon_steps)

    def forward(self, x: torch.Tensor, kpi_idx: torch.Tensor) -> torch.Tensor:
        # x: (batch, context_steps, 1), kpi_idx: (batch,)
        h0 = self.embed_to_hidden(self.kpi_embed(kpi_idx)).unsqueeze(0)  # (1, batch, hidden)
        c0 = torch.zeros_like(h0)
        _, (h_n, _) = self.lstm(x, (h0, c0))
        return self.head(h_n[-1])


class LSTMModel(ForecastModel):
    fit_mode = "shared"

    def __init__(self, n_kpis: int, kpi_to_idx: dict, hidden_size: int = 64, embed_dim: int = 16, horizon_steps: int = 15):
        self.kpi_to_idx = kpi_to_idx
        self.horizon_steps = horizon_steps
        self.device = torch.device("cpu")  # MPS ran out of memory at viable batch sizes; see compute-budget notes
        self.net = _LSTMNet(n_kpis, hidden_size, embed_dim, horizon_steps).to(self.device)

    def fit_shared(
        self,
        context: np.ndarray,
        horizon_value: np.ndarray,
        kpi_ids: np.ndarray,
        n_epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        checkpoint_dir: str | Path | None = None,
    ):
        """Not part of the abstract ForecastModel interface (fit() there takes a single
        TrainData) -- shared models need the full pooled window arrays, so this is the actual
        entry point the training script calls. fit() below is a thin no-op-compatible stub to
        satisfy the interface for code that expects it.

        If checkpoint_dir is given, saves {checkpoint_dir}/lstm_epoch{N}.pt after every epoch
        (model state_dict, optimizer state_dict, epoch number, mean training loss) -- lets a
        run resume or be inspected mid-training rather than losing everything if it's
        interrupted before the final epoch.
        """
        kpi_idx = np.array([self.kpi_to_idx[k] for k in kpi_ids])
        n = len(context)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(n_epochs):
            perm = np.random.permutation(n)
            total_loss, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                x = torch.tensor(context[idx], dtype=torch.float32, device=self.device).unsqueeze(-1)
                y = torch.tensor(horizon_value[idx], dtype=torch.float32, device=self.device)
                k = torch.tensor(kpi_idx[idx], dtype=torch.long, device=self.device)

                opt.zero_grad()
                pred = self.net(x, k)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
            mean_loss = total_loss / n_batches
            print(f"    [lstm] epoch {epoch+1}/{n_epochs}: mean MSE loss = {mean_loss:.4f}")

            if checkpoint_dir is not None:
                ckpt_path = checkpoint_dir / f"lstm_epoch{epoch+1}.pt"
                torch.save(
                    dict(
                        epoch=epoch + 1,
                        model_state_dict=self.net.state_dict(),
                        optimizer_state_dict=opt.state_dict(),
                        mean_loss=mean_loss,
                        kpi_to_idx=self.kpi_to_idx,
                    ),
                    ckpt_path,
                )
                print(f"    [lstm] saved checkpoint -> {ckpt_path}")

    def fit(self, train_data: TrainData) -> None:
        raise NotImplementedError("LSTMModel is fit_mode='shared' -- call fit_shared() with pooled window arrays, not fit()")

    def predict(self, context_window, horizon_steps, kpi_id=None, context_end_index=None) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(context_window, dtype=torch.float32, device=self.device).reshape(1, -1, 1)
            k = torch.tensor([self.kpi_to_idx[kpi_id]], dtype=torch.long, device=self.device)
            pred = self.net(x, k)
        self.net.train()
        return pred.squeeze(0).numpy()

    def predict_batch(self, contexts: np.ndarray, kpi_ids: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Batched equivalent of calling predict() once per row -- ~1000x fewer Python-level
        forward-pass calls, which matters a lot for a 350k-window eval set. Same math as
        predict(), just amortized over mini-batches instead of one row at a time.
        """
        self.net.eval()
        kpi_idx = np.array([self.kpi_to_idx[k] for k in kpi_ids])
        out = np.zeros((len(contexts), self.horizon_steps), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(contexts), batch_size):
                end = start + batch_size
                x = torch.tensor(contexts[start:end], dtype=torch.float32, device=self.device).unsqueeze(-1)
                k = torch.tensor(kpi_idx[start:end], dtype=torch.long, device=self.device)
                out[start:end] = self.net(x, k).numpy()
        self.net.train()
        return out
