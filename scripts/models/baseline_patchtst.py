"""PatchTST baseline: single shared-weights model trained across all 41 60s-resolution
AIOps18 KPIs at once (fit_mode="shared"), same architecture-decision rationale as
baseline_lstm.py. Patches the context window (patch_len, stride), linearly embeds each
patch, adds a learned positional encoding, runs a small Transformer encoder, then a linear
head over the flattened encoder output predicts all horizon_steps at once.

KPI identity is injected as a learned embedding added to every patch embedding (channel-wise
conditioning), rather than a separate token, to keep the sequence length (and therefore
attention cost) unchanged by adding KPI identity.

Measured ~4x faster per epoch than the LSTM on this dataset (patched attention parallelizes
across ~35 patches vs. the LSTM's 288 sequential steps) -- see compute-budget notes.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _PatchTSTNet(nn.Module):
    def __init__(
        self,
        n_kpis: int,
        context_steps: int = 288,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        horizon_steps: int = 15,
    ):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (context_steps - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.kpi_embed = nn.Embedding(n_kpis, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.n_patches, horizon_steps)

    def forward(self, x: torch.Tensor, kpi_idx: torch.Tensor) -> torch.Tensor:
        # x: (batch, context_steps), kpi_idx: (batch,)
        patches = x.unfold(1, self.patch_len, self.stride)  # (batch, n_patches, patch_len)
        emb = self.patch_embed(patches) + self.pos_embed + self.kpi_embed(kpi_idx).unsqueeze(1)
        enc = self.encoder(emb)
        return self.head(enc.flatten(1))


class PatchTSTModel:
    fit_mode = "shared"

    def __init__(self, n_kpis: int, kpi_to_idx: dict, context_steps: int = 288, horizon_steps: int = 15):
        self.kpi_to_idx = kpi_to_idx
        self.horizon_steps = horizon_steps
        self.device = torch.device("cpu")
        self.net = _PatchTSTNet(n_kpis, context_steps=context_steps, horizon_steps=horizon_steps).to(self.device)

    def fit_shared(
        self,
        context: np.ndarray,
        horizon_value: np.ndarray,
        kpi_ids: np.ndarray,
        n_epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
        checkpoint_dir: str | Path | None = None,
    ):
        """See baseline_lstm.py's fit_shared docstring for the checkpoint_dir contract --
        saves {checkpoint_dir}/patchtst_epoch{N}.pt after every epoch."""
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
                x = torch.tensor(context[idx], dtype=torch.float32, device=self.device)
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
            print(f"    [patchtst] epoch {epoch+1}/{n_epochs}: mean MSE loss = {mean_loss:.4f}")

            if checkpoint_dir is not None:
                ckpt_path = checkpoint_dir / f"patchtst_epoch{epoch+1}.pt"
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
                print(f"    [patchtst] saved checkpoint -> {ckpt_path}")

    def fit(self, train_data) -> None:
        raise NotImplementedError("PatchTSTModel is fit_mode='shared' -- call fit_shared() with pooled window arrays")

    def predict(self, context_window, horizon_steps, kpi_id=None, context_end_index=None) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(context_window, dtype=torch.float32, device=self.device).reshape(1, -1)
            k = torch.tensor([self.kpi_to_idx[kpi_id]], dtype=torch.long, device=self.device)
            pred = self.net(x, k)
        self.net.train()
        return pred.squeeze(0).numpy()

    def predict_batch(self, contexts: np.ndarray, kpi_ids: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Batched equivalent of calling predict() once per row -- see baseline_lstm.py's
        predict_batch for the rationale (350k individual forward-pass calls is far slower
        than a handful of batched ones)."""
        self.net.eval()
        kpi_idx = np.array([self.kpi_to_idx[k] for k in kpi_ids])
        out = np.zeros((len(contexts), self.horizon_steps), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(contexts), batch_size):
                end = start + batch_size
                x = torch.tensor(contexts[start:end], dtype=torch.float32, device=self.device)
                k = torch.tensor(kpi_idx[start:end], dtype=torch.long, device=self.device)
                out[start:end] = self.net(x, k).numpy()
        self.net.train()
        return out
