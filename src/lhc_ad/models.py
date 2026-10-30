"""Dense autoencoder; the anomaly score is the per-jet reconstruction MSE."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    def __init__(self, n_inputs: int, hidden: tuple[int, ...] = (256, 64), latent: int = 16):
        super().__init__()
        enc, dims = [], [n_inputs, *hidden]
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        enc.append(nn.Linear(dims[-1], latent))
        dec, dims = [], [latent, *reversed(hidden)]
        for a, b in zip(dims[:-1], dims[1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        dec.append(nn.Linear(dims[-1], n_inputs))
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    x_train: np.ndarray,
    x_val: np.ndarray | None = None,
    hidden: tuple[int, ...] = (256, 64),
    latent: int = 16,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[DenseAutoencoder, dict[str, list[float]]]:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseAutoencoder(x_train.shape[1], hidden, latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    history: dict[str, list[float]] = {"train": [], "val": []}
    gen = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(xt), batch_size):
            batch = xt[perm[i : i + batch_size]].to(device)
            loss = ((model(batch) - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        history["train"].append(total / len(xt))
        if x_val is not None:
            history["val"].append(float(reconstruction_error(model, x_val).mean()))
        if verbose:
            val = f" val={history['val'][-1]:.5g}" if x_val is not None else ""
            print(f"epoch {epoch + 1:3d} train={history['train'][-1]:.5g}{val}")
    return model, history


@torch.no_grad()
def reconstruction_error(model: nn.Module, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        out.append(((model(batch) - batch) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)
