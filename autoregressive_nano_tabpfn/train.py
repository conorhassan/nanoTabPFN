"""Training loop for ARTabPFN with online tabular data generation."""

import time
from typing import Optional, Dict, Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model import ARTabPFN, create_dense_mask, create_row_mask, clear_mask_cache
from .data import DataAttr, TabularSampler, OnlineTabularDataset


def train(
    model: ARTabPFN,
    dataset: OnlineTabularDataset,
    steps: int,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    device: str = "cuda",
    compile_model: bool = True,
    log_interval: int = 50,
) -> ARTabPFN:
    """
    Train ARTabPFN with online data generation.

    Args:
        model: ARTabPFN instance
        dataset: OnlineTabularDataset for data generation
        steps: Number of training steps
        lr: Learning rate
        grad_clip: Gradient clipping norm
        device: Device to train on
        compile_model: Whether to use torch.compile (CUDA only)
        log_interval: Steps between logging

    Returns:
        Trained model
    """
    model = model.to(device)

    if compile_model and device == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    # DataLoader with pre-batched data
    loader = DataLoader(dataset, batch_size=None, shuffle=False)
    data_iter = iter(loader)

    # Mask cache for different shapes
    mask_cache: Dict[tuple, tuple] = {}

    total_loss = 0.0
    t0 = time.perf_counter()

    for step in range(steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.to(device)

        # Get or create masks for this shape
        Nc, Nb, Nt = batch.xc.size(1), batch.xb.size(1), batch.xt.size(1)
        cache_key = (Nc, Nb, Nt)

        if cache_key not in mask_cache:
            mask_features = create_dense_mask(seq_len=1, device=device)
            mask_rows = create_row_mask(
                num_rows=Nc + Nb + Nt,
                context_len=Nc,
                buffer_len=Nb,
                device=device,
            )
            mask_cache[cache_key] = (mask_features, mask_rows)
        else:
            mask_features, mask_rows = mask_cache[cache_key]

        # Forward
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = model(
                x_context=batch.xc,
                y_context=batch.yc.squeeze(-1),
                x_buffer=batch.xb,
                y_buffer=batch.yb.squeeze(-1),
                x_target=batch.xt,
                mask_features=mask_features,
                mask_rows=mask_rows,
                y_target=batch.yt.squeeze(-1),
            )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            avg_loss = total_loss / log_interval
            steps_per_sec = log_interval / elapsed
            print(f"step {step+1:5d} | loss {avg_loss:.4f} | {steps_per_sec:.1f} steps/s")
            total_loss = 0.0
            t0 = time.perf_counter()

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Config
    d_model = 64
    n_heads = 4
    n_layers = 3
    d_ff = 128
    buffer_size = 8
    num_components = 5

    d_list = [3, 5, 10]
    nc_list = [16, 32, 64]
    num_buffer = buffer_size
    num_target = 64
    batch_size = 32
    num_batches = 100

    # Create dataset
    dataset = OnlineTabularDataset(
        batch_size=batch_size,
        num_batches=num_batches,
        d_list=d_list,
        nc_list=nc_list,
        num_buffer=num_buffer,
        num_target=num_target,
        normalize_y=True,
        dtype=torch.float32,
        device="cpu",
        seed=42,
    )

    # Create model (use max feature dim)
    model = ARTabPFN(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        num_features=max(d_list),
        buffer_size=buffer_size,
        num_components=num_components,
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_batches} steps...")

    trained_model = train(
        model,
        dataset,
        steps=num_batches,
        lr=1e-3,
        device=device,
        compile_model=(device == "cuda"),
        log_interval=10,
    )

    print("Training complete!")
