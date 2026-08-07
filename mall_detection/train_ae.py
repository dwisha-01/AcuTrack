#  train_ae.py — Train the LSTM Autoencoder on collected normal trajectories
#
#  Usage:
#    python train_ae.py
#    python train_ae.py --epochs 100 --data training_data.npy
#
#  Output:
#    trajectory_ae.pt      — model weights
#    ae_threshold.npy      — anomaly threshold (95th percentile of training errors)
#
#  Runtime: ~30-60 s on CPU for 500 sequences, 50 epochs.

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time

from trajectory_ae import TrajectoryAutoencoder, SEQ_LEN, FEAT_DIM

WEIGHTS_FILE   = "trajectory_ae.pt"
THRESHOLD_FILE = "ae_threshold.npy"


def train(data_file="training_data.npy", epochs=80, batch_size=64,
          lr=1e-3, val_split=0.15, percentile=95):

    print(f"\n{'='*55}")
    print(f"  AcuTrack — Autoencoder Training")
    print(f"{'='*55}")

    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.exists(data_file):
        print(f"\n  ERROR: {data_file} not found.")
        print("  Run collect_trajectories.py first to generate training data.\n")
        return

    data = np.load(data_file).astype(np.float32)   # (N, SEQ_LEN, FEAT_DIM)
    print(f"\n  Loaded {len(data)} sequences  (shape: {data.shape})")

    if len(data) < 50:
        print(f"  WARNING: only {len(data)} sequences — collect more for better results.")
        print("  Minimum recommended: 200 sequences.\n")

    # ── Train / val split ──────────────────────────────────────────────────────
    np.random.shuffle(data)
    split = max(1, int(len(data) * (1 - val_split)))
    X_train = torch.tensor(data[:split])
    X_val   = torch.tensor(data[split:])
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TrajectoryAutoencoder(feat_dim=FEAT_DIM, seq_len=SEQ_LEN)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loss_fn = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}  (CPU inference: ~{param_count*4//1024} KB)\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_state    = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        # Mini-batch shuffle
        perm = torch.randperm(len(X_train))
        train_loss = 0.0
        n_batches  = 0
        for i in range(0, len(X_train), batch_size):
            batch = X_train[perm[i:i+batch_size]]
            opt.zero_grad()
            recon = model(batch)
            loss  = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n_batches  += 1
        train_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), X_val).item()

        sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>3}/{epochs}  |  train: {train_loss:.5f}  "
                  f"|  val: {val_loss:.5f}  |  {elapsed:.1f}s elapsed")

    # ── Save best weights ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"\n  Best val loss: {best_val_loss:.5f}")
    print(f"  Saved weights → {WEIGHTS_FILE}")

    # ── Compute anomaly threshold ──────────────────────────────────────────────
    # Threshold = Nth percentile of reconstruction errors on training set.
    # Any score above this at inference time is flagged as anomalous.
    model.eval()
    with torch.no_grad():
        all_errors = []
        for i in range(0, len(X_train), batch_size):
            batch  = X_train[i:i+batch_size]
            recon  = model(batch)
            errors = torch.mean((recon - batch)**2, dim=[1, 2]).numpy()
            all_errors.extend(errors.tolist())

    all_errors = np.array(all_errors)
    threshold  = float(np.percentile(all_errors, percentile))

    np.save(THRESHOLD_FILE, np.array([threshold], dtype=np.float32))

    print(f"\n  Reconstruction error stats (training set):")
    print(f"    Min    : {all_errors.min():.5f}")
    print(f"    Mean   : {all_errors.mean():.5f}")
    print(f"    Median : {np.median(all_errors):.5f}")
    print(f"    95th % : {np.percentile(all_errors, 95):.5f}")
    print(f"    Max    : {all_errors.max():.5f}")
    print(f"\n  Anomaly threshold ({percentile}th pct): {threshold:.5f}")
    print(f"  Saved threshold → {THRESHOLD_FILE}")
    print(f"\n  Total training time: {time.time()-t0:.1f}s")
    print(f"\n  ✓ Ready — restart app.py to load the trained model.\n")
    print("="*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="training_data.npy")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--percentile", type=int,   default=95,
                        help="Percentile of training errors used as anomaly threshold")
    args = parser.parse_args()
    train(
        data_file  = args.data,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        percentile = args.percentile,
    )
