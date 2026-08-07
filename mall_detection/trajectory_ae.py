#  trajectory_ae.py — LSTM Autoencoder for trajectory anomaly detection
#  Unsupervised: trained on normal behaviour, flags high reconstruction error.
#  No GPU required — tiny model (~50K params), inference is microseconds on CPU.

import torch
import torch.nn as nn
import numpy as np

# ── CONSTANTS (shared between training and inference) ──────────────────────────
SEQ_LEN  = 30    # frames per window  (~3 s at 9-10 FPS)
FEAT_DIM = 8     # features per frame (see extract_features)

# Feature indices (for readability)
F_X     = 0   # foot_x normalised to [0,1]
F_Y     = 1   # foot_y normalised to [0,1]
F_VX    = 2   # x velocity, normalised & clamped
F_VY    = 3   # y velocity, normalised & clamped
F_ZA    = 4   # in Zone A (0 or 1)
F_ZB    = 5   # in Zone B (0 or 1)
F_ZC    = 6   # in Zone C (0 or 1)
F_DWELL = 7   # dwell time in current zone / 120 s (capped)


# ── MODEL ──────────────────────────────────────────────────────────────────────

class TrajectoryAutoencoder(nn.Module):
    """
    LSTM-based sequence autoencoder.
    Encoder compresses a 30-frame trajectory into a 16-dim latent vector.
    Decoder reconstructs the full sequence from that vector.
    Anomaly score = MSE(input, reconstruction).  High error = unusual behaviour.
    """
    def __init__(self, feat_dim=FEAT_DIM, hidden_dim=32, latent_dim=16, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len

        # Encoder: sequence → context vector
        self.enc_lstm   = nn.LSTM(feat_dim, hidden_dim, batch_first=True)
        self.enc_linear = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent vector → reconstructed sequence
        self.dec_linear = nn.Linear(latent_dim, hidden_dim)
        self.dec_lstm   = nn.LSTM(hidden_dim, feat_dim, batch_first=True)

    def encode(self, x):
        """x: (B, T, F) → latent: (B, latent_dim)"""
        _, (h, _) = self.enc_lstm(x)
        return self.enc_linear(h[-1])

    def decode(self, latent):
        """latent: (B, latent_dim) → recon: (B, T, F)"""
        h0   = self.dec_linear(latent)                            # (B, hidden)
        inp  = h0.unsqueeze(1).repeat(1, self.seq_len, 1)         # (B, T, hidden)
        recon, _ = self.dec_lstm(inp)
        return recon

    def forward(self, x):
        return self.decode(self.encode(x))

    def reconstruction_error(self, x):
        """
        x: (B, T, F) tensor  — returns per-sample MSE as a Python float list.
        Call with B=1 during live inference.
        """
        with torch.no_grad():
            recon = self.forward(x)
            # Mean over time & feature dims, keep batch dim
            return torch.mean((recon - x) ** 2, dim=[1, 2]).tolist()


# ── FEATURE EXTRACTION ─────────────────────────────────────────────────────────

def extract_features(foot_x, foot_y, prev_foot_x, prev_foot_y,
                     current_zone, dwell_seconds,
                     frame_w=1060, frame_h=660,
                     max_vel=30.0, max_dwell=120.0):
    """
    Builds one feature vector (numpy array of shape (FEAT_DIM,)) for a single frame.

    Parameters
    ----------
    foot_x, foot_y      : pixel position of person's foot this frame
    prev_foot_x/y       : pixel position previous frame (None on first frame)
    current_zone        : "Zone A" | "Zone B" | "Zone C" | None
    dwell_seconds       : time person has been in current_zone (0 if no zone)
    frame_w, frame_h    : frame dimensions for normalisation
    max_vel             : velocity clamp in pixels/frame
    max_dwell           : dwell normalisation ceiling in seconds
    """
    x_n = foot_x / frame_w
    y_n = foot_y / frame_h

    if prev_foot_x is not None and prev_foot_y is not None:
        vx_n = np.clip((foot_x - prev_foot_x) / max_vel, -1.0, 1.0)
        vy_n = np.clip((foot_y - prev_foot_y) / max_vel, -1.0, 1.0)
    else:
        vx_n = 0.0
        vy_n = 0.0

    za = 1.0 if current_zone == "Zone A" else 0.0
    zb = 1.0 if current_zone == "Zone B" else 0.0
    zc = 1.0 if current_zone == "Zone C" else 0.0

    dwell_n = min(dwell_seconds / max_dwell, 1.0) if current_zone else 0.0

    return np.array([x_n, y_n, vx_n, vy_n, za, zb, zc, dwell_n],
                    dtype=np.float32)


def seq_to_tensor(seq):
    """Convert list/deque of feature vectors → (1, T, F) float tensor."""
    return torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0)
