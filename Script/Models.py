import torch
import torch.nn as nn
from scipy.signal import firwin
from Config import FS

#Low pass filter
class FixedLowPass(nn.Module):
    """Zero-phase FIR low-pass applied as a fixed (non-trainable) convolution."""
    def __init__(self, cutoff_hz, numtaps=21):
        super().__init__()
        h = firwin(numtaps, cutoff_hz, fs=FS)           # symmetric, so no timing shift
        self.register_buffer('h', torch.tensor(h, dtype=torch.float32).view(1, 1, -1))
        self.pad = numtaps // 2

    def forward(self, x):
        return nn.functional.conv1d(x, self.h, padding=self.pad)


class SpikeCNN(nn.Module):
    def __init__(self, n_out, lowpass_hz=None):
        super().__init__()
        front = [FixedLowPass(lowpass_hz)] if lowpass_hz else []
        self.features = nn.Sequential(
            *front,
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),               

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),               

            nn.AdaptiveAvgPool1d(4),       
        )
        self.head = nn.Sequential(
            nn.Flatten(),                  
            nn.Dropout(0.3),
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.head(self.features(x))