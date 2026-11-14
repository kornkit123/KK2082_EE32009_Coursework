import torch.nn as nn

class SpikeCNN(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.features = nn.Sequential(
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