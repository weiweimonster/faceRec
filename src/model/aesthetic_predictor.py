from torch import nn
from typing import Optional
import torch

class AestheticPredictor(nn.Module):
    def __init__(self, input_size, device: str = None, model_path: Optional[str] = None):
        super().__init__()
        self.input_size = input_size
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

        if model_path:
            self._load_weights(model_path)

    def _load_weights(self, model_path: str):
        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    def forward(self, x):
        return self.layers(x)



