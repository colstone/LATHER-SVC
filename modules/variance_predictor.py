from __future__ import annotations

import torch
import torch.nn as nn

from modules.fastspeech.tts_modules import LayerNorm
from utils.hparams import hparams


class SVCVariancePredictor(nn.Module):
    OUTPUT_NAMES = ('breathiness', 'voicing', 'tension')

    def __init__(self, in_dims: int | None = None):
        super().__init__()
        cfg = hparams['variance_predictor']
        in_dims = in_dims or hparams['hidden_size']
        n_layers = int(cfg.get('backbone_layers', 5))
        n_channels = int(cfg.get('backbone_channels', in_dims))
        kernel_size = int(cfg.get('kernel_size', 5))
        dropout = float(cfg.get('dropout', 0.1))

        backbone = []
        for layer_idx in range(n_layers):
            in_channels = in_dims if layer_idx == 0 else n_channels
            backbone.append(nn.Sequential(
                nn.Conv1d(in_channels, n_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.ReLU(),
                LayerNorm(n_channels, dim=1),
                nn.Dropout(dropout),
            ))
        self.backbone = nn.ModuleList(backbone)
        self.heads = nn.ModuleDict({
            name: nn.Linear(n_channels, 1)
            for name in self.OUTPUT_NAMES
        })
        self.output_ranges = {
            'breathiness': (
                float(hparams.get('breathiness_db_min', -96.0)),
                float(hparams.get('breathiness_db_max', -20.0)),
            ),
            'voicing': (
                float(hparams.get('voicing_db_min', -96.0)),
                float(hparams.get('voicing_db_max', -12.0)),
            ),
            'tension': (
                float(hparams.get('tension_logit_min', -10.0)),
                float(hparams.get('tension_logit_max', 10.0)),
            ),
        }

    def forward(self, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        x = condition.transpose(1, 2)
        for layer in self.backbone:
            x = layer(x)
        x = x.transpose(1, 2)
        return {
            name: head(x).squeeze(-1)
            for name, head in self.heads.items()
        }

    def clamp(self, variances: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        clamped = {}
        for name, value in variances.items():
            vmin, vmax = self.output_ranges[name]
            clamped[name] = value.clamp(min=vmin, max=vmax)
        return clamped
