from __future__ import annotations

import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import XavierUniformInitLinear as Linear
from utils.hparams import hparams


@contextmanager
def legacy_checkpoint_torch_load():
    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        try:
            supports_weights_only = 'weights_only' in inspect.signature(original_torch_load).parameters
        except (TypeError, ValueError):
            supports_weights_only = False
        if supports_weights_only:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def align_frame_rate(features: torch.Tensor, target_length: int) -> torch.Tensor:
    if features.size(1) == target_length:
        return features
    if target_length <= 0:
        raise ValueError(f'target_length must be positive, got {target_length}')
    features = features.transpose(1, 2)
    features = F.interpolate(features, size=target_length, mode='linear', align_corners=False)
    return features.transpose(1, 2)


class LayerAttention(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.weights = nn.Parameter(torch.ones(n_layers) / max(1, n_layers))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 4:
            raise ValueError(f'Expected [B, L, T, C], got shape={tuple(hidden_states.shape)}')
        if hidden_states.size(1) != self.n_layers:
            raise ValueError(
                f'LayerAttention expects {self.n_layers} layers, got {hidden_states.size(1)}.'
            )
        weights = torch.softmax(self.weights, dim=0)
        return (hidden_states * weights[None, :, None, None]).sum(dim=1)


class ContentVecExtractor(nn.Module):
    def __init__(self, output_dim: int | None = None):
        super().__init__()
        cfg = hparams['contentvec']
        self.output_dim = output_dim or hparams['hidden_size']
        self.layer_mode = cfg.get('layer_mode', 'attention')
        self.target_layer = int(cfg.get('target_layer', 12))
        layer_range = cfg.get('layer_range', [self.target_layer, self.target_layer])
        if len(layer_range) != 2:
            raise ValueError(f'contentvec.layer_range must contain two integers, got {layer_range}')
        self.layer_start = int(layer_range[0])
        self.layer_end = int(layer_range[1])
        if self.layer_end < self.layer_start:
            raise ValueError(
                f'Invalid contentvec.layer_range: {self.layer_start}..{self.layer_end}'
            )
        self.freeze = bool(cfg.get('freeze', True))
        self.checkpoint_path = cfg.get('checkpoint_path')
        self.input_dim = int(cfg.get('input_dim', 768))
        self.selected_layer_count = self.layer_end - self.layer_start + 1
        self._fairseq_extraction_mode: str | None = None

        if self.layer_mode == 'attention':
            self.layer_attention = LayerAttention(self.selected_layer_count)
        elif self.layer_mode == 'single':
            self.layer_attention = None
        else:
            raise ValueError(f'Unsupported contentvec.layer_mode: {self.layer_mode}')

        self.proj = Linear(self.input_dim, self.output_dim)
        self.model, self.backend = self._load_model()
        if self.model is not None and self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_model(self):
        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            raise ValueError('contentvec.checkpoint_path must be configured.')
        checkpoint_path = str(Path(checkpoint_path))

        load_errors: List[str] = []
        try:
            from fairseq import checkpoint_utils  # type: ignore

            with legacy_checkpoint_torch_load():
                models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                    [checkpoint_path],
                    strict=False,
                )
            if not models:
                raise RuntimeError(f'No fairseq model loaded from {checkpoint_path}.')
            model = models[0]
            model.eval()
            return model, 'fairseq'
        except Exception as exc:  # pragma: no cover - depends on local runtime
            load_errors.append(f'fairseq: {exc}')

        try:
            from transformers import HubertModel  # type: ignore

            model = HubertModel.from_pretrained(checkpoint_path)
            model.eval()
            return model, 'transformers'
        except Exception as exc:  # pragma: no cover - depends on local runtime
            load_errors.append(f'transformers: {exc}')

        joined_errors = '; '.join(load_errors)
        raise RuntimeError(
            'Failed to load ContentVec model. '
            f'checkpoint={checkpoint_path}. Tried fairseq and transformers. Errors: {joined_errors}'
        )

    def _select_layers(self, hidden_states: Iterable[torch.Tensor]) -> torch.Tensor:
        hidden_states = list(hidden_states)
        if not hidden_states:
            raise RuntimeError('ContentVec extractor returned no hidden states.')

        if len(hidden_states) >= self.layer_end + 1:
            selected = hidden_states[self.layer_start:self.layer_end + 1]
        elif len(hidden_states) >= self.layer_end:
            selected = hidden_states[self.layer_start - 1:self.layer_end]
        else:
            raise RuntimeError(
                f'Not enough hidden states for layer_range={self.layer_start}:{self.layer_end}. '
                f'Only {len(hidden_states)} states were returned.'
            )
        return torch.stack(selected, dim=1)

    def _normalize_fairseq_layers(self, layer_results: Iterable) -> List[torch.Tensor]:
        normalized: List[torch.Tensor] = []
        for layer in layer_results:
            if isinstance(layer, (list, tuple)):
                layer = layer[0]
            if layer.dim() == 3 and layer.size(0) > layer.size(1):
                layer = layer.transpose(0, 1)
            normalized.append(layer)
        return normalized

    def _get_fairseq_encoder_layers(self):
        candidate_paths = [
            ('encoder', 'layers'),
            ('w2v_model', 'encoder', 'layers'),
            ('w2v_encoder', 'w2v_model', 'encoder', 'layers'),
        ]
        for path in candidate_paths:
            module = self.model
            found = True
            for attr in path:
                module = getattr(module, attr, None)
                if module is None:
                    found = False
                    break
            if found:
                return list(module)
        raise RuntimeError('Could not locate fairseq encoder layers for ContentVec extraction.')

    def _extract_fairseq_hidden_states_with_hooks(self, audio_16k: torch.Tensor) -> List[torch.Tensor]:
        captured_layers = []
        handles = []
        encoder_layers = self._get_fairseq_encoder_layers()
        if len(encoder_layers) < self.layer_end:
            raise RuntimeError(
                f'Fairseq ContentVec encoder exposes {len(encoder_layers)} layers, '
                f'but layer_end={self.layer_end} was requested.'
            )

        def capture_hook(_module, _inputs, output):
            captured_layers.append(output)

        for layer in encoder_layers[:self.layer_end]:
            handles.append(layer.register_forward_hook(capture_hook))

        try:
            with torch.no_grad():
                self.model(
                    source=audio_16k,
                    padding_mask=None,
                    mask=False,
                    features_only=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        if len(captured_layers) < self.layer_end:
            raise RuntimeError(
                'Fairseq hook fallback did not capture enough layers for ContentVec extraction. '
                f'Captured {len(captured_layers)} layer(s), expected at least {self.layer_end}.'
            )
        return self._normalize_fairseq_layers(captured_layers)

    def extract_hidden_states(self, audio_16k: torch.Tensor) -> torch.Tensor:
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)
        if audio_16k.dim() != 2:
            raise ValueError(f'Expected audio shape [B, T], got {tuple(audio_16k.shape)}')

        if self.backend == 'fairseq':
            if self._fairseq_extraction_mode == 'hooks':
                hidden_states = self._extract_fairseq_hidden_states_with_hooks(audio_16k)
            else:
                with torch.no_grad():
                    outputs = self.model(
                        source=audio_16k,
                        padding_mask=None,
                        mask=False,
                        features_only=True,
                        output_layer=self.layer_end,
                    )
                if isinstance(outputs, dict) and outputs.get('layer_results'):
                    self._fairseq_extraction_mode = 'layer_results'
                    hidden_states = self._normalize_fairseq_layers(outputs['layer_results'])
                else:
                    if self._fairseq_extraction_mode != 'hooks':
                        print('| info: fairseq ContentVec backend returned no layer_results; using encoder hooks.')
                    self._fairseq_extraction_mode = 'hooks'
                    hidden_states = self._extract_fairseq_hidden_states_with_hooks(audio_16k)
        elif self.backend == 'transformers':
            with torch.no_grad():
                outputs = self.model(
                    audio_16k,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = list(outputs.hidden_states or [])
        else:  # pragma: no cover
            raise RuntimeError(f'Unsupported ContentVec backend: {self.backend}')

        return self._select_layers(hidden_states)

    def _combine_layers(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 4:
            raise ValueError(f'Expected [B, L, T, C], got shape={tuple(hidden_states.shape)}')

        if self.layer_mode == 'attention':
            hidden_states = self.layer_attention(hidden_states)
        else:
            target_index = self.target_layer - self.layer_start
            if target_index < 0 or target_index >= hidden_states.size(1):
                raise ValueError(
                    f'target_layer={self.target_layer} is outside cached layer_range='
                    f'[{self.layer_start}, {self.layer_end}]'
                )
            hidden_states = hidden_states[:, target_index]
        return self.proj(hidden_states)

    def forward(self, audio_16k: torch.Tensor) -> torch.Tensor:
        hidden_states = self.extract_hidden_states(audio_16k)
        return self._combine_layers(hidden_states)

    def from_cached_features(self, cached: torch.Tensor) -> torch.Tensor:
        return self._combine_layers(cached)
