from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import modules.compat as compat
from basics.base_module import CategorizedModule
from modules.aux_decoder import AuxDecoderAdaptor
from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.condition_refiner import ConditionRefiner
from modules.content_encoder import ContentVecExtractor, align_frame_rate
from modules.core import RectifiedFlow
from modules.variance_predictor import SVCVariancePredictor
from utils.hparams import hparams


class ShallowDiffusionOutput:
    def __init__(self, *, aux_out=None, diff_out=None):
        self.aux_out = aux_out
        self.diff_out = diff_out


def lengths_to_padding_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device)[None]
    return positions >= lengths[:, None]


class LatherSVC(CategorizedModule):
    VARIANCE_NAMES = ('breathiness', 'voicing', 'tension')

    @property
    def category(self):
        return 'acoustic'

    def __init__(self, out_dims: int):
        super().__init__()
        hidden_size = hparams['hidden_size']
        self.content_encoder = ContentVecExtractor(output_dim=hidden_size)
        self.pitch_embed = Linear(1, hidden_size)

        self.use_spk_id = bool(hparams.get('use_spk_id', False))
        if self.use_spk_id:
            self.spk_embed = Embedding(hparams['num_spk'], hidden_size)

        self.condition_refiner = ConditionRefiner()
        self.variance_predictor = SVCVariancePredictor(hidden_size)

        self.variance_embed_list = []
        if hparams.get('use_breathiness_embed', False):
            self.variance_embed_list.append('breathiness')
        if hparams.get('use_voicing_embed', False):
            self.variance_embed_list.append('voicing')
        if hparams.get('use_tension_embed', False):
            self.variance_embed_list.append('tension')
        self.variance_embeds = nn.ModuleDict({
            name: Linear(1, hidden_size)
            for name in self.variance_embed_list
        })

        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        self.shallow_args = hparams.get('shallow_diffusion_args', {})
        if self.use_shallow_diffusion:
            self.train_aux_decoder = self.shallow_args['train_aux_decoder']
            self.train_diffusion = self.shallow_args['train_diffusion']
            self.aux_decoder_grad = self.shallow_args['aux_decoder_grad']
            self.aux_decoder = AuxDecoderAdaptor(
                in_dims=hidden_size,
                out_dims=out_dims,
                num_feats=1,
                spec_min=hparams['spec_min'],
                spec_max=hparams['spec_max'],
                aux_decoder_arch=self.shallow_args['aux_decoder_arch'],
                aux_decoder_args=self.shallow_args['aux_decoder_args'],
            )

        self.diffusion_type = hparams.get('diffusion_type', 'reflow')
        self.backbone_type = compat.get_backbone_type(hparams)
        self.backbone_args = compat.get_backbone_args(hparams, self.backbone_type)
        if self.diffusion_type != 'reflow':
            raise NotImplementedError(f'LatherSVC only supports reflow, got {self.diffusion_type}')
        self.diffusion = RectifiedFlow(
            out_dims=out_dims,
            num_feats=1,
            t_start=hparams['T_start'],
            time_scale_factor=hparams['time_scale_factor'],
            backbone_type=self.backbone_type,
            backbone_args=self.backbone_args,
            spec_min=hparams['spec_min'],
            spec_max=hparams['spec_max'],
        )

    def _prepare_content(self, contentvec: torch.Tensor, mel_length: int) -> torch.Tensor:
        if contentvec.dim() == 4:
            content = self.content_encoder.from_cached_features(contentvec)
        elif contentvec.dim() == 3 and contentvec.size(-1) == hparams['hidden_size']:
            content = contentvec
        else:
            raise ValueError(
                'contentvec must be cached hidden states [B, L, T, C] or projected features [B, T, H], '
                f'got {tuple(contentvec.shape)}'
            )
        return align_frame_rate(content, mel_length)

    def _collect_variances(
            self,
            variance_pred: Dict[str, torch.Tensor],
            infer: bool,
            overrides: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if infer:
            selected = self.variance_predictor.clamp(variance_pred)
        else:
            selected = {}
            for name in self.VARIANCE_NAMES:
                if name not in overrides or overrides[name] is None:
                    raise ValueError(f'Missing ground-truth variance during training: {name}')
                selected[name] = overrides[name]
        for name, value in overrides.items():
            if value is not None and infer:
                selected[name] = value
        return selected

    def _embed_variances(self, condition: torch.Tensor, variances: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.variance_embed_list:
            return condition
        variance_embed = torch.stack([
            self.variance_embeds[name](variances[name][:, :, None])
            for name in self.variance_embed_list
        ], dim=-1).sum(dim=-1)
        return condition + variance_embed

    def forward(
            self,
            contentvec: torch.Tensor,
            f0: torch.Tensor,
            mel_lengths: torch.Tensor,
            spk_id: torch.Tensor | None = None,
            gt_mel: torch.Tensor | None = None,
            infer: bool = True,
            **kwargs,
    ):
        max_mel_length = int(mel_lengths.max().item())
        condition = self._prepare_content(contentvec, max_mel_length)

        f0_mel = (1 + f0 / 700).log()
        condition = condition + self.pitch_embed(f0_mel[:, :, None])

        if self.use_spk_id:
            if spk_id is None:
                raise ValueError('spk_id is required when use_spk_id is true.')
            condition = condition + self.spk_embed(spk_id)[:, None, :]

        padding_mask = lengths_to_padding_mask(mel_lengths, max_length=max_mel_length)
        condition = self.condition_refiner(condition, padding_mask=padding_mask)

        variance_pred = self.variance_predictor(condition)
        variance_values = self._collect_variances(
            variance_pred,
            infer=infer,
            overrides={name: kwargs.get(name) for name in self.VARIANCE_NAMES},
        )
        condition = self._embed_variances(condition, variance_values)

        if infer:
            if self.use_shallow_diffusion:
                aux_mel_pred = self.aux_decoder(condition, infer=True)
                src_mel = aux_mel_pred
            else:
                aux_mel_pred = src_mel = None
            mel_pred = self.diffusion(condition, src_spec=src_mel, infer=True)
            frame_mask = (~padding_mask).float()[:, :, None]
            if aux_mel_pred is not None:
                aux_mel_pred = aux_mel_pred * frame_mask
            mel_pred = mel_pred * frame_mask
            return ShallowDiffusionOutput(aux_out=aux_mel_pred, diff_out=mel_pred)

        if self.use_shallow_diffusion:
            if self.train_aux_decoder:
                aux_cond = condition * self.aux_decoder_grad + condition.detach() * (1 - self.aux_decoder_grad)
                aux_out = self.aux_decoder(aux_cond, infer=False)
            else:
                aux_out = None
            if self.train_diffusion:
                diff_out = self.diffusion(condition, gt_spec=gt_mel, infer=False)
            else:
                diff_out = None
            return ShallowDiffusionOutput(aux_out=aux_out, diff_out=diff_out), variance_pred

        diff_out = self.diffusion(condition, gt_spec=gt_mel, infer=False)
        return ShallowDiffusionOutput(aux_out=None, diff_out=diff_out), variance_pred
