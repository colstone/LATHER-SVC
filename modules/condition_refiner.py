from __future__ import annotations

import torch
import torch.nn.functional as F

from modules.fastspeech.tts_modules import FastSpeech2Encoder
from utils.hparams import hparams


class ConditionRefiner(FastSpeech2Encoder):
    def __init__(self):
        super().__init__(
            hidden_size=hparams['hidden_size'],
            num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'],
            ffn_act=hparams['ffn_act'],
            dropout=hparams['dropout'],
            num_heads=hparams['num_heads'],
            use_pos_embed=hparams['use_pos_embed'],
            rel_pos=hparams.get('rel_pos', False),
            use_rope=hparams.get('use_rope', False),
            attention_type=hparams.get('enc_attention_type', 'normal'),
        )

    def forward_embedding(self, condition, extra_embed=None, padding_mask=None):
        x = condition
        if extra_embed is not None:
            x = x + extra_embed
        if self.use_pos_embed and self.embed_positions is not None:
            if self.rel_pos:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(~padding_mask)
                x = x + positions
        return F.dropout(x, p=self.dropout, training=self.training)

    def forward(self, condition, padding_mask=None, attn_mask=None, return_hiddens=False):
        if padding_mask is None:
            padding_mask = condition.new_zeros(condition.size(0), condition.size(1), dtype=torch.bool)
        x = self.forward_embedding(condition, padding_mask=padding_mask)
        nonpadding = (~padding_mask).float()[:, :, None]
        x = x * nonpadding
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding
            if return_hiddens:
                hiddens.append(x)
        x = self.layer_norm(x) * nonpadding
        if return_hiddens:
            return torch.stack(hiddens, dim=0)
        return x
