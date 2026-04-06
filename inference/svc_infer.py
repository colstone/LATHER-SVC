from __future__ import annotations

import json
import pathlib

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from basics.base_svs_infer import BaseSVSInfer
from modules.content_encoder import align_frame_rate
from modules.toplevel_svc import LatherSVC, ShallowDiffusionOutput
from modules.vocoders.registry import get_vocoder_cls
from utils import load_ckpt
from utils.hparams import hparams, set_hparams
from utils.infer_utils import resample_align_curve, save_wav


class SVCInfer(BaseSVSInfer):
    def __init__(self, config_path, exp_name='', device=None, load_vocoder=True, ckpt_steps=None):
        self.config_path = str(config_path)
        self.exp_name = exp_name
        self.ckpt_steps = ckpt_steps
        self._activate_hparams()
        super().__init__(device=device)

        self.spk_map = {}
        if hparams.get('use_spk_id', False):
            spk_map_path = pathlib.Path(hparams['work_dir']) / 'spk_map.json'
            if not spk_map_path.exists():
                raise FileNotFoundError(f'Speaker map not found: {spk_map_path}')
            with open(spk_map_path, 'r', encoding='utf-8') as f:
                self.spk_map = json.load(f)
        self.model = self.build_model(ckpt_steps=ckpt_steps)
        self.vocoder = self.build_vocoder() if load_vocoder else None

    def _activate_hparams(self):
        set_hparams(config=self.config_path, exp_name=self.exp_name, print_hparams=False)

    def build_model(self, ckpt_steps=None):
        self._activate_hparams()
        model = LatherSVC(out_dims=hparams['audio_num_mel_bins']).eval().to(self.device)
        load_ckpt(
            model,
            hparams['work_dir'],
            ckpt_steps=ckpt_steps,
            prefix_in_ckpt='model',
            strict=True,
            device=self.device,
        )
        return model

    def build_vocoder(self):
        self._activate_hparams()
        vocoder = get_vocoder_cls(hparams)()
        vocoder.to_device(self.device)
        return vocoder

    def speaker_names(self):
        if not self.spk_map:
            return []
        return [name for name, _ in sorted(self.spk_map.items(), key=lambda x: x[1])]

    def resolve_speaker(self, speaker):
        if not hparams.get('use_spk_id', False):
            return None
        if speaker is None or speaker == '':
            if len(self.spk_map) == 1:
                return next(iter(self.spk_map.values()))
            raise ValueError('speaker is required for multi-speaker inference.')
        if isinstance(speaker, str) and speaker in self.spk_map:
            return self.spk_map[speaker]
        try:
            speaker_id = int(speaker)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Unknown speaker: {speaker}') from exc
        if speaker_id not in self.spk_map.values():
            raise ValueError(f'Speaker id {speaker_id} not found in speaker map.')
        return speaker_id

    def _prepare_vocoder_inputs(self, mel: torch.Tensor, f0: torch.Tensor):
        target_sr = int(hparams.get('vocoder_sample_rate', hparams['audio_sample_rate']))
        target_hop = int(hparams.get('vocoder_hop_size', hparams['hop_size']))
        if target_sr == hparams['audio_sample_rate'] and target_hop == hparams['hop_size']:
            return mel, f0

        source_timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        target_timestep = target_hop / target_sr
        target_length = max(1, int(round(mel.shape[1] * source_timestep / target_timestep)))
        mel = F.interpolate(mel.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        f0_np = resample_align_curve(
            f0[0].detach().cpu().numpy().astype(np.float32),
            source_timestep,
            target_timestep,
            target_length,
        )
        f0 = torch.from_numpy(f0_np).to(mel.device)[None]
        return mel, f0

    def preprocess_input(self, audio_path: str, speaker=None, pitch_shift: float = 0.0):
        self._activate_hparams()
        waveform, _ = librosa.load(audio_path, sr=hparams['audio_sample_rate'], mono=True)
        mel_length = get_mel_length(waveform)
        waveform_16k, _ = librosa.load(audio_path, sr=16000, mono=True)

        audio_tensor = torch.from_numpy(waveform_16k).float().to(self.device)[None]
        content = self.model.content_encoder(audio_tensor)
        content = align_frame_rate(content, mel_length)

        pitch_extractor = initialize_pitch_extractor(self.device)
        f0, _ = pitch_extractor.get_pitch(
            waveform,
            samplerate=hparams['audio_sample_rate'],
            length=mel_length,
            hop_size=hparams['hop_size'],
            f0_min=hparams['f0_min'],
            f0_max=hparams['f0_max'],
            interp_uv=True,
        )
        if pitch_shift:
            f0 = f0 * (2 ** (pitch_shift / 12.0))

        batch = {
            'contentvec': content,
            'f0': torch.from_numpy(f0.astype(np.float32)).to(self.device)[None],
            'mel_lengths': torch.LongTensor([mel_length]).to(self.device),
        }
        if hparams.get('use_spk_id', False):
            batch['spk_ids'] = torch.LongTensor([self.resolve_speaker(speaker)]).to(self.device)
        return batch

    @torch.no_grad()
    def forward_model(self, sample):
        self._activate_hparams()
        output: ShallowDiffusionOutput = self.model(
            sample['contentvec'],
            sample['f0'],
            sample['mel_lengths'],
            spk_id=sample.get('spk_ids'),
            infer=True,
        )
        return output.diff_out

    @torch.no_grad()
    def run_vocoder(self, spec, f0):
        self._activate_hparams()
        spec, f0 = self._prepare_vocoder_inputs(spec, f0)
        return self.vocoder.spec2wav_torch(spec, f0=f0)

    def run_single(self, audio_path, speaker=None, pitch_shift: float = 0.0, output_path=None):
        batch = self.preprocess_input(audio_path, speaker=speaker, pitch_shift=pitch_shift)
        mel_pred = self.forward_model(batch)
        wav = self.run_vocoder(mel_pred, batch['f0']).detach().cpu().numpy()
        sample_rate = int(hparams.get('vocoder_sample_rate', hparams['audio_sample_rate']))
        if output_path is not None:
            output_path = pathlib.Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_wav(wav, str(output_path), sample_rate)
        return wav.astype(np.float32), sample_rate


def initialize_pitch_extractor(device):
    from modules.pe import initialize_pe

    return initialize_pe(device)


def get_mel_length(waveform: np.ndarray) -> int:
    from utils.binarizer_utils import get_mel_torch

    mel = get_mel_torch(
        waveform,
        hparams['audio_sample_rate'],
        num_mel_bins=hparams['audio_num_mel_bins'],
        hop_size=hparams['hop_size'],
        win_size=hparams['win_size'],
        fft_size=hparams['fft_size'],
        fmin=hparams['fmin'],
        fmax=hparams['fmax'],
        device='cpu',
    )
    return int(mel.shape[0])
