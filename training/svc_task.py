import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

import utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_vocoder import BaseVocoder
from modules.aux_decoder import build_aux_loss
from modules.content_encoder import align_frame_rate
from modules.losses import RectifiedFlowLoss
from modules.toplevel_svc import LatherSVC, ShallowDiffusionOutput
from modules.vocoders.registry import get_vocoder_cls
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve
from utils.plot import spec_to_figure

matplotlib.use('Agg')


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device)[None]
    return positions < lengths[:, None]


class SVCDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        super().__init__(prefix, hparams['dataset_size_key'], preload)

    def collater(self, samples):
        batch = super().collater(samples)
        if batch['size'] == 0:
            return batch

        mel = utils.collate_nd([s['mel'] for s in samples], 0.0)
        f0 = utils.collate_nd([s['f0'] for s in samples], 0.0)
        breathiness = utils.collate_nd([s['breathiness'] for s in samples], 0.0)
        voicing = utils.collate_nd([s['voicing'] for s in samples], 0.0)
        tension = utils.collate_nd([s['tension'] for s in samples], 0.0)
        contentvec = utils.collate_nd([s['contentvec'].transpose(0, 1) for s in samples], 0.0).permute(0, 2, 1, 3)
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])
        contentvec_lengths = torch.LongTensor([s['contentvec'].shape[1] for s in samples])
        spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
        batch.update({
            'mel': mel,
            'f0': f0,
            'breathiness': breathiness,
            'voicing': voicing,
            'tension': tension,
            'contentvec': contentvec,
            'mel_lengths': mel_lengths,
            'contentvec_lengths': contentvec_lengths,
            'spk_ids': spk_ids,
        })
        return batch


class SVCTask(BaseTask):
    variance_names = ('breathiness', 'voicing', 'tension')

    def __init__(self):
        super().__init__()
        self.dataset_cls = SVCDataset
        self.diffusion_type = hparams['diffusion_type']
        assert self.diffusion_type == 'reflow', f'Unsupported diffusion type: {self.diffusion_type}'
        self.use_shallow_diffusion = hparams['use_shallow_diffusion']
        if self.use_shallow_diffusion:
            self.shallow_args = hparams['shallow_diffusion_args']
            self.train_aux_decoder = self.shallow_args['train_aux_decoder']
            self.train_diffusion = self.shallow_args['train_diffusion']

        self.use_vocoder = hparams['infer'] or hparams['val_with_vocoder']
        if self.use_vocoder:
            self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.logged_gt_wav = set()
        self.lambda_var_loss = float(hparams.get('lambda_var_loss', 0.5))
        super()._finish_init()

    def _build_model(self):
        return LatherSVC(out_dims=hparams['audio_num_mel_bins'])

    def build_losses_and_metrics(self):
        if self.use_shallow_diffusion:
            self.aux_mel_loss = build_aux_loss(self.shallow_args['aux_decoder_arch'])
            self.lambda_aux_mel_loss = hparams['lambda_aux_mel_loss']
            self.register_validation_loss('aux_mel_loss')
        self.mel_loss = RectifiedFlowLoss(
            loss_type=hparams['main_loss_type'],
            log_norm=hparams['main_loss_log_norm'],
        )
        self.register_validation_loss('mel_loss')
        self.register_validation_loss('var_loss')

    def masked_mse(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = (prediction - target) ** 2
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

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

    def prepare_content_features(self, sample):
        features = []
        for batch_idx in range(sample['size']):
            content_length = int(sample['contentvec_lengths'][batch_idx].item())
            mel_length = int(sample['mel_lengths'][batch_idx].item())
            cached = sample['contentvec'][batch_idx:batch_idx + 1, :, :content_length, :]
            projected = self.model.content_encoder.from_cached_features(cached)
            projected = align_frame_rate(projected, mel_length)
            features.append(projected[0])
        return utils.collate_nd(features, 0.0)

    def run_model(self, sample, infer=False):
        target = sample['mel']
        spk_id = sample['spk_ids'] if hparams.get('use_spk_id', False) else None
        content_features = self.prepare_content_features(sample)

        if infer:
            return self.model(
                content_features,
                sample['f0'],
                sample['mel_lengths'],
                spk_id=spk_id,
                infer=True,
            )

        model_out, variance_pred = self.model(
            content_features,
            sample['f0'],
            sample['mel_lengths'],
            spk_id=spk_id,
            gt_mel=target,
            infer=False,
            breathiness=sample['breathiness'],
            voicing=sample['voicing'],
            tension=sample['tension'],
        )

        losses = {}
        if model_out.aux_out is not None:
            norm_gt = self.model.aux_decoder.norm_spec(target)
            losses['aux_mel_loss'] = self.lambda_aux_mel_loss * self.aux_mel_loss(model_out.aux_out, norm_gt)

        non_padding = lengths_to_mask(sample['mel_lengths'], target.size(1)).unsqueeze(-1).float()
        if model_out.diff_out is not None:
            v_pred, v_gt, t = model_out.diff_out
            losses['mel_loss'] = self.mel_loss(v_pred, v_gt, t=t, non_padding=non_padding)

        frame_mask = non_padding.squeeze(-1)
        variance_loss = 0.0
        for name in self.variance_names:
            variance_loss = variance_loss + self.masked_mse(variance_pred[name], sample[name], frame_mask)
        losses['var_loss'] = self.lambda_var_loss * (variance_loss / len(self.variance_names))
        return losses

    def on_train_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)

    def _on_validation_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        if sample['size'] > 0 and min(sample['indices']) < hparams['num_valid_plots']:
            mel_out: ShallowDiffusionOutput = self.run_model(sample, infer=True)
            for i in range(len(sample['indices'])):
                data_idx = sample['indices'][i].item()
                if data_idx < hparams['num_valid_plots']:
                    if self.use_vocoder:
                        self.plot_wav(
                            data_idx,
                            sample['mel'][i],
                            mel_out.aux_out[i] if mel_out.aux_out is not None else None,
                            mel_out.diff_out[i],
                            sample['f0'][i],
                        )
                    if mel_out.aux_out is not None:
                        self.plot_mel(data_idx, sample['mel'][i], mel_out.aux_out[i], 'auxmel')
                    if mel_out.diff_out is not None:
                        self.plot_mel(data_idx, sample['mel'][i], mel_out.diff_out[i], 'diffmel')
        return losses, sample['size']

    def plot_wav(self, data_idx, gt_mel, aux_mel, diff_mel, f0):
        f0_len = self.valid_dataset.metadata['f0'][data_idx]
        mel_len = self.valid_dataset.metadata['mel'][data_idx]
        gt_mel = gt_mel[:mel_len].unsqueeze(0)
        f0 = f0[:f0_len].unsqueeze(0)
        gt_mel, gt_f0 = self._prepare_vocoder_inputs(gt_mel, f0)
        if data_idx not in self.logged_gt_wav:
            gt_wav = self.vocoder.spec2wav_torch(gt_mel, f0=gt_f0)
            self.logger.all_rank_experiment.add_audio(
                f'gt_{data_idx}',
                gt_wav,
                sample_rate=int(hparams.get('vocoder_sample_rate', hparams['audio_sample_rate'])),
                global_step=self.global_step,
            )
            self.logged_gt_wav.add(data_idx)
        if aux_mel is not None:
            aux_mel = aux_mel[:mel_len].unsqueeze(0)
            aux_mel, aux_f0 = self._prepare_vocoder_inputs(aux_mel, f0)
            aux_wav = self.vocoder.spec2wav_torch(aux_mel, f0=aux_f0)
            self.logger.all_rank_experiment.add_audio(
                f'aux_{data_idx}',
                aux_wav,
                sample_rate=int(hparams.get('vocoder_sample_rate', hparams['audio_sample_rate'])),
                global_step=self.global_step,
            )
        if diff_mel is not None:
            diff_mel = diff_mel[:mel_len].unsqueeze(0)
            diff_mel, diff_f0 = self._prepare_vocoder_inputs(diff_mel, f0)
            diff_wav = self.vocoder.spec2wav_torch(diff_mel, f0=diff_f0)
            self.logger.all_rank_experiment.add_audio(
                f'diff_{data_idx}',
                diff_wav,
                sample_rate=int(hparams.get('vocoder_sample_rate', hparams['audio_sample_rate'])),
                global_step=self.global_step,
            )

    def plot_mel(self, data_idx, gt_spec, out_spec, name_prefix='mel'):
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        mel_len = self.valid_dataset.metadata['mel'][data_idx]
        spec_cat = torch.cat([(out_spec - gt_spec).abs() + vmin, gt_spec, out_spec], -1)
        title_text = (
            f"{self.valid_dataset.metadata['spk_names'][data_idx]} - "
            f"{self.valid_dataset.metadata['names'][data_idx]}"
        )
        self.logger.all_rank_experiment.add_figure(
            f'{name_prefix}_{data_idx}',
            spec_to_figure(spec_cat[:mel_len], vmin, vmax, title_text),
            global_step=self.global_step,
        )
