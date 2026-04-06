from __future__ import annotations

import json
import pathlib
import pickle
import random
from copy import deepcopy

import librosa
import numpy as np
import torch
from tqdm import tqdm

from modules.content_encoder import ContentVecExtractor
from modules.pe import initialize_pe
from utils.binarizer_utils import (
    SinusoidalSmoothingConv1d,
    get_breathiness,
    get_mel_torch,
    get_tension_base_harmonic,
    get_voicing,
)
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.multiprocess_utils import chunked_multiprocess_run

SVC_ITEM_ATTRIBUTES = [
    'spk_id',
    'mel',
    'f0',
    'contentvec',
    'breathiness',
    'voicing',
    'tension',
]

pitch_extractor = None
contentvec_extractor = None
breathiness_smooth = None
voicing_smooth = None
tension_smooth = None


class SVCBinarizer:
    def __init__(self):
        self.datasets = hparams['datasets']
        self.raw_data_dirs = [pathlib.Path(ds['raw_data_dir']) for ds in self.datasets]
        self.binary_data_dir = pathlib.Path(hparams['binary_data_dir'])
        self.data_attrs = SVC_ITEM_ATTRIBUTES
        self.binarization_args = hparams['binarization_args']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestep = hparams['hop_size'] / hparams['audio_sample_rate']

        self.spk_map = {}
        self.spk_ids = None
        self.items = {}
        self.item_names: list[str] = []
        self._train_item_names: list[str] = []
        self._valid_item_names: list[str] = []
        self.build_spk_map()

    def build_spk_map(self):
        spk_ids = [ds.get('spk_id') for ds in self.datasets]
        assigned_ids = {spk_id for spk_id in spk_ids if spk_id is not None}
        next_id = 0
        for idx in range(len(spk_ids)):
            if spk_ids[idx] is not None:
                continue
            while next_id in assigned_ids:
                next_id += 1
            spk_ids[idx] = next_id
            assigned_ids.add(next_id)
        if spk_ids:
            assert max(spk_ids) < hparams['num_spk'], (
                f'Assigned speaker ids {spk_ids} exceed num_spk={hparams["num_spk"]}.'
            )
        for spk_id, dataset in zip(spk_ids, self.datasets):
            spk_name = dataset['speaker']
            if spk_name in self.spk_map and self.spk_map[spk_name] != spk_id:
                raise ValueError(
                    f'Conflicting speaker id assignment for {spk_name}: '
                    f'{self.spk_map[spk_name]} vs {spk_id}'
                )
            self.spk_map[spk_name] = spk_id
        self.spk_ids = spk_ids
        print('| spk_map:', self.spk_map)

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id: int, spk: str) -> dict[str, dict]:
        meta = {}
        for wav_fn in sorted(raw_data_dir.rglob('*.wav')):
            rel_name = wav_fn.relative_to(raw_data_dir).with_suffix('').as_posix()
            item_name = f'{ds_id}:{rel_name}'
            meta[item_name] = {
                'wav_fn': str(wav_fn),
                'spk_id': self.spk_map[spk],
                'spk_name': spk,
                'name': rel_name,
                'ph_text': '',
            }
        return meta

    def split_train_valid_set(self, prefixes: list[str]):
        prefixes = {str(prefix): 1 for prefix in prefixes}
        valid_item_names = {}

        for prefix in deepcopy(prefixes):
            if prefix in self.item_names:
                valid_item_names[prefix] = 1
                prefixes.pop(prefix)
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.split(':', 1)[-1] == prefix:
                    valid_item_names[name] = 1
                    matched = True
            if matched:
                prefixes.pop(prefix)
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.startswith(prefix):
                    valid_item_names[name] = 1
                    matched = True
            if matched:
                prefixes.pop(prefix)
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.split(':', 1)[-1].startswith(prefix):
                    valid_item_names[name] = 1
                    matched = True
            if matched:
                prefixes.pop(prefix)

        valid_item_names = list(valid_item_names.keys())
        if not valid_item_names:
            seen_speakers = set()
            for item_name in self.item_names:
                spk_name = self.items[item_name]['spk_name']
                if spk_name not in seen_speakers:
                    valid_item_names.append(item_name)
                    seen_speakers.add(spk_name)
            if len(valid_item_names) >= len(self.item_names):
                if len(self.item_names) < 2:
                    raise RuntimeError('At least two wav files are required for SVC binarization.')
                valid_item_names = [self.item_names[0]]

        train_item_names = [x for x in self.item_names if x not in set(valid_item_names)]
        if not train_item_names:
            raise RuntimeError('Training split is empty. Please provide more than one wav file.')
        return train_item_names, valid_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._valid_item_names

    def meta_data_iterator(self, prefix):
        item_names = self.train_item_names if prefix == 'train' else self.valid_item_names
        for item_name in item_names:
            yield item_name, self.items[item_name]

    def process(self):
        test_prefixes = []
        for ds_id, dataset in enumerate(self.datasets):
            items = self.load_meta_data(
                pathlib.Path(dataset['raw_data_dir']),
                ds_id=ds_id,
                spk=dataset['speaker'],
            )
            self.items.update(items)
            test_prefixes.extend(
                f'{ds_id}:{prefix}'
                for prefix in dataset.get('test_prefixes', [])
            )
        self.item_names = sorted(self.items.keys())
        self._train_item_names, self._valid_item_names = self.split_train_valid_set(test_prefixes)

        if self.binarization_args.get('shuffle', False):
            random.shuffle(self.item_names)

        self.binary_data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.binary_data_dir / 'spk_map.json', 'w', encoding='utf-8') as f:
            json.dump(self.spk_map, f, ensure_ascii=False, indent=2)
        with open(self.binary_data_dir / 'lang_map.json', 'w', encoding='utf-8') as f:
            json.dump({}, f)

        self.process_dataset('valid')
        self.process_dataset('train', num_workers=int(self.binarization_args.get('num_workers', 0)))

    def process_dataset(self, prefix: str, num_workers: int = 0):
        builder = IndexedDatasetBuilder(self.binary_data_dir, prefix=prefix, allowed_attr=self.data_attrs)
        total_sec = {k: 0.0 for k in self.spk_map}
        extra_info = {'names': {}, 'ph_texts': {}, 'spk_ids': {}, 'spk_names': {}, 'lengths': {}}
        max_no = -1
        args = [[item_name, meta_data, self.binarization_args] for item_name, meta_data in self.meta_data_iterator(prefix)]

        def register_item(item):
            nonlocal max_no
            if item is None:
                return
            item_no = builder.add_item(item)
            max_no = max(max_no, item_no)
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    if key not in extra_info:
                        extra_info[key] = {}
                    extra_info[key][item_no] = value.shape[1] if key == 'contentvec' else value.shape[0]
            extra_info['names'][item_no] = item['name'].split(':', 1)[-1]
            extra_info['ph_texts'][item_no] = item['ph_text']
            extra_info['spk_ids'][item_no] = item['spk_id']
            extra_info['spk_names'][item_no] = item['spk_name']
            extra_info['lengths'][item_no] = item['length']
            total_sec[item['spk_name']] += item['seconds']

        try:
            if num_workers > 0:
                for item in tqdm(
                        chunked_multiprocess_run(self.process_item, args, num_workers=num_workers),
                        total=len(args)
                ):
                    register_item(item)
            else:
                for arg in tqdm(args):
                    register_item(self.process_item(*arg))
            for key in extra_info:
                assert set(extra_info[key]) == set(range(max_no + 1)), f'Item numbering is not consecutive for {key}.'
                extra_info[key] = [value for _, value in sorted(extra_info[key].items(), key=lambda x: x[0])]
        except KeyboardInterrupt:
            builder.finalize()
            raise

        builder.finalize()
        if prefix == 'train':
            extra_info.pop('names')
            extra_info.pop('ph_texts')
            extra_info.pop('spk_names')
        with open(self.binary_data_dir / f'{prefix}.meta', 'wb') as f:
            pickle.dump(extra_info, f)
        print(f'| {prefix} total duration: {sum(total_sec.values()):.2f}s')
        print('| {0} respective duration: {1}'.format(
            prefix,
            ', '.join(f'{k}={v:.2f}s' for k, v in total_sec.items())
        ))

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        waveform, _ = librosa.load(meta_data['wav_fn'], sr=hparams['audio_sample_rate'], mono=True)
        mel = get_mel_torch(
            waveform,
            hparams['audio_sample_rate'],
            num_mel_bins=hparams['audio_num_mel_bins'],
            hop_size=hparams['hop_size'],
            win_size=hparams['win_size'],
            fft_size=hparams['fft_size'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            device=self.device,
        )
        length = int(mel.shape[0])
        seconds = length * hparams['hop_size'] / hparams['audio_sample_rate']

        global pitch_extractor
        if pitch_extractor is None:
            pitch_extractor = initialize_pe(self.device)
        f0, uv = pitch_extractor.get_pitch(
            waveform,
            samplerate=hparams['audio_sample_rate'],
            length=length,
            hop_size=hparams['hop_size'],
            f0_min=hparams['f0_min'],
            f0_max=hparams['f0_max'],
            interp_uv=True,
        )
        if uv.all():
            print(f"Skipped '{item_name}': empty f0")
            return None

        dec_waveform = DecomposedWaveform(
            waveform,
            samplerate=hparams['audio_sample_rate'],
            f0=f0 * ~uv,
            hop_size=hparams['hop_size'],
            fft_size=hparams['fft_size'],
            win_size=hparams['win_size'],
            algorithm=hparams['hnsep'],
        )

        breathiness = get_breathiness(dec_waveform, None, None, length=length)
        voicing = get_voicing(dec_waveform, None, None, length=length)
        tension = get_tension_base_harmonic(dec_waveform, None, None, length=length, domain='logit')

        global breathiness_smooth
        if breathiness_smooth is None:
            breathiness_smooth = SinusoidalSmoothingConv1d(
                round(hparams['breathiness_smooth_width'] / self.timestep)
            ).eval().to(self.device)
        global voicing_smooth
        if voicing_smooth is None:
            voicing_smooth = SinusoidalSmoothingConv1d(
                round(hparams['voicing_smooth_width'] / self.timestep)
            ).eval().to(self.device)
        global tension_smooth
        if tension_smooth is None:
            tension_smooth = SinusoidalSmoothingConv1d(
                round(hparams['tension_smooth_width'] / self.timestep)
            ).eval().to(self.device)

        breathiness = breathiness_smooth(torch.from_numpy(breathiness).to(self.device)[None])[0]
        voicing = voicing_smooth(torch.from_numpy(voicing).to(self.device)[None])[0]
        tension = tension_smooth(torch.from_numpy(tension).to(self.device)[None])[0]
        if tension.isnan().any():
            print(f"Skipped '{item_name}': tension contains NaN")
            return None

        waveform_16k, _ = librosa.load(meta_data['wav_fn'], sr=16000, mono=True)
        global contentvec_extractor
        if contentvec_extractor is None:
            contentvec_extractor = ContentVecExtractor(output_dim=hparams['hidden_size']).to(self.device)
            contentvec_extractor.eval()
        cached = contentvec_extractor.extract_hidden_states(
            torch.from_numpy(waveform_16k).float().to(self.device)[None]
        )[0]

        return {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'spk_id': meta_data['spk_id'],
            'spk_name': meta_data['spk_name'],
            'seconds': seconds,
            'length': length,
            'ph_text': meta_data['ph_text'],
            'mel': mel.astype(np.float32),
            'f0': f0.astype(np.float32),
            'contentvec': cached.cpu().numpy().astype(np.float32),
            'breathiness': breathiness.cpu().numpy().astype(np.float32),
            'voicing': voicing.cpu().numpy().astype(np.float32),
            'tension': tension.cpu().numpy().astype(np.float32),
        }
