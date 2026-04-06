import json
import pathlib
from typing import Any, Dict, Optional

import torch
import yaml

from basics.base_vocoder import BaseVocoder
from modules.refinegan.generator import RefineGANGenerator
from modules.vocoders.registry import register_vocoder
from utils.hparams import hparams


def _load_config(config_path: Optional[pathlib.Path]) -> Dict[str, Any]:
    if not config_path or not config_path.exists():
        return {}
    suffix = config_path.suffix.lower()
    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


def _select_ckpt_file(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser()
    if path.is_file():
        return path

    if path.is_dir():
        ckpts = sorted(path.glob("*.ckpt"))
        preferred = [
            c for c in ckpts if c.stem.endswith("_G") or "generator" in c.stem.lower()
        ]
        if preferred:
            return preferred[0]
        if ckpts:
            return ckpts[0]

    raise FileNotFoundError(f"RefineGAN checkpoint not found at {path}")


def _select_config_file(path: pathlib.Path) -> Optional[pathlib.Path]:
    search_dir = path if path.is_dir() else path.parent
    candidates = [
        search_dir / "config.json",
        search_dir / "config.yaml",
        search_dir / "config.yml",
        search_dir / "config_v1.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _extract_gen_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_args = cfg.get("model_args") or cfg.get("generator") or {}
    def pick(key: str, default: Any):
        if key in model_args:
            return model_args[key]
        if key in cfg:
            return cfg[key]
        return default

    return {
        "sampling_rate": cfg.get("audio_sample_rate")
        or cfg.get("sampling_rate")
        or model_args.get("sampling_rate")
        or hparams["audio_sample_rate"],
        "num_mels": cfg.get("audio_num_mel_bins")
        or cfg.get("num_mels")
        or model_args.get("num_mels")
        or hparams["audio_num_mel_bins"],
        "hop_length": cfg.get("hop_size")
        or cfg.get("hop_length")
        or model_args.get("hop_length")
        or hparams["hop_size"],
        "downsample_rates": tuple(pick("downsample_rates", (2, 2, 8, 8))),
        "upsample_rates": tuple(pick("upsample_rates", (8, 8, 2, 2))),
        "leaky_relu_slope": float(pick("leaky_relu_slope", 0.2)),
        "start_channels": int(pick("start_channels", 16)),
        "template_generator": pick("template_generator", "comb"),
    }


def _extract_meta(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Gather the fields we can use for consistency checks
    meta = {}
    for key, aliases in {
        "audio_sample_rate": ["audio_sample_rate", "sampling_rate"],
        "audio_num_mel_bins": ["audio_num_mel_bins", "num_mels"],
        "hop_size": ["hop_size", "hop_length"],
        "fft_size": ["fft_size", "n_fft"],
        "win_size": ["win_size", "win_length"],
        "fmin": ["fmin", "f_min"],
        "fmax": ["fmax", "f_max"],
    }.items():
        for alias in aliases:
            if alias in cfg:
                meta[key] = cfg[alias]
                break
        if key not in meta and "generator" in cfg and isinstance(cfg["generator"], dict):
            for alias in aliases:
                if alias in cfg["generator"]:
                    meta[key] = cfg["generator"][alias]
                    break
    return meta


@register_vocoder
class RefineGAN(BaseVocoder):
    def __init__(self):
        raw_path = pathlib.Path(hparams["vocoder_ckpt"])
        self.ckpt_path = _select_ckpt_file(raw_path)
        self.config_path = _select_config_file(self.ckpt_path)
        self.config = _load_config(self.config_path)
        self.meta = _extract_meta(self.config)

        gen_kwargs = _extract_gen_kwargs(self.config)
        self.generator = RefineGANGenerator(**gen_kwargs)
        self.device = torch.device("cpu")

        print(f"| Load RefineGAN generator: {self.ckpt_path}")
        self._load_generator_state(self.ckpt_path)

    def _load_generator_state(self, ckpt_path: pathlib.Path):
        obj = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = None
        if isinstance(obj, dict):
            if "generator" in obj:
                state_dict = obj["generator"]
            elif "state_dict" in obj:
                raw_sd = obj["state_dict"]
                gen_sd = {
                    k.split("generator.", 1)[1]: v
                    for k, v in raw_sd.items()
                    if k.startswith("generator.")
                }
                state_dict = gen_sd or raw_sd
        if state_dict is None:
            state_dict = obj

        missing, unexpected = self.generator.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"| warn: RefineGAN missing keys: {len(missing)} (showing 8): {missing[:8]}")
        if unexpected:
            print(
                f"| warn: RefineGAN unexpected keys: {len(unexpected)} (showing 8): {unexpected[:8]}"
            )

    def to_device(self, device):
        self.device = device
        self.generator.to(device)

    def get_device(self):
        return self.device

    def _prepare_mel(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.to(self.device)
        if mel.dim() != 3:
            raise ValueError(f"RefineGAN expects 3-D mel, got shape {mel.shape}")

        if mel.shape[1] == self.generator.mel_conv.in_channels:
            mel_ch = mel
        elif mel.shape[2] == self.generator.mel_conv.in_channels:
            mel_ch = mel.transpose(1, 2)
        else:
            raise ValueError(
                f"Mel channel mismatch: got {mel.shape}, expected {self.generator.mel_conv.in_channels} bins"
            )

        mel_base = hparams.get("mel_base", "e")
        if mel_base != "e":
            mel_ch = 2.302585 * mel_ch
        return mel_ch.float()

    def _prepare_f0(self, f0: torch.Tensor) -> torch.Tensor:
        if f0 is None:
            raise ValueError("RefineGAN requires f0 input.")
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        elif f0.dim() == 3 and f0.shape[1] != 1:
            # If shape is [B, T, 1], move channel dimension
            if f0.shape[2] == 1:
                f0 = f0.transpose(1, 2)
        if f0.dim() != 3:
            raise ValueError(f"Unexpected f0 shape for RefineGAN: {f0.shape}")
        return f0.to(self.device).float()

    def _warn_mismatch(self):
        for key, hp_key in [
            ("audio_sample_rate", "audio_sample_rate"),
            ("audio_num_mel_bins", "audio_num_mel_bins"),
            ("hop_size", "hop_size"),
            ("fft_size", "fft_size"),
            ("win_size", "win_size"),
            ("fmin", "fmin"),
            ("fmax", "fmax"),
        ]:
            target = self.meta.get(key)
            if target is None:
                continue
            hp_val = hparams.get(hp_key)
            if hp_val is not None and hp_val != target:
                print(f"Mismatch parameters: hparams['{hp_key}']={hp_val} != {target} (RefineGAN)")

    def spec2wav_torch(self, mel, **kwargs):
        self._warn_mismatch()
        mel_t = self._prepare_mel(mel)
        f0 = self._prepare_f0(kwargs.get("f0"))

        with torch.no_grad():
            audio = self.generator(mel_t, f0)  # [B, 1, T]
            audio = audio.squeeze(1).contiguous().view(-1)
        return audio

    def spec2wav(self, mel, **kwargs):
        mel_np = torch.tensor(mel).unsqueeze(0) if not torch.is_tensor(mel) else mel
        f0 = kwargs.get("f0")
        if f0 is None:
            raise ValueError("RefineGAN requires f0 input.")
        f0_t = torch.tensor(f0).unsqueeze(0) if not torch.is_tensor(f0) else f0
        with torch.no_grad():
            wav = self.spec2wav_torch(mel_np, f0=f0_t)
        return wav.cpu().numpy()
