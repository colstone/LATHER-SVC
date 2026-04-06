# LATHER-SVC

LATHER-SVC is a speaker conversion variant built on top of the DSRX / DiffSinger codebase. It follows the `LATHER-SVC-PLAN-V2.md` design closely, while fixing a few practical issues in the original plan:

- `BaseTask` now supports phoneme-free tasks through `use_phoneme_dictionary: false`.
- `SVCBinarizer` is rewritten around raw wav datasets instead of forcing the original SVS binarization path.
- Fast mode keeps the 24 kHz acoustic model path but resamples mel / f0 to the vocoder frame rate before waveform reconstruction.
- Optional vocoder imports are made tolerant so unrelated backends do not break NSF-HiFiGAN inference.

## Implemented Components

- `modules/content_encoder.py`
  - ContentVec extraction, cached multi-layer loading, layer attention, frame-rate alignment.
- `modules/condition_refiner.py`
  - FastSpeech2-style Transformer refiner over `content + pitch + speaker`.
- `modules/variance_predictor.py`
  - Shared Conv1d backbone with 3 heads for `breathiness / voicing / tension`.
- `modules/toplevel_svc.py`
  - Main LATHER-SVC acoustic model with shallow diffusion.
- `preprocessing/svc_binarizer.py`
  - Raw wav scanning, mel / f0 / variance extraction, ContentVec caching, indexed dataset export.
- `training/svc_task.py`
  - SVC dataset collation, training losses, validation plots, vocoder preview path.
- `inference/svc_infer.py`
  - End-to-end single file inference.
- `scripts/infer.py`
  - CLI inference entry.
- `webui.py`
  - Gradio UI for quality / fast checkpoints.

## Data Layout

Each dataset entry in `configs/svc_base.yaml` points to a speaker folder:

```yaml
datasets:
  - raw_data_dir: data/svc/raw
    speaker: speaker_A
    spk_id: 0
    test_prefixes: []
```

`SVCBinarizer` scans `raw_data_dir` recursively for `*.wav`. Speaker identity comes from the dataset entry, not from filenames.

## Configs

- `configs/config_quality.yaml`
  - 44.1 kHz, hop 512, 20 sampling steps, quality-first.
- `configs/config_fast.yaml`
  - 24 kHz, hop 300, 4 sampling steps, speed-first.
- `configs/svc_base.yaml`
  - Shared SVC settings, ContentVec, variance predictor, shallow diffusion, vocoder settings.

Update at least these fields before training:

- `datasets`
- `binary_data_dir`
- `num_spk`
- `contentvec.checkpoint_path`
- `pe_ckpt`
- `hnsep_ckpt`
- `vocoder_ckpt`

## Usage

Preprocess:

```bash
python scripts/binarize.py --config configs/config_quality.yaml
python scripts/binarize.py --config configs/config_fast.yaml
```

Train:

```bash
python scripts/train.py --config configs/config_quality.yaml --exp_name lather_quality
python scripts/train.py --config configs/config_fast.yaml --exp_name lather_fast
```

CLI inference:

```bash
python scripts/infer.py --config configs/config_quality.yaml \
  --exp_name lather_quality \
  --input source.wav \
  --output output.wav \
  --speaker speaker_A \
  --pitch_shift 0
```

WebUI:

```bash
python webui.py \
  --quality_config ckpt/lather_quality/config.yaml \
  --fast_config ckpt/lather_fast/config.yaml \
  --port 7860
```

## Notes

- Training caches ContentVec hidden states during binarization. Runtime training does not re-run ContentVec.
- Validation / inference resample mel and f0 to the vocoder frame rate when the acoustic model and vocoder use different hop sizes.
- The current implementation assumes NSF-HiFiGAN as the main vocoder path.
- ContentVec loading first tries `fairseq`, then falls back to `transformers.HubertModel`.

## Validation Status

Static syntax compilation passed for the newly added SVC modules, binarizer, task, inference CLI, and WebUI.

Full end-to-end runtime validation was not completed in the current environment because the local Python environment has dependency conflicts around `lightning`, `transformers` / `huggingface-hub`, and `librosa` / `numba`.
