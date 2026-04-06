import argparse
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from inference.svc_infer import SVCInfer


def parse_args():
    parser = argparse.ArgumentParser(description='Run LATHER-SVC inference.')
    parser.add_argument('--config', required=True, help='Path to config.yaml or a training config.')
    parser.add_argument('--exp_name', default='', help='Experiment name when using a training config.')
    parser.add_argument('--input', required=True, help='Input wav path.')
    parser.add_argument('--output', required=True, help='Output wav path.')
    parser.add_argument('--speaker', default=None, help='Speaker name or speaker id.')
    parser.add_argument('--pitch_shift', type=float, default=0.0, help='Pitch shift in semitones.')
    parser.add_argument('--ckpt', type=int, default=None, help='Checkpoint step override.')
    parser.add_argument('--device', default=None, help='Inference device, e.g. cuda or cpu.')
    return parser.parse_args()


def main():
    args = parse_args()
    infer = SVCInfer(
        config_path=args.config,
        exp_name=args.exp_name,
        device=args.device,
        ckpt_steps=args.ckpt,
    )
    infer.run_single(
        audio_path=args.input,
        speaker=args.speaker,
        pitch_shift=args.pitch_shift,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
