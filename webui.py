import argparse
import os
import sys
from pathlib import Path

import gradio as gr

root_dir = Path(__file__).resolve().parent
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from inference.svc_infer import SVCInfer


class WebUIState:
    def __init__(self, configs: dict[str, tuple[str, str]]):
        self.configs = configs
        self.inferers = {}

    def get_inferer(self, mode: str) -> SVCInfer:
        if mode not in self.inferers:
            config_path, exp_name = self.configs[mode]
            self.inferers[mode] = SVCInfer(config_path=config_path, exp_name=exp_name)
        return self.inferers[mode]

    def speaker_choices(self, mode: str):
        inferer = self.get_inferer(mode)
        return inferer.speaker_names()


def run_inference(state: WebUIState, input_audio, mode, speaker, pitch_shift):
    if not input_audio:
        return None, 'Please provide an input wav file.'
    inferer = state.get_inferer(mode)
    wav, sample_rate = inferer.run_single(
        audio_path=input_audio,
        speaker=speaker,
        pitch_shift=float(pitch_shift),
    )
    return (sample_rate, wav), f'Inference finished with mode={mode}'


def update_speakers(state: WebUIState, mode: str):
    speakers = state.speaker_choices(mode)
    value = speakers[0] if speakers else None
    return gr.Dropdown.update(choices=speakers, value=value)


def build_ui(state: WebUIState):
    modes = list(state.configs.keys())
    default_mode = modes[0]
    default_speakers = state.speaker_choices(default_mode)
    default_speaker = default_speakers[0] if default_speakers else None

    with gr.Blocks(theme=gr.themes.Soft(primary_hue='blue', secondary_hue='slate')) as demo:
        gr.Markdown('# LATHER-SVC')
        with gr.Row():
            input_audio = gr.Audio(label='Source Audio', source='upload', type='filepath')
            output_audio = gr.Audio(label='Converted Audio', type='numpy')
        mode = gr.Radio(choices=modes, value=default_mode, label='Mode')
        speaker = gr.Dropdown(choices=default_speakers, value=default_speaker, label='Speaker')
        pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label='Pitch Shift')
        run_button = gr.Button('Run Inference', variant='primary')
        status = gr.Textbox(label='Status', interactive=False)

        mode.change(fn=lambda selected_mode: update_speakers(state, selected_mode), inputs=mode, outputs=speaker)
        run_button.click(
            fn=lambda audio, selected_mode, selected_speaker, shift: run_inference(
                state, audio, selected_mode, selected_speaker, shift
            ),
            inputs=[input_audio, mode, speaker, pitch_shift],
            outputs=[output_audio, status],
        )
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description='Launch LATHER-SVC WebUI.')
    parser.add_argument('--quality_config', default='', help='Config path for quality mode.')
    parser.add_argument('--quality_exp_name', default='', help='Experiment name for quality mode.')
    parser.add_argument('--fast_config', default='', help='Config path for fast mode.')
    parser.add_argument('--fast_exp_name', default='', help='Experiment name for fast mode.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    configs = {}
    if args.quality_config:
        configs['quality'] = (args.quality_config, args.quality_exp_name)
    if args.fast_config:
        configs['fast'] = (args.fast_config, args.fast_exp_name)
    if not configs:
        raise ValueError('At least one of --quality_config or --fast_config must be provided.')

    state = WebUIState(configs)
    demo = build_ui(state)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == '__main__':
    main()
