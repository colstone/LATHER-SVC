import argparse
import pathlib
import re

import torch

ID_LIST_PATTERN = r'(\d+)?(,\d+)*,?'

def _parse_csv_to_ints(raw: str):
    return [int(part.strip()) for part in raw.split(',') if part.strip()]

def _split_valid_ids(candidates, upper_bound):
    universe = set(range(upper_bound))
    valid = sorted({c for c in candidates if c in universe})
    invalid = sorted({c for c in candidates if c not in universe})
    return valid, invalid

def _compute_drop_ids(num_spk):
    if args.drop_all:
        return list(range(num_spk)), []
    if args.drop is not None:
        requested = _parse_csv_to_ints(args.drop)
        valid, invalid = _split_valid_ids(requested, num_spk)
        return valid, invalid
    requested = _parse_csv_to_ints(args.retain)
    valid, invalid = _split_valid_ids(requested, num_spk)
    retain_ids = set(valid)
    drop_ids = sorted(set(range(num_spk)) - retain_ids)
    return drop_ids, invalid

def modify_spk_embed(spk_embed):
    global _drop_summary_reported
    num_spk, hidden_size = spk_embed.shape
    drop_ids, invalid_ids = _compute_drop_ids(num_spk)
    if invalid_ids and not _invalid_reported[0]:
        print(f"| warning: ignoring speaker ids out of range: {invalid_ids}")
        _invalid_reported[0] = True
    if not drop_ids:
        if not _drop_summary_reported:
            print('| info: no speaker ids matched the request; embeddings left unchanged.')
            _drop_summary_reported = True
        return
    if args.fill == 'cyclic' and len(drop_ids) == num_spk:
        raise ValueError('Cannot use cyclic fill when all speakers are dropped.')
    drop_tensor = torch.tensor(drop_ids, dtype=torch.long, device=spk_embed.device)
    if args.fill == 'zeros':
        spk_embed.index_fill_(0, drop_tensor, 0.0)
    elif args.fill == 'random':
        spk_embed[drop_tensor] = torch.randn(
            (len(drop_ids), hidden_size),
            dtype=spk_embed.dtype,
            device=spk_embed.device
        )
    elif args.fill == 'mean':
        mean_vec = spk_embed.mean(dim=0, keepdim=True).clone()
        spk_embed[drop_tensor] = mean_vec
    elif args.fill == 'cyclic':
        retain_ids = sorted(set(range(num_spk)) - set(drop_ids))
        if not retain_ids:
            raise ValueError('Cannot use cyclic fill when all speakers are dropped.')
        source_rows = torch.stack([
            spk_embed[retain_ids[i % len(retain_ids)]].clone()
            for i in range(len(drop_ids))
        ], dim=0)
        spk_embed[drop_tensor] = source_rows
    else:
        raise ValueError(f'Unknown fill method: {args.fill}')
    if not _drop_summary_reported:
        print(f"| info: dropped {len(drop_ids)} speakers (ids={drop_ids}) with fill='{args.fill}'")
        _drop_summary_reported = True


parser = argparse.ArgumentParser(description='Drop or edit spk_embed in a checkpoint.')
parser.add_argument('input', type=str, help='Path to the input file')
parser.add_argument('output', type=str, help='Path to the output file')
drop_retain_group = parser.add_mutually_exclusive_group(required=True)
drop_retain_group.add_argument('--drop', type=str, required=False, metavar='ID,ID,...',
                               help='Drop specific speaker IDs.')
drop_retain_group.add_argument('--retain', type=str, required=False, metavar='ID,ID,...',
                               help='Retain specific speaker IDs and drop all the others.')
drop_retain_group.add_argument('--drop-all', action='store_true',
                               help='Drop every speaker embedding in the checkpoint.')
parser.add_argument('--fill', type=str, required=False, default='zeros', metavar='METHOD',
                    choices=['zeros', 'random', 'mean', 'cyclic'],
                    help='Specify a filling method for the dropped embedding. '
                         'Available methods: zeros, random, mean, cyclic')
parser.add_argument('--overwrite', required=False, default=False,
                    action='store_true', help='Overwrite if the output file exists.')
args = parser.parse_args()

if args.drop and not re.fullmatch(ID_LIST_PATTERN, args.drop):
    print(f"Invalid format for --drop: '{args.drop}'")
    exit(-1)
if args.retain and not re.fullmatch(ID_LIST_PATTERN, args.retain):
    print(f"Invalid format for --retain: '{args.retain}'")
    exit(-1)

_drop_summary_reported = False
_invalid_reported = [False]

input_ckpt = pathlib.Path(args.input).resolve()
output_ckpt = pathlib.Path(args.output).resolve()
assert input_ckpt.exists(), 'The input file does not exist.'
assert args.overwrite or not output_ckpt.exists(), \
    'The output file already exists or is the same as the input file.\n' \
    'This is not recommended because spk_embed dropping scripts may not be stable, ' \
    'and you may be at risk of losing your model.\n' \
    'If you are sure to OVERWRITE the existing file, please re-run this script with the \'--overwrite\' argument.'

ckpt_loaded = torch.load(input_ckpt, map_location='cpu')
state_dict = ckpt_loaded['state_dict']
try:
    if 'model.fs2.spk_embed.weight' in state_dict:
        modify_spk_embed(state_dict['model.fs2.spk_embed.weight'])
    if 'model.spk_embed.weight' in state_dict:
        modify_spk_embed(state_dict['model.spk_embed.weight'])
except ValueError as exc:
    print(f'Error: {exc}')
    exit(-1)

torch.save(ckpt_loaded, output_ckpt)
