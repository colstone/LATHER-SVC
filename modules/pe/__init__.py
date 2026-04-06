from utils import hparams

from .pm import ParselmouthPE
from .pw import HarvestPE
from .rmvpe import RMVPE


def initialize_pe(device=None):
    pe = hparams['pe']
    pe_ckpt = hparams['pe_ckpt']
    if pe == 'parselmouth':
        return ParselmouthPE()
    elif pe == 'rmvpe':
        return RMVPE(pe_ckpt, device=device)
    elif pe == 'harvest':
        return HarvestPE()
    else:
        raise ValueError(f" [x] Unknown f0 extractor: {pe}")
