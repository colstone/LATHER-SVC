from modules.vocoders import nsf_hifigan

try:  # Optional vocoders should not break the whole package.
    from modules.vocoders import ddsp
except Exception:
    ddsp = None

try:
    from modules.vocoders import refinegan
except Exception:
    refinegan = None
