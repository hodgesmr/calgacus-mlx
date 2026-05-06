"""Calgacus: text steganography via LLM rank choices.

Hide a meaningful text inside another coherent, plausible text of comparable
token length. Implementation of the Calgacus protocol (Norelli & Bronstein,
arXiv:2510.20075) on top of MLX.

The user-facing surface is the `calgacus` console script. For programmatic
use, see `calgacus.core` (encode/decode), `calgacus.keyfile` (load/save),
and `calgacus.model` (the MLX wrapper).
"""

import os

# Quiet noisy upstream output before any transformers/huggingface imports
# trigger. Set here (rather than in cli.py) so library users importing
# `calgacus` directly also get a clean stderr. `setdefault` so a user can
# still opt in by setting these in their environment first.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from importlib.metadata import version

__version__ = version("calgacus-mlx")
