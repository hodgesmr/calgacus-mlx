"""Calgacus: text steganography via LLM rank choices.

Hide a meaningful text inside another coherent, plausible text of comparable
token length. Implementation of the Calgacus protocol (Norelli & Bronstein,
arXiv:2510.20075) on top of MLX.

The user-facing surface is the `calgacus` console script. For programmatic
use, see `calgacus.core` (encode/decode), `calgacus.keyfile` (load/save),
and `calgacus.model` (the MLX wrapper).
"""

from importlib.metadata import version

__version__ = version("calgacus-mlx")
