"""Termination strategy: fixed trailer + EOS sentinel + natural tail.

The encoder composes the secret-side rank sequence as:

    e_tokens + trailer-tokens + [EOS]

where trailer-tokens are the tokenized form of `.\\n\\n` for GRACEFUL
mode, or empty for EOS_ONLY mode. The trailer is appended verbatim
regardless of what `e` ends with, so the decoder strips a fixed
number of tokens. Both sender and receiver must agree on the trailer
mode; it lives in the keyfile.

The trailer's job is to make EOS sit at a low rank. After `.\\n\\n`
the model rates EOS as a likely continuation, so the cover-side
token at the EOS position is unobtrusive. For typical secrets that
do not end with `.\\n\\n` patterns, the trailer tokens themselves
also sit at low ranks under their context, so the cover-side picks
at trailer positions look natural.

For unusual inputs (`e` already ending with `.`, `.\\n`, or `.\\n\\n`),
the secret-side context shows a doubled trailer, which slightly
elevates the trailer-token ranks. The cover-side picks at those
positions are moderately uncommon (rank in the hundreds to low
thousands) but stay in the natural-language register, not the
deep-tail glyph register.

After the rank-driven payload, the encoder appends a short
natural-sampled tail to the cover stegotext for a graceful ending.
The decoder ignores anything past EOS, so the tail length is a
sender-only knob.
"""

from __future__ import annotations

from enum import Enum

from .model import MLXModel


_TRAILER_TEXT = ".\n\n"


class Trailer(Enum):
    """Encoder-side trailer mode. Sender and receiver must agree on this;
    it lives in the keyfile.

    GRACEFUL (default): append tokenize(`.\\n\\n`) before EOS. Best EOS
    rank, most natural cover ending.

    EOS_ONLY: append nothing before EOS. EOS's rank depends on whatever
    `e` happens to end with. Slightly less overhead, slightly less
    reliable cover quality at the EOS position.
    """

    GRACEFUL = "graceful"
    EOS_ONLY = "eos-only"


def trailer_token_ids(model: MLXModel, trailer: Trailer) -> list[int]:
    """Return the trailer token IDs (NOT including EOS).

    Example (Llama 3.2 IDs):
        GRACEFUL  -> [13, 271]   # "." then "\\n\\n"
        EOS_ONLY  -> []
    """
    if trailer is Trailer.EOS_ONLY:
        return []
    return model.tokenize(_TRAILER_TEXT)


def secret_side_suffix(model: MLXModel, trailer: Trailer) -> list[int]:
    """Return the secret-side suffix appended after `e_tokens` during encode:

        trailer-tokens + [EOS]

    The caller composes the full secret-side rank sequence as
    `e_tokens + secret_side_suffix(...)`.

    Example (Llama 3.2 IDs, trailer=GRACEFUL):
        Returns: [13, 271, 128001]   # "." + "\\n\\n" + EOS
    """
    return trailer_token_ids(model, trailer) + [model.eos_token_id]


def strip_trailer(
    model: MLXModel, recovered_ids: list[int], trailer: Trailer
) -> list[int]:
    """Strip trailer tokens from the end of `recovered_ids`.

    `recovered_ids` is the secret-reconstruction output up to but not
    including EOS, i.e. `e_tokens + trailer-tokens`. Returns just
    `e_tokens` by removing the last `len(trailer_token_ids(...))`
    entries. No-op for Trailer.EOS_ONLY.

    Example (Llama 3.2 IDs, trailer=GRACEFUL):
        recovered_ids = [9906, 13, 271]   # "Hello" + "." + "\\n\\n"
        Returns:        [9906]            # "Hello"

    Raises ValueError if the recovered sequence is shorter than the
    expected trailer length, which usually means the stegotext was
    truncated or sender and receiver disagree on the trailer mode.
    """
    trailer_len = len(trailer_token_ids(model, trailer))
    if trailer_len == 0:
        return list(recovered_ids)
    if trailer_len > len(recovered_ids):
        raise ValueError(
            f"Cannot strip trailer of {trailer_len} tokens from "
            f"reconstructed sequence of {len(recovered_ids)} tokens. "
            f"The stegotext may be truncated, or sender and receiver "
            f"may disagree on the trailer mode."
        )
    return recovered_ids[:-trailer_len]
