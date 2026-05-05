"""MLX model wrapper.

The only place in calgacus that depends on MLX-specific APIs. The rest of
the codebase talks to a single `MLXModel` class that exposes a small
surface: tokenize/detokenize, next-token logits, rank operations, and the
EOS token ID (the protocol's stop sentinel).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx_lm import load


class MLXModel:
    """Wrapper over mlx-lm's `load()` with the small surface calgacus needs.

    Loaded once per CLI invocation. The model and tokenizer are cached.
    Construction is expensive (downloads weights on first run); reuse the
    same instance across all encode/decode calls in a session.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self._eos_token_id = self._resolve_eos()

    def _resolve_eos(self) -> int:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise RuntimeError(
                f"Tokenizer for {self.model_id!r} does not declare an EOS "
                f"token. Calgacus uses EOS as the protocol stop sentinel."
            )
        return int(eos)

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def tokenize(self, text: str, *, with_bos: bool = False) -> list[int]:
        """Tokenize text to token IDs.

        By default does NOT prepend BOS or other special tokens. Pass
        `with_bos=True` only when starting a fresh forward pass; subsequent
        appends to a context should not re-add BOS.
        """
        return self.tokenizer.encode(text, add_special_tokens=with_bos)

    def detokenize(self, ids: list[int]) -> str:
        """Detokenize a list of token IDs back to text.

        Skips special tokens (EOS, BOS, etc.) so the returned text is
        clean prose suitable for the user. The protocol uses EOS's
        token ID directly for sentinel detection, so stripping it on
        detokenize is always safe.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def next_token_logits(self, ids: list[int]) -> mx.array:
        """Return shape (vocab_size,) logits for the next position after `ids`.

        Runs a full forward pass (no KV cache).

        Raises ValueError if `ids` is empty.
        """
        if not ids:
            raise ValueError(
                "Cannot compute next-token logits with empty input."
            )
        inputs = mx.array(ids)[None, :]
        logits = self.model(inputs)
        return logits[0, -1, :]

    def rank_of_token(self, logits: mx.array, token_id: int) -> int:
        """Zero-indexed rank of `token_id` in `logits` under stable descending
        order, with ties broken by smaller token ID first.

        Encoder-side rank query: for a target token at position i in the
        secret, this returns r_i. The encoder records r_i; the decoder
        regenerates the same rank by running this on the same logits.

        Stable tie-break: encoder and decoder must agree on
        the ordering when two tokens have identical logits, otherwise the
        protocol breaks. We enforce stability by counting:

            rank = (tokens with strictly higher logit)
                 + (tokens with equal logit and smaller ID)
        """
        np_logits = np.asarray(logits)
        if not 0 <= token_id < len(np_logits):
            raise ValueError(
                f"token_id {token_id} out of range [0, {len(np_logits)})."
            )
        target = np_logits[token_id]
        higher = int((np_logits > target).sum())
        tie_lower_id = int(
            ((np_logits == target) & (np.arange(len(np_logits)) < token_id)).sum()
        )
        return higher + tie_lower_id

    def token_at_rank(self, logits: mx.array, rank: int) -> int:
        """Token ID at zero-indexed `rank` in stable descending order, with
        ties broken by smaller token ID first.

        Decoder-side rank query: pick the rank-r-th token from the cover
        distribution given the rank recovered from the stegotext.

        Uses numpy's stable lexsort: primary key is `-logits` ascending
        (i.e. logits descending), tie-break is `token_id` ascending. Same
        stable ordering convention as `rank_of_token`.
        """
        np_logits = np.asarray(logits)
        if not 0 <= rank < len(np_logits):
            raise ValueError(
                f"rank {rank} out of range [0, {len(np_logits)})."
            )
        n = len(np_logits)
        sorted_ids = np.lexsort((np.arange(n), -np_logits))
        return int(sorted_ids[rank])
