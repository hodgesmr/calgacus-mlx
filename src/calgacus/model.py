"""MLX model wrapper.

The only place in calgacus that depends on MLX-specific APIs. The rest of
the codebase talks to a single `MLXModel` class that exposes a small
surface: tokenize/detokenize, next-token logits, rank operations, and the
two protocol special-token IDs (EOS and BOUNDARY).
"""

from __future__ import annotations

from mlx_lm import load


# Per-model boundary-token lookup. The boundary marker is a tokenizer-reserved
# special token that never appears in normal text input (see the protocol's
# termination strategy). Different model families reserve different tokens,
# so we map by name and resolve names to IDs at load time.
BOUNDARY_TOKEN_NAMES_BY_FAMILY: dict[str, str] = {
    "llama": "<|reserved_special_token_0|>",
    "qwen": "<|extra_0|>",
    "gemma": "<unused0>",
}


def _detect_family(model_id: str) -> str:
    """Identify the model family from a Hugging Face repo ID.

    Used to look up the right boundary-token name. We are conservative:
    known families only, no fuzzy matching, since picking the wrong
    boundary token would silently break the protocol.
    """
    lo = model_id.lower()
    if "llama" in lo:
        return "llama"
    if "qwen" in lo:
        return "qwen"
    if "gemma" in lo:
        return "gemma"
    raise ValueError(
        f"Unknown model family for {model_id!r}. Calgacus needs a per-family "
        f"boundary-token mapping; see calgacus/model.py to add one."
    )


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
        self._boundary_token_id = self._resolve_boundary()

    def _resolve_eos(self) -> int:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise RuntimeError(
                f"Tokenizer for {self.model_id!r} does not declare an EOS "
                f"token. Calgacus uses EOS as the protocol stop sentinel."
            )
        return int(eos)

    def _resolve_boundary(self) -> int:
        family = _detect_family(self.model_id)
        name = BOUNDARY_TOKEN_NAMES_BY_FAMILY[family]
        ids = self.tokenizer.convert_tokens_to_ids(name)
        if ids is None:
            raise RuntimeError(
                f"Boundary token {name!r} not found in tokenizer for "
                f"{self.model_id!r}."
            )
        if isinstance(ids, list):
            if len(ids) != 1:
                raise RuntimeError(
                    f"Boundary token {name!r} resolves to {len(ids)} IDs, "
                    f"expected 1."
                )
            return int(ids[0])
        return int(ids)

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def boundary_token_id(self) -> int:
        return self._boundary_token_id

    def tokenize(self, text: str, *, with_bos: bool = False) -> list[int]:
        """Tokenize text to token IDs.

        By default does NOT prepend BOS or other special tokens. Pass
        `with_bos=True` only when starting a fresh forward pass; subsequent
        appends to a context should not re-add BOS.
        """
        return self.tokenizer.encode(text, add_special_tokens=with_bos)

    def detokenize(self, ids: list[int]) -> str:
        """Detokenize a list of token IDs back to text.

        Skips special tokens (EOS, BOUNDARY, BOS, etc.) so the returned
        text is clean prose suitable for the user. The protocol uses
        token IDs directly for sentinel detection, so stripping them on
        detokenize is always safe.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)
