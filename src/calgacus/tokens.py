"""Tokenization boundary helpers.

BPE tokenizers (Llama, Qwen, Gemma) merge across whitespace and
punctuation, so in general:

    tokenize(a) + tokenize(b) != tokenize(a + b)

For Calgacus this matters because the encoder needs `e`'s token IDs as
they appear in the joint tokenization of `prefix + e`, while the decoder
tokenizes `prefix` alone and reconstructs token-by-token. If the prefix
re-tokenizes differently when something is appended to it, encoder and
decoder see different contexts and the rank sequence does not decode.

`split_after_prefix` is the safe split: tokenize the joint string in one
call to get the canonical tokenization, then verify the prefix portion
matches the standalone prefix tokenization before slicing.
"""

from __future__ import annotations

from .model import MLXModel


def split_after_prefix(
    model: MLXModel, prefix: str, full_text: str
) -> tuple[list[int], list[int]]:
    """Tokenize `prefix + full_text` jointly, then split at the prefix boundary.

    Returns:
        prefix_ids: tokenize(prefix) on its own.
        suffix_ids: the portion of tokenize(prefix + full_text) that
            follows the prefix's tokens.

    Raises ValueError if the BPE tokenizer merges across the boundary,
    i.e. if the joint tokenization does not begin with the standalone
    prefix tokenization. This happens when the prefix's last character
    and the suffix's first character form a higher-priority merge rule
    together. The fix is usually to add an explicit separator (a space,
    newline, or punctuation) to the end of the prefix.
    """
    prefix_ids = model.tokenize(prefix)
    full_ids = model.tokenize(prefix + full_text)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        # Show the end of each tokenization. BPE merges are local
        # so the divergence is almost always at the boundary itself.
        sample_len = min(8, len(prefix_ids), len(full_ids))
        prefix_tail = prefix_ids[-sample_len:]
        joint_tail = full_ids[len(prefix_ids) - sample_len : len(prefix_ids)]
        prefix_tail_text = model.detokenize(prefix_tail)
        joint_tail_text = model.detokenize(joint_tail)
        suggestion = prefix_tail_text + " "
        raise ValueError(
            f"BPE tokenizer merged across the prefix/suffix boundary. "
            f"tokenize(prefix) ends with {prefix_tail} ({prefix_tail_text!r}); "
            f"tokenize(prefix + full_text) at the same position has "
            f"{joint_tail} ({joint_tail_text!r}). "
            f"Try ending the prefix with explicit whitespace or punctuation "
            f"(for example, {suggestion!r}) to avoid the merge."
        )
    suffix_ids = full_ids[len(prefix_ids):]
    return prefix_ids, suffix_ids
