"""Core protocol

The Calgacus protocol decomposes cleanly:

    encode = ranks_for_secret + cover_from_ranks
    decode = ranks_from_cover + secret_from_ranks

The encoder builds a secret-side rank sequence by walking
`secret_prefix + e + trailer + EOS` token by token under the LLM and
recording each token's rank. It then walks that rank sequence under
`cover_prompt` to generate the stegotext.

The decoder runs the inverse: tokenize `cover_prompt + stegotext`,
recover ranks for each stegotext token, then walk those ranks under
`secret_prefix` to reconstruct the secret.
"""

from __future__ import annotations

import numpy as np

from .model import MLXModel
from .termination import (
    Trailer,
    secret_side_suffix,
    strip_trailer,
)
from .tokens import split_after_prefix


def _initial_context(model: MLXModel, prefix_ids: list[int]) -> list[int]:
    """Prepend BOS to `prefix_ids` for a model-natural initial context.

    The model expects sequences to start with BOS. Empty prefix
    becomes just `[BOS]`, which is the safe minimum context to
    compute next-token logits. Non-empty prefix becomes
    `[BOS] + prefix_ids`. Both encoder and decoder do this, so they
    see the same context and produce the same rank sequence.

    If the tokenizer does not declare a BOS token, the prefix is
    returned as-is and the caller must ensure it is non-empty.
    """
    bos_id = model.tokenizer.bos_token_id
    if bos_id is None:
        return list(prefix_ids)
    return [int(bos_id)] + list(prefix_ids)


def _is_canonical(model: MLXModel, tokens: list[int]) -> bool:
    """Return True iff `tokens` is the canonical tokenization of its
    detokenized form.

    BPE tokenizers are deterministic but `detokenize -> tokenize` is not
    always the identity: BPE may produce different boundaries when given
    the raw text than when given the original tokens. We use this check
    to filter cover-side tokens to a subset where the round-trip holds.
    Without it, the decoder's re-tokenization of the visible stegotext
    would not match the encoder's tokens, and ranks would desynchronize.
    """
    text = model.detokenize(tokens)
    return model.tokenize(text) == tokens


def _stable_token_at_rank(
    model: MLXModel,
    prefix_no_bos: list[int],
    rank: int,
) -> int:
    """Pick the rank-r-th token from the canonical-stable continuation
    distribution at the end of `prefix_no_bos`.

    Walks the logit-sorted vocabulary in descending order, counting
    only tokens that pass `_is_canonical(prefix + [t])`. Returns the
    rank-th stable token.
    """
    inference_ctx = _initial_context(model, prefix_no_bos)
    logits = model.next_token_logits(inference_ctx)

    np_logits = np.asarray(logits)
    n = len(np_logits)
    sorted_ids = np.lexsort((np.arange(n), -np_logits))

    seen = 0
    for tid in sorted_ids:
        candidate = prefix_no_bos + [int(tid)]
        if _is_canonical(model, candidate):
            if seen == rank:
                return int(tid)
            seen += 1

    raise ValueError(
        f"Ran out of stable tokens before reaching rank {rank} "
        f"(vocab size {n})."
    )


def _stable_rank_of_token(
    model: MLXModel,
    prefix_no_bos: list[int],
    target: int,
) -> int:
    """Compute the rank of `target` in the canonical-stable continuation
    distribution.

    Counts only stable tokens with strictly higher logit (or equal
    logit and smaller token ID, matching the lexsort tie-break).
    """
    inference_ctx = _initial_context(model, prefix_no_bos)
    logits = model.next_token_logits(inference_ctx)

    np_logits = np.asarray(logits)
    n = len(np_logits)
    sorted_ids = np.lexsort((np.arange(n), -np_logits))

    seen = 0
    for tid in sorted_ids:
        tid_int = int(tid)
        if tid_int == target:
            return seen
        candidate = prefix_no_bos + [tid_int]
        if _is_canonical(model, candidate):
            seen += 1

    raise ValueError(f"Target token {target} not found in vocabulary.")


def ranks_for_secret(
    model: MLXModel,
    secret_prefix: str,
    secret_text: str,
    trailer: Trailer = Trailer.GRACEFUL,
) -> tuple[list[int], list[int]]:
    """Compute the rank sequence for `secret_text + trailer + [EOS]` under
    `secret_prefix`.

    Tokenizes `secret_prefix + secret_text` jointly (so BPE merges
    across the boundary are handled cleanly) and slices off the
    `e_tokens` portion. Appends the trailer-tokens and EOS. For each
    token in `e_tokens + trailer + [EOS]`, records its rank in the
    LLM's next-token distribution given the running context.

    Raises ValueError if `e_tokens` contains the EOS token ID, which
    would create an early-stop signal in the rank sequence.

    Returns:
        ranks: one int per (e + trailer + EOS) token.
        secret_ids: the actual token IDs that produced these ranks.
    """
    prefix_ids, e_tokens = split_after_prefix(model, secret_prefix, secret_text)

    # Encode-time check: a literal EOS token inside `e_tokens` would
    # cause the decoder to stop reconstruction early and recover only
    # part of the secret. Normal text input cannot tokenize to EOS, but
    # we check defensively in case of pathological input.
    eos = model.eos_token_id
    if eos in e_tokens:
        raise ValueError(
            f"Secret text tokenizes to include the EOS token (ID "
            f"{eos}). This would create an early-stop signal in the "
            f"rank sequence. Adjust the secret to not contain literal "
            f"end-of-text markers."
        )

    suffix = secret_side_suffix(model, trailer)
    rank_input = e_tokens + suffix

    context = _initial_context(model, prefix_ids)
    ranks: list[int] = []
    for token_id in rank_input:
        logits = model.next_token_logits(context)
        ranks.append(model.rank_of_token(logits, token_id))
        context.append(token_id)

    return ranks, rank_input


def cover_from_ranks(
    model: MLXModel,
    cover_prompt: str,
    ranks: list[int],
    tail_max_tokens: int = 32,
) -> str:
    """Generate the stegotext by walking `ranks` under `cover_prompt`,
    then appending up to `tail_max_tokens` of stable greedy tail.

    Both the rank-driven payload and the natural tail are picked from
    the canonical-stable continuation distribution at each step. This
    keeps the visible stegotext round-trippable through the decoder's
    re-tokenization. The natural tail stops on EOS, which is dropped
    so the visible stegotext stays clean prose.

    Returns the detokenized stegotext, without the cover-prompt prefix.
    """
    k_tokens = model.tokenize(cover_prompt)
    cover_tokens: list[int] = []

    for r in ranks:
        tid = _stable_token_at_rank(model, k_tokens + cover_tokens, r)
        cover_tokens.append(tid)

    eos = model.eos_token_id
    for _ in range(tail_max_tokens):
        tid = _stable_token_at_rank(model, k_tokens + cover_tokens, 0)
        if tid == eos:
            break
        cover_tokens.append(tid)

    return model.detokenize(cover_tokens)


def ranks_from_cover(
    model: MLXModel,
    cover_prompt: str,
    stego_text: str,
) -> list[int]:
    """Recover the rank sequence from `stego_text` given `cover_prompt`.

    Tokenizes `cover_prompt + stego_text` jointly via split_after_prefix
    to handle BPE merges across the boundary, then for each stegotext
    token records its rank in the canonical-stable continuation
    distribution.

    The decoder does not yet know which ranks are payload vs. tail;
    `secret_from_ranks` makes that distinction by stopping at EOS in
    the reconstructed secret stream.
    """
    k_tokens, s_tokens = split_after_prefix(model, cover_prompt, stego_text)

    ranks: list[int] = []
    cover_so_far: list[int] = []
    for tid in s_tokens:
        rank = _stable_rank_of_token(model, k_tokens + cover_so_far, tid)
        ranks.append(rank)
        cover_so_far.append(tid)

    return ranks


def secret_from_ranks(
    model: MLXModel,
    secret_prefix: str,
    ranks: list[int],
    trailer: Trailer = Trailer.GRACEFUL,
) -> str:
    """Reconstruct the secret from `ranks` under `secret_prefix`.

    Walks the rank sequence under `k'`, picking the rank-r-th token at
    each step, and stops the moment the picked token is EOS. The
    reconstructed sequence is then `e_tokens + trailer-tokens`. Strips
    the trailer-tokens by length, detokenizes, returns.

    Raises ValueError if the rank sequence is exhausted without
    reconstructing EOS (truncated stegotext or wrong key/model), or if
    the recovered pre-EOS sequence is shorter than the expected
    trailer length (mismatched trailer mode between sender and
    receiver).
    """
    prefix_ids = model.tokenize(secret_prefix)
    context = _initial_context(model, prefix_ids)
    eos = model.eos_token_id

    recovered: list[int] = []
    for r in ranks:
        logits = model.next_token_logits(context)
        token_id = model.token_at_rank(logits, r)
        if token_id == eos:
            break
        recovered.append(token_id)
        context.append(token_id)
    else:
        # The for/else here fires only if the loop completed without
        # `break`, which means we walked every rank and never saw EOS.
        # That signals truncation or a mismatched key.
        raise ValueError(
            f"Rank sequence exhausted ({len(ranks)} ranks) without "
            f"reconstructing EOS. The stegotext may be truncated, or "
            f"sender and receiver may be using different models or "
            f"different cover prompts."
        )

    e_tokens = strip_trailer(model, recovered, trailer)
    return model.detokenize(e_tokens)


def encode(
    model: MLXModel,
    cover_prompt: str,
    secret_text: str,
    secret_prefix: str = "",
    trailer: Trailer = Trailer.GRACEFUL,
    tail_max_tokens: int = 32,
) -> str:
    """Full encode pipeline: secret_text to stegotext.

    Convenience wrapper around `ranks_for_secret` followed by
    `cover_from_ranks`. The CLI's `encode` subcommand calls this.
    """
    ranks, _ = ranks_for_secret(model, secret_prefix, secret_text, trailer)
    return cover_from_ranks(model, cover_prompt, ranks, tail_max_tokens)


def decode(
    model: MLXModel,
    cover_prompt: str,
    stego_text: str,
    secret_prefix: str = "",
    trailer: Trailer = Trailer.GRACEFUL,
) -> str:
    """Full decode pipeline: stegotext to secret_text.

    Convenience wrapper around `ranks_from_cover` followed by
    `secret_from_ranks`. The CLI's `decode` subcommand calls this.
    """
    ranks = ranks_from_cover(model, cover_prompt, stego_text)
    return secret_from_ranks(model, secret_prefix, ranks, trailer)
