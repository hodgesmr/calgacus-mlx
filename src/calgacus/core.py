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

from .model import MLXModel
from .termination import (
    Trailer,
    generate_natural_tail,
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
    then appending up to `tail_max_tokens` of natural-sampled tail.

    For each rank `r` in the rank sequence, picks the rank-r-th most
    probable next token under the cover-side LLM. After the rank-driven
    payload, switches to greedy sampling for up to `tail_max_tokens`
    more tokens (or until natural EOS), discarding the EOS so the
    visible stegotext stays clean prose.

    Returns the detokenized stegotext, without the cover-prompt prefix.
    """
    k_tokens = model.tokenize(cover_prompt)
    context = _initial_context(model, k_tokens)

    payload: list[int] = []
    for r in ranks:
        logits = model.next_token_logits(context)
        token_id = model.token_at_rank(logits, r)
        payload.append(token_id)
        context.append(token_id)

    tail = generate_natural_tail(model, context, tail_max_tokens)
    return model.detokenize(payload + tail)


def ranks_from_cover(
    model: MLXModel,
    cover_prompt: str,
    stego_text: str,
) -> list[int]:
    """Recover the rank sequence from `stego_text` given `cover_prompt`.

    Tokenizes `cover_prompt + stego_text` jointly (BPE-safe split) and
    slices off the `s_tokens` portion. For each token in `s_tokens`,
    records its rank in the cover-side LLM's next-token distribution
    given the running context.

    The decoder does not yet know which ranks are payload vs. tail;
    `secret_from_ranks` makes that distinction by stopping at EOS in
    the reconstructed secret stream.
    """
    k_tokens, s_tokens = split_after_prefix(model, cover_prompt, stego_text)

    context = _initial_context(model, k_tokens)
    ranks: list[int] = []
    for token_id in s_tokens:
        logits = model.next_token_logits(context)
        ranks.append(model.rank_of_token(logits, token_id))
        context.append(token_id)

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
