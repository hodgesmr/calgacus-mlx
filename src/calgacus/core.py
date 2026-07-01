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

Forward-pass strategy:

All four functions use the same cache-based incremental forward
pattern: prefill a fresh KV cache with the initial context built by
`_initial_context` (`[BOS] + prompt` for tokenizers with a BOS token,
a BOS-less variant otherwise), then extend by one token per step via
`MLXModel.forward_step`. Using one strategy on every call site
guarantees that encoder and decoder see byte-identical logits at the
same context, which the rank ordering is sensitive to (small numerical
differences can flip ranks of close-logit tokens and desynchronize the
round-trip).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .model import MLXModel
from .termination import (
    Trailer,
    secret_side_suffix,
    strip_trailer,
)
from .tokens import split_after_prefix


def _initial_context(model: MLXModel, prefix_ids: list[int]) -> list[int]:
    """Build a non-empty, model-natural initial context from `prefix_ids`.

    A fresh forward pass needs at least one token to produce logits, and
    both encoder and decoder must build the *same* starting context or
    their rank sequences diverge. There are two tokenizer families:

    - Tokenizers with a BOS token (Llama 3, Gemma): the model expects
      sequences to start with it. Empty prefix becomes just `[BOS]`;
      non-empty prefix becomes `[BOS] + prefix_ids`.

    - Tokenizers without a BOS token (SmolLM3, Qwen): the model was
      trained on sequences that begin directly with content, so a
      non-empty prefix is already a valid start and passes through
      unchanged. An empty prefix, however, would leave the forward pass
      with zero tokens. We seed it with EOS: in the GPT-2/Llama-3/Qwen
      pretraining convention the end-of-text token is the document
      separator, so the next-token distribution right after it is the
      "document-initial" distribution — the natural stand-in for a
      start-of-sequence marker, and guaranteed to exist (calgacus
      requires an EOS token).

    Every call site runs this, so encoder and decoder stay in lockstep.
    """
    bos_id = model.tokenizer.bos_token_id
    if bos_id is not None:
        return [int(bos_id)] + list(prefix_ids)
    if prefix_ids:
        return list(prefix_ids)
    return [model.eos_token_id]


def _document_initial(model: MLXModel, prefix_ids: list[int]) -> bool:
    """True when the secret sits at a genuine document start.

    Holds only for tokenizers without a BOS token AND an empty secret
    prefix: there is no preceding context at all, so `_initial_context`
    seeds the forward pass with EOS (the pretraining document boundary)
    and the secret's first token should be document-initial — a
    capitalized word with no leading space, the form that naturally
    follows an end-of-text boundary.

    In every other case the secret continues after real prose (a BOS
    marker, or a non-empty prefix), where a leading-space first token is
    the natural form. Encoder (`ranks_for_secret`) and decoder
    (`secret_from_ranks`) both consult this so their space handling
    stays symmetric.
    """
    return model.tokenizer.bos_token_id is None and not prefix_ids


_CANONICALITY_WINDOW = 4

# Heuristic stopper for the natural-tail loop. The tail extends the
# rank-driven cover with greedy generation until the model wants to
# stop or we hit a sentence boundary. _TAIL_TERMINATORS are the
# sentence-final punctuation we treat as "done"; _TAIL_TRAILING_PUNCT
# is closing punctuation/quotes that may follow a terminator and
# should be peeled off before the terminator check.
_TAIL_TERMINATORS = ".!?"
_TAIL_TRAILING_PUNCT = "\"'`)]}"
_TAIL_LOOKBACK = 3


def _ends_on_sentence_terminator(model: MLXModel, cover_tokens: list[int]) -> bool:
    """Return True if the cover so far ends on a sentence terminator.

    Detokenizes the last few cover tokens, strips trailing whitespace
    and any closing quotes/brackets that legitimately follow a
    terminator (`."` `?'` `.)`), and checks whether the resulting
    last character is a sentence-ending punctuation. Used by the
    natural-tail loop to land on a graceful ending instead of running
    to `tail_max_tokens`.
    """
    if not cover_tokens:
        return False
    recent = model.detokenize(cover_tokens[-_TAIL_LOOKBACK:]).rstrip()
    while recent and recent[-1] in _TAIL_TRAILING_PUNCT:
        recent = recent[:-1]
    return bool(recent) and recent[-1] in _TAIL_TERMINATORS


def _is_canonical(model: MLXModel, tokens: list[int]) -> bool:
    """Return True iff appending `tokens[-1]` to `tokens[:-1]` keeps the
    tokenization canonical.

    BPE tokenizers are deterministic but `detokenize -> tokenize` is not
    always the identity: BPE may produce different boundaries when given
    the raw text than when given the original tokens. We use this check
    to filter cover-side tokens to a subset where the round-trip holds.
    Without it, the decoder's re-tokenization of the visible stegotext
    would not match the encoder's tokens, and ranks would desynchronize.

    Fast path: if the last token is in the model's "always stable" set
    (its decoded form starts with whitespace), we skip the check
    entirely. BPE merges essentially never bridge whitespace boundaries
    in natural text, so a leading-whitespace token cannot disrupt the
    canonicality of the preceding tokens.

    Slow path: a window-based local check. BPE merges are local, so a
    new token at the end can only disrupt the tokenization within a
    small distance of the boundary. We detokenize and re-tokenize only
    the last `_CANONICALITY_WINDOW + 1` tokens and compare. Roughly an
    order of magnitude faster than re-tokenizing the full sequence,
    with negligible false-positive risk for typical natural text.
    """
    if not tokens:
        return True
    if tokens[-1] in model.always_stable_token_ids:
        return True
    window = tokens[-(_CANONICALITY_WINDOW + 1):]
    text = model.detokenize(window)
    return model.tokenize(text) == window


def _stable_token_at_rank(
    model: MLXModel,
    logits: mx.array,
    prefix_no_bos: list[int],
    rank: int,
    *,
    leading_space_only: bool = False,
) -> int:
    """Pick the rank-r-th canonical-stable token, given precomputed logits.

    Walks the logit-sorted vocabulary in descending order, counting
    only tokens that pass `_is_canonical(prefix + [t])`. Returns the
    rank-th stable token.

    If `leading_space_only` is True, candidates are further restricted
    to tokens whose decoded form starts with a regular space (the
    model's `leading_space_token_ids` set, which excludes newline-
    and tab-prefix tokens). Used by the cover-side encoder at
    position 0 to guarantee the visible stegotext starts with one
    space and a word; the encoder then lstrips that one space and
    the decoder prepends one before tokenizing, so the stegotext is
    portable across transports that mangle leading whitespace.
    """
    np_logits = np.asarray(logits)
    n = len(np_logits)
    sorted_ids = np.lexsort((np.arange(n), -np_logits))

    seen = 0
    for tid in sorted_ids:
        tid_int = int(tid)
        if leading_space_only and tid_int not in model.leading_space_token_ids:
            continue
        candidate = prefix_no_bos + [tid_int]
        if _is_canonical(model, candidate):
            if seen == rank:
                return tid_int
            seen += 1

    raise ValueError(
        f"Ran out of stable tokens before reaching rank {rank} "
        f"(vocab size {n})."
    )


def _stable_rank_of_token(
    model: MLXModel,
    logits: mx.array,
    prefix_no_bos: list[int],
    target: int,
    *,
    leading_space_only: bool = False,
) -> int:
    """Compute the rank of `target` in the canonical-stable distribution,
    given precomputed logits.

    Counts only stable tokens with strictly higher logit (or equal
    logit and smaller token ID, matching the lexsort tie-break).

    If `leading_space_only` is True, only tokens whose decoded form
    starts with whitespace are counted (matching the encoder's
    cover-position-0 restriction).
    """
    np_logits = np.asarray(logits)
    n = len(np_logits)
    sorted_ids = np.lexsort((np.arange(n), -np_logits))

    seen = 0
    for tid in sorted_ids:
        tid_int = int(tid)
        if tid_int == target:
            return seen
        if leading_space_only and tid_int not in model.leading_space_token_ids:
            continue
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

    Tokenizes `secret_prefix` and `secret_text` **separately** and
    concatenates them. We deliberately do not use joint tokenization
    on the secret side: BPE often re-merges across the boundary
    (e.g. `":"` plus `" "` plus `"The"` becomes `":"` plus `" The"`
    when joint-tokenized), which would force the user to find a magic
    separator. With separate tokenization, the model sees a slightly
    non-canonical sequence at the boundary, but encoder and decoder
    agree on it perfectly, so round-trip is exact.

    Appends the trailer-tokens and EOS. For each token in
    `e_tokens + trailer + [EOS]`, records its rank in the LLM's
    next-token distribution given the running context.

    Uses an incremental KV cache: prefill once with `[BOS] + prefix_ids`
    and then extend by one token per step. The protocol uses the same
    cache-based pattern in `secret_from_ranks`, so encoder and decoder
    see byte-identical logits at each position and rank assignments
    line up exactly.

    Raises ValueError if `e_tokens` contains the EOS token ID, which
    would create an early-stop signal in the rank sequence.

    Returns:
        ranks: one int per (e + trailer + EOS) token.
        secret_ids: the actual token IDs that produced these ranks.
    """
    prefix_ids = model.tokenize(secret_prefix)
    # Prepend a space to `secret_text` before tokenizing so the first
    # secret token is the leading-space-aware variant (e.g. " hello"
    # rather than "hello"). Without this, secrets that start with a
    # non-whitespace character land at unusually high rank at position
    # 0 (the model expects a leading-space token after natural prose),
    # which the cover-side encoder then has to reach into the deep
    # tail of the cover distribution to encode, producing visibly
    # off-distribution tokens (foreign scripts, rare unicode) at the
    # start of the stegotext. The decoder strips this prepended space
    # in `secret_from_ranks`.
    #
    # Exception: a document-initial secret (no BOS token, empty prefix)
    # follows an end-of-text boundary rather than prose, where a
    # leading-space token is instead the unnatural, deep-tail form. There
    # we tokenize the secret as-is so its first token is a normal
    # document-initial word. `secret_from_ranks` mirrors this.
    lead = "" if _document_initial(model, prefix_ids) else " "
    e_tokens = model.tokenize(lead + secret_text)

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

    cache = model.make_cache()
    next_logits = model.forward_step(_initial_context(model, prefix_ids), cache)

    ranks: list[int] = []
    for token_id in rank_input:
        ranks.append(model.rank_of_token(next_logits, token_id))
        next_logits = model.forward_step([token_id], cache)

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

    Uses an incremental KV cache: prefill once with the cover prompt,
    then extend by one token per step.

    Returns the detokenized stegotext, without the cover-prompt prefix.
    """
    k_tokens = model.tokenize(cover_prompt)
    cover_tokens: list[int] = []

    cache = model.make_cache()
    next_logits = model.forward_step(_initial_context(model, k_tokens), cache)

    for i, r in enumerate(ranks):
        # Restrict the very first cover token to a leading-space-aware
        # token. The model's natural distribution at this position
        # already heavily favors leading-space tokens, so this rarely
        # changes behavior in practice; it just guarantees that the
        # visible stegotext starts with one space, which the encoder
        # then strips and the decoder prepends. That symmetric pair
        # makes the stegotext portable across transports that mangle
        # leading whitespace.
        tid = _stable_token_at_rank(
            model, next_logits, k_tokens + cover_tokens, r,
            leading_space_only=(i == 0),
        )
        cover_tokens.append(tid)
        next_logits = model.forward_step([tid], cache)

    eos = model.eos_token_id
    for _ in range(tail_max_tokens):
        # Stop if the cover already ends on a sentence terminator
        # (with optional trailing whitespace and closing
        # quotes/brackets). The rank-driven payload may have landed
        # cleanly, in which case we don't extend it. After the tail
        # appends a terminating token below, the next iteration's
        # check fires here and we stop.
        if _ends_on_sentence_terminator(model, cover_tokens):
            break
        # Stop when the model's actual greedy top is EOS.
        # `_stable_token_at_rank` filters EOS via canonicality
        # (detokenize strips special tokens, so the round-trip check
        # always fails for an EOS-ending candidate), so we check the
        # unfiltered top-1 directly. EOS as a cover token would also
        # break round-trip, so we never *append* EOS; we use it only
        # as a stop signal.
        if model.token_at_rank(next_logits, 0) == eos:
            break
        tid = _stable_token_at_rank(model, next_logits, k_tokens + cover_tokens, 0)
        cover_tokens.append(tid)
        next_logits = model.forward_step([tid], cache)

    text = model.detokenize(cover_tokens)
    # Strip a single leading space if present. The encoder's first
    # cover token is restricted to be leading-space-aware, so this
    # almost always strips exactly one space. The decoder prepends one
    # space before tokenizing, recovering the canonical form.
    if text.startswith(" "):
        text = text[1:]
    return text


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

    Uses an incremental KV cache, matching the encoder's
    `cover_from_ranks` pattern. This ensures encoder and decoder see
    byte-identical logits at each position and rank assignments line
    up exactly.

    The decoder does not yet know which ranks are payload vs. tail;
    `secret_from_ranks` makes that distinction by stopping at EOS in
    the reconstructed secret stream.
    """
    # Normalize leading whitespace: the encoder strips a single leading
    # space from its output for portability across whitespace-mangling
    # transports. We lstrip any leading whitespace the user's input
    # might have (extra spaces, accidentally added) and then prepend a
    # canonical single space, which matches the encoder's pre-strip
    # form and gives the joint tokenization the encoder produced.
    normalized_stego = " " + stego_text.lstrip()
    k_tokens, s_tokens = split_after_prefix(model, cover_prompt, normalized_stego)

    if not s_tokens:
        return []

    cache = model.make_cache()
    next_logits = model.forward_step(_initial_context(model, k_tokens), cache)

    ranks: list[int] = []
    cover_so_far: list[int] = []
    for i, tid in enumerate(s_tokens):
        rank = _stable_rank_of_token(
            model, next_logits, k_tokens + cover_so_far, tid,
            leading_space_only=(i == 0),
        )
        ranks.append(rank)
        cover_so_far.append(tid)
        next_logits = model.forward_step([tid], cache)

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

    Uses an incremental KV cache: prefill once with the secret prefix,
    then extend by one token per step.

    Raises ValueError if the rank sequence is exhausted without
    reconstructing EOS (truncated stegotext or wrong key/model), or if
    the recovered pre-EOS sequence is shorter than the expected
    trailer length (mismatched trailer mode between sender and
    receiver).
    """
    prefix_ids = model.tokenize(secret_prefix)
    eos = model.eos_token_id

    cache = model.make_cache()
    next_logits = model.forward_step(_initial_context(model, prefix_ids), cache)

    recovered: list[int] = []
    for r in ranks:
        token_id = model.token_at_rank(next_logits, r)
        if token_id == eos:
            break
        recovered.append(token_id)
        next_logits = model.forward_step([token_id], cache)
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
    text = model.detokenize(e_tokens)
    # Strip the single leading space that `ranks_for_secret` prepended
    # to `secret_text` before tokenizing. If the user's original secret
    # started with whitespace, only the encoder's added space is
    # removed; subsequent leading whitespace from the user's input is
    # preserved. A document-initial secret had no space prepended (see
    # `ranks_for_secret`), so we skip the strip to keep the round-trip
    # exact for secrets that legitimately begin with a space.
    if not _document_initial(model, prefix_ids) and text.startswith(" "):
        text = text[1:]
    return text


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
