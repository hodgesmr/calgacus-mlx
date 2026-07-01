# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

calgacus-mlx is an MLX implementation of the [Calgacus protocol](https://arxiv.org/abs/2510.20075) for LLM-based text steganography. The README is the canonical user-facing documentation; this file is for code-touching context that isn't there.

Apple Silicon only (MLX is Apple-specific). Default model is `mlx-community/SmolLM3-3B-Base-8bit`, downloaded to the HuggingFace cache on first use. It is a base (non-instruct) model with a BOS-less tokenizer; see invariant 6 for why the second matters.

## Running

```bash
uv sync                                                # install deps
uv run calgacus --help                                 # CLI overview
uv run calgacus init -o keyfile.toml                   # interactive keyfile creation
uv run calgacus encode -k keyfile.toml -s secret.txt -o stego.txt
uv run calgacus decode -k keyfile.toml -s stego.txt
```

There is **no test suite yet**. `pytest` is a dev dependency but no tests have been written. Smoke-test changes by running an encode/decode round-trip end-to-end against a fixed keyfile and confirming the recovered secret matches byte-for-byte.

## Module map

Everything lives in `src/calgacus/`:

- `core.py`: protocol primitives (`ranks_for_secret`, `cover_from_ranks`, `ranks_from_cover`, `secret_from_ranks`) plus the `encode` / `decode` wrappers that compose them. All cover-side rank selection routes through `_stable_token_at_rank` / `_stable_rank_of_token`, which apply the canonicality and position-0 leading-space filters.
- `model.py`: `MLXModel`, the only class that touches MLX-specific APIs. Wraps `mlx_lm.load()` and exposes the small surface every other module uses. Owns the lazy `always_stable_token_ids` (any leading-whitespace prefix) and `leading_space_token_ids` (regular-space prefix only) sets.
- `keyfile.py`: TOML keyfile dataclass plus `load_key` / `save_key`. Four fields: `model`, `cover_prompt`, `secret_prefix`, `trailer`.
- `termination.py`: `Trailer` enum (`graceful` / `eos-only`) and the trailer+EOS suffix construction. The trailer's purpose is to keep the model's EOS rank low at the secret-side stop position.
- `tokens.py`: `split_after_prefix`, the safe joint-tokenize-and-slice helper for the cover/stegotext boundary on the decoder side.
- `cli.py`: Click subcommands. Three commands: `init`, `encode`, `decode`. Most of the file is option resolution (CLI flag, then keyfile, then default).
- `__init__.py`: sets `HF_HUB_DISABLE_PROGRESS_BARS=1` and `TRANSFORMERS_VERBOSITY=error` before any HF/transformers import, then exposes `__version__`. Order of statements matters; do not move the `os.environ.setdefault` calls below the imports.

## Protocol invariants

Encoder and decoder must produce byte-identical logits at every position. Round-trip correctness is the design constraint, and several non-obvious choices exist only to maintain it:

1. **Cache-based forward only.** Every forward pass on both encoder and decoder side goes through `MLXModel.forward_step`, which uses the `mlx_lm` KV cache. Mixing a bulk forward pass on one side with a cache pass on the other gives slightly different bf16 logits and flips ranks on close-logit token pairs.
2. **Float32 at the wrapper boundary.** `_to_float32` in `model.py` casts logits to float32 before they leave the wrapper. Required for model architectures whose LM head emits bf16 (Qwen3, some Gemma variants); numpy cannot read bfloat16 through the buffer protocol.
3. **BPE canonicality filter.** `core._is_canonical` ensures cover-side picks do not disturb upstream BPE merges. Without it, the decoder's joint re-tokenization desynchronizes from the encoder's token stream and the round-trip silently fails.
4. **Position-0 leading-space filter.** The first cover token must be in `leading_space_token_ids` (regular space-prefix, *not* newline- or tab-prefix). Symmetric with the encoder's `lstrip(' ')` of the visible stegotext and the decoder's space-prepend before tokenizing.
5. **Asymmetric tokenization across the secret/cover halves.** `ranks_for_secret` tokenizes `k'` and the secret separately and prepends a leading space to the secret. The cover side uses joint tokenization via `split_after_prefix`. The asymmetry is intentional: the secret has no visible boundary to round-trip across, while the cover side does.
6. **BOS-aware initial context.** `core._initial_context` builds the first forward-pass context and must return the *same* tokens on encoder and decoder. Tokenizers with a BOS token (Llama, Gemma) get `[BOS] + prefix_ids`. BOS-less tokenizers (SmolLM3, Qwen) get `prefix_ids` as-is when non-empty, and `[EOS]` (the pretraining document boundary) when the prefix is empty — otherwise the forward pass would receive zero tokens and crash. The empty-prefix + BOS-less case is *document-initial*: `core._document_initial` gates it, and in that case `ranks_for_secret` skips the leading-space prepend from invariant 5 (a document-initial word has no leading space) and `secret_from_ranks` skips the matching strip. Both sides gate on the same predicate, so they stay in lockstep. This path is exercised by the default model (BOS-less) whenever `secret_prefix` is empty.

There is no protocol versioning. Any change that affects rank values, cover token selection order, the trailer/EOS scheme, or the position-0 filter breaks every stegotext encoded under the previous version. Treat such changes as breaking and call them out explicitly in commits.

## CLI subtleties

- All three subcommands accept text input three ways: positional argument, `--*-file` flag (with `-` for stdin), or stdin (default if neither is given). **The `-` stdin sentinel is honored only by `--*-file` flags, not by the positional argument.** `calgacus encode -` encodes the literal one-character string `"-"`.
- `init` is interactive by default; pass `--no-interactive` for scripted use.
- `encode` and `decode` both take `--quiet` to suppress `Loading model...` / `Encoding...` progress on stderr.
- `encode --tail-max-tokens N` bounds the natural-tail loop (default 32; 0 disables the tail entirely).
- Per-call flags on `encode` / `decode` (`--cover-prompt`, `--secret-prefix`, `--model`, `--trailer`) override the corresponding keyfile field for that invocation.
- Model load failures are wrapped in `MLXModel.__init__` and re-raised as `ValueError` with a friendlier message; the CLI converts these to `click.ClickException` so the user sees a clean error rather than an mlx-lm traceback.

## Cover prompt is an incipit, not an instruction

This is the single most important user-facing concept and worth internalizing before changing the README's cover-prompt guidance, the `init` interactive prompts, or any cover-prompt example in code or docs.

calgacus feeds the cover prompt to the LLM as raw text, with no chat-template formatting. The model treats it as the *opening of a passage* and continues from where it stops. Instruction-style prompts (`"Write a review of..."`) confuse instruction-tuned models in raw-text mode and produce off-topic, off-distribution covers. Incipit-style prompts (`"My review of X, after seeing it twice:"`) put the model in a clear continuation lane.

The README's "Cover prompt" subsection has the canonical guidance and the effective/ineffective examples. Keep any CLI examples and docstring examples consistent with that section.
