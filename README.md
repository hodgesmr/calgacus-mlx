# calgacus-mlx

LLM Steganography: Hide a meaningful text inside another plausible text of comparable length.

MLX implementation of the [Calgacus protocol](https://arxiv.org/abs/2510.20075) (Norelli & Bronstein, 2025) for LLM-based text steganography. Given a secret message, calgacus produces a stegotext that reads like ordinary prose, but encodes the secret losslessly via rank-driven selection from a language model's next-token distribution at each position. A receiver with the same model and the same key recovers the original message exactly.

This is a Python CLI that runs locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Demo

Generate a `keyfile.toml` that bundles the model, cover prompt, and protocol settings that sender and receiver share. See [Keyfile format](#keyfile format) for more details.

```bash
uv run calgacus init -o keyfile.toml
```

```toml
model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
cover_prompt = "The spaghetti was delicious and the breadsticks were a fantastic side dish, but the salad was stale."
secret_prefix = "The following is an important covert message. Please read it carefully."
trailer = "graceful"
```

With the following `secret.txt`:

> The plan has changed. We must meet Tuesday night. Come to my office at midnight and bring the money.


encode to `stegotext.txt`:

```bash
uv run calgacus encode \
    -k keyfile.toml \
    -s secret.txt \
    -o stegotext.txt
cat stegotext.txt
```

The `stegotext.txt` it produces: 

> The leftover sauce from the previous evening still somehow magically made the spaghetti from that night taste fresh, like its own feeder.Then, I had a conversation with a friend who had recently moved to a new city.

Anyone with the same `keyfile.toml` can recover it:

```bash
uv run calgacus decode \
    -k keyfile.toml \
    -s stegotext.txt
```

> The plan has changed. We must meet Tuesday night. Come to my office at midnight and bring the money.

The recovered text matches the original byte-for-byte. The stegotext, meanwhile, reads as a casual aside about a recent meal and an old friend, and gives no surface indication that it carries a hidden payload. There may be cover-quality artifacts in the stegotext at positions where the secret-side ranks are high. See [Tuning the keyfile](#tuning-the-keyfile) for how to push these down.

This enables other sneaky things. We can hide a shell command, `curl -sIL http://example.com`:

```bash
echo -n "curl -sIL http://example.com" \
  | uv run calgacus encode \
      -k keyfile.toml \
      --quiet
```

Gives the stegotext: 
> Najeh had always dreamed of being a video۱۳ game designer, and he had spent countless hours studying the craft and practicing his skills.

Decode and run in one shot:

```bash
$(
  echo "Najeh had always dreamed of being a video۱۳ game designer, and he had spent countless hours studying the craft and practicing his skills." \
    | uv run calgacus decode \
        -k keyfile.toml \
        -s - \
        --quiet
)
```

Our sentence about a video came career excuted our `curl` command:

```
HTTP/1.1 200 OK
Date: Fri, 08 May 2026 03:41:36 GMT
Content-Type: text/html
Connection: keep-alive
Server: cloudflare
Last-Modified: Wed, 06 May 2026 14:17:14 GMT
Allow: GET, HEAD
Accept-Ranges: bytes
Age: 3997
cf-cache-status: HIT
```

## How it works

A language model, given a context, defines a probability distribution over the next token; the same context gives the same distribution. Calgacus uses that distribution as a shared random oracle: the encoder picks a cover token by its rank in the cover-side distribution, where the rank value is the rank of the next secret token in a different (secret-side) distribution. The decoder runs the same two distributions and reverses the mapping.

Concretely:

- **Secret side**. The encoder tokenizes `k' + secret + trailer + EOS` (where `k'` is an optional secret prefix) and walks it position by position. At position `i`, it queries the model for the distribution under the prefix and records `r_i`, the rank of the actual secret token in that distribution.
- **Cover side**. The encoder generates one cover token per rank `r_i`, picking the rank-`r_i`-th most likely token under the cover prompt `k` (and the cover-so-far). These cover tokens, detokenized, are the stegotext.
- **Decode**. The decoder tokenizes `k + stegotext`, walks the cover tokens, looks up each one's rank under the cover-side distribution to recover `r_i`, then replays those ranks against the secret-side distribution. Stop when EOS is hit.

Two small mechanisms support those three steps. The **trailer** (default `.\n\n`) is a fixed sequence appended to the secret stream before EOS so the model rates EOS at low rank under the secret-side context, which keeps the cover-side pick at the EOS position from being a deep-tail glyph. The **natural tail** is a few greedy cover tokens appended after the rank-driven payload so the visible stegotext lands on a sentence boundary; the decoder discards anything past the recovered EOS, so the tail is purely cosmetic.

The stegotext's token count is roughly the secret's, plus a few tokens for the trailer and a short optional natural tail; the total is typically 1x to 1.5x of the secret's token count in practice. Both sides need the same model, the same cover prompt `k`, the same secret prefix `k'`, and the same trailer. The keyfile bundles all of these.

There are three engineering wrinkles the paper alludes to but doesn't fully address. They are easy to get wrong, and getting them wrong silently breaks round-trip:

1. **BPE tokenization stability.** Byte-pair-encoding tokenizers can re-merge tokens depending on their neighbors. If the encoder picks token A then token B, but `detokenize([A, B])` re-tokenizes back to a single token AB, the decoder sees a different token sequence than the encoder wrote. Calgacus filters cover-side picks to "canonical-stable" tokens whose addition to the prefix does not disturb earlier merges. We also fast-path tokens that begin with whitespace, which the BPE essentially never re-merges across.
2. **Numerical determinism.** With bf16 weights, the order of additions in a forward pass affects the last bit or two of the logits, which is enough to flip the rank of two close-together tokens. Bulk-pass and KV-cache forward passes don't always produce identical logits. Calgacus uses the KV cache uniformly on both sides so the rank queries get the same logit values during encode and decode. Calgacus also casts the final logits to float32 at the wrapper boundary, so downstream rank arithmetic in numpy works regardless of whether the model emits float32 or bfloat16 (some Qwen3 and Gemma variants emit bf16, which numpy can't read directly through the buffer protocol).
3. **Whitespace portability.** Stegotext travels through chat clients and email forwarders that mangle leading whitespace. Calgacus restricts the first cover token to a regular leading-space token (specifically space-prefixed, *not* newline- or tab-prefixed, so the encoder's `lstrip(' ')` stays symmetric with the decoder's space-prepend), lstrips one space before emitting the visible stegotext, and prepends one space on the decoder side before tokenizing. The visible text is whitespace-stable; the protocol stays in sync.

## Install

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Apple Silicon. MLX is Apple-specific, so this implementation is too. The protocol itself is portable; a CUDA or MPS port would be straightforward.

```bash
git clone https://github.com/hodgesmr/calgacus-mlx
cd calgacus-mlx
uv sync
uv run calgacus --help
```

The first encode or decode call downloads the default model (`mlx-community/Llama-3.2-3B-Instruct-4bit`, ~2GB) into your HuggingFace cache. Subsequent runs load from disk in a few seconds.

## Usage

### `calgacus init`

Scaffold a keyfile interactively or from flags. The keyfile is plain TOML and is what sender and receiver must share.

```bash
uv run calgacus init   # interactive
uv run calgacus init \
    -o keyfile.toml \
    --no-interactive \
    --cover-prompt "I have been wrestling with the grass in my front lawn all summer, and I finally" \
    --secret-prefix "The following is a fragment of a poem:" \
    --trailer graceful
```

Both parties need an identical copy of the keyfile. Treat it like a shared secret: anyone with the keyfile can decode anything you encode under it.

### `calgacus encode`

```bash
echo "hello world" | uv run calgacus encode -k keyfile.toml
uv run calgacus encode "hello world" -k keyfile.toml
uv run calgacus encode -k keyfile.toml -s secret.txt -o stego.txt
```

Reads the secret from a positional argument, `--secret-file` (with `-` for stdin), or stdin (default). Writes stegotext to stdout or `--output`.

Use `--tail-max-tokens N` to bound the natural tail; default is 32. Set to 0 to skip the tail entirely and end the stegotext at the rank-driven payload's last cover token (which may not land on a sentence boundary).

### `calgacus decode`

```bash
uv run calgacus decode -k keyfile.toml -s stego.txt
cat stego.txt | uv run calgacus decode -k keyfile.toml
```

Reads stegotext, recovers the secret. The decoder ignores any tail text past EOS, so a stegotext can carry a short natural-sounding ending without affecting recovery.

### Keyfile format

A keyfile is plain TOML with four fields:

```toml
model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
cover_prompt = "I have been wrestling with the grass in my front lawn all summer, and I finally"
secret_prefix = "The following is a fragment of a poem:"
trailer = "graceful"
```

| field | description |
| --- | --- |
| `model` | HuggingFace model ID. Both ends must use the same one (and the same quantization). |
| `cover_prompt` | The opening text of the cover passage. Treat it as the first sentence(s) the model continues into a stegotext, not an instruction. See [Cover prompt](#cover-prompt) for why. |
| `secret_prefix` | Optional incipit prepended to the secret. A genre-matched prefix lowers the secret-side ranks and improves cover quality, especially for non-prose secrets like code, chess PGN, or structured data. |
| `trailer` | `graceful` (default) appends `.\n\n` before EOS so the protocol's stop sentinel sits at a low rank in the cover distribution. `eos-only` skips the trailer; cover quality at the EOS position is slightly worse. |

Flags on `encode` and `decode` (`--cover-prompt`, `--secret-prefix`, `--model`, `--trailer`) override individual keyfile fields, so the same keyfile can be used as a base with per-call adjustments.

## Tuning the keyfile

The cover prompt, secret prefix, and model choice all directly affect stegotext quality. A weak prompt produces a stegotext that reads like nonsense even though round-trip is exact. A small model produces broad distributions and more deep-tail artifacts. Spend time on these.

### Cover prompt

The cover prompt is the most consequential setting, and the thing to internalize about it is that it is an **incipit, not an instruction**. Calgacus does not give the LLM a question and read its answer; it gives the LLM the *opening of a passage* and the LLM continues that passage. Write the first sentence (or two) of the kind of text you want as your stegotext, and stop where you want the model to take over.

| effective | ineffective |
| --- | --- |
| `My take on Kurt Gödel's incompleteness theorems, after rereading the 1931 paper:` | `Write a thoughtful overview of Gödel's incompleteness theorems.` |
| `Bella Vita is a small Italian restaurant in our neighborhood, and last Friday I` | `Compose a glowing review of an Italian restaurant called Bella Vita.` |
| `The spaghetti was delicious and the breadsticks were a fantastic side dish, but the salad was stale.` | `The spaghetti was good. The breadsticks were nice. The salad was stale.` |

Three rules of thumb for incipits:

1. **Set the genre up front.** Mention the kind of text it is (review, essay, memo, recipe). The model's continuation distribution is heavily shaped by what it thinks the document is.
2. **Pin down topic, register, and audience.** Specific incipits keep the natural distribution narrow, which means even moderate-rank cover picks stay readable. Vague incipits collapse into the deep tail and produce off-topic glyphs.
3. **Continue the thought.** The model picks up where you stop. Mid-sentence endings (colon, comma, or partway through) give the sharpest continuation lane, but a single complete thought ending in a period also works well; the model just keeps going in the same register. What hurts is multiple short period-terminated sentences in a row; that primes the model for a paragraph break, which often shows up in the cover as a stray newline mid-stegotext.

Why instructions don't work: instruction-tuned LLMs (like Llama 3.2 Instruct) expect chat-formatted input (`<|start_header_id|>user<|end_header_id|>...`) when given a question. Calgacus feeds the prompt as raw text, so the model treats it as a passage to continue. Instructions in raw-text mode confuse the model's next-token distribution; incipits do not.

### Secret prefix

The secret prefix `k'` conditions the secret-side distribution, which determines the rank values that get encoded into the cover. A well-matched `k'` makes the secret tokens more predictable, which means lower secret-side ranks, which means cover-side picks land at lower ranks in the cover distribution and read more naturally.

For prose secrets in ordinary English, a generic prefix is fine or can be omitted. For non-prose secrets (a chess game in PGN, a Python snippet, a memo with a fixed format), or for prose with a strong genre signal (a covert message, a poem, a technical email), telling the LLM what kind of text to expect can drop the average rank of secret tokens by an order of magnitude.

```toml
secret_prefix = "The following is a chess game in PGN format:"
secret_prefix = "The following is Python source code:"
secret_prefix = "The following is a meeting agenda:"
secret_prefix = "The following is a brief covert message:"
secret_prefix = "The following is a fragment of a poem:"
```

### Model

The default is `mlx-community/Llama-3.2-3B-Instruct-4bit`: small (~2 GB), fast on Apple Silicon, good enough for short secrets. It's also the cheapest model in the family, which means its next-token distributions are broader at typical positions; rank-`r_i`-th cover picks at moderate `r_i` fall into the tail more often than a larger model's would, and those tail picks are what produce the visible artifacts in the stegotext.

A larger model trades speed for cover quality. Roughly: a 7-8B model is ~3x slower per encoded token than the 3B default, but its sharper distributions push moderate-rank picks toward natural tokens and reduce the cascade effect along the cover.

Tested alternatives:

```toml
model = "mlx-community/Llama-3.1-8B-Instruct-4bit"      # same family, larger
model = "mlx-community/Qwen2.5-7B-Instruct-4bit"        # different family
model = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"   # different family
```

Both sides must use the **same model and the same quantization**. Mixing a 4-bit and an 8-bit copy of the same base model will not round-trip; the logits diverge enough to flip ranks.

## Limitations

- **Apple Silicon only.** MLX runs on Apple GPUs. A CUDA or MPS port is straightforward but not done here.
- **Model-locked.** Both ends need the exact same model and quantization. Mixing a 4-bit and an 8-bit Llama 3.2 3B will not round-trip; the logits diverge.
- **Cover prose has artifacts, and they cluster toward the end.** At positions where the secret-side rank is high, the cover-side pick is a low-probability token under the cover distribution, producing the occasional malformed word or odd phrasing. The effect compounds: each off-distribution pick widens the cover model's next-position distribution, making subsequent picks more likely to also fall into the tail, so artifacts cluster toward the end of the stegotext rather than spreading uniformly. Better cover prose comes from a larger model or a more specific cover prompt; the trade-off is speed.
- **Stegotext length is comparable, not exact.** Because of BPE stability filtering and the trailer, the stegotext is typically 1x to 1.5x the secret's token count, not a strict 1:1. The paper's title says "of the same length"; this implementation produces "of comparable length."
- **Not cryptographic.** The security claim is steganographic: a stegotext should be hard to distinguish from natural samples drawn from the cover distribution. There is no integrity guarantee, no forward secrecy, and no protection against active adversaries who know the keyfile is in use. If you need confidentiality against a serious adversary, use real cryptography (age, PGP, Signal). Stack steganography on top of cryptography only when you need plausible deniability of the message's existence.
- **Tail text is decorative.** The encoder appends a short natural-sampled tail past EOS for a graceful ending; the decoder discards anything after EOS. The tail does not encode anything.

## Citation

If you use this implementation in research, cite the protocol:

```bibtex
@misc{norelli2025calgacus,
    title={LLMs Can Hide Text in Other Text of the Same Length},
    author={Norelli, Antonio and Bronstein, Michael},
    year={2025},
    eprint={2510.20075},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2510.20075}
}
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
