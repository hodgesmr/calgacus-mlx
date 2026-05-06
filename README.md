# calgacus-mlx

> Hide a meaningful text inside another coherent, plausible text of comparable token length.

An MLX implementation of the [Calgacus protocol](https://arxiv.org/abs/2510.20075) (Norelli & Bronstein, 2025) for LLM-based text steganography. Given a secret message, calgacus produces a stegotext that reads like ordinary prose on the surface, but encodes the secret losslessly through the rank choices a language model makes at each token position. A receiver with the same model and the same key recovers the original message exactly.

This is a CLI built around a small Python library. It runs locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Demo

```bash
calgacus init -o demo.key.toml --no-interactive \
    --cover-prompt "My take on Kurt Gödel's incompleteness theorems, after rereading the original 1931 paper:" \
    --secret-prefix "The following is a brief covert message:" \
    --no-seed
```

The keyfile (`demo.key.toml`) bundles the model, cover prompt, and protocol settings that sender and receiver share:

```toml
model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
cover_prompt = "My take on Kurt Gödel's incompleteness theorems, after rereading the original 1931 paper:"
secret_prefix = "The following is a brief covert message:"
style_seed = ""
trailer = "graceful"
```

Encode a secret:

```bash
$ cat demo_secret.txt
We need to talk on Tuesday. The plan changed: the meeting is now at the lighthouse, midnight. Bring the manuscript and tell no one.

$ calgacus encode -k demo.key.toml -s demo_secret.txt -o demo_stego.txt

$ cat demo_stego.txt
We all know it is an easy exercise reading (I don't know if it's extra hard writing first) of course calculus, physics & engineering, compartmentposites, etc. to prove that the universe is finite.  But Gödel's theorems show that it is impossible to prove that the universe is finite.
```

Recover it:

```bash
$ calgacus decode -k demo.key.toml -s demo_stego.txt
We need to talk on Tuesday. The plan changed: the meeting is now at the lighthouse, midnight. Bring the manuscript and tell no one.
```

The recovered text matches the original byte-for-byte. The stegotext, meanwhile, reads as the opening of an essay on Gödel and gives no surface indication that it carries a hidden payload. (There are still cover-quality artifacts where the secret-side ranks are high, like `compartmentposites` and the wrong-way claim about proving the universe is finite. See [Tuning the keyfile](#tuning-the-keyfile) for how to push these down.)

## How it works

The core idea, in three sentences. A language model, given a context, defines a probability distribution over the next token; the same context gives the same distribution. Calgacus uses that distribution as a shared random oracle: the encoder picks a cover token by its rank in the cover-side distribution, where the rank value is the rank of the next secret token in a different (secret-side) distribution. The decoder runs the same two distributions and reverses the mapping.

Concretely:

- **Secret side**. The encoder tokenizes `k' + secret + trailer + EOS` (where `k'` is an optional secret prefix) and walks it position by position. At position `i`, it queries the model for the distribution under the prefix and records `r_i`, the rank of the actual secret token in that distribution.
- **Cover side**. The encoder generates one cover token per rank `r_i`, picking the rank-`r_i`-th most likely token under the cover prompt `k` (and the cover-so-far). These cover tokens, detokenized, are the stegotext.
- **Decode**. The decoder tokenizes `k + stegotext`, walks the cover tokens, looks up each one's rank under the cover-side distribution to recover `r_i`, then replays those ranks against the secret-side distribution. Stop when EOS is hit.

The stegotext's length tracks the secret's length plus a small fixed trailer. Both sides need the same model, the same cover prompt `k`, the same secret prefix `k'`, the same trailer mode, and (if used) the same style seed. The keyfile bundles all of these.

There are three engineering wrinkles the paper alludes to but doesn't fully address. They are easy to get wrong, and getting them wrong silently breaks round-trip:

1. **BPE tokenization stability.** Byte-pair-encoding tokenizers can re-merge tokens depending on their neighbors. If the encoder picks token A then token B, but `detokenize([A, B])` re-tokenizes back to a single token AB, the decoder sees a different token sequence than the encoder wrote. Calgacus filters cover-side picks to "canonical-stable" tokens whose addition to the prefix does not disturb earlier merges. We also fast-path tokens that begin with whitespace, which the BPE essentially never re-merges across.
2. **Numerical determinism.** With bf16 weights, the order of additions in a forward pass affects the last bit or two of the logits, which is enough to flip the rank of two close-together tokens. Bulk-pass and KV-cache forward passes don't always produce identical logits. Calgacus uses the KV cache uniformly on both sides so the rank queries get the same logit values during encode and decode.
3. **Whitespace portability.** Stegotext travels through chat clients and email forwarders that mangle leading whitespace. Calgacus restricts the first cover token to a leading-space-aware token, lstrips one space before emitting the visible stegotext, and prepends one space on the decoder side before tokenizing. The visible text is whitespace-stable; the protocol stays in sync.

## Install

Requires Python 3.11+ and Apple Silicon. MLX is Apple-specific, so this implementation is too. The protocol itself is portable; a CUDA or MPS port would be straightforward.

```bash
uv pip install calgacus-mlx
# or
pip install calgacus-mlx
```

The first encode or decode call downloads the default model (`mlx-community/Llama-3.2-3B-Instruct-4bit`, ~2GB) into your HuggingFace cache. Subsequent runs load from disk in a few seconds.

To run from source:

```bash
git clone https://github.com/hodgesmr/calgacus-mlx
cd calgacus-mlx
uv sync
uv run calgacus --help
```

## Usage

### `calgacus init`

Scaffold a keyfile interactively or from flags. The keyfile is plain TOML and is what sender and receiver must share.

```bash
calgacus init                                    # interactive
calgacus init -o my.key.toml --no-interactive \
    --cover-prompt "I have been wrestling with my front lawn all summer, and I finally" \
    --secret-prefix "The following is a fragment of a poem:" \
    --trailer graceful
```

Both parties need an identical copy of the keyfile. Treat it like a shared secret: anyone with the keyfile can decode anything you encode under it.

### `calgacus encode`

```bash
echo "hello world" | calgacus encode -k my.key.toml
calgacus encode "hello world" -k my.key.toml
calgacus encode -k my.key.toml -s secret.txt -o stego.txt
```

Reads the secret from a positional argument, `--secret-file` (with `-` for stdin), or stdin (default). Writes stegotext to stdout or `--output`.

### `calgacus decode`

```bash
calgacus decode -k my.key.toml -s stego.txt
cat stego.txt | calgacus decode -k my.key.toml
```

Reads stegotext, recovers the secret. The decoder ignores any tail text past EOS, so a stegotext can carry a short natural-sounding ending without affecting recovery.

### Keyfile format

```toml
model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
cover_prompt = "I have been wrestling with my front lawn all summer, and I finally"
secret_prefix = "The following is a personal essay:"
style_seed = "vault sustain orbit modify lounge fragile"
trailer = "graceful"
```

| field | description |
| --- | --- |
| `model` | HuggingFace model ID. Both ends must use the same one (and the same quantization). |
| `cover_prompt` | The opening text of the cover passage. Treat it as the first sentence(s) the model continues into a stegotext, not an instruction. See [Cover prompt](#cover-prompt) for why. |
| `secret_prefix` | Optional incipit prepended to the secret. A genre-matched prefix lowers the secret-side ranks and improves cover quality, especially for non-prose secrets like code, chess PGN, or structured data. |
| `style_seed` | Optional BIP39 word list appended to the cover prompt. Adds entropy to the keyfile without changing the cover topic. Without it, anyone who guesses the cover prompt can decode. |
| `trailer` | `graceful` (default) appends `.\n\n` before EOS so the protocol's stop sentinel sits at a low rank in the cover distribution. `eos-only` skips the trailer; cover quality at the EOS position is slightly worse. |

Flags on `encode` and `decode` (`--cover-prompt`, `--secret-prefix`, `--model`, `--trailer`) override individual keyfile fields, so the same keyfile can be used as a base with per-call adjustments.

## Tuning the keyfile

The cover prompt, secret prefix, and style seed all directly affect stegotext quality. A weak prompt produces a stegotext that reads like nonsense even though round-trip is exact. Spend time on these.

### Cover prompt

The cover prompt is the most consequential setting, and the thing to internalize about it is that it is an **incipit, not an instruction**. Calgacus does not give the LLM a question and read its answer; it gives the LLM the *opening of a passage* and the LLM continues that passage. Write the first sentence (or two) of the kind of text you want as your stegotext, and stop where you want the model to take over.

| works | does not |
| --- | --- |
| `My take on Kurt Gödel's incompleteness theorems, after rereading the 1931 paper:` | `Write a thoughtful overview of Gödel's incompleteness theorems.` |
| `Bella Vita is a small Italian restaurant in our neighborhood, and last Friday I` | `Compose a glowing review of an Italian restaurant called Bella Vita.` |
| `Subject: Friday's all-hands. Team,` | `Write a brief work memo announcing a meeting.` |

Three rules of thumb for incipits:

1. **Set the genre up front.** Mention the kind of text it is (review, essay, memo, recipe). The model's continuation distribution is heavily shaped by what it thinks the document is.
2. **Pin down topic, register, and audience.** Specific incipits keep the natural distribution narrow, which means even moderate-rank cover picks stay readable. Vague incipits collapse into the deep tail and produce off-topic glyphs.
3. **Stop mid-thought.** Ending the incipit on a colon, a comma, or mid-sentence pushes the model into a sharp, well-defined continuation lane. Ending on a period gives the model freedom to start a new thought, and the new thought may not match the topic.

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

### Style seed

Without a style seed, the only secret in the keyfile is the cover prompt. An adversary who guesses the prompt can decode. The style seed adds ~66 bits of entropy by appending six BIP39 words to the prompt; the LLM sees those words as part of its conditioning, but the cover topic and tone are dominated by the prompt, so the visible stegotext stays coherent. Generated automatically by `calgacus init` unless `--no-seed` is passed.

## Limitations

- **Apple Silicon only.** MLX runs on Apple GPUs. A CUDA or MPS port is straightforward but not done here.
- **Model-locked.** Both ends need the exact same model and quantization. Mixing a 4-bit and an 8-bit Llama 3.2 3B will not round-trip; the logits diverge.
- **Cover prose has artifacts.** At positions where the secret-side rank is high, the cover-side pick is a low-probability token under the cover distribution. These produce occasional malformed words or odd phrasings. Better cover prose comes from a larger model or a more permissive cover prompt; the trade-off is speed.
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
