"""BIP39-based style-seed generator for cover prompts.

The protocol's security against brute-forcing the cover prompt benefits
from a high-entropy random suffix that an attacker cannot guess even if
they can guess the prompt's topic from the visible stegotext. Using
English words from the BIP39 wordlist (rather than random base64 or
hex) keeps the seed in-distribution for the LLM, so it does not shift
the cover-side predictions toward code-like or out-of-distribution
fragments.

We use the `mnemonic` package  as a wordlist source. Words are drawn
directly with `secrets.choice`, which is cryptographically secure.
We deliberately do not use the package's full BIP39-with-checksum API.
"""

from __future__ import annotations

import secrets

from mnemonic import Mnemonic


# Loaded once at module import. The wordlist is small (~14 KB) and
# importing the `mnemonic` package itself is cheap, so eager loading
# is fine.
_WORDLIST: list[str] = Mnemonic("english").wordlist


# Punctuation that already terminates a clause cleanly. If the prompt
# ends with one of these, we append the seed directly without
# inserting a period first.
_CLAUSE_END = ".!?:;,"


def generate_seed(num_words: int = 6) -> str:
    """Return a space-separated string of `num_words` random BIP39 English words.

    Each word contributes ~11 bits of entropy (one of 2048). Default
    `num_words=6` gives ~66 bits, which is practically uncrackable
    under realistic threat models, since each brute-force trial
    requires running the LLM.

    Uses `secrets.choice` (cryptographically secure CSPRNG) over the
    BIP39 English wordlist exposed by the `mnemonic` package.
    """
    if num_words < 1:
        raise ValueError(f"num_words must be at least 1, got {num_words}.")
    return " ".join(secrets.choice(_WORDLIST) for _ in range(num_words))


def append_seed_to_prompt(prompt: str, seed: str) -> str:
    """Append `seed` to `prompt` as ` Style seed: <seed>.`

    Handles trailing whitespace and missing punctuation in `prompt` so
    the result reads as natural prose. If `prompt` already ends with
    clause-ending punctuation (., !, ?, :, ;, ,), the seed clause is
    appended directly. Otherwise a period is inserted first to
    separate the clauses cleanly. An empty (or whitespace-only)
    `prompt` returns just the seed clause.

    Examples:
        ("Write a recipe.", "river flame")
            -> "Write a recipe. Style seed: river flame."
        ("Write a recipe", "river flame")
            -> "Write a recipe. Style seed: river flame."
        ("Write a recipe   ", "river flame")
            -> "Write a recipe. Style seed: river flame."
        ("", "river flame")
            -> "Style seed: river flame."
    """
    cleaned = prompt.rstrip()
    if not cleaned:
        return f"Style seed: {seed}."
    if cleaned[-1] not in _CLAUSE_END:
        cleaned = cleaned + "."
    return f"{cleaned} Style seed: {seed}."
