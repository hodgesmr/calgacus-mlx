"""TOML keyfile load/save and the KeyFile dataclass.

A keyfile bundles the protocol parameters that sender and receiver
must agree on: model, cover prompt, secret prefix, and trailer mode.
It lives as a TOML file that two parties share verbatim.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

from .termination import Trailer


@dataclass
class KeyFile:
    """Protocol parameters shared between sender and receiver.

    Fields:
        model: HuggingFace MLX model ID (e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit").
            Both sides must use the same model for identical tokenization
            and identical rank sequences.
        cover_prompt: The cover-side prompt `k`. Steers the topic and tone
            of the stegotext.
        secret_prefix: Optional prefix `k'` placed before the secret to
            give the LLM context (e.g. "The following is a personal essay:").
            Empty by default.
        trailer: Trailer mode for the secret-side rank sequence
            (`GRACEFUL` or `EOS_ONLY`). Default `GRACEFUL`. Sender and
            receiver must use the same value or the decoder strips the
            wrong number of tokens.
    """

    model: str
    cover_prompt: str
    secret_prefix: str = ""
    trailer: Trailer = field(default=Trailer.GRACEFUL)


def load_key(path: Path) -> KeyFile:
    """Load a keyfile from `path`.

    Required fields: `model`, `cover_prompt`. Optional fields:
    `secret_prefix` (default ""), `trailer` (default "graceful").

    Raises ValueError on malformed or incomplete keyfiles, with the
    specific missing or invalid field named in the message.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    missing = [k for k in ("model", "cover_prompt") if k not in data]
    if missing:
        raise ValueError(
            f"Keyfile {path} is missing required field(s): "
            f"{', '.join(missing)}."
        )

    trailer_str = data.get("trailer", Trailer.GRACEFUL.value)
    try:
        trailer = Trailer(trailer_str)
    except ValueError:
        valid = ", ".join(t.value for t in Trailer)
        raise ValueError(
            f"Keyfile {path} has invalid trailer {trailer_str!r}. "
            f"Valid values: {valid}."
        ) from None

    return KeyFile(
        model=data["model"],
        cover_prompt=data["cover_prompt"],
        secret_prefix=data.get("secret_prefix", ""),
        trailer=trailer,
    )


def save_key(key: KeyFile, path: Path) -> None:
    """Save `key` to `path` as a TOML file.

    Always writes all fields, including empty strings for
    `secret_prefix`, so the resulting file makes its schema visible
    to a human reader.
    """
    data: dict[str, object] = {
        "model": key.model,
        "cover_prompt": key.cover_prompt,
        "secret_prefix": key.secret_prefix,
        "trailer": key.trailer.value,
    }
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
