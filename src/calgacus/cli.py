"""Click-based CLI: init, encode, decode.

Three subcommands:

- `init`: scaffold a TOML keyfile, interactively or from flags.
- `encode`: encode a secret into a stegotext under a cover prompt.
- `decode`: decode a stegotext back to the original secret.

Both `encode` and `decode` accept their text input three ways:
positional argument, `--*-file` flag (use `-` for stdin), or stdin
(default if neither is given). Files and stdin are decoded as UTF-8
strict; binary input is rejected with a clear error.

Configuration sources, in increasing precedence:
1. Built-in defaults.
2. A TOML keyfile loaded via `--key`.
3. CLI flags (`--cover-prompt`, `--secret-prefix`, `--model`, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import core
from .keyfile import KeyFile, load_key, save_key
from .model import MLXModel
from .termination import Trailer


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_TAIL_MAX_TOKENS = 32
TRAILER_CHOICES = [t.value for t in Trailer]


# Helpers --------------------------------------------------------------


def _read_stdin() -> str:
    """Read all of stdin as UTF-8 strict text."""
    raw = sys.stdin.buffer.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise click.UsageError(
            f"Stdin is not valid UTF-8: {e}. Calgacus operates on text "
            f"only, not binary data."
        ) from None


def _read_text_file(path: Path) -> str:
    """Read a file as UTF-8 strict, with `-` meaning stdin."""
    if str(path) == "-":
        return _read_stdin()
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as e:
        raise click.UsageError(
            f"File {path} is not valid UTF-8: {e}. Calgacus operates on "
            f"text only, not binary data."
        ) from None
    except FileNotFoundError:
        raise click.UsageError(f"File not found: {path}.") from None


def _resolve_text_input(
    text: str | None,
    file: Path | None,
    name: str,
    *,
    stdin_fallback: bool,
) -> str | None:
    """Resolve a piece of text input from positional/flag, file, or stdin.

    Returns the resolved string, or None if nothing was provided and
    `stdin_fallback` is False. Raises UsageError if both `text` and
    `file` are set (that is ambiguous).
    """
    if text is not None and file is not None:
        raise click.UsageError(
            f"Cannot pass both --{name} and --{name}-file."
        )
    if text is not None:
        return text
    if file is not None:
        return _read_text_file(file)
    if stdin_fallback:
        return _read_stdin()
    return None


def _resolve_trailer(
    trailer_flag: str | None, keyfile: KeyFile | None
) -> Trailer:
    """Pick the trailer mode: CLI flag wins, then keyfile, then GRACEFUL."""
    if trailer_flag is not None:
        return Trailer(trailer_flag)
    if keyfile is not None:
        return keyfile.trailer
    return Trailer.GRACEFUL


def _resolve_cover_prompt(
    cover_prompt: str | None,
    cover_prompt_file: Path | None,
    keyfile: KeyFile | None,
) -> str:
    """Pick the cover prompt seen by the LLM at runtime.

    Order of precedence: --cover-prompt / --cover-prompt-file flag,
    then the keyfile's `cover_prompt` field. Raises UsageError if
    neither path produces a prompt.
    """
    cli_text = _resolve_text_input(
        cover_prompt, cover_prompt_file, "cover-prompt",
        stdin_fallback=False,
    )
    if cli_text is not None:
        return cli_text
    if keyfile is not None:
        return keyfile.cover_prompt
    raise click.UsageError(
        "No cover prompt provided. Pass --cover-prompt, "
        "--cover-prompt-file, or --key. To create a reusable keyfile, "
        "run: calgacus init"
    )


def _resolve_secret_prefix(
    secret_prefix: str | None,
    secret_prefix_file: Path | None,
    keyfile: KeyFile | None,
) -> str:
    """Pick the effective secret prefix. CLI flag wins, then keyfile,
    then empty string."""
    cli_text = _resolve_text_input(
        secret_prefix, secret_prefix_file, "secret-prefix",
        stdin_fallback=False,
    )
    if cli_text is not None:
        return cli_text
    if keyfile is not None:
        return keyfile.secret_prefix
    return ""


def _resolve_model(model_flag: str | None, keyfile: KeyFile | None) -> str:
    """Pick the model: CLI flag wins, then keyfile, then default."""
    if model_flag:
        return model_flag
    if keyfile is not None:
        return keyfile.model
    return DEFAULT_MODEL


# Click group ----------------------------------------------------------


@click.group()
@click.version_option(package_name="calgacus-mlx")
def main() -> None:
    """Calgacus: hide a meaningful text inside another coherent, plausible
    text of comparable token length.
    """


# init -----------------------------------------------------------------


@main.command()
@click.option("-m", "--model", default=None, help="MLX model ID.")
@click.option(
    "-c", "--cover-prompt", default=None,
    help="Cover prompt k. Steers the topic and style of the stegotext.",
)
@click.option(
    "-C", "--cover-prompt-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Read cover prompt from file (- for stdin).",
)
@click.option(
    "-p", "--secret-prefix", default=None,
    help="Secret prefix k'. Optional context placed before the secret.",
)
@click.option(
    "-P", "--secret-prefix-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Read secret prefix from file (- for stdin).",
)
@click.option(
    "--trailer", type=click.Choice(TRAILER_CHOICES), default=None,
    help="Trailer mode (default: graceful).",
)
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("calgacus.key.toml"), show_default=True,
    help="Where to write the keyfile.",
)
@click.option(
    "--force", is_flag=True, help="Overwrite if output exists.",
)
@click.option(
    "--no-interactive", is_flag=True,
    help="Fail rather than prompt for missing values.",
)
def init(
    model: str | None,
    cover_prompt: str | None,
    cover_prompt_file: Path | None,
    secret_prefix: str | None,
    secret_prefix_file: Path | None,
    trailer: str | None,
    output: Path,
    force: bool,
    no_interactive: bool,
) -> None:
    """Scaffold a TOML keyfile.

    Interactive by default. Pass --no-interactive (and the appropriate
    flags) to script keyfile creation in CI or automation.
    """
    interactive = not no_interactive

    if output.exists() and not force:
        raise click.UsageError(
            f"Output file {output} already exists. Use --force to overwrite."
        )

    if interactive and model is None:
        click.echo()
        click.echo(
            "Welcome. Let's create a Calgacus keyfile. The keyfile bundles\n"
            "the model, cover prompt, and other settings that sender and\n"
            "receiver must share. Both parties need an identical copy of\n"
            "this file."
        )
        click.echo()

    resolved_model = model or (
        click.prompt("Model", default=DEFAULT_MODEL)
        if interactive else DEFAULT_MODEL
    )

    cover_prompt_text = _resolve_text_input(
        cover_prompt, cover_prompt_file, "cover-prompt",
        stdin_fallback=False,
    )
    if cover_prompt_text is None:
        if not interactive:
            raise click.UsageError(
                "Cover prompt required. Pass --cover-prompt or "
                "--cover-prompt-file."
            )
        cover_prompt_text = click.prompt("Cover prompt")

    secret_prefix_text = _resolve_text_input(
        secret_prefix, secret_prefix_file, "secret-prefix",
        stdin_fallback=False,
    )
    if secret_prefix_text is None:
        if not interactive:
            secret_prefix_text = ""
        else:
            secret_prefix_text = click.prompt(
                "Secret prefix", default="", show_default=False,
            )

    trailer_enum = Trailer(trailer) if trailer else Trailer.GRACEFUL

    key = KeyFile(
        model=resolved_model,
        cover_prompt=cover_prompt_text,
        secret_prefix=secret_prefix_text,
        trailer=trailer_enum,
    )
    save_key(key, output)

    click.echo()
    click.echo(f"Wrote {output}.")
    click.echo("Share this file with anyone who needs to decode your messages.")


# encode ---------------------------------------------------------------


@main.command()
@click.argument("secret_arg", required=False, default=None, metavar="SECRET")
@click.option(
    "-s", "--secret-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Read secret from file (- for stdin).",
)
@click.option(
    "-c", "--cover-prompt", default=None,
    help="Cover prompt k (overrides keyfile).",
)
@click.option(
    "-C", "--cover-prompt-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Read cover prompt from file (- for stdin).",
)
@click.option(
    "-p", "--secret-prefix", default=None,
    help="Secret prefix k' (overrides keyfile).",
)
@click.option(
    "-P", "--secret-prefix-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Read secret prefix from file (- for stdin).",
)
@click.option("-m", "--model", default=None, help="MLX model ID (overrides keyfile).")
@click.option(
    "--trailer", type=click.Choice(TRAILER_CHOICES), default=None,
    help="Trailer mode (overrides keyfile).",
)
@click.option(
    "--tail-max-tokens", type=int, default=DEFAULT_TAIL_MAX_TOKENS,
    show_default=True,
    help="Max greedy tail tokens to append after the rank-driven payload.",
)
@click.option(
    "-k", "--key",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Load model/k/k'/trailer from a TOML keyfile.",
)
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Write stegotext to file (default: stdout).",
)
@click.option(
    "--quiet", is_flag=True, help="Suppress progress on stderr.",
)
def encode(
    secret_arg: str | None,
    secret_file: Path | None,
    cover_prompt: str | None,
    cover_prompt_file: Path | None,
    secret_prefix: str | None,
    secret_prefix_file: Path | None,
    model: str | None,
    trailer: str | None,
    tail_max_tokens: int,
    key: Path | None,
    output: Path | None,
    quiet: bool,
) -> None:
    """Encode a secret into a stegotext.

    SECRET can be passed as a positional argument, via --secret-file
    (with `-` meaning stdin), or piped on stdin (default if neither
    is given).
    """
    keyfile = load_key(key) if key else None

    model_id = _resolve_model(model, keyfile)
    effective_prompt = _resolve_cover_prompt(
        cover_prompt, cover_prompt_file, keyfile,
    )
    secret_prefix_text = _resolve_secret_prefix(
        secret_prefix, secret_prefix_file, keyfile,
    )
    trailer_enum = _resolve_trailer(trailer, keyfile)

    secret_text = _resolve_text_input(
        secret_arg, secret_file, "secret", stdin_fallback=True,
    )
    if not secret_text:
        raise click.UsageError("Empty secret. Provide text to encode.")

    if not quiet:
        click.echo(f"Loading model {model_id}...", err=True)
    try:
        mdl = MLXModel(model_id)
    except ValueError as e:
        raise click.ClickException(str(e)) from None

    if not quiet:
        click.echo("Encoding...", err=True)
    try:
        stego = core.encode(
            mdl, effective_prompt, secret_text,
            secret_prefix=secret_prefix_text,
            trailer=trailer_enum,
            tail_max_tokens=tail_max_tokens,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from None

    if output:
        output.write_text(stego, encoding="utf-8")
    else:
        click.echo(stego, nl=False)
        if not stego.endswith("\n"):
            click.echo()

    if not key and not quiet:
        click.echo(
            "Tip: save these settings as a keyfile with `calgacus init`.",
            err=True,
        )


# decode ---------------------------------------------------------------


@main.command()
@click.argument("stego_arg", required=False, default=None, metavar="STEGO")
@click.option(
    "-s", "--stego-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Read stegotext from file (- for stdin).",
)
@click.option(
    "-c", "--cover-prompt", default=None,
    help="Cover prompt k (overrides keyfile).",
)
@click.option(
    "-C", "--cover-prompt-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Read cover prompt from file (- for stdin).",
)
@click.option(
    "-p", "--secret-prefix", default=None,
    help="Secret prefix k' (overrides keyfile).",
)
@click.option(
    "-P", "--secret-prefix-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Read secret prefix from file (- for stdin).",
)
@click.option("-m", "--model", default=None, help="MLX model ID (overrides keyfile).")
@click.option(
    "--trailer", type=click.Choice(TRAILER_CHOICES), default=None,
    help="Trailer mode (overrides keyfile).",
)
@click.option(
    "-k", "--key",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Load model/k/k'/trailer from a TOML keyfile.",
)
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None, help="Write recovered secret to file (default: stdout).",
)
@click.option("--quiet", is_flag=True, help="Suppress progress on stderr.")
def decode(
    stego_arg: str | None,
    stego_file: Path | None,
    cover_prompt: str | None,
    cover_prompt_file: Path | None,
    secret_prefix: str | None,
    secret_prefix_file: Path | None,
    model: str | None,
    trailer: str | None,
    key: Path | None,
    output: Path | None,
    quiet: bool,
) -> None:
    """Decode a stegotext back to the original secret.

    STEGO can be passed as a positional argument, via --stego-file
    (with `-` meaning stdin), or piped on stdin (default if neither
    is given).
    """
    keyfile = load_key(key) if key else None

    model_id = _resolve_model(model, keyfile)
    effective_prompt = _resolve_cover_prompt(
        cover_prompt, cover_prompt_file, keyfile,
    )
    secret_prefix_text = _resolve_secret_prefix(
        secret_prefix, secret_prefix_file, keyfile,
    )
    trailer_enum = _resolve_trailer(trailer, keyfile)

    stego_text = _resolve_text_input(
        stego_arg, stego_file, "stego", stdin_fallback=True,
    )
    if not stego_text:
        raise click.UsageError("Empty stegotext. Provide text to decode.")

    if not quiet:
        click.echo(f"Loading model {model_id}...", err=True)
    try:
        mdl = MLXModel(model_id)
    except ValueError as e:
        raise click.ClickException(str(e)) from None

    if not quiet:
        click.echo("Decoding...", err=True)
    try:
        secret_text = core.decode(
            mdl, effective_prompt, stego_text,
            secret_prefix=secret_prefix_text,
            trailer=trailer_enum,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from None

    if output:
        output.write_text(secret_text, encoding="utf-8")
    else:
        click.echo(secret_text, nl=False)
        if not secret_text.endswith("\n"):
            click.echo()


if __name__ == "__main__":
    main()
