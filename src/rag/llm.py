"""Pluggable answer generators behind one tiny interface.

Three providers, chosen by `--llm-provider`:

- `ollama` (default): a local model over HTTP. No key, no egress, no per-token
  cost, which matters when the whole point is to re-run the same question
  against five retrieval configurations.
- `openai_compatible`: the same chat-completions shape that OpenAI, vLLM,
  llama.cpp's server, Together and Groq all speak. One provider covers all of
  them because the wire format is the de-facto standard.
- `extractive`: no model at all. It stitches the retrieved chunks together with
  their citations. This is the hallucination baseline: every line of its output
  provably came from the index, so any claim the LLM makes that the extractive
  answer cannot support is the LLM's invention, not retrieval's.

HTTP is `urllib.request` rather than `requests`/`httpx`. Two JSON POSTs do not
justify a dependency.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from ..config import LLM_PROVIDERS, Config

# Enough of the chunk for the reader to judge the citation without scrolling.
EXTRACTIVE_LINES = 12


@dataclass(frozen=True)
class Block:
    """One retrieved chunk, as the prompt will present it."""

    index: int
    citation: str
    qualified_name: str
    symbol_type: str
    text: str

    def render(self) -> str:
        head = f"[{self.index}] {self.citation}"
        if self.qualified_name:
            head += f"  ({self.symbol_type} {self.qualified_name})"
        return f"{head}\n{self.text}"


@dataclass(frozen=True)
class Prompt:
    system: str
    question: str
    blocks: tuple[Block, ...]

    @property
    def user(self) -> str:
        context = "\n\n".join(b.render() for b in self.blocks) or "(no context retrieved)"
        return f"Retrieved context:\n\n{context}\n\nQuestion: {self.question}"

    @property
    def chars(self) -> int:
        return len(self.system) + len(self.user)


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    elapsed_ms: float


class LLM:
    """`complete(prompt) -> Completion`. That is the entire contract."""

    name = "llm"
    model = ""

    def complete(self, prompt: Prompt) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError

    def _done(self, text: str, started: float) -> Completion:
        return Completion(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


class ExtractiveLLM(LLM):
    """Quotes the top chunks verbatim. Cannot hallucinate, cannot synthesize."""

    name = "extractive"
    model = "none"

    def __init__(self, cfg: Config):
        self.max_blocks = max(1, cfg.llm_extract_blocks)

    def complete(self, prompt: Prompt) -> Completion:
        started = time.perf_counter()
        if not prompt.blocks:
            return self._done("The available context is insufficient: nothing was retrieved.", started)
        parts = [
            "No LLM was used. The passages below are the retrieved context, quoted verbatim.",
            "",
        ]
        for block in prompt.blocks[: self.max_blocks]:
            lines = block.text.splitlines()
            body = "\n".join(lines[:EXTRACTIVE_LINES])
            if len(lines) > EXTRACTIVE_LINES:
                body += f"\n    ... ({len(lines) - EXTRACTIVE_LINES} more lines)"
            parts.append(f"[{block.index}] {block.citation} {block.qualified_name}".rstrip())
            parts.append(body)
            parts.append("")
        return self._done("\n".join(parts), started)


class _HttpLLM(LLM):
    def __init__(self, cfg: Config):
        self.model = cfg.llm_model
        self.timeout = cfg.llm_timeout
        self.temperature = cfg.llm_temperature
        self.base_url = _checked_base_url(cfg.llm_base_url)

    def _post(self, path: str, body: dict, headers: dict[str, str] | None = None) -> dict:
        request = urllib.request.Request(  # noqa: S310 - scheme checked in _checked_base_url
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{self.name} {path} returned {exc.code}: {exc.read()[:200]!r}") from exc
        except OSError as exc:
            raise RuntimeError(f"{self.name} at {self.base_url} unreachable: {exc}") from exc


class OllamaLLM(_HttpLLM):
    """`POST /api/chat` with `stream:false`, so one request is one answer."""

    name = "ollama"

    def complete(self, prompt: Prompt) -> Completion:
        started = time.perf_counter()
        data = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                "stream": False,
                # Greedy decoding: an experiment you cannot re-run is not an
                # experiment. Two runs of the same question must agree.
                "options": {"temperature": self.temperature},
            },
        )
        return self._done(data.get("message", {}).get("content", ""), started)


class OpenAICompatibleLLM(_HttpLLM):
    """Anything speaking `/chat/completions`: OpenAI, vLLM, Groq, llama.cpp."""

    name = "openai_compatible"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.api_key_env = cfg.llm_api_key_env

    def complete(self, prompt: Prompt) -> Completion:
        started = time.perf_counter()
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"${self.api_key_env} is not set")
        data = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                "temperature": self.temperature,
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        choices = data.get("choices") or [{}]
        return self._done(choices[0].get("message", {}).get("content", ""), started)


def _checked_base_url(url: str) -> str:
    """Only http/https, and no `file:`/`gopher:` smuggled in through config."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"llm_base_url must be an http(s) url, got {url!r}")
    return url.rstrip("/")


def build(cfg: Config) -> LLM:
    match cfg.llm_provider:
        case "extractive":
            return ExtractiveLLM(cfg)
        case "ollama":
            return OllamaLLM(cfg)
        case "openai_compatible":
            return OpenAICompatibleLLM(cfg)
        case _:
            raise ValueError(
                f"unknown llm_provider {cfg.llm_provider!r}, expected one of {LLM_PROVIDERS}"
            )
