"""Prompt assembly, provider selection, and citation post-validation.

No network: the HTTP providers are exercised through a stubbed `_post`, and the
extractive provider needs no model at all.
"""

import pytest

from src.config import Config
from src.rag import llm as llm_mod
from src.rag.generator import SYSTEM_PROMPT, Generator, validate
from src.vector_store.qdrant_store import Hit


def hit(path="pkg/client.py", start=20, end=25, symbol="send_request", text="    return url"):
    return Hit(
        score=0.5,
        file_path=path,
        language="python",
        symbol_type="method",
        symbol_name=symbol,
        class_name="Client",
        start_line=start,
        end_line=end,
        text=text,
    )


@pytest.fixture
def hits():
    return [hit(), hit(path="pkg/pool.py", start=5, end=9, symbol="acquire", text="    pass")]


class StubLLM(llm_mod.LLM):
    """Returns whatever answer the test wants, and records the prompt it saw."""

    name, model = "stub", "stub-1"

    def __init__(self, text):
        self.text = text
        self.prompt = None

    def complete(self, prompt):
        self.prompt = prompt
        return self._done(self.text, 0.0)


# --- prompt assembly ---------------------------------------------------------


def test_context_is_numbered_blocks_headed_by_citations(hits):
    prompt = Generator(Config(), llm=StubLLM("")).prompt("how does it send?", hits)
    assert prompt.system == SYSTEM_PROMPT
    assert "[1] pkg/client.py:20-25  (method Client.send_request)" in prompt.user
    assert "[2] pkg/pool.py:5-9" in prompt.user
    assert prompt.user.rstrip().endswith("Question: how does it send?")


def test_empty_retrieval_still_produces_a_usable_prompt():
    prompt = Generator(Config(), llm=StubLLM("")).prompt("anything", [])
    assert "(no context retrieved)" in prompt.user


# --- citation validation -----------------------------------------------------


def test_exact_citation_is_grounded(hits):
    check = validate("See pkg/client.py:20-25 for the send path.", hits)
    assert check.grounded == ("pkg/client.py:20-25",)
    assert check.ok


def test_wrong_lines_in_a_real_file_is_drift_not_invention(hits):
    check = validate("See pkg/client.py:40-58.", hits)
    assert check.drifted == ("pkg/client.py:40-58",)
    assert check.invented == ()
    # Drift is sloppy, not dishonest: the file really was retrieved.
    assert check.ok


def test_a_file_that_was_never_retrieved_is_flagged(hits):
    check = validate("Retries live in src/core/retry.py:12-30.", hits)
    assert check.invented == ("src/core/retry.py:12-30",)
    assert not check.ok


def test_bare_path_to_a_retrieved_file_counts_as_grounded(hits):
    # The prompt asks for line numbers "when available", so their absence is
    # not a hallucination.
    assert validate("Defined in pkg/pool.py.", hits).grounded == ("pkg/pool.py",)


def test_dangling_block_reference_is_flagged(hits):
    check = validate("As shown in [1] and [7].", hits)
    assert check.bad_refs == (7,)
    assert not check.ok


def test_repeated_citation_is_counted_once(hits):
    check = validate("pkg/client.py:20-25 and again pkg/client.py:20-25", hits)
    assert check.total == 1


# --- providers ---------------------------------------------------------------


def test_extractive_quotes_only_retrieved_text(hits):
    answer = Generator(Config(llm_provider="extractive")).answer("why?", hits)
    # The whole point of the baseline: it cannot produce an ungrounded citation.
    assert answer.citations.invented == ()
    assert "pkg/client.py:20-25" in answer.text
    assert answer.provider == "extractive"


def test_extractive_says_so_when_nothing_was_retrieved():
    answer = Generator(Config(llm_provider="extractive")).answer("why?", [])
    assert answer.abstained


def test_ollama_sends_system_and_user_and_reads_message_content(hits, monkeypatch):
    sent = {}

    def fake_post(self, path, body, headers=None):
        sent.update(path=path, body=body)
        return {"message": {"content": "grounded in pkg/pool.py:5-9"}}

    monkeypatch.setattr(llm_mod._HttpLLM, "_post", fake_post)
    answer = Generator(Config(llm_provider="ollama")).answer("why?", hits)

    assert sent["path"] == "/api/chat"
    assert sent["body"]["stream"] is False
    assert sent["body"]["options"]["temperature"] == 0.0
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]
    assert answer.citations.grounded == ("pkg/pool.py:5-9",)


def test_openai_compatible_needs_a_key_and_reads_choices(hits, monkeypatch):
    monkeypatch.setattr(
        llm_mod._HttpLLM,
        "_post",
        lambda self, path, body, headers=None: {"choices": [{"message": {"content": "ok"}}]},
    )
    cfg = Config(llm_provider="openai_compatible", llm_base_url="https://api.example.com/v1")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm_mod.build(cfg).complete(Generator(cfg, llm=StubLLM("")).prompt("q", hits))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm_mod.build(cfg).complete(Generator(cfg, llm=StubLLM("")).prompt("q", hits)).text == "ok"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "not-a-url", "ftp://host/x"])
def test_non_http_base_url_is_rejected(url):
    with pytest.raises(ValueError, match="http"):
        llm_mod.build(Config(llm_provider="ollama", llm_base_url=url))


def test_unknown_provider_fails_validation():
    with pytest.raises(ValueError, match="llm_provider"):
        Config(llm_provider="gpt5-vibes").validate()
