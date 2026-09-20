# 15 — Answer generation, and checking the citations afterwards

Retrieval hands five chunks to a model. The model hands back prose. Everything
this project measured up to here — nDCG, recall, the reranker's failure, MMR's
trade — describes the chunks. This page is about the prose, and about the one
property of it that can be checked mechanically.

## The provider interface

`src/rag/llm.py` is one method: `complete(prompt) -> Completion`. Three
implementations.

| provider | what it is | why it exists |
| --- | --- | --- |
| `ollama` (default) | local model over `http://localhost:11434/api/chat` | no key, no egress, no per-token cost, so the same question can be re-run against five retrieval configs without a budget conversation |
| `openai_compatible` | `POST /chat/completions` | OpenAI, vLLM, llama.cpp's server, Groq and Together all speak this. One provider, every hosted option |
| `extractive` | no model at all | the hallucination baseline |

HTTP is `urllib.request`, not `requests` or `httpx`. Two JSON POSTs do not
justify a dependency, and the project already refuses LangChain for the same
reason.

`temperature=0` is not a quality choice, it is an experimental one. A generator
that answers differently on the second run cannot be used to attribute a change
to retrieval.

### The extractive provider is the interesting one

It stitches the top chunks together with their citations and returns them.
It cannot summarize, cannot synthesize across two files, and cannot answer
"why". It also cannot lie: every line it emits provably came out of the index.

That makes it the control. When the LLM says something the extractive answer
does not contain, that content came from the model's weights, not from this
repository. Running both is the cheapest hallucination test available, and it
works offline on a laptop with no model pulled.

## The prompt

Taken verbatim from the project brief and stored as `SYSTEM_PROMPT`:

```
You are a codebase analysis assistant.

Answer the user's question using ONLY the retrieved code context.

For every important claim:
- identify the file
- identify the class/function
- provide line numbers when available

Do not invent implementation details.

If the retrieved code is insufficient to answer the question,
explicitly say that the available context is insufficient.

Distinguish between:
1. Facts directly visible in the code.
2. Reasonable inferences.

Prefer concise technical explanations.
```

Context is formatted as numbered blocks, each headed by its citation:

```
[1] OpsSense/src/ingestion/indexer.py:12-13  (function _point_id)
def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

[2] OpsSense/data/incidents/incident_015.md:1-27  (section INC-1602)
...
```

The header is doing real work. A model asked to "cite the file" *composes* a
path from what the code looks like; a model handed `path:start-end` at the top
of each block copies it. The citation format is the retrieval key, not prose,
so it can be string-matched afterwards. When `--expand` is on, each block also
carries the imports and parent class signature from docs/13 — that feature was
built for this consumer and this is where it gets used.

## Post-validation

`validate(text, hits)` sorts every reference in the answer into four buckets:

- **grounded** — `path:start-end` that is exactly in the retrieved set, or a
  bare path to a retrieved file (the prompt says "line numbers when available",
  so their absence is not a violation).
- **drifted** — right file, line span the model made up. Sloppy, not dishonest.
- **invented** — a file that was never retrieved. This is the one that matters.
- **bad_refs** — `[7]` when only five blocks were supplied.

Drift and invention are kept apart deliberately. Collapsing them would report
one number that mixes "paraphrased a real citation" with "made up a file", and
those need different fixes.

Path matching is case-insensitive and accepts a bare filename for a retrieved
path. That is a deliberate loosening: llama3.2 emitted
`opsense/src/retrieval/vector_search.py` for a file really called
`OpsSense/src/retrieval/vector_search.py`, and a hallucination detector that
fires on a lowercased repo name is one nobody believes when it fires for real.

`ask` exits 2 when the check fails, so it is usable in a script.

## What the check is worth: a prompt ablation

Seven questions — two answerable, two lexical-gap, three unanswerable — through
llama3.2 with the project prompt, and again with a naive one
(`"You are a helpful coding assistant… cite the files and line numbers you
used."`). Same retrieval, same k=5, same corpus.

| prompt | abstained on unanswerable | invented citations | mean generation |
| --- | --- | --- | --- |
| project | **3 / 3** | 0 / 7 | 2.0 s |
| naive | **0 / 3** | 0 / 7 | 6.1 s |

Per query, project prompt:

| id | kind | abstained | grounded | drift | invented |
| --- | --- | --- | --- | --- | --- |
| q09 | answerable | no | 1 | 0 | - |
| q14 | answerable | no | 1 | 0 | - |
| q23 | lexical-gap | no | 2 | 0 | - |
| q28 | lexical-gap | no | 3 | 0 | - |
| q29 | unanswerable | yes | 0 | 0 | - |
| q30 | unanswerable | yes | 1 | 0 | - |
| q31 | unanswerable | yes | 0 | 0 | - |

Two things fall out of this.

**The abstention instruction is the load-bearing line.** Three of three versus
zero of three, from one paragraph of prompt, on a 3B local model. The
naive prompt also takes three times as long, because a model that will not say
"I don't know" has to produce something instead.

**Neither prompt invented a citation, and that does not mean neither
hallucinated.** Asked for a Kubernetes manifest the repo does not contain, the
naive prompt wrote one:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 2
  ...
```

Fabricated end to end, and it cites nothing, so citation validation reports
`ok — 0 grounded`. **A hallucination with no citations passes the citation
check.** The mechanism catches invented *references*, not invented *content*.
The honest reading of "0 invented" in the table above is "the model did not
attribute anything to a file that was not retrieved", which is a narrower and
much weaker claim than "the answer is true".

That is also the answer to why the extractive provider exists. Citation
validation bounds where claims are *attributed*; the extractive baseline bounds
what was *available* to claim.

## Cost of the context

From docs/13's token counts: k=5 is ~838 tokens of context, k=10 is ~1682, k=20
is ~3398. With `--expand` on, each block gains its imports and class signature —
roughly +30%. At k=5 expanded, a question costs about 1.1k prompt tokens, which
is why `rerank_top_n=20` is a *retrieval* depth and never a generation depth.

## Verify

```bash
# no model needed - the hallucination baseline
python -m src.cli ask "how are point ids generated" --llm-provider extractive

# see the exact prompt without spending a token
python -m src.cli ask "how are point ids generated" --show-prompt

# the real thing
ollama serve &
python -m src.cli ask "how are point ids generated" --expand

# the ablation
python -m scripts.rag_probe --prompt project
python -m scripts.rag_probe --prompt naive

pytest tests/test_rag.py
```

## Alternatives considered

- **Making the model emit structured JSON citations.** Stricter, and it fails
  differently: small local models break the JSON under pressure, and then you
  have neither an answer nor a citation. Free-text with a regex degrades
  gracefully.
- **Asking the model to self-report its confidence.** Correlates with fluency,
  not with correctness, and it costs tokens to be misled.
- **Refusing to print an answer whose citations fail.** Considered and
  rejected: the failure is the interesting output. Printing the answer next to
  `citations: SUSPECT` teaches more than suppressing it, and the exit code
  gives scripts the strict behaviour anyway.
- **An LLM judge scoring answer quality.** The honest version needs a second
  model and a labelled rubric, and it would be judging llama3.2 with something
  that shares its failure modes. The gold set already measures retrieval; this
  page deliberately only claims what can be checked by string matching.
