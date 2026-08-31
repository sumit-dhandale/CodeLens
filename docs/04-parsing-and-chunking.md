# 04 — Parsing and chunking: what a "unit of retrieval" is

A vector index stores fixed-size vectors, so the first real design decision in a
search engine is *what one vector represents*. Too big and the vector is an
average of ten unrelated ideas; too small and it has no context to be about
anything. `src/parser/code_parser.py` finds the natural boundaries and
`src/chunking/code_chunker.py` turns them into chunks.

## Parsing: `ast`, not regex

Python goes through the standard library's `ast`. Every node carries `lineno` and
`end_lineno`, so a function's extent is exact — no brace counting, no indentation
heuristics, no tree-sitter dependency. Two details that a naive walk gets wrong:

- **Decorators.** `ast` reports a function's `lineno` at the `def`, so slicing
  from there silently drops `@property` or `@retry(max=3)`. We take the minimum
  of the `def` line and the decorator lines.
- **Nesting.** `ast.walk` yields nested closures too, which would produce a chunk
  that overlaps its own parent and index the same lines twice. We recurse exactly
  two levels — module → class → method — and let closures live inside their
  parent's chunk.

A file that fails to parse still gets one whole-file symbol rather than
disappearing from the index, and non-Python, non-markdown languages get the same
fallback. Third-party source is untrusted input here: a Python 2 file raises
`SyntaxError`, and a deeply nested literal raises `RecursionError` from inside
`ast.parse`, which is a different exception on a different line. Both are caught,
because one pathological file should not abort a run over thousands.

Markdown splits on ATX headings, with a section running until the next heading of
any level. Fenced code blocks are tracked so that a `# comment` inside a bash
block is not mistaken for a heading. Each section records its ancestor headings in
`class_name`, which is the markdown equivalent of an owning class.

The output is `Symbol{file_path, language, symbol_type, symbol_name, class_name,
start_line, end_line, docstring, imports}`. `symbol_type` is one of `module`,
`class`, `function`, `method`, `section`. Module-level imports are attached to
every symbol in the file: they are a strong signal about what the file does, and
Milestone 5's context expansion needs them.

## Four strategies

| Strategy | One chunk is | Why you'd pick it |
| --- | --- | --- |
| `file` | a whole file | Maximum context, and the strategy most damaged by the model's token limit. |
| `class` | a class, or a top-level function | Keeps a class's methods together, so "how does the client work" retrieves the whole thing. |
| `function` | a method or function (default) | Highest precision: a hit points at the eight lines that answer the question. |
| `fixed` | an N-line window with overlap | Language-agnostic baseline. Splits mid-function, which is exactly the failure mode structural chunking exists to avoid. |

`class` and `function` fall back to the whole file when a file has no symbols — a
constants module like OpsSense's `src/config.py` would otherwise vanish from the
index entirely, and it is the correct answer to "where is chunk size configured".
Under `function`, a class with no methods is still emitted as a chunk for the same
reason.

## Two texts per chunk

This is the part that actually moves retrieval numbers.

- `text` is verbatim source. It is what gets displayed, cited, and fed to the LLM.
- `embed_text` is what gets embedded: the relative path, the language and
  qualified symbol name, the symbol name **split on snake_case and camelCase**,
  the docstring, then the body.

The query "where do we retry a failed request" shares no token with
`_shouldRetry`. Split into `should retry`, it does. Embedding the enriched form
and displaying the verbatim form gets both properties at once. The path is
included with separators loosened (`src / ingestion / chunker.py`) because
directory names carry real topical signal.

## Token budget

MiniLM's encoder truncates at 256 wordpieces and does so *silently*: you get a
vector back either way, but it represents the first third of a long chunk and
nothing after. `python -m src.cli chunk --stats` reports the truncation rate per
strategy so this is a measured number rather than an assumption. At `da938e09`:

| strategy | chunks | mean tok | p95 | max | trunc |
| --- | --- | --- | --- | --- | --- |
| file | 50 | 357.4 | 700 | 3209 | 66% |
| class | 82 | 201.8 | 380 | 458 | 33% |
| function | 85 | 195.6 | 380 | 458 | 32% |
| fixed | 49 | 366.3 | 739 | 845 | 69% |

Two thirds of file-level chunks are truncated, and that single fact is most of
the explanation for Experiment 2's results. The counts above come from a
heuristic estimator, because the exact wordpiece count needs `transformers`
installed; pass `--token-model sentence-transformers/all-MiniLM-L6-v2` once the
embedding dependencies are in place to get exact numbers.

Note that `fixed` at the default 60 lines is *worse* than file-level on this
corpus: the repo's files are short, so a 60-line window is nearly a file but with
functions cut in half. Line counts are a bad proxy for token counts.

## Making the comparison run cheap

`--stats` chunks the corpus four times, which profiling showed was doing three
times more work than necessary in two places. Both fixes are small and both
matter more as the corpus grows:

- **`ast.parse` ran once per strategy per file.** `code_parser.parse` is now
  `lru_cache`d on the (frozen, hashable) `SourceFile`. This costs no extra memory
  because the caller already holds every file in RAM, and it is why `parse`
  returns a tuple: a cached result is shared, so it must not be mutable.
- **Every chunk re-split the whole file into lines** to slice out its own span,
  making chunking O(chunks x file size). Lines are now split once per file and
  passed down.

Token estimation was the single largest cost — over half the profile, because it
runs a Python-level expression per word per chunk. Integer `(len + 3) >> 2`
instead of `math.ceil(len / 4)` gives identical counts for less work, and the
exact path is now batched (`count_tokens_batch`), since one tokenizer call per
chunk is dominated by call overhead rather than tokenization. Together: **0.32s →
0.14s** on a 1000-file synthetic corpus, byte-identical output.

An unresolvable `--token-model` logs one warning and falls back to estimates
rather than silently reporting heuristic numbers as if they were exact — the
`None` is cached alongside the real tokenizers so the warning fires once, not
once per chunk. When the flag is omitted, `chunk` uses `cfg.model` for exact
counts.

## Stable point IDs

Each chunk exposes `point_key` and `point_id` (a `uuid5` over the key). The key
includes `start_line` so two symbols with the same name in one file cannot
collide:

```
{file_path}:{class_name}:{symbol_name}:{start_line}:{strategy}
```

- **Stable across content edits** at the same line span: re-indexing updates the
  point in place.
- **Content hash in the payload** (Milestone 3) detects when the body changed
  without the span moving.
- **Sync step** (Milestone 3) deletes points whose `(path, start_line, strategy)`
  no longer exist after a refactor.

## Tooling

`pyproject.toml` configures ruff for both lint and format, with pyflakes, import
ordering, pyupgrade, bugbear, and bandit's security rules enabled. Style is not
worth a code review comment when a formatter can settle it:

```bash
ruff format src tests && ruff check src tests
```

## The gold set

`eval/opssense.json` holds 32 queries pinned to commit `da938e09`: 21 code
queries, 1 docs query, 6 lexical-gap queries against the incident write-ups, and 4
unanswerable ones. Labels are graded — 2 for the chunk that actually answers the
question, 1 for an acceptable neighbour — so nDCG has something to work with.

Unanswerable queries exist because a vector index always returns its top-k. Ask
about TLS certificate expiry in a repo full of latency incidents and you *will*
get a confident, wrong incident back. Measuring that is the only way to know
whether a similarity threshold is worth having.

`tests/test_gold_set.py` asserts that every label resolves to a chunk that
actually exists, so a re-parse or a strategy change cannot rot the labels quietly.

## Verify

```bash
python -m src.cli parse --list
python -m src.cli chunk --stats
python -m src.cli chunk --strategy function --list
pytest tests/test_chunker.py tests/test_gold_set.py
```

## Alternatives considered

- **tree-sitter** — the right answer for a multi-language indexer, and the only
  way to parse Ruby or TypeScript properly. It is a build dependency plus a
  grammar per language, for a project currently indexing Python and markdown.
  The `parse()` dispatch is a single `if` away from accommodating it.
- **Token-count windows instead of line windows** — more correct than `fixed`,
  but it requires the tokenizer at chunk time, which couples chunking to the
  embedding model. Worth revisiting once the embedder exists.
- **Chunking on `ast` statement groups** — splitting a 500-line function into
  logical blocks. Real technique, no 500-line functions in this corpus.
