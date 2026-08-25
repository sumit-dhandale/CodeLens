# 03 — File discovery: what gets indexed and what does not

`src/parser/repository_loader.py` walks a checkout and returns
`SourceFile{file_path, language, content, sha}`, with `file_path` relative to the
repo root in posix form. Relative paths matter because they are embedded into the
chunk text and printed as citations; absolute paths would leak the machine's
directory layout into the vectors themselves.

## The filters, and the cost of skipping each

Everything here exists because the alternative pollutes retrieval:

| Rule | Why |
| --- | --- |
| Extension must be in the language map | An unmapped extension has no parser and no sensible chunking. Silence beats a bad chunk. |
| Skip `.git`, `__pycache__`, `.venv`, `node_modules`, `dist`, `build` | Dependency and build output can outnumber first-party code 100:1, and every query would return a stranger's library. |
| Skip dotfiles and dot-directories | Config and CI noise, no retrieval value. |
| Skip lockfiles (`yarn.lock`, `poetry.lock`, …) | Thousands of near-identical lines that dominate a BM25 index. |
| Skip `*.min.js`, `*_pb2.py`, and friends | Generated. One "line" can be 200 KB. |
| Size cap (512 KB default) | Guards against a checked-in data blob with a source extension. |
| Skip symlinks | A symlink out of the tree reads files the user never asked to index. |
| Skip anything containing a NUL byte or failing UTF-8 decode | Binary content with a text extension. |

The `.py`/`.md` split is deliberate rather than incidental. The incident write-ups
in `data/incidents/*.md` are the best lexical-gap material available: a user types
"timeout" and the doc says "deadline exceeded" or "pool exhaustion". Tagging them
`language: markdown` means a payload filter can isolate code-only versus docs-only
retrieval and measure how much each contributes.

## Content hash

Each file carries `sha256(bytes)`. Later milestones store it in the Qdrant payload
so re-indexing can skip unchanged files, and so a point whose stored hash differs
from the current one is known to need re-embedding. Hashing the raw bytes rather
than the decoded string keeps it stable regardless of decode settings.

## Verify

```bash
python -m src.cli fetch --repo https://github.com/sumit-dhandale/OpsSense
python -m src.cli load  --repo https://github.com/sumit-dhandale/OpsSense --list
pytest tests/test_loader.py
```

At commit `da938e09` this prints 29 Python and 21 markdown files, 50 total.

## Alternatives considered

- **Honour the repo's `.gitignore`** — the correct long-term answer, and free if we
  shell out to `git ls-files`. It only works for git checkouts though, and the
  loader also needs to run against a plain `--path` directory. The hardcoded ignore
  set covers the same ground for the repos we target.
- **`pathspec` library** — a real gitignore matcher, but a dependency for a problem
  a frozenset of directory names currently solves.
