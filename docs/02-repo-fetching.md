# 02 — Fetching a repo, and why the commit SHA is not optional

`src/parser/repo_fetcher.py` turns a GitHub URL into a directory on disk plus a
commit SHA.

## Shallow clone

```
git clone --depth 1 --quiet https://github.com/<owner>/<name>.git <tmp>
```

`--depth 1` fetches one commit's tree instead of the entire history. For this
project history is pure overhead: we index the current state of the files, and
nothing downstream reads a parent commit.

## Why pin the SHA

The eval gold set maps a query to an expected `(file_path, symbol_name)` pair.
Those coordinates are only meaningful at a specific commit. If someone renames
`retry_request` to `_should_retry` upstream, an unpinned pipeline silently starts
scoring 0.0 on that query and you spend an afternoon debugging the retriever
instead of the label.

So the checkout directory is named `<owner>__<name>@<sha12>`, the SHA is recorded
alongside the eval set, and two different commits of the same repo coexist as two
directories rather than one mutating one.

## Atomic checkout

We clone into a temp directory inside `.repos/`, resolve `git rev-parse HEAD`, and
only then rename into the final `owner__name@sha` path. Rename within a filesystem
is atomic, so an interrupted clone can never leave a half-populated directory that
looks valid to the loader. If the destination already exists, the clone is thrown
away and the existing checkout reused — `fetch` is idempotent.

## Input validation

`--repo` is a trust boundary: it reaches `git` as an argument. Git accepts far more
than https URLs — local paths, `ssh://`, and `ext::<command>`, which executes a
shell command. So `parse_repo_url` allows only `https://github.com/<owner>/<name>`,
with owner and name matched against a conservative character class, and rejects
everything else before any subprocess starts. `--ref` is validated the same way.
Git is invoked with an argument list and never through a shell.

## Alternatives considered

- **GitHub tarball API** (`/archive/<sha>.tar.gz`) — smaller and no git dependency,
  but it needs a SHA up front and rate-limits unauthenticated.
- **Full clone then `git checkout <sha>`** — necessary if we ever index history or
  compute churn-based ranking features. Not today.
- **`git ls-remote` first, clone second** — saves the rename, costs a round trip.
  Worth it only if we start caching by SHA before downloading.
