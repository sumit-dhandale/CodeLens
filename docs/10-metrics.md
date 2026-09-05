# 10 — Metrics: measuring retrieval instead of eyeballing it

Until now every judgement in this project was "those results look reasonable".
That does not survive contact with a change: swap the chunking strategy and you
cannot tell whether it helped. `src/eval/metrics.py` replaces the eyeball.

Every metric takes the same input — the gold grade of each retrieved item in rank
order, `0` for unlabelled — so nothing in `metrics.py` knows what a chunk is.
A retriever returning `[irrelevant, primary, secondary]` produces `[0, 2, 1]`.

## The five numbers

| Metric | Question it answers | Blind spot |
| --- | --- | --- |
| **Hit@1** | Is the top result right? | Ignores everything below rank 1. |
| **Hit@5** | Is *anything* right in the top 5? | Binary: rank 1 and rank 5 score the same. |
| **Recall@5** | What fraction of known answers did we find? | Needs multiple gold items to mean anything. |
| **MRR** | How high is the first right answer? | Ignores the second one entirely. |
| **nDCG@5** | Right answers, right order, weighted by grade? | Hardest to explain; sensitive to the gain function. |

They disagree, and the disagreement is the information. `hybrid_weighted` at
w=0.5 has the best Hit@5 of any retriever (0.79) while `vector` has the best
nDCG@5 (0.56) — fusion finds an extra answer but ranks the whole list slightly
worse.

## Hit@k, not Recall@k

With one gold item, "Recall@5" and "Hit@5" are the same number, and the naming
matters more than it sounds. Divide by `k` instead of by the number of relevant
items and a perfect retriever scores 0.2 — I have seen that number reported as a
model failure. `recall_at_k` takes `total_relevant` explicitly for this reason,
and `tests/test_metrics.py` pins the case:

```python
assert recall_at_k([2, 0, 0, 0, 0], 5, total_relevant=1) == 1.0
```

## nDCG and the two decisions inside it

DCG sums each hit's gain, discounted by log of its rank. Two choices that people
usually inherit without noticing:

- **Gain.** `2^g - 1` (exponential) rather than `g` (linear), so a grade-2
  primary is worth 3x a grade-1 secondary rather than 2x. The gold set grades
  primary vs acceptable; exponential gain is what makes the distinction bite.
- **The ideal denominator.** Built from *all* gold grades for the query, not from
  what was retrieved. A retriever that finds the primary and misses the secondary
  must not score 1.0 — otherwise nDCG rewards the retriever for the gold set's
  generosity.

The log discount is the interesting part: rank 1 → 2 costs 37% of the gain,
rank 9 → 10 costs 3%. That matches how people read a result list, and it is why
nDCG separates retrievers that Hit@5 calls identical. All three of `vector`,
`hybrid_weighted` and `hybrid_rrf` score 0.75 on Hit@5; their nDCG@5 is 0.56,
0.54 and 0.45.

## Abstention is measured separately, not averaged in

Four gold queries have no answer in the corpus. They are excluded from the
ranking metrics — averaging them in would reward a retriever for returning
nothing and punish it for returning something when there is no right answer —
and reported as `falsecf`: the fraction of unanswerable queries answered with a
top score above a threshold.

That column is only interpretable within one score scale, and the table proves
it:

| retriever | falsecf @ 0.30 | why |
| --- | --- | --- |
| `vector` | 0.25 | raw cosine; 1 of 4 unanswerable queries cleared 0.30 |
| `bm25` | 1.00 | BM25 is unbounded, so *every* score clears 0.30 |
| `hybrid_weighted` | 1.00 | per-query min-max makes the top hit exactly 1.0, always |
| `hybrid_rrf` | 0.00 | RRF's top score is ~1/61, so nothing ever clears 0.30 |

The `hybrid_weighted` row is the real lesson, and it is a design cost, not a
measurement artifact: **normalizing scores per query destroys the absolute
confidence signal.** After min-max, the best result of a hopeless query is 1.0,
identical to the best result of a perfect query. A thresholded "I don't know"
is impossible on top of it. RRF is worse still — it discards scores by
construction. If abstention matters, it has to be decided on the raw cosine
before fusion. Per-query `top_score` values are in the JSON dumps so
Experiment 4 can sweep the threshold rather than trusting 0.30.

## Per-query dumps, because 28 queries have a mean that lies

`eval/results/{strategy}__{retriever}.json` holds every query's hits, grades, and
metrics. The aggregate table says `vector` and `hybrid_rrf` tie on Hit@5 at 0.75.
The dumps say they disagree on ten queries out of 28 and merely net out:

```
q05 code   vec=0 bm25=1 rrf=1   read the incident markdown files off disk
q15 code   vec=0 bm25=1 rrf=1   measure what fraction of the relevant documents we retrieved
q06 code   vec=1 bm25=0 rrf=0   extract the title, date and severity headers from a doc
q17 code   vec=1 bm25=0 rrf=0   compare two chunk sizes and report which retrieves better
```

RRF inherits BM25's two wins and loses two of the vector-only wins. Identical
Hit@5, different failures. Only the per-query view shows that, and it is the
input to docs/11's conclusion.

## A word on 28 queries

A difference of one query is 3.6 percentage points. Several gaps in the
comparison table are one or two queries wide, which is noise, and I will say so
rather than draw a conclusion from it. This is the risk the plan flagged at the
start; the mitigation is per-query deltas now and a second, larger corpus
(`encode/httpx`) before anything is called a result.

## Verify

```bash
python -m src.cli eval
python -m src.cli eval --retrievers vector --strategy class
pytest tests/test_metrics.py
```

## Alternatives considered

- **MAP** (mean average precision). Well-defined and standard, but it needs many
  relevant items per query to differ usefully from MRR; with one or two gold
  items it is nearly MRR with extra steps.
- **Precision@k.** Implemented and tested, but not in the table: with 1–2 gold
  items per query the ceiling is 0.4, so every retriever looks bad in the same
  uninformative way.
- **A single headline metric.** Tempting, and it is how you end up optimizing
  Hit@5 while the top result gets worse.
