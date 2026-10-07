# Unchanged HNSW refresh cost

Incremental item ingestion calls `AddVectors` once per item. An unchanged vector
refreshes its timestamp so cycle cleanup keeps it alive. Previously every such
call also recomputed the collection's memory gauge, scanning all item IDs and
graph upper-level arrays. This made singleton refresh cost grow with index size.

Gauge publication now runs only after an insertion, replacement or metadata
change. Deferred publication covers committed partial changes when a batch is
canceled. Timestamp updates, counters, search, ranking, filtering and persistence
rules retain their existing behavior. No configuration change is needed.

## Local measurements, 2026-10-07

Windows 11, i7-12700K, Go 1.27, GOMAXPROCS=2; synthetic 256-dimensional FP16
embedding fixtures. Baseline `ec30eb2` (tree-identical to `2adec332`). Five
250 ms unchanged-write repetitions per size, initialization excluded:

| Vectors | Baseline median | Changed median | Allocations |
| --- | ---: | ---: | --- |
| 10,000 | 9.837 us/write | 2.018 us/write | ~535 B, ~3/write both |
| 100,000 | 74.096 us/write | 2.597 us/write | ~536 B, ~3/write both |

The separate 100k CPU profile attributed 89.34% of baseline refresh-loop CPU
samples to the full byte estimate; that path was absent after the change.

Concurrent query tests used the same persisted graphs, topK=121, five 3-second
repetitions per scenario, and Windows QueryPerformanceCounter. Ordered IDs and
float32 score hashes matched before/after every run and across revisions.

| 100k query scenario | Samples before / after | Median run p99 before / after |
| --- | ---: | ---: |
| No writes | 127,216 / 128,288 | 200.7 / 200.4 us |
| Low writes (~0.67/s achieved) | 129,777 / 128,784 | 190.3 / 199.6 us |
| Synthetic stress (~99.66/s) | 126,713 / 128,159 | 196.9 / 197.0 us |

These results establish reduced refresh work, **not a repeatable query-tail or
production homepage speedup**. A short production passive sample observed only
10 unchanged updates across five collections in 34.09 seconds. GOMAXPROCS=2
does not make the local CPU, memory residency or storage equivalent to EC2.
The query generator was closed-loop with one reader; it is not an endpoint load
test. Variants ran sequentially. Initial coarse-clock query samples were rejected.

## Verification and reproduction

Dense/sparse tests verify timestamps survive cleanup, ordered results and scores,
counter/gauge consistency, real replacements/metadata changes, and cancellation
after a committed partial insertion followed by an unchanged refresh. Existing
HNSW tests cover persistence, compaction, precision, filtering and recall.

The included microbenchmark provides an aggregate-time reproduction:

```sh
GOMAXPROCS=2 go test ./storage/vectors -run '^$' \
  -bench '^BenchmarkHNSWUnchangedRefresh$' -benchtime=250ms -count=5 -benchmem
```

Run the identical benchmark source on the baseline and candidate revisions.
Initialization builds each fixture outside the measured interval. Local detailed
query samples, high-resolution harness and CPU profiles are retained in the
engineering handoff; the microbenchmark above does not reproduce query tails.

Deployment requires the serving Gorse component to use a rebuilt image. This
does not justify restarting a pressured production master solely to address
second-scale homepage delays. No snapshot/schema migration or resource-setting
change is required; a previous-image rollback remains compatible.
