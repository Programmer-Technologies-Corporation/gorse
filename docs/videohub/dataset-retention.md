# Release raw item labels after feature conversion

`LoadDataFromDatabase` previously retained every complete raw item in a
temporary slice through item loading and explicit-negative feedback loading,
even after converting numeric labels to compact ranking and CTR features.
Only configured non-personalized recommenders need those complete items later.
Sorting, feedback scan boundaries and item matching use only item IDs.

When the supplied non-personalized recommender list is empty, the temporary
slice now retains ID-only items. Original batches still feed every feature
conversion and the existing snapshot byte accounting. A nonempty list keeps
the complete raw-item path, including arbitrary raw-label expressions.
Processed labels that still share raw payloads remain live as needed.

There is no change to feedback selection, embeddings, categories, hidden flags,
freshness, schedules, item-to-item generation, ranking or storage formats.
No configuration change or migration is required. Activating this code requires
replacing the running master binary; an image-only rollback preserves the same
data and configuration.

## Local evidence

Comparison against `39614f450dcfd716b578c6ac9d71c8ac9379fbdd` used separate
compiled test binaries on Windows 11 / i7-12700K, Go 1.27, GOMAXPROCS=2.
There were 48 fresh-process samples: baseline/candidate x 10k/100k items x
empty/nonempty non-personalized configuration x diagnostic/ordinary GC mode x
three repetitions. Variant order alternated. Primary measurements wrote no
profiles; a separate ordinary 100k-item pair collected profiles.

The lazy synthetic stream generates independent decoded-JSON numeric arrays
with 256 bounded finite values per item. It retains no full raw-item fixture
outside the loader. Both modes share a pre-load GC. Diagnostic mode additionally
forces GC at the first feedback read after all item conversions, measuring live
retained heap above the pre-load baseline. Ordinary mode does not force GC during
loading; a 5 ms sampler records the maximum observed Go HeapAlloc.

Medians of three samples per version, with an empty recommender list:

| Measure | Items | Before | After |
| --- | ---: | ---: | ---: |
| Diagnostic retained heap delta | 10,000 | 94.00 MiB | 21.15 MiB |
| Diagnostic retained heap delta | 100,000 | 939.34 MiB | 210.84 MiB |
| Ordinary sampled heap maximum | 10,000 | 191.41 MiB | 191.42 MiB |
| Ordinary sampled heap maximum | 100,000 | 1,179.05 MiB | 767.59 MiB |
| Ordinary loader elapsed | 10,000 | 99.19 ms | 100.10 ms |
| Ordinary loader elapsed | 100,000 | 833.22 ms | 720.44 ms |

At 100k, ordinary sampled maxima ranged 1,141.92-1,186.27 MiB before and
692.50-854.74 MiB after. Total allocation stayed about 1,319.55 MiB per load;
the change shortens ownership rather than avoiding initial decoding. The
nonempty control retained about 939.35 MiB in both versions and its ordinary
heap/timing ranges overlapped. At 10k, GC did not reclaim the released data soon
enough to reduce ordinary sampled maximum heap.

All fixture output digests matched across versions and measurement modes.
These cover loaded ranking features, CTR features/embeddings, feedback,
snapshot counts/bytes and non-personalized scores. They are not an end-to-end
model-training or homepage-ranking benchmark.

These are local synthetic results. Sampled HeapAlloc is neither process RSS nor
an exact transient peak. The paused phase and sampling add instrumentation;
diagnostic elapsed time includes forced GC. Three samples do not establish a
latency distribution. Allocation profiles can lag GC epochs, so direct runtime
counters support the allocation comparison. No production memory, paging or
homepage speedup is established by these results.

## Reproduce and verify

Run the ordinary correctness test:

```sh
go test ./master -run '^TestDatasetRetentionLoader$' -count=1
go test ./master -run '^TestMaster$/(TestEmitSnapshot|TestLoadDataFromDatabase|TestLoadDataFromDatabaseInParallel|TestNonPersonalizedRecommend)$' -count=1
```

For comparable memory samples, copy both `dataset_retention_test.go` and
`dataset_retention_experiment_test.go` unchanged into a clean baseline worktree,
then compile each revision once with `go test -c ./master -o retention.test`.
Run every sample as a fresh process, alternating revision order:

```sh
GOMAXPROCS=2 GORSE_RETENTION_ITEMS=100000 \
GORSE_RETENTION_NONEMPTY=0 GORSE_RETENTION_DIAGNOSTIC=0 \
GORSE_RETENTION_PROFILE= ./retention.test \
  -test.run='^TestDatasetRetentionMemoryExperiment$' -test.v -test.timeout=3m
```

Set NONEMPTY to 1 for the raw-label consumer control and DIAGNOSTIC to 1 for
phase live-heap measurement. Use ITEMS=10000 for the smaller fixture. Profile
only separate samples by setting GORSE_RETENTION_PROFILE to an output prefix.
Do not mix those results into unprofiled timing comparisons. The experiment is
opt-in and skipped by normal test runs. Focused Linux CI runs the new retention
and serving regressions with the race detector, runs existing SQLite loader
tests as functional checks, and builds all three deployed components.

### Existing parallel-loader race limitation

Adding race detection to the existing SQLite suite exposed pre-existing races
in shared feedback counts/frequencies, progress counters, callback errors and
test teardown versus asynchronous reconciliation. The same mechanisms reproduced
on unchanged base `39614f4` in
[the baseline diagnostic run](https://github.com/Programmer-Technologies-Corporation/gorse/actions/runs/37690555781).
The retention edit is confined to single-threaded item loading before these
parallel phases; it does not repair their synchronization. Frequency races can
affect training weights and are not merely diagnostic noise. They need a
separate concurrency fix. The legacy suite's functional pass must not be
described as a race-clean parallel loader; the focused race gate remains enabled.
