# VideoHub fork of Gorse

This branch (`videohub`) carries the changes VideoHub needs on top of upstream
`gorse-io/gorse` `master`. The goal is to stay rebaseable: features live in new
files, hooks into upstream code are a few lines each, and every behavioural
change is opt-in through the `[videohub]` config section. Panic boundaries and
metrics are always on because they only add safety and observability.

## Branch layout and syncing with upstream

| Branch     | Purpose                                                     |
|------------|-------------------------------------------------------------|
| `master`   | mirror of upstream `master`, never carries VideoHub commits |
| `videohub` | upstream commits + the commits described here               |

The repository's ruleset requires linear history and forbids force pushes on
every branch, so `videohub` can neither merge upstream nor be rebased onto it.
Upstream commits are cherry-picked instead, with `-x` so each one names the
upstream commit it came from, on a topic branch that is merged into `videohub`
with **Rebase and merge** (a squash would fold upstream's commits into one).

Last upstream commit picked: `bc8ca79` (test: stabilize neural network training, #1386).

```bash
git fetch upstream
git checkout master && git merge --ff-only upstream/master && git push origin master
git checkout -b sync-upstream videohub
git cherry-pick -x <last upstream commit picked>..upstream/master
go build ./... && go test ./config/ ./logics/ ./server/ ./worker/ ./master/ ./model/ctr/ ./storage/... ./common/floats/
# update "Last upstream commit picked" above, push, open a PR into videohub
```

Files that are entirely ours (no merge conflicts expected):

- `config/config.go` `VideoHubConfig` (plus the `[videohub]` block in `config/config.toml`)
- `logics/item_to_item_incremental.go`
- `server/videohub.go`
- `worker/guard.go`
- `master/guard.go`
- `model/ctr/metrics.go`
- `storage/cache/redis_memory.go`
- `storage/scheme_videohub.go`, `storage/vectors/hnsw_*.go` (the `hnsw://` vector store)
- `common/floats/floats_videohub.go` (allocation-free FP16 helpers)

Small hooks into upstream files (each a few lines, marked with a
"VideoHub fork" comment where the intent is not obvious):

- `server/rest.go`: `QueryLimitFilter` in the filter chain, `createVideoHubRoutes`,
  `validateItemEmbeddings` / `indexItemVectors` in `batchInsertItems` and
  `modifyItem`, `queryItemToItem` in `SearchItemToItem`.
- `worker/pipeline.go`: per-user `recoverUserJob`, outcome counters, `safeBatchPredict`
  and the degrade-to-unranked fallback when ranking fails.
- `worker/worker.go`: model id gauges when a model is pulled.
- `master/tasks.go`: every task step wrapped in `runTask`, item-to-item stats,
  malformed embedding count, cache document counts in `collectGarbage`, model id gauges.
- `master/master.go`: model id gauges when meta is loaded at startup.
- `model/ctr/fm.go`, `fm_xla.go`: skipped-embedding counters.
- `logics/vector_writer.go`: `VectorWriterStats`.
- `logics/item_to_item.go`: missing collection means "no neighbors"; no error log per item without labels or an embedding.
- `logics/vector_writer.go`: `appendSparseVector` drops repeated ids (a user with several feedback types on one item made the vector store reject the user-to-user vector with "duplicate coordinate").
- `storage/cache/redis.go`: the document scan skips hashes deleted while scanning instead of failing garbage collection.
- `config/config.go`: `hnsw://` accepted by the `vector_store` validator.
- `logics/vector_writer.go` (`Clean`) and `master/tasks.go` (collaborative filtering index): call `Optimize` once the collection is written.
- `storage/vectors/xvec.go`: `xvec.NewCollectionOptions()` instead of the zero options (mmap), query results without vectors.
- `server/server.go`, `worker/worker.go`: `storage.IsEmbeddedVectorStore` decides whether the vector store is reached through the master (it was a check for the `xvec://` prefix).

## Configuration

```toml
[videohub]
# Index embedding-based item-to-item vectors on item insert/patch and answer
# neighbor queries from the stored embedding for items that are not indexed yet.
incremental_item_to_item = false
# PATCH /api/item/{item-id}/labels merges label keys (null deletes a key).
label_patch = false
# Reject item writes whose embedding column length differs (0 = off).
embedding_dimensions = 0
# Cap the "n" query parameter of every endpoint (0 = off).
max_query_n = 0
```

Environment overrides: `GORSE_VIDEOHUB_INCREMENTAL_ITEM_TO_ITEM`,
`GORSE_VIDEOHUB_LABEL_PATCH`, `GORSE_VIDEOHUB_EMBEDDING_DIMENSIONS`,
`GORSE_VIDEOHUB_MAX_QUERY_N`.

## 1. AFM/CTR tensor panics

Upstream status on `master` (all included in this branch):

- `#1331` ignores embeddings whose name was not seen during training.
- `#1347` sizes the feature tensors per batch instead of using the training-time
  `numDimension`, which is the actual fix for the "feature vector longer than
  training" index-out-of-range panic that killed workers on 0.5.11. The `v0.5.11`
  tag does **not** contain it; `release-0.5` and `master` do.
- `#1336` fixes a panic in the user/item feedback endpoint.

What the fork adds, because upstream still has no boundary between the model and
the process:

- **Worker**: every per-user recommendation job recovers panics
  (`gorse_worker_recommend_panics_total`) and continues with the next user. The
  ranking call itself runs in `safeBatchPredict`; a panic or a wrong output size
  is logged with the user id and candidate count, counted in
  `gorse_worker_ranking_failures_total{reason}`, and the user is served the
  unranked candidate order instead of nothing.
- **Master**: every task step (`load_dataset`, `item_to_item`,
  `train_click_through_rate`, ...) runs through `runTask`, so a training panic
  fails that step (`gorse_master_task_panics_total{task}`,
  `gorse_master_task_failures_total{task}`) and the task loop stays alive. Before
  this, `RunTasksLoop` recovered the panic once and then silently stopped
  scheduling tasks.
- **Boundary validation**: `[videohub].embedding_dimensions` rejects item writes
  (`POST /api/item(s)`, `PATCH /api/item/{id}`, the label patch) whose embedding
  has an unexpected length with `400` and `gorse_server_rejected_embeddings_total`.
  Malformed vectors that still reach the models are counted instead of silently
  zero-filled: `gorse_ctr_skipped_embeddings_total{reason}` at inference and
  `gorse_master_ctr_dataset_malformed_embeddings` when the training set is built.

## 2. Redis artifact lifecycle (audit)

On 0.5.11 every item-to-item and user-to-user neighbor list was materialised in
the cache store as one Redis hash **per neighbor** (`documents:item-to-item:<name>/<item>:<neighbor>`,
seven fields each, indexed by RediSearch), written by the master job for every
item and recommender. With `cache_size = 30`, two item-to-item recommenders and
60k items that is 3.6M hashes plus the search index, which is the memory
consumer that was observed. Those hashes had no TTL; they were only reclaimed by
the garbage collection scan when an item disappeared from the dataset.

Upstream `master` (this branch) no longer writes neighbors to the cache store at
all. Item-to-item and user-to-user similarity lives in the vector store (one
vector per item per recommender, `xvec` by default) and neighbors are computed
per request from the vector index. What remains in Redis:

| Collection / key                        | Bound                                            | Retention                                   |
|-----------------------------------------|--------------------------------------------------|---------------------------------------------|
| `recommend/<user>` documents            | `recommend.cache_size` × users with fresh recs   | replaced per refresh, stale entries deleted |
| `non-personalized/<name>` documents     | `recommend.cache_size` per recommender           | replaced per master cycle                   |
| `collaborative-filtering/<user>`        | `recommend.cache_size` × users                   | GC when the user leaves the dataset         |
| `last_modify_item_time/<item>` etc.     | one small key per item / user                    | never deleted                               |
| time series points                      | one point per metric per cycle                   | never deleted (upstream behaviour)          |

**Superseded generations.** A cache store that served 0.5.x still holds those
neighbor hashes (a VideoHub dev instance had ~28k of them) and upstream `master`
never reclaims them. The fork's garbage collection deletes every document in the
`item-to-item` and `user-to-user` collections, which nothing writes any more,
and logs how many it reclaimed.

Telemetry added so this stays visible: `gorse_master_cache_documents_total{collection}`
(counted during the existing garbage-collection scan, no extra scan) and
`gorse_master_cache_memory_bytes` (Redis `used_memory`). Vector store growth is
reported by `gorse_master_vector_store_vectors_total{collection}` and
`gorse_master_vector_store_estimated_bytes{collection}`.

Bounded neighbor counts are now a query-time property: `n` is capped by
`[videohub].max_query_n` and the per-item vector footprint is fixed by the
embedding dimension.

## 3. Incremental item-to-item availability

`[videohub].incremental_item_to_item = true` makes neighbors available as soon as
an item is written, without waiting for the master job:

- `POST /api/item`, `POST /api/items`, `PATCH /api/item/{id}` (labels, hidden flag
  or categories) and the label patch upsert the item's vector into the
  `item_to_item_<name>` collection of every **embedding**-type recommender. The
  collection is created on first use with the embedding's dimension; a vector
  with a different dimension is skipped and counted
  (`gorse_server_item_vector_index_total{recommender,result}`).
- Neighbor queries (`/api/item-to-item/{name}/{id}`, `/api/item/{id}/neighbors`)
  fall back to the item's stored embedding when nothing is indexed for it, so
  items inserted before the flag was enabled also get neighbors
  (`gorse_server_item_to_item_fallback_total{recommender,result}`). A missing
  collection (no master job has run yet) is treated as empty instead of a 500.
- The periodic master job keeps its role: it re-indexes every item, refines
  hidden/category flags and removes vectors of deleted items. Vectors written
  incrementally carry the write time as timestamp, so the job's cleanup (which
  deletes vectors older than its dataset snapshot) never removes them.
- A recommender whose collection has not been built yet (no item had a vector
  when the master job ran) now yields no neighbors instead of an error. Upstream
  failed the whole offline recommendation of every user on that error, so a
  catalog without embeddings received no recommendations at all.
- The master job no longer logs an error for every item that lacks the
  embedding column; the count is exposed as `gorse_master_item_to_item_items_without_vector`.
- Tags/users/auto recommenders are not indexed incrementally because their
  sparse vectors depend on dataset-wide IDF weights only the master job knows.
- Vector writes from server nodes reach the master's `xvec` store through the
  existing gRPC vector proxy; no additional service is needed.

## 4. Embedding mutation

`[videohub].label_patch = true` enables `PATCH /api/item/{item-id}/labels` with a
JSON object body. Top-level keys are merged into the existing label object,
`null` deletes a key, everything else is kept. This lets a client replace
`embedding` without resending title, metadata or categories. The endpoint refreshes
the item's last-modify time and, with incremental item-to-item enabled, re-indexes
the vector. Items whose labels are a legacy string array cannot be merged (400).
Concurrent patches of the same item are serialised with a striped lock inside the
node; `gorse_server_label_patch_total{result}` counts outcomes.

## 5. Instrumentation

All metrics are exposed on the existing Prometheus endpoints (`/metrics` on the
master, worker and server HTTP ports).

| Blind spot                     | Metric                                                                                         |
|--------------------------------|------------------------------------------------------------------------------------------------|
| task duration / age / failures | `gorse_master_task_duration_seconds{task}`, `gorse_master_task_last_success_timestamp_seconds{task}`, `gorse_master_task_failures_total{task}`, `gorse_master_task_panics_total{task}` |
| generated cache keys           | `gorse_master_cache_documents_total{collection}`                                               |
| cache bytes                    | `gorse_master_cache_memory_bytes`                                                              |
| vector store size              | `gorse_master_vector_store_vectors_total{collection}`, `gorse_master_vector_store_estimated_bytes{collection}` |
| items without embeddings       | `gorse_master_item_to_item_items_without_vector{recommender}`, `gorse_master_item_to_item_invalid_vectors{recommender}`, `gorse_master_item_to_item_vectors_total{recommender}` |
| model generation id            | `gorse_master_model_id{model}`, `gorse_worker_model_id{model}`                                 |
| neighbor generation age        | `gorse_master_item_to_item_last_update_timestamp_seconds{recommender}` (age = `time() - value`) |
| recommendation backlog         | `gorse_worker_recommend_outcomes_total{outcome}` (`updated`, `skipped_up_to_date`, `skipped_cold_start`, `skipped_no_cf_embedding`, `failed`, `panic`) |
| training failures              | `gorse_master_task_failures_total{task="train_click_through_rate"|"train_collaborative_filtering"}` |
| skipped malformed samples      | `gorse_ctr_skipped_embeddings_total{reason}`, `gorse_master_ctr_dataset_malformed_embeddings`, `gorse_server_rejected_embeddings_total{recommender}` |
| ranking failures               | `gorse_worker_ranking_failures_total{reason}`, `gorse_worker_recommend_panics_total`           |
| incremental indexing           | `gorse_server_item_vector_index_total{recommender,result}`, `gorse_server_item_to_item_fallback_total{recommender,result}` |
| label patches                  | `gorse_server_label_patch_total{result}`                                                       |
| clamped queries                | `gorse_server_query_n_clamped_total`                                                           |

## 6. Bounded-memory operation

Structures that scale with the catalog, with their bounds on this branch:

| Structure                                  | Size driver                                   | Bound / control                                              |
|--------------------------------------------|-----------------------------------------------|--------------------------------------------------------------|
| master dataset (items, labels, feedback)   | items × labels, positive feedback             | `recommend.data_source.item_ttl`, `positive_feedback_ttl`    |
| CTR training set embeddings (FP16)         | items × dim × 2 B                             | one embedding column, `embedding_dimensions`                 |
| item-to-item vector collections            | `hnsw://`: items × (2 × dim + 4 × M0 + 120) B, see section 7; `xvec://` holds several copies on the heap | one vector per item, `gorse_vector_index_memory_bytes` |
| collaborative filtering collections        | items × factors × 4 B, two generations kept   | upstream keeps the two newest complete models                |
| Redis documents                            | see section 2                                 | `recommend.cache_size`, `cache_documents_total`              |
| worker item cache                          | candidates touched in one cycle               | freed after each cycle; embeddings compressed to FP16        |
| API responses                              | `n` per request                               | `[videohub].max_query_n`                                     |

Operational guidance:

- Give every Gorse container an explicit memory limit **and** `GOMEMLIMIT`
  (about 85–90 % of the limit). The Go runtime then collects garbage aggressively
  before the kernel OOM-kills the process, which turns "worker restarts" into a
  visible `go_memstats` plateau.
- Give Redis a `maxmemory`; the VideoHub compose file uses `allkeys-lru` so the
  cache degrades gracefully instead of failing writes. Watch
  `gorse_master_cache_memory_bytes` against that limit.
- Item quotas (`[quota]`) exist upstream and are the hard cap on catalog growth;
  `max_items_count` is the right knob if ingestion should stop rather than
  grow memory.

## 7. Neighbor computation: the `hnsw://` vector store

Upstream `master` already moved item-to-item, user-to-user and collaborative
filtering search out of the in-memory HNSW (`common/ann`, now unused outside
tests) and into the vector store, which for an embedded deployment is `xvec`.
Reading `xvec` at the pinned version showed that it cannot serve Gorse's write
pattern:

- every upsert appends a new row and a tombstone, even for an identical
  document, and nothing is ever compacted because Gorse never calls `Flush` or
  `Optimize`; a collection stops accepting writes at 10 M retained rows;
- the first query after **any** write rebuilds the whole DiskANN index (Vamana
  graph plus product quantizer training) over every retained row, dead versions
  included, while holding the collection lock. The fork's incremental item
  indexing (section 3) turns every item insert into such a rebuild;
- documents, the decoded snapshot and the index all live on the Go heap, and a
  filtered query allocates per row before the graph search starts.

Measured with `GORSE_VECTOR_BENCH=100000 go test ./storage/vectors -run TestVectorStoreCycle -v`
(100 k vectors, 256 dimensions, one core, Windows/amd64):

| Step                              | `xvec://`            | `hnsw://`          |
|-----------------------------------|----------------------|--------------------|
| first cycle (load)                | 2.9 s, 2.8 GB alloc  | 17.7 s, 0.4 GB     |
| first query                       | 7.9 s, 4.4 GB alloc  | < 1 ms             |
| 1000 queries                      | 51.9 s, 118 GB alloc | 0.12 s, 2 MB alloc |
| next cycle, nothing changed       | 7.7 s, 6.8 GB alloc  | 0.09 s             |
| first query after that cycle      | 19.2 s, 8.9 GB alloc | < 1 ms             |
| one item upsert + one query       | 19.4 s, 8.9 GB alloc | 1 ms               |
| live heap after two cycles        | 1.1 GB               | ~90 MB             |

`hnsw://` is a vector store written for that pattern. It is selected by URL, so
`xvec://` keeps working and switching back is a one line change:

```toml
[database]
vector_store = "hnsw:///var/lib/gorse/master/hnsw"
```

What it does:

- **Incremental cycles.** One persistent HNSW graph per dense collection. An
  upsert whose vector is bit-identical only refreshes the timestamp; a change of
  `hidden` or categories is an in-place metadata update; only a changed vector
  is re-linked (new slot, old slot becomes a tombstone that still routes
  searches). `DeleteVectors` tombstones what the cycle did not touch. The cycle
  over an unchanged catalog is a comparison per item.
- **Compaction instead of rebuilds.** When tombstones pass `compact_ratio`
  (default 20 % of the slots, at least 1024) the graph is rebuilt in the
  background from a frozen prefix of the slabs; writes that arrive meanwhile are
  replayed before the swap. Queries and writes keep running. This is the only
  time a whole graph is built after the first load. Changing `m`, `m0` or
  `ef_construction` triggers the same rebuild.
- **FP16 storage.** Gorse hands embeddings over already rounded to FP16, so a
  collection whose first batch survives an FP16 round trip is stored as FP16
  (half the memory, and faster than FP32 here because distance evaluation is
  memory bound). Anything else (collaborative filtering factors) stays FP32.
  `precision=fp16|fp32` overrides the choice.
- **Graph parameters.** `M = 32`, `M0 = 64`, `ef_construction = 200`,
  `ef_search = 100`, with the HNSW neighbor selection heuristic instead of the
  plain "closest M" of `common/ann`. On embedding-like data recall@10 is 1.00 at
  `M = 32` and still 0.999 at `M = 16`; the old index (`M = 48`, no heuristic)
  reaches 0.994 while being 4.7x slower per query. `BenchmarkHNSWQuery` compares
  `M` and precision.
- **No allocation in searches.** A pooled scratch per index holds an
  epoch-marked `[]uint32` visited array, two typed binary heaps and the decode
  buffers. `TestHNSWSearchDoesNotAllocate` pins 0 allocations per search; the
  old `searchLayer` allocated about 107 KB per query.
- **SIMD distances without square roots.** Every metric is one SIMD dot product
  (`floats.Dot`, FP16 decoded by the SIMD `floats.ToFloat32To`): squared
  Euclidean distance is `|a|^2 + |q|^2 - 2a.q` with both norms precomputed. The
  score returned for Euclidean collections stays the negated squared distance,
  the same contract as the `xvec` backend, so `1 / (1 + d^2)` API scores do not
  change.
- **Inner product collections** (collaborative filtering) are indexed through
  the usual reduction to nearest neighbor search (one extra coordinate
  `sqrt(phi^2 - |v|^2)` on stored vectors); queries still rank by plain inner
  product. It lifts recall@10 on a hard synthetic set from 0.64 to 0.85.
- **Filters.** Category sets are interned, a filter is evaluated once per set
  instead of once per vector, and filters that match fewer than `scan_min`
  (4096) vectors or less than `scan_ratio` (5 %) are answered by an exact scan
  of the matches instead of walking the graph past everything they reject.
- **Sparse collections** (tag and feedback similarity) use an exact inverted
  index. Their vectors change every cycle because IDF weights move, so they are
  compacted in place, which is linear.
- **Persistence.** One snapshot file per collection
  (`<root>/<collection>/index.bin`, CRC32, written to a temporary file and
  renamed). Snapshots are taken when a collection changed and has been quiet for
  two seconds, every `snapshot_interval` (1 m), on `Optimize` and on `Close`.
  Timestamp-only changes are not worth a snapshot. The previous snapshot is
  kept as `index.bin.old` until the new one is in place (rename is not atomic
  on Windows) and is loaded when `index.bin` is missing or fails its checksum.
  A crash loses the writes since the last snapshot; the next master cycle
  restores them because it diffs against the store, and the fork's embedding
  fallback (section 3) covers the gap for new items. A collection with no
  readable snapshot is logged, counted and started empty instead of keeping
  the master from starting.
- **Quantization.** `database.vector.quantization_type` is rejected at config
  load for `hnsw://` and `xvec://` (both would fail every indexing task at
  runtime); `precision` is the hnsw knob.

### The `xvec://` fallback

Three fixes keep upstream's store usable as a fallback. They do not make it
competitive, the numbers below are 20 k vectors on xvec `0f0a064` (2026-09-21),
which behaves like the pinned version in this benchmark:

- **`Optimize` after every cycle.** Upstream never calls it, and every remote
  backend implements it as a no-op, so it was clearly meant for xvec. Without
  it the writing segment is never sealed: each query after a write rebuilds the
  index (2.6 s, 1.5 GB allocated) and dead rows pile up until the 10 M row
  limit. With it the index is built once per cycle (2.3 s), superseded rows are
  dropped (live heap 228 MB -> 102 MB) and a single upsert followed by a query
  only rebuilds the small writing segment (82 ms).
- **`xvec.NewCollectionOptions()`.** The zero `CollectionOptions` get the 64 MiB
  buffer from normalization but leave `EnableMmap` false, so sealed index
  artifacts were read into the heap. The flag is persisted in the collection
  manifest: collections created before this change keep mmap off until they are
  recreated. It only matters together with `Optimize`, nothing was ever sealed
  before.
- **No vectors in query results.** No caller reads them; decoding them and
  sending them through the master's gRPC proxy was pure cost.

What remains and cannot be fixed from Gorse: every filtered query (Gorse always
filters on `hidden = false`) forward-scans all rows and allocates about 24 MB at
20 k vectors (8.9 ms per query against 0.07 ms on `hnsw://`), documents stay on
the heap in any case, and `Optimize` is a full rewrite plus index build.

Options (URL query): `m`, `m0`, `ef_construction`, `ef_search`, `precision`,
`snapshot_interval`, `compact_ratio`, `scan_ratio`, `scan_min`.

Metrics (master `/metrics`):

| Metric | Meaning |
|---|---|
| `gorse_vector_index_vectors{collection}` | live vectors |
| `gorse_vector_index_tombstones{collection}` | vectors awaiting compaction |
| `gorse_vector_index_memory_bytes{collection}` | estimated heap of the collection |
| `gorse_vector_index_upserts_total{collection,outcome}` | `inserted`, `replaced`, `metadata`, `unchanged`: the share of `unchanged` is what a cycle saved |
| `gorse_vector_index_deletes_total{collection}` | vectors a cycle no longer produced |
| `gorse_vector_index_compactions_total`, `_compaction_seconds` | graph rebuilds |
| `gorse_vector_index_snapshot_seconds`, `_snapshot_bytes`, `_snapshot_failures_total` | persistence |
| `gorse_vector_index_load_failures_total{collection}` | snapshots rejected at startup |
| `gorse_vector_index_query_seconds{mode}` | latency by `graph`, `scan`, `inverted` |

Sizing: a dense FP16 collection costs about `items x (2 x dim + 4 x M0 + 120)`
bytes, 0.76 KB per item at 256 dimensions, so roughly 76 MB per 100 k items, plus
the same again while a compaction or a snapshot copy is in flight.

Migration from `xvec://`: point `vector_store` at a new directory and restart
the master. The store starts empty; the first item-to-item and user-to-user
cycle fills it and the next collaborative filtering fit recreates its
collection. Until then neighbors come from the embedding fallback. The old
`xvec` directory is not touched and can be removed once the switch has settled.
