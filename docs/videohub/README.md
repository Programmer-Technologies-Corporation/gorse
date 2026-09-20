# VideoHub fork of Gorse

This branch (`videohub`) carries the changes VideoHub needs on top of upstream
`gorse-io/gorse` `master`. The goal is to stay rebaseable: features live in new
files, hooks into upstream code are a few lines each, and every behavioural
change is opt-in through the `[videohub]` config section. Panic boundaries and
metrics are always on because they only add safety and observability.

## Branch layout and rebasing

| Branch     | Purpose                                                     |
|------------|-------------------------------------------------------------|
| `master`   | mirror of upstream `master`, never carries VideoHub commits |
| `videohub` | upstream `master` + the commits described here              |

To pick up upstream changes:

```bash
git fetch upstream
git checkout master && git merge --ff-only upstream/master && git push origin master
git checkout videohub && git rebase master   # or: git merge master
go build ./... && go test ./config/ ./logics/ ./server/ ./worker/ ./master/ ./model/ctr/ ./storage/cache/
```

Files that are entirely ours (no merge conflicts expected):

- `config/config.go` `VideoHubConfig` (plus the `[videohub]` block in `config/config.toml`)
- `logics/item_to_item_incremental.go`
- `server/videohub.go`
- `worker/guard.go`
- `master/guard.go`
- `model/ctr/metrics.go`
- `storage/cache/redis_memory.go`

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
| item-to-item vector collections (`xvec`)   | items × dim × 4 B per embedding recommender   | one vector per item, reported by `vector_store_estimated_bytes` |
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
