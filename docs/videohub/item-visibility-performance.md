# Filtered item visibility lookup

The item-neighbor serving paths previously loaded full item records from the data store, including embedding-bearing labels, solely to check current visibility and category membership. `FilterVisibleItemsByCategories` now uses the existing `Categories`, `SkipHidden` and `ReturnId` options and retains the original ordered score list by returned ID membership. Both embedding and tag neighbors benefit, as does the existing item-based recommendation caller.

Fresh database checks remain in place. Vector metadata is not trusted for current visibility/categories. Score values, order, duplicate score entries, candidate selection, source weights, pagination and downstream VideoHub SQL eligibility are unchanged. MongoDB's filtered lookup explicitly retains the previous interpretation of missing/null visibility flags; current writers store booleans. This does not introduce a general malformed-document validation policy.

## Local measurements

Go 1.27.0, Windows/amd64, Intel i7-12700K, `GOMAXPROCS=6`; MySQL 8.0 in a local Docker container accessed through loopback. A fixed pool of six open/idle connections avoids connection churn in the six-source fixture. The dataset contains 360 synthetic items, overlapping source pools of 120 candidates, hidden/category-mismatched items, and either zero or 1,536 embedding dimensions. No production data is used.

The retained baseline helper is the implementation from commit `e1d393e829dbeeb9ae4156bd748332b6cd3c6a78`. Baseline and changed helpers run in the same binary against the same database fixture. Complete ordered score equality is checked before timing. One operation is a single lookup or a batch of six concurrent lookups, as labeled. Runs use `-benchtime=500ms -count=3 -cpu=6 -benchmem`. The embedding/MySQL cases were repeated with method order reversed, producing six measured trials per method; controls have three trials.

| MySQL fixture | Before median | After median | Before allocated bytes/op | After allocated bytes/op |
| --- | ---: | ---: | ---: | ---: |
| 1 lookup, 1,536 dimensions | 84.75 ms | 1.45 ms | 15,379,037 | 176,219 |
| 6 lookups, 1,536 dimensions | 126.32 ms | 2.40 ms | 92,253,037 | 1,054,266 |
| 6 lookups, no embedding | 3.87 ms | 2.32 ms | 1,980,585 | 1,054,373 |

For the six-source embedding case, trials ranged from 120.92–140.85 ms before and 2.29–2.50 ms after. The six baseline trials executed 27 measured batches; the six changed trials executed 1,500 batches. Allocations fell from approximately 2,255,488 to 8,997 per batch. The baseline materialized 720 full item records and 720 labels values per batch; the changed helper materialized 525 matching ID-only records and **zero labels values**, with identical filtered outputs. The omitted labels represent 13,895,280 bytes of serialized synthetic fixture data; this is not a measured network-byte counter.

SQLite controls also improved: the six-source embedding median was 71.83 ms before and 4.24 ms after. See [item-visibility-results.json](item-visibility-results.json) for all 60 accepted benchmark samples, controls, ranges, iteration counts, allocation/materialization measurements and aggregate test results.

An initial harness run used the driver's default idle pool and hit Windows socket exhaustion during one repeated MySQL sample. That run was rejected in full. Final runs explicitly retain six connections and reject any `FAIL` in the output; both completed without failed samples. The application pool configuration is unchanged.

These results isolate a visibility-filter helper. They exclude vector search/RPC, API source queueing, SQL Server hydration, ranking, CORS, rendering and production contention. The production capture measured a 708.66 ms source stage and 226.39 ms API hydration stage within a 1,148.41 ms debug pipeline; it did not isolate the Gorse item-read substage. Do not substitute the local improvement percentage for production endpoint latency or claim that the 1–3 second report is resolved before rollout and comparable observation.

## Verification

`go test ./server ./storage ./storage/data ./logics -count=1 -json` passed all four packages: 181 named test/subtest pass events. Fixtures exercised real isolated MySQL 8.0 and MongoDB 6, SQLite and the gRPC database proxy. New cases cover all-category matching, case/Unicode/escaped values, empty filters, hidden/missing items, stale vector categories, changes without vector refresh, duplicate scores, order, projection, cancelled contexts and pagination through both named neighbor routes. Shared store tests exercise all three options together. The Mongo regression compares legacy false/null/missing flags with full-read behavior.

Six existing tests/subtests were skipped: PostgreSQL and ClickHouse were not configured; proxy Init/Purge/Timezone and MySQL Timezone explicitly skip. No coverage is claimed for those skipped cases. The unchanged Veya .NET test suite was not rerun for this separate Go change.

Independent Astra review found no actionable issues after the Mongo compatibility adjustment. Server, master and worker builds passed for native Windows/amd64 and Linux/amd64 with CGO disabled.

To reproduce on an isolated local fixture, set `GORSE_VISIBILITY_MYSQL_URI` to a loopback database whose name begins with `gorse_visibility`; the helper rejects other destinations. For aggregate MySQL/Mongo tests, set their normal test URI variables **only to disposable local databases**: the repository test suites recreate test databases. Never use production test URIs.

```text
go test ./server ./storage ./storage/data ./logics -count=1
go test ./server -run ^$ -bench ^BenchmarkItemVisibility$ -benchmem -benchtime=500ms -count=3 -cpu=6
# Repeat the MySQL embedding cases with GORSE_VISIBILITY_BENCH_ORDER=filtered-first:
go test ./server -run ^$ -bench ^BenchmarkItemVisibility$/mysql/dimensions=1536/ -benchmem -benchtime=500ms -count=3 -cpu=6
go build ./cmd/gorse-server ./cmd/gorse-master ./cmd/gorse-worker
```

## Dependencies and rollout

This is a change to `Programmer-Technologies-Corporation/gorse`, based on `e1d393e829dbeeb9ae4156bd748332b6cd3c6a78`. VideoHub must pin the resulting Gorse commit in `backend/vendor/gorse` before its normal image build can include the fix. It is independent of the earlier .NET candidate `1da217db05f347076de00a231cb29d68a2f211ec`; that candidate is preserved separately.

The production serving binary is in the **Gorse server image**. The VideoHub `gorse-images/Dockerfile` builds component images from the pinned `gorsesrc` context; follow the established image-release process with the new commit identity. Consistent component image tags can be used for the release, but no protocol, schema, vector index, model, source-count, timeout or cache-TTL migration is required. The .NET API alone cannot activate this fix.

After separately authorized publication/deployment, compare normally paced authenticated homepage requests and a small diagnostic sample, checking stage timings plus counts, filtering, freshness and diversity. No images, commits or pull requests were published and no production resources were changed as part of this local implementation.
