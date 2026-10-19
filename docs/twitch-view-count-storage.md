# Twitch live viewer-count storage

`database.mysql.separate_twitch_view_counts` is opt-in and defaults to false.
It applies only to items labeled `platform=twitch`, `externalKind=Live` with a
nonnegative integral `viewCount` representable as int64. Other labels, numeric
forms and providers retain their existing representation and behavior.

## Storage and contracts

When enabled, MySQL initialization creates `<prefix>item_view_counts` with a
primary/foreign item ID, a BIGINT count, and a nullable 32-byte projection hash.
The foreign key cascades item deletion. No SQL Server schema or public item
DTO changes are required. Initialization does not rewrite existing items.
Touched items convert lazily in the same transaction as their next full write.
Untouched legacy rows continue to read correctly.

Every metadata read (single, batch, latest, paginated, search and training
stream) reconstructs the numeric label in one SQL statement. A count still
present in legacy labels takes precedence. Missing count rows are repaired by
the next complete source projection; missing items are recreated normally.
The reader does not substitute zero for missing source data.

The optional single-item `POST /api/items?twitch_live_projection=true` intent allows an
unchanged source projection to update just the small count row. Its fingerprint
includes normalized timestamp, hidden state, categories, comment and all other
labels, including the embedding. Independent label patches do not change that
source fingerprint, so a later count-only projection cannot overwrite their
embedding or unrelated keys. A changed source fingerprint forces a full write,
including immediate lifecycle, visibility and safety transitions. Standard
POSTs retain their complete replacement semantics. Old servers ignore the
optional query parameter and retain the existing full-write behavior.

Writers lock the item before its count; multi-item batches deduplicate with
first occurrence winning, then lock by item ID. Full metadata/count updates
commit or roll back together. The labels PATCH endpoint merges against the
current locked item and validates the actual merged size and vectors before
commit. This prevents an embedding patch from restoring a stale count. A
failed write reaches the application as an error; VideoHub's pending retry,
application lock and newer-observation conditional acknowledgement are retained.

The projection handler shares the existing per-item HTTP mutex with label
patches through cache/vector refresh. After commit it re-reads and indexes the
stored representation, so a preserved embedding is not replaced by the old
incoming vector in the recommendation index. Projection cache metadata is
updated after successful persistence. Required post-commit read/cache/index work
uses a detached, bounded 10-second context so client cancellation cannot skip
the refresh immediately after a successful commit. All application writes must continue to
use the configured common Gorse master endpoint; this does not introduce a
new cross-server vector-cache ordering protocol.

The feature rejects READ-UNCOMMITTED sessions. Configure the application's
existing MySQL connection isolation option to READ-COMMITTED or stronger so a
read cannot observe half a transaction. This changes no server-global MySQL
setting. Search configurations indexing all `item.Labels`, or specifically
`item.Labels.viewCount`, are rejected rather than silently changing text-search
semantics. Explicit stable search columns remain supported.

## FM and reader audit

`GorseService` legacy readers continue to receive the same numeric JSON label;
the SQL API and stream paths restore it before returning an item. Tests verify
exact before/after SQL item, FM feature and embedding conversion parity, and
also check that a UseNumber JSON consumer gets the `viewCount.` FM feature.

The existing GORM JSON serializer returns float64 scalars, while the current
FM converter's scalar branch accepts json.Number. This pre-existing training
representation mismatch is not changed here. The optimization neither removes
the label nor silently changes model feature conversion or ranking weights.
It would be incorrect to claim this PR fixes that separate modeling issue.

## Decision counters

The split writer reports two Prometheus counters on the default registry,
`gorse_data_twitch_view_count_decisions_total{branch}` and
`gorse_data_twitch_view_count_commits_total{branch}`. The decision counter
moves when a branch is chosen inside the transaction; the commit counter moves
only after that transaction returned successfully, so a rolled-back write can
never look like a saving. The label set is fixed and carries no item IDs,
values or hashes. With the feature disabled neither counter moves.

Exactly one branch is chosen per item write, in this priority order:

| branch                | meaning                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `unsplittable`        | other provider, VOD, non-object labels or invalid `viewCount`; full write |
| `missing_item`        | no item row existed; insert plus count row                              |
| `legacy_conversion`   | stored labels still held `viewCount`; first-touch full write            |
| `missing_count`       | converted item without a count row; healing full write                  |
| `unmarked`            | Twitch live item without the projection marker; full replacement        |
| `short_path`          | marked projection, fingerprint unchanged; only the count row written    |
| `full_changed_source` | marked projection, fingerprint changed; full write                      |

Steady-state eligibility is `short_path` divided by `short_path +
full_changed_source` over committed writes, after `legacy_conversion` and
`missing_*` have settled. Independent label patches keep the source
fingerprint, so they do not push a projection out of `short_path`.

## Activation and mixed versions

Preparation and review do not authorize production activation.

1. Keep the feature disabled while upgrading all Gorse components to the new
   storage-aware binaries. VideoHub's projection marker is safe against either
   version. Default-off binaries retain the existing storage layout.
2. Validate search columns and the application connection isolation setting in
   isolated development. Update backup/export/restore checks to include
   `item_view_counts` in addition to items, users and feedback, and preserve a
   verified backup covering all data tables.
3. For separately approved activation, stop Gorse writers/readers, configure all
   components consistently (restart connections; do not rely on hot reload), and start the new master to create/verify the small
   table before starting workers and servers. Do not leave old direct-MySQL
   readers active once counts begin moving out of labels.
4. Resume consumers and verify a real live count, hidden/offline transitions,
   pending repair and training/reader parity. Allow first-touch conversion to
   settle before measuring steady-state write volume.

A rolling mixture of old direct-MySQL readers and enabled split writers is
unsupported: old readers cannot reconstruct a moved label. Use the coordinated
cutover above. This is a storage-layout compatibility boundary, even though
HTTP DTOs remain compatible.

For separately approved rollback, stop writers first and copy every hot value
back into labels in a transaction, using the actual verified table prefix:

```sql
START TRANSACTION;
UPDATE items AS i JOIN item_view_counts AS c ON i.item_id = c.item_id
SET i.labels = JSON_SET(i.labels, '$.viewCount', c.view_count);
COMMIT;
```

Verify row coverage and numeric values, then disable the feature on all
components before resuming old binaries. Preserve the small table and backup;
do not drop data during rollback. Re-enabling is tested: legacy count labels
force normalization even if an old projection fingerprint still exists.

## Rollout gates

Compatible readers, isolation/search validation, a restored complete backup and
the rollback rehearsal are prerequisites to a separately approved bounded
production trial. Actual eligibility and paired write measurements are then
acceptance gates during that trial, before confirming the release. Passing
isolated tests does not establish these production results.

1. Every master, server and worker that reads items directly from MySQL runs
   a storage-aware binary; no old direct-MySQL reader remains.
2. The application connection isolation setting and the search column
   configuration are validated against the enabled layout in isolation.
3. A complete backup of all data tables, including `item_view_counts`, exists
   and has been restored into an isolated database where every reader, the FM
   feature conversion and the embedding conversion were checked. The opt-in
   `TestMySQLTwitchViewCountsBackupRestore` is the rehearsal of that procedure.
4. First-touch conversion (`legacy_conversion`, `missing_item`,
   `missing_count`) is excluded from steady-state measurement windows.
5. The committed decision counters show the actual `short_path` share of the
   production projection stream; no estimate from probes substitutes for them.
6. A bounded, paired measurement of logical binlog bytes and physical device
   writes is taken over identical windows before and after activation.
7. The rollback procedure above copies hot counts back under quiesced writers
   and preserves both the count table and the backup; it has been rehearsed on
   the isolated restore from gate 3.

## Verification tools

The backup regression skips unless `TWITCH_COUNT_BACKUP_MYSQL_URI` is set. The
URI must point at loopback and name `gorse_counts_test`; the tool's server UUID
must match the Go connection before fixture creation. The test dumps only
its own prefixed tables with `mysqldump`, restores them into a uniquely named
`vc_backup_*` database it creates and drops itself, and never prints
credentials or dump contents. Set `TWITCH_COUNT_BACKUP_TOOL_CONTAINER` to a
`vh-count-tests-*` container (optionally with
`TWITCH_COUNT_BACKUP_DOCKER_CONTEXT`) to run the client tools inside a local
disposable container instead of from the host. Any other container name or a
non-loopback host is rejected before any tool runs. The focused CI workflow
runs it against its isolated MySQL 8.0.32 service and fails if it is skipped.

The write experiment needs `TWITCH_COUNT_BENCH_OUTPUT`,
`TWITCH_COUNT_BENCH_CONTAINER` (a `vh-count-tests-*` container) and optionally
`TWITCH_COUNT_BENCH_DOCKER_CONTEXT`. The focused workflow runs it for pull
requests and manual dispatch, starting and removing its own named container. Each
arm's receipt now includes the committed branch counters; a split arm is only
accepted when all 1,000 writes took `short_path`.

## Verification and limits

Real MySQL tests cover fresh creation, populated legacy upgrade, every item
reader, full replacement versus projection intent, concurrent count/embedding
patches, failed-write rollback/retry, legacy patch conversion, null deletion,
missing item/count healing, delete cascade and rollback/re-enable.

`TestMySQLTwitchViewCountWriteVolume` calls the actual SQLDatabase implementation
for 1,000 updates per arm. Both 8.0.32 and 8.0.46 produced 9,332,000 binlog bytes
for the old layout and 502,000 for the split layout (94.62% reduction); both
emitted zero for identical updates. Local kernel virtual-disk block-write
reductions were approximately 8.5–14% in the repeated changed-count arms, and
redo reductions approximately 83%. More statements/transactions add fixed
overhead, especially for unchanged requests. Do not infer physical savings from
binlog bytes or extrapolate these synthetic results to production EBS.

The experiment is synthetic and excludes HTTP, Redis cache and vector-index
writes. It is not a production load test or proof of application-level EBS
savings, and it does not estimate how many production projections are
short-path eligible. Production still requires the bounded paired
binlog/device-counter observation after a separately approved rollout. The previous production baseline remains 7,821,905 binlog bytes over
365.361022 seconds, 97.8828559% item-update-event bytes; it has no physical-disk
baseline and does not establish caller attribution.
