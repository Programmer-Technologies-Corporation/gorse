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
updated after successful persistence. All application writes must continue to
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

The experiment excludes HTTP cache/vector-index writes. It is not a production
load test or proof of application-level EBS savings. Production still requires
the bounded paired binlog/device-counter observation after a separately approved
rollout. The previous production baseline remains 7,821,905 binlog bytes over
365.361022 seconds, 97.8828559% item-update-event bytes; it has no physical-disk
baseline and does not establish caller attribution.
