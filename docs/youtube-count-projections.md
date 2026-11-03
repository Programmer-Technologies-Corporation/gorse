# YouTube count projections

The enabled MySQL split-count layout now stores YouTube Video counts in the
existing `item_view_counts` table. All public item readers still merge a numeric
`viewCount` into labels. No table, column, API, retention, row-image or durability
setting is added or changed. With `separate_twitch_view_counts=false`, behavior
is unchanged. Enabling that setting for the first time still requires the
[storage-layout checks](twitch-view-count-storage.md).

The change targets measured repeated YouTube count updates that otherwise log
both full before/after item images, including unchanged embeddings. Ordinary
batches lock items in sorted order, compare cold metadata, write changed items
in bulk and upsert counts in bulk in the same transaction. Identical counts
produce no row changes. First-touch conversion still rewrites the item once.
Full replacement semantics remain intact: the Twitch source-fingerprint intent
never applies to a YouTube item, including in a mixed batch.

JSON comparisons now run inside MySQL against `CAST(? AS JSON)`. MySQL's text
rendering of a stored double can differ from Go's original decimal spelling;
comparing those strings falsely classified identical embeddings as changed.
The query returns comparison booleans instead of the large JSON documents.
This is an exact database comparison, with no floating-point tolerance or
conversion of large integers to float64. Atomic label patches retain their
row lock and merge/validation path and use an equality predicate to avoid
counting identical labels as changed. An independent patch and count update
still serialize on the base item row.

## Measured local result

Compared the exact deployed revision `6dc49b64349a62e3cbe4dfd442ff28d23438a381`
with this candidate, using the same MySQL 8.0.46 fixture: ROW/FULL binlogging,
transaction compression OFF, `sync_binlog=1`, and
`innodb_flush_log_at_trx_commit=1`. Each arm updates 50 synthetic items ten
times, with a 256-dimensional vector and synthetic descriptive labels. Setup
and first-touch conversion are outside the measurement interval.

| Workload (500 item operations) | Before binlog bytes | After binlog bytes |
| --- | ---: | ---: |
| Count changes only | 3,543,770 | 30,060 |
| Identical projections | 0 | 0 |
| Comment and count change together | 3,540,120 | 3,555,280 |

Count-only binlog bytes fell 99.15%; SELECT/INSERT/UPDATE/DELETE statements
fell from 30 to 20 for the ten batches. Real metadata-plus-count changes cost
0.43% more binlog bytes because both representations legitimately change.
Identical requests still execute statements but create no row-event bytes.
The native comparison also avoided ten unnecessary base-row upsert statements
for 500 identical realistic float32-derived embedding projections. Those
upserts already produced zero row events in the old version, so this latter
improvement is SQL work and accounting, not additional binlog savings.

These measurements exclude HTTP/cache/vector work, and do not establish an
EBS write reduction or a production-wide percentage. Distribution varies over
time: live lifecycle transitions, embedding generation and canonical metadata
repair still need real writes. Measure post-conversion production growth over
comparable windows and inspect errors and committed counters before making
capacity claims.

## Tests and reproduction

Set `TWITCH_COUNT_MYSQL_URI` to an isolated disposable MySQL database and run:

```sh
go test ./storage/data ./server ./config
YOUTUBE_COUNT_BENCH_OUTPUT=/tmp/youtube.json go test ./storage/data -run '^TestMySQLYouTubeCountWriteExperiment$' -count=1
PROJECTION_NUMERIC_BENCH_OUTPUT=/tmp/numeric.json go test ./storage/data -run '^TestMySQLNumericProjectionWriteExperiment$' -count=1
```

The experiments use only uniquely prefixed synthetic fixture tables. Tests
cover readers, upgrade/conversion, ordinary and mixed replacement, invalid and
removed counts, failure rollback/retry, concurrent count/embedding patches,
missing-count and vector-index healing, recommendations, and exact JSON
comparisons including integers above 2^53, case and array order.

## Rollback boundary

The deployed `6dc49b64` reader already merges the count table generically when
the option stays enabled. A three-phase isolated probe verified that this
candidate's split YouTube row is readable by that exact revision, its next
ordinary full write restores the inline count and removes the small count row,
and this candidate reads that restored value correctly. Thus rolling back to
that known revision with the option kept enabled needs no copy-back or schema
work. The older feature-disabled/feature-unaware rollback procedure is a
different operation; do not disable the option or use an older incompatible
reader while converted rows remain.

This is a local candidate. Production publication and deployment remain
separately authorized steps. No production binlog payload was copied for this
work, and no logging or retention setting was reduced to obtain these results.
