# Opt-in lossless HNSW snapshot compression

This candidate is based on deployed Gorse commit
`044fa37d6f17eac7ae6fcb8ee74c8c9e24b3e5a5`. It changes snapshot encoding only.
The default remains the original, byte-identical v1 writer:

```text
hnsw:///var/lib/gorse/master/hnsw?snapshot_compression=none
```

An explicitly enabled writer uses streaming zstd:

```text
hnsw:///var/lib/gorse/master/hnsw?snapshot_compression=zstd
```

There is no change to vector precision, queries, snapshot interval, two-second
quiet requirement, dirty-version tracking, `Optimize`, or shutdown flushing.
Timestamp-only refreshes still do not dirty a collection. The existing scheduler
can defer snapshots during continuous writes; its one-minute interval is not
a newly established hard recovery bound.

## Format and limits

V2 is a 16-byte envelope: the existing four-byte magic, little-endian version
`2`, and an eight-byte decoded length. Its zstd stream contains exactly the v1
snapshot, including the original CRC32 trailer. Both the inner checksum and the
complete zstd frame are verified, along with exact decoded length. Readers accept
v1 and v2 independently of the configured writer format.

The encoder has one worker and a 1 MiB window. The streaming decoder has one
worker, a 1 MiB window limit, and an 8 MiB codec memory limit. Compressed snapshots
have a 1 GiB decoded-byte cap. A bounded streaming validation pass establishes
actual decoded length, CRC32 and frame integrity before collection allocations;
the decoder is then reset for reconstruction. No decoded copy is buffered or
written to disk. Explicit allocation accounting is capped at the smaller of
1 GiB and four times verified decoded bytes, including allowances for metadata,
maps, and slice headers.
These limits are independent of compressed physical file size; they are not a
guarantee about total process RSS. V1 retains its existing reader limits.
The writer checks decoded size and the reader's allocation charges before
encoding. A valid collection outside either v2 limit automatically uses the
original v1 writer even when compression is enabled, preserving checkpoint
capacity and freshness. Boundary tests inject smaller limits without creating
gigabyte fixtures. V1 fallback requires no change to precision or cadence.

Serialization remains bounded streaming, with no whole decoded snapshot buffer.
The temporary file is closed and removed on failure; the current file is rotated
only after encoding, checksum, file sync and close succeed. The existing current
and previous filename rotation and fallback remain in use. This does not add
directory fsync or a new filesystem durability guarantee.

## Rollout and old-binary rollback

Compression is disabled by default. Source/CI review and local synthetic tests
do not authorize changing the preview URL or reading live snapshots.

Before enabling compression, verify that every process which opens this vector
directory has the dual-format reader and preserve the existing recovery baseline.
Old binaries cannot read v2. After two compressed snapshots, neither recovery
file may be readable by an old binary.

To downgrade without weakening recovery:

1. Keep the new binary installed and coordinate writers as for the existing
   vector-store restart procedure.
2. Change its URL to `snapshot_compression=none` and reopen the store. Loading
   a v2 current snapshot schedules two v1 rotations. Loading a v1 current snapshot
   with a v2 previous snapshot schedules one v1 rotation. The normal maintenance,
   `Optimize`, or `Close` path completes these rotations even without new writes.
3. Verify successful conversion before replacing the binary. Both `index.bin`
   and `index.bin.old` must have magic `0x49485647` and version `1` in their
   first eight bytes. Use the new reader to verify both files, including CRC32.
   A conversion error means rollback is not ready; keep the dual-format binary.
4. Only then start the old binary. Do not delete either recovery file to force
   rollback. Tests cover both files independently and torn-current fallback.

Conversion writes two full legacy snapshots once; it is a rollback cost, not
steady-state write savings. An interrupted conversion may leave mixed versions;
reopening with the new binary and compression disabled finishes the conversion.

## Verification and measurement

The focused Linux CI race gate already includes every `TestHNSW*` test. The same
job runs the bounded synthetic study and uploads its JSON report. Local commands:

```sh
go test ./storage/vectors -run '^TestHNSW' -count=1 -timeout=5m
go test -race ./storage/vectors -run '^TestHNSW' -count=1 -timeout=5m
GORSE_SNAPSHOT_COMPRESSION_STUDY=1 \
  GORSE_SNAPSHOT_COMPRESSION_OUTPUT=/tmp/snapshot-compression-study.json \
  go test ./storage/vectors -run '^TestHNSWSnapshotCompressionStudy$' -count=1 -timeout=2m -v
```

CI additionally exports synthetic downgraded recovery files, checks out the
exact deployed `044fa37` revision into a separate directory, and runs the small
`hnsw_legacy_compat_test.go` probe against both files using that revision's
unchanged reader. It verifies vectors, tombstones, hidden state and query order.

Tests cover exact vectors and query output after reload, empty/dense FP16/dense
FP32/sparse snapshots, byte-identical default v1 output, timestamp-only refreshes,
mixed formats, conversion through `Close` and `Optimize`, corrupted/truncated
frames, oversized envelopes/allocation requests, failed temporary-file creation,
and concurrent writes/snapshots. Linux also tests actual temporary-file writes
against the `/dev/full` failure sink. These tests verify existing recovery files
remain intact and the saved version is not acknowledged on failure.

The study measures complete snapshot writing including CRC, sync and rotation,
then complete two-pass compressed validation/reconstruction on startup, across
five repetitions per arm. Fixtures
contain synthetic vectors, graph links, metadata and tombstones. It reports
file bytes, wall time, process CPU, allocated bytes and heap deltas. Repeated
sparse values deliberately demonstrate a highly compressible boundary; the
random sparse arm and non-FP16 FP32 arm limit that fixture bias. Windows CPU
clocks are coarse and local filesystem caches are warm. These are not peak-RSS
measurements or production disk-write savings. A separately authorized rollout
still needs paired container/device counters and restart/rebuild verification.
