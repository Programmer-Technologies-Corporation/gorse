package vectors

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/stretchr/testify/require"
)

func compressionFixture(t *testing.T, kind string) *hnswCollection {
	t.Helper()
	opts := defaultHNSWOptions()
	dim := 16
	if kind == "sparse" {
		dim = 0
	}
	if kind == "fp32" {
		opts.precision = hnswPrecisionFP32
	}
	c := newHNSWCollection("synthetic", t.TempDir(), dim, Dot, opts)
	var vectors []Vector
	for i := 0; i < 80; i++ {
		v := Vector{Id: fmt.Sprintf("id-%d", i), Timestamp: time.Unix(1700000000, 0), Categories: []string{fmt.Sprintf("category-%d", i%3)}, IsHidden: i%9 == 0}
		if dim == 0 {
			v.Indices = []uint32{1, uint32(i + 2)}
			v.Values = []float32{1, float32(i) / 13}
		} else {
			for j := 0; j < dim; j++ {
				value := float32(i + j)
				if kind == "fp32" {
					value /= 17
				}
				v.Values = append(v.Values, value)
			}
		}
		vectors = append(vectors, v)
	}
	if kind != "empty" {
		require.NoError(t, c.upsert(context.Background(), vectors))
		changed := vectors[7]
		changed.Values = append([]float32(nil), changed.Values...)
		changed.Values[0]++
		require.NoError(t, c.upsert(context.Background(), []Vector{changed}))
		c.mu.Lock()
		c.meta.tombstone(0)
		c.version++
		c.mu.Unlock()
	}
	return c
}

func TestHNSWCompressedSnapshotRoundTrip(t *testing.T) {
	for _, kind := range []string{"fp16", "fp32", "sparse", "empty"} {
		t.Run(kind, func(t *testing.T) {
			c := compressionFixture(t, kind)
			c.opts.snapshotCompression = "zstd"
			before := c.get([]string{"id-1", "id-7", "id-9"})
			q := Vector{Values: []float32{1, 2}}
			if c.dim > 0 {
				q.Values = make([]float32, c.dim)
				q.Values[0] = 1
			} else {
				q.Indices = []uint32{1, 3}
			}
			queries, err := c.query(q, []string{"category-1"}, 10)
			require.NoError(t, err)
			require.NoError(t, c.snapshot())
			path := filepath.Join(c.dir, hnswSnapshotFile)
			version, err := hnswSnapshotFileVersion(path)
			require.NoError(t, err)
			require.EqualValues(t, 2, version)
			after, err := loadHNSWSnapshot(path, c.name, c.dir, c.opts)
			require.NoError(t, err)
			require.Equal(t, before, after.get([]string{"id-1", "id-7", "id-9"}))
			require.Equal(t, c.tombstones(), after.tombstones())
			actual, err := after.query(q, []string{"category-1"}, 10)
			require.NoError(t, err)
			require.Equal(t, queries, actual)
			// Timestamp-only refreshes retain the existing no-snapshot rule.
			bytesBefore, err := os.ReadFile(path)
			require.NoError(t, err)
			if len(before) > 0 {
				before[0].Timestamp = before[0].Timestamp.Add(time.Minute)
				require.NoError(t, after.upsert(context.Background(), before[:1]))
			}
			require.NoError(t, after.snapshot())
			bytesAfter, err := os.ReadFile(path)
			require.NoError(t, err)
			require.Equal(t, bytesBefore, bytesAfter)
		})
	}
}

func TestHNSWLegacySnapshotBytesUnchanged(t *testing.T) {
	c := compressionFixture(t, "fp32")
	require.Equal(t, "none", c.opts.snapshotCompression)
	f := c.freeze()
	var expected bytes.Buffer
	w := &snapshotWriter{w: bufio.NewWriter(&expected)}
	f.encode(w)
	require.NoError(t, w.w.Flush())
	require.NoError(t, w.err)
	var trailer [4]byte
	binary.LittleEndian.PutUint32(trailer[:], crc32.ChecksumIEEE(expected.Bytes()))
	expected.Write(trailer[:])
	path := filepath.Join(c.dir, hnswSnapshotFile)
	_, err := f.writeFile(path, false)
	require.NoError(t, err)
	actual, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, expected.Bytes(), actual)
	for _, option := range []string{"snapshot_compression=none", "snapshot_compression=zstd"} {
		_, err := parseHNSWOptions(option)
		require.NoError(t, err)
	}
	_, err = parseHNSWOptions("snapshot_compression=gzip")
	require.Error(t, err)
}

func TestHNSWCompressedWriterFallsBackAtReaderLimits(t *testing.T) {
	for _, kind := range []string{"fp16", "fp32", "sparse", "empty"} {
		t.Run(kind, func(t *testing.T) {
			c := compressionFixture(t, kind)
			f := c.freeze()
			path := filepath.Join(c.dir, hnswSnapshotFile)
			_, err := f.writeFile(path, false)
			require.NoError(t, err)
			raw, err := os.ReadFile(path)
			require.NoError(t, err)
			decoded, allocated := f.snapshotSizes()
			require.EqualValues(t, len(raw), decoded)
			accounting := &snapshotReader{r: bufio.NewReader(bytes.NewReader(raw[:len(raw)-4])), limit: decoded, allocationLimit: hnswMaxSnapshotAllocationBytes}
			decodeHNSWCollection(accounting, c.name, c.dir, c.opts)
			require.NoError(t, accounting.err)
			require.Equal(t, allocated, accounting.allocated, "writer preflight must match the reader's allocation charges")
			for _, test := range []struct {
				name               string
				decoded, allocated uint64
				version            uint32
			}{
				{"decoded-below", decoded - 1, hnswMaxSnapshotAllocationBytes, 1},
				{"decoded-at", decoded, hnswMaxSnapshotAllocationBytes, 2},
				{"decoded-above", decoded + 1, hnswMaxSnapshotAllocationBytes, 2},
				{"allocation-below", hnswMaxDecodedSnapshotBytes, allocated - 1, 1},
				{"allocation-at", hnswMaxDecodedSnapshotBytes, allocated, 2},
			} {
				if test.name == "allocation-below" && allocated == 0 {
					continue
				}
				t.Run(test.name, func(t *testing.T) {
					_, err := f.writeFileWithLimits(path, true, test.decoded, test.allocated)
					require.NoError(t, err)
					version, err := hnswSnapshotFileVersion(path)
					require.NoError(t, err)
					require.Equal(t, test.version, version)
					loaded, err := loadHNSWSnapshot(path, c.name, c.dir, c.opts)
					require.NoError(t, err)
					require.Equal(t, c.meta.live, loaded.meta.live)
					if version == 1 {
						actual, err := os.ReadFile(path)
						require.NoError(t, err)
						require.Equal(t, raw, actual)
					}
				})
			}
		})
	}
}

func TestHNSWCompressedDowngradeBothRecoveryFiles(t *testing.T) {
	for _, currentCompressed := range []bool{false, true} {
		for _, previousCompressed := range []bool{false, true} {
			t.Run(fmt.Sprintf("current=%v/previous=%v", currentCompressed, previousCompressed), func(t *testing.T) {
				c := compressionFixture(t, "fp16")
				f := c.freeze()
				_, err := f.writeFile(filepath.Join(c.dir, hnswSnapshotFile), previousCompressed)
				require.NoError(t, err)
				_, err = f.writeFile(filepath.Join(c.dir, hnswSnapshotFile), currentCompressed)
				require.NoError(t, err)
				loaded, err := loadHNSWCollection(c.name, c.dir, defaultHNSWOptions())
				require.NoError(t, err)
				require.NoError(t, loaded.snapshot()) // same path used by Close and Optimize
				for _, name := range []string{hnswSnapshotFile, hnswPreviousSnapshot} {
					path := filepath.Join(c.dir, name)
					version, err := hnswSnapshotFileVersion(path)
					require.NoError(t, err)
					require.EqualValues(t, 1, version)
					file, err := os.Open(path)
					require.NoError(t, err)
					stat, err := file.Stat()
					require.NoError(t, err)
					legacy, err := readHNSWSnapshot(file, uint64(stat.Size()), c.name, c.dir, defaultHNSWOptions(), 0)
					require.NoError(t, err)
					require.NoError(t, file.Close())
					require.Equal(t, c.get([]string{"id-1"}), legacy.get([]string{"id-1"}))
				}
				// A torn current file must still recover from the v1 previous file.
				require.NoError(t, os.WriteFile(filepath.Join(c.dir, hnswSnapshotFile), []byte{1, 2, 3}, 0600))
				recovered, err := loadHNSWCollection(c.name, c.dir, defaultHNSWOptions())
				require.NoError(t, err)
				require.Equal(t, c.get([]string{"id-1"}), recovered.get([]string{"id-1"}))
			})
		}
	}
}

func TestHNSWCompressedSnapshotRejectsCorruption(t *testing.T) {
	c := compressionFixture(t, "fp16")
	path := filepath.Join(c.dir, hnswSnapshotFile)
	_, err := c.freeze().writeFile(path, true)
	require.NoError(t, err)
	good, err := os.ReadFile(path)
	require.NoError(t, err)
	cases := map[string][]byte{}
	for _, n := range []int{0, 1, 7, 12, len(good) - 1} {
		cases[fmt.Sprint("truncated-", n)] = append([]byte(nil), good[:n]...)
	}
	for _, n := range []int{len(good) / 2, len(good) - 1} {
		bad := append([]byte(nil), good...)
		bad[n] ^= 0xff
		cases[fmt.Sprint("corrupt-", n)] = bad
	}
	for _, size := range []uint64{0, 11, hnswMaxDecodedSnapshotBytes + 1, binary.LittleEndian.Uint64(good[8:]) + 1, binary.LittleEndian.Uint64(good[8:]) - 1} {
		bad := append([]byte(nil), good...)
		binary.LittleEndian.PutUint64(bad[8:], size)
		cases[fmt.Sprint("length-", size)] = bad
	}
	for label, bad := range cases {
		t.Run(label, func(t *testing.T) {
			require.NoError(t, os.WriteFile(path, bad, 0600))
			_, err := loadHNSWSnapshot(path, c.name, c.dir, c.opts)
			require.Error(t, err)
		})
	}
	// A tiny compressed body cannot ask for an enormous slot allocation, even
	// when its envelope falsely declares a permitted large decoded length.
	var raw bytes.Buffer
	w := &snapshotWriter{w: bufio.NewWriter(&raw)}
	c.freeze().encode(w)
	require.NoError(t, w.w.Flush())
	decoded := raw.Bytes()
	binary.LittleEndian.PutUint64(decoded[44:52], 1<<28)
	encoder, err := zstd.NewWriter(nil, zstd.WithEncoderConcurrency(1), zstd.WithWindowSize(hnswSnapshotWindowBytes))
	require.NoError(t, err)
	defer encoder.Close()
	bad := append([]byte(nil), good[:16]...)
	binary.LittleEndian.PutUint64(bad[8:], hnswMaxDecodedSnapshotBytes)
	bad = append(bad, encoder.EncodeAll(decoded, nil)...)
	require.NoError(t, os.WriteFile(path, bad, 0600))
	_, err = loadHNSWSnapshot(path, c.name, c.dir, c.opts)
	require.Error(t, err) // rejected during bounded prevalidation, before allocation
	// A correctly framed body with a disproportionate slot count must also
	// be rejected by the allocation allowance based on verified decoded bytes.
	verified := make([]byte, 9<<20)
	copy(verified, decoded)
	binary.LittleEndian.PutUint64(verified[44:52], 1<<20)
	binary.LittleEndian.PutUint32(verified[len(verified)-4:], crc32.ChecksumIEEE(verified[:len(verified)-4]))
	bad = append([]byte(nil), good[:16]...)
	binary.LittleEndian.PutUint64(bad[8:], uint64(len(verified)))
	bad = append(bad, encoder.EncodeAll(verified, nil)...)
	require.NoError(t, os.WriteFile(path, bad, 0600))
	_, err = loadHNSWSnapshot(path, c.name, c.dir, c.opts)
	require.ErrorContains(t, err, "allocation exceeds limit")
}

func TestHNSWCompressedInnerChecksumAndWindowLimits(t *testing.T) {
	c := compressionFixture(t, "fp16")
	path := filepath.Join(c.dir, hnswSnapshotFile)
	_, err := c.freeze().writeFile(path, false)
	require.NoError(t, err)
	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	raw[len(raw)-1] ^= 0xff
	encoder, err := zstd.NewWriter(nil, zstd.WithEncoderConcurrency(1), zstd.WithWindowSize(hnswSnapshotWindowBytes))
	require.NoError(t, err)
	defer encoder.Close()
	envelope := make([]byte, hnswCompressedHeaderBytes)
	binary.LittleEndian.PutUint32(envelope[:4], hnswSnapshotMagic)
	binary.LittleEndian.PutUint32(envelope[4:8], hnswCompressedSnapshotVersion)
	binary.LittleEndian.PutUint64(envelope[8:], uint64(len(raw)))
	envelope = append(envelope, encoder.EncodeAll(raw, nil)...)
	require.NoError(t, os.WriteFile(path, envelope, 0600))
	_, err = loadHNSWSnapshot(path, c.name, c.dir, c.opts)
	require.ErrorContains(t, err, "checksum")
	largeEncoder, err := zstd.NewWriter(nil, zstd.WithEncoderConcurrency(1), zstd.WithWindowSize(2*hnswSnapshotWindowBytes))
	require.NoError(t, err)
	defer largeEncoder.Close()
	large := make([]byte, 3*hnswSnapshotWindowBytes)
	copy(large, raw)
	largeEnvelope := append([]byte(nil), envelope[:hnswCompressedHeaderBytes]...)
	binary.LittleEndian.PutUint64(largeEnvelope[8:], uint64(len(large)))
	largeEnvelope = append(largeEnvelope, largeEncoder.EncodeAll(large, nil)...)
	require.NoError(t, os.WriteFile(path, largeEnvelope, 0600))
	_, err = loadHNSWSnapshot(path, c.name, c.dir, c.opts)
	require.ErrorContains(t, err, "window size")
}

func TestHNSWCompressedStoreCloseAndOptimizeDowngrade(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	db := openHNSW(t, root, "?snapshot_compression=zstd")
	require.NoError(t, db.AddCollection(ctx, "collection", 2, Dot, VectorConfig{}))
	require.NoError(t, db.AddVectors(ctx, "collection", []Vector{{Id: "persisted", Values: []float32{1, 2}}}))
	require.NoError(t, db.Optimize(ctx, "collection"))
	require.NoError(t, db.Close())
	path := filepath.Join(root, "collection", hnswSnapshotFile)
	version, err := hnswSnapshotFileVersion(path)
	require.NoError(t, err)
	require.EqualValues(t, 2, version)
	for _, flush := range []string{"Close", "Optimize"} {
		t.Run(flush, func(t *testing.T) {
			legacy := openHNSW(t, root, "?snapshot_compression=none")
			if flush == "Optimize" {
				require.NoError(t, legacy.Optimize(ctx, "collection"))
			}
			require.NoError(t, legacy.Close())
			for _, name := range []string{hnswSnapshotFile, hnswPreviousSnapshot} {
				v, err := hnswSnapshotFileVersion(filepath.Join(root, "collection", name))
				require.NoError(t, err)
				require.EqualValues(t, 1, v)
			}
			reopened := openHNSW(t, root, "")
			values, err := reopened.GetVectors(ctx, "collection", []string{"persisted"})
			require.NoError(t, err)
			require.Equal(t, []float32{1, 2}, values[0].Values)
			require.NoError(t, reopened.Close())
			// Prepare compressed recovery files again for the next conversion arm.
			compressed := openHNSW(t, root, "?snapshot_compression=zstd")
			require.NoError(t, compressed.AddVectors(ctx, "collection", []Vector{{Id: "next", Values: []float32{3, 4}}}))
			require.NoError(t, compressed.Optimize(ctx, "collection"))
			require.NoError(t, compressed.Close())
		})
	}
}

func TestHNSWCompressedSnapshotFailurePreservesCurrent(t *testing.T) {
	c := compressionFixture(t, "fp16")
	c.opts.snapshotCompression = "zstd"
	require.NoError(t, c.snapshot())
	path := filepath.Join(c.dir, hnswSnapshotFile)
	good, err := os.ReadFile(path)
	require.NoError(t, err)
	c.version++
	require.NoError(t, os.Mkdir(path+".tmp", 0700))
	require.Error(t, c.snapshot())
	actual, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, good, actual)
	require.NotEqual(t, c.version, c.savedVersion)
	_, err = loadHNSWSnapshot(path, c.name, c.dir, c.opts)
	require.NoError(t, err)
}

func TestHNSWCompressedConcurrentSnapshot(t *testing.T) {
	c := compressionFixture(t, "fp16")
	c.opts.snapshotCompression = "zstd"
	var wg sync.WaitGroup
	errors := make(chan error, 64)
	for worker := 0; worker < 3; worker++ {
		wg.Go(func() {
			for i := 0; i < 20; i++ {
				v := Vector{Id: "concurrent", Values: make([]float32, c.dim), Timestamp: time.Unix(int64(i), 0)}
				v.Values[0] = float32(i)
				if err := c.upsert(context.Background(), []Vector{v}); err != nil {
					errors <- err
				}
			}
		})
	}
	wg.Go(func() {
		for i := 0; i < 20; i++ {
			if err := c.snapshot(); err != nil {
				errors <- err
			}
		}
	})
	wg.Wait()
	close(errors)
	for err := range errors {
		require.NoError(t, err)
	}
	require.NoError(t, c.snapshot())
	loaded, err := loadHNSWSnapshot(filepath.Join(c.dir, hnswSnapshotFile), c.name, c.dir, c.opts)
	require.NoError(t, err)
	require.Equal(t, c.get([]string{"concurrent"}), loaded.get([]string{"concurrent"}))
}

func TestHNSWCompressedExportRollbackFixture(t *testing.T) {
	destination := os.Getenv("GORSE_SNAPSHOT_ROLLBACK_FIXTURE")
	if destination == "" {
		t.Skip("set GORSE_SNAPSHOT_ROLLBACK_FIXTURE to a new synthetic fixture directory")
	}
	require.NoError(t, os.Mkdir(destination, 0700))
	c := compressionFixture(t, "fp16")
	c.opts.snapshotCompression = "zstd"
	require.NoError(t, c.snapshot())
	c.version++
	require.NoError(t, c.snapshot())
	legacy, err := loadHNSWCollection(c.name, c.dir, defaultHNSWOptions())
	require.NoError(t, err)
	require.NoError(t, legacy.snapshot())
	for _, name := range []string{hnswSnapshotFile, hnswPreviousSnapshot} {
		data, err := os.ReadFile(filepath.Join(c.dir, name))
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(destination, name), data, 0600))
	}
}
