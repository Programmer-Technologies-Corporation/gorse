package vectors

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/shirou/gopsutil/v3/process"
	"github.com/stretchr/testify/require"
)

// Run explicitly: the fixtures contain only deterministic synthetic vectors.
func TestHNSWSnapshotCompressionStudy(t *testing.T) {
	if os.Getenv("GORSE_SNAPSHOT_COMPRESSION_STUDY") != "1" {
		t.Skip("set GORSE_SNAPSHOT_COMPRESSION_STUDY=1")
	}
	var rows []map[string]any
	for _, kind := range []string{"dense_fp16", "dense_fp32", "sparse_repeated", "sparse_random"} {
		t.Run(kind, func(t *testing.T) {
			root := t.TempDir()
			opts := defaultHNSWOptions()
			n, dim := 6000, 256
			if kind == "sparse_repeated" || kind == "sparse_random" {
				n, dim = 100000, 0
			}
			if kind == "dense_fp32" {
				opts.precision = hnswPrecisionFP32
			}
			c := newHNSWCollection(kind, root, dim, Dot, opts)
			timestamp := time.Unix(1700000000, 0)
			var data [][]float32
			if dim > 0 {
				data = embeddingLike(12, n, dim)
			}
			rng := rand.New(rand.NewSource(24))
			batch := make([]Vector, 0, 1024)
			for i := 0; i < n; i++ {
				v := Vector{Id: fmt.Sprintf("00000000-0000-4000-8000-%012x", i), Timestamp: timestamp,
					Categories: []string{fmt.Sprintf("category-%d", i%80), "language-en", "platform-synthetic"}, IsHidden: i%11 == 0}
				if dim > 0 {
					v.Values = data[i]
					if kind == "dense_fp32" {
						for j := range v.Values {
							v.Values[j] += rng.Float32() * 0.0001
						}
					}
				} else {
					for j := 0; j < 24; j++ {
						coordinate := uint32(i%100 + j*200)
						value := float32(j+1) / 24
						if kind == "sparse_random" {
							coordinate = uint32(j*4096 + rng.Intn(4096))
							value = rng.Float32()
						}
						v.Indices = append(v.Indices, coordinate)
						v.Values = append(v.Values, value)
					}
				}
				batch = append(batch, v)
				if len(batch) == cap(batch) || i == n-1 {
					require.NoError(t, c.upsert(context.Background(), batch))
					batch = batch[:0]
				}
			}
			// Include tombstones and graph links, rather than measuring vector arrays alone.
			c.mu.Lock()
			for i := 0; i < n; i += 10 {
				c.meta.tombstone(int32(i))
			}
			frozen := c.freeze()
			c.mu.Unlock()
			for _, mode := range []string{"raw", "zstd"} {
				path := filepath.Join(root, mode)
				metrics := studyMeasure(t, func() {
					_, err := frozen.writeFile(path, mode == "zstd")
					require.NoError(t, err)
				})
				stat, err := os.Stat(path)
				require.NoError(t, err)
				metrics["kind"], metrics["mode"], metrics["bytes"] = kind, mode, stat.Size()
				read := studyMeasure(t, func() {
					loaded, err := loadHNSWSnapshot(path, kind, root, opts)
					require.NoError(t, err)
					require.Equal(t, c.meta.live, loaded.meta.live)
				})
				metrics["startup"] = read
				rows = append(rows, metrics)
				encoded, err := json.Marshal(metrics)
				require.NoError(t, err)
				t.Log(string(encoded))
			}
		})
	}
	if output := os.Getenv("GORSE_SNAPSHOT_COMPRESSION_OUTPUT"); output != "" {
		report := map[string]any{"baseline": "044fa37d6f17eac7ae6fcb8ee74c8c9e24b3e5a5", "sourceCommit": os.Getenv("GORSE_SNAPSHOT_SOURCE_COMMIT"),
			"go": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "iterations": studyIterations, "rows": rows,
			"limits": "Deterministic synthetic data; warm local filesystem; CPU is process-wide and Windows clocks are coarse; allocated bytes are not peak RSS; sparse repeated fixture is deliberately highly compressible; not production IO savings."}
		encoded, err := json.MarshalIndent(report, "", "  ")
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(output, encoded, 0600))
	}
}

const studyIterations = 5

func studyMeasure(t *testing.T, action func()) map[string]any {
	t.Helper()
	runtime.GC()
	p, err := process.NewProcess(int32(os.Getpid()))
	require.NoError(t, err)
	cpuBefore, err := p.Times()
	require.NoError(t, err)
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	for i := 0; i < studyIterations; i++ {
		action()
	}
	elapsed := time.Since(start)
	runtime.ReadMemStats(&after)
	cpuAfter, err := p.Times()
	require.NoError(t, err)
	return map[string]any{"wall_ms": float64(elapsed) / float64(time.Millisecond) / studyIterations, "cpu_ms": (cpuAfter.Total() - cpuBefore.Total()) * 1000 / studyIterations,
		"allocated_bytes": (after.TotalAlloc - before.TotalAlloc) / studyIterations, "heap_delta_bytes_total": int64(after.HeapAlloc) - int64(before.HeapAlloc)}
}
