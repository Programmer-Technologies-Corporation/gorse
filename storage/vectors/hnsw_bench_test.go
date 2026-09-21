// Copyright 2026 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vectors

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/gorse-io/gorse/common/floats"
	"github.com/gorse-io/gorse/storage"
	"github.com/stretchr/testify/require"
)

// embeddingLike returns unit vectors with a low intrinsic dimension whose
// values are exact in FP16: the shape of the text embeddings VideoHub stores.
func embeddingLike(seed int64, n, dim int) [][]float32 {
	const latent, clusters = 24, 200
	shared := rand.New(rand.NewSource(99))
	projection := make([][]float32, dim)
	for d := range projection {
		projection[d] = make([]float32, latent)
		for j := range projection[d] {
			projection[d][j] = float32(shared.NormFloat64())
		}
	}
	centers := make([][]float32, clusters)
	for i := range centers {
		centers[i] = make([]float32, latent)
		for j := range centers[i] {
			centers[i][j] = float32(shared.NormFloat64()) * 2
		}
	}
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	z := make([]float32, latent)
	for i := range out {
		center := centers[rng.Intn(clusters)]
		for j := range z {
			z[j] = center[j] + float32(rng.NormFloat64())*0.6
		}
		v := make([]float32, dim)
		var norm float64
		for d := range v {
			v[d] = floats.Dot(projection[d], z) + float32(rng.NormFloat64())*0.05
			norm += float64(v[d] * v[d])
		}
		for d := range v {
			v[d] /= float32(math.Sqrt(norm))
		}
		out[i] = floats.ToFloat32(floats.FromFloat32(v))
	}
	return out
}

func BenchmarkHNSWQuery(b *testing.B) {
	const n, dim = 50000, 256
	data := embeddingLike(1, n, dim)
	for _, fp16 := range []bool{true, false} {
		for _, m := range []int{16, 32, 48} {
			b.Run(fmt.Sprintf("fp16=%v/M=%d", fp16, m), func(b *testing.B) {
				index := newDenseIndex(dim, Euclidean, fp16, hnswParams{M: m, M0: 2 * m, EFConstruction: 200, EFSearch: 100})
				for _, v := range data {
					index.add(v, nil)
				}
				scratch := index.getScratch()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					index.search(scratch, data[i%n], 12, 100, nil)
				}
			})
		}
	}
}

// TestVectorStoreCycle compares what one master cycle costs on xvec and on
// hnsw: load N vectors, query, rewrite all of them unchanged (what every cycle
// after the first does), query again. Run it with GORSE_VECTOR_BENCH=<N>.
func TestVectorStoreCycle(t *testing.T) {
	n, err := strconv.Atoi(os.Getenv("GORSE_VECTOR_BENCH"))
	if err != nil || n <= 0 {
		t.Skip("set GORSE_VECTOR_BENCH to the number of vectors")
	}
	const dim, batch = 256, 1024
	ctx := context.Background()
	data := embeddingLike(1, n, dim)
	for _, prefix := range []string{storage.HNSWPrefix, storage.XvecPrefix} {
		db, err := Open(prefix+t.TempDir(), "")
		require.NoError(t, err)
		require.NoError(t, db.Init())
		require.NoError(t, db.AddCollection(ctx, "bench_vectors", dim, Euclidean, VectorConfig{}))
		step := func(name string, fn func()) {
			var before, after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)
			start := time.Now()
			fn()
			elapsed := time.Since(start)
			runtime.ReadMemStats(&after)
			runtime.GC()
			var live runtime.MemStats
			runtime.ReadMemStats(&live)
			t.Logf("%-8s %-22s %10v  allocated %6d MB  live heap %5d MB", prefix, name,
				elapsed.Round(time.Millisecond), (after.TotalAlloc-before.TotalAlloc)>>20, live.HeapAlloc>>20)
		}
		write := func(timestamp time.Time) {
			for start := 0; start < n; start += batch {
				end := min(start+batch, n)
				require.NoError(t, db.AddVectors(ctx, "bench_vectors", offsetVectors(data, start, end, timestamp)))
			}
			require.NoError(t, db.DeleteVectors(ctx, "bench_vectors", timestamp))
		}
		query := func(count int) {
			for i := 0; i < count; i++ {
				_, err := db.QueryVectors(ctx, "bench_vectors", Vector{Values: data[(i*7919)%n]}, nil, 12)
				require.NoError(t, err)
			}
		}
		cycle := time.Now().UTC().Truncate(time.Millisecond)
		step("first cycle (load)", func() { write(cycle) })
		step("first query", func() { query(1) })
		step("1000 queries", func() { query(1000) })
		step("next cycle (unchanged)", func() { write(cycle.Add(time.Hour)) })
		step("first query after", func() { query(1) })
		step("single upsert + query", func() {
			require.NoError(t, db.AddVectors(ctx, "bench_vectors", []Vector{{Id: "fresh", Values: data[0], Timestamp: cycle.Add(2 * time.Hour)}}))
			query(1)
		})
		require.NoError(t, db.Close())
	}
}

func offsetVectors(data [][]float32, start, end int, timestamp time.Time) []Vector {
	vectors := make([]Vector, 0, end-start)
	for i := start; i < end; i++ {
		vectors = append(vectors, Vector{Id: fmt.Sprintf("item-%d", i), Values: data[i], Timestamp: timestamp})
	}
	return vectors
}
