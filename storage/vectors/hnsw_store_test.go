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
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/gorse/common/floats"
	"github.com/gorse-io/gorse/common/log"
	"github.com/gorse-io/gorse/storage"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type HNSWTestSuite struct {
	vectorsTestSuite
}

func (suite *HNSWTestSuite) SetupSuite() {
	log.SetTestLogger(suite.T())
	var err error
	suite.Database, err = Open(storage.HNSWPrefix+suite.T().TempDir(), "gorse_")
	suite.Require().NoError(err)
	suite.Require().NoError(suite.Database.Init())
}

func (suite *HNSWTestSuite) TearDownSuite() {
	suite.NoError(suite.Database.Close())
}

func TestHNSW(t *testing.T) {
	suite.Run(t, new(HNSWTestSuite))
}

func openHNSW(t testing.TB, root, options string) *HNSWStore {
	t.Helper()
	db, err := Open(storage.HNSWPrefix+root+options, "")
	require.NoError(t, err)
	require.NoError(t, db.Init())
	return db.(*HNSWStore)
}

// fp16Vectors returns clustered vectors whose values are exact in FP16, like
// the embeddings Gorse hands to the vector store.
func fp16Vectors(seed int64, n, dim, clusters int) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for j := range centers[i] {
			centers[i][j] = float32(rng.NormFloat64())
		}
	}
	vectors := make([][]float32, n)
	for i := range vectors {
		center := centers[rng.Intn(clusters)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = center[j] + float32(rng.NormFloat64())*0.35
		}
		vectors[i] = floats.ToFloat32(floats.FromFloat32(v))
	}
	return vectors
}

func exactNeighbors(data [][]float32, query []float32, k int, distance Distance, keep func(i int) bool) []string {
	type scored struct {
		id    string
		score float32
	}
	var all []scored
	for i, v := range data {
		if keep != nil && !keep(i) {
			continue
		}
		var score float32
		if distance == Euclidean {
			for j := range v {
				d := v[j] - query[j]
				score -= d * d
			}
		} else {
			score = floats.Dot(v, query)
		}
		all = append(all, scored{fmt.Sprintf("item-%d", i), score})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].score > all[j].score })
	ids := make([]string, 0, k)
	for _, s := range all[:min(k, len(all))] {
		ids = append(ids, s.id)
	}
	return ids
}

func recall(expected []string, actual []ScoredVector) float64 {
	want := make(map[string]struct{}, len(expected))
	for _, id := range expected {
		want[id] = struct{}{}
	}
	hits := 0
	for _, result := range actual {
		if _, ok := want[result.Id]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(expected))
}

func toVectors(data [][]float32, timestamp time.Time, categories func(i int) []string) []Vector {
	vectors := make([]Vector, len(data))
	for i, v := range data {
		vectors[i] = Vector{Id: fmt.Sprintf("item-%d", i), Values: v, Timestamp: timestamp}
		if categories != nil {
			vectors[i].Categories = categories(i)
		}
	}
	return vectors
}

func TestHNSWOptions(t *testing.T) {
	opts, err := parseHNSWOptions("m=24&ef_construction=150&ef_search=80&precision=fp32&snapshot_interval=30s&compact_ratio=0.5")
	require.NoError(t, err)
	require.Equal(t, hnswParams{M: 24, M0: 48, EFConstruction: 150, EFSearch: 80}, opts.params)
	require.Equal(t, hnswPrecisionFP32, opts.precision)
	require.Equal(t, 30*time.Second, opts.snapshotInterval)
	require.Equal(t, 0.5, opts.compactRatio)

	opts, err = parseHNSWOptions("")
	require.NoError(t, err)
	require.Equal(t, hnswParams{M: 32, M0: 64, EFConstruction: 200, EFSearch: 100}, opts.params)

	for _, invalid := range []string{"m=1", "precision=int8", "unknown=1", "snapshot_interval=0s", "compact_ratio=2"} {
		_, err = parseHNSWOptions(invalid)
		require.Error(t, err, invalid)
	}
	require.True(t, storage.IsEmbeddedVectorStore("hnsw:///var/lib/gorse"))
	require.True(t, storage.IsEmbeddedVectorStore("xvec:///var/lib/gorse"))
	require.False(t, storage.IsEmbeddedVectorStore("qdrant://localhost:6334"))
}

func TestHNSWRecall(t *testing.T) {
	log.SetTestLogger(t)
	ctx := context.Background()
	const n, dim, k = 6000, 64, 10
	data := fp16Vectors(1, n, dim, 40)
	queries := fp16Vectors(2, 100, dim, 40)
	for _, distance := range []Distance{Euclidean, Dot} {
		t.Run(fmt.Sprint(distance), func(t *testing.T) {
			db := openHNSW(t, t.TempDir(), "")
			defer db.Close()
			require.NoError(t, db.AddCollection(ctx, "c", dim, distance, VectorConfig{}))
			require.NoError(t, db.AddVectors(ctx, "c", toVectors(data, time.Now(), nil)))
			collection, err := db.collection("c")
			require.NoError(t, err)
			require.True(t, collection.dense.fp16, "lossless FP16 input is stored as FP16")

			var total float64
			for _, query := range queries {
				results, err := db.QueryVectors(ctx, "c", Vector{Values: query}, nil, k)
				require.NoError(t, err)
				require.Len(t, results, k)
				for i := 1; i < len(results); i++ {
					require.GreaterOrEqual(t, results[i-1].Score, results[i].Score)
				}
				total += recall(exactNeighbors(data, query, k, distance, nil), results)
			}
			// Inner product is the harder problem even with the norm augmentation.
			minimum := map[Distance]float64{Euclidean: 0.97, Dot: 0.93}[distance]
			require.GreaterOrEqual(t, total/float64(len(queries)), minimum)
		})
	}
}

func TestHNSWEuclideanScoreIsNegatedSquaredDistance(t *testing.T) {
	ctx := context.Background()
	db := openHNSW(t, t.TempDir(), "")
	defer db.Close()
	require.NoError(t, db.AddCollection(ctx, "c", 2, Euclidean, VectorConfig{}))
	require.NoError(t, db.AddVectors(ctx, "c", []Vector{{Id: "a", Values: []float32{0, 0}}, {Id: "b", Values: []float32{3, 4}}}))
	results, err := db.QueryVectors(ctx, "c", Vector{Values: []float32{0, 0}}, nil, 2)
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Equal(t, "a", results[0].Id)
	require.InDelta(t, 0, results[0].Score, 1e-6)
	require.InDelta(t, -25, results[1].Score, 1e-4)
}

func TestHNSWPrecision(t *testing.T) {
	ctx := context.Background()
	db := openHNSW(t, t.TempDir(), "")
	defer db.Close()
	// 0.1 is not representable in FP16: the collection must keep FP32.
	require.NoError(t, db.AddCollection(ctx, "lossy", 2, Euclidean, VectorConfig{}))
	require.NoError(t, db.AddVectors(ctx, "lossy", []Vector{{Id: "a", Values: []float32{0.1, 0.7}}}))
	collection, err := db.collection("lossy")
	require.NoError(t, err)
	require.False(t, collection.dense.fp16)
	stored, err := db.GetVectors(ctx, "lossy", []string{"a"})
	require.NoError(t, err)
	require.Equal(t, []float32{0.1, 0.7}, stored[0].Values)

	require.NoError(t, db.AddCollection(ctx, "exact", 2, Euclidean, VectorConfig{}))
	require.NoError(t, db.AddVectors(ctx, "exact", []Vector{{Id: "a", Values: []float32{0.5, -0.25}}}))
	collection, err = db.collection("exact")
	require.NoError(t, err)
	require.True(t, collection.dense.fp16)

	require.Error(t, db.AddVectors(ctx, "exact", []Vector{{Id: "b", Values: []float32{1}}}), "dimension mismatch")
	require.ErrorIs(t, db.AddCollection(ctx, "sq", 2, Euclidean, VectorConfig{Type: QuantizationSQ}), storage.ErrNotSupported)
}

// A cycle that rewrites every vector must only touch the ones that changed.
func TestHNSWIncrementalCycle(t *testing.T) {
	log.SetTestLogger(t)
	ctx := context.Background()
	const n, dim = 2000, 32
	data := fp16Vectors(3, n, dim, 10)
	db := openHNSW(t, t.TempDir(), "")
	defer db.Close()
	require.NoError(t, db.AddCollection(ctx, "c", dim, Euclidean, VectorConfig{}))
	cycle1 := time.Now().UTC().Truncate(time.Millisecond)
	require.NoError(t, db.AddVectors(ctx, "c", toVectors(data, cycle1, nil)))
	collection, err := db.collection("c")
	require.NoError(t, err)
	require.Equal(t, n, collection.totalSlots())
	version := collection.version

	// Cycle 2: identical vectors. Nothing is re-linked, nothing needs saving.
	cycle2 := cycle1.Add(time.Hour)
	require.NoError(t, db.AddVectors(ctx, "c", toVectors(data, cycle2, nil)))
	require.NoError(t, db.DeleteVectors(ctx, "c", cycle2))
	require.Equal(t, n, collection.totalSlots())
	require.Zero(t, collection.tombstones())
	require.Equal(t, version, collection.version)
	count, err := db.CountVectors(ctx, "c")
	require.NoError(t, err)
	require.EqualValues(t, n, count)

	// Cycle 3: item-0 gets a new embedding, item-1 is hidden, item-2 changes
	// category, item-3 disappears, item-new appears.
	cycle3 := cycle2.Add(time.Hour)
	next := toVectors(data, cycle3, nil)
	next[0].Values = data[1]
	next[1].IsHidden = true
	next[2].Categories = []string{"youtube"}
	next = append(next[:3], next[4:]...)
	next = append(next, Vector{Id: "item-new", Values: data[0], Timestamp: cycle3})
	require.NoError(t, db.AddVectors(ctx, "c", next))
	require.NoError(t, db.DeleteVectors(ctx, "c", cycle3))
	require.Equal(t, n+2, collection.totalSlots(), "one replaced and one new vector")
	require.Equal(t, 2, collection.tombstones(), "the replaced and the removed vector")
	count, err = db.CountVectors(ctx, "c")
	require.NoError(t, err)
	require.EqualValues(t, n, count)

	results, err := db.QueryVectors(ctx, "c", Vector{Values: data[0]}, nil, 3)
	require.NoError(t, err)
	require.Equal(t, "item-new", results[0].Id, "item-0 moved away, item-new took its place")
	results, err = db.QueryVectors(ctx, "c", Vector{Values: data[1]}, nil, 3)
	require.NoError(t, err)
	require.Equal(t, "item-0", results[0].Id, "item-1 is hidden, item-0 now has its vector")
	for _, result := range results {
		require.NotEqual(t, "item-1", result.Id)
	}
	results, err = db.QueryVectors(ctx, "c", Vector{Values: data[2]}, []string{"youtube"}, 3)
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "item-2", results[0].Id)
	missing, err := db.GetVectors(ctx, "c", []string{"item-3"})
	require.NoError(t, err)
	require.Empty(t, missing)
}

func TestHNSWPersistence(t *testing.T) {
	log.SetTestLogger(t)
	ctx := context.Background()
	root := t.TempDir()
	const n, dim, k = 3000, 48, 10
	data := fp16Vectors(4, n, dim, 20)
	categories := func(i int) []string { return []string{[]string{"youtube", "twitch", "kick"}[i%3], "en"} }
	timestamp := time.Now().UTC().Truncate(time.Millisecond)

	db := openHNSW(t, root, "")
	require.NoError(t, db.AddCollection(ctx, "dense", dim, Euclidean, VectorConfig{}))
	require.NoError(t, db.AddCollection(ctx, "empty", dim, Dot, VectorConfig{}))
	require.NoError(t, db.AddCollection(ctx, "sparse", 0, Dot, VectorConfig{}))
	require.NoError(t, db.AddVectors(ctx, "dense", toVectors(data, timestamp, categories)))
	require.NoError(t, db.AddVectors(ctx, "dense", []Vector{{Id: "item-7", Values: data[8], Timestamp: timestamp, Categories: categories(7)}}))
	require.NoError(t, db.AddVectors(ctx, "sparse", []Vector{
		{Id: "a", Indices: []uint32{1, 5}, Values: []float32{1, 2}, Timestamp: timestamp},
		{Id: "b", Indices: []uint32{5, 9}, Values: []float32{3, 1}, Timestamp: timestamp, Categories: []string{"x"}},
	}))
	before := make([][]ScoredVector, 20)
	for i := range before {
		var err error
		before[i], err = db.QueryVectors(ctx, "dense", Vector{Values: data[i*17]}, []string{"twitch"}, k)
		require.NoError(t, err)
	}
	require.NoError(t, db.Close())

	db = openHNSW(t, root, "")
	defer db.Close()
	names, err := db.ListCollections(ctx)
	require.NoError(t, err)
	require.Equal(t, []string{"dense", "empty", "sparse"}, names)
	info, err := db.DescribeCollection(ctx, "empty")
	require.NoError(t, err)
	require.Equal(t, &CollectionInfo{Name: "empty", Dimension: dim, Distance: Dot}, info)
	count, err := db.CountVectors(ctx, "dense")
	require.NoError(t, err)
	require.EqualValues(t, n, count)
	collection, err := db.collection("dense")
	require.NoError(t, err)
	require.True(t, collection.dense.fp16)
	require.Equal(t, 1, collection.tombstones())
	for i := range before {
		after, err := db.QueryVectors(ctx, "dense", Vector{Values: data[i*17]}, []string{"twitch"}, k)
		require.NoError(t, err)
		require.Equal(t, len(before[i]), len(after))
		for j := range after {
			require.Equal(t, before[i][j].Id, after[j].Id)
			require.Equal(t, before[i][j].Score, after[j].Score)
		}
	}
	stored, err := db.GetVectors(ctx, "dense", []string{"item-7"})
	require.NoError(t, err)
	require.Equal(t, []Vector{{Id: "item-7", Values: data[8], Timestamp: timestamp, Categories: []string{"twitch", "en"}}}, stored)
	results, err := db.QueryVectors(ctx, "sparse", Vector{Indices: []uint32{5}, Values: []float32{1}}, nil, 5)
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Equal(t, "b", results[0].Id)
	require.EqualValues(t, 3, results[0].Score)

	// The reloaded graph keeps accepting writes.
	require.NoError(t, db.AddVectors(ctx, "dense", []Vector{{Id: "late", Values: data[0], Timestamp: timestamp}}))
	late, err := db.QueryVectors(ctx, "dense", Vector{Values: data[0]}, nil, 2)
	require.NoError(t, err)
	require.ElementsMatch(t, []string{"item-0", "late"}, []string{late[0].Id, late[1].Id})
}

func TestHNSWCorruptSnapshotStartsEmpty(t *testing.T) {
	log.SetTestLogger(t)
	ctx := context.Background()
	root := t.TempDir()
	db := openHNSW(t, root, "")
	require.NoError(t, db.AddCollection(ctx, "good", 4, Dot, VectorConfig{}))
	require.NoError(t, db.AddCollection(ctx, "bad", 4, Dot, VectorConfig{}))
	require.NoError(t, db.AddVectors(ctx, "good", []Vector{{Id: "a", Values: []float32{1, 0, 0, 0}}}))
	require.NoError(t, db.AddVectors(ctx, "bad", []Vector{{Id: "a", Values: []float32{1, 0, 0, 0}}}))
	require.NoError(t, db.Close())

	path := filepath.Join(root, "bad", hnswSnapshotFile)
	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	raw[len(raw)/2] ^= 0xff
	require.NoError(t, os.WriteFile(path, raw, 0o644))
	// Without a previous snapshot to fall back to, the collection starts empty.
	require.NoError(t, os.Remove(filepath.Join(root, "bad", hnswPreviousSnapshot)))

	db = openHNSW(t, root, "")
	defer db.Close()
	names, err := db.ListCollections(ctx)
	require.NoError(t, err)
	require.Equal(t, []string{"good"}, names)
	// The writer recreates what it cannot find.
	require.NoError(t, db.AddCollection(ctx, "bad", 4, Dot, VectorConfig{}))
}

func TestHNSWTornSnapshotRecoversPrevious(t *testing.T) {
	log.SetTestLogger(t)
	ctx := context.Background()
	root := t.TempDir()
	db := openHNSW(t, root, "")
	require.NoError(t, db.AddCollection(ctx, "c", 4, Dot, VectorConfig{}))
	require.NoError(t, db.AddVectors(ctx, "c", []Vector{{Id: "a", Values: []float32{1, 0, 0, 0}}}))
	require.NoError(t, db.Optimize(ctx, "c"))
	require.NoError(t, db.AddVectors(ctx, "c", []Vector{{Id: "b", Values: []float32{0, 1, 0, 0}}}))
	require.NoError(t, db.Close())

	dir := filepath.Join(root, "c")
	_, err := os.Stat(filepath.Join(dir, hnswPreviousSnapshot))
	require.NoError(t, err, "the previous snapshot is kept")
	// Simulate a crash between the two renames: the current file is truncated.
	require.NoError(t, os.WriteFile(filepath.Join(dir, hnswSnapshotFile), []byte{1, 2, 3}, 0o644))

	db = openHNSW(t, root, "")
	defer db.Close()
	names, err := db.ListCollections(ctx)
	require.NoError(t, err)
	require.Equal(t, []string{"c"}, names)
	count, err := db.CountVectors(ctx, "c")
	require.NoError(t, err)
	require.EqualValues(t, 1, count, "the state of the previous snapshot")
	require.NoError(t, db.Optimize(ctx, "c"))
	stat, err := os.Stat(filepath.Join(dir, hnswSnapshotFile))
	require.NoError(t, err)
	require.Greater(t, stat.Size(), int64(3), "a fresh snapshot replaced the torn file")
}

func TestHNSWCompaction(t *testing.T) {
	log.SetTestLogger(t)
	ctx := context.Background()
	const n, dim, k = 4000, 32, 10
	data := fp16Vectors(5, 2*n, dim, 16)
	db := openHNSW(t, t.TempDir(), "?compact_ratio=0.2")
	defer db.Close()
	require.NoError(t, db.AddCollection(ctx, "c", dim, Euclidean, VectorConfig{}))
	require.NoError(t, db.AddVectors(ctx, "c", toVectors(data[:n], time.Now(), nil)))
	collection, err := db.collection("c")
	require.NoError(t, err)

	// Replace half of the vectors while a compaction may already be running,
	// and keep writing new ones during it.
	replaced := toVectors(data[n:n+n/2], time.Now(), nil)
	var writers sync.WaitGroup
	writers.Add(1)
	go func() {
		defer writers.Done()
		for i := 0; i < 200; i++ {
			_ = db.AddVectors(ctx, "c", []Vector{{Id: fmt.Sprintf("extra-%d", i), Values: data[n+n/2+i]}})
		}
	}()
	require.NoError(t, db.AddVectors(ctx, "c", replaced))
	writers.Wait()
	require.NoError(t, db.Optimize(ctx, "c"))
	require.Eventually(t, func() bool { return !collection.needsCompaction() }, 30*time.Second, 50*time.Millisecond)

	collection.mu.RLock()
	require.Less(t, collection.tombstones(), n/2, "tombstones were reclaimed")
	require.Equal(t, n+200, collection.meta.live)
	require.Equal(t, collection.totalSlots(), collection.dense.slots())
	collection.mu.RUnlock()

	// item-i now holds data[n+i] for the replaced half and data[i] otherwise.
	current := make([][]float32, n)
	for i := range current {
		current[i] = data[i]
		if i < n/2 {
			current[i] = data[n+i]
		}
	}
	var total float64
	for i := 0; i < 50; i++ {
		query := data[n+i]
		results, err := db.QueryVectors(ctx, "c", Vector{Values: query}, nil, k)
		require.NoError(t, err)
		require.Equal(t, fmt.Sprintf("item-%d", i), results[0].Id)
		total += recallIgnoringExtras(exactNeighbors(current, query, k, Euclidean, nil), results)
	}
	require.GreaterOrEqual(t, total/50, 0.9)
}

// recallIgnoringExtras compares against the exact neighbors among item-*
// vectors; extra-* vectors written during the compaction are not in the
// reference set.
func recallIgnoringExtras(expected []string, actual []ScoredVector) float64 {
	want := make(map[string]struct{}, len(expected))
	for _, id := range expected {
		want[id] = struct{}{}
	}
	hits, considered := 0, 0
	for _, result := range actual {
		if len(result.Id) >= 6 && result.Id[:6] == "extra-" {
			continue
		}
		considered++
		if _, ok := want[result.Id]; ok {
			hits++
		}
	}
	if considered == 0 {
		return 1
	}
	return float64(hits) / float64(considered)
}

func TestHNSWFilteredSearch(t *testing.T) {
	log.SetTestLogger(t)
	ctx := context.Background()
	const n, dim, k = 5000, 32, 10
	data := fp16Vectors(6, n, dim, 16)
	// "rare" covers 1% of the vectors, "common" half of them.
	categories := func(i int) []string {
		switch {
		case i%100 == 0:
			return []string{"rare", "common"}
		case i%2 == 0:
			return []string{"common"}
		default:
			return []string{"other"}
		}
	}
	for name, options := range map[string]string{"scan": "", "graph": "?scan_min=0&scan_ratio=0"} {
		t.Run(name, func(t *testing.T) {
			db := openHNSW(t, t.TempDir(), options)
			defer db.Close()
			require.NoError(t, db.AddCollection(ctx, "c", dim, Euclidean, VectorConfig{}))
			require.NoError(t, db.AddVectors(ctx, "c", toVectors(data, time.Now(), categories)))
			for _, category := range []string{"rare", "common"} {
				var total float64
				for i := 0; i < 40; i++ {
					query := data[i*7+1]
					results, err := db.QueryVectors(ctx, "c", Vector{Values: query}, []string{category}, k)
					require.NoError(t, err)
					require.Len(t, results, k)
					for _, result := range results {
						require.Contains(t, result.Categories, category)
					}
					want := exactNeighbors(data, query, k, Euclidean, func(i int) bool {
						return category == "common" && i%2 == 0 || i%100 == 0
					})
					total += recall(want, results)
				}
				require.GreaterOrEqual(t, total/40, 0.95, category)
			}
			results, err := db.QueryVectors(ctx, "c", Vector{Values: data[0]}, []string{"unknown"}, k)
			require.NoError(t, err)
			require.Empty(t, results)
		})
	}
}

func TestHNSWSparseMatchesExactScores(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(7))
	db := openHNSW(t, t.TempDir(), "")
	defer db.Close()
	require.NoError(t, db.AddCollection(ctx, "s", 0, Dot, VectorConfig{}))
	const n = 500
	stored := make([]map[uint32]float32, n)
	batch := make([]Vector, n)
	for i := range batch {
		stored[i] = make(map[uint32]float32)
		for len(stored[i]) < 8 {
			stored[i][uint32(rng.Intn(60))] = float32(rng.Intn(8)+1) / 4
		}
		v := Vector{Id: fmt.Sprintf("item-%d", i)}
		for index, value := range stored[i] {
			// Unsorted on purpose: the store orders coordinates itself.
			v.Indices, v.Values = append(v.Indices, index), append(v.Values, value)
		}
		batch[i] = v
	}
	require.NoError(t, db.AddVectors(ctx, "s", batch))
	require.NoError(t, db.AddVectors(ctx, "s", batch), "re-upserting the same vectors")
	collection, err := db.collection("s")
	require.NoError(t, err)
	require.Equal(t, n, collection.totalSlots())

	query := Vector{Indices: []uint32{3, 17, 42}, Values: []float32{1, 0.5, 2}}
	results, err := db.QueryVectors(ctx, "s", query, nil, 20)
	require.NoError(t, err)
	require.NotEmpty(t, results)
	exact := make([]float32, n)
	for i := range stored {
		for j, index := range query.Indices {
			exact[i] += stored[i][index] * query.Values[j]
		}
	}
	sorted := append([]float32(nil), exact...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
	for i, result := range results {
		require.InDelta(t, sorted[i], result.Score, 1e-5)
		var index int
		_, err = fmt.Sscanf(result.Id, "item-%d", &index)
		require.NoError(t, err)
		require.InDelta(t, exact[index], result.Score, 1e-5)
	}
	require.Error(t, db.AddVectors(ctx, "s", []Vector{{Id: "dup", Indices: []uint32{2, 2}, Values: []float32{1, 1}}}))

	// Rewriting every vector (IDF weights move each cycle) is reclaimed in place.
	for i := range batch {
		batch[i].Values = append([]float32(nil), batch[i].Values...)
		batch[i].Values[0] += 0.25
	}
	for cycle := 0; cycle < 4; cycle++ {
		require.NoError(t, db.AddVectors(ctx, "s", batch))
		require.NoError(t, db.Optimize(ctx, "s"))
	}
	require.Less(t, collection.totalSlots(), 2*n+hnswMinTombstones+1)
}

// Searches reuse pooled scratch: no visited set, heap or result buffer is
// allocated per query.
func TestHNSWSearchDoesNotAllocate(t *testing.T) {
	const n, dim = 3000, 64
	data := fp16Vectors(8, n, dim, 12)
	index := newDenseIndex(dim, Euclidean, true, defaultHNSWOptions().params)
	for _, v := range data {
		index.add(v, nil)
	}
	scratch := index.getScratch()
	query := data[11]
	index.search(scratch, query, 10, 100, nil)
	allocations := testing.AllocsPerRun(200, func() {
		index.search(scratch, query, 10, 100, nil)
	})
	require.Zero(t, allocations)
	if raceDetector {
		return
	}
	allocations = testing.AllocsPerRun(50, func() {
		index.add(query, nil)
	})
	require.LessOrEqual(t, allocations, 4.0, "an insert only allocates when a slab grows or a node gets upper layers")
}
