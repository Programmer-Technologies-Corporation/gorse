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

package logics

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/gorse-io/gorse/common/log"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// VideoHub fork: the upstream similarity suites, run against the hnsw:// store.

func openHNSWStore(t *testing.T) *vectors.HNSWStore {
	client, err := vectors.Open(fmt.Sprintf("hnsw://%s/vectors", t.TempDir()), "")
	require.NoError(t, err)
	require.NoError(t, client.Init())
	return client.(*vectors.HNSWStore)
}

type HNSWItemToItemTestSuite struct {
	ItemToItemTestSuite
}

func (suite *HNSWItemToItemTestSuite) SetupTest() {
	log.SetTestLogger(suite.T())
	suite.vectorClient = openHNSWStore(suite.T())
}

func TestItemToItemOnHNSW(t *testing.T) {
	suite.Run(t, new(HNSWItemToItemTestSuite))
}

type HNSWUserToUserTestSuite struct {
	UserToUserTestSuite
}

func (suite *HNSWUserToUserTestSuite) SetupTest() {
	log.SetTestLogger(suite.T())
	suite.vectorClient = openHNSWStore(suite.T())
}

func TestUserToUserOnHNSW(t *testing.T) {
	suite.Run(t, new(HNSWUserToUserTestSuite))
}

// The master rewrites every item each cycle. The store must recognize unchanged vectors, keep them in FP16
// and only re-link what changed.
func TestEmbeddingCycleOnHNSWIsIncremental(t *testing.T) {
	log.SetTestLogger(t)
	store := openHNSWStore(t)
	defer store.Close()
	cfg := config.ItemToItemConfig{Name: "neighbors", Type: "embedding", Column: "item.Labels.embedding"}
	collection := vectors.ItemToItemCollection(cfg.Name)

	const n, dim = 300, 16
	rng := rand.New(rand.NewSource(1))
	items := make([]data.Item, n)
	for i := range items {
		embedding := make([]any, dim)
		for j := range embedding {
			embedding[j] = rng.Float64() - 0.5 // labels arrive from the data store as float64
		}
		items[i] = data.Item{ItemId: fmt.Sprintf("item-%d", i), Categories: []string{"youtube"}, Labels: map[string]any{"embedding": embedding}}
	}
	cycle := func(timestamp time.Time) {
		recommender, err := NewItemToItem(cfg, timestamp, &ItemToItemOptions{Context: t.Context(), VectorClient: store})
		require.NoError(t, err)
		for i := range items {
			require.NoError(t, recommender.Add(&items[i], nil))
		}
		require.NoError(t, recommender.Clean())
	}

	start := time.Now().UTC().Truncate(time.Millisecond)
	cycle(start)
	stats, err := store.CollectionStats(collection)
	require.NoError(t, err)
	require.Equal(t, vectors.HNSWCollectionStats{Vectors: n, Precision: "fp16", Bytes: stats.Bytes}, stats)

	cycle(start.Add(time.Hour))
	stats, err = store.CollectionStats(collection)
	require.NoError(t, err)
	require.Equal(t, n, stats.Vectors)
	require.Zero(t, stats.Tombstones, "an unchanged cycle must not rewrite vectors")

	// One embedding changes, one item disappears.
	items[0].Labels = map[string]any{"embedding": items[1].Labels.(map[string]any)["embedding"]}
	items = items[:n-1]
	cycle(start.Add(2 * time.Hour))
	stats, err = store.CollectionStats(collection)
	require.NoError(t, err)
	require.Equal(t, n-1, stats.Vectors)
	require.Equal(t, 2, stats.Tombstones)

	scores, err := QueryItemToItem(t.Context(), store, cfg, "item-0", []string{"youtube"}, 3)
	require.NoError(t, err)
	require.Equal(t, "item-1", scores[0].Id, "item-0 now shares item-1's embedding")
	require.InDelta(t, 1, scores[0].Score, 1e-6)
}
