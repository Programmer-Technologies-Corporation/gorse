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

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/samber/lo"
	"github.com/steinfletcher/apitest"
)

func (suite *ServerTestSuite) enableIncrementalItemToItem() {
	suite.Config.VideoHub.IncrementalItemToItem = true
	suite.Config.Recommend.ItemToItem = []config.ItemToItemConfig{
		{Name: "neighbors", Type: "embedding", Column: "item.Labels.embedding"},
		{Name: "label_neighbors", Type: "tags", Column: "item.Labels.tags"},
	}
}

func (suite *ServerTestSuite) neighborIds(path string) []string {
	request := httptest.NewRequest(http.MethodGet, path, nil)
	request.Header.Set("X-API-Key", apiKey)
	recorder := httptest.NewRecorder()
	suite.handler.ServeHTTP(recorder, request)
	suite.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())
	var scores []cache.Score
	suite.Require().NoError(json.Unmarshal(recorder.Body.Bytes(), &scores))
	return lo.Map(scores, func(score cache.Score, _ int) string { return score.Id })
}

func (suite *ServerTestSuite) TestIncrementalItemToItemOnInsert() {
	suite.enableIncrementalItemToItem()
	ctx := suite.T().Context()
	items := []Item{
		{ItemId: "source", Labels: map[string]any{"embedding": []float64{0, 0}}},
		{ItemId: "near", Labels: map[string]any{"embedding": []float64{0.1, 0}}, Categories: []string{"movie"}},
		{ItemId: "far", Labels: map[string]any{"embedding": []float64{10, 0}}},
		{ItemId: "hidden", Labels: map[string]any{"embedding": []float64{0.02, 0}}, IsHidden: true},
		{ItemId: "plain", Labels: map[string]any{"tags": []string{"a"}}},
	}
	apitest.New().Handler(suite.handler).Post("/api/items").Header("X-API-Key", apiKey).JSON(items).
		Expect(suite.T()).Status(http.StatusOK).End()

	// neighbors are available immediately, without a master job
	suite.Equal([]string{"near", "far"}, suite.neighborIds("/api/item-to-item/neighbors/source"))
	suite.Equal([]string{"near"}, suite.neighborIds("/api/item-to-item/neighbors/source?category=movie"))
	// hidden items keep their own neighbors but are not neighbors of others
	suite.Equal([]string{"source", "near", "far"}, suite.neighborIds("/api/item-to-item/neighbors/hidden"))
	// only the embedding recommender is indexed
	count, err := suite.VectorClient.CountVectors(ctx, vectors.ItemToItemCollection("neighbors"))
	suite.NoError(err)
	suite.Equal(int64(4), count)
	_, err = suite.VectorClient.DescribeCollection(ctx, vectors.ItemToItemCollection("label_neighbors"))
	suite.Error(err)
	suite.Empty(suite.neighborIds("/api/item-to-item/label_neighbors/source"))
}

func (suite *ServerTestSuite) TestIncrementalItemToItemDisabled() {
	suite.Config.Recommend.ItemToItem = []config.ItemToItemConfig{{Name: "neighbors", Type: "embedding", Column: "item.Labels.embedding"}}
	apitest.New().Handler(suite.handler).Post("/api/items").Header("X-API-Key", apiKey).JSON([]Item{
		{ItemId: "source", Labels: map[string]any{"embedding": []float64{0, 0}}},
		{ItemId: "near", Labels: map[string]any{"embedding": []float64{0.1, 0}}},
	}).Expect(suite.T()).Status(http.StatusOK).End()
	_, err := suite.VectorClient.DescribeCollection(suite.T().Context(), vectors.ItemToItemCollection("neighbors"))
	suite.Error(err, "upstream behaviour: nothing is indexed outside the master job")
}

func (suite *ServerTestSuite) TestItemToItemFallbackFromStoredEmbedding() {
	suite.enableIncrementalItemToItem()
	ctx := suite.T().Context()
	now := time.Now()
	collection := vectors.ItemToItemCollection("neighbors")
	suite.NoError(suite.VectorClient.AddCollection(ctx, collection, 2, vectors.Euclidean, vectors.VectorConfig{}))
	suite.NoError(suite.VectorClient.AddVectors(ctx, collection, []vectors.Vector{
		{Id: "near", Values: []float32{0.1, 0}, Timestamp: now},
		{Id: "far", Values: []float32{10, 0}, Timestamp: now},
	}))
	// the new item exists only in the data store, e.g. inserted while the flag was off
	suite.NoError(suite.DataClient.BatchInsertItems(ctx, []data.Item{
		{ItemId: "new", Labels: map[string]any{"embedding": []float32{0, 0}}, Timestamp: now},
		{ItemId: "near", Labels: map[string]any{"embedding": []float32{0.1, 0}}, Timestamp: now},
		{ItemId: "far", Labels: map[string]any{"embedding": []float32{10, 0}}, Timestamp: now},
	}))
	suite.Equal([]string{"near", "far"}, suite.neighborIds("/api/item-to-item/neighbors/new"))
	suite.Equal([]string{"near", "far"}, suite.neighborIds("/api/item/new/neighbors"))
	suite.Equal([]string{"far"}, suite.neighborIds("/api/item-to-item/neighbors/new?n=1&offset=1"))

	suite.Config.VideoHub.IncrementalItemToItem = false
	suite.Empty(suite.neighborIds("/api/item-to-item/neighbors/new"))
}

func (suite *ServerTestSuite) TestModifyItemReindexes() {
	suite.enableIncrementalItemToItem()
	ctx := suite.T().Context()
	apitest.New().Handler(suite.handler).Post("/api/items").Header("X-API-Key", apiKey).JSON([]Item{
		{ItemId: "source", Labels: map[string]any{"embedding": []float64{0, 0}}},
		{ItemId: "near", Labels: map[string]any{"title": "no embedding yet"}},
	}).Expect(suite.T()).Status(http.StatusOK).End()
	suite.Empty(suite.neighborIds("/api/item-to-item/neighbors/source"))

	// a label patch that adds the embedding indexes the item
	apitest.New().Handler(suite.handler).Patch("/api/item/near").Header("X-API-Key", apiKey).
		JSON(data.ItemPatch{Labels: map[string]any{"embedding": []float64{0.1, 0}}}).
		Expect(suite.T()).Status(http.StatusOK).End()
	suite.Equal([]string{"near"}, suite.neighborIds("/api/item-to-item/neighbors/source"))

	// hiding the item removes it from other items' neighbors
	apitest.New().Handler(suite.handler).Patch("/api/item/near").Header("X-API-Key", apiKey).
		JSON(data.ItemPatch{IsHidden: lo.ToPtr(true)}).
		Expect(suite.T()).Status(http.StatusOK).End()
	stored, err := suite.VectorClient.GetVectors(ctx, vectors.ItemToItemCollection("neighbors"), []string{"near"})
	suite.NoError(err)
	suite.Require().Len(stored, 1)
	suite.True(stored[0].IsHidden)
	suite.Empty(suite.neighborIds("/api/item-to-item/neighbors/source"))
}

func (suite *ServerTestSuite) TestPatchItemLabels() {
	ctx := suite.T().Context()
	suite.enableIncrementalItemToItem()
	suite.NoError(suite.DataClient.BatchInsertItems(ctx, []data.Item{
		{ItemId: "a", Labels: map[string]any{"title": "old", "embedding": []float64{0, 0}, "keep": "yes"}, Timestamp: time.Now()},
		{ItemId: "legacy", Labels: []string{"tag"}, Timestamp: time.Now()},
	}))

	// disabled by default
	apitest.New().Handler(suite.handler).Patch("/api/item/a/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{"title": "new"}).
		Expect(suite.T()).Status(http.StatusNotFound).End()
	suite.Config.VideoHub.LabelPatch = true

	// merge: replace embedding, delete title, add a key, keep the rest
	apitest.New().Handler(suite.handler).Patch("/api/item/a/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{"embedding": []float64{1, 0}, "title": nil, "added": "x"}).
		Expect(suite.T()).Status(http.StatusOK).End()
	item, err := suite.DataClient.GetItem(ctx, "a")
	suite.NoError(err)
	labels, ok := item.Labels.(map[string]any)
	suite.Require().True(ok)
	suite.Equal("yes", labels["keep"])
	suite.Equal("x", labels["added"])
	suite.NotContains(labels, "title")
	suite.Equal([]any{1.0, 0.0}, labels["embedding"])
	// the incremental index follows the patch
	stored, err := suite.VectorClient.GetVectors(ctx, vectors.ItemToItemCollection("neighbors"), []string{"a"})
	suite.NoError(err)
	suite.Require().Len(stored, 1)
	suite.Equal([]float32{1, 0}, stored[0].Values)
	// Repeating the same logical patch still heals derived index loss. The
	// split-count SQL implementation suppresses the identical persistent row
	// update independently of this cache/index refresh.
	suite.NoError(suite.VectorClient.DeleteVectors(ctx, vectors.ItemToItemCollection("neighbors"), time.Now().Add(time.Minute)))
	count, err := suite.VectorClient.CountVectors(ctx, vectors.ItemToItemCollection("neighbors"))
	suite.NoError(err)
	suite.Zero(count)
	apitest.New().Handler(suite.handler).Patch("/api/item/a/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{"embedding": []float64{1, 0}}).
		Expect(suite.T()).Status(http.StatusOK).End()
	stored, err = suite.VectorClient.GetVectors(ctx, vectors.ItemToItemCollection("neighbors"), []string{"a"})
	suite.NoError(err)
	suite.Require().Len(stored, 1)
	suite.Equal([]float32{1, 0}, stored[0].Values)
	// the modify timestamp is refreshed
	modified, err := suite.CacheClient.Get(ctx, cache.Key(cache.LastModifyItemTime, "a")).Time()
	suite.NoError(err)
	suite.WithinDuration(time.Now(), modified, time.Minute)

	// rejected inputs
	apitest.New().Handler(suite.handler).Patch("/api/item/a/labels").Header("X-API-Key", apiKey).
		Body(`["not", "an", "object"]`).Header("Content-Type", "application/json").
		Expect(suite.T()).Status(http.StatusBadRequest).End()
	apitest.New().Handler(suite.handler).Patch("/api/item/a/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{}).
		Expect(suite.T()).Status(http.StatusBadRequest).End()
	apitest.New().Handler(suite.handler).Patch("/api/item/missing/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{"title": "new"}).
		Expect(suite.T()).Status(http.StatusNotFound).End()
	apitest.New().Handler(suite.handler).Patch("/api/item/legacy/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{"title": "new"}).
		Expect(suite.T()).Status(http.StatusBadRequest).End()
}

func (suite *ServerTestSuite) TestEmbeddingDimensionsValidation() {
	suite.enableIncrementalItemToItem()
	suite.Config.VideoHub.LabelPatch = true
	suite.Config.VideoHub.EmbeddingDimensions = 2

	apitest.New().Handler(suite.handler).Post("/api/items").Header("X-API-Key", apiKey).JSON([]Item{
		{ItemId: "ok", Labels: map[string]any{"embedding": []float64{0, 0}}},
		{ItemId: "no_embedding", Labels: map[string]any{"title": "x"}},
	}).Expect(suite.T()).Status(http.StatusOK).End()
	apitest.New().Handler(suite.handler).Post("/api/items").Header("X-API-Key", apiKey).JSON([]Item{
		{ItemId: "bad", Labels: map[string]any{"embedding": []float64{0, 0, 0}}},
	}).Expect(suite.T()).Status(http.StatusBadRequest).End()
	_, err := suite.DataClient.GetItem(suite.T().Context(), "bad")
	suite.Error(err, "rejected items must not reach the data store")

	apitest.New().Handler(suite.handler).Post("/api/item").Header("X-API-Key", apiKey).
		JSON(Item{ItemId: "bad", Labels: map[string]any{"embedding": []float64{1}}}).
		Expect(suite.T()).Status(http.StatusBadRequest).End()
	apitest.New().Handler(suite.handler).Patch("/api/item/ok").Header("X-API-Key", apiKey).
		JSON(data.ItemPatch{Labels: map[string]any{"embedding": []float64{1, 2, 3}}}).
		Expect(suite.T()).Status(http.StatusBadRequest).End()
	apitest.New().Handler(suite.handler).Patch("/api/item/ok/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{"embedding": []float64{1, 2, 3}}).
		Expect(suite.T()).Status(http.StatusBadRequest).End()
	apitest.New().Handler(suite.handler).Patch("/api/item/ok/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{"embedding": []float64{1, 2}}).
		Expect(suite.T()).Status(http.StatusOK).End()
}

func (suite *ServerTestSuite) TestQueryLimitFilter() {
	suite.NoError(suite.DataClient.BatchInsertItems(suite.T().Context(), []data.Item{
		{ItemId: "a", Timestamp: time.Now()}, {ItemId: "b", Timestamp: time.Now()},
		{ItemId: "c", Timestamp: time.Now()}, {ItemId: "d", Timestamp: time.Now()},
	}))
	listItems := func(n string) int {
		request := httptest.NewRequest(http.MethodGet, "/api/items?n="+n, nil)
		request.Header.Set("X-API-Key", apiKey)
		recorder := httptest.NewRecorder()
		suite.handler.ServeHTTP(recorder, request)
		suite.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())
		var iterator ItemIterator
		suite.Require().NoError(json.Unmarshal(recorder.Body.Bytes(), &iterator))
		return len(iterator.Items)
	}
	suite.Equal(4, listItems("10"))
	suite.Config.VideoHub.MaxQueryN = 2
	suite.Equal(2, listItems("10"))
	suite.Equal(1, listItems("1"))
	suite.Config.VideoHub.MaxQueryN = 0
	suite.Equal(4, listItems("10"))
}
