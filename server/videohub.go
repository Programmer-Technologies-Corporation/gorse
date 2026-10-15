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

// VideoHub fork: REST extensions controlled by the [videohub] config section.
//
//   - incremental_item_to_item: index embedding vectors on item writes and
//     answer neighbor queries from stored embeddings for unindexed items.
//   - label_patch: PATCH /api/item/{item-id}/labels merges label keys so a
//     vector can be replaced without resending the whole item.
//   - embedding_dimensions: reject embeddings with an unexpected length at the
//     API boundary instead of letting them reach the models.
//   - max_query_n: cap the "n" query parameter of every endpoint.

package server

import (
	"context"
	"hash/fnv"
	"net/http"
	"strconv"
	"sync"
	"time"

	restfulspec "github.com/emicklei/go-restful-openapi/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/gorse-io/gorse/common/log"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/logics"
	"github.com/gorse-io/gorse/storage"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/pkg/errors"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

var (
	ItemVectorIndexTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "server",
		Name:      "item_vector_index_total",
		Help:      "Incremental item-to-item vector writes by recommender and result.",
	}, []string{"recommender", "result"})
	ItemToItemFallbackTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "server",
		Name:      "item_to_item_fallback_total",
		Help:      "Item-to-item queries answered from the item's stored embedding because no vector was indexed.",
	}, []string{"recommender", "result"})
	LabelPatchTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "server",
		Name:      "label_patch_total",
		Help:      "Label merge patches by result.",
	}, []string{"result"})
	RejectedEmbeddingsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "server",
		Name:      "rejected_embeddings_total",
		Help:      "Item writes rejected because an embedding column had an unexpected dimension.",
	}, []string{"recommender"})
	QueryNClampedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "server",
		Name:      "query_n_clamped_total",
		Help:      "Requests whose n parameter was reduced to [videohub].max_query_n.",
	})
)

// itemLocks serialize read-merge-write label patches per item. A fixed set of
// stripes bounds memory regardless of the number of items.
var itemLocks [256]sync.Mutex

func itemLock(itemId string) *sync.Mutex {
	h := fnv.New32a()
	_, _ = h.Write([]byte(itemId))
	return &itemLocks[h.Sum32()%uint32(len(itemLocks))]
}

func (s *RestServer) vectorConfig() vectors.VectorConfig {
	return vectors.VectorConfig{
		Type: vectors.QuantizationType(s.Config.Database.Vector.QuantizationType),
		Bits: s.Config.Database.Vector.QuantizationBits,
	}
}

func (s *RestServer) embeddingItemToItemConfigs() []config.ItemToItemConfig {
	configs := make([]config.ItemToItemConfig, 0, len(s.Config.Recommend.ItemToItem))
	for _, cfg := range s.Config.Recommend.ItemToItem {
		if cfg.Type == "embedding" {
			configs = append(configs, cfg)
		}
	}
	return configs
}

// createVideoHubRoutes registers the fork's endpoints. Routes are always
// registered so that a config reload can enable them; disabled features
// answer 404.
func (s *RestServer) createVideoHubRoutes(ws *restful.WebService) {
	ws.Route(ws.PATCH("/item/{item-id}/labels").To(s.patchItemLabels).
		Doc("Merge label keys into an item. Keys set to null are removed. Requires [videohub].label_patch.").
		Metadata(restfulspec.KeyOpenAPITags, []string{ItemsAPITag}).
		Param(ws.HeaderParameter("X-API-Key", "API key").DataType("string")).
		Param(ws.PathParameter("item-id", "Item ID").DataType("string")).
		Reads(map[string]any{}).
		Returns(http.StatusOK, "OK", Success{}).
		Writes(Success{}))
}

// QueryLimitFilter caps the n query parameter to [videohub].max_query_n.
func (s *RestServer) QueryLimitFilter(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	if limit := s.Config.VideoHub.MaxQueryN; limit > 0 {
		query := req.Request.URL.Query()
		if raw := query.Get("n"); raw != "" {
			if n, err := strconv.Atoi(raw); err == nil && n > limit {
				query.Set("n", strconv.Itoa(limit))
				req.Request.URL.RawQuery = query.Encode()
				QueryNClampedTotal.Inc()
			}
		}
	}
	chain.ProcessFilter(req, resp)
}

// validateItemEmbeddings enforces [videohub].embedding_dimensions on the
// embedding columns of all embedding-based item-to-item recommenders.
func (s *RestServer) validateItemEmbeddings(items []data.Item) error {
	expected := s.Config.VideoHub.EmbeddingDimensions
	if expected <= 0 {
		return nil
	}
	for _, cfg := range s.embeddingItemToItemConfigs() {
		for i := range items {
			embedding, present, err := logics.ItemEmbedding(cfg, &items[i])
			if err != nil {
				RejectedEmbeddingsTotal.WithLabelValues(cfg.Name).Inc()
				return err
			}
			if present && len(embedding) != expected {
				RejectedEmbeddingsTotal.WithLabelValues(cfg.Name).Inc()
				return errors.Errorf("item %s: column %s has %d dimensions, expected %d",
					items[i].ItemId, cfg.Column, len(embedding), expected)
			}
		}
	}
	return nil
}

// indexItemVectors writes the embedding vectors of items into the item-to-item
// collections when [videohub].incremental_item_to_item is enabled. Failures are
// logged and counted but never fail the request: the item is already persisted
// and the next master job re-indexes it.
func (s *RestServer) indexItemVectors(ctx context.Context, items []data.Item) {
	if !s.Config.VideoHub.IncrementalItemToItem {
		return
	}
	now := time.Now()
	for _, cfg := range s.embeddingItemToItemConfigs() {
		for i := range items {
			indexed, err := logics.IndexItemVector(ctx, s.VectorClient, cfg, s.vectorConfig(), &items[i], now)
			switch {
			case errors.Is(err, logics.ErrEmbeddingDimensionMismatch):
				ItemVectorIndexTotal.WithLabelValues(cfg.Name, "dimension_mismatch").Inc()
				log.Logger().Warn("skip indexing item vector",
					zap.String("item_id", items[i].ItemId), zap.String("recommender", cfg.Name), zap.Error(err))
			case err != nil:
				ItemVectorIndexTotal.WithLabelValues(cfg.Name, "error").Inc()
				log.Logger().Error("failed to index item vector",
					zap.String("item_id", items[i].ItemId), zap.String("recommender", cfg.Name), zap.Error(err))
			case indexed:
				ItemVectorIndexTotal.WithLabelValues(cfg.Name, "indexed").Inc()
			default:
				ItemVectorIndexTotal.WithLabelValues(cfg.Name, "skipped").Inc()
			}
		}
	}
}

// reindexItem reloads an item after a patch and indexes its vector so that
// hidden/category changes and new embeddings are visible to neighbor queries.
func (s *RestServer) reindexItem(ctx context.Context, itemId string, patch data.ItemPatch) {
	if !s.Config.VideoHub.IncrementalItemToItem {
		return
	}
	if patch.Labels == nil && patch.IsHidden == nil && patch.Categories == nil {
		return
	}
	item, err := s.DataClient.GetItem(ctx, itemId)
	if err != nil {
		log.Logger().Error("failed to load item for re-indexing", zap.String("item_id", itemId), zap.Error(err))
		return
	}
	s.indexItemVectors(ctx, []data.Item{item})
}

// queryItemToItem wraps logics.QueryItemToItem with the stored-embedding
// fallback. A failing fallback degrades to "no neighbors" because the primary
// path already answered that nothing is indexed.
func (s *RestServer) queryItemToItem(ctx context.Context, cfg config.ItemToItemConfig, itemId string, n int) ([]cache.Score, error) {
	scores, usedFallback, err := logics.QueryItemToItemWithFallback(ctx, s.VectorClient, s.DataClient, cfg, itemId, nil, n,
		s.Config.VideoHub.IncrementalItemToItem)
	if !usedFallback {
		return scores, err
	}
	switch {
	case err != nil:
		ItemToItemFallbackTotal.WithLabelValues(cfg.Name, "error").Inc()
		log.Logger().Warn("item-to-item fallback failed",
			zap.String("item_id", itemId), zap.String("recommender", cfg.Name), zap.Error(err))
		return nil, nil
	case len(scores) == 0:
		ItemToItemFallbackTotal.WithLabelValues(cfg.Name, "miss").Inc()
	default:
		ItemToItemFallbackTotal.WithLabelValues(cfg.Name, "hit").Inc()
	}
	return scores, nil
}

func (s *RestServer) patchItemLabels(request *restful.Request, response *restful.Response) {
	ctx := context.Background()
	if request != nil && request.Request != nil {
		ctx = request.Request.Context()
	}
	if !s.Config.VideoHub.LabelPatch {
		PageNotFound(response, errors.New("label patch is disabled by [videohub].label_patch"))
		return
	}
	itemId := request.PathParameter("item-id")
	var patch map[string]any
	if err := request.ReadEntity(&patch); err != nil {
		LabelPatchTotal.WithLabelValues("rejected").Inc()
		BadRequest(response, err)
		return
	}
	if len(patch) == 0 {
		LabelPatchTotal.WithLabelValues("rejected").Inc()
		BadRequest(response, errors.New("label patch must be a non-empty JSON object"))
		return
	}
	if err := data.ValidateLabels(patch); err != nil {
		LabelPatchTotal.WithLabelValues("rejected").Inc()
		BadRequest(response, err)
		return
	}

	lock := itemLock(itemId)
	lock.Lock()
	defer lock.Unlock()

	item, err := s.DataClient.GetItem(ctx, itemId)
	if err != nil {
		if errors.Is(err, storage.ErrNotFound) {
			LabelPatchTotal.WithLabelValues("not_found").Inc()
			PageNotFound(response, err)
		} else {
			LabelPatchTotal.WithLabelValues("error").Inc()
			InternalServerError(response, err)
		}
		return
	}
	merged, err := mergeLabels(item.Labels, patch)
	if err != nil {
		LabelPatchTotal.WithLabelValues("rejected").Inc()
		BadRequest(response, err)
		return
	}
	if err = s.checkLabelsSize(merged); err != nil {
		LabelPatchTotal.WithLabelValues("rejected").Inc()
		TooManyRequests(response, err)
		return
	}
	item.Labels = merged
	if err = s.validateItemEmbeddings([]data.Item{item}); err != nil {
		LabelPatchTotal.WithLabelValues("rejected").Inc()
		BadRequest(response, err)
		return
	}
	// Split-count storage merges against the locked current row. Reusing the
	// earlier read here would overwrite a newer count during an embedding patch.
	handled := false
	validationStatus := 0
	if patcher, ok := s.DataClient.(data.AtomicItemLabelPatcher); ok {
		var current data.Item
		current, handled, err = patcher.PatchItemLabelsAtomic(ctx, itemId, patch, func(actual data.Item) error {
			if validationError := s.checkLabelsSize(actual.Labels); validationError != nil {
				validationStatus = 429
				return validationError
			}
			if validationError := s.validateItemEmbeddings([]data.Item{actual}); validationError != nil {
				validationStatus = 400
				return validationError
			}
			return nil
		})
		if handled && err == nil {
			item = current
		}
	}
	if !handled {
		err = s.DataClient.ModifyItem(ctx, itemId, data.ItemPatch{Labels: merged})
	}
	if err != nil {
		if errors.Is(err, storage.ErrNotFound) {
			LabelPatchTotal.WithLabelValues("not_found").Inc()
			PageNotFound(response, err)
		} else if validationStatus == 429 {
			LabelPatchTotal.WithLabelValues("rejected").Inc()
			TooManyRequests(response, err)
		} else if validationStatus == 400 {
			LabelPatchTotal.WithLabelValues("rejected").Inc()
			BadRequest(response, err)
		} else {
			LabelPatchTotal.WithLabelValues("error").Inc()
			InternalServerError(response, err)
		}
		return
	}
	if err = s.CacheClient.Set(ctx, cache.Time(cache.Key(cache.LastModifyItemTime, itemId), time.Now())); err != nil {
		LabelPatchTotal.WithLabelValues("error").Inc()
		InternalServerError(response, err)
		return
	}
	s.indexItemVectors(ctx, []data.Item{item})
	LabelPatchTotal.WithLabelValues("ok").Inc()
	Ok(response, Success{RowAffected: 1})
}

// mergeLabels applies a top-level key merge: keys in patch replace keys in the
// existing label object and null values delete keys. Items whose labels are not
// an object (legacy string arrays) cannot be merged.
func mergeLabels(existing any, patch map[string]any) (map[string]any, error) {
	merged := make(map[string]any)
	switch labels := existing.(type) {
	case nil:
	case map[string]any:
		for key, value := range labels {
			merged[key] = value
		}
	default:
		return nil, errors.New("item labels are not a JSON object and cannot be merged")
	}
	for key, value := range patch {
		if value == nil {
			delete(merged, key)
		} else {
			merged[key] = value
		}
	}
	return merged, nil
}
