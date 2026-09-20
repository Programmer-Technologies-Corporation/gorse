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

// VideoHub fork: incremental item-to-item availability.
//
// Upstream only fills the item-to-item vector collections from the periodic
// master job, so a freshly inserted item has no neighbors until the next run.
// The helpers in this file let the REST layer index an item's embedding as soon
// as it is written and answer neighbor queries from the stored embedding when
// the item has not been indexed yet. Only embedding-based recommenders are
// supported: tags/users/auto vectors depend on dataset-wide IDF statistics that
// are only known to the master job.

package logics

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"time"

	"github.com/expr-lang/expr"
	"github.com/expr-lang/expr/vm"
	"github.com/gorse-io/gorse/common/floats"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/storage"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/pkg/errors"
)

// ErrEmbeddingDimensionMismatch is returned when an item embedding does not
// match the dimension of an existing item-to-item collection.
var ErrEmbeddingDimensionMismatch = errors.New("embedding dimension mismatch")

var itemColumnPrograms sync.Map // column expression -> *vm.Program

func compileItemColumn(column string) (*vm.Program, error) {
	if program, ok := itemColumnPrograms.Load(column); ok {
		return program.(*vm.Program), nil
	}
	program, err := expr.Compile(column, expr.Env(map[string]any{"item": data.Item{}}))
	if err != nil {
		return nil, errors.WithStack(err)
	}
	itemColumnPrograms.Store(column, program)
	return program, nil
}

// ItemEmbedding evaluates the embedding column of an embedding-based
// item-to-item recommender for one item. present reports whether the column
// holds a value; err reports a value that is present but is not a numeric
// vector. Values are rounded through FP16 exactly like the master job so the
// incrementally indexed vector equals the one the periodic job would write.
func ItemEmbedding(cfg config.ItemToItemConfig, item *data.Item) (embedding []float32, present bool, err error) {
	if cfg.Type != "embedding" {
		return nil, false, nil
	}
	program, err := compileItemColumn(cfg.Column)
	if err != nil {
		return nil, false, err
	}
	if item.Labels == nil {
		return nil, false, nil
	}
	result, err := expr.Run(program, map[string]any{"item": item})
	if err != nil || result == nil {
		// The column cannot be resolved for this label shape (missing key,
		// legacy string-array labels): the item simply has no embedding.
		return nil, false, nil
	}
	values, ok := toFloat32Slice(result)
	if !ok {
		return nil, true, errors.Errorf("column %s of item %s is not a numeric vector", cfg.Column, item.ItemId)
	}
	if len(values) == 0 {
		return nil, false, nil
	}
	return floats.ToFloat32(floats.FromFloat32(values)), true, nil
}

// toFloat32Slice converts the label representations produced by the REST
// layer (json.Number elements), by the data stores (float64 elements) and by
// Go callers ([]float32, []float64, FP16 []uint16) into a float32 vector.
func toFloat32Slice(v any) ([]float32, bool) {
	switch typed := v.(type) {
	case []float32:
		return typed, true
	case []float64:
		values := make([]float32, len(typed))
		for i, value := range typed {
			values[i] = float32(value)
		}
		return values, true
	case []uint16:
		return floats.ToFloat32(typed), true
	case []any:
		values := make([]float32, len(typed))
		for i, element := range typed {
			if number, ok := element.(json.Number); ok {
				f, err := number.Float64()
				if err != nil {
					return nil, false
				}
				values[i] = float32(f)
				continue
			}
			converted, ok := floats.FromAny([]any{element})
			if !ok || len(converted) != 1 {
				return nil, false
			}
			values[i] = floats.ToFloat32(converted)[0]
		}
		return values, true
	}
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Slice {
		return nil, false
	}
	converted, ok := floats.FromAny(v)
	if !ok {
		return nil, false
	}
	return floats.ToFloat32(converted), true
}

// IndexItemVector upserts the embedding-based item-to-item vector of an item.
// It creates the collection on first use and refuses vectors whose dimension
// differs from an existing collection with ErrEmbeddingDimensionMismatch. The
// returned flag reports whether a vector was written; items without an
// embedding and non-embedding recommenders are skipped without error.
func IndexItemVector(
	ctx context.Context,
	client vectors.Database,
	cfg config.ItemToItemConfig,
	vectorConfig vectors.VectorConfig,
	item *data.Item,
	timestamp time.Time,
) (bool, error) {
	embedding, present, err := ItemEmbedding(cfg, item)
	if err != nil || !present {
		return false, err
	}
	collection := vectors.ItemToItemCollection(cfg.Name)
	if err = ensureDenseCollection(ctx, client, collection, len(embedding), vectorConfig); err != nil {
		return false, err
	}
	if err = client.AddVectors(ctx, collection, []vectors.Vector{{
		Id:         item.ItemId,
		Values:     embedding,
		IsHidden:   item.IsHidden,
		Categories: item.Categories,
		Timestamp:  timestamp,
	}}); err != nil {
		return false, errors.WithStack(err)
	}
	return true, nil
}

func ensureDenseCollection(ctx context.Context, client vectors.Database, collection string, dimension int, vectorConfig vectors.VectorConfig) error {
	checkDimension := func(info *vectors.CollectionInfo) error {
		if info.Dimension != dimension {
			return errors.Wrapf(ErrEmbeddingDimensionMismatch,
				"collection %s expects %d dimensions, got %d", collection, info.Dimension, dimension)
		}
		return nil
	}
	info, err := client.DescribeCollection(ctx, collection)
	if err == nil {
		return checkDimension(info)
	}
	if !errors.Is(err, storage.ErrNotFound) {
		return errors.WithStack(err)
	}
	if err = client.AddCollection(ctx, collection, dimension, vectors.Euclidean, vectorConfig); err != nil {
		// A concurrent writer may have created the collection first.
		if info, describeErr := client.DescribeCollection(ctx, collection); describeErr == nil {
			return checkDimension(info)
		}
		return errors.WithStack(err)
	}
	return nil
}

// QueryItemToItemWithFallback returns the indexed neighbors of an item. When
// nothing is indexed for the item, fallback is enabled and the recommender is
// embedding-based, the item's stored embedding is used as the query vector so
// that newly ingested content has neighbors before the next master job. The
// second result reports whether the fallback path produced the scores.
func QueryItemToItemWithFallback(
	ctx context.Context,
	client vectors.Database,
	dataClient data.Database,
	cfg config.ItemToItemConfig,
	itemId string,
	categories []string,
	n int,
	fallback bool,
) ([]cache.Score, bool, error) {
	scores, err := QueryItemToItem(ctx, client, cfg, itemId, categories, n)
	if fallback && errors.Is(err, storage.ErrNotFound) {
		// The collection does not exist yet (no master job has run): treat it
		// as empty instead of failing the request.
		scores, err = nil, nil
	}
	if err != nil || len(scores) > 0 || !fallback || cfg.Type != "embedding" {
		return scores, false, err
	}
	item, err := dataClient.GetItem(ctx, itemId)
	if err != nil {
		if errors.Is(err, storage.ErrNotFound) {
			return nil, true, nil
		}
		return nil, true, errors.WithStack(err)
	}
	embedding, present, err := ItemEmbedding(cfg, &item)
	if err != nil || !present {
		return nil, true, err
	}
	neighbors, err := client.QueryVectors(ctx, vectors.ItemToItemCollection(cfg.Name), vectors.Vector{Values: embedding}, categories, n+1)
	if err != nil {
		if errors.Is(err, storage.ErrNotFound) {
			return nil, true, nil
		}
		return nil, true, errors.WithStack(err)
	}
	scores = make([]cache.Score, 0, min(n, len(neighbors)))
	for _, neighbor := range neighbors {
		if neighbor.Id == itemId {
			continue
		}
		// Mirrors QueryItemToItem: Euclidean stores report negated distances.
		scores = append(scores, cache.Score{
			Id:         neighbor.Id,
			Score:      1 / (1 - float64(neighbor.Score)),
			Categories: neighbor.Categories,
		})
		if len(scores) == n {
			break
		}
	}
	return scores, true, nil
}
