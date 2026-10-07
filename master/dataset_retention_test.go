// Copyright 2026 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package master

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/gorse/common/event"
	"github.com/gorse-io/gorse/common/expression"
	"github.com/gorse-io/gorse/common/floats"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/dataset"
	"github.com/gorse-io/gorse/logics"
	"github.com/gorse-io/gorse/model/ctr"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/require"
)

var retentionTimestamp = time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)

// Each streamed item owns independently generated decoded-JSON numeric arrays.
// The fake holds no item fixture after a batch is consumed.
func retentionItem(i, dim int, malformed bool) data.Item {
	size := dim
	if malformed && i%7 == 0 {
		size--
	}
	embedding := make([]any, size)
	for j := range embedding {
		embedding[j] = float64((i+j)%2001-1000) / 1024
	}
	return data.Item{
		ItemId: fmt.Sprintf("item-%06d", i), IsHidden: i%11 == 0,
		Categories: []string{fmt.Sprintf("category-%d", i%3)}, Timestamp: retentionTimestamp,
		Labels:  map[string]any{"profile": map[string]any{"embedding": embedding, "tag": "shared", "weight": json.Number("2.5"), "mixed": []any{float64(1), "unsupported"}}},
		Comment: "preserved comment",
	}
}

type retentionStreamStore struct {
	data.Database
	n, dim                int
	malformed, noFeedback bool
	phase, release        chan struct{}
	once                  sync.Once
	mu                    sync.Mutex
	scans                 []data.ScanOptions
	itemLimit             *time.Time
}

func (d *retentionStreamStore) CountUsers(context.Context) (int, error) { return 2, nil }
func (d *retentionStreamStore) CountItems(context.Context) (int, error) { return d.n, nil }
func (d *retentionStreamStore) CountFeedback(context.Context) (int, error) {
	return len(d.feedback()), nil
}
func retentionStream[T any](ctx context.Context, values []T) (chan []T, chan error) {
	out, errs := make(chan []T), make(chan error, 1)
	go func() {
		defer close(out)
		defer close(errs)
		if len(values) > 0 {
			select {
			case out <- values:
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			}
		}
		errs <- nil
	}()
	return out, errs
}
func (d *retentionStreamStore) GetUserStream(ctx context.Context, _ int) (chan []data.User, chan error) {
	return retentionStream(ctx, []data.User{{UserId: "u0", Labels: map[string]any{"tag": "shared"}}, {UserId: "u1", Labels: map[string]any{"tag": "shared"}}})
}
func (d *retentionStreamStore) GetItemStream(ctx context.Context, size int, limit *time.Time) (chan []data.Item, chan error) {
	d.itemLimit = limit
	out, errs := make(chan []data.Item), make(chan error, 1)
	go func() {
		defer close(out)
		defer close(errs)
		for start := 0; start < d.n; start += size {
			batch := make([]data.Item, 0, min(size, d.n-start))
			for i := start; i < min(start+size, d.n); i++ {
				batch = append(batch, retentionItem(i, d.dim, d.malformed))
			}
			select {
			case out <- batch:
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			}
		}
		errs <- nil
	}()
	return out, errs
}
func (d *retentionStreamStore) feedback() []data.Feedback {
	if d.noFeedback {
		return nil
	}
	var result []data.Feedback
	for _, f := range []struct {
		kind, user string
		item       int
	}{
		{"positive", "u0", 1}, {"positive", "u0", 1}, {"negative", "u0", 1}, // negative takes precedence
		{"positive", "u0", 2}, {"positive", "u1", 3}, {"read", "u0", 2}, {"read", "u1", 4}, {"negative", "u1", 5},
		{"positive", "missing", 6}, {"read", "u0", 999999},
	} {
		result = append(result, data.Feedback{FeedbackKey: data.FeedbackKey{FeedbackType: f.kind, UserId: f.user, ItemId: fmt.Sprintf("item-%06d", f.item)}, Timestamp: retentionTimestamp.Add(time.Duration(len(result)) * time.Second), Value: 1})
	}
	return result
}
func (d *retentionStreamStore) GetFeedbackStream(ctx context.Context, _ int, options ...data.ScanOption) (chan []data.Feedback, chan error) {
	var scan data.ScanOptions
	for _, option := range options {
		if option != nil {
			option(&scan)
		}
	}
	d.mu.Lock()
	d.scans = append(d.scans, scan)
	d.mu.Unlock()
	if d.phase != nil {
		d.once.Do(func() {
			close(d.phase)
			select {
			case <-d.release:
			case <-ctx.Done():
			}
		})
	}
	var result []data.Feedback
	for _, f := range d.feedback() {
		if scan.BeginItemId != nil && f.ItemId < *scan.BeginItemId {
			continue
		}
		if scan.EndItemId != nil && f.ItemId > *scan.EndItemId {
			continue
		}
		if scan.BeginTime != nil && f.Timestamp.Before(*scan.BeginTime) {
			continue
		}
		if scan.EndTime != nil && f.Timestamp.After(*scan.EndTime) {
			continue
		}
		if len(scan.FeedbackTypes) > 0 && !expression.MatchFeedbackTypeExpressions(scan.FeedbackTypes, f.FeedbackType, f.Value) {
			continue
		}
		result = append(result, f)
	}
	if scan.OrderByItemId {
		sort.SliceStable(result, func(i, j int) bool { return result[i].ItemId < result[j].ItemId })
	}
	return retentionStream(ctx, result)
}
func retentionLoader(t *testing.T, nonempty bool, n int) (*Master, []*logics.NonPersonalized) {
	cfg := config.GetDefaultConfig()
	cfg.Master.NumJobs = 1
	m := &Master{Config: cfg}
	if !nonempty {
		return m, nil
	}
	recommender, err := logics.NewNonPersonalized(config.NonPersonalizedConfig{Name: "raw", Score: "float(item.Labels.profile.embedding[0]) + len(feedback)", Filter: "item.Labels.profile.tag == 'shared'"}, n, retentionTimestamp)
	require.NoError(t, err)
	return m, []*logics.NonPersonalized{recommender}
}
func retentionTypes(kind string) []expression.FeedbackTypeExpression {
	return []expression.FeedbackTypeExpression{{FeedbackType: kind}}
}

// Canonicalize preexisting map/set iteration order while preserving feature names,
// values, FP16 bits, row identities, targets, timestamps and item stream order.
func retentionDigest(t *testing.T, ranking *dataset.Dataset, click *ctr.Dataset, snapshot event.Snapshot, scores []cache.Score) string {
	snapshot.Timestamp = time.Time{}
	type row struct {
		User, Item string
		Target     float32
		Timestamp  time.Time
	}
	rows := make([]row, click.Count())
	for i := range rows {
		rows[i] = row{click.Index.GetUsers()[click.Users[i]], click.Index.GetItems()[click.Items[i]], click.Target[i], click.Timestamps[i]}
	}
	sort.Slice(rows, func(i, j int) bool {
		a, _ := json.Marshal(rows[i])
		b, _ := json.Marshal(rows[j])
		return string(a) < string(b)
	})
	itemFeatures := make([]map[string]float32, len(click.ItemLabels))
	for i, labels := range click.ItemLabels {
		itemFeatures[i] = map[string]float32{}
		for _, v := range labels {
			itemFeatures[i][click.Index.GetItemLabels()[v.A]] = v.B
		}
	}
	userFeatures := make([]map[string]float32, len(click.UserLabels))
	for i, labels := range click.UserLabels {
		userFeatures[i] = map[string]float32{}
		for _, v := range labels {
			userFeatures[i][click.Index.GetUserLabels()[v.A]] = v.B
		}
	}
	embeddings := make([]map[string][]uint16, len(click.ItemEmbeddings))
	dims := map[string]int{}
	for i, dim := range click.ItemEmbeddingDimension {
		dims[click.ItemEmbeddingIndex.ToName(int32(i))] = dim
	}
	for i, values := range click.ItemEmbeddings {
		embeddings[i] = map[string][]uint16{}
		for j, v := range values {
			embeddings[i][click.ItemEmbeddingIndex.ToName(int32(j))] = v
		}
	}
	for i := range scores {
		sort.Strings(scores[i].Categories)
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].Id < scores[j].Id })
	result := struct {
		Items                      []data.Item
		Users                      []data.User
		UserFeedback, ItemFeedback [][]int32
		Rows                       []row
		ItemFeatures, UserFeatures []map[string]float32
		Embeddings                 []map[string][]uint16
		Dimensions                 map[string]int
		Snapshot                   event.Snapshot
		Scores                     []cache.Score
	}{ranking.GetItems(), ranking.GetUsers(), ranking.GetUserFeedback(), ranking.GetItemFeedback(), rows, itemFeatures, userFeatures, embeddings, dims, snapshot, scores}
	encoded, err := json.Marshal(result)
	require.NoError(t, err)
	return fmt.Sprintf("%x", sha256.Sum256(encoded))
}

func TestDatasetRetentionLoader(t *testing.T) {
	for _, nonempty := range []bool{false, true} {
		for _, noFeedback := range []bool{false, true} {
			t.Run(fmt.Sprintf("nonpersonalized=%v/no_feedback=%v", nonempty, noFeedback), func(t *testing.T) {
				store := &retentionStreamStore{n: 12, dim: 4, malformed: true, noFeedback: noFeedback}
				m, recommenders := retentionLoader(t, nonempty, store.n)
				click, ranking, snapshot, err := m.LoadDataFromDatabase(t.Context(), store, retentionTypes("positive"), retentionTypes("negative"), retentionTypes("read"), 0, 0, NewOnlineEvaluator(nil, nil), recommenders)
				require.NoError(t, err)
				require.Equal(t, 12, ranking.CountItems())
				require.Equal(t, 2, ranking.CountUsers())
				var raw []data.Item
				for i := 0; i < store.n; i++ {
					item := retentionItem(i, store.dim, true)
					raw = append(raw, item)
					got := ranking.GetItems()[i]
					require.Equal(t, item.ItemId, got.ItemId)
					require.Equal(t, item.IsHidden, got.IsHidden)
					require.Equal(t, item.Categories, got.Categories)
					require.Equal(t, item.Timestamp, got.Timestamp)
					require.Equal(t, item.Comment, got.Comment)
					profile := got.Labels.(map[string]any)["profile"].(map[string]any)
					original := item.Labels.(map[string]any)["profile"].(map[string]any)
					expected, ok := floats.FromAny(original["embedding"])
					require.True(t, ok)
					require.Equal(t, expected, profile["embedding"])
					idx := click.ItemEmbeddingIndex.ToNumber("profile.embedding")
					require.Equal(t, 4, click.ItemEmbeddingDimension[idx])
					if i%7 == 0 {
						require.Nil(t, click.ItemEmbeddings[i][idx])
					} else {
						require.Equal(t, expected, click.ItemEmbeddings[i][idx])
					}
					features := map[string]float32{}
					for _, v := range click.ItemLabels[i] {
						features[click.Index.GetItemLabels()[v.A]] = v.B
					}
					require.Equal(t, map[string]float32{"profile.tag.shared": 1, "profile.weight.": 2.5}, features)
				}
				require.Equal(t, int64(12), snapshot.ItemCount)
				require.Equal(t, deepSize(raw), snapshot.ItemBytes)
				require.Equal(t, int64(2), snapshot.UserCount)
				require.False(t, snapshot.Timestamp.IsZero())
				require.Len(t, store.scans, 3)
				require.True(t, store.scans[0].OrderByItemId)
				require.True(t, store.scans[1].OrderByItemId)
				require.False(t, store.scans[2].OrderByItemId)
				var scores []cache.Score
				if nonempty {
					scores = recommenders[0].PopAll()
					require.Len(t, scores, 10)
					for _, score := range scores {
						require.NotEqual(t, "item-000000", score.Id)
						require.NotEqual(t, "item-000011", score.Id)
					}
				}
				if noFeedback {
					require.Zero(t, click.Count())
					require.Zero(t, ranking.CountFeedback())
					require.Zero(t, snapshot.FeedbackCount)
				} else {
					require.Equal(t, 2, click.PositiveCount)
					require.Equal(t, 3, click.NegativeCount)
					require.Equal(t, 2, ranking.CountFeedback())
					require.Equal(t, int64(9), snapshot.FeedbackCount)
				}
				t.Logf("RETENTION_CORRECTNESS nonpersonalized=%v no_feedback=%v sha256=%s snapshot=%+v", nonempty, noFeedback, retentionDigest(t, ranking, click, snapshot, scores), snapshot)
			})
		}
	}
}
