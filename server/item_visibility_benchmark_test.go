package server

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/require"
)

type visibilityMaterializationCounter struct {
	data.Database
	rows   atomic.Int64
	labels atomic.Int64
}

func (db *visibilityMaterializationCounter) BatchGetItems(ctx context.Context, ids []string, opts data.GetOptions) ([]data.Item, error) {
	items, err := db.Database.BatchGetItems(ctx, ids, opts)
	db.rows.Add(int64(len(items)))
	var labels int64
	for _, item := range items {
		if item.Labels != nil {
			labels++
		}
	}
	db.labels.Add(labels)
	return items, err
}

// Compare the retained pre-change helper with the production helper in the same
// process/database. One operation is one lookup or six concurrent source lookups.
// Synthetic labels include a 1536-dimensional embedding; no production data is used.
func BenchmarkItemVisibility(b *testing.B) {
	for _, backend := range []string{"sqlite", "mysql"} {
		b.Run(backend, func(b *testing.B) {
			for _, dimensions := range []int{0, 1536} {
				b.Run(fmt.Sprintf("dimensions=%d", dimensions), func(b *testing.B) {
					embedding := make([]float32, dimensions)
					for i := range embedding {
						embedding[i] = float32(math.Sin(float64(i)*.37) / 40)
					}
					labels := map[string]any{"embedding": embedding, "language": "en", "normalizedLabels": []string{"language:en", "topic:gaming", "format:vod"}}
					encodedLabels, err := json.Marshal(labels)
					require.NoError(b, err)
					items := make([]data.Item, 360)
					for i := range items {
						categories := []string{"language:en", "topic:gaming"}
						if i%5 == 0 {
							categories = []string{"language:fr", "topic:gaming"}
						}
						items[i] = data.Item{ItemId: fmt.Sprintf("item-%04d", i), IsHidden: i%11 == 0, Categories: categories,
							Labels: labels, Comment: "Synthetic recommendation item for visibility benchmark"}
					}
					db := visibilityDatabase(b, backend, items)
					categories := []string{"language:en", "topic:gaming"}
					pools := make([][]cache.Score, 6)
					for source := range pools {
						for i := 0; i < 120; i++ {
							pools[source] = append(pools[source], cache.Score{Id: items[source*36+i].ItemId, Score: float64(120-i) / 120})
						}
					}
					for _, workers := range []int{1, 6} {
						b.Run(fmt.Sprintf("sources=%d", workers), func(b *testing.B) {
							wantCounts := make([]int, workers)
							for source := 0; source < workers; source++ {
								before, err := filterVisibleItemsFullMetadata(b.Context(), db, pools[source], categories)
								require.NoError(b, err)
								after, err := FilterVisibleItemsByCategories(b.Context(), db, pools[source], categories)
								require.NoError(b, err)
								require.Equal(b, before, after, "complete ordered score equality before timing")
								wantCounts[source] = len(before)
							}
							methods := []struct {
								name   string
								filter func(context.Context, data.Database, []cache.Score, []string) ([]cache.Score, error)
							}{{"full-metadata", filterVisibleItemsFullMetadata}, {"filtered-ids", FilterVisibleItemsByCategories}}
							if os.Getenv("GORSE_VISIBILITY_BENCH_ORDER") == "filtered-first" {
								methods[0], methods[1] = methods[1], methods[0]
							}
							for _, method := range methods {
								b.Run(method.name, func(b *testing.B) {
									measured := &visibilityMaterializationCounter{Database: db}
									errs := make([]error, workers)
									query := func(source int) {
										got, err := method.filter(b.Context(), measured, pools[source], categories)
										if err == nil && len(got) != wantCounts[source] {
											err = fmt.Errorf("source %d returned %d items, expected %d", source, len(got), wantCounts[source])
										}
										errs[source] = err
									}
									b.ReportAllocs()
									b.ResetTimer()
									for i := 0; i < b.N; i++ {
										if workers == 1 {
											query(0)
										} else {
											var work sync.WaitGroup
											for source := 0; source < workers; source++ {
												work.Add(1)
												go func(source int) { defer work.Done(); query(source) }(source)
											}
											work.Wait()
										}
										for _, err := range errs {
											if err != nil {
												b.Fatal(err)
											}
										}
									}
									b.StopTimer()
									b.ReportMetric(float64(measured.rows.Load())/float64(b.N), "rows/op")
									b.ReportMetric(float64(measured.labels.Load())/float64(b.N), "labels/op")
									// Serialized fixture label bytes represented by materialized labels;
									// excludes protocol overhead and is not a wire-traffic measurement.
									b.ReportMetric(float64(measured.labels.Load()*int64(len(encodedLabels)))/float64(b.N), "label_B/op")
								})
							}
						})
					}
				})
			}
		})
	}
}
