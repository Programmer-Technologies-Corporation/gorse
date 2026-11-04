package data_test

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/require"
)

func youtubeItem(id string, count int64) data.Item {
	item := liveItem(id, count)
	item.Labels.(map[string]any)["platform"] = "youtube"
	item.Labels.(map[string]any)["externalKind"] = "Video"
	item.Categories = []string{"format:video"}
	return item
}

func TestMySQLYouTubeCountsUpgradeAndTransitions(t *testing.T) {
	ctx := context.Background()
	legacy, store, db, prefix := countFixture(t)
	item := youtubeItem("youtube", 1000)
	require.NoError(t, legacy.BatchInsertItems(ctx, []data.Item{item}))
	require.NoError(t, store.Init())
	check := func(expected float64, split bool) {
		t.Helper()
		got, err := store.GetItem(ctx, item.ItemId)
		require.NoError(t, err)
		require.Equal(t, expected, viewCount(t, got))
		var countRows, inline int
		require.NoError(t, db.QueryRow("SELECT COUNT(*) FROM "+prefix+"item_view_counts WHERE item_id=?", item.ItemId).Scan(&countRows))
		require.NoError(t, db.QueryRow("SELECT JSON_CONTAINS_PATH(labels,'one','$.viewCount') FROM "+prefix+"items WHERE item_id=?", item.ItemId).Scan(&inline))
		if split {
			require.Equal(t, 1, countRows)
			require.Zero(t, inline)
		} else {
			require.Zero(t, countRows)
			require.Equal(t, 1, inline)
		}
	}
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	check(1000, true)
	batch, err := store.BatchGetItems(ctx, []string{item.ItemId}, data.GetOptions{})
	require.NoError(t, err)
	require.Len(t, batch, 1)
	require.Equal(t, float64(1000), viewCount(t, batch[0]))
	_, batch, err = store.GetItems(ctx, "", 10, nil)
	require.NoError(t, err)
	require.Len(t, batch, 1)
	require.Equal(t, float64(1000), viewCount(t, batch[0]))
	batch, err = store.GetLatestItems(ctx, 10, nil, nil)
	require.NoError(t, err)
	require.Len(t, batch, 1)
	require.Equal(t, float64(1000), viewCount(t, batch[0]))
	stream, streamErrors := store.GetItemStream(ctx, 10, nil)
	for batch := range stream {
		require.Equal(t, float64(1000), viewCount(t, batch[0]))
	}
	require.NoError(t, <-streamErrors)
	item.Labels.(map[string]any)["viewCount"] = int64(2000)
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	check(2000, true)
	// Metadata-only patches must not drop the separate count.
	hidden := true
	require.NoError(t, store.ModifyItem(ctx, item.ItemId, data.ItemPatch{IsHidden: &hidden}))
	check(2000, true)
	// A full labels replacement retains the same public contract.
	item.Labels.(map[string]any)["viewCount"] = int64(3000)
	require.NoError(t, store.ModifyItem(ctx, item.ItemId, data.ItemPatch{Labels: item.Labels}))
	check(3000, true)
	patcher := store.(data.AtomicItemLabelPatcher)
	_, handled, err := patcher.PatchItemLabelsAtomic(ctx, item.ItemId, map[string]any{"viewCount": json.Number("4000")}, func(data.Item) error { return nil })
	require.True(t, handled)
	require.NoError(t, err)
	check(4000, true)
	// An unsupported provider and invalid count return to the inline form.
	item.Labels.(map[string]any)["platform"] = "twitch"
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	check(3000, false)
	item.Labels.(map[string]any)["platform"] = "youtube"
	item.Labels.(map[string]any)["viewCount"] = json.Number("1.5")
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	check(1.5, false)
	item.Labels.(map[string]any)["viewCount"] = int64(5000)
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	check(5000, true)
	// Missing count rows heal. Item deletion cascades without leaving stale state.
	_, err = db.Exec("DELETE FROM "+prefix+"item_view_counts WHERE item_id=?", item.ItemId)
	require.NoError(t, err)
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	check(5000, true)
	_, handled, err = patcher.PatchItemLabelsAtomic(ctx, item.ItemId, map[string]any{"viewCount": nil}, func(data.Item) error { return nil })
	require.True(t, handled)
	require.NoError(t, err)
	got, err := store.GetItem(ctx, item.ItemId)
	require.NoError(t, err)
	require.NotContains(t, got.Labels, "viewCount")
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	require.NoError(t, store.DeleteItem(ctx, item.ItemId))
	var remaining int
	require.NoError(t, db.QueryRow("SELECT COUNT(*) FROM "+prefix+"item_view_counts").Scan(&remaining))
	require.Zero(t, remaining)
}

func TestMySQLYouTubeCountFailureRollsBackBatchAndRetries(t *testing.T) {
	ctx := context.Background()
	_, store, db, prefix := countFixture(t)
	require.NoError(t, store.Init())
	items := []data.Item{youtubeItem("a", 10), youtubeItem("b", 20)}
	require.NoError(t, store.BatchInsertItems(ctx, items))
	_, err := db.Exec("ALTER TABLE " + prefix + "item_view_counts ADD CHECK(view_count<100)")
	require.NoError(t, err)
	items[0].Comment = "must roll back"
	items[0].Labels.(map[string]any)["viewCount"] = int64(99)
	items[1].Labels.(map[string]any)["viewCount"] = int64(100)
	require.Error(t, store.BatchInsertItems(ctx, items))
	got, err := store.GetItem(ctx, "a")
	require.NoError(t, err)
	require.Equal(t, "synthetic stream", got.Comment)
	require.Equal(t, float64(10), viewCount(t, got))
	items[1].Labels.(map[string]any)["viewCount"] = int64(98)
	require.NoError(t, store.BatchInsertItems(ctx, items))
	got, err = store.GetItem(ctx, "a")
	require.NoError(t, err)
	require.Equal(t, "must roll back", got.Comment)
	require.Equal(t, float64(99), viewCount(t, got))
}

func TestMySQLYouTubeCountAndEmbeddingPatchesAreAtomic(t *testing.T) {
	ctx := context.Background()
	_, store, _, _ := countFixture(t)
	require.NoError(t, store.Init())
	item := youtubeItem("a", 10)
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	patcher := store.(data.AtomicItemLabelPatcher)
	validate := func(data.Item) error { return nil }
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for _, countWriter := range []bool{true, false} {
		wg.Add(1)
		go func(countWriter bool) {
			defer wg.Done()
			for i := 0; i < 30; i++ {
				patch := map[string]any{"embedding": []float64{.75, .875}}
				if countWriter {
					patch = map[string]any{"viewCount": int64(11 + i)}
				}
				_, _, err := patcher.PatchItemLabelsAtomic(ctx, "a", patch, validate)
				if err != nil {
					errs <- err
					return
				}
			}
		}(countWriter)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
	got, err := store.GetItem(ctx, "a")
	require.NoError(t, err)
	require.Equal(t, float64(40), viewCount(t, got))
	require.Equal(t, []any{.75, .875}, got.Labels.(map[string]any)["embedding"])
	// A general full replacement remains a replacement, including in a marked
	// mixed batch. Only Twitch live projections may preserve independent patches.
	require.NoError(t, store.BatchInsertItems(data.WithTwitchLiveProjection(ctx), []data.Item{item, liveItem("live", 5)}))
	got, err = store.GetItem(ctx, "a")
	require.NoError(t, err)
	require.Equal(t, float64(10), viewCount(t, got))
	require.Equal(t, []any{.25, .5}, got.Labels.(map[string]any)["embedding"])
}

func TestMySQLNativeProjectionComparisonPreservesChanges(t *testing.T) {
	ctx := context.Background()
	_, store, _, _ := countFixture(t)
	require.NoError(t, store.Init())
	item := youtubeItem("exact", 10)
	item.Labels.(map[string]any)["integer"] = json.Number("9007199254740993")
	item.Labels.(map[string]any)["embedding"] = []float64{1e-7, -0.09971780329942703, .047125864773988724}
	require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item}))
	require.Equal(t, float64(1), counterDelta(t, data.ItemProjectionCommitsTotal, data.ItemProjectionBranchUnchanged, func() { require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item})) }))
	for _, change := range []func(){
		func() { item.Labels.(map[string]any)["integer"] = json.Number("9007199254740992") },
		func() {
			item.Labels.(map[string]any)["embedding"] = []float64{1e-7, -0.09971780329942703, math.Nextafter(.047125864773988724, 1)}
		},
		func() { item.Labels.(map[string]any)["platform"] = "YouTube" },
		func() { item.Categories = []string{"one", "two"} },
		func() { item.Categories = []string{"two", "one"} },
		func() { item.Comment = "SYNTHETIC STREAM" },
		func() { item.Timestamp = item.Timestamp.Add(time.Second) },
	} {
		change()
		require.Equal(t, float64(1), counterDelta(t, data.ItemProjectionCommitsTotal, data.ItemProjectionBranchChanged, func() { require.NoError(t, store.BatchInsertItems(ctx, []data.Item{item})) }))
	}
}

func TestMySQLYouTubeBulkCountWritersSerialize(t *testing.T) {
	ctx := context.Background()
	_, store, _, _ := countFixture(t)
	require.NoError(t, store.Init())
	// Exercise the larger parameterized comparison and bulk count paths together.
	items := make([]data.Item, 1000)
	for i := range items {
		items[i] = youtubeItem(fmt.Sprintf("bulk-%04d", i), 0)
	}
	require.NoError(t, store.BatchInsertItems(ctx, items))
	var wg sync.WaitGroup
	errs := make(chan error, 4)
	for writer := 1; writer <= 4; writer++ {
		wg.Add(1)
		go func(writer int) {
			defer wg.Done()
			batch := make([]data.Item, len(items))
			for i := range batch {
				// Opposite input ordering must still acquire locks consistently.
				id := i
				if writer%2 == 0 {
					id = len(items) - 1 - i
				}
				batch[i] = youtubeItem(fmt.Sprintf("bulk-%04d", id), int64(writer))
			}
			errs <- store.BatchInsertItems(ctx, batch)
		}(writer)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
	_, got, err := store.GetItems(ctx, "", 1001, nil)
	require.NoError(t, err)
	require.Len(t, got, len(items))
	last := viewCount(t, got[0])
	require.GreaterOrEqual(t, last, float64(1))
	require.LessOrEqual(t, last, float64(4))
	for _, item := range got {
		require.Equal(t, last, viewCount(t, item), "whole batches must commit together")
	}
}

func countWriteSnapshot(t *testing.T, db *sql.DB) (string, uint64, map[string]uint64) {
	t.Helper()
	var file, a, b, c string
	var pos uint64
	require.NoError(t, db.QueryRow("SHOW MASTER STATUS").Scan(&file, &pos, &a, &b, &c))
	stats := map[string]uint64{}
	rows, err := db.Query("SHOW GLOBAL STATUS WHERE Variable_name IN ('Com_insert','Com_select','Com_update','Com_delete','Innodb_rows_updated','Innodb_rows_inserted','Innodb_os_log_written','Bytes_sent','Bytes_received')")
	require.NoError(t, err)
	defer rows.Close()
	for rows.Next() {
		var key string
		var value uint64
		require.NoError(t, rows.Scan(&key, &value))
		stats[key] = value
	}
	require.NoError(t, rows.Err())
	return file, pos, stats
}

func TestMySQLYouTubeCountWriteExperiment(t *testing.T) {
	output := os.Getenv("YOUTUBE_COUNT_BENCH_OUTPUT")
	if output == "" {
		t.Skip("explicit isolated experiment not requested")
	}
	ctx := context.Background()
	_, store, db, _ := countFixture(t)
	require.NoError(t, store.Init())
	vector := make([]float64, 256)
	rng := rand.New(rand.NewSource(42))
	for i := range vector {
		vector[i] = math.Round((rng.Float64()*2-1)*1e7) / 1e7
	}
	results := []map[string]any{}
	for _, mode := range []string{"count_changed", "identical", "metadata_changed"} {
		items := make([]data.Item, 50)
		for i := range items {
			items[i] = youtubeItem(fmt.Sprintf("%s-%d", mode, i), 1000)
			labels := items[i].Labels.(map[string]any)
			labels["embedding"] = vector
			labels["description"] = strings.Repeat("synthetic ", 50)
		}
		require.NoError(t, store.BatchInsertItems(ctx, items))
		file, before, statsBefore := countWriteSnapshot(t, db)
		start := time.Now()
		for round := 0; round < 10; round++ {
			for i := range items {
				if mode != "identical" {
					items[i].Labels.(map[string]any)["viewCount"] = int64(1001 + round)
				}
				if mode == "metadata_changed" {
					items[i].Comment = fmt.Sprintf("changed-%d", round)
				}
			}
			require.NoError(t, store.BatchInsertItems(ctx, items))
		}
		seconds := time.Since(start).Seconds()
		afterFile, after, statsAfter := countWriteSnapshot(t, db)
		require.Equal(t, file, afterFile)
		delta := map[string]uint64{}
		for k, v := range statsAfter {
			delta[k] = v - statsBefore[k]
		}
		for _, expected := range items {
			got, err := store.GetItem(ctx, expected.ItemId)
			require.NoError(t, err)
			require.Equal(t, float64(expected.Labels.(map[string]any)["viewCount"].(int64)), viewCount(t, got))
			require.Equal(t, expected.Comment, got.Comment)
			require.Len(t, got.Labels.(map[string]any)["embedding"], 256)
		}
		results = append(results, map[string]any{"mode": mode, "operations": 500, "seconds": seconds, "binlog_bytes": after - before, "status_delta": delta, "verified_items": 50})
	}
	encoded, err := json.MarshalIndent(results, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(output, encoded, 0600))
	t.Log(string(encoded))
}
