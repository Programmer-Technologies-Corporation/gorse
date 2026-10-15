package data_test

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/model/ctr"
	"github.com/gorse-io/gorse/storage"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/require"
)

func countFixture(t *testing.T) (data.Database, data.Database, *sql.DB, string) {
	t.Helper()
	uri := os.Getenv("TWITCH_COUNT_MYSQL_URI")
	if uri == "" {
		t.Skip("TWITCH_COUNT_MYSQL_URI is not set")
	}
	prefix := fmt.Sprintf("vc_%d_", time.Now().UnixNano())
	legacy, err := data.Open(uri, prefix, storage.WithIsolationLevel("READ-COMMITTED"))
	require.NoError(t, err)
	require.NoError(t, legacy.Init())
	modern, err := data.Open(uri, prefix, storage.WithIsolationLevel("READ-COMMITTED"), storage.WithSeparateTwitchViewCounts(true))
	require.NoError(t, err)
	client, err := sql.Open("mysql", uri[len(storage.MySQLPrefix):])
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, modern.Purge())
		require.NoError(t, modern.Close())
		require.NoError(t, legacy.Close())
		require.NoError(t, client.Close())
	})
	return legacy, modern, client, prefix
}

func liveItem(id string, count int64) data.Item {
	return data.Item{ItemId: id, Categories: []string{"format:live"}, Timestamp: time.Date(2026, 10, 4, 12, 0, 0, 0, time.UTC), Comment: "synthetic stream",
		Labels: map[string]any{"platform": "twitch", "externalKind": "Live", "viewCount": count, "embedding": []float64{.25, .5}, "creatorName": "creator"}}
}

func viewCount(t *testing.T, item data.Item) float64 {
	t.Helper()
	return item.Labels.(map[string]any)["viewCount"].(float64)
}

func sortedFeatures(labels any) []ctr.Label {
	features := ctr.ConvertLabels(labels)
	sort.Slice(features, func(i, j int) bool { return features[i].Name < features[j].Name })
	return features
}

func TestMySQLTwitchViewCountsUpgradeReadersAndHealing(t *testing.T) {
	ctx := context.Background()
	legacy, db, sqlDB, prefix := countFixture(t)
	item := liveItem("live", 8731)
	require.NoError(t, legacy.BatchInsertItems(ctx, []data.Item{item}))
	before, err := legacy.GetItem(ctx, item.ItemId)
	require.NoError(t, err)
	require.NoError(t, db.Init()) // forward creation on populated legacy schema
	old, err := db.GetItem(ctx, item.ItemId)
	require.NoError(t, err)
	require.Equal(t, before, old)
	require.NoError(t, db.BatchInsertItems(ctx, []data.Item{item}))
	after, err := db.GetItem(ctx, item.ItemId)
	require.NoError(t, err)
	require.Equal(t, before, after)
	// Preserve the actual SQL loader's FM/embedding representations, and also
	// prove the public numeric JSON label still feeds a UseNumber consumer.
	require.Equal(t, sortedFeatures(before.Labels), sortedFeatures(after.Labels))
	require.Equal(t, ctr.ConvertEmbeddings(before.Labels), ctr.ConvertEmbeddings(after.Labels))
	encoded, err := json.Marshal(after.Labels)
	require.NoError(t, err)
	var numeric any
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	require.NoError(t, decoder.Decode(&numeric))
	require.Contains(t, sortedFeatures(numeric), ctr.Label{Name: "viewCount.", Value: 8731})
	var present int
	require.NoError(t, sqlDB.QueryRow("SELECT JSON_CONTAINS_PATH(labels,'one','$.viewCount') FROM "+prefix+"items WHERE item_id='live'").Scan(&present))
	require.Zero(t, present)
	rows, err := db.BatchGetItems(ctx, []string{"live"}, data.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, float64(8731), viewCount(t, rows[0]))
	_, rows, err = db.GetItems(ctx, "", 10, nil)
	require.NoError(t, err)
	require.Equal(t, float64(8731), viewCount(t, rows[0]))
	rows, err = db.GetLatestItems(ctx, 10, nil, nil)
	require.NoError(t, err)
	require.Equal(t, float64(8731), viewCount(t, rows[0]))
	stream, errs := db.GetItemStream(ctx, 10, nil)
	var streamed []data.Item
	for batch := range stream {
		streamed = append(streamed, batch...)
	}
	require.NoError(t, <-errs)
	require.Equal(t, float64(8731), viewCount(t, streamed[0]))
	require.NoError(t, db.Reconcile(config.SearchConfig{Columns: []string{"item.Comment"}}))
	search, err := db.SearchItems(ctx, "synthetic", 10)
	require.NoError(t, err)
	require.NotEmpty(t, search)
	require.Equal(t, float64(8731), viewCount(t, search[0].Item))
	require.Error(t, db.Reconcile(config.SearchConfig{Columns: []string{"item.Labels"}}))
	hidden := true
	require.NoError(t, db.ModifyItem(ctx, "live", data.ItemPatch{IsHidden: &hidden}))
	after, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.True(t, after.IsHidden)
	require.Equal(t, float64(8731), viewCount(t, after))
	// Missing derived count and missing entire item both heal from full SQL projection.
	_, err = sqlDB.Exec("DELETE FROM " + prefix + "item_view_counts WHERE item_id='live'")
	require.NoError(t, err)
	require.NoError(t, db.BatchInsertItems(ctx, []data.Item{item}))
	after, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(8731), viewCount(t, after))
	require.NoError(t, db.DeleteItem(ctx, "live"))
	var count int
	require.NoError(t, sqlDB.QueryRow("SELECT COUNT(*) FROM "+prefix+"item_view_counts").Scan(&count))
	require.Zero(t, count)
	require.NoError(t, db.BatchInsertItems(ctx, []data.Item{item}))
	after, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(8731), viewCount(t, after))
	// Rollback preparation restores the old physical representation. Re-enabling
	// must normalize it again rather than trust the old projection fingerprint.
	_, err = sqlDB.Exec("UPDATE " + prefix + "items i JOIN " + prefix + "item_view_counts c ON i.item_id=c.item_id SET i.labels=JSON_SET(i.labels,'$.viewCount',c.view_count)")
	require.NoError(t, err)
	restored, err := legacy.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(8731), viewCount(t, restored))
	item.Labels.(map[string]any)["viewCount"] = int64(9000)
	require.NoError(t, db.BatchInsertItems(data.WithTwitchLiveProjection(ctx), []data.Item{item}))
	current, err := db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(9000), viewCount(t, current))
}

func TestMySQLTwitchViewCountsLegacyPatchAndFailedProjection(t *testing.T) {
	ctx := context.Background()
	legacy, db, sqlDB, prefix := countFixture(t)
	item := liveItem("live", 1000)
	require.NoError(t, legacy.BatchInsertItems(ctx, []data.Item{item}))
	require.NoError(t, db.Init())
	patcher := db.(data.AtomicItemLabelPatcher)
	_, handled, err := patcher.PatchItemLabelsAtomic(ctx, "live", map[string]any{"embedding": []float64{.75, .875}}, func(data.Item) error { return nil })
	require.True(t, handled)
	require.NoError(t, err)
	require.NoError(t, db.BatchInsertItems(data.WithTwitchLiveProjection(ctx), []data.Item{item}))
	current, err := db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, []any{.75, .875}, current.Labels.(map[string]any)["embedding"])

	_, err = sqlDB.Exec("ALTER TABLE " + prefix + "item_view_counts ADD CHECK (view_count < 50000)")
	require.NoError(t, err)
	item.Comment = "changed metadata"
	item.Labels.(map[string]any)["viewCount"] = int64(50000)
	require.Error(t, db.BatchInsertItems(data.WithTwitchLiveProjection(ctx), []data.Item{item}))
	current, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, "synthetic stream", current.Comment)
	require.Equal(t, float64(1000), viewCount(t, current))
	item.Labels.(map[string]any)["viewCount"] = int64(49999)
	require.NoError(t, db.BatchInsertItems(data.WithTwitchLiveProjection(ctx), []data.Item{item}))
	current, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, "changed metadata", current.Comment)
	require.Equal(t, float64(49999), viewCount(t, current))
}

func TestMySQLTwitchViewCountsOtherProviderBatchRetainsLabels(t *testing.T) {
	ctx := context.Background()
	_, db, client, prefix := countFixture(t)
	require.NoError(t, db.Init())
	item := liveItem("a", 1000)
	require.NoError(t, db.BatchInsertItems(ctx, []data.Item{item}))
	item.Labels.(map[string]any)["platform"] = "youtube"
	item.Labels.(map[string]any)["viewCount"] = int64(2000)
	other := liveItem("b", 3000)
	other.Labels.(map[string]any)["platform"] = "youtube"
	require.NoError(t, db.BatchInsertItems(ctx, []data.Item{item, other, liveItem("b", 4000)}))
	current, err := db.GetItem(ctx, "a")
	require.NoError(t, err)
	require.Equal(t, float64(2000), viewCount(t, current))
	current, err = db.GetItem(ctx, "b")
	require.NoError(t, err)
	require.Equal(t, float64(3000), viewCount(t, current), "first duplicate still wins")
	var rows int
	require.NoError(t, client.QueryRow("SELECT COUNT(*) FROM "+prefix+"item_view_counts").Scan(&rows))
	require.Zero(t, rows)
}

func TestMySQLTwitchViewCountsPreserveConcurrentLabelsAndTransitions(t *testing.T) {
	ctx := context.Background()
	_, db, _, _ := countFixture(t)
	require.NoError(t, db.Init())
	item := liveItem("live", 1000)
	require.NoError(t, db.BatchInsertItems(ctx, []data.Item{item}))
	patcher := db.(data.AtomicItemLabelPatcher)
	validate := func(data.Item) error { return nil }
	var wg sync.WaitGroup
	failures := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			next := liveItem("live", int64(1001+i))
			if err := db.BatchInsertItems(data.WithTwitchLiveProjection(ctx), []data.Item{next}); err != nil {
				failures <- err
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_, handled, err := patcher.PatchItemLabelsAtomic(ctx, "live", map[string]any{"embedding": []float64{.75, .875}, "extra": "retained"}, validate)
			if err != nil {
				failures <- err
				return
			}
			if !handled {
				failures <- fmt.Errorf("atomic path not used")
				return
			}
		}
	}()
	wg.Wait()
	close(failures)
	for err := range failures {
		require.NoError(t, err)
	}
	current, err := db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(1050), viewCount(t, current))
	labels := current.Labels.(map[string]any)
	require.Equal(t, "retained", labels["extra"])
	require.Equal(t, []any{.75, .875}, labels["embedding"])
	// A general-purpose full upsert still replaces independent label patches.
	require.NoError(t, db.BatchInsertItems(ctx, []data.Item{liveItem("live", 1050)}))
	replaced, err := db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.NotContains(t, replaced.Labels, "extra")
	require.Equal(t, []any{.25, .5}, replaced.Labels.(map[string]any)["embedding"])
	// Changed safety/lifecycle metadata forces a full projection immediately.
	item.IsHidden = true
	item.Comment = "offline"
	item.Categories = []string{"status:offline"}
	item.Labels.(map[string]any)["isLive"] = "false"
	require.NoError(t, db.BatchInsertItems(ctx, []data.Item{item}))
	current, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.True(t, current.IsHidden)
	require.Equal(t, "offline", current.Comment)
	require.Equal(t, []string{"status:offline"}, current.Categories)
	// Failed validation rolls back both representations and remains retryable.
	_, handled, err := patcher.PatchItemLabelsAtomic(ctx, "live", map[string]any{"viewCount": json.Number("2")}, func(data.Item) error { return fmt.Errorf("reject") })
	require.True(t, handled)
	require.Error(t, err)
	current, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(1000), viewCount(t, current))
	_, handled, err = patcher.PatchItemLabelsAtomic(ctx, "live", map[string]any{"viewCount": nil}, validate)
	require.True(t, handled)
	require.NoError(t, err)
	current, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.NotContains(t, current.Labels, "viewCount")
}
