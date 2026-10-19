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
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

// branchCounts snapshots a branch counter vector without prometheus/testutil.
func branchCounts(t *testing.T, vec *prometheus.CounterVec) map[string]float64 {
	t.Helper()
	out := make(map[string]float64, len(data.ViewCountBranches))
	for _, branch := range data.ViewCountBranches {
		var metric dto.Metric
		require.NoError(t, vec.WithLabelValues(branch).Write(&metric))
		out[branch] = metric.Counter.GetValue()
	}
	return out
}

// countDecisions runs fn and returns the nonzero evaluated and committed
// branch deltas it produced, so a failed transaction is visibly uncommitted.
func countDecisions(t *testing.T, fn func()) (evaluated, committed map[string]float64) {
	t.Helper()
	beforeEvaluated, beforeCommitted := branchCounts(t, data.TwitchViewCountDecisionsTotal), branchCounts(t, data.TwitchViewCountCommitsTotal)
	fn()
	evaluated, committed = map[string]float64{}, map[string]float64{}
	for branch, after := range branchCounts(t, data.TwitchViewCountDecisionsTotal) {
		if delta := after - beforeEvaluated[branch]; delta != 0 {
			evaluated[branch] = delta
		}
	}
	for branch, after := range branchCounts(t, data.TwitchViewCountCommitsTotal) {
		if delta := after - beforeCommitted[branch]; delta != 0 {
			committed[branch] = delta
		}
	}
	return evaluated, committed
}

func expectBranch(t *testing.T, branch string, fn func()) {
	t.Helper()
	evaluated, committed := countDecisions(t, fn)
	require.Equal(t, map[string]float64{branch: 1}, evaluated)
	require.Equal(t, map[string]float64{branch: 1}, committed)
}

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

// Exercise the runbook operation on owned synthetic tables with writers quiet.
// Table names are test-generated identifiers, already qualified when needed.
func copyBackViewCounts(t *testing.T, client *sql.DB, itemsTable, countsTable string) int64 {
	t.Helper()
	tx, err := client.BeginTx(context.Background(), nil)
	require.NoError(t, err)
	defer tx.Rollback()
	result, err := tx.Exec("UPDATE " + itemsTable + " i JOIN " + countsTable + " c ON i.item_id=c.item_id SET i.labels=JSON_SET(i.labels,'$.viewCount',c.view_count) WHERE JSON_TYPE(i.labels)='OBJECT' AND NOT JSON_CONTAINS_PATH(i.labels,'one','$.viewCount')")
	require.NoError(t, err)
	affected, err := result.RowsAffected()
	require.NoError(t, err)
	require.NoError(t, tx.Commit())
	return affected
}

func TestMySQLTwitchViewCountsRollbackMixedRepresentations(t *testing.T) {
	ctx := context.Background()
	legacy, modern, client, prefix := countFixture(t)
	require.NoError(t, modern.Init())
	converted := liveItem("converted", 80)
	require.NoError(t, modern.BatchInsertItems(data.WithTwitchLiveProjection(ctx), []data.Item{converted}))
	// A concurrent enrichment can own a much larger independent embedding.
	vector := make([]float64, 256)
	for i := range vector {
		vector[i] = float64(i) / 256
	}
	_, handled, err := modern.(data.AtomicItemLabelPatcher).PatchItemLabelsAtomic(ctx, "converted", map[string]any{"embedding": vector, "independent": true}, func(data.Item) error { return nil })
	require.NoError(t, err)
	require.True(t, handled)
	untouched := liveItem("legacy", 42)
	nullKey := liveItem("null-key", 0)
	nullKey.Labels.(map[string]any)["viewCount"] = nil
	zero := liveItem("zero", 0)
	missing := liveItem("missing-count", 0)
	delete(missing.Labels.(map[string]any), "viewCount")
	array := liveItem("array", 0)
	array.Labels = []any{"unrelated", "labels"}
	jsonNull := liveItem("json-null", 0)
	jsonNull.Labels = nil // The schema is NOT NULL; this stores JSON null.
	require.NoError(t, legacy.BatchInsertItems(ctx, []data.Item{untouched, nullKey, zero, missing, array, jsonNull}))
	// Retained rows from an earlier rollback must never override legacy keys,
	// including an explicit JSON null or numeric zero, or non-object labels.
	for i, id := range []string{"legacy", "null-key", "zero", "array", "json-null"} {
		_, err = client.Exec("INSERT INTO "+prefix+"item_view_counts (item_id,view_count,projection_hash) VALUES (?,?,UNHEX(REPEAT('ab',32)))", id, i+7)
		require.NoError(t, err)
	}
	_, before, err := modern.GetItems(ctx, "", 10, nil)
	require.NoError(t, err)
	require.Len(t, before, 7)
	byID := func(items []data.Item) map[string]data.Item {
		out := make(map[string]data.Item, len(items))
		for _, item := range items {
			out[item.ItemId] = item
		}
		return out
	}
	want := byID(before)
	require.Equal(t, float64(42), viewCount(t, want["legacy"]))
	require.Equal(t, float64(0), viewCount(t, want["zero"]))
	require.Nil(t, want["null-key"].Labels.(map[string]any)["viewCount"])
	require.NotContains(t, want["missing-count"].Labels.(map[string]any), "viewCount")
	// Compare physical bytes for every excluded row as well as public readers.
	untouchedLabels := func() map[string]string {
		rows, err := client.Query("SELECT item_id,labels FROM " + prefix + "items WHERE item_id<>'converted'")
		require.NoError(t, err)
		defer rows.Close()
		out := map[string]string{}
		for rows.Next() {
			var id, labels string
			require.NoError(t, rows.Scan(&id, &labels))
			out[id] = labels
		}
		require.NoError(t, rows.Err())
		return out
	}
	physicalBefore := untouchedLabels()
	require.Equal(t, int64(1), copyBackViewCounts(t, client, prefix+"items", prefix+"item_view_counts"))
	require.Equal(t, physicalBefore, untouchedLabels(), "all non-converted labels are byte-preserved")
	_, after, err := legacy.GetItems(ctx, "", 10, nil)
	require.NoError(t, err)
	require.Equal(t, want, byID(after), "split and old readers agree after guarded copy-back")
	for _, item := range after {
		// The existing converters reset accumulated features on a nil map value,
		// so explicit-null input is map-order dependent even without any write.
		// Its exact raw/public labels are checked above; assert feature parity on
		// the remaining deterministic cases, including the independent vector.
		if item.ItemId == "null-key" {
			continue
		}
		require.Equal(t, sortedFeatures(want[item.ItemId].Labels), sortedFeatures(item.Labels))
		require.Equal(t, ctr.ConvertEmbeddings(want[item.ItemId].Labels), ctr.ConvertEmbeddings(item.Labels))
	}
	var numericType string
	require.NoError(t, client.QueryRow("SELECT JSON_TYPE(JSON_EXTRACT(labels,'$.viewCount')) FROM "+prefix+"items WHERE item_id='converted'").Scan(&numericType))
	require.Equal(t, "INTEGER", numericType, "copy-back restores a numeric label, not a string")
	require.Zero(t, copyBackViewCounts(t, client, prefix+"items", prefix+"item_view_counts"), "a second copy-back is a no-op")
	converted.Labels.(map[string]any)["viewCount"] = int64(81)
	converted.Labels.(map[string]any)["embedding"] = vector
	expectBranch(t, data.ViewCountBranchLegacyConversion, func() {
		require.NoError(t, modern.BatchInsertItems(data.WithTwitchLiveProjection(ctx), []data.Item{converted}))
	})
	current, err := modern.GetItem(ctx, "converted")
	require.NoError(t, err)
	require.Equal(t, float64(81), viewCount(t, current))
	require.NotContains(t, current.Labels.(map[string]any), "independent", "re-enable converts a complete source projection with replacement semantics")
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
	require.Equal(t, int64(1), copyBackViewCounts(t, sqlDB, prefix+"items", prefix+"item_view_counts"))
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

func TestMySQLTwitchViewCountsDecisionCounters(t *testing.T) {
	ctx := context.Background()
	legacy, db, sqlDB, prefix := countFixture(t)
	require.NoError(t, db.Init())
	marked := data.WithTwitchLiveProjection(ctx)
	labelPresent := func(id string) int {
		var present int
		require.NoError(t, sqlDB.QueryRow("SELECT JSON_CONTAINS_PATH(labels,'one','$.viewCount') FROM "+prefix+"items WHERE item_id=?", id).Scan(&present))
		return present
	}

	// The default writer (split OFF) never touches the counters.
	evaluated, committed := countDecisions(t, func() {
		require.NoError(t, legacy.BatchInsertItems(marked, []data.Item{liveItem("legacy", 10)}))
	})
	require.Empty(t, evaluated)
	require.Empty(t, committed)

	expectBranch(t, data.ViewCountBranchMissingItem, func() {
		require.NoError(t, db.BatchInsertItems(marked, []data.Item{liveItem("live", 1000)}))
	})
	expectBranch(t, data.ViewCountBranchShortPath, func() {
		require.NoError(t, db.BatchInsertItems(marked, []data.Item{liveItem("live", 1001)}))
	})
	// An independent embedding patch keeps the source fingerprint, so the next
	// count-only projection is still short-path eligible and keeps the patch.
	patcher := db.(data.AtomicItemLabelPatcher)
	_, handled, err := patcher.PatchItemLabelsAtomic(ctx, "live", map[string]any{"embedding": []float64{.75, .875}}, func(data.Item) error { return nil })
	require.True(t, handled)
	require.NoError(t, err)
	expectBranch(t, data.ViewCountBranchShortPath, func() {
		require.NoError(t, db.BatchInsertItems(marked, []data.Item{liveItem("live", 1002)}))
	})
	current, err := db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, float64(1002), viewCount(t, current))
	require.Equal(t, []any{.75, .875}, current.Labels.(map[string]any)["embedding"])

	// Changed lifecycle/safety metadata forces a full write under the marker.
	offline := liveItem("live", 1003)
	offline.IsHidden = true
	offline.Comment = "offline"
	expectBranch(t, data.ViewCountBranchFullChangedSource, func() {
		require.NoError(t, db.BatchInsertItems(marked, []data.Item{offline}))
	})
	current, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.True(t, current.IsHidden)
	// The same source without the marker is an ordinary full replacement.
	expectBranch(t, data.ViewCountBranchUnmarked, func() {
		require.NoError(t, db.BatchInsertItems(ctx, []data.Item{offline}))
	})

	// First touch of a legacy row converts it even though it is marked.
	require.Equal(t, 1, labelPresent("legacy"))
	expectBranch(t, data.ViewCountBranchLegacyConversion, func() {
		require.NoError(t, db.BatchInsertItems(marked, []data.Item{liveItem("legacy", 11)}))
	})
	require.Zero(t, labelPresent("legacy"))
	_, err = sqlDB.Exec("DELETE FROM " + prefix + "item_view_counts WHERE item_id='legacy'")
	require.NoError(t, err)
	expectBranch(t, data.ViewCountBranchMissingCount, func() {
		require.NoError(t, db.BatchInsertItems(marked, []data.Item{liveItem("legacy", 12)}))
	})
	current, err = db.GetItem(ctx, "legacy")
	require.NoError(t, err)
	require.Equal(t, float64(12), viewCount(t, current))

	// Other providers and invalid counts in a mixed batch are unsplittable;
	// a batch without any Twitch live item takes the bulk path per item.
	youtube := liveItem("yt", 5)
	youtube.Labels.(map[string]any)["platform"] = "youtube"
	invalid := liveItem("bad", 0)
	invalid.Labels.(map[string]any)["viewCount"] = "not a number"
	evaluated, committed = countDecisions(t, func() {
		require.NoError(t, db.BatchInsertItems(ctx, []data.Item{youtube, invalid, liveItem("legacy", 12)}))
	})
	require.Equal(t, map[string]float64{data.ViewCountBranchUnsplittable: 2, data.ViewCountBranchUnmarked: 1}, evaluated)
	require.Equal(t, evaluated, committed)
	evaluated, committed = countDecisions(t, func() {
		other := youtube
		other.ItemId = "yt2"
		require.NoError(t, db.BatchInsertItems(ctx, []data.Item{youtube, other}))
	})
	require.Equal(t, map[string]float64{data.ViewCountBranchUnsplittable: 2}, evaluated)
	require.Equal(t, evaluated, committed)

	// A rolled-back transaction is evaluated but never committed, including a
	// short-path item that shared the transaction with the failing one.
	_, err = sqlDB.Exec("ALTER TABLE " + prefix + "item_view_counts ADD CHECK (view_count < 50000)")
	require.NoError(t, err)
	failing := liveItem("live", 50000)
	failing.Comment = "changed"
	evaluated, committed = countDecisions(t, func() {
		require.Error(t, db.BatchInsertItems(marked, []data.Item{failing}))
	})
	require.Equal(t, map[string]float64{data.ViewCountBranchFullChangedSource: 1}, evaluated)
	require.Empty(t, committed)
	evaluated, committed = countDecisions(t, func() {
		require.Error(t, db.BatchInsertItems(marked, []data.Item{offline, liveItem("zz", 60000)}))
	})
	require.Equal(t, map[string]float64{data.ViewCountBranchShortPath: 1, data.ViewCountBranchMissingItem: 1}, evaluated)
	require.Empty(t, committed)
	current, err = db.GetItem(ctx, "live")
	require.NoError(t, err)
	require.Equal(t, "offline", current.Comment)
	require.Equal(t, float64(1003), viewCount(t, current))
	_, err = db.GetItem(ctx, "zz")
	require.ErrorIs(t, err, storage.ErrNotFound)
}
