package data_test

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/require"
)

// Runs only against the explicitly configured, disposable MySQL fixture.
// No production rows, traffic or binlog payloads are used.
func TestMySQLNumericProjectionWriteExperiment(t *testing.T) {
	output := os.Getenv("PROJECTION_NUMERIC_BENCH_OUTPUT")
	if output == "" {
		t.Skip("explicit synthetic write experiment not requested")
	}
	_, store, db, prefix := countFixture(t)
	require.NoError(t, store.Init())
	ctx := context.Background()
	vector := make([]float64, 256)
	rng := rand.New(rand.NewSource(42))
	for i := range vector {
		vector[i] = float64(float32((rng.Float64()*2 - 1) * 0.1))
	}
	vector[0] = 1e-7
	vector[1] = -1e-8
	items := make([]data.Item, 50)
	for i := range items {
		items[i] = data.Item{ItemId: fmt.Sprintf("numeric-%d", i), Categories: []string{"video"}, Timestamp: time.Date(2026, 10, 25, 12, 0, 0, 0, time.UTC), Comment: "synthetic",
			Labels: map[string]any{"platform": "twitch", "externalKind": "Video", "embedding": vector, "viewCount": int64(1234)}}
	}
	require.NoError(t, store.BatchInsertItems(ctx, items))
	var storedJSON string
	require.NoError(t, db.QueryRow("SELECT labels FROM "+prefix+"items WHERE item_id=?", items[0].ItemId).Scan(&storedJSON))
	var stored map[string]any
	decoder := json.NewDecoder(strings.NewReader(storedJSON))
	decoder.UseNumber()
	require.NoError(t, decoder.Decode(&stored))
	logged := 0
	for i, number := range stored["embedding"].([]any) {
		encoded, _ := json.Marshal(vector[i])
		if string(encoded) != string(number.(json.Number)) && logged < 8 {
			t.Logf("synthetic embedding %d: sent=%s MySQL=%s", i, encoded, number)
			logged++
		}
	}
	snapshot := func() (uint64, map[string]uint64) {
		var file, a, b, c string
		var position uint64
		require.NoError(t, db.QueryRow("SHOW MASTER STATUS").Scan(&file, &position, &a, &b, &c))
		stats := map[string]uint64{}
		rows, err := db.Query("SHOW GLOBAL STATUS WHERE Variable_name IN ('Com_insert','Com_update','Com_select','Innodb_rows_updated','Innodb_rows_inserted','Innodb_os_log_written')")
		require.NoError(t, err)
		defer rows.Close()
		for rows.Next() {
			var key string
			var value uint64
			require.NoError(t, rows.Scan(&key, &value))
			stats[key] = value
		}
		require.NoError(t, rows.Err())
		return position, stats
	}
	results := []map[string]any{}
	for _, mode := range []string{"ordinary_identical", "patch_identical", "ordinary_actual_change"} {
		before, statsBefore := snapshot()
		start := time.Now()
		var decisions map[string]float64
		work := func() {
			for round := 0; round < 10; round++ {
				if mode == "patch_identical" {
					for _, item := range items {
						_, handled, err := store.(data.AtomicItemLabelPatcher).PatchItemLabelsAtomic(ctx, item.ItemId, map[string]any{"embedding": vector}, func(data.Item) error { return nil })
						require.True(t, handled)
						require.NoError(t, err)
					}
				} else {
					if mode == "ordinary_actual_change" {
						for i := range items {
							items[i].Comment = fmt.Sprintf("changed-%d", round)
						}
					}
					require.NoError(t, store.BatchInsertItems(ctx, items))
				}
			}
		}
		if mode == "patch_identical" {
			work()
		} else {
			decisions = countProjectionCommits(t, work)
		}
		elapsed := time.Since(start).Seconds()
		after, statsAfter := snapshot()
		delta := map[string]uint64{}
		for key, value := range statsAfter {
			delta[key] = value - statsBefore[key]
		}
		results = append(results, map[string]any{"mode": mode, "operations": 500, "seconds": elapsed, "binlog_bytes": after - before, "status_delta": delta, "projection_decisions": decisions})
	}
	encoded, err := json.MarshalIndent(results, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(output, encoded, 0600))
	t.Log(string(encoded))
}
