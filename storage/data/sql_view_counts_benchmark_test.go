package data_test

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/require"
)

type writeSnapshot struct {
	File         string            `json:"binlog_file"`
	Position     uint64            `json:"binlog_position"`
	Status       map[string]uint64 `json:"status"`
	KernelWrites map[string]uint64 `json:"kernel_write_bytes_by_device"`
}

func benchmarkSnapshot(t *testing.T, db *sql.DB, container, dockerContext string) writeSnapshot {
	t.Helper()
	s := writeSnapshot{Status: map[string]uint64{}, KernelWrites: map[string]uint64{}}
	var ignored1, ignored2, ignored3 string
	require.NoError(t, db.QueryRow("SHOW MASTER STATUS").Scan(&s.File, &s.Position, &ignored1, &ignored2, &ignored3))
	rows, err := db.Query("SHOW GLOBAL STATUS WHERE Variable_name IN ('Innodb_os_log_written','Innodb_data_written','Innodb_rows_updated','Innodb_rows_inserted')")
	require.NoError(t, err)
	for rows.Next() {
		var key, value string
		require.NoError(t, rows.Scan(&key, &value))
		s.Status[key], err = strconv.ParseUint(value, 10, 64)
		require.NoError(t, err)
	}
	require.NoError(t, rows.Err())
	require.NoError(t, rows.Close())
	// This is a disposable test-only container named by the harness or CI job
	// (never a production container); the docker context is explicit as well.
	// Kernel virtual-disk writes are not Windows host SSD or production EBS I/O.
	require.True(t, strings.HasPrefix(container, backupContainerPrefix), "TWITCH_COUNT_BENCH_CONTAINER must name a disposable %q container", backupContainerPrefix)
	args := []string{}
	if dockerContext != "" {
		args = append(args, "--context", dockerContext)
	}
	command := exec.Command("docker", append(args, "exec", container, "cat", "/sys/fs/cgroup/io.stat")...)
	output, err := command.Output()
	require.NoError(t, err)
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		for _, field := range fields[1:] {
			if strings.HasPrefix(field, "wbytes=") {
				s.KernelWrites[fields[0]], err = strconv.ParseUint(strings.TrimPrefix(field, "wbytes="), 10, 64)
				require.NoError(t, err)
			}
		}
	}
	require.NotEmpty(t, s.KernelWrites, "physical block-counter coverage is required for this experiment")
	return s
}

func settleWrites(t *testing.T, db *sql.DB) {
	t.Helper()
	for i := 0; i < 45; i++ {
		var key string
		var dirty uint64
		require.NoError(t, db.QueryRow("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_pages_dirty'").Scan(&key, &dirty))
		if dirty == 0 {
			time.Sleep(time.Second)
			require.NoError(t, db.QueryRow("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_pages_dirty'").Scan(&key, &dirty))
			if dirty == 0 {
				return
			}
		}
		time.Sleep(time.Second)
	}
	t.Fatal("dirty pages did not drain within bounded wait")
}

func TestMySQLTwitchViewCountWriteVolume(t *testing.T) {
	outputPath := os.Getenv("TWITCH_COUNT_BENCH_OUTPUT")
	if outputPath == "" {
		t.Skip("explicit isolated write experiment not requested")
	}
	container := os.Getenv("TWITCH_COUNT_BENCH_CONTAINER")
	dockerContext := os.Getenv("TWITCH_COUNT_BENCH_DOCKER_CONTEXT")
	ctx := context.Background()
	legacy, modern, db, prefix := countFixture(t)
	require.NoError(t, modern.Init())
	var version, format, image, partial string
	var compression int
	require.NoError(t, db.QueryRow("SELECT VERSION(),@@binlog_format,@@binlog_row_image,@@binlog_row_value_options,@@binlog_transaction_compression").Scan(&version, &format, &image, &partial, &compression))
	require.Equal(t, "ROW", format)
	require.Equal(t, "FULL", image)
	require.Empty(t, partial)
	require.Zero(t, compression)
	random := rand.New(rand.NewSource(4))
	// Match the production configuration inspected for this trial. The remaining
	// item metadata and traffic are synthetic, not a production replay.
	vector := make([]float64, 256)
	for i := range vector {
		vector[i] = math.Round((random.Float64()*2-1)*1e7) / 1e7
	}
	makeItem := func(id int, count int64) data.Item {
		item := liveItem(fmt.Sprintf("00000000-0000-0000-0000-%012d", id), count)
		item.Labels.(map[string]any)["embedding"] = vector
		item.Labels.(map[string]any)["description"] = "Synthetic benchmark item; no production data."
		return item
	}
	makeOrdinaryItem := func(id int, comment string) data.Item {
		return data.Item{
			ItemId:     fmt.Sprintf("00000000-0000-0000-0000-%012d", id),
			Categories: []string{"format:video"},
			Timestamp:  time.Date(2026, 10, 4, 12, 0, 0, 0, time.UTC),
			Comment:    comment,
			Labels: map[string]any{"platform": "youtube", "externalKind": "Video", "embedding": vector,
				"description": "Synthetic benchmark item; no production data."},
		}
	}
	type result struct {
		Mode                string             `json:"mode"`
		Operations          int                `json:"operations"`
		Seconds             float64            `json:"statement_seconds"`
		BinlogBytes         uint64             `json:"binlog_bytes"`
		Before              writeSnapshot      `json:"before"`
		After               writeSnapshot      `json:"after"`
		VerifiedItems       int                `json:"verified_items"`
		ViewCountDecisions  map[string]float64 `json:"committed_view_count_decisions_by_branch,omitempty"`
		ProjectionDecisions map[string]float64 `json:"committed_projection_decisions_by_branch,omitempty"`
		LabelPatchCommits   map[string]float64 `json:"committed_label_patch_results,omitempty"`
	}
	results := []result{}
	for _, mode := range []string{"legacy_changed", "split_changed", "legacy_identical", "split_identical", "split_changed", "legacy_changed", "ordinary_changed", "ordinary_identical", "ordinary_identical", "ordinary_changed", "patch_changed", "patch_identical", "patch_identical", "patch_changed"} {
		_, err := db.Exec("DELETE FROM " + prefix + "items")
		require.NoError(t, err)
		store := legacy
		writeCtx := ctx
		if strings.HasPrefix(mode, "split_") {
			store = modern
			writeCtx = data.WithTwitchLiveProjection(ctx)
		} else if strings.HasPrefix(mode, "ordinary_") || strings.HasPrefix(mode, "patch_") {
			store = modern
		}
		seed := make([]data.Item, 100)
		for i := range seed {
			if strings.HasPrefix(mode, "ordinary_") || strings.HasPrefix(mode, "patch_") {
				seed[i] = makeOrdinaryItem(i, "stable")
			} else {
				seed[i] = makeItem(i, 1000)
			}
		}
		require.NoError(t, store.BatchInsertItems(writeCtx, seed))
		settleWrites(t, db)
		before := benchmarkSnapshot(t, db, container, dockerContext)
		started := time.Now()
		var seconds float64
		// The committed branch counters tie the measured bytes to the actual
		// eligibility decision: a split arm is only meaningful if every one of
		// its 1000 writes took the short path after the seed conversion.
		writeWorkload := func() {
			for round := 0; round < 10; round++ {
				count := int64(1001 + round)
				if strings.HasSuffix(mode, "identical") {
					count = 1000
				}
				for i := 0; i < 100; i++ {
					if strings.HasPrefix(mode, "patch_") {
						embedding := vector
						if strings.HasSuffix(mode, "changed") {
							embedding = append([]float64(nil), vector...)
							if round%2 == 0 {
								embedding[0] = -0.75
							} else {
								embedding[0] = 0.75
							}
						}
						_, handled, err := store.(data.AtomicItemLabelPatcher).PatchItemLabelsAtomic(ctx,
							makeOrdinaryItem(i, "stable").ItemId, map[string]any{"embedding": embedding}, func(data.Item) error { return nil })
						require.True(t, handled)
						require.NoError(t, err)
						continue
					}
					item := makeItem(i, count)
					if strings.HasPrefix(mode, "ordinary_") {
						comment := "stable"
						if strings.HasSuffix(mode, "changed") {
							comment = fmt.Sprintf("changed-%d", round)
						}
						item = makeOrdinaryItem(i, comment)
					}
					require.NoError(t, store.BatchInsertItems(writeCtx, []data.Item{item}))
				}
			}
			seconds = time.Since(started).Seconds()
		}
		var committed, projectionCommitted, labelPatchCommitted map[string]float64
		if strings.HasPrefix(mode, "ordinary_") {
			projectionCommitted = countProjectionCommits(t, writeWorkload)
		} else if strings.HasPrefix(mode, "patch_") {
			result := "unchanged"
			if strings.HasSuffix(mode, "changed") {
				result = "changed"
			}
			labelPatchCommitted = map[string]float64{result: counterDelta(t, data.ItemLabelPatchCommitsTotal, result, writeWorkload)}
		} else {
			_, committed = countDecisions(t, writeWorkload)
		}
		if strings.HasPrefix(mode, "split_") {
			require.Equal(t, map[string]float64{data.ViewCountBranchShortPath: 1000}, committed)
		} else {
			require.Empty(t, committed, "the split-OFF writer must not report decisions")
		}
		if strings.HasPrefix(mode, "ordinary_") {
			branch := data.ItemProjectionBranchUnchanged
			if strings.HasSuffix(mode, "changed") {
				branch = data.ItemProjectionBranchChanged
			}
			require.Equal(t, map[string]float64{branch: 1000}, projectionCommitted)
		}
		if strings.HasPrefix(mode, "patch_") {
			result := "unchanged"
			if strings.HasSuffix(mode, "changed") {
				result = "changed"
			}
			require.Equal(t, map[string]float64{result: 1000}, labelPatchCommitted)
		}
		settleWrites(t, db)
		after := benchmarkSnapshot(t, db, container, dockerContext)
		require.Equal(t, before.File, after.File)
		items, err := store.BatchGetItems(ctx, func() []string {
			ids := make([]string, 100)
			for i := range ids {
				ids[i] = makeItem(i, 0).ItemId
			}
			return ids
		}(), data.GetOptions{})
		require.NoError(t, err)
		require.Len(t, items, 100)
		for _, item := range items {
			if strings.HasPrefix(mode, "patch_") {
				embedding := item.Labels.(map[string]any)["embedding"].([]any)
				expected := vector[0]
				if strings.HasSuffix(mode, "changed") {
					expected = 0.75
				}
				require.Equal(t, expected, embedding[0])
				continue
			}
			if strings.HasPrefix(mode, "ordinary_") {
				expected := "stable"
				if strings.HasSuffix(mode, "changed") {
					expected = "changed-9"
				}
				require.Equal(t, expected, item.Comment)
				continue
			}
			expected := float64(1010)
			if strings.HasSuffix(mode, "identical") {
				expected = 1000
			}
			require.Equal(t, expected, viewCount(t, item))
			require.Equal(t, vector, func() []float64 {
				values := item.Labels.(map[string]any)["embedding"].([]any)
				out := make([]float64, len(values))
				for i, v := range values {
					out[i] = v.(float64)
				}
				return out
			}())
		}
		results = append(results, result{Mode: mode, Operations: 1000, Seconds: seconds,
			BinlogBytes: after.Position - before.Position, Before: before, After: after,
			VerifiedItems: len(items), ViewCountDecisions: committed, ProjectionDecisions: projectionCommitted,
			LabelPatchCommits: labelPatchCommitted})
		t.Logf("%s: %d binlog bytes over 1000 actual datastore writes", mode, after.Position-before.Position)
	}
	receipt := map[string]any{"mysql_version": version, "binlog_format": format, "binlog_row_image": image, "partial_json": partial, "compression": compression, "items": 100, "embedding_dimensions": len(vector), "results": results,
		"coverage": "Synthetic items against actual Gorse SQLDatabase methods, not raw SQL emulation and not production data or traffic. Kernel counters measure the named container's virtual-disk block writes, not production EBS. HTTP, Redis cache and vector-index writes are excluded; seed conversion is excluded from every arm."}
	encoded, err := json.MarshalIndent(receipt, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(outputPath, encoded, 0600))
}
