package vectors

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
)

func TestHNSWRefreshPreservesResultsAndGauges(t *testing.T) {
	for _, sparse := range []bool{false, true} {
		name := "refresh_dense"
		dim := 2
		distance := Euclidean
		if sparse {
			name, dim = "refresh_sparse", 0
			distance = Dot
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			db := openHNSW(t, t.TempDir(), "")
			defer db.Close()
			require.NoError(t, db.AddCollection(ctx, name, dim, distance, VectorConfig{}))
			now := time.Now().UTC().Truncate(time.Millisecond)
			batch := []Vector{
				{Id: "a", Values: []float32{1, 2}, Categories: []string{"live"}, Timestamp: now},
				{Id: "b", Values: []float32{2, 1}, Categories: []string{"live"}, Timestamp: now},
			}
			if sparse {
				for i := range batch {
					batch[i].Indices = []uint32{1, 2}
				}
			}
			require.NoError(t, db.AddVectors(ctx, name, batch))
			collection, err := db.collection(name)
			require.NoError(t, err)
			metricValue := func(m prometheus.Metric) float64 {
				t.Helper()
				var value dto.Metric
				require.NoError(t, m.Write(&value))
				if value.Gauge != nil {
					return value.Gauge.GetValue()
				}
				return value.Counter.GetValue()
			}
			checkGauges := func() {
				t.Helper()
				collection.mu.RLock()
				defer collection.mu.RUnlock()
				require.Equal(t, float64(collection.bytesLocked()), metricValue(hnswBytes.WithLabelValues(name)))
				require.Equal(t, float64(collection.meta.live), metricValue(hnswVectors.WithLabelValues(name)))
				require.Equal(t, float64(collection.tombstones()), metricValue(hnswTombstones.WithLabelValues(name)))
			}
			checkGauges()
			before, err := db.QueryVectors(ctx, name, batch[0], []string{"live"}, 2)
			require.NoError(t, err)
			unchanged := metricValue(hnswUpsertsTotal.WithLabelValues(name, "unchanged"))
			refreshed := now.Add(time.Hour)
			for i := range batch {
				batch[i].Timestamp = refreshed
				require.NoError(t, db.AddVectors(ctx, name, batch[i:i+1]))
				checkGauges()
			}
			require.Equal(t, unchanged+2, metricValue(hnswUpsertsTotal.WithLabelValues(name, "unchanged")))
			require.NoError(t, db.DeleteVectors(ctx, name, refreshed))
			after, err := db.QueryVectors(ctx, name, batch[0], []string{"live"}, 2)
			require.NoError(t, err)
			require.Len(t, after, len(before))
			for i := range before {
				require.True(t, after[i].Timestamp.Equal(refreshed))
				before[i].Timestamp = after[i].Timestamp
			}
			require.Equal(t, before, after)
			stored, err := db.GetVectors(ctx, name, []string{"a", "b"})
			require.NoError(t, err)
			require.Equal(t, batch, stored)
			checkGauges()

			// A mixed batch must still publish gauges after a real replacement.
			batch[1].Values = []float32{3, 1}
			require.NoError(t, db.AddVectors(ctx, name, batch))
			checkGauges()
			batch[0].IsHidden = true
			require.NoError(t, db.AddVectors(ctx, name, batch[:1]))
			checkGauges()

			// Cancellation after one committed insertion must not leave stale
			// gauges, including when subsequent traffic only refreshes timestamps.
			partial := batch[0]
			partial.Id = "partial"
			cancelCtx, cancel := context.WithCancel(ctx)
			defer cancel()
			interrupted := &cancelOnSecondPoll{Context: cancelCtx, cancel: cancel}
			require.ErrorIs(t, db.AddVectors(interrupted, name, []Vector{partial, batch[1]}), context.Canceled)
			checkGauges()
			partial.Timestamp = refreshed.Add(time.Millisecond)
			require.NoError(t, db.AddVectors(ctx, name, []Vector{partial}))
			checkGauges()
			require.NoError(t, db.DeleteVectors(ctx, name, refreshed.Add(time.Millisecond)))
			checkGauges()
		})
	}
}

// upsert polls before each vector; cancel after the first vector is committed.
type cancelOnSecondPoll struct {
	context.Context
	cancel context.CancelFunc
	polls  int
}

func (c *cancelOnSecondPoll) Err() error {
	c.polls++
	if c.polls == 2 {
		c.cancel()
	}
	return c.Context.Err()
}
