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
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorse-io/gorse/common/event"
	"github.com/gorse-io/gorse/dataset"
	"github.com/gorse-io/gorse/model/ctr"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/stretchr/testify/require"
)

// Opt-in harness: run each sample in a fresh compiled-test process. Item creation
// stays lazy, and diagnostic GC is explicitly separated from ordinary sampling.
func retentionAtomicMax(peak *atomic.Uint64, value uint64) {
	for old := peak.Load(); value > old; old = peak.Load() {
		if peak.CompareAndSwap(old, value) {
			return
		}
	}
}
func TestDatasetRetentionMemoryExperiment(t *testing.T) {
	countString := os.Getenv("GORSE_RETENTION_ITEMS")
	if countString == "" {
		t.Skip("set GORSE_RETENTION_ITEMS for the local memory experiment")
	}
	n, err := strconv.Atoi(countString)
	require.NoError(t, err)
	nonempty := os.Getenv("GORSE_RETENTION_NONEMPTY") == "1"
	diagnostic := os.Getenv("GORSE_RETENTION_DIAGNOSTIC") == "1"
	store := &retentionStreamStore{n: n, dim: 256, phase: make(chan struct{}), release: make(chan struct{})}
	m, recommenders := retentionLoader(t, nonempty, min(n, 100))
	runtime.GC() // Common pre-load starting point; ordinary runs never force GC during loading.
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	var peak atomic.Uint64
	stop, sampled := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(sampled)
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				var stats runtime.MemStats
				runtime.ReadMemStats(&stats)
				retentionAtomicMax(&peak, stats.HeapAlloc)
			case <-stop:
				return
			}
		}
	}()
	type loadResult struct {
		click    *ctr.Dataset
		ranking  *dataset.Dataset
		snapshot event.Snapshot
		err      error
	}
	done := make(chan loadResult, 1)
	started := time.Now()
	go func() {
		click, ranking, snapshot, err := m.LoadDataFromDatabase(context.Background(), store, retentionTypes("positive"), retentionTypes("negative"), retentionTypes("read"), 0, 0, NewOnlineEvaluator(nil, nil), recommenders)
		done <- loadResult{click, ranking, snapshot, err}
	}()
	select {
	case <-store.phase:
	case <-time.After(2 * time.Minute):
		close(store.release)
		t.Fatal("first feedback phase was not reached")
	}
	if diagnostic {
		runtime.GC()
	}
	var phase runtime.MemStats
	runtime.ReadMemStats(&phase)
	retentionAtomicMax(&peak, phase.HeapAlloc)
	if profile := os.Getenv("GORSE_RETENTION_PROFILE"); profile != "" {
		f, err := os.Create(profile + "-phase-heap.pprof")
		require.NoError(t, err)
		require.NoError(t, pprof.Lookup("heap").WriteTo(f, 0))
		require.NoError(t, f.Close())
	}
	close(store.release)
	result := <-done
	require.NoError(t, result.err)
	elapsed := time.Since(started)
	close(stop)
	<-sampled
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	retentionAtomicMax(&peak, after.HeapAlloc)
	// Capture allocation profile before correctness serialization adds allocations.
	if profile := os.Getenv("GORSE_RETENTION_PROFILE"); profile != "" {
		f, err := os.Create(profile + "-loader-allocs.pprof")
		require.NoError(t, err)
		require.NoError(t, pprof.Lookup("allocs").WriteTo(f, 0))
		require.NoError(t, f.Close())
	}
	// Stream the hash to avoid an externally held serialized full-data fixture.
	hash := sha256.New()
	encoder := json.NewEncoder(hash)
	snapshot := result.snapshot
	snapshot.Timestamp = time.Time{}
	require.NoError(t, encoder.Encode(snapshot))
	for i, item := range result.ranking.GetItems() {
		require.NoError(t, encoder.Encode(item))
		features := map[string]float32{}
		for _, v := range result.click.ItemLabels[i] {
			features[result.click.Index.GetItemLabels()[v.A]] = v.B
		}
		embeddings := map[string][]uint16{}
		for j, v := range result.click.ItemEmbeddings[i] {
			embeddings[result.click.ItemEmbeddingIndex.ToName(int32(j))] = v
		}
		require.NoError(t, encoder.Encode(features))
		require.NoError(t, encoder.Encode(embeddings))
	}
	// Include all remaining feature/feedback state using the small canonical digest.
	// Large items/embeddings were hashed above, so hash a shared-slice shallow view
	// containing only the first twelve items here; no full fixture is retained.
	tiny := dataset.NewDataset(retentionTimestamp, 0, 0)
	for _, user := range result.ranking.GetUsers() {
		tiny.AddUser(user)
	}
	for _, item := range result.ranking.GetItems()[:min(12, n)] {
		tiny.AddItem(item)
	}
	for user, items := range result.ranking.GetUserFeedback() {
		for _, item := range items {
			if item < 12 {
				tiny.AddFeedback(result.click.Index.GetUsers()[user], result.click.Index.GetItems()[item], retentionTimestamp)
			}
		}
	}
	clickView := *result.click
	clickView.ItemLabels = clickView.ItemLabels[:min(12, n)]
	clickView.ItemEmbeddings = clickView.ItemEmbeddings[:min(12, n)]
	var scores []cache.Score
	if nonempty {
		scores = recommenders[0].PopAll()
	}
	require.NoError(t, encoder.Encode(retentionDigest(t, tiny, &clickView, snapshot, scores)))
	measurement := struct {
		N                                                                                  int `json:"n"`
		Nonempty, Diagnostic                                                               bool
		StartHeap, PhaseHeap, PhaseRetainedDelta, PeakHeap, TotalAllocated, PhaseAllocated uint64
		GCCount                                                                            uint32
		ElapsedNanoseconds                                                                 int64
		Digest                                                                             string
		Snapshot                                                                           event.Snapshot
	}{n, nonempty, diagnostic, before.HeapAlloc, phase.HeapAlloc, phase.HeapAlloc - before.HeapAlloc, peak.Load(), after.TotalAlloc - before.TotalAlloc, phase.TotalAlloc - before.TotalAlloc, after.NumGC - before.NumGC, elapsed.Nanoseconds(), fmt.Sprintf("%x", hash.Sum(nil)), snapshot}
	encoded, err := json.Marshal(measurement)
	require.NoError(t, err)
	t.Logf("RETENTION_MEMORY %s", encoded)
	runtime.KeepAlive(result)
}
