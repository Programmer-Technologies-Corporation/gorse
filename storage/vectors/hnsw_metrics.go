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

package vectors

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// VideoHub fork: metrics of the hnsw:// vector store. They are registered on
// the default registry, which the master already exposes on /metrics.
var (
	hnswVectors = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "vectors",
		Help: "Live vectors per collection of the hnsw vector store.",
	}, []string{"collection"})
	hnswTombstones = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "tombstones",
		Help: "Replaced or deleted vectors waiting for compaction.",
	}, []string{"collection"})
	hnswBytes = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "memory_bytes",
		Help: "Estimated heap held by a collection (vectors, graph, metadata).",
	}, []string{"collection"})
	hnswUpsertsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "upserts_total",
		Help: "Upserted vectors by outcome: inserted, replaced (vector changed), metadata (only hidden/categories changed), unchanged.",
	}, []string{"collection", "outcome"})
	hnswDeletesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "deletes_total",
		Help: "Vectors removed because a cycle no longer produced them.",
	}, []string{"collection"})
	hnswCompactionsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "compactions_total",
		Help: "Full rebuilds of a collection to reclaim tombstones.",
	}, []string{"collection"})
	hnswCompactionSeconds = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "compaction_seconds",
		Help: "Duration of the last compaction.",
	}, []string{"collection"})
	hnswSnapshotSeconds = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "snapshot_seconds",
		Help: "Duration of the last snapshot.",
	}, []string{"collection"})
	hnswSnapshotBytes = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "snapshot_bytes",
		Help: "Size of the last snapshot file.",
	}, []string{"collection"})
	hnswSnapshotFailuresTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "snapshot_failures_total",
		Help: "Snapshots that could not be written.",
	}, []string{"collection"})
	hnswLoadFailuresTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "load_failures_total",
		Help: "Collections whose snapshot could not be read at startup and were started empty.",
	}, []string{"collection"})
	hnswQuerySeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "gorse", Subsystem: "vector_index", Name: "query_seconds",
		Help:    "Query latency by search mode: graph, scan (selective filter) or inverted (sparse).",
		Buckets: prometheus.ExponentialBuckets(0.00005, 2, 16),
	}, []string{"mode"})
)

// deleteHNSWGauges drops the series of a removed collection. Collaborative
// filtering creates one collection per model, so they would pile up otherwise.
func deleteHNSWGauges(collection string) {
	labels := prometheus.Labels{"collection": collection}
	for _, vec := range []*prometheus.GaugeVec{hnswVectors, hnswTombstones, hnswBytes, hnswCompactionSeconds, hnswSnapshotSeconds, hnswSnapshotBytes} {
		vec.DeletePartialMatch(labels)
	}
	for _, vec := range []*prometheus.CounterVec{hnswUpsertsTotal, hnswDeletesTotal, hnswCompactionsTotal, hnswSnapshotFailuresTotal, hnswLoadFailuresTotal} {
		vec.DeletePartialMatch(labels)
	}
}
