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
	"context"
	"fmt"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorse-io/gorse/common/floats"
	"github.com/gorse-io/gorse/common/log"
	"github.com/gorse-io/gorse/storage"
	"github.com/pkg/errors"
	"go.uber.org/zap"
)

// VideoHub fork: hnsw:// is an embedded vector store built for the way Gorse
// uses one. The master rewrites every similarity collection each cycle and the
// server upserts single items in between, so the store
//
//   - keeps one persistent HNSW graph per dense collection and updates it in
//     place: an upsert whose vector did not change only refreshes metadata, a
//     changed vector is re-linked, a removed one becomes a tombstone;
//   - reclaims tombstones with a background rebuild once they pass a ratio,
//     the only time a whole graph is built again;
//   - stores vectors as FP16 when that is lossless (Gorse embeddings already
//     are FP16), otherwise as FP32;
//   - serves sparse collections from an exact inverted index;
//   - persists each collection as one snapshot file, written atomically.
//
// Options are URL query parameters, for example
//
//	hnsw:///var/lib/gorse/master/vectors?m=32&ef_construction=200&ef_search=100
//
// A crash loses the writes since the last snapshot (at most snapshot_interval).
// The master's next cycle restores them because it diffs against the store.

const (
	hnswSnapshotFile      = "index.bin"
	hnswPreviousSnapshot  = "index.bin.old"
	hnswPrecisionAuto     = "auto"
	hnswPrecisionFP16     = "fp16"
	hnswPrecisionFP32     = "fp32"
	hnswMinTombstones     = 1024
	hnswSnapshotQuietTime = 2 * time.Second
	// A Dot collection learns its norm bound from the first batch; later
	// vectors may be longer, and those past the bound only lose the benefit.
	hnswPhiHeadroom = 1.5
)

func init() {
	Register([]string{storage.HNSWPrefix}, func(path, tablePrefix string, _ ...storage.Option) (Database, error) {
		return newHNSWStore(strings.TrimPrefix(path, storage.HNSWPrefix), tablePrefix)
	})
}

type hnswOptions struct {
	params           hnswParams
	precision        string
	snapshotInterval time.Duration
	compactRatio     float64 // rebuild a graph when tombstones exceed this share of its slots
	scanRatio        float64 // filters matching less than this share are answered by an exact scan
	scanMin          int     // ... as are filters matching fewer vectors than this
}

func defaultHNSWOptions() hnswOptions {
	return hnswOptions{
		params:           hnswParams{M: 32, M0: 64, EFConstruction: 200, EFSearch: 100},
		precision:        hnswPrecisionAuto,
		snapshotInterval: time.Minute,
		compactRatio:     0.2,
		scanRatio:        0.05,
		scanMin:          4096,
	}
}

func parseHNSWOptions(rawQuery string) (hnswOptions, error) {
	opts := defaultHNSWOptions()
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return opts, errors.WithStack(err)
	}
	intOption := func(key string, target *int, minimum, maximum int) error {
		if !values.Has(key) {
			return nil
		}
		parsed, err := strconv.Atoi(values.Get(key))
		if err != nil || parsed < minimum || parsed > maximum {
			return errors.Errorf("hnsw option %s must be an integer in [%d, %d]", key, minimum, maximum)
		}
		*target = parsed
		return nil
	}
	ratioOption := func(key string, target *float64) error {
		if !values.Has(key) {
			return nil
		}
		parsed, err := strconv.ParseFloat(values.Get(key), 64)
		if err != nil || parsed < 0 || parsed > 1 {
			return errors.Errorf("hnsw option %s must be a number in [0, 1]", key)
		}
		*target = parsed
		return nil
	}
	if err = intOption("m", &opts.params.M, 4, 256); err != nil {
		return opts, err
	}
	opts.params.M0 = 2 * opts.params.M
	if err = intOption("m0", &opts.params.M0, opts.params.M, 1024); err != nil {
		return opts, err
	}
	if err = intOption("ef_construction", &opts.params.EFConstruction, 8, 4096); err != nil {
		return opts, err
	}
	if err = intOption("ef_search", &opts.params.EFSearch, 1, 4096); err != nil {
		return opts, err
	}
	if err = intOption("scan_min", &opts.scanMin, 0, math.MaxInt32); err != nil {
		return opts, err
	}
	if err = ratioOption("compact_ratio", &opts.compactRatio); err != nil {
		return opts, err
	}
	if err = ratioOption("scan_ratio", &opts.scanRatio); err != nil {
		return opts, err
	}
	if values.Has("precision") {
		opts.precision = values.Get("precision")
		switch opts.precision {
		case hnswPrecisionAuto, hnswPrecisionFP16, hnswPrecisionFP32:
		default:
			return opts, errors.Errorf("hnsw option precision must be auto, fp16 or fp32")
		}
	}
	if values.Has("snapshot_interval") {
		if opts.snapshotInterval, err = time.ParseDuration(values.Get("snapshot_interval")); err != nil || opts.snapshotInterval <= 0 {
			return opts, errors.Errorf("hnsw option snapshot_interval must be a positive duration")
		}
	}
	for key := range values {
		switch key {
		case "m", "m0", "ef_construction", "ef_search", "scan_min", "compact_ratio", "scan_ratio", "precision", "snapshot_interval":
		default:
			return opts, errors.Errorf("unknown hnsw option %s", key)
		}
	}
	return opts, nil
}

// HNSWStore implements Database on local disk.
type HNSWStore struct {
	root        string
	tablePrefix string
	opts        hnswOptions

	mu          sync.RWMutex
	collections map[string]*hnswCollection
	closed      bool
	started     bool

	wake chan struct{}
	stop chan struct{}
	done chan struct{}
}

func newHNSWStore(path, tablePrefix string) (*HNSWStore, error) {
	root, rawQuery, _ := strings.Cut(path, "?")
	if root == "" {
		return nil, errors.New("hnsw path is empty")
	}
	opts, err := parseHNSWOptions(rawQuery)
	if err != nil {
		return nil, err
	}
	return &HNSWStore{
		root:        root,
		tablePrefix: tablePrefix,
		opts:        opts,
		collections: make(map[string]*hnswCollection),
		wake:        make(chan struct{}, 1),
		stop:        make(chan struct{}),
		done:        make(chan struct{}),
	}, nil
}

func (db *HNSWStore) collectionDir(name string) string {
	return filepath.Join(db.root, url.PathEscape(db.tablePrefix+name))
}

func (db *HNSWStore) Init() error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed {
		return errors.New("hnsw store is closed")
	}
	if db.started {
		return nil
	}
	if err := os.MkdirAll(db.root, 0o755); err != nil {
		return errors.WithStack(err)
	}
	entries, err := os.ReadDir(db.root)
	if err != nil {
		return errors.WithStack(err)
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		physical, err := url.PathUnescape(entry.Name())
		if err != nil || !strings.HasPrefix(physical, db.tablePrefix) || physical == db.tablePrefix {
			continue
		}
		name := strings.TrimPrefix(physical, db.tablePrefix)
		start := time.Now()
		collection, err := loadHNSWCollection(name, filepath.Join(db.root, entry.Name()), db.opts)
		if err != nil {
			// A collection that cannot be read is rebuilt by the next cycle;
			// refusing to start would take every other collection down with it.
			log.Logger().Error("failed to load vector collection, starting it empty",
				zap.String("collection", name), zap.Error(err))
			hnswLoadFailuresTotal.WithLabelValues(name).Inc()
			continue
		}
		db.collections[name] = collection
		collection.publishGauges()
		log.Logger().Info("loaded vector collection",
			zap.String("collection", name),
			zap.Int("vectors", collection.meta.live),
			zap.Int("tombstones", collection.tombstones()),
			zap.Duration("used_time", time.Since(start)))
	}
	db.started = true
	go db.maintain()
	return nil
}

func (db *HNSWStore) Close() error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return nil
	}
	db.closed = true
	started := db.started
	collections := make([]*hnswCollection, 0, len(db.collections))
	for _, collection := range db.collections {
		collections = append(collections, collection)
	}
	db.mu.Unlock()

	close(db.stop)
	if started {
		<-db.done
	}
	var firstErr error
	for _, collection := range collections {
		if err := collection.snapshot(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (db *HNSWStore) collection(name string) (*hnswCollection, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.closed {
		return nil, errors.New("hnsw store is closed")
	}
	collection, found := db.collections[name]
	if !found {
		return nil, fmt.Errorf("collection %s: %w", name, storage.ErrNotFound)
	}
	return collection, nil
}

// Optimize persists the collection now and reclaims its tombstones if they
// have passed the compaction ratio.
func (db *HNSWStore) Optimize(ctx context.Context, name string) error {
	collection, err := db.collection(name)
	if err != nil {
		return err
	}
	if collection.needsCompaction() {
		collection.compact(db.stop)
	}
	return collection.snapshot()
}

func (db *HNSWStore) ListCollections(ctx context.Context) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, errors.WithStack(err)
	}
	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.closed {
		return nil, errors.New("hnsw store is closed")
	}
	names := make([]string, 0, len(db.collections))
	for name := range db.collections {
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

func (db *HNSWStore) DescribeCollection(ctx context.Context, name string) (*CollectionInfo, error) {
	if err := ctx.Err(); err != nil {
		return nil, errors.WithStack(err)
	}
	collection, err := db.collection(name)
	if err != nil {
		return nil, err
	}
	return &CollectionInfo{Name: name, Dimension: collection.dim, Distance: collection.distance}, nil
}

func (db *HNSWStore) AddCollection(ctx context.Context, name string, dimensions int, distance Distance, config VectorConfig) error {
	if err := ctx.Err(); err != nil {
		return errors.WithStack(err)
	}
	if dimensions < 0 || dimensions > math.MaxInt32 {
		return errors.Errorf("invalid vector dimension %d", dimensions)
	}
	if config.Type != QuantizationNone {
		return fmt.Errorf("quantization type %s for hnsw %w", config.Type, storage.ErrNotSupported)
	}
	switch distance {
	case Cosine, Euclidean, Dot:
	default:
		return fmt.Errorf("distance method %v %w", distance, storage.ErrNotSupported)
	}
	if dimensions == 0 && distance != Dot {
		return fmt.Errorf("distance method for sparse vector %w", storage.ErrNotSupported)
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed {
		return errors.New("hnsw store is closed")
	}
	if _, found := db.collections[name]; found {
		return fmt.Errorf("collection %s %w", name, storage.ErrAlreadyExists)
	}
	dir := db.collectionDir(name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return errors.WithStack(err)
	}
	collection := newHNSWCollection(name, dir, dimensions, distance, db.opts)
	// Persist the empty collection so that it survives a restart before its
	// first vector arrives.
	if err := collection.snapshot(); err != nil {
		return err
	}
	db.collections[name] = collection
	collection.publishGauges()
	return nil
}

func (db *HNSWStore) DeleteCollection(ctx context.Context, name string) error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return errors.New("hnsw store is closed")
	}
	collection, found := db.collections[name]
	if !found {
		db.mu.Unlock()
		return fmt.Errorf("collection %s: %w", name, storage.ErrNotFound)
	}
	delete(db.collections, name)
	db.mu.Unlock()

	collection.mu.Lock()
	collection.dropped = true
	collection.mu.Unlock()
	// Snapshots are written under ioMu, so no file appears after this point.
	collection.ioMu.Lock()
	defer collection.ioMu.Unlock()
	deleteHNSWGauges(name)
	return errors.WithStack(os.RemoveAll(collection.dir))
}

func (db *HNSWStore) CountVectors(ctx context.Context, name string) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, errors.WithStack(err)
	}
	collection, err := db.collection(name)
	if err != nil {
		return 0, err
	}
	collection.mu.RLock()
	defer collection.mu.RUnlock()
	return int64(collection.meta.live), nil
}

func (db *HNSWStore) AddVectors(ctx context.Context, name string, vectors []Vector) error {
	if len(vectors) == 0 {
		return nil
	}
	collection, err := db.collection(name)
	if err != nil {
		return err
	}
	if err = collection.upsert(ctx, vectors); err != nil {
		return err
	}
	db.wakeMaintenance()
	return nil
}

func (db *HNSWStore) GetVectors(ctx context.Context, name string, ids []string) ([]Vector, error) {
	if len(ids) == 0 {
		return []Vector{}, nil
	}
	collection, err := db.collection(name)
	if err != nil {
		return nil, err
	}
	return orderVectors(ids, collection.get(ids)), nil
}

func (db *HNSWStore) DeleteVectors(ctx context.Context, name string, timestamp time.Time) error {
	collection, err := db.collection(name)
	if err != nil {
		return err
	}
	collection.deleteBefore(timestamp)
	db.wakeMaintenance()
	return nil
}

func (db *HNSWStore) QueryVectors(ctx context.Context, name string, q Vector, categories []string, topK int) ([]ScoredVector, error) {
	if topK <= 0 {
		return []ScoredVector{}, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, errors.WithStack(err)
	}
	collection, err := db.collection(name)
	if err != nil {
		return nil, err
	}
	return collection.query(q, categories, topK)
}

func (db *HNSWStore) wakeMaintenance() {
	select {
	case db.wake <- struct{}{}:
	default:
	}
}

// maintain is the store's only background goroutine. It snapshots collections
// that changed and rebuilds graphs that carry too many tombstones, one at a
// time so that maintenance never competes with itself for memory.
func (db *HNSWStore) maintain() {
	defer close(db.done)
	ticker := time.NewTicker(db.opts.snapshotInterval)
	defer ticker.Stop()
	for {
		snapshots := false
		select {
		case <-db.stop:
			return
		case <-ticker.C:
			snapshots = true
		case <-db.wake:
		}
		db.mu.RLock()
		collections := make([]*hnswCollection, 0, len(db.collections))
		for _, collection := range db.collections {
			collections = append(collections, collection)
		}
		db.mu.RUnlock()
		for _, collection := range collections {
			select {
			case <-db.stop:
				return
			default:
			}
			if collection.needsCompaction() {
				collection.compact(db.stop)
			}
			if snapshots && collection.quietFor(hnswSnapshotQuietTime) {
				if err := collection.snapshot(); err != nil {
					log.Logger().Error("failed to snapshot vector collection",
						zap.String("collection", collection.name), zap.Error(err))
				}
			}
		}
	}
}

// slotMeta is the per-slot metadata shared by dense and sparse collections.
// Category sets are interned: items share a handful of category combinations,
// so a slot stores one int32 and a filter is evaluated once per set, not once
// per visited vector.
type slotMeta struct {
	ids        []string
	hidden     []bool
	deleted    []bool
	timestamps []int64
	catSet     []int32

	slots map[string]int32

	catNames   []string
	catIDs     map[string]uint32
	sets       [][]uint32 // categories in the order they were given
	sortedSets [][]uint32 // the same sets sorted, for matching
	setIDs     map[string]int32
	// setVisible counts live, visible slots per set; it sizes a filter's match.
	setVisible []int

	live    int
	visible int
}

func newSlotMeta() *slotMeta {
	return &slotMeta{
		slots:  make(map[string]int32),
		catIDs: make(map[string]uint32),
		setIDs: make(map[string]int32),
	}
}

func (m *slotMeta) internSet(categories []string) int32 {
	ids := make([]uint32, 0, len(categories))
	for _, category := range categories {
		id, found := m.catIDs[category]
		if !found {
			id = uint32(len(m.catNames))
			m.catNames = append(m.catNames, category)
			m.catIDs[category] = id
		}
		if !slices.Contains(ids, id) {
			ids = append(ids, id)
		}
	}
	if set, found := m.setIDs[setKey(ids)]; found {
		return set
	}
	return m.addSet(ids)
}

func (m *slotMeta) addSet(ids []uint32) int32 {
	set := int32(len(m.sets))
	sorted := slices.Clone(ids)
	slices.Sort(sorted)
	m.sets = append(m.sets, ids)
	m.sortedSets = append(m.sortedSets, sorted)
	m.setVisible = append(m.setVisible, 0)
	m.setIDs[setKey(ids)] = set
	return set
}

func setKey(ids []uint32) string {
	var builder strings.Builder
	for _, id := range ids {
		builder.WriteString(strconv.FormatUint(uint64(id), 36))
		builder.WriteByte(',')
	}
	return builder.String()
}

func (m *slotMeta) categories(slot int32) []string {
	set := m.sets[m.catSet[slot]]
	if len(set) == 0 {
		return nil
	}
	categories := make([]string, len(set))
	for i, id := range set {
		categories[i] = m.catNames[id]
	}
	return categories
}

func (m *slotMeta) countVisible(slot int32, delta int) {
	if !m.hidden[slot] {
		m.visible += delta
		m.setVisible[m.catSet[slot]] += delta
	}
}

func (m *slotMeta) appendSlot(id string, hidden bool, timestamp int64, set int32) {
	slot := int32(len(m.ids))
	m.ids = append(m.ids, id)
	m.hidden = append(m.hidden, hidden)
	m.deleted = append(m.deleted, false)
	m.timestamps = append(m.timestamps, timestamp)
	m.catSet = append(m.catSet, set)
	m.slots[id] = slot
	m.live++
	m.countVisible(slot, 1)
}

func (m *slotMeta) tombstone(slot int32) {
	if m.deleted[slot] {
		return
	}
	m.countVisible(slot, -1)
	m.deleted[slot] = true
	m.live--
	if m.slots[m.ids[slot]] == slot {
		delete(m.slots, m.ids[slot])
	}
}

// liveAcceptor links new nodes to live slots only.
type liveAcceptor struct{ meta *slotMeta }

func (a liveAcceptor) accept(slot int32) bool { return !a.meta.deleted[slot] }

// queryAcceptor applies the query filter: live, visible and, when categories
// were requested, a member of a matching category set.
type queryAcceptor struct {
	meta  *slotMeta
	match []int8 // per category set, nil without a category filter
}

func (a *queryAcceptor) accept(slot int32) bool {
	if a.meta.deleted[slot] || a.meta.hidden[slot] {
		return false
	}
	return a.match == nil || a.match[a.meta.catSet[slot]] == 1
}

// matchSets marks the category sets containing all requested categories and
// returns how many visible vectors they cover.
func (m *slotMeta) matchSets(categories []string, buffer []int8) ([]int8, int) {
	if len(categories) == 0 {
		return nil, m.visible
	}
	wanted := make([]uint32, 0, len(categories))
	for _, category := range categories {
		id, found := m.catIDs[category]
		if !found {
			return nil, 0
		}
		wanted = append(wanted, id)
	}
	if cap(buffer) < len(m.sets) {
		buffer = make([]int8, len(m.sets))
	}
	buffer = buffer[:len(m.sets)]
	matching := 0
	for i, set := range m.sortedSets {
		buffer[i] = 0
		if containsAllUint32(set, wanted) {
			buffer[i] = 1
			matching += m.setVisible[i]
		}
	}
	return buffer, matching
}

func containsAllUint32(sortedSet, wanted []uint32) bool {
	for _, id := range wanted {
		if _, found := slices.BinarySearch(sortedSet, id); !found {
			return false
		}
	}
	return true
}

type hnswCollection struct {
	name     string
	dir      string
	dim      int // 0 for sparse collections
	distance Distance
	opts     hnswOptions

	mu     sync.RWMutex
	meta   *slotMeta
	dense  *denseIndex // created by the first vector, which decides the precision
	sparse *sparseIndex

	version      uint64 // bumped by every change worth persisting
	savedVersion uint64
	lastWrite    time.Time
	dropped      bool

	// ioMu serializes snapshots, compaction and removal of the directory.
	ioMu sync.Mutex
}

func newHNSWCollection(name, dir string, dim int, distance Distance, opts hnswOptions) *hnswCollection {
	collection := &hnswCollection{name: name, dir: dir, dim: dim, distance: distance, opts: opts, meta: newSlotMeta(), version: 1}
	if dim == 0 {
		collection.sparse = newSparseIndex()
	}
	return collection
}

func (c *hnswCollection) totalSlots() int { return len(c.meta.ids) }

func (c *hnswCollection) tombstones() int { return c.totalSlots() - c.meta.live }

func (c *hnswCollection) quietFor(d time.Duration) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return time.Since(c.lastWrite) >= d
}

func (c *hnswCollection) needsCompaction() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.dropped {
		return false
	}
	if c.dense != nil && c.meta.live > 0 {
		// A graph loaded from a snapshot keeps the parameters it was built
		// with; rebuild it when the configured ones differ.
		built, configured := c.dense.params, c.opts.params
		if built.M != configured.M || built.M0 != configured.M0 || built.EFConstruction != configured.EFConstruction {
			return true
		}
	}
	dead := c.tombstones()
	return dead >= hnswMinTombstones && float64(dead) > c.opts.compactRatio*float64(c.totalSlots())
}

func (c *hnswCollection) publishGauges() {
	c.mu.RLock()
	defer c.mu.RUnlock()
	c.publishGaugesLocked()
}

func (c *hnswCollection) publishGaugesLocked() {
	if c.dropped {
		return
	}
	hnswVectors.WithLabelValues(c.name).Set(float64(c.meta.live))
	hnswTombstones.WithLabelValues(c.name).Set(float64(c.tombstones()))
	hnswBytes.WithLabelValues(c.name).Set(float64(c.bytesLocked()))
}

func (c *hnswCollection) bytesLocked() int64 {
	size := int64(len(c.meta.ids)) * (16 + 1 + 1 + 8 + 4 + 48)
	for _, id := range c.meta.ids {
		size += int64(len(id))
	}
	if c.dense != nil {
		size += c.dense.bytes()
	}
	if c.sparse != nil {
		size += c.sparse.bytes()
	}
	return size
}

func validateDense(vector Vector, dim int) error {
	if vector.IsSparse() || vector.Len() != dim {
		return errors.Errorf("vector %s has dimension %d, collection expects %d", vector.Id, vector.Len(), dim)
	}
	if len(vector.HValues) > 0 {
		for _, bits := range vector.HValues {
			if bits&0x7c00 == 0x7c00 { // FP16 exponent all ones: Inf or NaN
				return errors.Errorf("vector %s contains a non-finite value", vector.Id)
			}
		}
		return nil
	}
	for _, value := range vector.Values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return errors.Errorf("vector %s contains a non-finite value", vector.Id)
		}
	}
	return nil
}

// normalizeSparse returns the coordinates in increasing index order.
func normalizeSparse(vector Vector) ([]uint32, []float32, error) {
	if len(vector.Indices) == 0 || len(vector.Indices) != len(vector.Values) {
		return nil, nil, errors.Errorf("vector %s is not a sparse vector", vector.Id)
	}
	sorted := true
	for i := 1; i < len(vector.Indices); i++ {
		if vector.Indices[i] <= vector.Indices[i-1] {
			sorted = false
			break
		}
	}
	indices, values := vector.Indices, vector.Values
	if !sorted {
		order := make([]int, len(indices))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(i, j int) bool { return vector.Indices[order[i]] < vector.Indices[order[j]] })
		indices, values = make([]uint32, len(order)), make([]float32, len(order))
		for i, position := range order {
			indices[i], values[i] = vector.Indices[position], vector.Values[position]
			if i > 0 && indices[i] == indices[i-1] {
				return nil, nil, errors.Errorf("vector %s has a duplicate coordinate %d", vector.Id, indices[i])
			}
		}
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, errors.Errorf("vector %s contains a non-finite value", vector.Id)
		}
	}
	return indices, values, nil
}

// maxNorm is the largest Euclidean norm of the batch; see denseIndex.phi.
func maxNorm(vectors []Vector) float32 {
	var largest float32
	for _, vector := range vectors {
		values := vector.Float32Values()
		largest = max(largest, floats.Dot(values, values))
	}
	return float32(math.Sqrt(float64(largest)))
}

// losslessFP16 reports whether every value survives a round trip through FP16.
func losslessFP16(vectors []Vector) bool {
	var encoded []uint16
	var decoded []float32
	for _, vector := range vectors {
		if len(vector.HValues) > 0 {
			continue // already FP16 bits
		}
		if cap(encoded) < len(vector.Values) {
			encoded, decoded = make([]uint16, len(vector.Values)), make([]float32, len(vector.Values))
		}
		encoded, decoded = encoded[:len(vector.Values)], decoded[:len(vector.Values)]
		floats.FromFloat32To(encoded, vector.Values)
		floats.ToFloat32To(decoded, encoded)
		for i, value := range vector.Values {
			if decoded[i] != value {
				return false
			}
		}
	}
	return true
}

func (c *hnswCollection) upsert(ctx context.Context, vectors []Vector) error {
	type sparseValues struct {
		indices []uint32
		values  []float32
	}
	var normalized []sparseValues
	for _, vector := range vectors {
		if vector.Id == "" {
			return errors.New("vector id is empty")
		}
		if c.dim > 0 {
			if err := validateDense(vector, c.dim); err != nil {
				return err
			}
			continue
		}
		indices, values, err := normalizeSparse(vector)
		if err != nil {
			return err
		}
		normalized = append(normalized, sparseValues{indices, values})
	}

	var inserted, replaced, unchanged, updated int
	var encoded []uint16
	for i, vector := range vectors {
		if err := ctx.Err(); err != nil {
			return errors.WithStack(err)
		}
		// One lock per vector: a bulk load must not starve neighbor queries.
		c.mu.Lock()
		if c.dropped {
			c.mu.Unlock()
			return fmt.Errorf("collection %s: %w", c.name, storage.ErrNotFound)
		}
		if c.dim > 0 && c.dense == nil {
			fp16 := c.opts.precision == hnswPrecisionFP16 ||
				c.opts.precision == hnswPrecisionAuto && losslessFP16(vectors)
			c.dense = newDenseIndex(c.dim, c.distance, fp16, c.opts.params)
			c.dense.phi = hnswPhiHeadroom * maxNorm(vectors)
		}
		timestamp := vector.Timestamp.UnixMilli()
		set := c.meta.internSet(vector.Categories)
		slot, exists := c.meta.slots[vector.Id]
		same := false
		if exists {
			if c.dim > 0 {
				same = c.sameDense(slot, vector, &encoded)
			} else {
				same = c.sparse.equal(slot, normalized[i].indices, normalized[i].values)
			}
		}
		switch {
		case same && c.meta.hidden[slot] == vector.IsHidden && c.meta.catSet[slot] == set:
			// Only the timestamp moves. It keeps the vector alive through the
			// cycle's DeleteVectors and is rewritten by every cycle, so it is
			// not worth a snapshot on its own.
			c.meta.timestamps[slot] = timestamp
			unchanged++
		case same:
			c.meta.countVisible(slot, -1)
			c.meta.hidden[slot] = vector.IsHidden
			c.meta.catSet[slot] = set
			c.meta.countVisible(slot, 1)
			c.meta.timestamps[slot] = timestamp
			c.version++
			updated++
		default:
			if exists {
				c.meta.tombstone(slot)
				replaced++
			} else {
				inserted++
			}
			if c.dim > 0 {
				c.dense.addVector(vector, liveAcceptor{c.meta})
			} else {
				c.sparse.add(normalized[i].indices, normalized[i].values)
			}
			c.meta.appendSlot(vector.Id, vector.IsHidden, timestamp, set)
			c.version++
		}
		c.lastWrite = time.Now()
		c.mu.Unlock()
	}
	hnswUpsertsTotal.WithLabelValues(c.name, "inserted").Add(float64(inserted))
	hnswUpsertsTotal.WithLabelValues(c.name, "replaced").Add(float64(replaced))
	hnswUpsertsTotal.WithLabelValues(c.name, "unchanged").Add(float64(unchanged))
	hnswUpsertsTotal.WithLabelValues(c.name, "metadata").Add(float64(updated))
	c.publishGauges()
	return nil
}

func (c *hnswCollection) sameDense(slot int32, vector Vector, encoded *[]uint16) bool {
	offset := int(slot) * c.dim
	if !c.dense.fp16 {
		stored := c.dense.vec32[offset : offset+c.dim]
		for i, value := range vector.Float32Values() {
			if math.Float32bits(stored[i]) != math.Float32bits(value) {
				return false
			}
		}
		return true
	}
	bits := vector.HValues
	if len(bits) == 0 {
		if cap(*encoded) < c.dim {
			*encoded = make([]uint16, c.dim)
		}
		*encoded = (*encoded)[:c.dim]
		floats.FromFloat32To(*encoded, vector.Values)
		bits = *encoded
	}
	return slices.Equal(c.dense.vec16[offset:offset+c.dim], bits)
}

func (c *hnswCollection) deleteBefore(timestamp time.Time) {
	cutoff := timestamp.UnixMilli()
	c.mu.Lock()
	deleted := 0
	for slot := range c.meta.ids {
		if !c.meta.deleted[slot] && c.meta.timestamps[slot] < cutoff {
			c.meta.tombstone(int32(slot))
			deleted++
		}
	}
	if deleted > 0 {
		c.version++
		c.lastWrite = time.Now()
	}
	c.publishGaugesLocked()
	c.mu.Unlock()
	hnswDeletesTotal.WithLabelValues(c.name).Add(float64(deleted))
}

func (c *hnswCollection) get(ids []string) []Vector {
	c.mu.RLock()
	defer c.mu.RUnlock()
	vectors := make([]Vector, 0, len(ids))
	for _, id := range ids {
		slot, found := c.meta.slots[id]
		if !found {
			continue
		}
		vector := Vector{
			Id:         id,
			IsHidden:   c.meta.hidden[slot],
			Categories: c.meta.categories(slot),
			Timestamp:  time.UnixMilli(c.meta.timestamps[slot]).UTC(),
		}
		if c.dim > 0 {
			vector.Values = c.dense.vector(make([]float32, c.dim), slot)
		} else {
			indices, values := c.sparse.vector(slot)
			vector.Indices = append([]uint32(nil), indices...)
			vector.Values = append([]float32(nil), values...)
		}
		vectors = append(vectors, vector)
	}
	return vectors
}

func (c *hnswCollection) query(q Vector, categories []string, topK int) ([]ScoredVector, error) {
	start := time.Now()
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.dim > 0 {
		if q.IsSparse() || q.Len() != c.dim {
			return nil, errors.Errorf("query has dimension %d, collection %s expects %d", q.Len(), c.name, c.dim)
		}
	} else if len(q.Indices) == 0 || len(q.Indices) != len(q.Values) {
		return nil, errors.Errorf("collection %s expects a sparse query", c.name)
	}

	mode := "graph"
	var found []heapItem
	if c.dim == 0 {
		mode = "inverted"
		indices, values, err := normalizeSparse(q)
		if err != nil {
			return nil, err
		}
		match, matching := c.meta.matchSets(categories, nil)
		if matching == 0 {
			return []ScoredVector{}, nil
		}
		found = c.sparse.search(indices, values, topK, &queryAcceptor{meta: c.meta, match: match})
	} else {
		if c.dense == nil {
			return []ScoredVector{}, nil
		}
		s := c.dense.getScratch()
		defer c.dense.putScratch(s)
		match, matching := c.meta.matchSets(categories, s.setMatch)
		if match != nil {
			s.setMatch = match
		}
		if matching == 0 {
			return []ScoredVector{}, nil
		}
		accept := &queryAcceptor{meta: c.meta, match: match}
		k := min(topK, matching)
		values := q.Values
		if len(q.HValues) > 0 {
			values = s.query[:c.dim]
			floats.ToFloat32To(values, q.HValues)
		}
		if matching < c.opts.scanMin || float64(matching) < c.opts.scanRatio*float64(c.totalSlots()) {
			mode = "scan"
			found = c.dense.scan(s, values, k, accept)
		} else {
			found = c.dense.search(s, values, k, max(c.opts.params.EFSearch, k), accept)
		}
	}

	results := make([]ScoredVector, len(found))
	for i, item := range found {
		results[i] = ScoredVector{
			Vector: Vector{
				Id:         c.meta.ids[item.slot],
				Categories: c.meta.categories(item.slot),
				Timestamp:  time.UnixMilli(c.meta.timestamps[item.slot]),
			},
			// Same convention as the xvec backend: larger is better, and the
			// Euclidean score is the negated squared distance.
			Score: -item.dist,
		}
	}
	hnswQuerySeconds.WithLabelValues(mode).Observe(time.Since(start).Seconds())
	return results, nil
}

// compact rebuilds the collection without its tombstones. The dense graph is
// rebuilt from a frozen prefix of the slabs without holding the lock; writes
// that arrive meanwhile are replayed onto the new graph before the swap.
func (c *hnswCollection) compact(stop <-chan struct{}) {
	c.ioMu.Lock()
	defer c.ioMu.Unlock()
	start := time.Now()
	var ok bool
	if c.dim == 0 {
		ok = c.compactSparse()
	} else {
		ok = c.compactDense(stop)
	}
	if !ok {
		return
	}
	hnswCompactionsTotal.WithLabelValues(c.name).Inc()
	hnswCompactionSeconds.WithLabelValues(c.name).Set(time.Since(start).Seconds())
	c.publishGauges()
	log.Logger().Info("compacted vector collection",
		zap.String("collection", c.name), zap.Duration("used_time", time.Since(start)))
}

// remapMeta builds the metadata of a compacted collection. order lists the old
// slot of every new slot.
func (c *hnswCollection) remapMeta(order []int32) *slotMeta {
	old := c.meta
	next := newSlotMeta()
	next.catNames, next.catIDs, next.sets, next.sortedSets, next.setIDs = old.catNames, old.catIDs, old.sets, old.sortedSets, old.setIDs
	next.setVisible = make([]int, len(old.sets))
	for _, slot := range order {
		next.appendSlot(old.ids[slot], old.hidden[slot], old.timestamps[slot], old.catSet[slot])
		if old.deleted[slot] {
			next.tombstone(int32(len(next.ids) - 1))
		}
	}
	// appendSlot claimed the id for tombstoned slots too; point ids at their
	// live slot again.
	for slot, id := range next.ids {
		if !next.deleted[slot] {
			next.slots[id] = int32(slot)
		}
	}
	return next
}

func (c *hnswCollection) compactSparse() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dropped {
		return false
	}
	next := newSparseIndex()
	order := make([]int32, 0, c.meta.live)
	for slot := range c.meta.ids {
		if c.meta.deleted[slot] {
			continue
		}
		indices, values := c.sparse.vector(int32(slot))
		next.add(indices, values)
		order = append(order, int32(slot))
	}
	c.meta = c.remapMeta(order)
	c.sparse = next
	c.version++
	return true
}

func (c *hnswCollection) compactDense(stop <-chan struct{}) bool {
	c.mu.RLock()
	old := c.dense
	if c.dropped || old == nil {
		c.mu.RUnlock()
		return false
	}
	frozen := old.slots()
	// Slots below frozen are immutable, so these headers stay valid while
	// writers append behind them.
	vec16, vec32 := old.vec16, old.vec32
	deleted := append([]bool(nil), c.meta.deleted[:frozen]...)
	c.mu.RUnlock()

	next := newDenseIndex(c.dim, c.distance, old.fp16, c.opts.params)
	order := make([]int32, 0, frozen)
	buffer := make([]float32, c.dim)
	read := func(slot int) []float32 {
		offset := slot * c.dim
		if old.fp16 {
			floats.ToFloat32To(buffer, vec16[offset:offset+c.dim])
		} else {
			copy(buffer, vec32[offset:offset+c.dim])
		}
		return buffer
	}
	if c.distance == Dot {
		// A rebuild knows every vector, so the norm bound can be tight.
		var largest float32
		for slot := 0; slot < frozen; slot++ {
			if !deleted[slot] {
				v := read(slot)
				largest = max(largest, floats.Dot(v, v))
			}
		}
		next.phi = float32(math.Sqrt(float64(largest))) * 1.1
	}
	for slot := 0; slot < frozen; slot++ {
		if deleted[slot] {
			continue
		}
		if slot%1024 == 0 {
			select {
			case <-stop:
				return false
			default:
			}
			c.mu.RLock()
			dropped := c.dropped
			c.mu.RUnlock()
			if dropped {
				return false
			}
		}
		next.add(read(slot), nil)
		order = append(order, int32(slot))
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dropped || c.dense != old {
		return false
	}
	// Replay what was written during the rebuild. Slots that died meanwhile
	// are carried over as tombstones by remapMeta.
	for slot := frozen; slot < old.slots(); slot++ {
		if c.meta.deleted[slot] {
			continue
		}
		next.add(old.vector(buffer, int32(slot)), nil)
		order = append(order, int32(slot))
	}
	c.meta = c.remapMeta(order)
	c.dense = next
	c.version++
	return true
}

// HNSWCollectionStats describes one collection of the hnsw:// store.
type HNSWCollectionStats struct {
	Vectors    int    // live vectors
	Tombstones int    // replaced or deleted vectors awaiting compaction
	Bytes      int64  // estimated heap
	Precision  string // fp16, fp32, sparse, or empty before the first vector
}

// CollectionStats reports the size and storage precision of a collection.
func (db *HNSWStore) CollectionStats(name string) (HNSWCollectionStats, error) {
	collection, err := db.collection(name)
	if err != nil {
		return HNSWCollectionStats{}, err
	}
	collection.mu.RLock()
	defer collection.mu.RUnlock()
	stats := HNSWCollectionStats{Vectors: collection.meta.live, Tombstones: collection.tombstones(), Bytes: collection.bytesLocked()}
	switch {
	case collection.sparse != nil:
		stats.Precision = "sparse"
	case collection.dense == nil:
	case collection.dense.fp16:
		stats.Precision = hnswPrecisionFP16
	default:
		stats.Precision = hnswPrecisionFP32
	}
	return stats, nil
}
