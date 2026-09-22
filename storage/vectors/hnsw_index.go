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
	"math"
	"math/rand"
	"slices"
	"sync"

	"github.com/gorse-io/gorse/common/floats"
)

// VideoHub fork: the dense index of the hnsw:// vector store. It is an HNSW
// graph kept in flat slabs (vectors, level-0 links) so that it persists with a
// handful of sequential writes, grows by appending and costs the garbage
// collector nothing per vector. Searches borrow a pooled scratch (epoch-marked
// visited array, typed heaps, decode buffers) and do not allocate.
//
// The index is not safe for concurrent mutation: hnswCollection serializes
// writers and lets searches share a read lock.

const hnswMaxLevel = 16

type hnswParams struct {
	M              int // links per node on the upper layers
	M0             int // links per node on layer 0
	EFConstruction int
	EFSearch       int
}

// heapItem is a slot with its distance to the query; smaller is closer.
type heapItem struct {
	dist float32
	slot int32
}

// distHeap is a binary heap of heapItem without interface boxing. With max set
// the farthest item is on top (the result set), otherwise the closest (the
// candidate queue).
type distHeap struct {
	items []heapItem
	max   bool
}

func (h *distHeap) reset(max bool) {
	h.items = h.items[:0]
	h.max = max
}

func (h *distHeap) len() int { return len(h.items) }

func (h *distHeap) top() heapItem { return h.items[0] }

func (h *distHeap) before(a, b heapItem) bool {
	if h.max {
		return a.dist > b.dist
	}
	return a.dist < b.dist
}

func (h *distHeap) push(item heapItem) {
	h.items = append(h.items, item)
	i := len(h.items) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if !h.before(h.items[i], h.items[parent]) {
			break
		}
		h.items[i], h.items[parent] = h.items[parent], h.items[i]
		i = parent
	}
}

func (h *distHeap) pop() heapItem {
	top := h.items[0]
	last := len(h.items) - 1
	h.items[0] = h.items[last]
	h.items = h.items[:last]
	i := 0
	for {
		left, right, best := 2*i+1, 2*i+2, i
		if left < last && h.before(h.items[left], h.items[best]) {
			best = left
		}
		if right < last && h.before(h.items[right], h.items[best]) {
			best = right
		}
		if best == i {
			break
		}
		h.items[i], h.items[best] = h.items[best], h.items[i]
		i = best
	}
	return top
}

// searchScratch is everything one search or insert needs. It is pooled per
// index so that a neighbor query for every item does not turn into one visited
// set, two heaps and a result slice of garbage per query.
type searchScratch struct {
	visited    []uint32
	epoch      uint32
	candidates distHeap
	results    distHeap
	decode     []float32 // FP16 -> FP32 buffer for the stored side of a distance
	base       []float32 // FP32 copy of a stored vector used as the query side
	query      []float32 // FP32 copy of the vector being inserted
	linked     []heapItem
	baseSlot   int32
	selected   []heapItem
	pool       []heapItem
	entry      []heapItem
	setMatch   []int8
}

func (s *searchScratch) nextEpoch(slots int) {
	if len(s.visited) < slots {
		s.visited = append(s.visited, make([]uint32, slots-len(s.visited))...)
	}
	s.epoch++
	if s.epoch == 0 {
		clear(s.visited)
		s.epoch = 1
	}
}

// slotAcceptor decides which slots may appear in a result set. Traversal still
// walks through rejected slots, otherwise a filter would disconnect the graph.
type slotAcceptor interface {
	accept(slot int32) bool
}

type denseIndex struct {
	dim       int
	distance  Distance
	fp16      bool
	params    hnswParams
	levelMult float64
	rng       *rand.Rand

	// Slot-indexed slabs. A slot is never rewritten once its vector is stored;
	// a changed vector gets a new slot and the old one becomes a tombstone that
	// keeps routing searches until the next compaction.
	vec16  []uint16
	vec32  []float32
	norms  []float32 // Euclidean: |v|^2, Cosine: 1/|v|, Dot: sqrt(phi^2 - |v|^2)
	levels []uint8
	links0 []int32
	count0 []uint16
	upper  [][]int32 // level l >= 1 lives at [(l-1)*(M+1)]: count, then M links

	entry    int32
	maxLevel int

	// phi bounds the norm of the vectors of a Dot collection. Inner product is
	// not a metric, so a graph built on it links poorly; appending the
	// coordinate sqrt(phi^2 - |v|^2) to every stored vector turns maximum inner
	// product into nearest neighbor search while a query (extra coordinate 0)
	// still ranks by plain inner product.
	phi float32

	scratch sync.Pool
}

func newDenseIndex(dim int, distance Distance, fp16 bool, params hnswParams) *denseIndex {
	x := &denseIndex{
		dim:       dim,
		distance:  distance,
		fp16:      fp16,
		params:    params,
		levelMult: 1 / math.Log(float64(params.M)),
		rng:       rand.New(rand.NewSource(int64(dim)*7919 + int64(params.M))),
		entry:     -1,
	}
	x.scratch.New = func() any {
		return &searchScratch{decode: make([]float32, dim), base: make([]float32, dim), query: make([]float32, dim), baseSlot: -1}
	}
	return x
}

func (x *denseIndex) slots() int { return len(x.levels) }

func (x *denseIndex) getScratch() *searchScratch {
	s := x.scratch.Get().(*searchScratch)
	s.baseSlot = -1
	return s
}

func (x *denseIndex) putScratch(s *searchScratch) { x.scratch.Put(s) }

// queryNorm is the query side of the norm trick, see distanceTo.
func (x *denseIndex) queryNorm(q []float32) float32 {
	switch x.distance {
	case Euclidean:
		return floats.Dot(q, q)
	case Cosine:
		n := floats.Dot(q, q)
		if n == 0 {
			return 0
		}
		return float32(1 / math.Sqrt(float64(n)))
	default:
		return 0
	}
}

// distanceTo evaluates every metric with one SIMD dot product. Euclidean is
// |a|^2 + |q|^2 - 2a.q with both norms known up front: the graph only needs an
// ordering, so nothing here takes a square root.
func (x *denseIndex) distanceTo(s *searchScratch, q []float32, qn float32, slot int32) float32 {
	offset := int(slot) * x.dim
	var dot float32
	if x.fp16 {
		dot = floats.DotFP16(x.vec16[offset:offset+x.dim], q, s.decode)
	} else {
		dot = floats.Dot(x.vec32[offset:offset+x.dim], q)
	}
	switch x.distance {
	case Euclidean:
		d := x.norms[slot] + qn - 2*dot
		if d < 0 {
			return 0
		}
		return d
	case Cosine:
		return -dot * x.norms[slot] * qn
	default:
		// qn is the extra coordinate of a stored vector and 0 for a query.
		return -dot - qn*x.norms[slot]
	}
}

// vector copies the stored vector of a slot into dst as FP32.
func (x *denseIndex) vector(dst []float32, slot int32) []float32 {
	offset := int(slot) * x.dim
	dst = dst[:x.dim]
	if x.fp16 {
		floats.ToFloat32To(dst, x.vec16[offset:offset+x.dim])
	} else {
		copy(dst, x.vec32[offset:offset+x.dim])
	}
	return dst
}

// between is the distance between two stored vectors.
func (x *denseIndex) between(s *searchScratch, a, b int32) float32 {
	if s.baseSlot != a {
		x.vector(s.base, a)
		s.baseSlot = a
	}
	return x.distanceTo(s, s.base, x.norms[a], b)
}

func (x *denseIndex) neighbors(slot int32, level int) []int32 {
	if level == 0 {
		offset := int(slot) * x.params.M0
		return x.links0[offset : offset+int(x.count0[slot])]
	}
	block := x.upper[slot][(level-1)*(x.params.M+1):]
	return block[1 : 1+block[0]]
}

func (x *denseIndex) setNeighbors(slot int32, level int, list []heapItem) {
	if level == 0 {
		offset := int(slot) * x.params.M0
		for i, item := range list {
			x.links0[offset+i] = item.slot
		}
		x.count0[slot] = uint16(len(list))
		return
	}
	block := x.upper[slot][(level-1)*(x.params.M+1):]
	for i, item := range list {
		block[1+i] = item.slot
	}
	block[0] = int32(len(list))
}

func (x *denseIndex) capacity(level int) int {
	if level == 0 {
		return x.params.M0
	}
	return x.params.M
}

// appendVector stores a vector in a new slot without linking it.
func (x *denseIndex) appendVector(v []float32) int32 {
	slot := int32(len(x.levels))
	if x.fp16 {
		x.vec16 = slices.Grow(x.vec16, x.dim)[:len(x.vec16)+x.dim]
		floats.FromFloat32To(x.vec16[len(x.vec16)-x.dim:], v)
	} else {
		x.vec32 = append(x.vec32, v...)
	}
	x.appendSlotMeta(slot)
	return slot
}

// appendSlotMeta computes the norm of the vector just stored in slot and
// allocates its link storage.
func (x *denseIndex) appendSlotMeta(slot int32) {
	// Norms are taken from the stored precision so that the distance of a
	// vector to itself is exactly zero.
	s := x.getScratch()
	stored := x.vector(s.decode, slot)
	norm := x.queryNorm(stored)
	if x.distance == Dot {
		norm = 0
		if rest := x.phi*x.phi - floats.Dot(stored, stored); rest > 0 {
			norm = float32(math.Sqrt(float64(rest)))
		}
	}
	x.putScratch(s)
	x.norms = append(x.norms, norm)
	x.levels = append(x.levels, 0)
	x.links0 = append(x.links0, make([]int32, x.params.M0)...)
	x.count0 = append(x.count0, 0)
	x.upper = append(x.upper, nil)
}

// add stores a vector and links it into the graph. accept limits which slots
// the new node links to (tombstones stay reachable but gain no new links).
func (x *denseIndex) add(v []float32, accept slotAcceptor) int32 {
	slot := x.appendVector(v)
	x.link(slot, accept)
	return slot
}

// addVector stores a Vector that may carry FP16 bits (HValues). Bits go into
// an FP16 index without a conversion; anything else takes the FP32 path.
func (x *denseIndex) addVector(v Vector, accept slotAcceptor) int32 {
	if len(v.HValues) == 0 || !x.fp16 {
		return x.add(v.Float32Values(), accept)
	}
	slot := int32(len(x.levels))
	x.vec16 = append(x.vec16, v.HValues...)
	x.appendSlotMeta(slot)
	x.link(slot, accept)
	return slot
}

func (x *denseIndex) link(slot int32, accept slotAcceptor) {
	level := int(-math.Log(1-x.rng.Float64()) * x.levelMult)
	level = min(level, hnswMaxLevel)
	x.levels[slot] = uint8(level)
	if level > 0 {
		x.upper[slot] = make([]int32, level*(x.params.M+1))
	}
	if x.entry < 0 {
		x.entry, x.maxLevel = slot, level
		return
	}

	s := x.getScratch()
	defer x.putScratch(s)
	q := x.vector(s.query, slot)
	qn := x.norms[slot]

	current := heapItem{slot: x.entry, dist: x.distanceTo(s, q, qn, x.entry)}
	for l := x.maxLevel; l > level; l-- {
		current = x.greedy(s, q, qn, current, l)
	}
	s.entry = append(s.entry[:0], current)
	for l := min(level, x.maxLevel); l >= 0; l-- {
		found := x.searchLayer(s, q, qn, s.entry, x.params.EFConstruction, l, accept)
		// found is ascending by distance; keep it as the next layer's entry set.
		s.entry = append(s.entry[:0], found...)
		if len(s.entry) == 0 {
			s.entry = append(s.entry, current)
		}
		s.pool = append(s.pool[:0], found...)
		selected := x.selectNeighbors(s, s.pool, x.params.M)
		x.setNeighbors(slot, l, selected)
		// connect reuses s.selected, so walk a copy of the selection.
		s.linked = append(s.linked[:0], selected...)
		for _, neighbor := range s.linked {
			x.connect(s, neighbor.slot, slot, neighbor.dist, l)
		}
	}
	if level > x.maxLevel {
		x.entry, x.maxLevel = slot, level
	}
}

// connect adds the link from -> to, shrinking from's list with the selection
// heuristic when it is full.
func (x *denseIndex) connect(s *searchScratch, from, to int32, dist float32, level int) {
	current := x.neighbors(from, level)
	if slices.Contains(current, to) {
		return
	}
	limit := x.capacity(level)
	if len(current) < limit {
		if level == 0 {
			x.links0[int(from)*x.params.M0+len(current)] = to
			x.count0[from]++
		} else {
			block := x.upper[from][(level-1)*(x.params.M+1):]
			block[1+block[0]] = to
			block[0]++
		}
		return
	}
	s.pool = append(s.pool[:0], heapItem{slot: to, dist: dist})
	for _, neighbor := range current {
		s.pool = append(s.pool, heapItem{slot: neighbor, dist: x.between(s, from, neighbor)})
	}
	slices.SortFunc(s.pool, compareHeapItems)
	x.setNeighbors(from, level, x.selectNeighbors(s, s.pool, limit))
}

func compareHeapItems(a, b heapItem) int {
	switch {
	case a.dist < b.dist:
		return -1
	case a.dist > b.dist:
		return 1
	default:
		return int(a.slot - b.slot)
	}
}

// selectNeighbors is the HNSW selection heuristic: walking candidates from the
// closest, keep one only when it is closer to the base than to every neighbor
// already kept. It spreads links over directions instead of spending all of
// them on one cluster, which is what lets a smaller M keep its recall.
// candidates must be ascending by distance to the base.
func (x *denseIndex) selectNeighbors(s *searchScratch, candidates []heapItem, m int) []heapItem {
	s.selected = s.selected[:0]
	if len(candidates) <= m {
		return append(s.selected, candidates...)
	}
	for _, candidate := range candidates {
		if len(s.selected) == m {
			break
		}
		keep := true
		for _, kept := range s.selected {
			if x.between(s, candidate.slot, kept.slot) < candidate.dist {
				keep = false
				break
			}
		}
		if keep {
			s.selected = append(s.selected, candidate)
		}
	}
	return s.selected
}

func (x *denseIndex) greedy(s *searchScratch, q []float32, qn float32, current heapItem, level int) heapItem {
	for improved := true; improved; {
		improved = false
		for _, neighbor := range x.neighbors(current.slot, level) {
			if d := x.distanceTo(s, q, qn, neighbor); d < current.dist {
				current, improved = heapItem{slot: neighbor, dist: d}, true
			}
		}
	}
	return current
}

// searchLayer returns up to ef accepted slots closest to q, ascending. The
// returned slice belongs to the scratch and is valid until its next use.
func (x *denseIndex) searchLayer(s *searchScratch, q []float32, qn float32, entry []heapItem, ef, level int, accept slotAcceptor) []heapItem {
	s.nextEpoch(x.slots())
	s.candidates.reset(false)
	s.results.reset(true)
	for _, item := range entry {
		if s.visited[item.slot] == s.epoch {
			continue
		}
		s.visited[item.slot] = s.epoch
		s.candidates.push(item)
		if accept == nil || accept.accept(item.slot) {
			s.results.push(item)
		}
	}
	for s.candidates.len() > 0 {
		candidate := s.candidates.pop()
		if s.results.len() >= ef && candidate.dist > s.results.top().dist {
			break
		}
		for _, neighbor := range x.neighbors(candidate.slot, level) {
			if s.visited[neighbor] == s.epoch {
				continue
			}
			s.visited[neighbor] = s.epoch
			d := x.distanceTo(s, q, qn, neighbor)
			if s.results.len() >= ef && d >= s.results.top().dist {
				continue
			}
			s.candidates.push(heapItem{slot: neighbor, dist: d})
			if accept == nil || accept.accept(neighbor) {
				s.results.push(heapItem{slot: neighbor, dist: d})
				if s.results.len() > ef {
					s.results.pop()
				}
			}
		}
	}
	// Drain the max-heap back to front to get ascending order in place.
	n := s.results.len()
	s.pool = slices.Grow(s.pool[:0], n)[:n]
	for i := n - 1; i >= 0; i-- {
		s.pool[i] = s.results.pop()
	}
	return s.pool
}

// search returns the k accepted slots closest to q, ascending by distance.
func (x *denseIndex) search(s *searchScratch, q []float32, k, ef int, accept slotAcceptor) []heapItem {
	if x.entry < 0 || k <= 0 {
		return nil
	}
	qn := x.queryNorm(q)
	current := heapItem{slot: x.entry, dist: x.distanceTo(s, q, qn, x.entry)}
	for l := x.maxLevel; l > 0; l-- {
		current = x.greedy(s, q, qn, current, l)
	}
	s.entry = append(s.entry[:0], current)
	found := x.searchLayer(s, q, qn, s.entry, max(ef, k), 0, accept)
	if len(found) > k {
		found = found[:k]
	}
	return found
}

// scan is the exact search used when a filter leaves so few slots that walking
// the graph past everything it rejects would cost more than looking at them.
func (x *denseIndex) scan(s *searchScratch, q []float32, k int, accept slotAcceptor) []heapItem {
	qn := x.queryNorm(q)
	s.results.reset(true)
	for slot := int32(0); int(slot) < x.slots(); slot++ {
		if !accept.accept(slot) {
			continue
		}
		d := x.distanceTo(s, q, qn, slot)
		if s.results.len() < k {
			s.results.push(heapItem{slot: slot, dist: d})
		} else if d < s.results.top().dist {
			s.results.pop()
			s.results.push(heapItem{slot: slot, dist: d})
		}
	}
	n := s.results.len()
	s.pool = slices.Grow(s.pool[:0], n)[:n]
	for i := n - 1; i >= 0; i-- {
		s.pool[i] = s.results.pop()
	}
	return s.pool
}

// bytes estimates the heap held by the index.
func (x *denseIndex) bytes() int64 {
	size := int64(len(x.vec16))*2 + int64(len(x.vec32))*4 + int64(len(x.norms))*4 +
		int64(len(x.levels)) + int64(len(x.links0))*4 + int64(len(x.count0))*2 + int64(len(x.upper))*24
	for _, links := range x.upper {
		size += int64(len(links)) * 4
	}
	return size
}
