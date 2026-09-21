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
	"slices"
	"sync"
)

// VideoHub fork: the sparse index of the hnsw:// vector store. Sparse
// collections (tag and feedback based similarity) are scored by inner product,
// so an inverted index gives exact results while only touching vectors that
// share a coordinate with the query.

type sparsePosting struct {
	slot  int32
	value float32
}

type sparseScratch struct {
	scores  []float32
	seen    []uint32
	epoch   uint32
	touched []int32
	results distHeap
	out     []heapItem
}

type sparseIndex struct {
	// Slot i owns indices[starts[i]:starts[i+1]] and the matching values.
	starts   []uint64
	indices  []uint32
	values   []float32
	postings map[uint32][]sparsePosting

	scratch sync.Pool
}

func newSparseIndex() *sparseIndex {
	x := &sparseIndex{starts: []uint64{0}, postings: make(map[uint32][]sparsePosting)}
	x.scratch.New = func() any { return new(sparseScratch) }
	return x
}

func (x *sparseIndex) slots() int { return len(x.starts) - 1 }

func (x *sparseIndex) vector(slot int32) ([]uint32, []float32) {
	from, to := x.starts[slot], x.starts[slot+1]
	return x.indices[from:to], x.values[from:to]
}

func (x *sparseIndex) equal(slot int32, indices []uint32, values []float32) bool {
	storedIndices, storedValues := x.vector(slot)
	return slices.Equal(storedIndices, indices) && slices.Equal(storedValues, values)
}

// add stores a vector whose indices are strictly increasing.
func (x *sparseIndex) add(indices []uint32, values []float32) int32 {
	slot := int32(x.slots())
	x.indices = append(x.indices, indices...)
	x.values = append(x.values, values...)
	x.starts = append(x.starts, uint64(len(x.indices)))
	for i, index := range indices {
		x.postings[index] = append(x.postings[index], sparsePosting{slot: slot, value: values[i]})
	}
	return slot
}

// search returns the k accepted slots with the largest positive inner product,
// best first. heapItem.dist holds the negated score.
func (x *sparseIndex) search(indices []uint32, values []float32, k int, accept slotAcceptor) []heapItem {
	s := x.scratch.Get().(*sparseScratch)
	defer x.scratch.Put(s)
	if slots := x.slots(); len(s.scores) < slots {
		s.scores = append(s.scores, make([]float32, slots-len(s.scores))...)
		s.seen = append(s.seen, make([]uint32, slots-len(s.seen))...)
	}
	s.epoch++
	if s.epoch == 0 {
		clear(s.seen)
		s.epoch = 1
	}
	s.touched = s.touched[:0]
	for i, index := range indices {
		weight := values[i]
		for _, posting := range x.postings[index] {
			if s.seen[posting.slot] != s.epoch {
				s.seen[posting.slot] = s.epoch
				s.scores[posting.slot] = 0
				s.touched = append(s.touched, posting.slot)
			}
			s.scores[posting.slot] += weight * posting.value
		}
	}
	s.results.reset(true)
	for _, slot := range s.touched {
		score := s.scores[slot]
		if score == 0 || !accept.accept(slot) {
			continue
		}
		item := heapItem{slot: slot, dist: -score}
		if s.results.len() < k {
			s.results.push(item)
		} else if item.dist < s.results.top().dist {
			s.results.pop()
			s.results.push(item)
		}
	}
	n := s.results.len()
	out := make([]heapItem, n)
	for i := n - 1; i >= 0; i-- {
		out[i] = s.results.pop()
	}
	return out
}

func (x *sparseIndex) bytes() int64 {
	size := int64(len(x.starts))*8 + int64(len(x.indices))*4 + int64(len(x.values))*4
	for _, postings := range x.postings {
		size += int64(cap(postings))*8 + 48
	}
	return size
}
