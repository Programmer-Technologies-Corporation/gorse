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
	"bufio"
	"encoding/binary"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/gorse-io/gorse/common/log"
	"github.com/pkg/errors"
	"go.uber.org/zap"
)

// VideoHub fork: snapshot format of an hnsw:// collection. One little-endian
// file holds the metadata, the vectors and the graph, followed by a CRC32 of
// everything before it. A new snapshot is written to a temporary file, the
// previous snapshot is moved aside and the temporary file renamed into place.
// Rename is atomic on POSIX but not on Windows, so a crash between the two
// renames can leave index.bin missing or partial; loading then falls back to
// the moved-aside file, which is complete, so at most the writes since the
// previous snapshot are lost.

const (
	hnswSnapshotMagic   = 0x49485647 // "GVHI"
	hnswSnapshotVersion = 1

	hnswPrecisionUnset byte = 0
	hnswPrecision32    byte = 1
	hnswPrecision16    byte = 2
)

// hnswFrozen is a consistent view of a collection taken under its read lock.
// Append-only data is shared by slice header, data that changes in place is
// copied, so the file can be written without blocking writers.
type hnswFrozen struct {
	version  uint64
	dim      int
	distance Distance

	ids        []string
	hidden     []bool
	deleted    []bool
	timestamps []int64
	catSet     []int32
	catNames   []string
	sets       [][]uint32

	precision byte
	params    hnswParams
	entry     int32
	maxLevel  int
	phi       float32
	vec16     []uint16
	vec32     []float32
	norms     []float32
	levels    []uint8
	links0    []int32
	count0    []uint16
	upper     [][]int32

	starts  []uint64
	indices []uint32
	values  []float32
}

func (c *hnswCollection) freeze() *hnswFrozen {
	n := c.totalSlots()
	frozen := &hnswFrozen{
		version:    c.version,
		dim:        c.dim,
		distance:   c.distance,
		ids:        c.meta.ids[:n],
		hidden:     append([]bool(nil), c.meta.hidden...),
		deleted:    append([]bool(nil), c.meta.deleted...),
		timestamps: append([]int64(nil), c.meta.timestamps...),
		catSet:     append([]int32(nil), c.meta.catSet...),
		catNames:   c.meta.catNames[:len(c.meta.catNames)],
		sets:       c.meta.sets[:len(c.meta.sets)],
		entry:      -1,
	}
	if x := c.dense; x != nil {
		frozen.precision = hnswPrecision32
		if x.fp16 {
			frozen.precision = hnswPrecision16
		}
		frozen.params, frozen.entry, frozen.maxLevel, frozen.phi = x.params, x.entry, x.maxLevel, x.phi
		frozen.vec16, frozen.vec32 = x.vec16[:len(x.vec16)], x.vec32[:len(x.vec32)]
		frozen.norms, frozen.levels = x.norms[:n], x.levels[:n]
		frozen.links0 = append([]int32(nil), x.links0...)
		frozen.count0 = append([]uint16(nil), x.count0...)
		frozen.upper = make([][]int32, n)
		for slot, links := range x.upper {
			if links != nil {
				frozen.upper[slot] = append([]int32(nil), links...)
			}
		}
	}
	if x := c.sparse; x != nil {
		frozen.starts, frozen.indices, frozen.values = x.starts[:n+1], x.indices[:len(x.indices)], x.values[:len(x.values)]
	}
	return frozen
}

// snapshot persists the collection if it changed since the last snapshot.
func (c *hnswCollection) snapshot() error {
	c.ioMu.Lock()
	defer c.ioMu.Unlock()
	c.mu.RLock()
	if c.dropped || c.version == c.savedVersion {
		c.mu.RUnlock()
		return nil
	}
	frozen := c.freeze()
	c.mu.RUnlock()

	start := time.Now()
	size, err := frozen.writeFile(filepath.Join(c.dir, hnswSnapshotFile))
	if err != nil {
		hnswSnapshotFailuresTotal.WithLabelValues(c.name).Inc()
		return errors.WithStack(err)
	}
	c.mu.Lock()
	c.savedVersion = frozen.version
	c.mu.Unlock()
	hnswSnapshotSeconds.WithLabelValues(c.name).Set(time.Since(start).Seconds())
	hnswSnapshotBytes.WithLabelValues(c.name).Set(float64(size))
	log.Logger().Debug("saved vector collection",
		zap.String("collection", c.name), zap.Int64("bytes", size), zap.Duration("used_time", time.Since(start)))
	return nil
}

func (f *hnswFrozen) writeFile(path string) (int64, error) {
	temporary := path + ".tmp"
	file, err := os.Create(temporary)
	if err != nil {
		return 0, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = file.Close()
			_ = os.Remove(temporary)
		}
	}()
	checksum := crc32.NewIEEE()
	w := &snapshotWriter{w: bufio.NewWriterSize(io.MultiWriter(file, checksum), 1<<20)}
	f.encode(w)
	if w.err == nil {
		w.err = w.w.Flush()
	}
	if w.err != nil {
		return 0, w.err
	}
	var trailer [4]byte
	binary.LittleEndian.PutUint32(trailer[:], checksum.Sum32())
	if _, err = file.Write(trailer[:]); err != nil {
		return 0, err
	}
	if err = file.Sync(); err != nil {
		return 0, err
	}
	size, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	if err = file.Close(); err != nil {
		return 0, err
	}
	previous := filepath.Join(filepath.Dir(path), hnswPreviousSnapshot)
	if err = os.Rename(path, previous); err != nil && !os.IsNotExist(err) {
		return 0, err
	}
	if err = os.Rename(temporary, path); err != nil {
		return 0, err
	}
	committed = true
	return size, nil
}

func (f *hnswFrozen) encode(w *snapshotWriter) {
	w.u32(hnswSnapshotMagic)
	w.u32(hnswSnapshotVersion)
	w.u32(uint32(f.dim))
	w.u32(uint32(f.distance))
	w.u32(uint32(f.precision))
	w.u32(uint32(f.params.M))
	w.u32(uint32(f.params.M0))
	w.u32(uint32(f.params.EFConstruction))
	w.u32(uint32(int32(f.entry)))
	w.u32(uint32(f.maxLevel))
	w.u32(math.Float32bits(f.phi))
	w.u64(uint64(len(f.ids)))

	w.u32(uint32(len(f.catNames)))
	for _, name := range f.catNames {
		w.str(name)
	}
	w.u32(uint32(len(f.sets)))
	for _, set := range f.sets {
		w.u32(uint32(len(set)))
		w.u32s(set)
	}
	for _, id := range f.ids {
		w.str(id)
	}
	w.bools(f.hidden)
	w.bools(f.deleted)
	for _, timestamp := range f.timestamps {
		w.u64(uint64(timestamp))
	}
	w.i32s(f.catSet)

	if f.dim == 0 {
		for _, start := range f.starts {
			w.u64(start)
		}
		w.u32s(f.indices)
		w.f32s(f.values)
		return
	}
	if f.precision == hnswPrecisionUnset {
		return
	}
	if f.precision == hnswPrecision16 {
		w.u16s(f.vec16)
	} else {
		w.f32s(f.vec32)
	}
	w.f32s(f.norms)
	w.bytes(f.levels)
	w.u16s(f.count0)
	w.i32s(f.links0)
	upper := 0
	for _, links := range f.upper {
		if links != nil {
			upper++
		}
	}
	w.u32(uint32(upper))
	for slot, links := range f.upper {
		if links != nil {
			w.u32(uint32(slot))
			w.i32s(links)
		}
	}
}

func loadHNSWCollection(name, dir string, opts hnswOptions) (*hnswCollection, error) {
	collection, err := loadHNSWSnapshot(filepath.Join(dir, hnswSnapshotFile), name, dir, opts)
	if err == nil {
		return collection, nil
	}
	previous := filepath.Join(dir, hnswPreviousSnapshot)
	if _, statErr := os.Stat(previous); statErr != nil {
		return nil, err
	}
	log.Logger().Warn("vector collection snapshot is unreadable, loading the previous one",
		zap.String("collection", name), zap.Error(err))
	collection, previousErr := loadHNSWSnapshot(previous, name, dir, opts)
	if previousErr != nil {
		return nil, err
	}
	// The recovered state is older than what was written last; mark it dirty so
	// the next maintenance pass writes a fresh index.bin.
	collection.savedVersion = 0
	return collection, nil
}

func loadHNSWSnapshot(path, name, dir string, opts hnswOptions) (*hnswCollection, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, errors.WithStack(err)
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return nil, errors.WithStack(err)
	}
	if stat.Size() < 4 {
		return nil, errors.Errorf("snapshot %s is truncated", path)
	}
	checksum := crc32.NewIEEE()
	r := &snapshotReader{
		r:     bufio.NewReaderSize(io.TeeReader(io.LimitReader(file, stat.Size()-4), checksum), 1<<20),
		limit: uint64(stat.Size()),
	}
	collection := decodeHNSWCollection(r, name, dir, opts)
	if r.err != nil {
		return nil, errors.Wrapf(r.err, "read snapshot %s", path)
	}
	// Drain what a newer writer may have appended so the checksum covers it.
	if _, err = io.Copy(io.Discard, r.r); err != nil {
		return nil, errors.WithStack(err)
	}
	var trailer [4]byte
	if _, err = file.ReadAt(trailer[:], stat.Size()-4); err != nil {
		return nil, errors.WithStack(err)
	}
	if binary.LittleEndian.Uint32(trailer[:]) != checksum.Sum32() {
		return nil, errors.Errorf("snapshot %s failed its checksum", path)
	}
	return collection, nil
}

func decodeHNSWCollection(r *snapshotReader, name, dir string, opts hnswOptions) *hnswCollection {
	if r.u32() != hnswSnapshotMagic {
		r.fail("not a vector snapshot")
		return nil
	}
	if version := r.u32(); version != hnswSnapshotVersion {
		r.fail("unsupported snapshot version")
		return nil
	}
	dim := int(r.u32())
	distance := Distance(r.u32())
	precision := byte(r.u32())
	params := hnswParams{M: int(r.u32()), M0: int(r.u32()), EFConstruction: int(r.u32()), EFSearch: opts.params.EFSearch}
	entry := int32(r.u32())
	maxLevel := int(r.u32())
	phi := math.Float32frombits(r.u32())
	n := r.count(1)
	if r.err != nil {
		return nil
	}

	collection := newHNSWCollection(name, dir, dim, distance, opts)
	meta := collection.meta
	meta.catNames = make([]string, r.count(4))
	for i := range meta.catNames {
		meta.catNames[i] = r.str()
		meta.catIDs[meta.catNames[i]] = uint32(i)
	}
	for i, sets := 0, r.count(4); i < sets && r.err == nil; i++ {
		set := r.u32s(r.count(4))
		for _, id := range set {
			if int(id) >= len(meta.catNames) {
				r.fail("category set refers to an unknown category")
				return nil
			}
		}
		meta.addSet(set)
	}
	meta.ids = make([]string, n)
	for i := range meta.ids {
		meta.ids[i] = r.str()
	}
	meta.hidden = r.bools(n)
	meta.deleted = r.bools(n)
	meta.timestamps = make([]int64, n)
	for i := range meta.timestamps {
		meta.timestamps[i] = int64(r.u64())
	}
	meta.catSet = r.i32s(n)
	if r.err != nil {
		return nil
	}
	for slot := range meta.ids {
		if set := meta.catSet[slot]; set < 0 || int(set) >= len(meta.sets) {
			r.fail("slot refers to an unknown category set")
			return nil
		}
		if meta.deleted[slot] {
			continue
		}
		meta.slots[meta.ids[slot]] = int32(slot)
		meta.live++
		meta.countVisible(int32(slot), 1)
	}

	if dim == 0 {
		x := collection.sparse
		x.starts = make([]uint64, n+1)
		for i := range x.starts {
			x.starts[i] = r.u64()
		}
		if r.err != nil {
			return nil
		}
		total := x.starts[n]
		for i := 1; i <= n; i++ {
			if x.starts[i] < x.starts[i-1] {
				r.fail("sparse offsets are not increasing")
				return nil
			}
		}
		if total > r.limit {
			r.fail("sparse data is larger than the file")
			return nil
		}
		x.indices = r.u32s(int(total))
		x.values = r.f32s(int(total))
		if r.err != nil {
			return nil
		}
		for slot := 0; slot < n; slot++ {
			if meta.deleted[slot] {
				continue
			}
			indices, values := x.vector(int32(slot))
			for i, index := range indices {
				x.postings[index] = append(x.postings[index], sparsePosting{slot: int32(slot), value: values[i]})
			}
		}
		collection.savedVersion = collection.version
		return collection
	}

	if precision == hnswPrecisionUnset {
		if n != 0 {
			r.fail("dense collection has slots but no vectors")
			return nil
		}
		collection.savedVersion = collection.version
		return collection
	}
	if params.M < 1 || params.M0 < 1 || params.M0 > math.MaxUint16 || maxLevel > hnswMaxLevel {
		r.fail("invalid graph parameters")
		return nil
	}
	x := newDenseIndex(dim, distance, precision == hnswPrecision16, params)
	if x.fp16 {
		x.vec16 = r.u16s(n * dim)
	} else {
		x.vec32 = r.f32s(n * dim)
	}
	x.norms = r.f32s(n)
	x.levels = r.bytes(n)
	x.count0 = r.u16s(n)
	x.links0 = r.i32s(n * params.M0)
	x.upper = make([][]int32, n)
	upper := r.count(8)
	for i := 0; i < upper && r.err == nil; i++ {
		slot := int(r.u32())
		if slot >= n {
			r.fail("upper layer refers to an unknown slot")
			return nil
		}
		x.upper[slot] = r.i32s(int(x.levels[slot]) * (params.M + 1))
	}
	if r.err != nil {
		return nil
	}
	x.entry, x.maxLevel, x.phi = entry, maxLevel, phi
	if !x.validGraph() {
		r.fail("graph links are out of range")
		return nil
	}
	collection.dense = x
	collection.savedVersion = collection.version
	return collection
}

// validGraph checks every link before the index is trusted with unchecked
// slab arithmetic.
func (x *denseIndex) validGraph() bool {
	n := int32(x.slots())
	if n == 0 {
		return x.entry == -1
	}
	if x.entry < 0 || x.entry >= n || int(x.levels[x.entry]) != x.maxLevel {
		return false
	}
	for slot := int32(0); slot < n; slot++ {
		if int(x.count0[slot]) > x.params.M0 {
			return false
		}
		level := int(x.levels[slot])
		if level > 0 && len(x.upper[slot]) != level*(x.params.M+1) {
			return false
		}
		for l := 0; l <= level; l++ {
			if l > 0 {
				if count := x.upper[slot][(l-1)*(x.params.M+1)]; count < 0 || int(count) > x.params.M {
					return false
				}
			}
			for _, neighbor := range x.neighbors(slot, l) {
				if neighbor < 0 || neighbor >= n || int(x.levels[neighbor]) < l {
					return false
				}
			}
		}
	}
	return true
}

type snapshotWriter struct {
	w       *bufio.Writer
	scratch [8]byte
	chunk   [1 << 15]byte
	err     error
}

func (w *snapshotWriter) write(p []byte) {
	if w.err == nil {
		_, w.err = w.w.Write(p)
	}
}

func (w *snapshotWriter) u32(v uint32) {
	binary.LittleEndian.PutUint32(w.scratch[:4], v)
	w.write(w.scratch[:4])
}

func (w *snapshotWriter) u64(v uint64) {
	binary.LittleEndian.PutUint64(w.scratch[:8], v)
	w.write(w.scratch[:8])
}

func (w *snapshotWriter) str(s string) {
	w.u32(uint32(len(s)))
	if w.err == nil {
		_, w.err = w.w.WriteString(s)
	}
}

func (w *snapshotWriter) bytes(p []byte) { w.write(p) }

func (w *snapshotWriter) bools(values []bool) {
	for _, value := range values {
		b := byte(0)
		if value {
			b = 1
		}
		if w.err == nil {
			w.err = w.w.WriteByte(b)
		}
	}
}

func (w *snapshotWriter) u16s(values []uint16) {
	for len(values) > 0 && w.err == nil {
		n := min(len(values), len(w.chunk)/2)
		for i, value := range values[:n] {
			binary.LittleEndian.PutUint16(w.chunk[2*i:], value)
		}
		w.write(w.chunk[:2*n])
		values = values[n:]
	}
}

func (w *snapshotWriter) u32s(values []uint32) {
	for len(values) > 0 && w.err == nil {
		n := min(len(values), len(w.chunk)/4)
		for i, value := range values[:n] {
			binary.LittleEndian.PutUint32(w.chunk[4*i:], value)
		}
		w.write(w.chunk[:4*n])
		values = values[n:]
	}
}

func (w *snapshotWriter) i32s(values []int32) {
	for len(values) > 0 && w.err == nil {
		n := min(len(values), len(w.chunk)/4)
		for i, value := range values[:n] {
			binary.LittleEndian.PutUint32(w.chunk[4*i:], uint32(value))
		}
		w.write(w.chunk[:4*n])
		values = values[n:]
	}
}

func (w *snapshotWriter) f32s(values []float32) {
	for len(values) > 0 && w.err == nil {
		n := min(len(values), len(w.chunk)/4)
		for i, value := range values[:n] {
			binary.LittleEndian.PutUint32(w.chunk[4*i:], math.Float32bits(value))
		}
		w.write(w.chunk[:4*n])
		values = values[n:]
	}
}

type snapshotReader struct {
	r       *bufio.Reader
	limit   uint64 // file size: no count in a valid file can exceed it
	scratch [8]byte
	chunk   [1 << 15]byte
	err     error
}

func (r *snapshotReader) fail(message string) {
	if r.err == nil {
		r.err = errors.New(message)
	}
}

func (r *snapshotReader) read(p []byte) {
	if r.err == nil {
		_, r.err = io.ReadFull(r.r, p)
	}
}

func (r *snapshotReader) u32() uint32 {
	r.read(r.scratch[:4])
	if r.err != nil {
		return 0
	}
	return binary.LittleEndian.Uint32(r.scratch[:4])
}

func (r *snapshotReader) u64() uint64 {
	r.read(r.scratch[:8])
	if r.err != nil {
		return 0
	}
	return binary.LittleEndian.Uint64(r.scratch[:8])
}

// count reads an element count and rejects one that cannot fit in the file, so
// a corrupt header cannot ask for an absurd allocation. width is the smallest
// encoding of one element; 1 reads a 64 bit count, anything else 32 bit.
func (r *snapshotReader) count(width uint64) int {
	var n uint64
	if width == 1 {
		n = r.u64()
	} else {
		n = uint64(r.u32())
	}
	if r.err == nil && n*width > r.limit {
		r.fail("count is larger than the file")
	}
	if r.err != nil {
		return 0
	}
	return int(n)
}

func (r *snapshotReader) sized(n, width int) bool {
	if r.err == nil && (n < 0 || uint64(n)*uint64(width) > r.limit) {
		r.fail("array is larger than the file")
	}
	return r.err == nil
}

func (r *snapshotReader) str() string {
	n := r.count(4)
	if !r.sized(n, 1) {
		return ""
	}
	buffer := make([]byte, n)
	r.read(buffer)
	return string(buffer)
}

func (r *snapshotReader) bytes(n int) []byte {
	if !r.sized(n, 1) {
		return nil
	}
	buffer := make([]byte, n)
	r.read(buffer)
	return buffer
}

func (r *snapshotReader) bools(n int) []bool {
	raw := r.bytes(n)
	values := make([]bool, len(raw))
	for i, b := range raw {
		values[i] = b != 0
	}
	return values
}

// words reads n little-endian words of the given width in chunks.
func (r *snapshotReader) words(n, width int, store func(i int, word []byte)) bool {
	if !r.sized(n, width) {
		return false
	}
	for i := 0; i < n && r.err == nil; {
		m := min(n-i, len(r.chunk)/width)
		r.read(r.chunk[:m*width])
		if r.err != nil {
			break
		}
		for j := 0; j < m; j++ {
			store(i+j, r.chunk[j*width:])
		}
		i += m
	}
	return r.err == nil
}

func (r *snapshotReader) u16s(n int) []uint16 {
	if !r.sized(n, 2) {
		return nil
	}
	values := make([]uint16, n)
	r.words(n, 2, func(i int, word []byte) { values[i] = binary.LittleEndian.Uint16(word) })
	return values
}

func (r *snapshotReader) u32s(n int) []uint32 {
	if !r.sized(n, 4) {
		return nil
	}
	values := make([]uint32, n)
	r.words(n, 4, func(i int, word []byte) { values[i] = binary.LittleEndian.Uint32(word) })
	return values
}

func (r *snapshotReader) i32s(n int) []int32 {
	if !r.sized(n, 4) {
		return nil
	}
	values := make([]int32, n)
	r.words(n, 4, func(i int, word []byte) { values[i] = int32(binary.LittleEndian.Uint32(word)) })
	return values
}

func (r *snapshotReader) f32s(n int) []float32 {
	if !r.sized(n, 4) {
		return nil
	}
	values := make([]float32, n)
	r.words(n, 4, func(i int, word []byte) { values[i] = math.Float32frombits(binary.LittleEndian.Uint32(word)) })
	return values
}
