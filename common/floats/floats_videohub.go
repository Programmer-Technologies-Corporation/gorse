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

package floats

// VideoHub fork: allocation-free variants of the FP16 helpers for hot loops
// (the hnsw:// vector store decodes a stored vector per distance evaluation).

// ToFloat32To decodes IEEE 754 FP16 bits into dst using the SIMD kernel of the
// platform. dst must be at least as long as a.
func ToFloat32To(dst []float32, a []uint16) {
	if len(a) == 0 {
		return
	}
	if len(dst) < len(a) {
		panic("floats: destination is shorter than source")
	}
	feature.toFloat32(a, dst[:len(a)])
}

// FromFloat32To encodes FP32 values as IEEE 754 FP16 bits into dst. dst must be
// at least as long as a.
func FromFloat32To(dst []uint16, a []float32) {
	if len(a) == 0 {
		return
	}
	if len(dst) < len(a) {
		panic("floats: destination is shorter than source")
	}
	feature.fromFloat32(a, dst[:len(a)])
}

// DotFP16 returns the dot product of an FP16 encoded vector and an FP32 vector.
// scratch must hold len(a) values; it is overwritten with the decoded vector.
func DotFP16(a []uint16, b, scratch []float32) float32 {
	scratch = scratch[:len(a)]
	ToFloat32To(scratch, a)
	return Dot(scratch, b)
}
