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

package storage

import "strings"

// HNSWPrefix selects the VideoHub fork's embedded vector store: a persistent,
// incrementally maintained HNSW index owned by the master node.
const HNSWPrefix = "hnsw://"

// IsEmbeddedVectorStore reports whether the vector store lives inside the
// master process, in which case servers and workers reach it through the
// master's vector store proxy instead of opening it themselves.
func IsEmbeddedVectorStore(path string) bool {
	return strings.HasPrefix(path, XvecPrefix) || strings.HasPrefix(path, HNSWPrefix)
}
