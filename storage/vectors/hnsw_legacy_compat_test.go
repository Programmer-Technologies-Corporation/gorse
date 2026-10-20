package vectors

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// This test is also copied into the unmodified deployed 044fa37 source tree.
// It deliberately uses only that revision's reader API and format constants.
func TestHNSWLegacyRollbackFixture(t *testing.T) {
	root := os.Getenv("GORSE_SNAPSHOT_ROLLBACK_FIXTURE")
	if root == "" {
		t.Skip("synthetic downgrade fixture not supplied")
	}
	for _, name := range []string{hnswSnapshotFile, hnswPreviousSnapshot} {
		t.Run(name, func(t *testing.T) {
			c, err := loadHNSWSnapshot(filepath.Join(root, name), "synthetic", root, defaultHNSWOptions())
			require.NoError(t, err)
			require.Equal(t, 79, c.meta.live)
			require.Equal(t, 2, c.tombstones())
			vectors := c.get([]string{"id-1", "id-7", "id-9"})
			require.Len(t, vectors, 3)
			expected := make([]float32, 16)
			for j := range expected {
				expected[j] = float32(j + 1)
			}
			require.Equal(t, expected, vectors[0].Values)
			expected[0] = 8
			for j := 1; j < len(expected); j++ {
				expected[j] = float32(j + 7)
			}
			require.Equal(t, expected, vectors[1].Values)
			require.True(t, vectors[2].IsHidden)
			query := Vector{Values: make([]float32, 16)}
			query.Values[0] = 1
			results, err := c.query(query, []string{"category-1"}, 3)
			require.NoError(t, err)
			require.Len(t, results, 3)
			require.Equal(t, []string{"id-79", "id-76", "id-73"}, []string{results[0].Id, results[1].Id, results[2].Id})
		})
	}
}
