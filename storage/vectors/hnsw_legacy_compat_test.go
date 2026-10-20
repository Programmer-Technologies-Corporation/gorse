package vectors

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/gorse-io/gorse/storage"
	"github.com/stretchr/testify/require"
)

// This test is also copied into the unmodified deployed 044fa37 source tree.
// It deliberately uses only that revision's reader API and format constants.
func TestHNSWLegacyRollbackFixture(t *testing.T) {
	root := os.Getenv("GORSE_SNAPSHOT_ROLLBACK_FIXTURE")
	if root == "" {
		t.Skip("synthetic downgrade fixture not supplied")
	}
	legacyURL := storage.HNSWPrefix + root + "?snapshot_interval=1m"
	if os.Getenv("GORSE_SNAPSHOT_EXPECT_LEGACY_PARSER") == "1" {
		for _, mode := range []string{"none", "zstd"} {
			_, err := Open(legacyURL+"&snapshot_compression="+mode, "")
			require.ErrorContains(t, err, "unknown hnsw option snapshot_compression")
		}
	}
	// Exercise the deployed URL parser, initialization and public reader with
	// the compression option removed, retaining a supported legacy parameter.
	db, err := Open(legacyURL, "")
	require.NoError(t, err)
	require.NoError(t, db.Init())
	defer func() { require.NoError(t, db.Close()) }()
	count, err := db.CountVectors(context.Background(), "synthetic")
	require.NoError(t, err)
	require.EqualValues(t, 79, count)
	publicVectors, err := db.GetVectors(context.Background(), "synthetic", []string{"id-1", "id-7", "id-9"})
	require.NoError(t, err)
	require.Len(t, publicVectors, 3)
	require.True(t, publicVectors[2].IsHidden)
	directory := filepath.Join(root, "synthetic")
	for _, name := range []string{hnswSnapshotFile, hnswPreviousSnapshot} {
		t.Run(name, func(t *testing.T) {
			c, err := loadHNSWSnapshot(filepath.Join(directory, name), "synthetic", directory, db.(*HNSWStore).opts)
			require.NoError(t, err)
			require.Equal(t, 79, c.meta.live)
			require.Equal(t, 2, c.tombstones())
			vectors := c.get([]string{"id-1", "id-7", "id-9"})
			require.Equal(t, publicVectors, vectors)
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
