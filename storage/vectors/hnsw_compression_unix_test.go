//go:build !windows

package vectors

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// /dev/full is a disposable write-failure sink, never a data file. The
// symlink stays inside the synthetic fixture and cleanup removes only it.
func TestHNSWCompressedSnapshotWriteFailurePreservesRecoveryFiles(t *testing.T) {
	if _, err := os.Stat("/dev/full"); err != nil {
		t.Skip("write-failure sink unavailable")
	}
	for _, mode := range []string{"none", "zstd"} {
		t.Run(mode, func(t *testing.T) {
			c := compressionFixture(t, "fp16")
			c.opts.snapshotCompression = mode
			require.NoError(t, c.snapshot())
			c.version++
			require.NoError(t, c.snapshot())
			current := filepath.Join(c.dir, hnswSnapshotFile)
			previous := filepath.Join(c.dir, hnswPreviousSnapshot)
			beforeCurrent, err := os.ReadFile(current)
			require.NoError(t, err)
			beforePrevious, err := os.ReadFile(previous)
			require.NoError(t, err)
			c.version++
			require.NoError(t, os.Symlink("/dev/full", current+".tmp"))
			require.Error(t, c.snapshot())
			afterCurrent, err := os.ReadFile(current)
			require.NoError(t, err)
			require.Equal(t, beforeCurrent, afterCurrent)
			afterPrevious, err := os.ReadFile(previous)
			require.NoError(t, err)
			require.Equal(t, beforePrevious, afterPrevious)
			require.NotEqual(t, c.version, c.savedVersion)
			_, err = loadHNSWSnapshot(current, c.name, c.dir, c.opts)
			require.NoError(t, err)
			_, err = loadHNSWSnapshot(previous, c.name, c.dir, c.opts)
			require.NoError(t, err)
		})
	}
}
