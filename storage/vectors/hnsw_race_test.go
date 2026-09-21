//go:build race

package vectors

// The race detector makes sync.Pool drop items at random, so allocation counts
// of pooled code are only meaningful without it.
const raceDetector = true
