package data

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Branches of the MySQL split-count item writer. Exactly one branch is chosen
// per item write, in the priority order of classifyViewCountWrite. The set is
// fixed; no item identifier, value, hash or title is ever used as a label.
const (
	// A marked projection whose source fingerprint matched the stored one for an
	// already converted item: only the small count row was written.
	ViewCountBranchShortPath = "short_path"
	// A marked projection of a converted item whose source fingerprint changed
	// (lifecycle, visibility, categories, comment, timestamp or any other
	// label): full item write plus count row.
	ViewCountBranchFullChangedSource = "full_changed_source"
	// The stored item still carried a legacy viewCount label: first-touch
	// conversion full write regardless of marker or fingerprint.
	ViewCountBranchLegacyConversion = "legacy_conversion"
	// No item row existed: ordinary insert plus count row.
	ViewCountBranchMissingItem = "missing_item"
	// A converted-shape item existed without a count row: healing full write
	// plus count row.
	ViewCountBranchMissingCount = "missing_count"
	// A splittable Twitch live item written without the projection marker:
	// ordinary full replacement, even when the fingerprint matched.
	ViewCountBranchUnmarked = "unmarked"
	// Other providers, VODs, non-object labels or an invalid viewCount: full
	// write, no count row kept.
	ViewCountBranchUnsplittable = "unsplittable"
)

// ViewCountBranches lists every branch so series exist at zero.
var ViewCountBranches = []string{
	ViewCountBranchShortPath, ViewCountBranchFullChangedSource, ViewCountBranchLegacyConversion,
	ViewCountBranchMissingItem, ViewCountBranchMissingCount, ViewCountBranchUnmarked, ViewCountBranchUnsplittable,
}

// Both counters are registered on the default registry like the vector-index
// metrics, so every component exposing /metrics reports them. They only move
// when separate_twitch_view_counts is enabled; the default writer is untouched.
var (
	TwitchViewCountDecisionsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse", Subsystem: "data", Name: "twitch_view_count_decisions_total",
		Help: "Item writes evaluated by the MySQL split-count writer by branch, counted before the transaction outcome is known.",
	}, []string{"branch"})
	TwitchViewCountCommitsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse", Subsystem: "data", Name: "twitch_view_count_commits_total",
		Help: "Item writes whose transaction committed, by the same branch as the decisions counter; a rolled-back write is never counted here.",
	}, []string{"branch"})
)

func init() {
	for _, branch := range ViewCountBranches {
		TwitchViewCountDecisionsTotal.WithLabelValues(branch)
		TwitchViewCountCommitsTotal.WithLabelValues(branch)
	}
}

// viewCountWriteState is what the writer learned about one item while holding
// its row lock. classifyViewCountWrite turns it into the single branch that
// both drives the write and is reported, so the metric cannot drift from the
// actual decision.
type viewCountWriteState struct {
	marked          bool // request carried the twitch_live_projection intent
	splittable      bool // labels yielded a valid Twitch live count
	exists          bool // item row present
	legacyLabel     bool // stored labels still contain viewCount
	countRow        bool // item_view_counts row present
	sameFingerprint bool // stored projection hash equals the incoming cold fingerprint
}

func classifyViewCountWrite(s viewCountWriteState) string {
	switch {
	case !s.splittable:
		return ViewCountBranchUnsplittable
	case !s.exists:
		return ViewCountBranchMissingItem
	case s.legacyLabel:
		return ViewCountBranchLegacyConversion
	case !s.countRow:
		return ViewCountBranchMissingCount
	case !s.marked:
		return ViewCountBranchUnmarked
	case s.sameFingerprint:
		return ViewCountBranchShortPath
	default:
		return ViewCountBranchFullChangedSource
	}
}

// viewCountDecisions accumulates branches chosen inside one transaction so the
// committed counter is only advanced after the transaction returned nil.
type viewCountDecisions map[string]int

func (d viewCountDecisions) evaluate(branch string, n int) {
	TwitchViewCountDecisionsTotal.WithLabelValues(branch).Add(float64(n))
	d[branch] += n
}

func (d viewCountDecisions) committed() {
	for branch, n := range d {
		TwitchViewCountCommitsTotal.WithLabelValues(branch).Add(float64(n))
	}
}
