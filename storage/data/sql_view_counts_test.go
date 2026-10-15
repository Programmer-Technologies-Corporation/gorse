package data

import (
	"encoding/json"
	"testing"

	"github.com/gorse-io/gorse/storage"
	"github.com/stretchr/testify/require"
)

func TestSplitTwitchViewCount(t *testing.T) {
	for _, test := range []struct {
		name, platform, kind string
		count                any
		split                bool
	}{
		{"live", "twitch", "Live", json.Number("8731"), true},
		{"zero", "twitch", "Live", json.Number("0"), true},
		{"large", "twitch", "Live", json.Number("9223372036854775807"), true},
		{"negative", "twitch", "Live", json.Number("-1"), false},
		{"decimal", "twitch", "Live", json.Number("1.5"), false},
		{"null", "twitch", "Live", nil, false},
		{"string", "twitch", "Live", "8731", false},
		{"vod", "twitch", "Video", json.Number("8731"), false},
		{"youtube", "youtube", "Live", json.Number("8731"), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			labels := map[string]any{"platform": test.platform, "externalKind": test.kind, "viewCount": test.count, "embedding": []float32{.25, .5}}
			cold, count, err := splitTwitchViewCount(Item{ItemId: "a", Labels: labels})
			require.NoError(t, err)
			require.Equal(t, test.split, count != nil)
			require.Contains(t, labels, "viewCount", "the caller's projection must not be mutated")
			if test.split {
				require.NotContains(t, cold.Labels, "viewCount")
				require.Contains(t, cold.Labels, "embedding")
			}
		})
	}
	for _, labels := range []any{nil, []string{"legacy"}} {
		cold, count, err := splitTwitchViewCount(Item{Labels: labels})
		require.NoError(t, err)
		require.Nil(t, count)
		require.Equal(t, labels, cold.Labels)
	}
}

func TestSplitTwitchViewCountsRejectsUncommittedIsolation(t *testing.T) {
	_, err := Open("mysql://unused", "", storage.WithSeparateTwitchViewCounts(true))
	require.ErrorContains(t, err, "READ-COMMITTED or stronger")
}
