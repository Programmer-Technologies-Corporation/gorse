package data

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	mysqldriver "github.com/go-sql-driver/mysql"
	"github.com/gorse-io/gorse/storage"
	"github.com/stretchr/testify/require"
)

func TestIsMySQLDeadlock(t *testing.T) {
	deadlock := &mysqldriver.MySQLError{Number: 1213, Message: "synthetic deadlock"}
	require.True(t, isMySQLDeadlock(deadlock))
	require.True(t, isMySQLDeadlock(fmt.Errorf("wrapped: %w", deadlock)))
	require.False(t, isMySQLDeadlock(&mysqldriver.MySQLError{Number: 1205, Message: "lock wait timeout"}))
	require.False(t, isMySQLDeadlock(fmt.Errorf("other")))
}

func TestRetryMySQLDeadlocksWithBackoff(t *testing.T) {
	deadlock := &mysqldriver.MySQLError{Number: 1213, Message: "synthetic deadlock"}
	nonDeadlock := &mysqldriver.MySQLError{Number: 1205, Message: "lock wait timeout"}

	t.Run("success after retry", func(t *testing.T) {
		calls, waits := 0, 0
		err := retryMySQLDeadlocksWithBackoff(context.Background(), func() error {
			calls++
			if calls == 1 {
				return fmt.Errorf("wrapped: %w", deadlock)
			}
			return nil
		}, func(context.Context, int) error {
			waits++
			return nil
		})
		require.NoError(t, err)
		require.Equal(t, 2, calls)
		require.Equal(t, 1, waits)
	})

	t.Run("bounded", func(t *testing.T) {
		calls, waits := 0, 0
		err := retryMySQLDeadlocksWithBackoff(context.Background(), func() error {
			calls++
			return deadlock
		}, func(context.Context, int) error {
			waits++
			return nil
		})
		require.ErrorIs(t, err, deadlock)
		require.Equal(t, ordinaryItemDeadlockRetries+1, calls)
		require.Equal(t, ordinaryItemDeadlockRetries, waits)
	})

	t.Run("non deadlock is not retried", func(t *testing.T) {
		calls := 0
		err := retryMySQLDeadlocksWithBackoff(context.Background(), func() error {
			calls++
			return nonDeadlock
		}, func(context.Context, int) error {
			t.Fatal("unexpected backoff")
			return nil
		})
		require.ErrorIs(t, err, nonDeadlock)
		require.Equal(t, 1, calls)
	})

	t.Run("backoff cancellation stops retry", func(t *testing.T) {
		calls := 0
		err := retryMySQLDeadlocksWithBackoff(context.Background(), func() error {
			calls++
			return deadlock
		}, func(context.Context, int) error {
			return context.Canceled
		})
		require.ErrorIs(t, err, context.Canceled)
		require.Equal(t, 1, calls)
	})
}

func TestSameSQLItemProjectionUsesCanonicalContent(t *testing.T) {
	instant := time.Date(2026, 10, 25, 20, 0, 0, 0, time.UTC)
	left := NewSQLItem(Item{ItemId: "item", Timestamp: instant, Categories: []string{"video"}, Labels: map[string]any{"b": 2, "a": 1}})
	right := NewSQLItem(Item{ItemId: "item", Timestamp: instant.In(time.FixedZone("offset", -7*60*60)), Categories: []string{"video"}, Labels: map[string]any{"a": json.Number("1"), "b": json.Number("2")}})
	require.True(t, sameSQLItemProjection(left, right), "map order, JSON number representation, and time location are not logical changes")
	right.Categories = `[ "video" ]`
	right.Labels = `{ "b": 2, "a": 1 }`
	require.True(t, sameSQLItemProjection(left, right), "database JSON whitespace and key order are not logical changes")
	right.Labels = `{"a":9007199254740993,"b":2}`
	require.False(t, sameSQLItemProjection(left, right), "large JSON integers must not compare through float64 rounding")
	right.Labels = left.Labels
	right.Comment = "changed"
	require.False(t, sameSQLItemProjection(left, right))
}

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

func TestClassifyViewCountWrite(t *testing.T) {
	converted := viewCountWriteState{marked: true, splittable: true, exists: true, countRow: true, sameFingerprint: true}
	for _, test := range []struct {
		name   string
		state  viewCountWriteState
		branch string
	}{
		{"short path", converted, ViewCountBranchShortPath},
		{"changed source", func() viewCountWriteState { s := converted; s.sameFingerprint = false; return s }(), ViewCountBranchFullChangedSource},
		{"unmarked wins over same fingerprint", func() viewCountWriteState { s := converted; s.marked = false; return s }(), ViewCountBranchUnmarked},
		{"missing count wins over marker", func() viewCountWriteState { s := converted; s.countRow = false; s.sameFingerprint = false; return s }(), ViewCountBranchMissingCount},
		{"legacy label wins over stale fingerprint", func() viewCountWriteState { s := converted; s.legacyLabel = true; return s }(), ViewCountBranchLegacyConversion},
		{"missing item", viewCountWriteState{marked: true, splittable: true}, ViewCountBranchMissingItem},
		{"unsplittable existing", viewCountWriteState{marked: true, exists: true, countRow: true}, ViewCountBranchUnsplittable},
		{"unsplittable missing", viewCountWriteState{}, ViewCountBranchUnsplittable},
	} {
		t.Run(test.name, func(t *testing.T) {
			require.Equal(t, test.branch, classifyViewCountWrite(test.state))
			require.Contains(t, ViewCountBranches, test.branch)
		})
	}
	// Every state maps to exactly one listed branch: the label set is closed.
	for bits := 0; bits < 1<<6; bits++ {
		s := viewCountWriteState{marked: bits&1 != 0, splittable: bits&2 != 0, exists: bits&4 != 0,
			legacyLabel: bits&8 != 0, countRow: bits&16 != 0, sameFingerprint: bits&32 != 0}
		require.Contains(t, ViewCountBranches, classifyViewCountWrite(s))
	}
}

func TestSplitTwitchViewCountsRejectsUncommittedIsolation(t *testing.T) {
	_, err := Open("mysql://unused", "", storage.WithSeparateTwitchViewCounts(true))
	require.ErrorContains(t, err, "READ-COMMITTED or stronger")
}
