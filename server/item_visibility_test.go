package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	mapset "github.com/deckarep/golang-set/v2"
	"github.com/go-sql-driver/mysql"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/storage"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/samber/lo"
	"github.com/steinfletcher/apitest"
	"github.com/stretchr/testify/require"
)

// Retain the pre-optimization implementation as an equivalence/benchmark oracle.
func filterVisibleItemsFullMetadata(ctx context.Context, db data.Database, scores []cache.Score, categories []string) ([]cache.Score, error) {
	items, err := db.BatchGetItems(ctx, cache.ConvertDocumentsToValues(scores), data.GetOptions{})
	if err != nil {
		return nil, err
	}
	visible := mapset.NewSet[string]()
	for _, item := range items {
		matches := len(categories) == 0 || lo.EveryBy(categories, func(category string) bool {
			return lo.Contains(item.Categories, category)
		})
		if !item.IsHidden && matches {
			visible.Add(item.ItemId)
		}
	}
	return lo.Filter(scores, func(score cache.Score, _ int) bool { return visible.Contains(score.Id) }), nil
}

func visibilityDatabase(tb testing.TB, backend string, items []data.Item) data.Database {
	tb.Helper()
	uri := "sqlite://" + filepath.ToSlash(filepath.Join(tb.TempDir(), "visibility.db"))
	if backend == "mysql" {
		uri = os.Getenv("GORSE_VISIBILITY_MYSQL_URI")
		if uri == "" {
			tb.Skip("GORSE_VISIBILITY_MYSQL_URI is not set; isolated loopback MySQL required")
		}
		if !strings.HasPrefix(uri, "mysql://") {
			tb.Fatal("visibility fixture requires a mysql:// URI")
		}
		dsn, err := mysql.ParseDSN(strings.TrimPrefix(uri, "mysql://"))
		require.NoError(tb, err)
		host, _, err := net.SplitHostPort(dsn.Addr)
		require.NoError(tb, err)
		ip := net.ParseIP(host)
		if dsn.Net != "tcp" || (host != "localhost" && (ip == nil || !ip.IsLoopback())) || !strings.HasPrefix(dsn.DBName, "gorse_visibility") {
			tb.Fatal("visibility fixture only accepts a loopback gorse_visibility database")
		}
	}
	// Keep the six-source benchmark's MySQL connections warm instead of
	// measuring TCP churn from database/sql's default two idle connections.
	db, err := data.Open(uri, fmt.Sprintf("visibility_%d_", time.Now().UnixNano()),
		storage.WithMaxOpenConns(6), storage.WithMaxIdleConns(6))
	require.NoError(tb, err)
	tb.Cleanup(func() { require.NoError(tb, db.Close()) })
	require.NoError(tb, db.Init())
	for i := range items {
		if items[i].Timestamp.IsZero() {
			items[i].Timestamp = time.Date(2026, 10, 6, 0, 0, 0, 0, time.UTC)
		}
	}
	require.NoError(tb, db.BatchInsertItems(tb.Context(), items))
	return db
}

func TestFilterVisibleItemsEquivalence(t *testing.T) {
	for _, backend := range []string{"sqlite", "mysql"} {
		t.Run(backend, func(t *testing.T) {
			db := visibilityDatabase(t, backend, []data.Item{
				{ItemId: "one", Categories: []string{"topic:game", "language:en", "quote:\"", "日本語"}, Labels: map[string]any{"embedding": []float32{.1, .2}}},
				{ItemId: "two", Categories: []string{"topic:game"}},
				{ItemId: "hidden", IsHidden: true, Categories: []string{"topic:game", "language:en"}},
				{ItemId: "empty", Categories: []string{}},
				{ItemId: "nil", Categories: nil},
				{ItemId: "blank", Categories: []string{""}},
			})
			// Scores are deliberately unsorted, duplicated and carry stale categories.
			scores := []cache.Score{
				{Id: "two", Score: 3, Categories: []string{"stale"}},
				{Id: "hidden", Score: 9}, {Id: "missing", Score: 8},
				{Id: "one", Score: 2, Categories: []string{"stale"}},
				{Id: "two", Score: 1}, {Id: "empty"}, {Id: "nil"}, {Id: "blank"},
			}
			cases := []struct {
				name       string
				categories []string
				want       []string
			}{
				{"unfiltered", nil, []string{"two", "one", "two", "empty", "nil", "blank"}},
				{"empty-filter", []string{}, []string{"two", "one", "two", "empty", "nil", "blank"}},
				{"one-category", []string{"topic:game"}, []string{"two", "one", "two"}},
				{"all-categories", []string{"topic:game", "language:en"}, []string{"one"}},
				{"duplicate-category", []string{"topic:game", "topic:game"}, []string{"two", "one", "two"}},
				{"case-sensitive", []string{"Topic:Game"}, []string{}},
				{"escaped-unicode", []string{"quote:\"", "日本語"}, []string{"one"}},
				{"blank-category", []string{""}, []string{"blank"}},
				{"no-match", []string{"absent"}, []string{}},
			}
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					want, err := filterVisibleItemsFullMetadata(t.Context(), db, scores, tc.categories)
					require.NoError(t, err)
					got, err := FilterVisibleItemsByCategories(t.Context(), db, scores, tc.categories)
					require.NoError(t, err)
					require.Equal(t, want, got, "preserve complete scores, order and duplicates")
					require.Equal(t, tc.want, cache.ConvertDocumentsToValues(got))
				})
			}
			empty, err := FilterVisibleItemsByCategories(t.Context(), db, nil, []string{"topic:game"})
			require.NoError(t, err)
			require.Empty(t, empty)

			// The same candidate list must observe writes without waiting for vector refresh.
			hidden := true
			require.NoError(t, db.ModifyItem(t.Context(), "one", data.ItemPatch{IsHidden: &hidden}))
			require.NoError(t, db.ModifyItem(t.Context(), "two", data.ItemPatch{Categories: []string{"other"}}))
			got, err := FilterVisibleItemsByCategories(t.Context(), db, scores, []string{"topic:game"})
			require.NoError(t, err)
			require.Empty(t, got)
			hidden = false
			require.NoError(t, db.ModifyItem(t.Context(), "one", data.ItemPatch{IsHidden: &hidden}))
			got, err = FilterVisibleItemsByCategories(t.Context(), db, scores, []string{"topic:game"})
			require.NoError(t, err)
			require.Equal(t, []cache.Score{scores[3]}, got)

			// Exercise the actual projection: no labels, categories or comments materialize.
			items, err := db.BatchGetItems(t.Context(), []string{"one", "two", "hidden", "missing"}, data.GetOptions{Categories: []string{"topic:game"}, SkipHidden: true, ReturnId: true})
			require.NoError(t, err)
			require.Equal(t, []data.Item{{ItemId: "one"}}, items)
			cancelled, cancel := context.WithCancel(t.Context())
			cancel()
			got, err = FilterVisibleItemsByCategories(cancelled, db, scores, nil)
			require.ErrorIs(t, err, context.Canceled)
			require.Nil(t, got)
		})
	}
}

type failingVisibilityDatabase struct {
	data.Database
	err error
}

func (db failingVisibilityDatabase) BatchGetItems(context.Context, []string, data.GetOptions) ([]data.Item, error) {
	return nil, db.err
}

func TestFilterVisibleItemsPropagatesErrors(t *testing.T) {
	for _, want := range []error{errors.New("data store unavailable"), context.Canceled, context.DeadlineExceeded} {
		got, err := FilterVisibleItemsByCategories(t.Context(), failingVisibilityDatabase{err: want}, []cache.Score{{Id: "one"}}, nil)
		require.ErrorIs(t, err, want)
		require.Nil(t, got)
	}
}

type staleVisibilityVectors struct{ vectors.NoDatabase }

func (staleVisibilityVectors) GetVectors(context.Context, string, []string) ([]vectors.Vector, error) {
	return []vectors.Vector{{Id: "anchor", Values: []float32{1, 0}}}, nil
}

func (staleVisibilityVectors) QueryVectors(context.Context, string, vectors.Vector, []string, int) ([]vectors.ScoredVector, error) {
	var result []vectors.ScoredVector
	for i, id := range []string{"hidden", "missing", "wrong-category", "one", "two", "three"} {
		result = append(result, vectors.ScoredVector{Vector: vectors.Vector{Id: id, Categories: []string{"wanted"}}, Score: float32(0.6 - float64(i)*.1)})
	}
	return result, nil
}

func (suite *ServerTestSuite) TestItemToItemFreshVisibilityAndPagination() {
	// The vector provider intentionally returns stale visibility/categories. The
	// data store is real; both named routes must filter it before applying offset.
	original := suite.VectorClient
	suite.VectorClient = staleVisibilityVectors{}
	defer func() { suite.VectorClient = original }()
	suite.Config.Recommend.ItemToItem = []config.ItemToItemConfig{
		{Name: "neighbors", Type: "embedding"}, {Name: "label_neighbors", Type: "tags"},
	}
	suite.Require().NoError(suite.DataClient.BatchInsertItems(suite.T().Context(), []data.Item{
		{ItemId: "hidden", IsHidden: true, Categories: []string{"wanted"}},
		{ItemId: "wrong-category", Categories: []string{"other"}},
		{ItemId: "one", Categories: []string{"wanted"}},
		{ItemId: "two", Categories: []string{"wanted"}},
		{ItemId: "three", Categories: []string{"wanted"}},
	}))
	for _, name := range []string{"neighbors", "label_neighbors"} {
		apitest.New().Handler(suite.handler).Get("/api/item-to-item/"+name+"/anchor").
			QueryCollection(map[string][]string{"category": {"wanted"}, "n": {"1"}, "offset": {"1"}}).
			Header("X-API-Key", apiKey).Expect(suite.T()).Status(http.StatusOK).
			Assert(func(response *http.Response, _ *http.Request) error {
				var scores []cache.Score
				err := json.NewDecoder(response.Body).Decode(&scores)
				if err != nil {
					return err
				}
				suite.Equal([]string{"two"}, cache.ConvertDocumentsToValues(scores))
				return nil
			}).End()
	}
}
