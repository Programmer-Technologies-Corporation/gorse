package server

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/gorse-io/gorse/storage"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/steinfletcher/apitest"
)

type projectionRecorder struct {
	data.Database
	projection bool
}

func (d *projectionRecorder) BatchInsertItems(ctx context.Context, items []data.Item) error {
	d.projection = data.IsTwitchLiveProjection(ctx)
	return d.Database.BatchInsertItems(ctx, items)
}

func (suite *ServerTestSuite) TestTwitchProjectionIntentDoesNotChangeOrdinaryUpserts() {
	original := suite.DataClient
	recorder := &projectionRecorder{Database: original}
	suite.DataClient = recorder
	defer func() { suite.DataClient = original }()
	items := []Item{{ItemId: "live", Comment: "stream"}}
	apitest.New().Handler(suite.handler).Post("/api/items").Query("twitch_live_projection", "true").
		Header("X-API-Key", apiKey).JSON(items).Expect(suite.T()).Status(http.StatusOK).End()
	suite.True(recorder.projection)
	apitest.New().Handler(suite.handler).Post("/api/items").
		Header("X-API-Key", apiKey).JSON(items).Expect(suite.T()).Status(http.StatusOK).End()
	suite.False(recorder.projection)
	apitest.New().Handler(suite.handler).Post("/api/items").Query("twitch_live_projection", "true").
		Header("X-API-Key", apiKey).JSON(append(items, Item{ItemId: "other"})).Expect(suite.T()).Status(http.StatusBadRequest).End()
}

func (suite *ServerTestSuite) TestMySQLTwitchProjectionKeepsPatchedVectorInIndex() {
	uri := os.Getenv("TWITCH_COUNT_MYSQL_URI")
	if uri == "" {
		suite.T().Skip("TWITCH_COUNT_MYSQL_URI is not set")
	}
	original := suite.DataClient
	db, err := data.Open(uri, fmt.Sprintf("api_%d_", time.Now().UnixNano()),
		storage.WithIsolationLevel("READ-COMMITTED"), storage.WithSeparateTwitchViewCounts(true))
	suite.Require().NoError(err)
	suite.Require().NoError(db.Init())
	suite.DataClient = db
	defer func() { suite.DataClient = original; suite.NoError(db.Purge()); suite.NoError(db.Close()) }()
	suite.enableIncrementalItemToItem()
	suite.Config.VideoHub.LabelPatch = true
	item := Item{ItemId: "live", Timestamp: "2026-10-04T12:00:00Z", Labels: map[string]any{
		"platform": "twitch", "externalKind": "Live", "viewCount": 1000,
		"embedding": []float64{0, 0},
	}}
	project := func() {
		apitest.New().Handler(suite.handler).Post("/api/items").Query("twitch_live_projection", "true").
			Header("X-API-Key", apiKey).JSON([]Item{item}).Expect(suite.T()).Status(http.StatusOK).End()
	}
	project()
	apitest.New().Handler(suite.handler).Patch("/api/item/live/labels").Header("X-API-Key", apiKey).
		JSON(map[string]any{"embedding": []float64{1, 0}, "extra": "retained"}).Expect(suite.T()).Status(http.StatusOK).End()
	item.Labels.(map[string]any)["viewCount"] = 1001
	project() // incoming vector is still old; the committed patched vector must win
	stored, err := db.GetItem(suite.T().Context(), "live")
	suite.Require().NoError(err)
	suite.Equal(float64(1001), stored.Labels.(map[string]any)["viewCount"])
	suite.Equal("retained", stored.Labels.(map[string]any)["extra"])
	indexed, err := suite.VectorClient.GetVectors(suite.T().Context(), vectors.ItemToItemCollection("neighbors"), []string{"live"})
	suite.Require().NoError(err)
	suite.Require().Len(indexed, 1)
	suite.Equal([]float32{1, 0}, indexed[0].Values)
}
