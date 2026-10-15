package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"time"

	"github.com/gorse-io/gorse/storage"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/steinfletcher/apitest"
)

type projectionRecorder struct {
	data.Database
	projection bool
	afterWrite func()
}

func (d *projectionRecorder) BatchInsertItems(ctx context.Context, items []data.Item) error {
	d.projection = data.IsTwitchLiveProjection(ctx)
	err := d.Database.BatchInsertItems(ctx, items)
	if err == nil && d.afterWrite != nil {
		d.afterWrite()
	}
	return err
}

func (d *projectionRecorder) PatchItemLabelsAtomic(ctx context.Context, id string, patch map[string]any, validate func(data.Item) error) (data.Item, bool, error) {
	patcher, ok := d.Database.(data.AtomicItemLabelPatcher)
	if !ok {
		return data.Item{}, false, nil
	}
	item, handled, err := patcher.PatchItemLabelsAtomic(ctx, id, patch, validate)
	if err == nil && handled && d.afterWrite != nil {
		d.afterWrite()
	}
	return item, handled, err
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

	// An atomic label patch also completes its post-commit refresh after the
	// caller cancels, without restoring an old viewer count.
	patchCtx, cancelPatch := context.WithCancel(context.Background())
	defer cancelPatch()
	suite.DataClient = &projectionRecorder{Database: db, afterWrite: cancelPatch}
	request := httptest.NewRequest(http.MethodPatch, "/api/item/live/labels", strings.NewReader(`{"embedding":[0,1]}`)).WithContext(patchCtx)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-API-Key", apiKey)
	response := httptest.NewRecorder()
	suite.handler.ServeHTTP(response, request)
	suite.Require().ErrorIs(patchCtx.Err(), context.Canceled)
	suite.Equal(http.StatusOK, response.Code, response.Body.String())
	indexed, err = suite.VectorClient.GetVectors(suite.T().Context(), vectors.ItemToItemCollection("neighbors"), []string{"live"})
	suite.Require().NoError(err)
	suite.Require().Len(indexed, 1)
	suite.Equal([]float32{0, 1}, indexed[0].Values)
}

func (suite *ServerTestSuite) TestTwitchProjectionRefreshSurvivesRequestCancellation() {
	suite.enableIncrementalItemToItem()
	original := suite.DataClient
	apitest.New().Handler(suite.handler).Post("/api/items").Header("X-API-Key", apiKey).
		JSON([]Item{{ItemId: "canceled", Timestamp: "2026-10-04T12:00:00Z", Categories: []string{"old"}, Labels: map[string]any{"embedding": []float64{0, 1}}}}).
		Expect(suite.T()).Status(http.StatusOK).End()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	suite.DataClient = &projectionRecorder{Database: original, afterWrite: cancel}
	defer func() { suite.DataClient = original }()
	item := Item{ItemId: "canceled", Timestamp: "2026-10-04T12:00:00Z", IsHidden: true,
		Categories: []string{"restricted"}, Labels: map[string]any{"embedding": []float64{1, 0}}}
	body, err := json.Marshal([]Item{item})
	suite.Require().NoError(err)
	request := httptest.NewRequest(http.MethodPost, "/api/items?twitch_live_projection=true", bytes.NewReader(body)).WithContext(ctx)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-API-Key", apiKey)
	recorder := httptest.NewRecorder()
	suite.handler.ServeHTTP(recorder, request)
	suite.Require().ErrorIs(ctx.Err(), context.Canceled)
	suite.Equal(http.StatusOK, recorder.Code, recorder.Body.String())
	indexed, err := suite.VectorClient.GetVectors(context.Background(), vectors.ItemToItemCollection("neighbors"), []string{"canceled"})
	suite.Require().NoError(err)
	suite.Require().Len(indexed, 1)
	suite.True(indexed[0].IsHidden)
	suite.Equal([]string{"restricted"}, indexed[0].Categories)
	suite.Equal([]float32{1, 0}, indexed[0].Values)
}
