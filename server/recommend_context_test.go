// Copyright 2026 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/emicklei/go-restful/v3"
	"github.com/gorse-io/gorse/common/expression"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/stretchr/testify/require"
)

type recommendFeedbackStore struct {
	data.Database
	entered               chan context.Context
	release               chan struct{}
	feedback              []data.Feedback
	returnPartialOnCancel bool
}

func (d *recommendFeedbackStore) GetUserFeedback(ctx context.Context, userId string, endTime *time.Time, feedbackTypes ...expression.FeedbackTypeExpression) ([]data.Feedback, error) {
	d.entered <- ctx
	select {
	case <-ctx.Done():
		if d.returnPartialOnCancel {
			return d.feedback, nil
		}
		return nil, ctx.Err()
	case <-d.release:
		return d.feedback, nil
	}
}

type recommendCacheSpy struct {
	cache.Database
	calls  atomic.Int32
	scores []cache.Score
}

func (c *recommendCacheSpy) SearchScores(context.Context, string, string, []string, int, int) ([]cache.Score, error) {
	c.calls.Add(1)
	return c.scores, nil
}

type recommendVectorSpy struct {
	vectors.Database
	calls atomic.Int32
}

func (v *recommendVectorSpy) GetVectors(context.Context, string, []string) ([]vectors.Vector, error) {
	v.calls.Add(1)
	return nil, nil
}

func (v *recommendVectorSpy) QueryVectors(context.Context, string, vectors.Vector, []string, int) ([]vectors.ScoredVector, error) {
	v.calls.Add(1)
	return nil, nil
}

func feedbackRecommendHandler(cfg *config.Config, d data.Database, c cache.Database, v vectors.Database) http.Handler {
	s := &RestServer{Config: cfg, DataClient: d, CacheClient: c, VectorClient: v}
	service := new(restful.WebService).Path("/api").Produces(restful.MIME_JSON)
	service.Route(service.GET("/recommend/{user-id}").To(s.getRecommend))
	handler := restful.NewContainer()
	handler.Add(service)
	return handler
}

func TestGetRecommendFeedbackCancellation(t *testing.T) {
	for _, partial := range []bool{false, true} {
		t.Run(fmt.Sprintf("partial_feedback=%v", partial), func(t *testing.T) {
			store := &recommendFeedbackStore{
				entered: make(chan context.Context, 1), release: make(chan struct{}), returnPartialOnCancel: partial,
				feedback: []data.Feedback{{FeedbackKey: data.FeedbackKey{UserId: "user", ItemId: "partial", FeedbackType: "click"}}},
			}
			cacheClient := &recommendCacheSpy{scores: []cache.Score{{Id: "candidate"}}}
			vectorClient := &recommendVectorSpy{}
			cfg := config.GetDefaultConfig()
			cfg.Recommend.Ranker.Type = "fm"
			cfg.Recommend.Fallback.Recommenders = nil
			handler := feedbackRecommendHandler(cfg, store, cacheClient, vectorClient)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			request := httptest.NewRequest(http.MethodGet, "/api/recommend/user?n=1", nil).WithContext(ctx)
			response := httptest.NewRecorder()
			done := make(chan struct{})
			go func() {
				defer close(done)
				handler.ServeHTTP(response, request)
			}()
			// Explicit release keeps the baseline failure bounded and cleans up the handler.
			defer func() { close(store.release); <-done; t.Log("handler joined after cleanup release") }()
			select {
			case <-store.entered:
			case <-time.After(time.Second):
				t.Fatal("feedback read did not start")
			}
			cancelStarted := time.Now()
			cancel()
			select {
			case <-done:
				t.Logf("canceled feedback read and handler exited without explicit release after %s", time.Since(cancelStarted))
			case <-time.After(time.Second):
				t.Fatalf("handler ignored request cancellation for %s until explicit feedback release", time.Since(cancelStarted))
			}
			require.Equal(t, http.StatusInternalServerError, response.Code)
			require.Contains(t, response.Body.String(), context.Canceled.Error())
			require.Zero(t, cacheClient.calls.Load(), "canceled initialization must stop before cache work")
			require.Zero(t, vectorClient.calls.Load(), "canceled initialization must stop before vector work")
		})
	}
}
func TestGetRecommendFeedbackSuccess(t *testing.T) {
	feedback := []data.Feedback{
		{FeedbackKey: data.FeedbackKey{UserId: "user", ItemId: "positive", FeedbackType: "click"}},
		{FeedbackKey: data.FeedbackKey{UserId: "user", ItemId: "negative", FeedbackType: "dislike"}},
		{FeedbackKey: data.FeedbackKey{UserId: "user", ItemId: "read", FeedbackType: "read"}},
	}
	for _, replacement := range []bool{false, true} {
		t.Run(fmt.Sprintf("replacement=%v", replacement), func(t *testing.T) {
			store := &recommendFeedbackStore{entered: make(chan context.Context, 1), release: make(chan struct{}), feedback: feedback}
			close(store.release)
			cacheClient := &recommendCacheSpy{scores: []cache.Score{{Id: "negative"}, {Id: "positive"}, {Id: "read"}, {Id: "candidate"}}}
			vectorClient := &recommendVectorSpy{}
			cfg := config.GetDefaultConfig()
			cfg.Recommend.Ranker.Type = "fm"
			cfg.Recommend.Fallback.Recommenders = nil
			cfg.Recommend.Replacement.EnableReplacement = replacement
			cfg.Recommend.DataSource.PositiveFeedbackTypes = []expression.FeedbackTypeExpression{{FeedbackType: "click"}}
			cfg.Recommend.DataSource.NegativeFeedbackTypes = []expression.FeedbackTypeExpression{{FeedbackType: "dislike"}}
			handler := feedbackRecommendHandler(cfg, store, cacheClient, vectorClient)
			request := httptest.NewRequest(http.MethodGet, "/api/recommend/user?n=3", nil).WithContext(t.Context())
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			require.Equal(t, http.StatusOK, response.Code)
			if replacement {
				require.JSONEq(t, `["positive","read","candidate"]`, response.Body.String())
			} else {
				require.JSONEq(t, `["candidate"]`, response.Body.String())
			}
			require.Equal(t, int32(1), cacheClient.calls.Load())
			require.Zero(t, vectorClient.calls.Load())
		})
	}
}
