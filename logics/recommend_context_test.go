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

package logics

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/gorse-io/gorse/common/expression"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/require"
)

type constructorFeedbackStore struct {
	data.Database
	getFeedback func(context.Context, string, *time.Time, ...expression.FeedbackTypeExpression) ([]data.Feedback, error)
}

func (d *constructorFeedbackStore) GetUserFeedback(ctx context.Context, userId string, endTime *time.Time, feedbackTypes ...expression.FeedbackTypeExpression) ([]data.Feedback, error) {
	return d.getFeedback(ctx, userId, endTime, feedbackTypes...)
}

func TestNewRecommenderFeedbackCancellation(t *testing.T) {
	for _, partial := range []bool{false, true} {
		t.Run(fmt.Sprintf("partial_feedback=%v", partial), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			entered := make(chan context.Context, 1)
			release := make(chan struct{})
			store := &constructorFeedbackStore{getFeedback: func(ctx context.Context, _ string, _ *time.Time, _ ...expression.FeedbackTypeExpression) ([]data.Feedback, error) {
				entered <- ctx
				select {
				case <-ctx.Done():
					if partial {
						// A store may stop iterating on cancellation but omit the cursor error.
						return []data.Feedback{{FeedbackKey: data.FeedbackKey{ItemId: "partial", FeedbackType: "click"}}}, nil
					}
					return nil, ctx.Err()
				case <-release:
					return nil, nil
				}
			}}
			type result struct {
				recommender *Recommender
				err         error
			}
			done := make(chan result, 1)
			go func() {
				recommender, err := NewRecommender(ctx, config.RecommendConfig{}, nil, store, nil, true, "user", nil)
				done <- result{recommender, err}
			}()
			defer close(release)
			select {
			case passed := <-entered:
				require.Same(t, ctx, passed)
			case <-time.After(time.Second):
				t.Fatal("feedback read did not start")
			}
			cancel()
			select {
			case result := <-done:
				require.Nil(t, result.recommender)
				require.ErrorIs(t, result.err, context.Canceled)
			case <-time.After(time.Second):
				t.Fatal("constructor did not stop after feedback cancellation")
			}
		})
	}
}
func TestNewRecommenderFullFeedback(t *testing.T) {
	feedback := []data.Feedback{
		{FeedbackKey: data.FeedbackKey{UserId: "user", ItemId: "positive", FeedbackType: "click"}},
		{FeedbackKey: data.FeedbackKey{UserId: "user", ItemId: "negative", FeedbackType: "dislike"}},
		{FeedbackKey: data.FeedbackKey{UserId: "user", ItemId: "read", FeedbackType: "read"}},
		{FeedbackKey: data.FeedbackKey{UserId: "user", ItemId: "positive", FeedbackType: "dislike"}},
	}
	for _, test := range []struct {
		name                string
		replacement, online bool
		feedback            []data.Feedback
		excluded            []string
		coldstart           bool
	}{
		{"online", false, true, feedback, []string{"positive", "negative", "read"}, false},
		{"online replacement", true, true, feedback, []string{"positive", "negative"}, false},
		{"offline replacement", true, false, feedback, []string{"positive", "negative", "read"}, false},
		{"negative cold start", true, true, feedback[1:3], []string{"negative"}, true},
		{"no feedback cold start", true, true, nil, nil, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := t.Context()
			before := time.Now()
			calls := 0
			store := &constructorFeedbackStore{getFeedback: func(passed context.Context, userId string, endTime *time.Time, feedbackTypes ...expression.FeedbackTypeExpression) ([]data.Feedback, error) {
				calls++
				require.Equal(t, ctx, passed)
				require.Equal(t, "user", userId)
				require.NotNil(t, endTime)
				require.False(t, endTime.Before(before))
				require.False(t, endTime.After(time.Now()))
				require.Empty(t, feedbackTypes, "load every feedback type")
				return test.feedback, nil
			}}
			cfg := config.RecommendConfig{CacheSize: 1, ContextSize: 1}
			cfg.DataSource.PositiveFeedbackTypes = []expression.FeedbackTypeExpression{{FeedbackType: "click"}}
			cfg.DataSource.NegativeFeedbackTypes = []expression.FeedbackTypeExpression{{FeedbackType: "dislike"}}
			cfg.Replacement.EnableReplacement = test.replacement
			recommender, err := NewRecommender(ctx, cfg, nil, store, nil, test.online, "user", []string{"movie"})
			require.NoError(t, err)
			require.Equal(t, 1, calls)
			require.Equal(t, test.feedback, recommender.UserFeedback(), "feedback must not be limited by cache/context sizes")
			require.ElementsMatch(t, test.excluded, recommender.ExcludeSet().ToSlice())
			require.Equal(t, test.coldstart, recommender.IsColdStart())
			require.Equal(t, []string{"movie"}, recommender.categories)
		})
	}
}
