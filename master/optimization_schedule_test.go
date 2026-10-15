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

package master

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/gorse-io/gorse/dataset"
	"github.com/gorse-io/gorse/model/ctr"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/require"
)

func TestOptimizationPeriodBoundary(t *testing.T) {
	now := time.Date(2026, 10, 15, 0, 0, 0, 0, time.UTC)
	m := Master{optimizationNow: func() time.Time { return now }}
	runs := 0
	run := func() error {
		return m.runOptimizationTask(context.Background(), "period_boundary", 6*time.Hour,
			&m.collaborativeFilteringLastOptimize, func() (bool, error) {
				runs++
				now = now.Add(2 * time.Minute) // The checkpoint is completion, not start.
				return true, nil
			})
	}
	require.NoError(t, run()) // First eligible cycle, including startup.
	completed := now
	require.Equal(t, completed, m.collaborativeFilteringLastOptimize)
	for hour := 1; hour < 6; hour++ {
		now = completed.Add(time.Duration(hour) * time.Hour)
		require.NoError(t, run()) // Periodic, config/import/restore-triggered cycles share this gate.
	}
	now = completed.Add(6*time.Hour - time.Nanosecond)
	require.NoError(t, run())
	require.Equal(t, 1, runs)
	now = completed.Add(6 * time.Hour)
	require.NoError(t, run())
	require.Equal(t, 2, runs)
	require.Equal(t, now, m.collaborativeFilteringLastOptimize)
	require.NoError(t, run())
	require.Equal(t, 2, runs, "another scheduled notification must not bypass the period")
}

func TestOptimizationPeriodsAreIndependent(t *testing.T) {
	start := time.Date(2026, 10, 15, 0, 0, 0, 0, time.UTC)
	now := start
	m := Master{optimizationNow: func() time.Time { return now }}
	var collaborative, ranker []int
	for hour := 0; hour <= 6; hour++ {
		now = start.Add(time.Duration(hour) * time.Hour)
		require.NoError(t, m.runOptimizationTask(context.Background(), "period_collaborative", 6*time.Hour,
			&m.collaborativeFilteringLastOptimize, func() (bool, error) {
				collaborative = append(collaborative, hour)
				return true, nil
			}))
		require.NoError(t, m.runOptimizationTask(context.Background(), "period_ranker", 2*time.Hour,
			&m.clickThroughRateLastOptimize, func() (bool, error) {
				ranker = append(ranker, hour)
				return true, nil
			}))
	}
	require.Equal(t, []int{0, 6}, collaborative)
	require.Equal(t, []int{0, 2, 4, 6}, ranker)
	last := m.clickThroughRateLastOptimize
	called := false
	require.NoError(t, m.runOptimizationTask(context.Background(), "period_disabled", 0,
		&m.clickThroughRateLastOptimize, func() (bool, error) { called = true; return true, nil }))
	require.False(t, called)
	require.Equal(t, last, m.clickThroughRateLastOptimize)
	require.NoError(t, m.runOptimizationTask(context.Background(), "period_reenabled", 2*time.Hour,
		&m.clickThroughRateLastOptimize, func() (bool, error) { called = true; return true, nil }))
	require.False(t, called, "re-enabling must not erase a completed search's checkpoint")
	// A changed positive period takes effect without discarding the successful checkpoint.
	now = now.Add(time.Hour)
	require.NoError(t, m.runOptimizationTask(context.Background(), "period_changed", time.Hour,
		&m.clickThroughRateLastOptimize, func() (bool, error) { called = true; return true, nil }))
	require.True(t, called)
}

func TestOptimizationUnsuccessfulWorkDoesNotConsumePeriod(t *testing.T) {
	for _, scenario := range []string{"empty", "error", "panic", "cancel_before", "cancel_during"} {
		t.Run(scenario, func(t *testing.T) {
			now := time.Date(2026, 10, 15, 0, 0, 0, 0, time.UTC)
			m := Master{optimizationNow: func() time.Time { return now }}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if scenario == "cancel_before" {
				cancel()
			}
			called := false
			name := "period_" + scenario
			failures := metricValue(TaskFailuresTotalVec.WithLabelValues(name))
			panics := metricValue(TaskPanicsTotalVec.WithLabelValues(name))
			err := m.runOptimizationTask(ctx, name, 6*time.Hour, &m.collaborativeFilteringLastOptimize,
				func() (bool, error) {
					called = true
					switch scenario {
					case "empty":
						return false, nil
					case "error":
						return false, errors.New("synthetic optimizer failure")
					case "panic":
						panic("synthetic optimizer panic")
					case "cancel_during":
						cancel()
					}
					return true, nil
				})
			if scenario == "empty" {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.Equal(t, failures+1, metricValue(TaskFailuresTotalVec.WithLabelValues(name)))
			}
			if scenario == "panic" {
				require.Equal(t, panics+1, metricValue(TaskPanicsTotalVec.WithLabelValues(name)))
			}
			require.Equal(t, scenario != "cancel_before", called)
			require.True(t, m.collaborativeFilteringLastOptimize.IsZero())
			// Retrying is owned by the existing next task cycle, not a new tight loop.
			now = now.Add(time.Hour)
			require.NoError(t, m.runOptimizationTask(context.Background(), name, 6*time.Hour,
				&m.collaborativeFilteringLastOptimize, func() (bool, error) { return true, nil }))
			require.Equal(t, now, m.collaborativeFilteringLastOptimize)
		})
	}
}

func TestOptimizationFailureDoesNotBlockOtherModel(t *testing.T) {
	now := time.Date(2026, 10, 15, 0, 0, 0, 0, time.UTC)
	m := Master{optimizationNow: func() time.Time { return now }}
	require.Error(t, m.runOptimizationTask(context.Background(), "period_failed_cf", time.Hour,
		&m.collaborativeFilteringLastOptimize, func() (bool, error) { return false, errors.New("failed") }))
	require.NoError(t, m.runOptimizationTask(context.Background(), "period_successful_ctr", time.Hour,
		&m.clickThroughRateLastOptimize, func() (bool, error) { return true, nil }))
	require.True(t, m.collaborativeFilteringLastOptimize.IsZero())
	require.Equal(t, now, m.clickThroughRateLastOptimize)
}

func (s *MasterTestSuite) TestOptimizationEmptyDataDoesNotCheckpoint() {
	defer func() {
		s.optimizationNow = nil
		s.collaborativeFilteringLastOptimize = time.Time{}
		s.clickThroughRateLastOptimize = time.Time{}
	}()
	now := time.Date(2026, 10, 15, 0, 0, 0, 0, time.UTC)
	s.optimizationNow = func() time.Time { return now }
	s.collaborativeFilteringLastOptimize = time.Time{}
	s.clickThroughRateLastOptimize = time.Time{}
	emptyCF := dataset.NewDataset(now, 0, 0)
	emptyCTR := &ctr.Dataset{Index: dataset.NewUnifiedDirectIndex(0)}
	s.NoError(s.runOptimizationTask(s.T().Context(), "empty_cf", time.Hour,
		&s.collaborativeFilteringLastOptimize, func() (bool, error) {
			return s.optimizeCollaborativeFiltering(s.T().Context(), emptyCF, emptyCF)
		}))
	s.NoError(s.runOptimizationTask(s.T().Context(), "empty_ctr", time.Hour,
		&s.clickThroughRateLastOptimize, func() (bool, error) {
			return s.optimizeClickThroughRatePrediction(s.T().Context(), emptyCTR, emptyCTR)
		}))
	s.True(s.collaborativeFilteringLastOptimize.IsZero())
	s.True(s.clickThroughRateLastOptimize.IsZero())
}

func (s *MasterTestSuite) TestOptimizationCooldownKeepsDataRefreshAndFitting() {
	defer func() {
		s.optimizationNow = nil
		s.collaborativeFilteringLastOptimize = time.Time{}
		s.clickThroughRateLastOptimize = time.Time{}
	}()
	now := time.Date(2026, 10, 15, 0, 0, 0, 0, time.UTC)
	s.optimizationNow = func() time.Time { return now }
	s.collaborativeFilteringLastOptimize = now
	s.clickThroughRateLastOptimize = now
	s.Config.Recommend.Collaborative.Type = "mf"
	s.Config.Recommend.Collaborative.OptimizePeriod = 6 * time.Hour
	s.Config.Recommend.Ranker.Type = "fm"
	s.Config.Recommend.Ranker.OptimizePeriod = 6 * time.Hour
	s.NoError(s.runLoadDatasetTask(s.T().Context()))
	s.backgroundWaitGroup.Wait()
	s.Equal(float64(0), metricValue(UsersTotal))
	s.Require().NoError(s.DataClient.BatchInsertUsers(s.T().Context(), []data.User{{UserId: "synthetic"}}))
	now = now.Add(time.Hour)
	s.NoError(s.runLoadDatasetTask(s.T().Context()))
	s.backgroundWaitGroup.Wait()
	s.Equal(float64(1), metricValue(UsersTotal), "fresh input is loaded during optimization cooldown")
	s.Greater(metricValue(TaskLastSuccessTimestampVec.WithLabelValues("train_collaborative_filtering")), float64(0))
	s.Greater(metricValue(TaskLastSuccessTimestampVec.WithLabelValues("train_click_through_rate")), float64(0))
	s.Equal(now.Add(-time.Hour), s.collaborativeFilteringLastOptimize)
	s.Equal(now.Add(-time.Hour), s.clickThroughRateLastOptimize)
}
