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
	"time"
)

// runOptimizationTask is called only by the serialized master task loop. Unlike
// ordinary fitting and data refreshes, hyperparameter searches observe their own
// cooldown after a completed search. Empty data, failures, panics and cancelled
// work do not consume the period; they can retry on the next normal task cycle.
func (m *Master) runOptimizationTask(ctx context.Context, name string, period time.Duration,
	lastSuccess *time.Time, fn func() (bool, error)) error {
	now := m.optimizationNow
	if now == nil {
		now = time.Now
	}
	if period <= 0 || (!lastSuccess.IsZero() && now().Sub(*lastSuccess) < period) {
		return nil
	}
	completed := false
	err := m.runTask(name, func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		var err error
		completed, err = fn()
		if err != nil {
			return err
		}
		return ctx.Err()
	})
	if err == nil && completed {
		*lastSuccess = now()
	}
	return err
}
