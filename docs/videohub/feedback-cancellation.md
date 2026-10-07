# Ranker feedback cancellation

Recommendation initialization formerly loaded user feedback with
`context.Background()`. An expired HTTP request could leave that database read
running. `NewRecommender` now takes the owner's context; server/master HTTP
handlers and the worker pipeline pass their existing contexts through.

Initialization also checks cancellation after the read. Some datastore readers
can return partial feedback with a nil error after streaming is interrupted;
those rows must not become a successful profile with incomplete exclusions.
The full-feedback query and exclusion, replacement and coldstart rules remain
unchanged for live requests. No new timeout, cache, feedback limit, configuration
or schema migration is introduced.

The same controlled HTTP-handler regression on baseline `ec30eb2` and candidate
starts a blocked feedback read, cancels the request, and verifies exit before
explicit fake release. Baseline ignores cancellation until the one-second guard
fails and cleanup releases the read. Candidate exits before the guard, including
a fake reader that returns partial feedback with no error on cancellation. No
cache or vector calls follow. Successful controls with replacement enabled and
disabled return the same ordered IDs on both versions. Constructor tests cover
full history, negative priority, offline exclusions and coldstart behavior.

These are cancellation/liveness tests, not a successful-request latency benchmark.
Physical database cancellation delay and production connection-pool relief remain
unmeasured. The existing canceled-request error response is retained. General
datastore row/cursor-error propagation is a separate pre-existing concern.

```sh
go test ./logics ./server ./master ./worker \
  -run '^(TestNewRecommender(FeedbackCancellation|FullFeedback)|TestGetRecommendFeedback(Cancellation|Success))$' \
  -count=1
```

The master/worker packages compile their tests in this focused command but do not
execute matching tests. All deployed component binaries must also compile. A
rebuilt serving image is needed to activate the fix; previous-image rollback is
compatible. This patch alone does not establish a production homepage speedup.
