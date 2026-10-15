# Hyperparameter optimization cadence

`fit_period` controls the existing master task ticker. `optimize_period` is a
separate interval for hyperparameter searches, as documented in the upstream
[configuration reference](https://gorse.io/docs/config). A positive six-hour
optimization period must not mean a new ten-trial search on every hourly fit.

The collaborative and FM ranking searches now have independent, task-loop-owned
checkpoints. Each search is eligible on its first master cycle with usable data.
After a completed search, its configured positive interval must elapse before
another search can run. The checkpoint is taken at completion; execution occurs
on the next existing task cycle at or after that boundary. The ticker itself is
unchanged, so the interval is a minimum cooldown rather than a new exact timer.
Zero continues to disable optimization.

Dataset loading, neighbors, ordinary model fitting, recommendation updates and
garbage collection remain in their existing order and cadence. File reload,
dashboard config update/reset, user/item/feedback import and backup restore still
schedule those tasks. None is an explicit optimize-now endpoint: they share the
same optimization cooldown and cannot silently bypass it. Configured periods are
read on each cycle, so a changed period applies to the existing successful
checkpoint. Disabling and re-enabling does not erase that checkpoint.

Empty datasets are not a completed search. Errors, recovered panics and cancelled
work also leave the checkpoint unchanged and can retry on the next normal cycle;
there is no added retry loop. The existing task guard and metrics still wrap
actual attempts. A failed collaborative search does not suppress the ranking
search. Existing target-model selection and the next fitting pass are unchanged.

Checkpoints are deliberately process-local: a newly started master remains
eligible for startup optimization, matching existing startup behavior. They are
not model blobs, database migrations or Prometheus timestamps. A restart can
therefore repeat a recent search; this change does not claim restart-persistent
scheduling, eliminate startup load, or fix memory peaks or serving timeouts.

Controlled-time tests cover startup, before/exact/after boundaries, completion
time, independent periods, disable/re-enable and changed periods, empty data,
failure, panic, cancellation and the next-cycle retry. The SQLite master fixture
also verifies data refresh and fitting during optimization cooldown. Existing
configuration/import/restore scheduling tests remain applicable.

The hotfix branch is based on deployed revision
`34c968e88e62fbfb1cb20703523ebc608a6e284d`. It excludes the independent Twitch
viewer-count changes on the newer integration branch. PR head tests are pinned
to that exact source. A merge into the newer branch retains its independent
changes; any production artifact must explicitly identify the reviewed hotfix
source and be compared with the running baseline before deployment. No storage
flag, worker/schema repair, infrastructure, credential, job-count or memory-limit
change is part of this scheduler fix.
