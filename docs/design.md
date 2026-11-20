# Design notes

The decisions that shaped the system, and what each one costs.

## One feature function, two callers

Session features (last 50 items, 30-minute counts, skip rate, session length) are computed by a
single incremental function, `recsys.features.update`. The stream processor calls it per event
and pushes the result to Feast. Offline, `session_feature_log` replays the same events through
the same function and writes the rows that Feast's point-in-time join reads when building
training data. Nothing is reimplemented in SQL or pandas for training, so there is no second
implementation to drift.

The alternative is defining features declaratively (Feast on-demand transformations, a stream SQL
engine) and letting the platform compute both paths. That scales better to many feature
authors, but on a project this size it would have added a runtime and a DSL. The trade-off I
accepted: the offline log is built in a Python loop (about 30 s for the 16 days it needs), which
would not scale to the 5B dataset without a distributed replay.

What keeps the two paths honest is a test, not the design: `tests/test_features.py` drives the
real stream code against a real online store and compares every feature with Feast's
point-in-time join at the same timestamp. It caught a bug the design did not prevent.
Yambda bins timestamps to 5 seconds, and Feast's Redis store silently drops a write whose
timestamp is not newer than the stored one, so the second event in a bin never reached Redis
while the offline join used it. Each event in a bin now gets a microsecond offset.

## Feast push source instead of the stream writing to Redis

The stream processor writes through `FeatureStore.push`, not with raw Redis commands. That costs
a few milliseconds per micro-batch and pulls Feast into the stream image, but the entity key
serialisation, the feature view schema and the read path in the API all come from one registry.
Writing to Redis directly would be faster, and it would also be the most likely way to
introduce a serving-only bug. The `feast` compose service is a one-shot job that applies the
definitions and materialises the daily user snapshot; there is no Feast feature server, since
the API reads Redis through the SDK in 1-2 ms.

Item features (popularity, played ratio, age) are not in Feast at all. They change once a day,
are needed for ~600 candidates per request, and fit in memory, so they ship inside the model
bundle. Fetching 600 entities from Redis per request would cost more than the whole ranker.

## Redpanda for the event log

The replayer and the stream processor talk through a Kafka topic keyed by user, so per-user
order is guaranteed and the stream can be restarted and re-read from the beginning. Redpanda is
Kafka-compatible, ships as one container with no ZooKeeper/KRaft setup, and runs in 200 MB with
`--memory 512M`, which matters inside an 8 GB Colima VM. The cost is that the stream's state lives
in process memory: on restart it rebuilds from the bootstrap snapshot and re-reads the topic
(about 40 s). A production version would checkpoint state or use a stream framework with managed
state.

## Latency budget and where it goes

The target was p99 under 50 ms at the API. In isolation a request takes about 5 ms at the median:
0.6 ms for the Feast read, 2.5 ms retrieval (IVF-SQ8 Faiss with nprobe=128, the co-occurrence
lookup, the cold-start pool), 1.4 ms for LightGBM over ~650 candidates and 0.2 ms for MMR. The
budget is lost under concurrency, not per request: one Python process is GIL-bound, so the tail
grows once the request rate approaches what one core can do, and on a laptop the VM's CPUs are
also shared with everything else. Shadow scoring runs a second full pipeline in the same process,
which is why only a sample of requests is shadowed. Things that kept the per-request cost low:
single-threaded Faiss and LightGBM (thread pools cost more than they save for one query), a
contiguous copy of the cold-pool vectors instead of a gather per request, and a warm-up before
the health check passes. Bundle arrays are memory-mapped so that several workers could share
one copy; in the 8 GB VM three workers still hit the container's memory limit and got slower, so
the measured configuration is one worker.

Every request has a 25 ms timeout on the feature read. On timeout, Redis errors or a model
exception, the API returns the recent-popularity list and labels the response with the reason;
the fallback rate is a Prometheus metric with an alert at 5%.

## Shadow mode before canary

The challenger scores the same request after the response is sent (a FastAPI background task)
and only overlap@k is recorded. Shadow mode measures how different the challenger is without
exposing a single user to it, and needs no traffic splitting. It cannot measure whether users
prefer the challenger; that needs a canary or an A/B test with outcome logging, which a replayed
dataset cannot answer anyway. The costs are CPU (above) and that overlap is only a proxy: a
challenger with 60% overlap could be better or worse. Promotion is an alias move in the MLflow
registry plus a reload call (`scripts/promote.py`); rollback is the reverse, since the previous
champion is kept as the `@previous` alias.

## Evaluation that matches serving

Offline metrics are computed by sending every test request through `recsys.serving.recommend`,
the same function the API calls, with features from the point-in-time join. Requests are placed
one second before a random listen during the test day, and positives are tracks played at least
50% in the following hour, minus anything in the user's last 50 listens. That last rule
matters: 69% of test-day positives are repeats, and since the product never recommends a
recently played track, counting them would reward a model for something it is not allowed to do.
