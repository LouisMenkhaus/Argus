# Changelog

## 1.3.0 — Decoupled tracking layer, latency bug fix (2026-07)

### Fixed
- **One Euro filter froze when frames arrived faster than real time.** The
  filter derives its response from wall-clock time between samples, which is
  correct for a live camera but collapses toward zero in benchmark mode, video
  file playback, and batch replay — the smoothing factor went with it and the
  output stopped following its input. Measured on a synthetic walk, keypoint
  lag was **19.5 px against 2.7 px for the box** in the same run; after
  clamping dt to a plausible frame interval, keypoint lag is **2.7 px**. Two
  regression tests cover both the fast-delivery case and normal framerate.
- Keypoint filter state is now primed on a track's first frame, so per-track
  smoothing state is consistent from the start.

### Changed
- **Tracking logic decoupled from inference.** `MotionSmoother`,
  `GlobalIDManager`, `TrackState`, and rendering moved to `core/smoothing.py`,
  which imports only OpenCV/NumPy; `core/tracker.py` retains the YOLO/ByteTrack
  layer and re-exports the moved names, so existing imports keep working.
  This closed a real test gap: 7 tests had been *silently* skipping whenever
  torch was absent (an early `return` counts as a pass), and running them for
  real immediately surfaced the latency bug above.
- CI rewritten: the suite no longer needs torch/ultralytics, so runs finish in
  roughly a minute across Python 3.11 and 3.12. Tests, flake8, and bandit are
  blocking; mypy is advisory during incremental type adoption.
- Cleared all 33 line-length violations and replaced every bare
  `try/except/pass` with documented `contextlib.suppress` blocks.

### Added
- Suite grown to 54 tests, all genuinely executing (no silent skips).


## 1.2.0 — Application-ready polish (2026-07)

- **Model selection from the CLI**: `--model yolo11s-pose.pt` (and
  `--keypoint-filter one_euro|ema`) without editing config. See the README's
  model-selection notes for the accuracy/latency tradeoff.
- **Full-pipeline integration tests** (4): scripted detection sequences flow
  through smoothing -> spatial -> behavior, asserting identity persistence,
  pipeline-level tracking lag, monotonic distance on approach, a scripted
  collapse raising a fall event, and lost-track expiry. Suite now 50 tests.
- Docs: model-selection guidance; test counts updated.

## 1.1.1 — Hotfix (2026-07)

- **Fixed a startup race in threaded capture** that presented as "camera
  can't open": the main loop could poll before the grabber thread delivered
  its first frame, triggering a reconnect that reopened the device and
  restarted the wait — an endless reconnect storm. The capture wrapper now
  blocks up to 8 s for the first frame at construction (webcam/RTSP
  handshakes legitimately take 1-3 s) and `read_latest` polls for ~4 frame
  intervals instead of failing in the instant between grabs. Two regression
  tests added, including an immediate-read-after-construction test that
  reproduces the original race with a slow-start fake device.


## 1.1.0 — Latency release (2026-07)

Perceived skeleton lag traced to three stacked delays and addressed:

- **One Euro filter for keypoints (new default).** The previous fixed EMA
  (alpha_slow 0.88) gave new detections only ~12% weight at low motion — the
  skeleton visibly trailed the person. The One Euro filter adapts its cutoff
  to speed: tests show <10 px lag during 600 px/s motion (vs 40+ px for the
  EMA) with jitter at rest still reduced >40%. Legacy EMA remains available
  via `tracking.keypoint_filter: ema`.
- **Threaded capture (new default).** A background grabber continuously
  drains the camera and serves the freshest frame, removing camera I/O from
  the critical path and dropping stale frames instead of processing them.
- **Latency visibility.** HUD now shows inference ms and the active device;
  startup prints an explicit hint (with the install command) when inference
  is running on CPU with a likely-idle GPU.
- 9 new tests (44 total): One Euro step-response vs legacy EMA, rest-jitter
  suppression, threaded-capture freshness, health ticking, and lifecycle.


## 1.0.0 — Portfolio release (2026-07)

First stable, reviewed release. Consolidates the v1–v6 prototype line into a
single versioned project ("Argus").

### Fixed
- **Kalman filter crashed on NumPy >= 2.0** (the default tracking mode).
  `cv2.KalmanFilter` state vectors are column vectors; scalar reads now index
  `[row, 0]` instead of relying on 1-element array conversion, which NumPy 2.x
  removed. Added `KalmanTracker.predict()` so occlusion handling goes through
  the same safe path.
- **Track overlay colors** — every identity rendered black; restored a
  deterministic 10-color palette keyed by global ID.
- **Missing config file crashed startup** — `AppConfig.from_yaml` now falls
  back to built-in defaults with a console notice.
- GPU Dockerfile silently swallowed dependency install failures (`|| true`).

### Changed
- Project renamed to **Argus**; entry point is `main.py`.
- JWT secret is now read from the environment (`JWT_SECRET` by default,
  configurable) instead of only a CLI flag; the API prints a clear warning
  when started without auth.
- The API's auth primitives (token bucket, per-client rate limiter, RBAC) are
  importable and unit-tested without the optional FastAPI stack installed.
- CI: pip caching, coverage reporting on core modules, updated entrypoints.

### Added
- Behavioral test suite (35 tests): Kalman convergence/occlusion, camera
  health hysteresis, monocular distance monotonicity, behavior events,
  RBAC/rate-limit security properties, ReID similarity, config overrides.
- `rbac.example.json` and `scripts/make_token.py` for demonstrating the
  secured API end to end.
- MIT license, `.gitignore`, this changelog.

## Pre-1.0 (v1–v6 prototype history)

Iterative builds: single-camera YOLO pose demo → ByteTrack integration →
multi-camera with global IDs → adaptive/Kalman smoothing, simple ReID, RTSP
support, camera health monitoring, telemetry (CSV/JSONL), session replay and
MP4 recording, benchmark mode, FastAPI control surface with JWT + RBAC,
Prometheus metrics, Docker images, CI scaffolding.
