# Argus — Multi-Camera Pose Tracking & Behavior Analysis

Real-time human pose tracking across multiple cameras with persistent identities,
monocular spatial estimation, behavior event detection, and production-style
operational tooling: camera health monitoring, structured telemetry, session
replay, a JWT-secured control API, Prometheus metrics, and Docker/CI support.

Built and iterated across seven versions as a self-directed systems project —
from a single-camera YOLO demo to a multi-camera tracker with an authenticated
control surface.

<!-- After pushing to GitHub, replace USERNAME/REPO and uncomment:
![CI](https://github.com/USERNAME/REPO/actions/workflows/ci.yml/badge.svg)
-->

> **Demo:** drop a short screen capture at `assets/demo.gif` (see
> [Recording a demo](#recording-a-demo)).

---

## What it does

- **Pose detection & tracking** — YOLO11 pose estimation with ByteTrack
  association per camera; per-track Kalman filtering (8-state constant-velocity
  box model) or an adaptive exponential smoother, selectable at runtime.
- **Cross-camera identity** — a global ID manager maps per-camera track IDs to
  persistent global IDs, with optional appearance re-identification
  (color-histogram + HOG cosine similarity).
- **Occlusion handling** — lost tracks coast on the Kalman prediction and expire
  after a configurable number of frames or confidence decay.
- **Monocular spatial estimation** — per-person distance from visible body
  height and camera vertical FOV, plus azimuth and coarse body heading; per-ID
  temporal smoothing to damp jitter.
- **Behavior events** — fast motion, fall-like posture, and raised hands,
  derived from track kinematics and keypoint geometry.
- **Camera health monitoring** — hysteresis-based state machine
  (`healthy / degraded / dropping_frames`) that resists flapping, with
  automatic capture reconnection for webcams and RTSP streams.
- **Operations surface**
  - Structured telemetry per session (CSV + JSONL) and JSON session replay
  - Optional MP4 recording
  - FastAPI control API with **JWT auth, role-based access control, and
    per-client rate limiting** (view status, toggle overlays remotely)
  - Optional Prometheus metrics endpoint
  - Audit log with optional webhook for ERROR-level events
- **Benchmark mode** — FPS, P95 latency, and memory delta on a fixed frame count.

## Architecture

```mermaid
flowchart LR
    subgraph Cameras
        C0[Webcam / RTSP 0] --> CW0[CaptureWrapper + Health]
        C1[Webcam / RTSP N] --> CW1[CaptureWrapper + Health]
    end
    CW0 --> P[YOLO11 Pose + ByteTrack per camera]
    CW1 --> P
    P --> S[MotionSmoother: Kalman or adaptive EMA]
    S --> G[GlobalIDManager + optional ReID]
    G --> SA[SpatialAnalyzer: distance / azimuth / heading]
    G --> BA[BehaviorAnalyzer: events]
    SA --> OUT[Annotated display + Telemetry CSV/JSONL + Replay/MP4]
    BA --> OUT
    OUT --> API[FastAPI control API: JWT + RBAC + rate limit]
    OUT --> MET[Prometheus metrics]
    OUT --> AUD[Audit log + webhook]
```

Module layout mirrors the diagram: `camera/` (capture + health), `core/`
(tracking, smoothing, ReID, spatial, behavior, config), `telemetry/`,
`recording/`, `api/`, `benchmark/`.

The tracking layer is deliberately decoupled from inference:
`core/smoothing.py` (Kalman/One Euro smoothing, global identity, rendering)
imports only OpenCV and NumPy, while `core/tracker.py` composes it with
YOLO/ByteTrack. The algorithmic core is therefore testable — and reusable
behind a different detector — without a multi-gigabyte model stack, which is
also why CI runs the full suite in about a minute.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --sources 0           # webcam 0; Windows: add --prefer-dshow
```

Multiple sources (mix webcams, files, RTSP):

```bash
python main.py --sources "0,rtsp://user:pass@192.168.1.20/stream1"
```

Keyboard: `Q` quit · `N` night-view enhancement · `1/2/3` toggle
pose/boxes/IDs · `E` export replay JSON.

Every run writes a session folder (`sessions/session_<timestamp>/`) containing
telemetry CSV/JSONL, an audit log, and optionally `replay.json` / `session.mp4`.

## Secured control API

```bash
pip install -r requirements-api.txt
cp rbac.example.json rbac.json
export JWT_SECRET="a-long-random-secret"     # PowerShell: $env:JWT_SECRET="..."
python main.py --sources 0 --dashboard
```

Mint a token and call the API:

```bash
python scripts/make_token.py --role operator --minutes 60
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/status
curl -X POST -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" -d '{"value": false}' \
     http://127.0.0.1:8000/toggle/boxes
```

Security behavior, all unit-tested without the web stack installed:

| Layer | Behavior |
|---|---|
| Authentication | HS256 JWT bearer tokens; missing/invalid → 401 |
| Authorization | RBAC from `rbac.json`; role lacking a capability → 403 (viewer can read, operator can toggle, admin `*`) |
| Rate limiting | Per-client token bucket → 429 |
| No secret configured | API runs as anonymous `viewer` and prints an explicit warning — it never pretends to be secured |

The API binds to `127.0.0.1` by default; exposing it beyond localhost is a
deliberate deployment decision, not a default.

## Configuration

Everything lives in `config/default.yaml` (model, cameras, tracking filter,
spatial/FOV parameters, behavior thresholds, telemetry, security). CLI flags
override the file — see `python main.py --help`.

## Benchmarks

```bash
python main.py --benchmark --benchmark-frames 500
```

Reports FPS, P95 end-to-end latency, and memory delta for your exact
hardware/model combination. Numbers vary widely between machines, so this repo
intentionally publishes the *method* rather than a table of unreproducible
results — run it and you'll have honest numbers for your setup in under a
minute.

## Testing & code quality

```bash
pip install -r requirements-dev.txt
pytest
flake8 . && mypy main.py core api camera telemetry recording benchmark && bandit -c .bandit -r .
```

The suite (59 tests) asserts behavior, not existence: Kalman convergence on
moving targets and variance reduction under noise, occlusion prediction and
track expiry, health-monitor hysteresis (no flapping, escalation and recovery
thresholds), distance monotonicity and azimuth sign, behavior-event geometry,
RBAC/rate-limit security properties, and config override semantics. CI runs
lint, type-checking, a bandit security scan, and the tests with coverage on
every push.

## Docker

```bash
docker build -f docker/Dockerfile.cpu -t argus:cpu .
docker build -f docker/Dockerfile.gpu -t argus:gpu .    # CUDA runtime base
```

## Recording a demo

```bash
python main.py --sources 0 --record-video
```

then convert a short clip to a GIF for the README, e.g.
`ffmpeg -i sessions/<session>/session.mp4 -t 8 -vf "fps=12,scale=640:-1" assets/demo.gif`.

## Model selection

The default weights (`yolo11n-pose.pt`, the nano model) prioritize real-time
latency on modest hardware. If your machine has headroom — or accuracy matters
more than framerate for your use — swap models with one flag:

```bash
python main.py --sources 0 --model yolo11s-pose.pt   # small: noticeably better keypoints
python main.py --sources 0 --model yolo11m-pose.pt   # medium: better again, GPU recommended
```

Weights download automatically on first use. The honest guidance: on CPU, the
nano model is usually the right call — a more accurate skeleton at 8 FPS reads
worse than a slightly noisier one at 25. On a CUDA GPU, `s` is typically the
sweet spot. `--benchmark` reports the real numbers for your hardware either
way.

## Latency

Perceived skeleton lag is the product of three delays, and each is addressed
explicitly:

1. **Smoothing lag** — keypoints use a **One Euro filter** (Casiez et al.,
   CHI 2012): speed-adaptive cutoff, so the skeleton hugs the raw detections
   during motion (measured in tests: <10 px lag at 600 px/s vs 40+ px for a
   fixed EMA) while still suppressing jitter at rest. Tune
   `tracking.one_euro_beta` up for even less lag, `one_euro_min_cutoff` down
   for more calm at rest.
2. **Capture lag** — a background grab thread continuously drains the camera
   and always hands the pipeline the freshest frame (`cameras.threaded_capture`),
   instead of processing frames that queued up during inference.
3. **Inference time** — shown live on the HUD (`infer <n> ms`, plus the active
   device). If it says `CPU` and you have an NVIDIA GPU, install CUDA PyTorch:
   `pip install torch --index-url https://download.pytorch.org/whl/cu121` —
   typically a 5-10x inference speedup.

## Design decisions

- **One Euro for keypoints, Kalman for boxes.** The One Euro filter is the
  interactive-tracking standard (VR/AR, MediaPipe-style pipelines) because it
  optimizes what humans actually perceive: no lag during motion, no jitter at
  rest. Boxes keep an 8-state constant-velocity Kalman filter because it also
  provides principled occlusion coasting, which a memoryless smoother cannot.
- **Health monitoring with hysteresis.** Raw frame-interval checks flap
  between states at the boundary; requiring N consecutive bad checks to
  degrade and M good checks to recover (plus a cooldown) produces stable,
  actionable states.
- **Optional dependencies degrade gracefully.** The tracker core needs only
  OpenCV/NumPy/ultralytics; FastAPI, PyJWT, and Prometheus are extras, and
  their absence disables the feature rather than crashing the app.
- **Standard config precedence.** Flag > environment > file > default (e.g.
  the JWT secret resolves `--jwt-secret`, then `JWT_SECRET`, then off-with-a-warning).

## Honest limitations

- Distance estimation is monocular and assumes a configured average person
  height — it's an estimate for situational awareness, not measurement
  (typical error grows with range and partial occlusion).
- ReID is a lightweight color-histogram + HOG similarity, useful for
  short-gap re-association under stable lighting; it is not a deep-embedding
  ReID and will confuse similarly dressed people.
- Behavior events are geometric heuristics, tuned for demo robustness rather
  than validated against a labeled dataset.
- Cross-camera global IDs currently rely on ReID similarity, not calibrated
  camera geometry.

## Roadmap

- Deep-embedding ReID (e.g. OSNet) behind the same `SimpleReID` interface
- Camera calibration + homography for true cross-camera handoff
- Event webhooks for behavior detections (the audit webhook plumbing exists)
- Track-level analytics: dwell time, zone entry/exit
- WebSocket live status stream for the control API

## License

MIT — see [LICENSE](LICENSE).
