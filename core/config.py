from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass
class ModelConfig:
    path: str = "yolo11n-pose.pt"
    conf: float = 0.50
    iou: float = 0.50
    max_det: int = 10


@dataclass
class CamerasConfig:
    sources: List[str] = field(default_factory=lambda: ["0"])
    width: int = 1280
    height: int = 720
    fps: int = 60
    prefer_dshow: bool = True
    threaded_capture: bool = True   # background grab thread → always freshest frame
    buffer_drops: int = 2           # used only when threaded_capture is false
    reconnect_attempts: int = 120


@dataclass
class BehaviorsConfig:
    fast_motion_threshold: float = 0.9
    fall_aspect_ratio: float = 1.2
    raised_hands_margin_px: int = 20


@dataclass
class SpatialConfig:
    person_height_m: float = 1.72
    fov_deg: float = 69.0
    vert_fov_deg: float = 38.8
    distance_alpha: float = 0.75
    keypoint_conf_threshold: float = 0.20


@dataclass
class TrackingConfig:
    filter: str = "kalman"              # box filter: kalman | adaptive
    keypoint_filter: str = "one_euro"   # keypoint smoothing: one_euro | ema
    one_euro_min_cutoff: float = 1.0    # Hz — jitter suppression at rest
    one_euro_beta: float = 0.05         # speed responsiveness (higher = less lag)
    alpha_fast: float = 0.45            # (ema mode) weight during fast motion
    alpha_slow: float = 0.88            # (ema mode) weight at rest
    motion_thresh: float = 0.025
    history_len: int = 600
    reid: bool = True
    reid_threshold: float = 0.65


@dataclass
class TelemetryConfig:
    out_dir: str = "sessions"
    write_csv: bool = True
    write_jsonl: bool = True
    dashboard: bool = False
    dashboard_port: int = 8000
    metrics: bool = False
    metrics_port: int = 9090


@dataclass
class SecurityConfig:
    api: bool = False
    jwt_secret_env: str = "JWT_SECRET"
    rbac_config: str = "rbac.json"
    rate_limit: float = 5.0
    rate_burst: int = 10


@dataclass
class AlertsConfig:
    webhook_url: str = ""


@dataclass
class PlaybackConfig:
    enabled: bool = True
    autosave_replay_json: bool = True


@dataclass
class HealthConfig:
    bad_checks_required: int = 3
    good_checks_required: int = 5
    degraded_mean_interval_s: float = 0.12
    dropping_frame_gap_s: float = 0.50
    cooldown_sec: float = 2.0


@dataclass
class ProcessorConfig:
    auto_fallback: bool = True
    fallback_model: str = "yolo11n-pose.pt"
    slow_infer_threshold_ms: float = 50.0


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cameras: CamerasConfig = field(default_factory=CamerasConfig)
    behaviors: BehaviorsConfig = field(default_factory=BehaviorsConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load config from YAML, falling back to defaults if the file is
        missing. A missing config file is a normal condition (first run,
        container without a mounted config) and must not crash the app."""
        p = Path(path)
        if not p.exists():
            print(f"[CONFIG] '{p}' not found — using built-in defaults.")
            return cls()
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        return cls(
            model=ModelConfig(**data.get("model", {})),
            cameras=CamerasConfig(**data.get("cameras", {})),
            behaviors=BehaviorsConfig(**data.get("behaviors", {})),
            spatial=SpatialConfig(**data.get("spatial", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            telemetry=TelemetryConfig(**data.get("telemetry", {})),
            security=SecurityConfig(**data.get("security", {})),
            alerts=AlertsConfig(**data.get("alerts", {})),
            playback=PlaybackConfig(**data.get("playback", {})),
            health=HealthConfig(**data.get("health", {})),
            processor=ProcessorConfig(**data.get("processor", {})),
        )
