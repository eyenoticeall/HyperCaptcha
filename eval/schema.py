"""Shared data structures, TOML schema, recording parsers, and exact Rust-parity math.

All probability clipping, log-odds transformations, 4-connected neighbor pooling, box
reach geometry, and 4x4/3x3 decision evaluation match `src/providers.rs`, `src/boxes.rs`,
and `src/plan.rs` verbatim.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

PROB_EPS: float = 1e-4
LOG_ODDS_3: float = 1.0 / (1.0 + math.exp(-3.0))  # 0.9525741268224334

VALID_SERVICES = ("recaptcha", "hcaptcha")
VALID_PROVIDERS = ("vllm", "liquid", "jev", "custom")
VALID_DIALECTS = ("systemone-assets", "systemone-images")
VALID_IMAGE_FORMATS = ("jpeg", "png")
VALID_CROP_FILTERS = ("bilinear", "nearest")
VALID_DECISIONS = ("logistic", "average")
EFFORT_NAMES = ("low", "medium", "high", "max")


def _build_nbm(side: int = 4) -> np.ndarray:
    n = side * side
    mask = np.zeros((n, n), dtype=bool)
    for idx in range(n):
        row, col = divmod(idx, side)
        for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= rr < side and 0 <= cc < side:
                mask[idx, rr * side + cc] = True
    return mask


# 4x4 4-connected (up/down/left/right) edge-neighbor adjacency matrix.
NBM: np.ndarray = _build_nbm(4)
NBM.flags.writeable = False


def clip_prob(p: np.ndarray | float) -> np.ndarray | float:
    """Clip probability to [1e-4, 1 - 1e-4], matching Rust `clamp(1e-4, 1.0 - 1e-4)`."""
    if isinstance(p, (int, float)):
        return min(max(float(p), PROB_EPS), 1.0 - PROB_EPS)
    return np.clip(np.asarray(p, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)


def log_odds(p: np.ndarray | float) -> np.ndarray | float:
    """Clipped log-odds `ln(p / (1 - p))`, matching Rust `providers::log_odds`."""
    if isinstance(p, (int, float)):
        clipped = min(max(float(p), PROB_EPS), 1.0 - PROB_EPS)
        return math.log(clipped / (1.0 - clipped))
    clipped = np.clip(np.asarray(p, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Logistic sigmoid `1 / (1 + exp(-z))`."""
    if isinstance(z, (int, float)):
        return 1.0 / (1.0 + math.exp(-float(z)))
    z_arr = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z_arr))


def edge_neighbors(tile: int, side: int = 4) -> list[int]:
    """1-based 4-connected edge neighbors for a tile in a `side x side` grid."""
    row, col = divmod(tile - 1, side)
    out: list[int] = []
    if row > 0:
        out.append(tile - side)
    if row + 1 < side:
        out.append(tile + side)
    if col > 0:
        out.append(tile - 1)
    if col + 1 < side:
        out.append(tile + 1)
    return out


def nbmax(x: np.ndarray) -> np.ndarray:
    """Strongest 4-connected edge-adjacent value for each tile in a 4x4 grid."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[-1] != 16:
        raise ValueError(f"nbmax expects trailing axis of length 16, got {arr.shape[-1]}")
    return np.where(NBM, arr[..., None, :], -np.inf).max(axis=-1)


def average_close_up_passes(passes: Sequence[np.ndarray | Sequence[float]]) -> np.ndarray:
    """Average close-up passes in log-odds; a single pass is returned unchanged."""
    if not passes:
        raise ValueError("at least one close-up pass is required")
    if len(passes) == 1:
        return np.asarray(passes[0], dtype=np.float64)
    total = np.zeros_like(np.asarray(passes[0], dtype=np.float64), dtype=np.float64)
    for p in passes:
        total = total + log_odds(np.asarray(p, dtype=np.float64))
    return np.asarray(sigmoid(total / float(len(passes))), dtype=np.float64)


def features_3x3(whole: np.ndarray | Sequence[float], crops: np.ndarray | Sequence[float]) -> np.ndarray:
    """Build 3x3 logistic features `[lo(whole), lo(crops)]`, shape `(..., 9, 2)`."""
    return np.stack([log_odds(np.asarray(whole, dtype=np.float64)), log_odds(np.asarray(crops, dtype=np.float64))], axis=-1)


def features_4x4(whole: np.ndarray | Sequence[float], crops: np.ndarray | Sequence[float]) -> np.ndarray:
    """Build 4x4 logistic features `[lo(W), lo(C), nbmax(lo(W)), nbmax(lo(C))]`, shape `(..., 16, 4)`."""
    lo_w = np.asarray(log_odds(np.asarray(whole, dtype=np.float64)), dtype=np.float64)
    lo_c = np.asarray(log_odds(np.asarray(crops, dtype=np.float64)), dtype=np.float64)
    return np.stack([lo_w, lo_c, nbmax(lo_w), nbmax(lo_c)], axis=-1)


def logistic_scores_3x3(
    whole: np.ndarray | Sequence[float],
    crops: np.ndarray | Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    """Compute 3x3 logistic probabilities matching `providers::logistic_scores`."""
    if len(weights) != 3:
        raise ValueError(f"3x3 logistic rule requires 3 weights, got {len(weights)}")
    a, b, bias = (float(weights[0]), float(weights[1]), float(weights[2]))
    lo_w = np.asarray(log_odds(np.asarray(whole, dtype=np.float64)), dtype=np.float64)
    lo_c = np.asarray(log_odds(np.asarray(crops, dtype=np.float64)), dtype=np.float64)
    return np.asarray(sigmoid(a * lo_w + b * lo_c + bias), dtype=np.float64)


def logistic_scores_4x4(
    whole: np.ndarray | Sequence[float],
    crops: np.ndarray | Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    """Compute 4x4 neighbor-aware logistic probabilities matching `providers::logistic_scores`."""
    if len(weights) != 5:
        raise ValueError(f"4x4 logistic rule requires 5 weights, got {len(weights)}")
    a, b, nw, nc, bias = (float(x) for x in weights)
    lo_w = np.asarray(log_odds(np.asarray(whole, dtype=np.float64)), dtype=np.float64)
    lo_c = np.asarray(log_odds(np.asarray(crops, dtype=np.float64)), dtype=np.float64)
    z = a * lo_w + b * lo_c + nw * nbmax(lo_w) + nc * nbmax(lo_c) + bias
    return np.asarray(sigmoid(z), dtype=np.float64)


def average_scores(
    whole: np.ndarray | Sequence[float],
    crops: np.ndarray | Sequence[float],
    whole_weight: float = 0.5,
) -> np.ndarray:
    """Weighted probability average matching `Combine::Average`."""
    w = np.asarray(whole, dtype=np.float64)
    c = np.asarray(crops, dtype=np.float64)
    return w * float(whole_weight) + c * (1.0 - float(whole_weight))


# ---------------------------------------------------------------------------
# Box geometry (matching src/boxes.rs)
# ---------------------------------------------------------------------------


def tile_rect(tile: int, side: int = 4) -> tuple[float, float, float, float]:
    """Normalized rectangle `(left, top, right, bottom)` for 1-based `tile`."""
    row, col = divmod(tile - 1, side)
    s = float(side)
    return (col / s, row / s, (col + 1) / s, (row + 1) / s)


def normalize_box(raw: Sequence[float] | Mapping[str, Any]) -> tuple[float, float, float, float, float] | None:
    """Normalize a box entry from fixture `[l,t,r,b,c]` or log `{'rect': ..., 'confidence': ...}`."""
    if isinstance(raw, Mapping):
        rect = raw.get("rect")
        if rect is None:
            box = raw.get("box")
            if not isinstance(box, Mapping):
                return None
            rect = (box["left"], box["top"], box["right"], box["bottom"])
        if len(rect) != 4:
            return None
        conf = float(raw.get("confidence", 0.0))
        return (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]), conf)
    if len(raw) >= 5:
        return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]), float(raw[4]))
    return None


def box_reach_metrics(rect: Sequence[float], tile: int, side: int = 4) -> tuple[float, float, float]:
    """Return `(cover, inside, box_area)` for a box against 1-based `tile`."""
    left, top, right, bottom = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    x0, y0, x1, y1 = tile_rect(tile, side)
    width = min(right, x1) - max(left, x0)
    height = min(bottom, y1) - max(top, y0)
    shared = max(width, 0.0) * max(height, 0.0)
    area = (right - left) * (bottom - top)
    cover = shared * float(side * side)
    inside = (shared / area) if area > 0.0 else -1.0
    return (cover, inside, area)


def box_reaches_tile(
    boxes: Sequence[Sequence[float] | Mapping[str, Any]],
    tile: int,
    confidence: float,
    cover: float,
    inside: float,
    side: int = 4,
) -> bool:
    """Return whether any box reaches 1-based `tile`, matching `boxes::reached` exactly."""
    x0, y0, x1, y1 = tile_rect(tile, side)
    tile_scale = float(side * side)
    for raw in boxes:
        norm = normalize_box(raw)
        if norm is None:
            continue
        left, top, right, bottom, conf = norm
        if conf < confidence:
            continue
        width = min(right, x1) - max(left, x0)
        height = min(bottom, y1) - max(top, y0)
        shared = max(width, 0.0) * max(height, 0.0)
        area = (right - left) * (bottom - top)
        if shared * tile_scale >= cover and area > 0.0 and shared / area >= inside:
            return True
    return False


def precompute_box_reach(
    boxes: Sequence[Sequence[float] | Mapping[str, Any]],
    side: int = 4,
) -> list[list[tuple[float, float, float]]]:
    """Per-tile (0-indexed) list of `(confidence, cover, inside)` for positive-area boxes."""
    n_tiles = side * side
    tile_scale = float(side * side)
    rects = [tile_rect(t + 1, side) for t in range(n_tiles)]
    per_tile: list[list[tuple[float, float, float]]] = [[] for _ in range(n_tiles)]
    for raw in boxes:
        norm = normalize_box(raw)
        if norm is None:
            continue
        left, top, right, bottom, conf = norm
        area = (right - left) * (bottom - top)
        if area <= 0.0:
            continue
        for t, (x0, y0, x1, y1) in enumerate(rects):
            width = min(right, x1) - max(left, x0)
            height = min(bottom, y1) - max(top, y0)
            shared = max(width, 0.0) * max(height, 0.0)
            if shared > 0.0:
                per_tile[t].append((conf, shared * tile_scale, shared / area))
    return per_tile


# ---------------------------------------------------------------------------
# Grid correctness metrics
# ---------------------------------------------------------------------------


def grid_correct(
    selected: np.ndarray,
    target: np.ndarray,
    empty: np.ndarray,
) -> np.ndarray:
    """Vectorized grid correctness: every target selected and no empty selected."""
    return ~((selected & empty) | (~selected & target)).any(axis=-1)


def is_grid_correct(
    picks: Sequence[int],
    targets: Sequence[int],
    slivers: Sequence[int] = (),
    *,
    strict_slivers: bool = False,
) -> bool:
    """Scalar grid correctness check for 1-based tile indices."""
    p_set = set(int(x) for x in picks)
    t_set = set(int(x) for x in targets)
    s_set = set(int(x) for x in slivers)
    allowed = t_set if strict_slivers else (t_set | s_set)
    return t_set.issubset(p_set) and p_set.issubset(allowed)


# ---------------------------------------------------------------------------
# Prompt parsing (matching src/captcha/recaptcha/mod.rs)
# ---------------------------------------------------------------------------


def allows_skipping_prompt(prompt: str, service: str = "recaptcha") -> bool:
    """Whether the prompt offers skipping (reCAPTCHA 4x4 skip prompts)."""
    if service != "recaptcha":
        return False
    lines = prompt.splitlines()
    return any("skip" in line.lower() for line in lines[2:])


def parse_prompt(prompt: str, service: str = "recaptcha") -> dict[str, Any]:
    """Split a CAPTCHA prompt into lead, target, note, and allows_skipping."""
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if service == "hcaptcha":
        return {
            "lead": "Select all matching pictures",
            "target": " ".join(lines) if lines else prompt.strip(),
            "note": "",
            "allows_skipping": False,
        }
    if len(lines) >= 2:
        lead = lines[0]
        target = lines[1]
        note = " ".join(lines[2:])
    elif len(lines) == 1:
        lead = "Select all matching tiles"
        target = lines[0]
        note = ""
    else:
        lead, target, note = "", "", ""
    return {
        "lead": lead,
        "target": target,
        "note": note,
        "allows_skipping": allows_skipping_prompt(prompt, service),
    }


# ---------------------------------------------------------------------------
# TOML / Plan dataclasses matching rust-core schema contract exactly
# ---------------------------------------------------------------------------


def normalize_close_up_spec(item: str | int | None) -> str | int:
    """Normalize a close-up pass spec to `"capture"` or positive int pixel side."""
    if item is None or item == "capture":
        return "capture"
    if isinstance(item, int) and not isinstance(item, bool):
        if item <= 0:
            raise ValueError(f"close-up pixel side must be positive, got {item}")
        return item
    if isinstance(item, str):
        text = item.strip().lower()
        if text == "capture":
            return "capture"
        if text.isdigit() and int(text) > 0:
            return int(text)
    raise ValueError(f"invalid close_ups entry: {item!r}")


@dataclass(frozen=True)
class GapRule:
    under: float
    more_than: float

    def to_dict(self) -> dict[str, float]:
        return {"under": float(self.under), "more_than": float(self.more_than)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GapRule:
        return cls(under=float(data["under"]), more_than=float(data["more_than"]))


@dataclass(frozen=True)
class BoxRule:
    exempt_from: float
    whole: float | None = None
    every: float | None = None
    windows: float | None = None
    cover: float = 0.01
    inside: float = 0.0

    def active_asks(self) -> tuple[str, ...]:
        asks: list[str] = []
        if self.whole is not None:
            asks.append("whole")
        if self.every is not None:
            asks.append("every")
        if self.windows is not None:
            asks.append("windows")
        return tuple(asks)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"exempt_from": float(self.exempt_from)}
        if self.whole is not None:
            out["whole"] = float(self.whole)
        if self.every is not None:
            out["every"] = float(self.every)
        if self.windows is not None:
            out["windows"] = float(self.windows)
        out["cover"] = float(self.cover)
        out["inside"] = float(self.inside)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoxRule:
        return cls(
            exempt_from=float(data["exempt_from"]),
            whole=None if data.get("whole") is None else float(data["whole"]),
            every=None if data.get("every") is None else float(data["every"]),
            windows=None if data.get("windows") is None else float(data["windows"]),
            cover=float(data.get("cover", 0.01)),
            inside=float(data.get("inside", 0.0)),
        )


@dataclass(frozen=True)
class Plan:
    close_ups: tuple[str | int, ...] = ("capture",)
    select_from: float = 0.5
    skip_below: float | None = None
    boxes: BoxRule | None = None
    gap: GapRule | None = None

    def __post_init__(self) -> None:
        normalized = tuple(normalize_close_up_spec(x) for x in self.close_ups)
        if not normalized:
            raise ValueError("Plan.close_ups must be non-empty")
        object.__setattr__(self, "close_ups", normalized)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "close_ups": list(self.close_ups),
            "select_from": float(self.select_from),
        }
        if self.skip_below is not None:
            out["skip_below"] = float(self.skip_below)
        if self.gap is not None:
            out["gap"] = self.gap.to_dict()
        if self.boxes is not None:
            out["boxes"] = self.boxes.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Plan:
        raw_passes = data.get("close_ups", data.get("passes", ("capture",)))
        return cls(
            close_ups=tuple(normalize_close_up_spec(x) for x in raw_passes),
            select_from=float(data.get("select_from", 0.5)),
            skip_below=None if data.get("skip_below") is None else float(data["skip_below"]),
            boxes=BoxRule.from_dict(data["boxes"]) if data.get("boxes") is not None else None,
            gap=GapRule.from_dict(data["gap"]) if data.get("gap") is not None else None,
        )


@dataclass(frozen=True)
class Logistic3x3:
    weights: tuple[float, float, float]
    cutoff: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {"weights": [float(x) for x in self.weights], "cutoff": float(self.cutoff)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Logistic3x3:
        w = data["weights"]
        if len(w) != 3:
            raise ValueError(f"logistic_3x3.weights must have 3 values, got {len(w)}")
        return cls(weights=(float(w[0]), float(w[1]), float(w[2])), cutoff=float(data.get("cutoff", 0.5)))


@dataclass(frozen=True)
class Efforts:
    low: Plan
    medium: Plan
    high: Plan
    max: Plan

    def plan(self, effort: str) -> Plan:
        if effort not in EFFORT_NAMES:
            raise ValueError(f"unknown effort tier: {effort!r}")
        return getattr(self, effort)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: self.plan(name).to_dict() for name in EFFORT_NAMES}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Efforts:
        return cls(
            low=Plan.from_dict(data["low"]),
            medium=Plan.from_dict(data["medium"]),
            high=Plan.from_dict(data["high"]),
            max=Plan.from_dict(data["max"]),
        )


@dataclass(frozen=True)
class EndpointConfig:
    dialect: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    extra_api_keys_env: str | None = None
    image_format: str | None = None
    crop_filter: str | None = None
    gzip: bool | None = None
    max_images: int | None = None
    multi_state_windows: bool | None = None
    supports_every_box: bool | None = None
    return_meta: bool | None = None
    hedge_on_stall: bool | None = None
    warm_connections: bool | None = None
    min_patch_tokens: bool | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EndpointConfig:
        return cls(
            dialect=data.get("dialect"),
            base_url=data.get("base_url"),
            api_key_env=data.get("api_key_env"),
            extra_api_keys_env=data.get("extra_api_keys_env"),
            image_format=data.get("image_format"),
            crop_filter=data.get("crop_filter"),
            gzip=data.get("gzip"),
            max_images=None if data.get("max_images") is None else int(data["max_images"]),
            multi_state_windows=data.get("multi_state_windows"),
            supports_every_box=data.get("supports_every_box"),
            return_meta=data.get("return_meta"),
            hedge_on_stall=data.get("hedge_on_stall"),
            warm_connections=data.get("warm_connections"),
            min_patch_tokens=data.get("min_patch_tokens"),
            extra_headers={str(k): str(v) for k, v in (data.get("extra_headers") or {}).items()},
        )


@dataclass(frozen=True)
class TuningConfig:
    default_decision: str = "average"
    logistic_3x3: Logistic3x3 | None = None
    logistic_4x4: tuple[float, float, float, float, float] | None = None
    average_4x4: Plan | None = None
    efforts: Efforts | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TuningConfig:
        l4 = data.get("logistic_4x4")
        weights_4x4: tuple[float, float, float, float, float] | None = None
        if l4 is not None:
            raw_w = l4["weights"] if isinstance(l4, Mapping) else l4
            if len(raw_w) != 5:
                raise ValueError(f"logistic_4x4.weights must have 5 values, got {len(raw_w)}")
            weights_4x4 = tuple(float(x) for x in raw_w)  # type: ignore[assignment]
        return cls(
            default_decision=str(data.get("default_decision", "average")),
            logistic_3x3=Logistic3x3.from_dict(data["logistic_3x3"]) if data.get("logistic_3x3") is not None else None,
            logistic_4x4=weights_4x4,
            average_4x4=Plan.from_dict(data["average_4x4"]) if data.get("average_4x4") is not None else None,
            efforts=Efforts.from_dict(data["efforts"]) if data.get("efforts") is not None else None,
        )


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int
    service: str
    name: str
    provider: str
    model_id: str
    endpoint: EndpointConfig | None = None
    tuning: TuningConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        cfg = cls(
            schema_version=int(data["schema_version"]),
            service=str(data["service"]),
            name=str(data["name"]),
            provider=str(data["provider"]),
            model_id=str(data["model_id"]),
            endpoint=EndpointConfig.from_dict(data["endpoint"]) if data.get("endpoint") is not None else None,
            tuning=TuningConfig.from_dict(data["tuning"]) if data.get("tuning") is not None else None,
        )
        validate_model_config(cfg)
        return cfg


_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def _check_prob(value: float, label: str) -> None:
    if not (0.0 <= value <= 1.0) or math.isnan(value):
        raise ValueError(f"{label} must be in [0.0, 1.0], got {value}")


def _validate_base_url(base_url: str) -> None:
    """Validate API base URL matching Rust `providers::validate_base_url`."""
    if not base_url or any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in base_url):
        raise ValueError(f"invalid endpoint.base_url {base_url!r}: must be a valid HTTP or HTTPS URL")
    if "?" in base_url or "#" in base_url:
        raise ValueError("endpoint.base_url cannot contain credentials, a query, or a fragment")
    try:
        parts = urlsplit(base_url)
        _ = parts.port
    except ValueError as exc:
        raise ValueError(f"invalid endpoint.base_url {base_url!r}: {exc}") from exc
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(f"endpoint.base_url must use HTTP or HTTPS with a host, got {base_url!r}")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError("endpoint.base_url cannot contain credentials, a query, or a fragment")


def _validate_extra_headers(headers: Mapping[str, str]) -> None:
    """Validate HTTP header names and values matching Rust `WireProfile::validate`."""
    for name, value in headers.items():
        if not isinstance(name, str) or not _HEADER_NAME_RE.match(name):
            raise ValueError(f"invalid endpoint.extra_headers name {name!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid endpoint.extra_headers value for {name!r}: must be non-empty")
        if any((ord(ch) < 0x20 and ch != "\t") or ord(ch) >= 0x7F for ch in value):
            raise ValueError(f"invalid endpoint.extra_headers value for {name!r}")


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate a ModelConfig against the shared Rust/Python TOML schema contract."""
    if cfg.schema_version != 1:
        raise ValueError(f"unsupported schema_version: {cfg.schema_version} (expected 1)")
    if cfg.service not in VALID_SERVICES:
        raise ValueError(f"invalid service {cfg.service!r}; expected one of {VALID_SERVICES}")
    if cfg.provider not in VALID_PROVIDERS:
        raise ValueError(f"invalid provider {cfg.provider!r}; expected one of {VALID_PROVIDERS}")
    if not cfg.name.strip():
        raise ValueError("name must be non-empty")
    if not cfg.model_id.strip():
        raise ValueError("model_id must be non-empty")

    supports_every = cfg.provider == "liquid"
    effective_dialect = (
        cfg.endpoint.dialect
        if (cfg.endpoint is not None and cfg.endpoint.dialect is not None)
        else ("systemone-images" if cfg.provider == "liquid" else "systemone-assets")
    )
    if cfg.endpoint is not None:
        ep = cfg.endpoint
        if ep.dialect is not None and ep.dialect not in VALID_DIALECTS:
            raise ValueError(f"invalid endpoint.dialect {ep.dialect!r}")
        if ep.base_url is not None:
            _validate_base_url(ep.base_url)
        if ep.api_key_env is not None and not ep.api_key_env.strip():
            raise ValueError("endpoint.api_key_env must be non-empty")
        if ep.extra_api_keys_env is not None and not ep.extra_api_keys_env.strip():
            raise ValueError("endpoint.extra_api_keys_env must be non-empty when set")
        if ep.image_format is not None and ep.image_format not in VALID_IMAGE_FORMATS:
            raise ValueError(f"invalid endpoint.image_format {ep.image_format!r}")
        if ep.crop_filter is not None and ep.crop_filter not in VALID_CROP_FILTERS:
            raise ValueError(f"invalid endpoint.crop_filter {ep.crop_filter!r}")
        if ep.max_images is not None:
            if isinstance(ep.max_images, bool):
                raise ValueError(f"endpoint.max_images must be an integer, got {ep.max_images!r}")
            min_images = 2 if effective_dialect == "systemone-images" else 1
            if ep.max_images < min_images:
                raise ValueError(
                    f"endpoint.max_images must be >= {min_images} for {effective_dialect}, got {ep.max_images}"
                )
        if ep.supports_every_box is not None:
            supports_every = ep.supports_every_box
        if ep.extra_headers:
            _validate_extra_headers(ep.extra_headers)

    if cfg.provider == "custom":
        if cfg.endpoint is None:
            raise ValueError("provider = 'custom' requires an [endpoint] section")
        if not cfg.endpoint.dialect or not cfg.endpoint.base_url or not cfg.endpoint.api_key_env:
            raise ValueError("provider = 'custom' requires endpoint.dialect, base_url, and api_key_env")

    if cfg.tuning is not None:
        t = cfg.tuning
        if t.default_decision not in VALID_DECISIONS:
            raise ValueError(f"invalid tuning.default_decision {t.default_decision!r}")
        if t.logistic_3x3 is not None:
            if not all(math.isfinite(w) for w in t.logistic_3x3.weights):
                raise ValueError("tuning.logistic_3x3.weights must all be finite")
            _check_prob(t.logistic_3x3.cutoff, "tuning.logistic_3x3.cutoff")
        if t.logistic_4x4 is not None:
            if not all(math.isfinite(w) for w in t.logistic_4x4):
                raise ValueError("tuning.logistic_4x4.weights must all be finite")
        plans: list[tuple[str, Plan]] = []
        if t.average_4x4 is not None:
            plans.append(("tuning.average_4x4", t.average_4x4))
        if t.efforts is not None:
            for tier in EFFORT_NAMES:
                plans.append((f"tuning.efforts.{tier}", t.efforts.plan(tier)))
        for prefix, plan in plans:
            _check_prob(plan.select_from, f"{prefix}.select_from")
            if plan.skip_below is not None:
                _check_prob(plan.skip_below, f"{prefix}.skip_below")
            if plan.gap is not None:
                _check_prob(plan.gap.under, f"{prefix}.gap.under")
                _check_prob(plan.gap.more_than, f"{prefix}.gap.more_than")
            if plan.boxes is not None:
                b = plan.boxes
                if not b.active_asks():
                    raise ValueError(f"{prefix}.boxes must set at least one of whole, every, or windows")
                _check_prob(b.exempt_from, f"{prefix}.boxes.exempt_from")
                _check_prob(b.cover, f"{prefix}.boxes.cover")
                _check_prob(b.inside, f"{prefix}.boxes.inside")
                if b.whole is not None:
                    _check_prob(b.whole, f"{prefix}.boxes.whole")
                if b.every is not None:
                    _check_prob(b.every, f"{prefix}.boxes.every")
                    if not supports_every:
                        raise ValueError(f"{prefix}.boxes.every requires supports_every_box = true")
                if b.windows is not None:
                    _check_prob(b.windows, f"{prefix}.boxes.windows")


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a TOML ModelConfig via stdlib `tomllib`."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return ModelConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Exact decision evaluation matching src/providers.rs & src/boxes.rs
# ---------------------------------------------------------------------------


def _resolve_close_up_pass(
    close_up_passes: Mapping[Any, Sequence[float]],
    spec: str | int,
) -> np.ndarray:
    candidates: list[Any] = [spec]
    if spec == "capture":
        candidates.extend([None, "captured", "p1"])
    elif isinstance(spec, int):
        candidates.extend([str(spec), f"at_{spec}"])
        if spec == 506:
            candidates.append("p2")
    for key in candidates:
        if key in close_up_passes and close_up_passes[key] is not None:
            return np.asarray(close_up_passes[key], dtype=np.float64)
    raise KeyError(f"missing close-up pass {spec!r}; available keys: {list(close_up_passes)}")


def decide_3x3(
    whole: Sequence[float] | np.ndarray,
    crops: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | None = None,
    cutoff: float = 0.5,
    decision: str = "logistic",
    whole_weight: float = 0.5,
) -> list[int]:
    """Evaluate a 3x3 grid and return sorted 1-based tile selections."""
    w = np.asarray(whole, dtype=np.float64)
    c = np.asarray(crops, dtype=np.float64)
    if decision == "logistic":
        if weights is None:
            raise ValueError("logistic decision requires 3 weights")
        scores = logistic_scores_3x3(w, c, weights)
    elif decision == "average":
        scores = average_scores(w, c, whole_weight)
    else:
        raise ValueError(f"unknown decision rule {decision!r}")
    return [t + 1 for t in range(9) if float(scores[t]) >= float(cutoff)]


def decide_4x4(
    whole: Sequence[float] | np.ndarray,
    close_up_passes: Mapping[Any, Sequence[float]] | Sequence[Sequence[float]],
    plan: Plan,
    *,
    weights: Sequence[float] | None = None,
    decision: str = "logistic",
    whole_weight: float = 0.5,
    presence: float | None = None,
    whole_boxes: Sequence[Sequence[float] | Mapping[str, Any]] | None = None,
    every_boxes: Sequence[Sequence[float] | Mapping[str, Any]] | None = None,
    window_boxes: Sequence[Sequence[float] | Mapping[str, Any]] | None = None,
) -> list[int]:
    """Evaluate a 4x4 grid under `plan`, reproducing Rust `src/providers.rs` exactly.

    Steps in order:
    1. Average configured close-up passes in log-odds (single pass unchanged).
    2. Combine whole-grid and close-up scores via logistic or weighted average.
    3. Select tiles scoring `>= plan.select_from`.
    4. Skip check: if `plan.skip_below` is set and `presence < plan.skip_below`, return `[]`.
    5. Gap rule: drop tiles `< gap.under` where `close_up - whole > gap.more_than`.
    6. Box rule: drop tiles `< boxes.exempt_from` unless EVERY configured box ask reaches it.
    """
    w = np.asarray(whole, dtype=np.float64)
    if isinstance(close_up_passes, Mapping):
        selected_passes = [_resolve_close_up_pass(close_up_passes, spec) for spec in plan.close_ups]
    else:
        if len(close_up_passes) != len(plan.close_ups):
            raise ValueError(
                f"expected {len(plan.close_ups)} close-up passes for plan, got {len(close_up_passes)}"
            )
        selected_passes = [np.asarray(p, dtype=np.float64) for p in close_up_passes]

    close_ups = average_close_up_passes(selected_passes)
    if decision == "logistic":
        if weights is None:
            raise ValueError("logistic decision requires 5 weights")
        scores = logistic_scores_4x4(w, close_ups, weights)
    elif decision == "average":
        scores = average_scores(w, close_ups, whole_weight)
    else:
        raise ValueError(f"unknown decision rule {decision!r}")

    picks = [t for t in range(1, 17) if float(scores[t - 1]) >= plan.select_from]

    if (
        plan.skip_below is not None
        and presence is not None
        and not math.isnan(float(presence))
        and float(presence) < plan.skip_below
    ):
        return []

    if plan.gap is not None:
        gap = plan.gap
        picks = [
            t
            for t in picks
            if float(scores[t - 1]) >= gap.under
            or (float(close_ups[t - 1]) - float(w[t - 1])) <= gap.more_than
        ]

    if plan.boxes is not None:
        rule = plan.boxes
        kept: list[int] = []
        for t in picks:
            if float(scores[t - 1]) >= rule.exempt_from:
                kept.append(t)
                continue
            ok = True
            if rule.whole is not None:
                ok = ok and box_reaches_tile(whole_boxes or (), t, rule.whole, rule.cover, rule.inside)
            if ok and rule.every is not None:
                ok = ok and box_reaches_tile(every_boxes or (), t, rule.every, rule.cover, rule.inside)
            if ok and rule.windows is not None:
                ok = ok and box_reaches_tile(window_boxes or (), t, rule.windows, rule.cover, rule.inside)
            if ok:
                kept.append(t)
        picks = kept

    return picks


# ---------------------------------------------------------------------------
# JSONL manifest, label, and recording loaders shared by fit / sweep / export
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSON lines from a file."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def load_dataset(path: str | Path, layout: int | None = None) -> list[dict[str, Any]]:
    """Load `dataset.jsonl` manifest rows, optionally filtered by layout."""
    rows = read_jsonl(path)
    if layout is not None:
        rows = [r for r in rows if int(r["layout"]) == layout]
    return rows


def load_labels(path: str | Path, *, include_unclear: bool = False) -> dict[str, dict[str, Any]]:
    """Load `labels.jsonl` keeping the latest entry per grid ID."""
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        gid = str(row.get("id", row.get("grid")))
        target = sorted({int(x) for x in row.get("target", row.get("contains", ()))})
        sliver = sorted({int(x) for x in row.get("sliver", row.get("barely", ()))})
        unclear = bool(row.get("unclear", False))
        latest[gid] = {
            "id": gid,
            "layout": row.get("layout"),
            "target": target,
            "sliver": [t for t in sliver if t not in set(target)],
            "unclear": unclear,
            "saved_at": row.get("saved_at", row.get("savedAt")),
        }
    if not include_unclear:
        return {gid: rec for gid, rec in latest.items() if not rec["unclear"]}
    return latest


def extract_tile_vector(raw: Mapping[str, Any] | Sequence[float] | None, n_tiles: int) -> np.ndarray | None:
    """Extract ordered tile probabilities `[tile_1, ..., tile_N]` from dict or list."""
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        vals: list[float] = []
        for t in range(1, n_tiles + 1):
            val = raw.get(str(t), raw.get(t))
            if val is None:
                return None
            vals.append(float(val))
        return np.array(vals, dtype=np.float64)
    if len(raw) == n_tiles:
        return np.asarray(raw, dtype=np.float64)
    return None


def parse_recording_entry(record: Mapping[str, Any], layout: int) -> dict[str, Any] | None:
    """Parse one `recordings.jsonl` record into standardized scores, boxes, and tokens."""
    n_tiles = layout * layout
    result = record.get("result")
    if not isinstance(result, Mapping) or result.get("status") != "answered":
        return None

    comp = result.get("vllm_components") or result.get("jev_components") or {}
    if not isinstance(comp, Mapping):
        return None

    whole_comp = comp.get("whole_grid") or {}
    variant_comp = comp.get("variant") or {}
    whole = extract_tile_vector(whole_comp.get("tile_probabilities"), n_tiles)
    if whole is None:
        return None

    passes_map: dict[str | int, np.ndarray] = {}
    raw_passes = variant_comp.get("passes")
    if layout == 4 and isinstance(raw_passes, Sequence) and len(raw_passes) > 0:
        if len(raw_passes) == 1:
            p0 = extract_tile_vector(raw_passes[0].get("tile_probabilities"), n_tiles)
            if p0 is None:
                return None
            pixels = raw_passes[0].get("grid_pixels") or []
            key: str | int = 506 if (len(pixels) == 2 and max(pixels) == 506) else "capture"
            passes_map[key] = p0
        else:
            p0 = extract_tile_vector(raw_passes[0].get("tile_probabilities"), n_tiles)
            p1 = extract_tile_vector(raw_passes[1].get("tile_probabilities"), n_tiles)
            if p0 is None or p1 is None:
                return None
            # Plan::record sends [None ("capture"), Some(506)] in order; check pixels if swapped.
            pix0 = raw_passes[0].get("grid_pixels") or []
            pix1 = raw_passes[1].get("grid_pixels") or []
            if len(pix0) == 2 and max(pix0) == 506 and (len(pix1) != 2 or max(pix1) != 506):
                passes_map[506] = p0
                passes_map["capture"] = p1
            else:
                passes_map["capture"] = p0
                passes_map[506] = p1
    else:
        crops = extract_tile_vector(variant_comp.get("tile_probabilities"), n_tiles)
        if crops is None:
            return None
        passes_map["capture"] = crops

    presence_comp = comp.get("presence") or {}
    presence_val = presence_comp.get("probability")
    presence = None if presence_val is None else float(presence_val)

    boxes_comp = comp.get("boxes") or {}
    whole_boxes_raw = (boxes_comp.get("whole_grid") or {}).get("boxes")
    every_boxes_raw = (boxes_comp.get("every") or {}).get("boxes")
    window_boxes_raw = (boxes_comp.get("windows") or {}).get("boxes")

    def _norm_boxes(raw_list: Any) -> list[list[float]] | None:
        if raw_list is None:
            return None
        out: list[list[float]] = []
        for item in raw_list:
            b = normalize_box(item)
            if b is not None:
                out.append([b[0], b[1], b[2], b[3], b[4]])
        return out

    def _input_tokens(node: Any) -> int | None:
        if not isinstance(node, Mapping):
            return None
        usage = node.get("usage")
        if isinstance(usage, Mapping) and usage.get("input_tokens") is not None:
            return int(usage["input_tokens"])
        return None

    win_meta = (boxes_comp.get("windows") or {}).get("meta") or {}
    window_requests = int(win_meta.get("requests", 1)) if isinstance(win_meta, Mapping) else 1

    tokens = {
        "whole": _input_tokens(whole_comp),
        "variant": _input_tokens(variant_comp),
        "presence": _input_tokens(presence_comp),
        "whole_boxes": _input_tokens(boxes_comp.get("whole_grid")),
        "every_boxes": _input_tokens(boxes_comp.get("every")),
        "window_boxes": _input_tokens(boxes_comp.get("windows")),
        "total": _input_tokens(result),
    }

    return {
        "grid_id": str(record.get("grid_id", record.get("grid"))),
        "pass_index": int(record.get("pass_index", 0)),
        "layout": layout,
        "whole": whole,
        "passes": passes_map,
        "presence": presence,
        "whole_boxes": _norm_boxes(whole_boxes_raw),
        "every_boxes": _norm_boxes(every_boxes_raw),
        "window_boxes": _norm_boxes(window_boxes_raw),
        "window_requests": window_requests,
        "tokens": tokens,
        "resolved_model": result.get("resolved_model"),
    }


def make_group_folds(
    group_ids: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Deterministic grouped K-fold assignment matching `lib.py:load`."""
    if isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    groups = sorted(set(str(g) for g in group_ids))
    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError(f"grouped CV requires at least 2 distinct groups, got {n_groups}")
    k = min(n_folds, n_groups)
    idx_map = {g: i for i, g in enumerate(groups)}
    gi = np.array([idx_map[str(g)] for g in group_ids], dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_groups)
    fold_of_group = np.empty(n_groups, dtype=np.int64)
    fold_of_group[order] = np.arange(n_groups, dtype=np.int64) % k
    return gi, fold_of_group[gi], n_groups, k
