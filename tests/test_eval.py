"""Comprehensive offline test suite for HyperCaptcha's `eval` toolkit."""

from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from eval.cli import main as cli_main
from eval.dataset import UNIFORM_3X3_PICTURES, build_dataset
from eval.export import export_model, render_model_toml, write_parity_fixture
from eval.fit import fit_dataset
from eval.label_server import (
    LabelStore,
    _is_loopback_host_header,
    compare_labels,
    create_label_server,
)
from eval.record import _has_recording_error, record_dataset
from eval.schema import (
    EFFORT_NAMES,
    LOG_ODDS_3,
    BoxRule,
    Efforts,
    EndpointConfig,
    GapRule,
    Logistic3x3,
    ModelConfig,
    Plan,
    TuningConfig,
    average_close_up_passes,
    decide_3x3,
    decide_4x4,
    is_grid_correct,
    load_model_config,
    logistic_scores_3x3,
    logistic_scores_4x4,
    make_group_folds,
    validate_model_config,
)
from eval.summarize import format_summary_markdown, summarize_files, summarize_trials, wilson_proportion
from eval.sweep import grouped_bootstrap_diff, sweep_4x4

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Tiny valid 1x1 PNG bytes for offline dataset/label/record tests.
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c63f8cfc000000301010018dd8db00000000049454e44ae426082"
)


# ---------------------------------------------------------------------------
# 1. Exact Rust-parity checks against shipped .json.gz fixtures & unit vectors
# ---------------------------------------------------------------------------


def _vllm_d3_ultra_config() -> ModelConfig:
    return ModelConfig(
        schema_version=1,
        service="recaptcha",
        name="vLLM/d3-ultra",
        provider="vllm",
        model_id="vllm-sr/d3-ultra",
        tuning=TuningConfig(
            default_decision="logistic",
            logistic_3x3=Logistic3x3(weights=(0.571457, 0.939705, -1.719964), cutoff=0.5),
            logistic_4x4=(0.223152, 1.092908, 0.213385, 0.18743, -1.509533),
            average_4x4=Plan(close_ups=("capture", 506), select_from=0.5, skip_below=0.7),
            efforts=Efforts(
                low=Plan(
                    close_ups=(506,),
                    select_from=0.55,
                    skip_below=0.7,
                    gap=GapRule(under=0.6, more_than=0.3),
                ),
                medium=Plan(
                    close_ups=(506,),
                    select_from=0.55,
                    skip_below=0.7,
                    boxes=BoxRule(exempt_from=LOG_ODDS_3, whole=0.3, cover=0.01, inside=0.0),
                ),
                high=Plan(
                    close_ups=("capture", 506),
                    select_from=0.5,
                    skip_below=0.7,
                    boxes=BoxRule(exempt_from=LOG_ODDS_3, whole=0.3, cover=0.01, inside=0.01),
                ),
                max=Plan(
                    close_ups=("capture", 506),
                    select_from=0.5,
                    skip_below=0.7,
                    boxes=BoxRule(exempt_from=LOG_ODDS_3, whole=0.02, windows=0.5, cover=0.01, inside=0.0),
                ),
            ),
        ),
    )


def _liquid_d1_2_config() -> ModelConfig:
    with_boxes = Plan(
        close_ups=("capture",),
        select_from=0.2,
        skip_below=0.5,
        boxes=BoxRule(exempt_from=0.9, every=0.3, cover=0.01, inside=0.0),
    )
    return ModelConfig(
        schema_version=1,
        service="recaptcha",
        name="Liquid/d1.2",
        provider="liquid",
        model_id="d1.2",
        tuning=TuningConfig(
            default_decision="logistic",
            logistic_4x4=(0.02249, 2.075111, 0.598377, 0.516936, -0.64329),
            average_4x4=Plan(close_ups=("capture",), select_from=0.5),
            efforts=Efforts(
                low=Plan(close_ups=("capture",), select_from=0.5, skip_below=0.5),
                medium=with_boxes,
                high=with_boxes,
                max=with_boxes,
            ),
        ),
    )


def _liquid_d1_2_ultra_config() -> ModelConfig:
    boxes = BoxRule(exempt_from=0.9, every=0.3, cover=0.01, inside=0.0)
    high = Plan(close_ups=("capture", 506), select_from=0.35, skip_below=0.5, boxes=boxes)
    return ModelConfig(
        schema_version=1,
        service="recaptcha",
        name="Liquid/d1.2-ultra",
        provider="liquid",
        model_id="d1.2-ultra",
        tuning=TuningConfig(
            default_decision="logistic",
            logistic_4x4=(0.325707, 1.717633, 0.364472, 0.151428, -2.083722),
            average_4x4=Plan(close_ups=("capture",), select_from=0.5),
            efforts=Efforts(
                low=Plan(close_ups=(506,), select_from=0.4, skip_below=0.5),
                medium=Plan(close_ups=(506,), select_from=0.4, skip_below=0.5, boxes=boxes),
                high=high,
                max=high,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected_count", "cfg_factory"),
    [
        ("vllm_d3_ultra_4x4.json.gz", 644, _vllm_d3_ultra_config),
        ("liquid_d1_2_4x4.json.gz", 339, _liquid_d1_2_config),
        ("liquid_d1_2_ultra_4x4.json.gz", 339, _liquid_d1_2_ultra_config),
    ],
)
def test_fixture_parity_against_rust_reference(
    tmp_path: Path,
    fixture_name: str,
    expected_count: int,
    cfg_factory: Any,
) -> None:
    """Verify 100% pick parity across all shipped fixtures (and TOML round-trip)."""
    fixture_path = FIXTURES_DIR / fixture_name
    assert fixture_path.is_file(), f"missing fixture {fixture_path}"

    # Round-trip the ModelConfig through TOML serialization + tomllib parsing first.
    toml_text = render_model_toml(cfg_factory())
    toml_path = tmp_path / f"{fixture_name}.toml"
    toml_path.write_text(toml_text, encoding="utf-8")
    cfg = load_model_config(toml_path)
    assert cfg.tuning is not None and cfg.tuning.efforts is not None and cfg.tuning.logistic_4x4 is not None

    with gzip.open(fixture_path, "rt", encoding="utf-8") as handle:
        grids = json.load(handle)
    assert len(grids) == expected_count

    for tier in EFFORT_NAMES:
        plan = cfg.tuning.efforts.plan(tier)
        mismatches: list[str] = []
        for g in grids:
            got = decide_4x4(
                g["whole"],
                {"capture": g["captured"], 506: g["at_506"]},
                plan,
                weights=cfg.tuning.logistic_4x4,
                decision="logistic",
                presence=g["presence"],
                whole_boxes=g.get("whole_boxes"),
                every_boxes=g.get("every_boxes"),
                window_boxes=g.get("window_boxes"),
            )
            want = g["picks"][tier]
            if got != want:
                mismatches.append(f"{g['id']}: got {got}, want {want}")
        assert not mismatches, f"{fixture_name} [{tier}] had {len(mismatches)} mismatches: {mismatches[:5]}"


def test_logistic_reference_vectors_match_rust_unit_tests() -> None:
    """Cross-check 3x3 and 4x4 logistic vectors from `src/providers.rs` unit tests."""
    # 3x3 vector from src/providers.rs:2563-2598
    w3 = [0.910962, 0.0212042, 0.996278, 0.999845, 0.999755, 0.00590522, 0.00108685, 0.0034534, 0.000995191]
    c3 = [0.497478, 0.0124854, 0.999416, 0.999385, 0.999831, 0.0157936, 0.00317473, 0.00289398, 0.000319073]
    s3 = logistic_scores_3x3(w3, c3, [0.571457, 0.939705, -1.719964])
    want3 = [0.401178, 0.000329711, 0.999791, 0.999964, 0.999986, 0.000196957, 1.63392e-05, 2.90287e-05, 1.78865e-06]
    assert np.allclose(s3, want3, atol=1e-5)
    assert decide_3x3(w3, c3, weights=[0.571457, 0.939705, -1.719964], cutoff=0.5) == [3, 4, 5]

    # 4x4 vector from src/providers.rs:2968-2982
    w4 = [
        0.240337, 0.750125, 0.69387, 0.613134, 0.957105, 0.995155, 0.991341, 0.993727,
        0.182279, 0.910848, 0.821835, 0.848625, 0.931801, 0.985958, 0.988057, 0.992624,
    ]
    c4 = [
        0.463749, 0.983213, 0.942063, 0.669614, 0.998127, 0.999454, 0.996333, 0.998662,
        0.019002, 0.972076, 0.725535, 0.974846, 0.996249, 0.988316, 0.991891, 0.991116,
    ]
    s4 = logistic_scores_4x4(w4, c4, [0.223152, 1.092908, 0.213385, 0.18743, -1.509533])
    want4 = [
        0.478572, 0.996758, 0.97775, 0.843709, 0.999814, 0.999958, 0.99973, 0.999865,
        0.0131823, 0.995649, 0.876105, 0.994475, 0.999007, 0.998126, 0.998719, 0.998614,
    ]
    assert np.allclose(s4, want4, atol=1e-5)

    # Log-odds pass average matches Rust test: mean of ln(9) and 0 is ln(3) -> 0.75.
    avg = average_close_up_passes([[0.9], [0.5]])
    assert math.isclose(float(avg[0]), 0.75, abs_tol=1e-12)
    assert is_grid_correct([1, 2], [1], [2], strict_slivers=False)
    assert not is_grid_correct([1, 2], [1], [2], strict_slivers=True)


# ---------------------------------------------------------------------------
# 2. Dataset builder deduplication, grouping, and filtering
# ---------------------------------------------------------------------------


def test_dataset_builder_dedup_grouping_and_filtering(tmp_path: Path) -> None:
    run_rc = tmp_path / "run_recaptcha"
    run_hc = tmp_path / "run_hcaptcha"
    (run_rc / "trial-A").mkdir(parents=True)
    (run_rc / "trial-B").mkdir(parents=True)
    (run_hc / "hc-1").mkdir(parents=True)

    img1 = run_rc / "trial-A" / "round-01.png"
    img2 = run_rc / "trial-A" / "round-02.png"
    img_dup = run_rc / "trial-B" / "round-01.png"
    img3x3 = run_rc / "trial-B" / "round-02.png"
    img_hc = run_hc / "hc-1" / "grid-01.png"

    img1.write_bytes(TINY_PNG + b"grid-a1")
    img2.write_bytes(TINY_PNG + b"grid-a2")
    img_dup.write_bytes(TINY_PNG + b"grid-a1")  # exact duplicate of img1
    img3x3.write_bytes(TINY_PNG + b"grid-b3")
    img_hc.write_bytes(TINY_PNG + b"grid-hc1")

    rc_trials = [
        {
            "trial_id": "trial-A",
            "rounds": [
                {"prompt": "Select all squares with\nbicycles\nIf there are none, click skip", "layout": 4, "image": str(img1)},
                {"prompt": "Select all squares with\nbuses", "layout": 4, "image": str(img2)},
            ],
        },
        {
            "trial_id": "trial-B",
            "rounds": [
                {"prompt": "Select all squares with\nbicycles\nIf there are none, click skip", "layout": 4, "image": str(img_dup)},
                {"prompt": "Select all images with\ncrosswalks", "layout": 3, "image": str(img3x3)},
            ],
        },
    ]
    (run_rc / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in rc_trials) + "\n", encoding="utf-8")

    hc_trials = [
        {
            "trial_id": "hc-1",
            "captcha": "hcaptcha",
            "rounds": [{"prompt": "Please click each image containing a tiger", "page": 1, "pages": 1}],
        }
    ]
    (run_hc / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in hc_trials) + "\n", encoding="utf-8")

    out_all = tmp_path / "manifests" / "all.jsonl"
    stats = build_dataset([run_rc, run_hc], out_all)
    assert stats["written"] == 4
    assert stats["duplicates_skipped"] == 1
    assert stats["groups"] == 3

    rows = [json.loads(line) for line in out_all.read_text(encoding="utf-8").splitlines()]
    # Both rounds from trial-A share group_id="trial-A" so CV keeps them together.
    a_rows = [r for r in rows if r["group_id"] == "trial-A"]
    assert len(a_rows) == 2
    assert any(r["allows_skipping"] for r in a_rows)
    assert any(not r["allows_skipping"] for r in a_rows)
    # Relative paths resolve back to real files from the manifest's directory.
    for r in rows:
        assert not Path(r["image"]).is_absolute()
        assert (out_all.parent / r["image"]).is_file()

    # Filter to 4x4 only via CLI.
    out_4x4 = tmp_path / "manifests" / "only4.jsonl"
    rc = cli_main(["dataset", str(run_rc), str(run_hc), "--layout", "4", "-o", str(out_4x4)])
    assert rc == 0
    rows4 = [json.loads(line) for line in out_4x4.read_text(encoding="utf-8").splitlines()]
    assert len(rows4) == 2
    assert all(r["layout"] == 4 for r in rows4)


# ---------------------------------------------------------------------------
# 3. Localhost label server HTTP API round-trip, persistence & repeatability
# ---------------------------------------------------------------------------


def test_label_server_http_roundtrip_and_repeatability(tmp_path: Path) -> None:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    p1 = img_dir / "g1.png"
    p2 = img_dir / "g2.png"
    p1.write_bytes(TINY_PNG)
    p2.write_bytes(TINY_PNG)

    ds_path = tmp_path / "dataset.jsonl"
    items = [
        {"id": "g1", "group_id": "t1", "service": "recaptcha", "layout": 4, "prompt": "Select all squares with\nbuses\nIf there are none, click skip", "image": "imgs/g1.png", "allows_skipping": True},
        {"id": "g2", "group_id": "t2", "service": "recaptcha", "layout": 4, "prompt": "Select all squares with\ncars", "image": "imgs/g2.png", "allows_skipping": False},
    ]
    ds_path.write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")
    labels_path = tmp_path / "labels.jsonl"

    server = create_label_server(ds_path, labels_path, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(trust_env=False) as client:
            # HTML UI must have zero claude.ai dependencies.
            html = client.get(f"{base}/")
            assert html.status_code == 200
            assert "window.claude" not in html.text
            assert "/api/labels" in html.text

            got_items = client.get(f"{base}/api/items").json()
            assert [x["id"] for x in got_items] == ["g1", "g2"]
            assert got_items[0]["target"] == "buses"

            img_resp = client.get(f"{base}/api/image/g1")
            assert img_resp.status_code == 200
            assert img_resp.content == TINY_PNG

            assert client.get(f"{base}/api/labels").json() == {}

            # Reject overlapping target & sliver.
            bad = client.post(f"{base}/api/labels", json={"id": "g1", "target": [2], "sliver": [2], "unclear": False})
            assert bad.status_code == 400

            # Save g1, then update g1, then save g2.
            r1 = client.post(f"{base}/api/labels", json={"id": "g1", "target": [5, 6], "sliver": [1], "unclear": False})
            assert r1.status_code == 200
            r1_upd = client.post(f"{base}/api/labels", json={"id": "g1", "target": [6], "sliver": [5], "unclear": False})
            assert r1_upd.status_code == 200
            r2 = client.post(f"{base}/api/labels", json={"id": "g2", "target": [10, 11], "sliver": [], "unclear": False})
            assert r2.status_code == 200

            saved = client.get(f"{base}/api/labels").json()
            assert saved["g1"]["target"] == [6]
            assert saved["g1"]["sliver"] == [5]
            assert saved["g2"]["target"] == [10, 11]
    finally:
        server.shutdown()
        server.server_close()

    # Reload persistence into a brand-new LabelStore instance.
    reloaded = LabelStore(ds_path, labels_path)
    assert reloaded.get_labels()["g1"]["target"] == [6]
    assert reloaded.get_labels()["g1"]["sliver"] == [5]
    assert reloaded.get_labels()["g2"]["target"] == [10, 11]

    # Inter-rater repeatability check between labels_a (labels_path) and labels_b.
    labels_b = tmp_path / "labels_b.jsonl"
    b_rows = [
        # On g1: A picked [6] (sliver 5), B picked [6, 5] -> B right under A (5 is A's sliver), A NOT right under B (missed B's target 5).
        {"id": "g1", "layout": 4, "target": [5, 6], "sliver": [], "unclear": False},
        # On g2: exact agreement.
        {"id": "g2", "layout": 4, "target": [10, 11], "sliver": [], "unclear": False},
    ]
    labels_b.write_text("\n".join(json.dumps(x) for x in b_rows) + "\n", encoding="utf-8")

    rep = compare_labels(labels_path, labels_b)
    assert rep["common_grids"] == 2
    assert rep["a_right_under_b"]["count"] == 1
    assert rep["b_right_under_a"]["count"] == 2
    assert rep["mutual_agreement"]["count"] == 1
    assert rep["square_confusion"]["sliver"]["target"] == 1
    assert rep["square_confusion"]["target"]["target"] == 3
    assert rep["total_squares"] == 32
    assert rep["cohen_kappa_target"] is not None and rep["cohen_kappa_target"] > 0.7


# ---------------------------------------------------------------------------
# 4. Recording client: retry, grid_png stripping, multi-pass, and resume
# ---------------------------------------------------------------------------


def test_record_retries_strips_png_and_resumes(tmp_path: Path) -> None:
    img = tmp_path / "g1.png"
    img.write_bytes(TINY_PNG)
    ds_path = tmp_path / "dataset.jsonl"
    ds_path.write_text(
        json.dumps({"id": "g1", "group_id": "t1", "service": "recaptcha", "layout": 4, "prompt": "Select all squares with\nbuses", "image": "g1.png"}) + "\n",
        encoding="utf-8",
    )

    calls = {"count": 0}

    class MockLab(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            assert self.path == "/model"
            assert self.headers.get("Origin") == f"http://{self.headers.get('Host')}"
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            calls["count"] += 1
            if calls["count"] == 1:
                body = json.dumps({"result": {"status": "api_error", "error": "transient 503"}}).encode()
            else:
                probs = {str(i): 0.5 for i in range(1, 17)}
                body = json.dumps(
                    {
                        "grid_png": "SHOULD_BE_STRIPPED",
                        "result": {
                            "status": "answered",
                            "grid_png": "ALSO_STRIPPED",
                            "prediction": {"type": "tiles", "value": [1]},
                            "vllm_components": {
                                "whole_grid": {"tile_probabilities": probs},
                                "variant": {"tile_probabilities": probs, "passes": [{"grid_pixels": [389, 389], "tile_probabilities": probs}]},
                            },
                        },
                    }
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockLab)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        out_path = tmp_path / "recordings.jsonl"
        stats = record_dataset(
            ds_path,
            out_path,
            passes=2,
            workers=1,
            retries=3,
            backoff_seconds=0.01,
            server_url=f"http://127.0.0.1:{port}",
        )
        assert stats["recorded"] == 2
        assert stats["failed"] == []
        assert calls["count"] == 3  # 1 transient failure + 2 succeeded passes

        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert {r["pass_index"] for r in rows} == {0, 1}
        assert all("grid_png" not in r["result"] for r in rows)

        # Resume should skip both completed passes without hitting the server.
        before = calls["count"]
        stats2 = record_dataset(
            ds_path,
            out_path,
            passes=2,
            workers=1,
            server_url=f"http://127.0.0.1:{port}",
        )
        assert stats2["skipped_existing"] == 2
        assert stats2["recorded"] == 0
        assert calls["count"] == before

        # Non-loopback URL must be rejected.
        with pytest.raises(ValueError, match="loopback"):
            record_dataset(ds_path, tmp_path / "bad.jsonl", server_url="https://example.com")
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# 5. End-to-end synthetic pipeline: fit (3x3 & 4x4) -> sweep -> export
# ---------------------------------------------------------------------------


def _tile_box_rect(tile: int) -> list[float]:
    row, col = divmod(tile - 1, 4)
    return [col / 4.0 + 0.02, row / 4.0 + 0.02, (col + 1) / 4.0 - 0.02, (row + 1) / 4.0 - 0.02]


def _build_synthetic_pipeline_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(42)
    ds_rows: list[dict[str, Any]] = []
    lbl_rows: list[dict[str, Any]] = []
    rec_rows: list[dict[str, Any]] = []

    # 15 synthetic 3x3 grids across 8 groups
    for i in range(15):
        gid = f"s3-{i:02d}"
        ds_rows.append(
            {
                "id": gid,
                "group_id": f"g3-{i % 8}",
                "service": "recaptcha",
                "layout": 3,
                "prompt": "Select all images with\nbicycles",
                "image": f"{gid}.png",
                "allows_skipping": False,
            }
        )
        targets = sorted(int(x) for x in rng.choice(range(1, 10), size=3, replace=False))
        sliver = [t for t in range(1, 10) if t not in targets][:1]
        lbl_rows.append({"id": gid, "layout": 3, "target": targets, "sliver": sliver, "unclear": False})
        w = {str(t): float(np.clip((0.85 if t in targets else 0.12) + rng.normal(0, 0.08), 0.01, 0.99)) for t in range(1, 10)}
        c = {str(t): float(np.clip((0.90 if t in targets else 0.08) + rng.normal(0, 0.07), 0.01, 0.99)) for t in range(1, 10)}
        rec_rows.append(
            {
                "grid_id": gid,
                "pass_index": 0,
                "layout": 3,
                "wall_ms": 120,
                "result": {
                    "status": "answered",
                    "resolved_model": "synthetic-v1",
                    "usage": {"input_tokens": 1200, "output_tokens": 10},
                    "vllm_components": {
                        "whole_grid": {"tile_probabilities": w, "usage": {"input_tokens": 300}},
                        "variant": {"tile_probabilities": c, "usage": {"input_tokens": 900}},
                    },
                },
            }
        )

    # 24 synthetic 4x4 grids across 10 groups, 2 passes each, with skip + whole/every/window boxes
    for i in range(24):
        gid = f"s4-{i:02d}"
        is_empty_grid = i < 4  # 4 skip grids with no target
        ds_rows.append(
            {
                "id": gid,
                "group_id": f"g4-{i % 10}",
                "service": "recaptcha",
                "layout": 4,
                "prompt": "Select all squares with\nbuses\nIf there are none, click skip" if i < 12 else "Select all squares with\nbuses",
                "image": f"{gid}.png",
                "allows_skipping": i < 12,
            }
        )
        if is_empty_grid:
            targets, slivers = [], []
        else:
            # Contiguous 2-tile target + 1 adjacent sliver + 1 false-positive trap tile
            base = [6, 7] if i % 2 == 0 else [10, 11]
            targets = base
            slivers = [base[0] - 4]
        lbl_rows.append({"id": gid, "layout": 4, "target": targets, "sliver": slivers, "unclear": False})

        for p_idx in range(2):
            w_probs: dict[str, float] = {}
            c1_probs: dict[str, float] = {}
            c2_probs: dict[str, float] = {}
            for t in range(1, 17):
                if t in targets:
                    w_probs[str(t)] = float(np.clip(0.84 + rng.normal(0, 0.06), 0.02, 0.99))
                    c1_probs[str(t)] = float(np.clip(0.88 + rng.normal(0, 0.06), 0.02, 0.99))
                    c2_probs[str(t)] = float(np.clip(0.92 + rng.normal(0, 0.04), 0.02, 0.99))
                elif is_empty_grid and t == 6:
                    # Look-alike false positive caught by skip presence question!
                    w_probs[str(t)] = 0.72
                    c1_probs[str(t)] = 0.80
                    c2_probs[str(t)] = 0.82
                elif (not is_empty_grid) and t == 16 and i % 3 == 0:
                    # Borderline false positive on some grids: high enough to pass cutoff,
                    # below exempt_from, and NOT covered by true boxes -> box rule removes it!
                    w_probs[str(t)] = 0.55
                    c1_probs[str(t)] = 0.72
                    c2_probs[str(t)] = 0.70
                else:
                    w_probs[str(t)] = float(np.clip(0.08 + rng.normal(0, 0.05), 0.005, 0.4))
                    c1_probs[str(t)] = float(np.clip(0.06 + rng.normal(0, 0.05), 0.005, 0.4))
                    c2_probs[str(t)] = float(np.clip(0.05 + rng.normal(0, 0.04), 0.005, 0.4))

            presence = (0.18 if is_empty_grid else 0.94) if i < 12 else None
            # Boxes cover true targets (not false-positive tile 16).
            wb = [{"name": f"b{k}", "rect": _tile_box_rect(t), "confidence": 0.85} for k, t in enumerate(targets)]
            eb = [{"name": f"e{k}", "rect": _tile_box_rect(t), "confidence": 0.88} for k, t in enumerate(targets)]
            sw = [{"name": f"s{t:02d}", "rect": _tile_box_rect(t), "confidence": 0.90} for t in targets]

            rec_rows.append(
                {
                    "grid_id": gid,
                    "pass_index": p_idx,
                    "layout": 4,
                    "wall_ms": 450,
                    "result": {
                        "status": "answered",
                        "resolved_model": "synthetic-v1",
                        "usage": {"input_tokens": 9000, "output_tokens": 20},
                        "vllm_components": {
                            "whole_grid": {"tile_probabilities": w_probs, "usage": {"input_tokens": 600}},
                            "variant": {
                                "tile_probabilities": c2_probs,
                                "usage": {"input_tokens": 4400},
                                "passes": [
                                    {"grid_pixels": [389, 389], "tile_probabilities": c1_probs},
                                    {"grid_pixels": [506, 506], "tile_probabilities": c2_probs},
                                ],
                            },
                            "presence": (
                                {"probability": presence, "usage": {"input_tokens": 500}}
                                if presence is not None
                                else {}
                            ),
                            "boxes": {
                                "whole_grid": {"boxes": wb, "usage": {"input_tokens": 1500}},
                                "every": {"boxes": eb, "usage": {"input_tokens": 1100}},
                                "windows": {"boxes": sw, "usage": {"input_tokens": 3500}, "meta": {"requests": 1}},
                            },
                        },
                    },
                }
            )

    ds_path = tmp_path / "dataset.jsonl"
    lbl_path = tmp_path / "labels.jsonl"
    rec_path = tmp_path / "recordings.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in ds_rows) + "\n", encoding="utf-8")
    lbl_path.write_text("\n".join(json.dumps(r) for r in lbl_rows) + "\n", encoding="utf-8")
    rec_path.write_text("\n".join(json.dumps(r) for r in rec_rows) + "\n", encoding="utf-8")
    return ds_path, lbl_path, rec_path


def test_end_to_end_fit_sweep_export_and_fixture(tmp_path: Path) -> None:
    ds_path, lbl_path, rec_path = _build_synthetic_pipeline_files(tmp_path)

    # 1. Fit 3x3 and 4x4 weights via grouped K-fold CV.
    fit_path = tmp_path / "fit.json"
    assert cli_main(["fit", "--dataset", str(ds_path), "--labels", str(lbl_path), "--recordings", str(rec_path), "-o", str(fit_path)]) == 0
    fit_res = json.loads(fit_path.read_text(encoding="utf-8"))
    assert len(fit_res["layout_3x3"]["weights"]) == 3
    assert fit_res["layout_3x3"]["oof_at_50"]["rate"] >= 0.8
    assert set(fit_res["layout_4x4"]["weights"].keys()) == {"capture", "506", "avg"}
    assert all(len(v) == 5 for v in fit_res["layout_4x4"]["weights"].values())

    # 2. Sweep 4x4 effort tiers across 2 passes with whole + every + window boxes.
    sweep_path = tmp_path / "sweep_result.json"
    md_path = tmp_path / "sweep.md"
    assert (
        cli_main(
            [
                "sweep",
                "--dataset",
                str(ds_path),
                "--labels",
                str(lbl_path),
                "--recordings",
                str(rec_path),
                "--fit",
                str(fit_path),
                "--fast",
                "--boot",
                "500",
                "-o",
                str(sweep_path),
                "--markdown",
                str(md_path),
            ]
        )
        == 0
    )
    sw = json.loads(sweep_path.read_text(encoding="utf-8"))
    assert len(sw["chosen_weights"]) == 5
    assert set(sw["efforts"].keys()) == set(EFFORT_NAMES)
    assert sw["dataset_stats"]["passes"] == 2
    assert sw["efforts"]["medium"]["metrics"]["in_sample_rate"] >= sw["efforts"]["low"]["metrics"]["in_sample_rate"]
    assert sw["efforts"]["medium"]["metrics"]["in_sample_rate"] >= 0.9
    assert sw["efforts"]["low"]["metrics"]["cross_pass_right"] is not None
    assert sw["efforts"]["low"]["metrics"]["pass_flip_grids"] is not None
    assert "medium_minus_low" in sw["comparisons"]
    assert "low_minus_baseline" in sw["comparisons"]
    assert md_path.read_text(encoding="utf-8").startswith("## 4x4 Effort-Level Sweep")

    # 3. Export custom provider TOML + deterministic .json.gz parity fixture.
    toml_out = tmp_path / "my_model.toml"
    fixture_out = tmp_path / "my_model_4x4.json.gz"
    assert (
        cli_main(
            [
                "export",
                "--sweep",
                str(sweep_path),
                "--fit",
                str(fit_path),
                "--service",
                "recaptcha",
                "--name",
                "MyProvider/my-model",
                "--provider",
                "custom",
                "--model-id",
                "my-model-v1",
                "--dialect",
                "systemone-assets",
                "--base-url",
                "https://example.com/v1",
                "--api-key-env",
                "MY_MODEL_API_KEY",
                "--image-format",
                "jpeg",
                "--crop-filter",
                "bilinear",
                "--no-gzip",
                "--max-images",
                "16",
                "--multi-state-windows",
                "--supports-every-box",
                "--no-return-meta",
                "--no-hedge-on-stall",
                "--no-warm-connections",
                "--no-min-patch-tokens",
                "--extra-header",
                "X-Test-Header=enabled",
                "--recordings",
                str(rec_path),
                "--dataset",
                str(ds_path),
                "--fixture",
                str(fixture_out),
                "-o",
                str(toml_out),
            ]
        )
        == 0
    )

    # Parse back with stdlib tomllib and verify exact schema contract!
    raw_toml = tomllib.loads(toml_out.read_text(encoding="utf-8"))
    assert raw_toml["schema_version"] == 1
    assert raw_toml["service"] == "recaptcha"
    assert raw_toml["name"] == "MyProvider/my-model"
    assert raw_toml["provider"] == "custom"
    assert raw_toml["model_id"] == "my-model-v1"
    assert raw_toml["endpoint"]["dialect"] == "systemone-assets"
    assert raw_toml["endpoint"]["base_url"] == "https://example.com/v1"
    assert raw_toml["endpoint"]["api_key_env"] == "MY_MODEL_API_KEY"
    assert raw_toml["endpoint"]["supports_every_box"] is True
    assert raw_toml["endpoint"]["extra_headers"] == {"X-Test-Header": "enabled"}
    assert raw_toml["tuning"]["default_decision"] == "logistic"
    assert len(raw_toml["tuning"]["logistic_3x3"]["weights"]) == 3
    assert len(raw_toml["tuning"]["logistic_4x4"]["weights"]) == 5
    for tier in EFFORT_NAMES:
        assert tier in raw_toml["tuning"]["efforts"]

    cfg = load_model_config(toml_out)
    assert cfg.tuning is not None and cfg.tuning.efforts is not None and cfg.tuning.logistic_4x4 is not None

    # Verify deterministic fixture (gzip mtime=0, image-free, label-free, 100% pick parity).
    raw_gz_bytes = fixture_out.read_bytes()
    assert raw_gz_bytes[4:8] == b"\x00\x00\x00\x00"  # mtime == 0
    with gzip.open(fixture_out, "rt", encoding="utf-8") as handle:
        fx = json.load(handle)
    assert len(fx) == 24
    expected_keys = {"id", "presence", "whole", "captured", "at_506", "whole_boxes", "every_boxes", "window_boxes", "picks"}
    for row in fx:
        assert set(row.keys()) == expected_keys
        for tier in EFFORT_NAMES:
            got = decide_4x4(
                row["whole"],
                {"capture": row["captured"], 506: row["at_506"]},
                cfg.tuning.efforts.plan(tier),
                weights=cfg.tuning.logistic_4x4,
                presence=row["presence"],
                whole_boxes=row["whole_boxes"],
                every_boxes=row["every_boxes"],
                window_boxes=row["window_boxes"],
            )
            assert got == row["picks"][tier]

    # Validation check: custom provider without [endpoint] must fail.
    with pytest.raises(ValueError, match="endpoint"):
        export_model(
            service="recaptcha",
            name="Bad/custom",
            provider="custom",
            model_id="bad-1",
            output_toml=tmp_path / "bad.toml",
            sweep_path=sweep_path,
            endpoint=None,
        )

    # Validation check: vllm provider must reject `boxes.every`.
    bad_every = ModelConfig(
        schema_version=1,
        service="recaptcha",
        name="vLLM/bad",
        provider="vllm",
        model_id="bad",
        tuning=TuningConfig(
            default_decision="logistic",
            logistic_4x4=(0.2, 1.0, 0.2, 0.2, -1.5),
            efforts=Efforts(
                low=Plan(close_ups=(506,), select_from=0.5),
                medium=Plan(close_ups=(506,), select_from=0.5, boxes=BoxRule(exempt_from=0.9, every=0.3)),
                high=Plan(close_ups=(506,), select_from=0.5),
                max=Plan(close_ups=(506,), select_from=0.5),
            ),
        ),
    )
    with pytest.raises(ValueError, match="supports_every_box"):
        render_model_toml(bad_every)

    # Validation check: BoxRule with no whole/every/windows must be rejected (matches Rust).
    empty_boxes = ModelConfig(
        schema_version=1,
        service="recaptcha",
        name="vLLM/empty-box",
        provider="vllm",
        model_id="bad",
        tuning=TuningConfig(
            default_decision="logistic",
            logistic_4x4=(0.2, 1.0, 0.2, 0.2, -1.5),
            efforts=Efforts(
                low=Plan(close_ups=(506,), select_from=0.5),
                medium=Plan(close_ups=(506,), select_from=0.5, boxes=BoxRule(exempt_from=0.9)),
                high=Plan(close_ups=(506,), select_from=0.5),
                max=Plan(close_ups=(506,), select_from=0.5),
            ),
        ),
    )
    with pytest.raises(ValueError, match="at least one of whole, every, or windows"):
        render_model_toml(empty_boxes)

    # Schema parity: omitted `tuning.default_decision` in TOML defaults to "average" (matches Rust).
    minimal_toml = tmp_path / "minimal_average.toml"
    minimal_toml.write_text(
        'schema_version = 1\nservice = "hcaptcha"\nname = "vLLM/test"\nprovider = "vllm"\nmodel_id = "test"\n\n[tuning]\n',
        encoding="utf-8",
    )
    loaded_min = load_model_config(minimal_toml)
    assert loaded_min.tuning is not None
    assert loaded_min.tuning.default_decision == "average"


def test_hcaptcha_dataset_and_record_preserve_pictures(tmp_path: Path) -> None:
    """Verify hCaptcha `dataset` -> `record` sends valid 9-picture rects for both grid-* and task-* PNGs."""
    run_dir = tmp_path / "hc_run"
    (run_dir / "t-grid").mkdir(parents=True)
    (run_dir / "t-task").mkdir(parents=True)

    (run_dir / "t-grid" / "grid-01.png").write_bytes(TINY_PNG + b"hc-grid-01")
    (run_dir / "t-task" / "task-01.png").write_bytes(TINY_PNG + b"hc-task-01")

    custom_pictures = [[0.1 + 0.25 * c, 0.05 + 0.3 * r, 0.2, 0.25] for r in range(3) for c in range(3)]
    trials = [
        {
            "trial_id": "t-grid",
            "captcha": "hcaptcha",
            "rounds": [{"prompt": "Pick the flying animal", "page": 1, "pages": 1, "pictures": custom_pictures}],
        },
        {
            "trial_id": "t-task",
            "captcha": "hcaptcha",
            "rounds": [{"prompt": "Pick the tiger", "page": 1, "pages": 1, "pictures": custom_pictures}],
        },
    ]
    (run_dir / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials) + "\n", encoding="utf-8")

    ds_path = tmp_path / "hc_dataset.jsonl"
    stats = build_dataset([run_dir], ds_path)
    assert stats["written"] == 2
    rows = [json.loads(line) for line in ds_path.read_text(encoding="utf-8").splitlines()]
    by_group = {r["group_id"]: r for r in rows}
    assert by_group["t-grid"]["pictures"] == UNIFORM_3X3_PICTURES
    assert by_group["t-task"]["pictures"] == custom_pictures

    # Also add a pre-composed grid entry with `pictures` omitted to test record's fallback.
    fallback_img = tmp_path / "grid-99.png"
    fallback_img.write_bytes(TINY_PNG + b"hc-grid-99")
    with ds_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "hc-fallback",
                    "group_id": "t-fb",
                    "service": "hcaptcha",
                    "layout": 3,
                    "prompt": "Pick the boat",
                    "image": "grid-99.png",
                }
            )
            + "\n"
        )

    received: list[dict[str, Any]] = []

    class MockHcaptchaLab(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            req_body = json.loads(self.rfile.read(length).decode("utf-8"))
            received.append(req_body)
            probs = {str(i): 0.4 for i in range(1, 10)}
            body = json.dumps(
                {
                    "result": {
                        "status": "answered",
                        "prediction": {"type": "tiles", "value": []},
                        "vllm_components": {
                            "whole_grid": {"tile_probabilities": probs},
                            "variant": {"tile_probabilities": probs},
                        },
                    }
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHcaptchaLab)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rec_out = tmp_path / "hc_rec.jsonl"
        res = record_dataset(
            ds_path,
            rec_out,
            passes=1,
            workers=1,
            server_url=f"http://hypercaptcha.localhost:{port}",
        )
        assert res["recorded"] == 3
        assert res["failed"] == []
    finally:
        server.shutdown()
        server.server_close()

    by_prompt = {r["prompt"]: r for r in received}
    assert "layout" not in by_prompt["Pick the flying animal"]
    assert by_prompt["Pick the flying animal"]["pictures"] == UNIFORM_3X3_PICTURES
    assert by_prompt["Pick the tiger"]["pictures"] == custom_pictures
    assert by_prompt["Pick the boat"]["pictures"] == UNIFORM_3X3_PICTURES


def test_record_spawn_error_reports_numeric_exit_code(tmp_path: Path) -> None:
    """Verify silent early server exit reports numeric exit code instead of `code None`."""
    img = tmp_path / "g1.png"
    img.write_bytes(TINY_PNG)
    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(
        json.dumps({"id": "g1", "group_id": "t1", "service": "recaptcha", "layout": 4, "prompt": "Select all squares with\nbuses", "image": "g1.png"}) + "\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "fake_hypercaptcha.sh"
    fake_bin.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    fake_bin.chmod(0o755)

    with pytest.raises(RuntimeError, match="server exited with code 42"):
        record_dataset(ds_path, tmp_path / "out.jsonl", binary=str(fake_bin))


def test_sweep_handles_whole_only_and_no_box_recordings(tmp_path: Path) -> None:
    """Verify sweep handles recordings without `every` boxes and recordings with zero boxes."""
    ds_path, lbl_path, rec_path = _build_synthetic_pipeline_files(tmp_path)
    orig_rows = [json.loads(line) for line in rec_path.read_text(encoding="utf-8").splitlines()]

    # Variant A: vLLM-style recordings (whole_grid + windows only, no `every`).
    vllm_rec = tmp_path / "rec_vllm.jsonl"
    vllm_rows = []
    for row in orig_rows:
        r = json.loads(json.dumps(row))
        boxes = (r["result"].get("vllm_components") or {}).get("boxes")
        if boxes:
            boxes.pop("every", None)
        vllm_rows.append(r)
    vllm_rec.write_text("\n".join(json.dumps(r) for r in vllm_rows) + "\n", encoding="utf-8")
    sw_vllm = sweep_4x4(ds_path, lbl_path, vllm_rec, fast=True, n_boot=200)
    assert sw_vllm["model_meta"]["recorded_boxes"] == {"whole_grid": True, "every": False, "windows": True}
    for tier in EFFORT_NAMES:
        b = sw_vllm["efforts"][tier]["plan"].get("boxes")
        assert b is None or b.get("every") is None

    # Variant B: recordings with no boxes and single pass (`capture` only).
    nobox_rec = tmp_path / "rec_nobox.jsonl"
    nobox_rows = []
    for row in orig_rows:
        if row["layout"] != 4 or row["pass_index"] != 0:
            continue
        r = json.loads(json.dumps(row))
        comp = r["result"]["vllm_components"]
        comp.pop("boxes", None)
        comp["variant"]["passes"] = comp["variant"]["passes"][:1]
        nobox_rows.append(r)
    nobox_rec.write_text("\n".join(json.dumps(r) for r in nobox_rows) + "\n", encoding="utf-8")
    sw_nobox = sweep_4x4(ds_path, lbl_path, nobox_rec, fast=True, n_boot=200)
    assert sw_nobox["dataset_stats"]["passes"] == 1
    assert sw_nobox["efforts"]["low"]["metrics"]["cross_pass_right"] is None
    assert sw_nobox["efforts"]["max"]["plan"].get("boxes") is None


def test_summarize_hcaptcha_and_condition_grouping(tmp_path: Path) -> None:
    trials = [
        {
            "trial_id": "hc-1",
            "captcha": "hcaptcha",
            "condition": "d3_hc",
            "outcome": "model_solved",
            "checkbox_to_verification_ms": 4200,
            "hcaptcha": {"result": {"valid": True}},
            "passed_over": [{"prompt": "Drag the puzzle piece", "settled": True, "ms": 900}],
            "rounds": [
                {
                    "prompt": "Pick the flying animal",
                    "page": 1,
                    "pages": 2,
                    "model_ms": 620,
                    "model": {"result": {"status": "answered", "usage": {"input_tokens": 1800, "output_tokens": 9}}},
                },
                {
                    "prompt": "Pick the flying animal",
                    "page": 2,
                    "pages": 2,
                    "model_ms": 610,
                    "model": {"result": {"status": "answered", "usage": {"input_tokens": 1800, "output_tokens": 9}}},
                },
            ],
        },
        {
            "trial_id": "hc-2",
            "captcha": "hcaptcha",
            "condition": "d3_hc",
            "outcome": "model_failed",
            "rejections": 1,
            "passed_over": [],
            "rounds": [
                {
                    "prompt": "Pick the tiger",
                    "page": 1,
                    "pages": 1,
                    "result": "rejected",
                    "model_ms": 590,
                    "model": {"result": {"status": "answered", "usage": {"input_tokens": 1750, "output_tokens": 9}}},
                }
            ],
        },
    ]
    p = tmp_path / "hc_trials.jsonl"
    p.write_text("\n".join(json.dumps(t) for t in trials) + "\n", encoding="utf-8")

    rep = summarize_files([p], group_by_condition=True)
    assert "d3_hc" in rep["by_condition"]
    ov = rep["overall"]
    assert ov["passed_over_tasks"] == 1
    assert ov["verified_challenged"]["passed"] == 1
    assert ov["verified_challenged"]["total"] == 2
    assert set(ov["by_challenge_type"].keys()) == {"3x3_hcaptcha_single", "3x3_hcaptcha_multi"}
    assert ov["tokens"]["total_input_tokens"] == 1800 + 1800 + 1750
    md = format_summary_markdown(rep)
    assert "Condition: d3_hc" in md
    assert "Non-grid tasks passed over" in md


# ---------------------------------------------------------------------------
# 6. Live trials summarizer (Wilson CI, challenge breakdown, timelines, tokens)
# ---------------------------------------------------------------------------


def test_summarize_trials_wilson_ci_timelines_and_tokens(tmp_path: Path) -> None:
    trials = [
        # 1. Checkbox-only pass
        {
            "trial_id": "t1",
            "condition": "d3",
            "outcome": "checkbox_only",
            "checkbox_to_verification_ms": 800,
            "google": {"result": {"valid": True}},
            "rounds": [],
        },
        # 2. Static 3x3 solved in 1 round
        {
            "trial_id": "t2",
            "condition": "d3",
            "outcome": "model_solved",
            "checkbox_to_verification_ms": 2400,
            "google": {"result": {"valid": True}},
            "rounds": [
                {
                    "layout": 3,
                    "prompt": "Select all images with\nbicycles",
                    "timeline_ms": {"grid_loaded": 500, "captured": 600, "model_answered": 1300, "tiles_clicked": 1600, "verify_clicked": 1800},
                    "model": {"latency_ms": 680, "result": {"status": "answered", "usage": {"input_tokens": 2200, "output_tokens": 12}}},
                }
            ],
        },
        # 3. Dynamic 3x3 failed after 1 rejection
        {
            "trial_id": "t3",
            "condition": "d3",
            "outcome": "model_failed",
            "rounds": [
                {
                    "layout": 3,
                    "prompt": "Select all images with\ncars\nClick verify once there are none left",
                    "rejection": "Please try again.",
                    "timeline_ms": {"grid_loaded": 600, "captured": 700, "model_answered": 1500, "tiles_clicked": 1900, "verify_clicked": 2500},
                    "model": {"latency_ms": 790, "result": {"status": "answered", "usage": {"input_tokens": 2400, "output_tokens": 14}}},
                }
            ],
        },
        # 4. 4x4 non-skip solved
        {
            "trial_id": "t4",
            "condition": "d3",
            "outcome": "model_solved",
            "checkbox_to_verification_ms": 3600,
            "google": {"result": {"valid": True}},
            "rounds": [
                {
                    "layout": 4,
                    "prompt": "Select all squares with\ntraffic lights",
                    "timeline_ms": {"grid_loaded": 700, "captured": 850, "model_answered": 2200, "tiles_clicked": 2700, "verify_clicked": 3000},
                    "model": {"latency_ms": 1320, "result": {"status": "answered", "usage": {"input_tokens": 12000, "output_tokens": 30}}},
                }
            ],
        },
        # 5. 4x4 skip solved after 1 rejection
        {
            "trial_id": "t5",
            "condition": "d3",
            "outcome": "model_solved",
            "checkbox_to_verification_ms": 5800,
            "google": {"result": {"valid": True}},
            "rounds": [
                {
                    "layout": 4,
                    "prompt": "Select all squares with\nbuses\nIf there are none, click skip",
                    "judged": True,
                    "timeline_ms": {"grid_loaded": 550, "captured": 650, "model_answered": 2000, "tiles_clicked": 2300, "verify_clicked": 2600},
                    "model": {"latency_ms": 1340, "result": {"status": "answered", "usage": {"input_tokens": 12500, "output_tokens": 28}}},
                },
                {
                    "layout": 4,
                    "prompt": "Select all squares with\ncrosswalks\nIf there are none, click skip",
                    "timeline_ms": {"grid_loaded": 3100, "captured": 3200, "model_answered": 4500, "tiles_clicked": 4900, "verify_clicked": 5200},
                    "model": {"latency_ms": 1290, "result": {"status": "answered", "usage": {"input_tokens": 12500, "output_tokens": 28}}},
                },
            ],
        },
    ]

    trials_path = tmp_path / "trials.jsonl"
    trials_path.write_text("\n".join(json.dumps(t) for t in trials) + "\n", encoding="utf-8")

    # Direct Wilson CI sanity check: 4/5 = 0.80, CI strictly inside (0.3, 0.99).
    p = wilson_proportion(4, 5)
    assert p["rate"] == 0.8
    assert 0.35 < p["wilson95"][0] < 0.40
    assert 0.95 < p["wilson95"][1] < 0.99

    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    assert cli_main(["summarize", str(trials_path), "--json", str(out_json), "--markdown", str(out_md)]) == 0

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    ov = rep["overall"]
    assert ov["attempts"] == 5
    assert ov["verified_overall"]["passed"] == 4
    assert ov["verified_challenged"]["passed"] == 3
    assert ov["verified_challenged"]["total"] == 4
    assert ov["checkbox_only"]["passed"] == 1
    assert ov["rejections"]["total"] == 2
    assert set(ov["by_challenge_type"].keys()) == {"3x3_static", "3x3_dynamic", "4x4_no_skip", "4x4_skip"}
    assert ov["timelines_ms"]["grid_loaded"]["count"] == 5
    assert ov["timelines_ms"]["checkbox_to_verification_ms"]["median"] == 3600.0
    assert ov["tokens"]["total_input_tokens"] == 2200 + 2400 + 12000 + 12500 + 12500
    assert "## Live Trials Summary" in format_summary_markdown(rep)


def test_python_module_cli_help() -> None:
    """Verify `python -m eval --help` and every subcommand `--help` succeed cleanly."""
    subcommands = ("dataset", "label", "record", "fit", "sweep", "export", "summarize")
    proc = subprocess.run(
        [sys.executable, "-m", "eval", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for sub in subcommands:
        assert sub in proc.stdout
        sub_proc = subprocess.run(
            [sys.executable, "-m", "eval", sub, "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"hypercaptcha-eval {sub}" in sub_proc.stdout


# ---------------------------------------------------------------------------
# 7. Regression tests for audit fixes across eval/*
# ---------------------------------------------------------------------------


def test_summarize_regressions_condition_captured_null_layout_and_nested_dirs(tmp_path: Path) -> None:
    """Verify summarize fallback grouping, captured trials, null layout guard, and nested dirs."""
    run_a = tmp_path / "runs" / "model_a"
    run_b = tmp_path / "runs" / "model_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    # Real runner logs omit top-level condition/model_name and store rounds[i].model.model_name.
    trials_a = [
        {
            "trial_id": "a-1",
            "outcome": "model_solved",
            "google": {"result": {"valid": True}},
            "rounds": [
                {
                    "layout": None,
                    "prompt": "Select all images with\nbicycles",
                    "model": {
                        "model_name": "vLLM/d3-ultra",
                        "latency_ms": 420,
                        "result": {"status": "answered", "usage": {"input_tokens": 1100, "output_tokens": 8}},
                    },
                }
            ],
        },
        # Capture-only trial should NOT count as a failed challenged attempt.
        {
            "trial_id": "a-cap",
            "outcome": "captured",
            "rounds": [
                {
                    "layout": 4,
                    "prompt": "Select all squares with\nbuses",
                }
            ],
        },
    ]
    trials_b = [
        {
            "trial_id": "b-1",
            "captcha": "hcaptcha",
            "outcome": "model_failed",
            "rejections": 1,
            "rounds": [
                {
                    "prompt": "Pick the tiger",
                    "pages": None,
                    "model_ms": 510,
                    "model": {
                        "model_name": "Liquid/d1.2-ultra",
                        "result": {"status": "answered", "usage": {"input_tokens": 1500, "output_tokens": 6}},
                    },
                }
            ],
        },
        # Explicit top-level model_name takes precedence over nested round model_name.
        {
            "trial_id": "b-2",
            "model_name": "TopLevel/model",
            "outcome": "checkbox_only",
            "google": {"result": {"valid": True}},
            "rounds": [],
        },
    ]
    (run_a / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials_a) + "\n", encoding="utf-8")
    (run_b / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials_b) + "\n", encoding="utf-8")

    # Directory without top-level trials.jsonl discovers nested trials.jsonl files.
    rep = summarize_files([tmp_path / "runs"], group_by_condition=True)
    assert set(rep["by_condition"].keys()) == {"vLLM/d3-ultra", "Liquid/d1.2-ultra", "TopLevel/model", "default"}

    ov = rep["overall"]
    assert ov["attempts"] == 4
    assert ov["outcomes"]["captured"] == 1
    # Only a-1 (solved) and b-1 (failed) are challenged; a-cap (captured) and b-2 (checkbox_only) are excluded.
    assert ov["verified_challenged"]["passed"] == 1
    assert ov["verified_challenged"]["total"] == 2
    assert ov["first_stage_pass"]["passed"] == 1
    assert ov["first_stage_pass"]["total"] == 2
    assert set(ov["by_challenge_type"].keys()) == {"3x3_static", "3x3_hcaptcha_single"}
    assert ov["tokens"]["by_layout_input"]["3x3"]["count"] == 2

    # Condition vLLM/d3-ultra has 1/1 challenged solved; default (capture-only) has 0/0 challenged.
    assert rep["by_condition"]["vLLM/d3-ultra"]["verified_challenged"]["passed"] == 1
    assert rep["by_condition"]["vLLM/d3-ultra"]["verified_challenged"]["total"] == 1
    assert rep["by_condition"]["default"]["verified_challenged"]["total"] == 0

    # Empty directory with no trials.jsonl raises FileNotFoundError.
    empty_dir = tmp_path / "empty_runs"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no trials.jsonl"):
        summarize_files([empty_dir])

    # Direct summarize_trials with null layout and captured trial.
    direct = summarize_trials(trials_a)
    assert direct["verified_challenged"]["passed"] == 1
    assert direct["verified_challenged"]["total"] == 1


def test_dataset_hcaptcha_passed_over_explicit_and_legacy_fallback(tmp_path: Path) -> None:
    """Verify hCaptcha image alignment after passed_over tasks for explicit & legacy logs."""
    custom_pictures = [[0.1 + 0.25 * c, 0.05 + 0.3 * r, 0.2, 0.25] for r in range(3) for c in range(3)]
    run_dir = tmp_path / "hc_passed_over"

    # 1. Explicit per-round `image` and `layout` (new hcaptcha_lab.py format).
    (run_dir / "t-explicit").mkdir(parents=True)
    (run_dir / "t-explicit" / "task-01.png").write_bytes(TINY_PNG + b"passed-over-drag")
    (run_dir / "t-explicit" / "task-02.png").write_bytes(TINY_PNG + b"explicit-task-02")
    (run_dir / "t-explicit" / "grid-02.png").write_bytes(TINY_PNG + b"explicit-grid-02")
    (run_dir / "t-explicit" / "task-03.png").write_bytes(TINY_PNG + b"passed-over-point")
    (run_dir / "t-explicit" / "task-04.png").write_bytes(TINY_PNG + b"explicit-task-04")

    # 2. Legacy log WITHOUT per-round `image` and WITHOUT `grid-*.png`, WITH ms/shown_ms timestamps.
    (run_dir / "t-legacy-ts").mkdir(parents=True)
    (run_dir / "t-legacy-ts" / "task-01.png").write_bytes(TINY_PNG + b"legacy-passed-1")
    (run_dir / "t-legacy-ts" / "task-02.png").write_bytes(TINY_PNG + b"legacy-grid-round-0")
    (run_dir / "t-legacy-ts" / "task-03.png").write_bytes(TINY_PNG + b"legacy-passed-2")
    (run_dir / "t-legacy-ts" / "task-04.png").write_bytes(TINY_PNG + b"legacy-grid-round-1")

    # 3. Legacy log WITHOUT `image`, WITHOUT `grid-*.png`, and WITHOUT timestamps -> must NOT pair task-01.png!
    (run_dir / "t-legacy-ambig").mkdir(parents=True)
    (run_dir / "t-legacy-ambig" / "task-01.png").write_bytes(TINY_PNG + b"ambig-passed-1")
    (run_dir / "t-legacy-ambig" / "task-02.png").write_bytes(TINY_PNG + b"ambig-task-2")

    # 4. Legacy log with passed_over[].image set (no timestamps, no round.image, no grid-*.png).
    (run_dir / "t-passed-img").mkdir(parents=True)
    (run_dir / "t-passed-img" / "task-01.png").write_bytes(TINY_PNG + b"pimg-passed-1")
    (run_dir / "t-passed-img" / "task-02.png").write_bytes(TINY_PNG + b"pimg-round-0")

    trials = [
        {
            "trial_id": "t-explicit",
            "captcha": "hcaptcha",
            "passed_over": [
                {"prompt": "Drag puzzle", "settled": True, "ms": 800, "image": "task-01.png"},
                {"prompt": "Click point", "settled": True, "ms": 3000, "image": "task-03.png"},
            ],
            "rounds": [
                {
                    "prompt": "Pick the flying animal",
                    "layout": 3,
                    "image": "grid-02.png",
                    "page": 1,
                    "pages": 2,
                    "shown_ms": 1500,
                    "pictures": custom_pictures,
                },
                {
                    "prompt": "Pick the boat",
                    "layout": None,
                    "image": "task-04.png",
                    "page": 2,
                    "pages": 2,
                    "shown_ms": 4200,
                    "pictures": custom_pictures,
                },
            ],
        },
        {
            "trial_id": "t-legacy-ts",
            "captcha": "hcaptcha",
            "passed_over": [
                {"prompt": "Drag puzzle", "settled": True, "ms": 900},
                {"prompt": "Follow object", "settled": False, "ms": 3500},
            ],
            "rounds": [
                {
                    "prompt": "Pick the cat",
                    "page": 1,
                    "pages": 2,
                    "shown_ms": 2100,
                    "pictures": custom_pictures,
                },
                {
                    "prompt": "Pick the dog",
                    "page": 2,
                    "pages": 2,
                    "shown_ms": 5200,
                    "pictures": custom_pictures,
                },
            ],
        },
        {
            "trial_id": "t-legacy-ambig",
            "captcha": "hcaptcha",
            "passed_over": [{"prompt": "Drag puzzle"}],
            "rounds": [
                {
                    "prompt": "Pick the frog",
                    "page": 1,
                    "pages": 1,
                    "pictures": custom_pictures,
                }
            ],
        },
        {
            "trial_id": "t-passed-img",
            "captcha": "hcaptcha",
            "passed_over": [{"prompt": "Drag puzzle", "image": "task-01.png"}],
            "rounds": [
                {
                    "prompt": "Pick the owl",
                    "page": 1,
                    "pages": 1,
                    "pictures": custom_pictures,
                }
            ],
        },
    ]
    (run_dir / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials) + "\n", encoding="utf-8")

    ds_path = tmp_path / "hc_aligned.jsonl"
    stats = build_dataset([run_dir], ds_path)
    assert stats["written"] == 5
    assert stats["missing_images"] == 1  # t-legacy-ambig refused to misalign

    rows = [json.loads(line) for line in ds_path.read_text(encoding="utf-8").splitlines()]
    by_prompt = {r["prompt"]: r for r in rows}
    assert "Pick the frog" not in by_prompt

    assert Path(by_prompt["Pick the flying animal"]["image"]).name == "grid-02.png"
    assert by_prompt["Pick the flying animal"]["pictures"] == UNIFORM_3X3_PICTURES
    assert Path(by_prompt["Pick the boat"]["image"]).name == "task-04.png"
    assert by_prompt["Pick the boat"]["layout"] == 3
    assert by_prompt["Pick the boat"]["pictures"] == custom_pictures

    # Legacy timestamped fallback picked task-02.png and task-04.png (skipping task-01.png & task-03.png).
    assert Path(by_prompt["Pick the cat"]["image"]).name == "task-02.png"
    assert by_prompt["Pick the cat"]["pictures"] == custom_pictures
    assert Path(by_prompt["Pick the dog"]["image"]).name == "task-04.png"
    assert by_prompt["Pick the dog"]["pictures"] == custom_pictures

    # passed_over[].image exclusion picked task-02.png.
    assert Path(by_prompt["Pick the owl"]["image"]).name == "task-02.png"


def test_record_retries_presence_error_and_filters_mixed_services(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify presence.error triggers retry and mixed-service manifests filter cleanly by --captcha."""
    # Direct unit test for _has_recording_error on presence.error.
    assert _has_recording_error(
        {
            "status": "answered",
            "vllm_components": {
                "presence": {"error": "HTTP 400 Bad Request", "minimum": 0.0, "skipped": False},
            },
        }
    ) == "presence: HTTP 400 Bad Request"
    assert _has_recording_error(
        {
            "status": "answered",
            "jev_components": {
                "presence": {"error": "timeout"},
            },
        }
    ) == "presence: timeout"
    assert _has_recording_error(
        {
            "status": "answered",
            "vllm_components": {"presence": {"probability": 0.85, "minimum": 0.7, "skipped": False}},
        }
    ) is None

    rc_img = tmp_path / "rc1.png"
    hc_img = tmp_path / "hc1.png"
    rc_img.write_bytes(TINY_PNG + b"rc")
    hc_img.write_bytes(TINY_PNG + b"hc")
    ds_path = tmp_path / "mixed.jsonl"
    ds_rows = [
        {"id": "hc-1", "group_id": "gh1", "service": "hcaptcha", "layout": 3, "prompt": "Pick the tiger", "image": "hc1.png", "pictures": UNIFORM_3X3_PICTURES},
        {"id": "rc-1", "group_id": "gr1", "service": "recaptcha", "layout": 4, "prompt": "Select all squares with\nbuses\nIf there are none, click skip", "image": "rc1.png"},
    ]
    ds_path.write_text("\n".join(json.dumps(r) for r in ds_rows) + "\n", encoding="utf-8")

    calls: list[dict[str, Any]] = []

    class MockServer(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            calls.append(req)
            probs = {str(i): 0.5 for i in range(1, 17)}
            # First call returns presence.error (must be retried); second call succeeds.
            if len(calls) == 1:
                presence_obj: dict[str, Any] = {"error": "HTTP 503 transient", "minimum": 0.0, "skipped": False}
            else:
                presence_obj = {"probability": 0.91, "minimum": 0.0, "skipped": False}
            body = json.dumps(
                {
                    "result": {
                        "status": "answered",
                        "prediction": {"type": "tiles", "value": [1]},
                        "vllm_components": {
                            "whole_grid": {"tile_probabilities": probs},
                            "variant": {
                                "tile_probabilities": probs,
                                "passes": [{"grid_pixels": [389, 389], "tile_probabilities": probs}],
                            },
                            "presence": presence_obj,
                        },
                    }
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockServer)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        out_path = tmp_path / "rec_rc.jsonl"
        # --limit 1 must apply AFTER service filtering so rc-1 is recorded even though hc-1 appears first!
        rc = cli_main(
            [
                "record",
                "--dataset",
                str(ds_path),
                "-o",
                str(out_path),
                "--captcha",
                "recaptcha",
                "--limit",
                "1",
                "--workers",
                "1",
                "--retries",
                "3",
                "--server-url",
                f"http://127.0.0.1:{port}",
            ]
        )
        assert rc == 0
        captured_stdout = capsys.readouterr().out
        assert "1 other-service skipped" in captured_stdout
        assert len(calls) == 2  # 1st failed with presence.error, 2nd succeeded
        assert all("layout" in c and c["layout"] == 4 for c in calls)
        written = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert len(written) == 1
        assert written[0]["grid_id"] == "rc-1"
        assert "error" not in written[0]["result"]["vllm_components"]["presence"]

        # Requesting a service with 0 matching rows fails clearly.
        rc_only = tmp_path / "rc_only.jsonl"
        rc_only.write_text(json.dumps(ds_rows[1]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no dataset items match service 'hcaptcha'"):
            record_dataset(rc_only, tmp_path / "none.jsonl", service="hcaptcha", server_url=f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()
        server.server_close()


def test_schema_and_export_rust_parity_validations(tmp_path: Path) -> None:
    """Verify TOML schema parity with Rust: shipped TOMLs, hCaptcha defaults, base_url, headers, max_images, pass_index."""
    # All shipped models/**/*.toml must load and validate cleanly.
    shipped_tomls = sorted((REPO_ROOT / "models").rglob("*.toml"))
    assert len(shipped_tomls) >= 6
    for toml_path in shipped_tomls:
        cfg = load_model_config(toml_path)
        assert cfg.schema_version == 1

    # hCaptcha export without 3x3 fit defaults to default_decision = "average".
    hc_toml = tmp_path / "hc_default.toml"
    assert (
        cli_main(
            [
                "export",
                "--service",
                "hcaptcha",
                "--name",
                "Custom/hc-model",
                "--provider",
                "custom",
                "--model-id",
                "hc-1",
                "--dialect",
                "systemone-assets",
                "--base-url",
                "https://api.example.com/v1",
                "--api-key-env",
                "HC_API_KEY",
                "-o",
                str(hc_toml),
            ]
        )
        == 0
    )
    hc_cfg = load_model_config(hc_toml)
    assert hc_cfg.tuning is not None
    assert hc_cfg.tuning.default_decision == "average"

    # hCaptcha export WITH 3x3 fit defaults to default_decision = "logistic".
    ds_path, lbl_path, rec_path = _build_synthetic_pipeline_files(tmp_path)
    fit_path = tmp_path / "fit.json"
    sweep_path = tmp_path / "sweep.json"
    assert cli_main(["fit", "--dataset", str(ds_path), "--labels", str(lbl_path), "--recordings", str(rec_path), "-o", str(fit_path)]) == 0
    assert (
        cli_main(
            [
                "sweep",
                "--dataset",
                str(ds_path),
                "--labels",
                str(lbl_path),
                "--recordings",
                str(rec_path),
                "--fit",
                str(fit_path),
                "--fast",
                "--boot",
                "50",
                "-o",
                str(sweep_path),
            ]
        )
        == 0
    )

    hc_fitted_toml = tmp_path / "hc_fitted.toml"
    export_model(
        service="hcaptcha",
        name="vLLM/hc-fitted",
        provider="vllm",
        model_id="vllm-sr/d3-ultra",
        fit_path=fit_path,
        output_toml=hc_fitted_toml,
    )
    hc_fitted_cfg = load_model_config(hc_fitted_toml)
    assert hc_fitted_cfg.tuning is not None
    assert hc_fitted_cfg.tuning.default_decision == "logistic"
    assert hc_fitted_cfg.tuning.logistic_3x3 is not None

    # hCaptcha export with sweep only (no 3x3 fit) still defaults to "average".
    hc_sweep_toml = tmp_path / "hc_sweep.toml"
    export_model(
        service="hcaptcha",
        name="vLLM/hc-sweep",
        provider="vllm",
        model_id="vllm-sr/d3-ultra",
        sweep_path=sweep_path,
        output_toml=hc_sweep_toml,
    )
    hc_sweep_cfg = load_model_config(hc_sweep_toml)
    assert hc_sweep_cfg.tuning is not None
    assert hc_sweep_cfg.tuning.default_decision == "average"

    # Explicit --service hcaptcha --default-decision logistic without 3x3 fit fails clearly.
    with pytest.raises(ValueError, match="3x3 logistic fit"):
        export_model(
            service="hcaptcha",
            name="vLLM/hc-bad",
            provider="vllm",
            model_id="vllm-sr/d3-ultra",
            default_decision="logistic",
            output_toml=tmp_path / "hc_bad.toml",
        )

    # endpoint.base_url validation parity with Rust.
    bad_urls = [
        "ftp://bad.example.com/v1",
        "https://example.com/v1?x=1",
        "https://example.com/v1?",
        "https://example.com/v1#frag",
        "https://user:pass@example.com/v1",
        "https:///missing-host",
        "not-a-url",
        "https://example.com:99999/v1",
    ]
    for bad_url in bad_urls:
        cfg_bad_url = ModelConfig(
            schema_version=1,
            service="recaptcha",
            name="Custom/bad-url",
            provider="custom",
            model_id="m1",
            endpoint=EndpointConfig(
                dialect="systemone-assets",
                base_url=bad_url,
                api_key_env="KEY",
            ),
        )
        with pytest.raises(ValueError, match="base_url"):
            validate_model_config(cfg_bad_url)

    # endpoint.extra_headers validation parity with Rust.
    bad_headers_cases = [
        {"": "val"},
        {"Bad Header": "val"},
        {"Bad:Header": "val"},
        {"X-Test": ""},
        {"X-Test": "   "},
        {"X-Test": "line1\r\nline2"},
        {"X-Test": "caf\u00e9"},
    ]
    for bad_headers in bad_headers_cases:
        cfg_bad_hdr = ModelConfig(
            schema_version=1,
            service="recaptcha",
            name="vLLM/bad-hdr",
            provider="vllm",
            model_id="m1",
            endpoint=EndpointConfig(extra_headers=bad_headers),
        )
        with pytest.raises(ValueError, match="extra_headers"):
            validate_model_config(cfg_bad_hdr)

    # max_images >= 2 for systemone-images (both liquid default dialect & explicit dialect), >= 1 for systemone-assets.
    for cfg_max1_img in (
        ModelConfig(
            schema_version=1,
            service="recaptcha",
            name="Liquid/bad-max",
            provider="liquid",
            model_id="d1.2",
            endpoint=EndpointConfig(max_images=1),
        ),
        ModelConfig(
            schema_version=1,
            service="recaptcha",
            name="Custom/bad-max",
            provider="custom",
            model_id="m1",
            endpoint=EndpointConfig(
                dialect="systemone-images",
                base_url="https://api.example.com/v1",
                api_key_env="KEY",
                max_images=1,
            ),
        ),
    ):
        with pytest.raises(ValueError, match="max_images.*2.*systemone-images"):
            validate_model_config(cfg_max1_img)

    # max_images = 1 is valid for systemone-assets, max_images = 2 is valid for systemone-images.
    validate_model_config(
        ModelConfig(
            schema_version=1,
            service="recaptcha",
            name="vLLM/min-assets",
            provider="vllm",
            model_id="m1",
            endpoint=EndpointConfig(max_images=1),
        )
    )
    validate_model_config(
        ModelConfig(
            schema_version=1,
            service="recaptcha",
            name="Liquid/min-images",
            provider="liquid",
            model_id="d1.2",
            endpoint=EndpointConfig(max_images=2),
        )
    )

    # Nonexistent --pass-index in write_parity_fixture raises clearly instead of falling back.
    rc_toml = tmp_path / "rc_for_fx.toml"
    export_model(
        service="recaptcha",
        name="vLLM/fx-test",
        provider="vllm",
        model_id="vllm-sr/d3-ultra",
        sweep_path=sweep_path,
        fit_path=fit_path,
        output_toml=rc_toml,
    )
    rc_cfg = load_model_config(rc_toml)
    with pytest.raises(ValueError, match=r"pass_index=99.*\[0, 1\]"):
        write_parity_fixture(rc_cfg, rec_path, tmp_path / "bad_pass.json.gz", pass_index=99)
    with pytest.raises(ValueError, match="pass_index must be >= 0"):
        write_parity_fixture(rc_cfg, rec_path, tmp_path / "neg_pass.json.gz", pass_index=-1)
    fx_p1 = tmp_path / "pass1.json.gz"
    info_p1 = write_parity_fixture(rc_cfg, rec_path, fx_p1, pass_index=1)
    assert info_p1["grids"] == 24


def test_label_server_regressions_layout_filter_ipv6_host_and_repeat_check_mkdir(tmp_path: Path) -> None:
    """Verify LabelStore --layout label filtering, IPv6 [::1] Host check, and --repeat-check --json mkdir."""
    # Direct Host header validation checks.
    assert _is_loopback_host_header("127.0.0.1:8787")
    assert _is_loopback_host_header("localhost:8787")
    assert _is_loopback_host_header("[::1]:8787")
    assert _is_loopback_host_header("[::1]")
    assert not _is_loopback_host_header("evil.example.com:8787")
    assert not _is_loopback_host_header("127.0.0.1@evil.example.com")
    assert not _is_loopback_host_header("[::1]:notaport")
    assert not _is_loopback_host_header("127.0.0.1/path")

    img3 = tmp_path / "g3.png"
    img4 = tmp_path / "g4.png"
    img3.write_bytes(TINY_PNG)
    img4.write_bytes(TINY_PNG)
    ds_path = tmp_path / "ds_mixed_layout.jsonl"
    ds_path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"id": "g3", "group_id": "t3", "service": "recaptcha", "layout": 3, "prompt": "Select all images with\nbicycles", "image": "g3.png"},
                {"id": "g4", "group_id": "t4", "service": "recaptcha", "layout": 4, "prompt": "Select all squares with\nbuses", "image": "g4.png"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels_mixed.jsonl"
    labels_path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"id": "g3", "layout": 3, "target": [1], "sliver": [], "unclear": False},
                {"id": "g4", "layout": 4, "target": [2], "sliver": [], "unclear": False},
                {"id": "stale-other", "layout": 4, "target": [3], "sliver": [], "unclear": False},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store4 = LabelStore(ds_path, labels_path, layout=4)
    assert list(store4.get_labels().keys()) == ["g4"]

    server = create_label_server(ds_path, labels_path, port=0, layout=4)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(trust_env=False) as client:
            # /api/labels filters to layout=4 items only.
            resp_labels = client.get(f"{base}/api/labels")
            assert resp_labels.status_code == 200
            assert list(resp_labels.json().keys()) == ["g4"]

            # IPv6 loopback Host header is accepted.
            resp_v6 = client.get(f"{base}/api/items", headers={"Host": f"[::1]:{port}"})
            assert resp_v6.status_code == 200
            assert [it["id"] for it in resp_v6.json()] == ["g4"]

            # Non-loopback Host header is rejected with 403.
            resp_bad = client.get(f"{base}/api/items", headers={"Host": f"evil.com:{port}"})
            assert resp_bad.status_code == 403
    finally:
        server.shutdown()
        server.server_close()

    # --repeat-check --json creates missing parent directories automatically.
    nested_json = tmp_path / "deep" / "nested" / "repeat_report.json"
    assert (
        cli_main(
            [
                "label",
                "--repeat-check",
                str(labels_path),
                str(labels_path),
                "--json",
                str(nested_json),
            ]
        )
        == 0
    )
    assert nested_json.is_file()
    rep = json.loads(nested_json.read_text(encoding="utf-8"))
    assert rep["common_grids"] == 3


def test_folds_and_boot_validation_in_schema_fit_and_sweep(tmp_path: Path) -> None:
    """Verify n_folds >= 2 and --boot >= 1 are enforced in Python APIs and CLIs."""
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        make_group_folds(["g1", "g2", "g3"], n_folds=0)
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        make_group_folds(["g1", "g2", "g3"], n_folds=1)

    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        grouped_bootstrap_diff(
            np.array([0, 1]),
            2,
            np.array([[True, False]]),
            np.array([[False, False]]),
            n_boot=0,
        )

    ds_path, lbl_path, rec_path = _build_synthetic_pipeline_files(tmp_path)

    for bad_folds in (0, 1, -2):
        with pytest.raises(ValueError, match="folds must be >= 2"):
            fit_dataset(ds_path, lbl_path, rec_path, folds=bad_folds)
        with pytest.raises(ValueError, match="folds must be >= 2"):
            sweep_4x4(ds_path, lbl_path, rec_path, folds=bad_folds, fast=True, n_boot=10)
        with pytest.raises(SystemExit) as exc_fit:
            cli_main(
                [
                    "fit",
                    "--dataset",
                    str(ds_path),
                    "--labels",
                    str(lbl_path),
                    "--recordings",
                    str(rec_path),
                    "--folds",
                    str(bad_folds),
                ]
            )
        assert exc_fit.value.code == 2
        with pytest.raises(SystemExit) as exc_sweep:
            cli_main(
                [
                    "sweep",
                    "--dataset",
                    str(ds_path),
                    "--labels",
                    str(lbl_path),
                    "--recordings",
                    str(rec_path),
                    "-o",
                    str(tmp_path / "unused.json"),
                    "--folds",
                    str(bad_folds),
                ]
            )
        assert exc_sweep.value.code == 2

    for bad_boot in (0, -5):
        with pytest.raises(ValueError, match="n_boot must be >= 1"):
            sweep_4x4(ds_path, lbl_path, rec_path, n_boot=bad_boot, fast=True)
        with pytest.raises(SystemExit) as exc_boot:
            cli_main(
                [
                    "sweep",
                    "--dataset",
                    str(ds_path),
                    "--labels",
                    str(lbl_path),
                    "--recordings",
                    str(rec_path),
                    "-o",
                    str(tmp_path / "unused.json"),
                    "--boot",
                    str(bad_boot),
                ]
            )
        assert exc_boot.value.code == 2


# ---------------------------------------------------------------------------
# 8. DX streamlining: --base-config, --service/--captcha aliases, record progress,
#    pyproject extras, and Camoufox-only runner scripts
# ---------------------------------------------------------------------------


def test_export_base_config_inheritance_overrides_and_errors(tmp_path: Path) -> None:
    """Verify `export --base-config` inherits metadata + [endpoint], allows CLI overrides, and errors cleanly."""
    ds_path, lbl_path, rec_path = _build_synthetic_pipeline_files(tmp_path)
    fit_path = tmp_path / "fit.json"
    sweep_path = tmp_path / "sweep.json"
    assert cli_main(["fit", "--dataset", str(ds_path), "--labels", str(lbl_path), "--recordings", str(rec_path), "-o", str(fit_path)]) == 0
    assert (
        cli_main(
            [
                "sweep",
                "--dataset",
                str(ds_path),
                "--labels",
                str(lbl_path),
                "--recordings",
                str(rec_path),
                "--fit",
                str(fit_path),
                "--fast",
                "--boot",
                "50",
                "-o",
                str(sweep_path),
            ]
        )
        == 0
    )

    # 1. Pure inheritance from the shipped custom example TOML (no --name/--provider/--model-id/--service needed).
    starter = REPO_ROOT / "models" / "examples" / "custom-decision-model.toml"
    out_inherited = tmp_path / "inherited.toml"
    assert (
        cli_main(
            [
                "export",
                "--base-config",
                str(starter),
                "--fit",
                str(fit_path),
                "--sweep",
                str(sweep_path),
                "-o",
                str(out_inherited),
            ]
        )
        == 0
    )
    inherited = load_model_config(out_inherited)
    assert inherited.service == "recaptcha"
    assert inherited.name == "MyProvider/my-decision-model"
    assert inherited.provider == "custom"
    assert inherited.model_id == "my-decision-model-v1"
    assert inherited.endpoint is not None
    assert inherited.endpoint.dialect == "systemone-assets"
    assert inherited.endpoint.base_url == "https://api.example.com/v1"
    assert inherited.endpoint.api_key_env == "MY_DECISION_API_KEY"
    assert inherited.endpoint.max_images == 16
    assert inherited.endpoint.gzip is False
    # Freshly exported tuning replaces starter's average-only tuning.
    assert inherited.tuning is not None
    assert inherited.tuning.default_decision == "logistic"
    assert inherited.tuning.logistic_3x3 is not None
    assert inherited.tuning.logistic_4x4 is not None
    assert inherited.tuning.efforts is not None

    # 2. Custom hCaptcha base config with extra_headers: verify service inheritance + selective CLI overrides.
    hc_base = tmp_path / "hc_starter.toml"
    hc_base.write_text(
        """schema_version = 1
service = "hcaptcha"
name = "Base/hc-model"
provider = "custom"
model_id = "base-hc-v1"

[endpoint]
dialect = "systemone-assets"
base_url = "https://base.example.com/v1"
api_key_env = "BASE_KEY"
gzip = false
max_images = 12

[endpoint.extra_headers]
"X-Keep" = "keep-me"
"X-Override" = "old-val"
""",
        encoding="utf-8",
    )
    out_hc_inherited = tmp_path / "hc_inherited.toml"
    assert cli_main(["export", "--base-config", str(hc_base), "--fit", str(fit_path), "-o", str(out_hc_inherited)]) == 0
    hc_cfg = load_model_config(out_hc_inherited)
    assert hc_cfg.service == "hcaptcha"
    assert hc_cfg.endpoint is not None
    assert hc_cfg.endpoint.extra_headers == {"X-Keep": "keep-me", "X-Override": "old-val"}

    out_overridden = tmp_path / "hc_overridden.toml"
    assert (
        cli_main(
            [
                "export",
                "--base-config",
                str(hc_base),
                "--captcha",
                "recaptcha",
                "--name",
                "Override/rc-model",
                "--model-id",
                "override-rc-v2",
                "--base-url",
                "https://override.example.com/v2",
                "--gzip",
                "--max-images",
                "16",
                "--extra-header",
                "X-Override=new-val",
                "--extra-header",
                "X-Added=added-val",
                "--fit",
                str(fit_path),
                "--sweep",
                str(sweep_path),
                "-o",
                str(out_overridden),
            ]
        )
        == 0
    )
    ov_cfg = load_model_config(out_overridden)
    assert ov_cfg.service == "recaptcha"
    assert ov_cfg.name == "Override/rc-model"
    assert ov_cfg.provider == "custom"  # inherited
    assert ov_cfg.model_id == "override-rc-v2"
    assert ov_cfg.endpoint is not None
    assert ov_cfg.endpoint.dialect == "systemone-assets"  # inherited
    assert ov_cfg.endpoint.api_key_env == "BASE_KEY"  # inherited
    assert ov_cfg.endpoint.base_url == "https://override.example.com/v2"
    assert ov_cfg.endpoint.gzip is True
    assert ov_cfg.endpoint.max_images == 16
    assert ov_cfg.endpoint.extra_headers == {
        "X-Keep": "keep-me",
        "X-Override": "new-val",
        "X-Added": "added-val",
    }

    # 3. Error handling: missing required flags without --base-config exits with code 2 (CLI) / ValueError (API).
    with pytest.raises(SystemExit) as exc_missing:
        cli_main(["export", "-o", str(tmp_path / "missing.toml")])
    assert exc_missing.value.code == 2

    with pytest.raises(ValueError, match="--name.*--provider.*--model-id"):
        export_model(output_toml=tmp_path / "missing_api.toml")

    # 4. Nonexistent or invalid --base-config fails clearly.
    with pytest.raises(FileNotFoundError):
        export_model(base_config_path=tmp_path / "nope.toml", output_toml=tmp_path / "out.toml")

    bad_base = tmp_path / "bad_base.toml"
    bad_base.write_text('schema_version = 99\nservice = "recaptcha"\nname = "x"\nprovider = "vllm"\nmodel_id = "y"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        export_model(base_config_path=bad_base, output_toml=tmp_path / "out.toml")


def test_service_and_captcha_aliases_and_record_stderr_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--service`/`--captcha` work on dataset/record/export and `record` logs progress to stderr only."""
    run_rc = tmp_path / "rc"
    run_hc = tmp_path / "hc"
    (run_rc / "t1").mkdir(parents=True)
    (run_hc / "h1").mkdir(parents=True)
    (run_rc / "t1" / "round-01.png").write_bytes(TINY_PNG + b"rc-alias")
    (run_hc / "h1" / "grid-01.png").write_bytes(TINY_PNG + b"hc-alias")
    (run_rc / "trials.jsonl").write_text(
        json.dumps({"trial_id": "t1", "rounds": [{"prompt": "Select all squares with\nbuses", "layout": 4, "image": "round-01.png"}]}) + "\n",
        encoding="utf-8",
    )
    (run_hc / "trials.jsonl").write_text(
        json.dumps({"trial_id": "h1", "captcha": "hcaptcha", "rounds": [{"prompt": "Pick the tiger", "layout": 3, "image": "grid-01.png"}]}) + "\n",
        encoding="utf-8",
    )

    # 1. `dataset`: both --service and --captcha filter identically.
    ds_svc = tmp_path / "ds_svc.jsonl"
    ds_cap = tmp_path / "ds_cap.jsonl"
    assert cli_main(["dataset", str(run_rc), str(run_hc), "--service", "hcaptcha", "-o", str(ds_svc)]) == 0
    assert cli_main(["dataset", str(run_rc), str(run_hc), "--captcha", "hcaptcha", "-o", str(ds_cap)]) == 0
    rows_svc = [json.loads(line) for line in ds_svc.read_text(encoding="utf-8").splitlines()]
    rows_cap = [json.loads(line) for line in ds_cap.read_text(encoding="utf-8").splitlines()]
    assert len(rows_svc) == 1 and rows_svc[0]["service"] == "hcaptcha"
    assert rows_svc == rows_cap

    # 2. `export`: --captcha alias sets service in TOML.
    ex_toml = tmp_path / "ex_hc.toml"
    assert (
        cli_main(
            [
                "export",
                "--captcha",
                "hcaptcha",
                "--name",
                "vLLM/hc-alias",
                "--provider",
                "vllm",
                "--model-id",
                "vllm-sr/d3-ultra",
                "-o",
                str(ex_toml),
            ]
        )
        == 0
    )
    assert load_model_config(ex_toml).service == "hcaptcha"
    capsys.readouterr()  # clear previous stdout/stderr

    # 3. `record`: --service alias works AND progress is written to stderr without polluting stdout or JSONL.
    ds_all = tmp_path / "ds_all.jsonl"
    assert cli_main(["dataset", str(run_rc), str(run_hc), "-o", str(ds_all)]) == 0
    capsys.readouterr()

    class MockProgressLab(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            probs = {str(i): 0.5 for i in range(1, 17)}
            body = json.dumps(
                {
                    "result": {
                        "status": "answered",
                        "prediction": {"type": "tiles", "value": [1]},
                        "vllm_components": {
                            "whole_grid": {"tile_probabilities": probs},
                            "variant": {"tile_probabilities": probs, "passes": [{"grid_pixels": [389, 389], "tile_probabilities": probs}]},
                        },
                    }
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockProgressLab)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rec_out = tmp_path / "rec_progress.jsonl"
        rc = cli_main(
            [
                "record",
                "--dataset",
                str(ds_all),
                "-o",
                str(rec_out),
                "--service",
                "recaptcha",
                "--passes",
                "2",
                "--workers",
                "1",
                "--server-url",
                f"http://127.0.0.1:{port}",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "[1/2] pass=0" in captured.err
        assert "[2/2] pass=1" in captured.err
        assert "[1/2]" not in captured.out
        assert "Recorded 2 new rounds" in captured.out
        assert "1 other-service skipped" in captured.out
        # JSONL must contain strictly valid JSON entries (zero progress lines).
        jsonl_rows = [json.loads(line) for line in rec_out.read_text(encoding="utf-8").splitlines()]
        assert len(jsonl_rows) == 2
        assert {r["pass_index"] for r in jsonl_rows} == {0, 1}
    finally:
        server.shutdown()
        server.server_close()


def test_pyproject_extras_and_deleted_legacy_scripts() -> None:
    """Verify `pyproject.toml` consolidates browser/camoufox into a single `browser` extra and legacy scripts are gone."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert extras["browser"] == ["camoufox==0.5.9"]
    assert extras["dev"] == ["pytest>=8.0"]
    assert "camoufox" not in extras

    assert not (REPO_ROOT / "scripts" / "chrome_lab.py").exists()
    assert not (REPO_ROOT / "scripts" / "camoufox_lab.py").exists()
    for name in ("_common.py", "recaptcha_lab.py", "hcaptcha_lab.py"):
        src = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "playwright.chromium" not in src
        assert "sync_playwright" not in src


def test_scripts_common_helpers_and_mode_runners(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `scripts/_common.py` URL validation, parsers, model helpers, and interactive/smoke/vision-check helpers."""
    import argparse

    from scripts import _common

    # URL validation: localhost, *.localhost, IPv4/IPv6 loopback, and loopback-resolving /etc/hosts names.
    assert _common.validate_lab_url("http://localhost:8765") == "http://localhost:8765"
    assert _common.validate_lab_url("http://localhost:8765/") == "http://localhost:8765"
    assert _common.validate_lab_url("http://hypercaptcha.localhost:8765") == "http://hypercaptcha.localhost:8765"
    assert _common.validate_lab_url("http://127.0.0.1:8765") == "http://127.0.0.1:8765"
    assert _common.validate_lab_url("http://[::1]:8765") == "http://[::1]:8765"

    fake_loopback_resolver = lambda host, port: [(2, 1, 6, "", ("127.0.0.1", 0))]  # noqa: E731
    fake_mixed_resolver = lambda host, port: [  # noqa: E731
        (2, 1, 6, "", ("127.0.0.1", 0)),
        (2, 1, 6, "", ("93.184.216.34", 0)),
    ]
    assert _common.validate_lab_url("http://lab.test:8765", resolver=fake_loopback_resolver) == "http://lab.test:8765"
    assert _common.validate_lab_url("http://lab.test:8765", resolver=fake_mixed_resolver) is None

    for bad_url in (
        "https://localhost:8765",
        "http://example.com:8765",
        "http://user:pass@localhost:8765",
        "http://localhost:8765/path",
        "http://localhost:8765/?q=1",
        "http://localhost:8765/#frag",
        "http://localhost:99999",
    ):
        assert _common.validate_lab_url(bad_url, resolver=lambda h, p: [(2, 1, 6, "", ("93.184.216.34", 0))]) is None

    # Numeric argument parsers.
    assert _common.positive_int("1") == 1
    assert _common.positive_int("50") == 50
    for bad in ("0", "-3", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            _common.positive_int(bad)
    assert _common.non_negative_int("0") == 0
    assert _common.non_negative_int("10") == 10
    for bad in ("-1", "xyz"):
        with pytest.raises(argparse.ArgumentTypeError):
            _common.non_negative_int(bad)
    assert _common.positive_float("2.0") == 2.0
    for bad in ("0", "-1.5", "nan", "inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            _common.positive_float(bad)

    # Captcha config validation with cross-script hints.
    assert _common.validate_captcha_config({"captcha": "recaptcha"}, "recaptcha") is None
    assert _common.validate_captcha_config({"captcha": "hcaptcha"}, "hcaptcha") is None
    hint_hc = _common.validate_captcha_config({"captcha": "hcaptcha"}, "recaptcha")
    assert hint_hc is not None and "scripts/hcaptcha_lab.py" in hint_hc
    hint_rc = _common.validate_captcha_config({"captcha": "recaptcha"}, "hcaptcha")
    assert hint_rc is not None and "scripts/recaptcha_lab.py" in hint_rc

    # Model name resolution and reply error helpers.
    assert _common.resolve_model_name({"model_name": "vLLM/d3-ultra"}, {}) == "vLLM/d3-ultra"
    assert _common.resolve_model_name({}, {"rounds": [{"model": {"model_name": "Liquid/d1.2"}}]}) == "Liquid/d1.2"
    assert _common.resolve_model_name({"provider": "vllm", "model": "d3"}, {}) == "vllm/d3"
    assert _common.resolve_model_name({}, {}) is None

    assert not _common.model_reply_failed({"result": {"status": "answered"}})
    assert _common.model_reply_failed({"result": {"status": "api_error"}})
    assert _common.model_reply_failed("not-a-dict")
    assert _common.format_model_http_error((400, "bad payload"), "ignored") == "Model request failed (HTTP 400): bad payload"
    assert _common.format_model_http_error((400, None), "status msg") == "Model request failed (HTTP 400): status msg"
    assert _common.format_model_http_error(None, "timeout") == "Model request failed: timeout"

    # Fake Playwright browser/context/page to verify run_vision_check, run_interactive, and run_smoke offline.
    class FakeResp:
        def __init__(self, status: int, payload: Any) -> None:
            self.status = status
            self.ok = 200 <= status < 300
            self._payload = payload

        def json(self) -> Any:
            return self._payload

    class FakeLocator:
        def __init__(self, text: str = "") -> None:
            self._text = text
            self.waited = False

        def wait_for(self, **_kwargs: Any) -> None:
            self.waited = True

        def inner_text(self) -> str:
            return self._text

    class FakeRequest:
        def __init__(self) -> None:
            self.posts: list[tuple[str, dict[str, Any]]] = []

        def post(self, url: str, headers: dict[str, str], data: dict[str, Any], timeout: int) -> FakeResp:
            self.posts.append((url, data))
            return FakeResp(200, {"result": {"passed": True}})

    class FakePage:
        def __init__(self, result_text: str = '{"result": {"valid": true}}') -> None:
            self.url = ""
            self.request = FakeRequest()
            self.headings_waited: list[str] = []
            self.closed_waited = False
            self._result_text = result_text

        def goto(self, url: str, **_kwargs: Any) -> FakeResp:
            self.url = url
            return FakeResp(200, {})

        def get_by_role(self, role: str, name: str) -> FakeLocator:
            self.headings_waited.append(f"{role}:{name}")
            return FakeLocator()

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(self._result_text)

        def wait_for_event(self, event: str, timeout: int = 0) -> None:
            assert event == "close" and timeout == 0
            self.closed_waited = True

    class FakeContext:
        def __init__(self, page: FakePage) -> None:
            self._page = page

        def __enter__(self) -> FakeContext:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def new_page(self) -> FakePage:
            return self._page

    class FakeBrowser:
        version = "135.0-fake"

        def __init__(self, page: FakePage) -> None:
            self._page = page

        def new_context(self, **_kwargs: Any) -> FakeContext:
            return FakeContext(self._page)

    # Vision check writes vision-checks.jsonl with source="camoufox".
    vc_page = FakePage()
    vc_dir = tmp_path / "vc_out"
    vc_id, check = _common.run_vision_check(FakeBrowser(vc_page), "http://localhost:8765", vc_dir)
    assert vc_id and check["result"]["passed"] is True
    assert vc_page.request.posts[0][1]["source"] == "camoufox"
    assert (vc_dir / "vision-checks.jsonl").is_file()

    # Interactive runner opens #source=camoufox, waits for heading and close event.
    int_page = FakePage()
    _common.run_interactive(FakeBrowser(int_page), "http://localhost:8765", heading="Google widget")
    assert int_page.url == "http://localhost:8765/#source=camoufox"
    assert int_page.headings_waited == ["heading:Google widget"]
    assert int_page.closed_waited is True

    # Smoke runner clicks checkbox callback and returns parsed JSON record.
    clicked = {"yes": False}
    smoke_page = FakePage('{"result": {"valid": true}}')
    rec = _common.run_smoke(
        FakeBrowser(smoke_page),
        "http://hypercaptcha.localhost:8765",
        heading="hCaptcha widget",
        click_checkbox=lambda p: clicked.__setitem__("yes", True),
        result_selector="#verdict",
    )
    assert clicked["yes"] is True
    assert smoke_page.url == "http://hypercaptcha.localhost:8765/#source=camoufox"
    assert rec == {"result": {"valid": True}}
    capsys.readouterr()


def test_recaptcha_and_hcaptcha_cli_parsing_validation_and_score_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI defaults, lifted trial cap, flag conflict rejection, config checks, and score error handling."""
    import base64

    from scripts import hcaptcha_lab, recaptcha_lab

    # --- recaptcha_lab.py ---
    assert recaptcha_lab.BROWSER == "camoufox"
    assert recaptcha_lab.SOURCE == "camoufox"

    _, rc_default, rc_url = recaptcha_lab.parse_args([])
    assert rc_url == "http://localhost:8765"
    assert rc_default.output == Path("reports/recaptcha")
    assert rc_default.trials == 1
    assert rc_default.max_rounds == 6
    assert rc_default.max_stages == 0
    assert rc_default.fast is True
    assert rc_default.early_tiles is True
    assert rc_default.headed is False
    assert rc_default.submit is False

    # Lifted --trials cap (> 20 works), custom loopback URL, --headed, --no-early-tiles, --no-fast.
    _, rc_many, rc_custom_url = recaptcha_lab.parse_args(
        [
            "--url",
            "http://127.0.0.1:9000/",
            "--trials",
            "50",
            "--max-rounds",
            "30",
            "--max-stages",
            "5",
            "--submit",
            "--no-early-tiles",
            "--no-fast",
            "--headed",
            "--output",
            "reports/custom_rc",
        ]
    )
    assert rc_custom_url == "http://127.0.0.1:9000"
    assert rc_many.trials == 50
    assert rc_many.max_rounds == 30
    assert rc_many.max_stages == 5
    assert rc_many.submit is True
    assert rc_many.early_tiles is False
    assert rc_many.fast is False
    assert rc_many.headed is True
    assert rc_many.output == Path("reports/custom_rc")

    # --browser flag is removed; special modes are mutually exclusive; invalid combinations rejected.
    bad_rc_invocations = [
        ["--browser", "chrome"],
        ["--url", "https://example.com"],
        ["--trials", "0"],
        ["--max-rounds", "0"],
        ["--max-stages", "-1"],
        ["--interactive", "--smoke"],
        ["--smoke", "--score"],
        ["--interactive", "--score"],
        ["--interactive", "--submit"],
        ["--smoke", "--submit"],
        ["--score", "--submit"],
        ["--smoke", "--no-early-tiles"],
        ["--score", "--fast"],
        ["--interactive", "--no-fast"],
        ["--smoke", "--trials", "2"],
        ["--score", "--output", "reports/x"],
    ]
    for argv in bad_rc_invocations:
        with pytest.raises(SystemExit) as exc:
            recaptcha_lab.parse_args(argv)
        assert exc.value.code == 2, f"expected exit code 2 for recaptcha_lab {argv}"

    # Valid special modes with --headed.
    for mode_flag in ("--interactive", "--smoke", "--score"):
        _, parsed, _ = recaptcha_lab.parse_args([mode_flag, "--headed"])
        assert parsed.headed is True

    # Config validation on recaptcha_lab.
    err_wrong = recaptcha_lab.validate_settings(rc_default, {"captcha": "hcaptcha", "mode": "checkbox"})
    assert err_wrong is not None and "scripts/hcaptcha_lab.py" in err_wrong

    _, rc_smoke_args, _ = recaptcha_lab.parse_args(["--smoke"])
    assert recaptcha_lab.validate_settings(rc_smoke_args, {"captcha": "recaptcha", "mode": "v2", "test": True, "score_based": False}) is None
    assert recaptcha_lab.validate_settings(rc_smoke_args, {"captcha": "recaptcha", "mode": "v2", "test": False, "score_based": False}) is not None

    _, rc_score_args, _ = recaptcha_lab.parse_args(["--score"])
    assert recaptcha_lab.validate_settings(rc_score_args, {"captcha": "recaptcha", "mode": "v3", "score_based": True}) is None
    assert recaptcha_lab.validate_settings(rc_score_args, {"captcha": "recaptcha", "mode": "v2", "score_based": False}) is not None

    _, rc_int_args, _ = recaptcha_lab.parse_args(["--interactive"])
    assert recaptcha_lab.validate_settings(rc_int_args, {"captcha": "recaptcha", "mode": "v3", "score_based": True}) is None

    assert recaptcha_lab.validate_settings(rc_default, {"captcha": "recaptcha", "mode": "v2", "test": True}) is not None
    assert recaptcha_lab.validate_settings(rc_default, {"captcha": "recaptcha", "mode": "v2", "test": False}) is None
    assert recaptcha_lab.validate_settings(rc_many, {"captcha": "recaptcha", "mode": "v2", "test": False, "supports_vision": False}) is not None
    assert recaptcha_lab.validate_settings(rc_many, {"captcha": "recaptcha", "mode": "v2", "test": False, "supports_vision": True}) is None

    # evaluate_early_answer edge cases.
    b64_png = base64.b64encode(TINY_PNG).decode("ascii")
    drawn, problem = recaptcha_lab.evaluate_early_answer(
        {
            "grid_png": b64_png,
            "tile_threshold": 0.5,
            "result": {"status": "answered", "prediction": {"type": "tiles", "value": [1]}, "tile_probabilities": {"1": 0.9}},
        }
    )
    assert drawn == TINY_PNG and problem is None
    _, close_problem = recaptcha_lab.evaluate_early_answer(
        {
            "grid_png": b64_png,
            "tile_threshold": 0.5,
            "result": {"status": "answered", "prediction": {"type": "tiles", "value": [1]}, "tile_probabilities": {"1": 0.55}},
        }
    )
    assert close_problem == "a score close to the cutoff"

    # run_score fast HTTP 400 error handling (offline fake page).
    class ScoreResp:
        def __init__(self, url: str, status: int, payload: Any) -> None:
            self.url = url
            self.status = status
            self.ok = 200 <= status < 300
            self._payload = payload

        def json(self) -> Any:
            return self._payload

    class ScoreLocator:
        def __init__(self, text: str = "") -> None:
            self._text = text

        def wait_for(self, **_kwargs: Any) -> None:
            return None

        def click(self, **_kwargs: Any) -> None:
            return None

        def inner_text(self) -> str:
            return self._text

    class ScoreWaitHandle:
        def __init__(self, val: dict[str, Any]) -> None:
            self._val = val

        def json_value(self) -> dict[str, Any]:
            return self._val

    class ScorePage:
        def __init__(self, wait_outcome: dict[str, Any], http_err_resp: ScoreResp | None = None) -> None:
            self._listeners: list[Any] = []
            self._wait_outcome = wait_outcome
            self._http_err_resp = http_err_resp

        def on(self, event: str, fn: Any) -> None:
            if event == "response":
                self._listeners.append(fn)

        def goto(self, url: str, **_kwargs: Any) -> ScoreResp:
            return ScoreResp(url, 200, {})

        def get_by_role(self, role: str, name: str) -> ScoreLocator:
            return ScoreLocator()

        def locator(self, selector: str) -> ScoreLocator:
            return ScoreLocator('{"result": {"valid": true, "score": 0.9}}')

        def wait_for_function(self, js: str, **_kwargs: Any) -> ScoreWaitHandle:
            if self._http_err_resp is not None:
                for fn in self._listeners:
                    fn(self._http_err_resp)
            return ScoreWaitHandle(self._wait_outcome)

    class ScoreBrowser:
        version = "135.0-fake"

        def __init__(self, page: ScorePage) -> None:
            self._page = page

        def new_context(self, **_kwargs: Any) -> Any:
            page = self._page

            class Ctx:
                def __enter__(self) -> Ctx:
                    return self

                def __exit__(self, *args: Any) -> None:
                    return None

                def new_page(self) -> ScorePage:
                    return page

            return Ctx()

    capsys.readouterr()
    err_page = ScorePage(
        {"kind": "error", "text": "Bad request"},
        ScoreResp("http://localhost:8765/model", 400, {"error": "provider rejected score trace"}),
    )
    with pytest.raises(SystemExit) as exc_400:
        recaptcha_lab.run_score(ScoreBrowser(err_page), "http://localhost:8765", {"provider": "openai-completions"})
    assert exc_400.value.code == 1
    assert "HTTP 400" in capsys.readouterr().err

    non_ans_page = ScorePage({"kind": "result", "text": '{"result": {"status": "refused"}}'})
    with pytest.raises(SystemExit) as exc_refused:
        recaptcha_lab.run_score(ScoreBrowser(non_ans_page), "http://localhost:8765", {"provider": "openai-completions"})
    assert exc_refused.value.code == 1

    # --- hcaptcha_lab.py ---
    assert hcaptcha_lab.BROWSER == "camoufox"
    assert hcaptcha_lab.SOURCE == "camoufox"

    _, hc_default, hc_url = hcaptcha_lab.parse_args([])
    assert hc_url == "http://hypercaptcha.localhost:8765"
    assert hc_default.output == Path("reports/hcaptcha")
    assert hc_default.trials == 1
    assert hc_default.max_rounds == 24
    assert hc_default.max_stages == 1
    assert hc_default.max_refreshes == 12
    assert hc_default.fast is True
    assert hc_default.headed is False
    assert hc_default.submit is False

    _, hc_custom, hc_custom_url = hcaptcha_lab.parse_args(
        ["--url", "http://[::1]:8765", "--trials", "5", "--submit", "--headed", "--max-refreshes", "4"]
    )
    assert hc_custom_url == "http://[::1]:8765"
    assert hc_custom.trials == 5
    assert hc_custom.submit is True
    assert hc_custom.headed is True
    assert hc_custom.max_refreshes == 4

    bad_hc_invocations = [
        ["--url", "https://example.com"],
        ["--trials", "0"],
        ["--max-stages", "0"],
        ["--max-refreshes", "0"],
        ["--interactive", "--smoke"],
        ["--interactive", "--submit"],
        ["--smoke", "--submit"],
        ["--smoke", "--trials", "2"],
        ["--interactive", "--max-refreshes", "3"],
        ["--smoke", "--no-fast"],
        ["--interactive", "--output", "reports/x"],
    ]
    for argv in bad_hc_invocations:
        with pytest.raises(SystemExit) as exc:
            hcaptcha_lab.parse_args(argv)
        assert exc.value.code == 2, f"expected exit code 2 for hcaptcha_lab {argv}"

    # Config validation on hcaptcha_lab.
    err_wrong_hc = hcaptcha_lab.validate_settings(hc_default, {"captcha": "recaptcha", "mode": "v2"})
    assert err_wrong_hc is not None and "scripts/recaptcha_lab.py" in err_wrong_hc

    _, hc_smoke_args, _ = hcaptcha_lab.parse_args(["--smoke"])
    assert hcaptcha_lab.validate_settings(hc_smoke_args, {"captcha": "hcaptcha", "test": True}) is None
    assert hcaptcha_lab.validate_settings(hc_smoke_args, {"captcha": "hcaptcha", "test": False}) is not None

    _, hc_int_args, _ = hcaptcha_lab.parse_args(["--interactive"])
    assert hcaptcha_lab.validate_settings(hc_int_args, {"captcha": "hcaptcha", "test": True}) is None

    assert hcaptcha_lab.validate_settings(hc_default, {"captcha": "hcaptcha", "test": True}) is not None
    assert hcaptcha_lab.validate_settings(hc_default, {"captcha": "hcaptcha", "test": False}) is None
    assert hcaptcha_lab.validate_settings(hc_custom, {"captcha": "hcaptcha", "test": False, "supports_vision": False}) is not None
    assert hcaptcha_lab.validate_settings(hc_custom, {"captcha": "hcaptcha", "test": False, "supports_vision": True}) is None
    capsys.readouterr()
