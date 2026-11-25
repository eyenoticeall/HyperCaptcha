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
from eval.dataset import build_dataset
from eval.export import export_model, render_model_toml
from eval.fit import fit_dataset
from eval.label_server import LabelStore, compare_labels, create_label_server
from eval.record import record_dataset
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
)
from eval.summarize import format_summary_markdown, summarize_files, wilson_proportion
from eval.sweep import sweep_4x4

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
