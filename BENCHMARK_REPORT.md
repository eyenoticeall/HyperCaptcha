# Live reCAPTCHA benchmark and failure review

> **Historical benchmark note:** This report preserves the original November 15, 2026 50-attempt live comparison of `typesafe/Jev` and `vLLM/d3-ultra` using equal-weight 50/50 multi-view averaging. For all subsequent multi-model evaluations (`vLLM/d3-ultra`, `Liquid/d1.2-ultra`, `Liquid/d1.2`, `typesafe/Jev`), logistic and box-veto effort ladders, later live runs, and hCaptcha status, see [docs/RESULTS.md](docs/RESULTS.md).

**Run date** November 15, 2026, 04:01–04:17 Singapore time

## Research context

HyperCaptcha is a Rust research harness for measuring how modern vision and decision models perform against Google reCAPTCHA in a controlled lab. It supports testing live widgets on researcher-controlled localhost keys alongside replaying saved puzzles. Live runs measure whether Google verifies the interaction. Saved grids let us compare model inputs on the same image and investigate individual tile decisions.

The broader project covers checkbox and score-based integrations, including Enterprise and non-Enterprise variants. This run compares Jev with d3-ultra on image challenges from a live **reCAPTCHA v2 Enterprise checkbox key** in Camoufox. An earlier JPEG pilot verified 7 of 7 image attempts for Jev and 8 of 8 for d3-ultra, with three additional checkbox-only passes across the two models. The pilot included two 4×4 image attempts, both for d3-ultra. The larger run estimates pass rates separately by grid size. [Harness overview](README.md) · Local pilot trial log: `reports/decision-jpeg-comparison/live.jsonl` *(gitignored)*

## Method

- Ran 50 attempts per model in headless Camoufox `158.0.1-beta.41`. Each attempt used a fresh browser context and a live localhost key. The runner alternated which model went first in each pair and kept one widget active at a time. Google served its own puzzle sequence to each model.
- Compared TypeSafe Jev `jev-1.14.0` with System One `vllm-sr/d3-ultra`. Both passed the same red and blue vision controls before the run. The harness saved the widget screenshots as PNGs and encoded the inputs sent to both providers as JPEG at quality 90.
- For 3×3, the harness sent two parallel requests, one for the whole grid and one for the nine individual photos. For 4×4, it sent three parallel requests, one for the whole grid and two batches of eight crops. Each 4×4 crop batch also received the whole scene. The harness averaged the two view scores per tile with equal weight and selected scores at or above 60% on 3×3 and 50% on 4×4.
- Counted a pass when Google returned an assessment that the lab validated for `localhost`. A visible widget rejection or the six-round limit ended the attempt as a failure. Subgroup rates use the first grid in each attempt. Model-round timings use the layout of each individual round.

[Browser runner](scripts/chrome_lab.py) · [Provider pipeline](src/providers.rs) · Local comparison runner: `reports/decision-live-50/run.py` *(gitignored)*

## Benchmark findings

All 100 attempts opened an image challenge. Both models answered every scored round. The run captured 145 grids, 67 for Jev and 78 for d3-ultra.

| Outcome | Jev | d3-ultra |
| --- | ---: | ---: |
| Google-verified attempts | 34/50 (68.0%) | 40/50 (80.0%) |
| 95% Wilson interval for overall pass rate | 54.2–79.2% | 67.0–88.8% |
| Starting on 3×3 | 28/34 (82.4%) | 28/34 (82.4%) |
| Starting on 3×3 without tile replacement | 27/30 (90.0%) | 26/27 (96.3%) |
| Starting on 3×3 with tile replacement | 1/4 (25.0%) | 2/7 (28.6%) |
| Starting on 4×4 | 6/16 (37.5%) | 12/16 (75.0%) |
| Visible widget rejections | 15 | 8 |
| Attempts that reached the round limit | 1 | 2 |

d3-ultra’s six additional verifications all came from attempts starting on 4×4. That observed advantage comes from 16 such attempts per model, with different scenes and target mixes. The 3×3 totals were tied even though d3-ultra saw more challenges that replaced tiles. The layout-specific rates describe this run on this key, and the small subgroups leave the size of the difference uncertain. One d3-ultra attempt switched from 4×4 to 3×3 and remains in the 4×4 attempt group.

### Timing

| Metric | Jev | d3-ultra |
| --- | ---: | ---: |
| Median model round, 3×3 | 1.16 s | 1.74 s |
| Median model round, 4×4 | 1.48 s | 2.35 s |
| Median click to verification, successful attempts | 3.28 s | 4.26 s |
| 90th-percentile click to verification, successful attempts | 6.13 s | 8.65 s |
| Median wall time, all attempts | 4.54 s | 5.17 s |
| 90th-percentile wall time, all attempts | 8.83 s | 21.83 s |

Model-round time includes JPEG encoding and the parallel provider calls for a single scored grid. Click to verification starts immediately before the checkbox click and ends at the browser-observed backend verdict, so it includes Google’s challenge flow and any replacement rounds. Attempt wall time starts at navigation and includes unsuccessful attempts. A verified six-round d3-ultra attempt took 30.96 seconds from checkbox click to verdict.

## d3-ultra failure review

The review covers all 28 captured rounds from d3-ultra’s 10 unsuccessful attempts. Five of the ten started with tile replacement, including both attempts that reached the cap. Tile IDs below run left to right and then top to bottom, starting at 1. Scores are the model’s outputs, rounded for display. Google supplies puzzle-level feedback. Tile-level observations come from visual inspection of the saved grids and leave some rejection causes unresolved. Local screenshot paths below live inside the gitignored `reports/` directory.

| Attempt | Target and rounds | Recorded outcome and local artifact reference |
| --- | --- | --- |
| 10 | 4×4 stairs → 3×3 traffic lights, 2 rounds | Google supplied a fresh grid after the stairs submission and rejected the traffic-light round. A stair edge is visible in tile 15 of round 1, which averaged 38.2%. A distant signal candidate in tile 3 of round 2 averaged 50.0%, below the 60% cutoff. Local captures: `reports/decision-live-50/grids/vllm_jpeg/72a385ce-6a84-40df-961e-54343ed6ff7a/round-01.png`, `round-02.png` |
| 12 | 3×3 cars, 1 round | The widget asked to try again. Selected tile 6 shows a large bus-like vehicle and another vehicle partly visible alongside it. The screenshot leaves the specific error unclear. Local capture: `reports/decision-live-50/grids/vllm_jpeg/cf005264-7ef4-47dd-8d0f-e24068cad7c5/round-01.png` |
| 17 | 4×4 buses, 1 round | The widget asked to try again. The lower edge of the bus extends into tile 15. The whole-grid score was 17.3% and the scene-plus-crop score was 81.8%, averaging 49.6% against the 50% cutoff. Local capture: `reports/decision-live-50/grids/vllm_jpeg/2d7a1b83-85d6-4b32-85ac-2f000175479c/round-01.png` |
| 29 | 3×3 bus with replacement, 2 rounds | “Please select all matching images.” Tile 6 shows a bus-like vehicle cut off at the edge and stayed unchanged across both rounds. The whole-grid and crop scores were 99.96% and 5.07%, averaging 52.5%. The runner pressed Verify in round 2 with an empty selection. Local captures: `reports/decision-live-50/grids/vllm_jpeg/fd0cf9b0-5dce-4484-abbb-02108bc8a8a3/round-01.png`, `round-02.png` |
| 30 | 3×3 hydrants with replacement, 6 rounds | Six-round limit. After several hydrant replacements, the runner pressed Verify with no picks in round 5. Google served a new grid in round 6. The model picked three visible hydrants and the cap stopped the attempt before another grid evaluation. Local captures: `reports/decision-live-50/grids/vllm_jpeg/59c91be1-0335-426b-8091-379bbd666090/round-05.png`, `round-06.png` |
| 34 | 4×4 traffic lights, 1 round | “Please select all matching images.” The model picked only tile 6. The signal housing crosses the divider into nearby tiles. Tile 7 averaged 44.4%. Tile 8, which shows the pole and support arm, received 84.0% from the scene-plus-crop view. Local capture: `reports/decision-live-50/grids/vllm_jpeg/0f8f7319-b7bf-4e49-81de-a8c952d010e4/round-01.png` |
| 36 | 4×4 motorcycles, 1 round | The widget asked to try again. A rear tire extends into tile 9. The whole-grid view scored it at 2.4% and the scene-plus-crop view at 86.9%. The average was 44.6%, so tile 9 was skipped. Local capture: `reports/decision-live-50/grids/vllm_jpeg/71c61858-b9f8-4d72-8b0a-085216c47ed9/round-01.png` |
| 41 | 3×3 hydrants with replacement, 4 rounds | The widget asked to try again after several visible hydrants were replaced and the runner pressed Verify with no selections in round 4. The final grid leaves the specific cause unresolved. Local capture: `reports/decision-live-50/grids/vllm_jpeg/48368a6c-f4f7-46eb-8820-11e80b72ff4a/round-04.png` |
| 45 | 3×3 bus with replacement, 4 rounds | The widget asked to try again after an empty selection in round 4. The unchanged tile 6 shows a partially obscured vehicle. In the final round the views scored it at 96.3% and 12.0%, averaging 54.2%. Tile 9 also had a distant vehicle candidate and fell below 60%. The vehicle classes in these tiles are uncertain. Local capture: `reports/decision-live-50/grids/vllm_jpeg/a30c5ed1-54bf-4ed3-9ab0-7822d259d0d5/round-04.png` |
| 47 | 3×3 bicycles with replacement, 6 rounds | Six-round limit. Tile 1 was selected in round 2 and kept producing replacements. A visible bicycle by the storefront in round 6 scored 99.7% on average. The attempt reached the cap during the replacement sequence. Local capture: `reports/decision-live-50/grids/vllm_jpeg/9850d9b8-973d-40f0-800b-698282941524/round-06.png` |

### What the saved grids let us test

The strongest visual evidence concerns object parts at tile boundaries. In attempts 17 and 36, the close-up preserved useful evidence that got diluted by averaging. Attempt 34 shows the complementary risk of treating nearby supports as part of the target. On independent 3×3 photos, the disagreement can go the other direction, as in the bus candidate in attempt 29. Together these cases motivate measuring how the two views behave on human-labeled boundary fragments, distant objects, and adjacent structures before comparing any scoring change. A held-out set should include successful and unsuccessful grids because this review selected failures by design.

Replacement rounds also test how the runner manages a sequence. The limit stopped two d3-ultra attempts while valid-looking targets were still being selected. For further evaluation, record time spent on model inference, replacement waits, and verification separately on those sequences. That would distinguish changes in tile recognition from changes in how the runner accounts for rounds and time. The screenshots alone cannot establish whether either capped attempt would eventually have verified.

## Evidence and scope

The logs connect the screenshots, model replies, and Google outcomes by trial ID. The repository ignores `reports/` so raw CAPTCHA captures and run logs stay out of git history; local run artifacts from this benchmark reside at:

- Summary JSON: `reports/decision-live-50/summary.json`
- Trial log: `reports/decision-live-50/live.jsonl`
- Jev server log: `reports/decision-live-50/live-jev_jpeg-server.jsonl`
- d3-ultra server log: `reports/decision-live-50/live-vllm_jpeg-server.jsonl`
- Original screenshots: `reports/decision-live-50/grids/`
- Summary verifier script: `reports/decision-live-50/summarize.py`

When running with local artifacts present, rebuild the summary and check the artifacts against the logged SHA-256 hashes with:

```sh
python3 reports/decision-live-50/summarize.py --expect 50 --verify-files
```

The comparison applies to this browser build, local key, and observed mix of image challenges. Comparisons across keys, browser settings, model versions, or v3 score-based integrations remain open.
