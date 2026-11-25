# Empirical Results & Benchmark Report

This document summarizes all verified offline and live measurements collected with HyperCaptcha across decision models (`vLLM/d3-ultra`, `Liquid/d1.2-ultra`, `Liquid/d1.2`, and `typesafe/Jev`), scoring rules, 4×4 box-veto effort levels, transport optimizations, and hCaptcha grid segmentation.

> **Read before comparing numbers:** Three human label revisions (`v0`, `v1`, and `v2`) exist across the experimental timeline, and live Google verification runs predate the 4×4 box rule and `--effort` ladder. Every table below explicitly tags its label version and evaluation scope. See [What Has Not Been Measured Live Yet](#7-what-has-not-been-measured-live-yet) for the exact boundaries of live evidence.

## Contents

1. [Dataset & Labeling Genealogy](#1-dataset--labeling-genealogy)
2. [Multi-View & Decision Rule Progression (`vLLM/d3-ultra`)](#2-multi-view--decision-rule-progression-vllmd3-ultra)
3. [Cross-Model 4×4 Effort-Level Comparison](#3-cross-model-44-effort-level-comparison)
4. [Key Negative & Surprising Findings](#4-key-negative--surprising-findings)
5. [Live reCAPTCHA Benchmarks](#5-live-recaptcha-benchmarks)
6. [hCaptcha Status](#6-hcaptcha-status)
7. [What Has Not Been Measured Live Yet](#7-what-has-not-been-measured-live-yet)

---

## 1. Dataset & Labeling Genealogy

### Saved Grid Pools

Offline comparisons replay saved PNG grids back through the lab server so that every model sees the exact crop geometry and JPEG/PNG encoding used during live operation.

| Pool / Subset | Grids | Groups | Composition & Labeling Method |
| --- | ---: | ---: | --- |
| **Initial saved population** | 582 | — | `143` static 3×3 + `202` older 4×4 + `237` dynamic 3×3. Not exhaustively labelled: evaluated via stratified human review of all `97` grids where `vLLM/d3-ultra` logistic and average rules disagreed (blind to picks) plus a random sample of `99` agreeing grids (`196` reviewed total; yields `123` labelled 4×4 grids and `18` dynamic 3×3 disagreement grids). |
| **First 4×4 set** | 123 | — | Human-labelled 4×4 subset drawn from the `202`-grid older 4×4 pool (`122` of the `186` combined first+`res` set started from model-pick pre-fills, including `10` from larger-size picks). |
| **Re-captured (`res`) 4×4 set** | 63 | — | Newly captured 4×4 grids combined with the first `123` to form the **`186`-grid** 4×4 set (`64` blind-labelled). |
| **Blind skip (`s5`) 4×4 set** | 125 | — | Grids from live 4×4 skip challenges, labelled completely blind (model picks and Google verdicts hidden). |
| **Blind-labelled 4×4 subset** | **189** | — | `64` blind grids from the `186` set + `125` blind skip grids. Best subset for evaluating baseline accuracy without pre-fill influence. |
| **Square-by-square labelled 4×4** | **311** | 136 | `123 + 63 + 125 = 311` labelled 4×4 grids grouped by challenge/trial (`136` groups). Fairer benchmark when comparing box vs. boxless levels. |
| **Live-graded 4×4 set** | **28** | 28 | Drawn from a `297`-attempt live capture (`337` 4×4 grids → `333` unique). These `28` are the exact grids where `d3-ultra`'s initial `0.3` box rule removed a pick (`29` picks across `26` attempts). Graded **non-blind** with the rule's selections pre-filled and removed squares flagged (median `9.6 s`/grid). |
| **Total graded 4×4 benchmark** | **339** | 164 | `311` labelled + `28` live-graded (`136 + 28 = 164` groups). |

### Grading Rule & Label Versions (`v0`, `v1`, `v2`)

A grid counts as **right** if and only if every `target` square is selected and no `empty` square is selected (`targets ⊆ picked ⊆ targets ∪ slivers`). Squares that the target only barely enters (**slivers**) are optional.

| Label Version | Date / Scope | How Created | `vLLM/d3-ultra` Anchors | Where Used |
| --- | --- | --- | --- | --- |
| **`v0`** *(original whole-grid)* | Initial 582-grid & 186-grid studies | Whole-grid human labels before single-square adjudication. | 186-grid 4×4: `122` (1-pass) → `129` (2-pass) → `133` (+ skip). | Older `143` static 3×3 / `202` 4×4 progression table. |
| **`v1`** *(earlier adjudicated)* | 2026-11-16 | Every square where pre-box `d3-ultra` disagreed with `v0` was re-shown one outlined square at a time mixed with agreed squares (`0/25` agreed on `186` and `0/24` agreed on `125` changed). On `186`: of `59` "wrong picks", `15` target, `21` sliver, `23` real errors (`12` background, `7` look-alikes, `4` riders); of `27` "missed", `14` sliver, `3` not target, `10` real misses (`133 → 159/186`). On `125` blind skip: of `51` "wrong picks", `6` target, `19` sliver, `26` real errors (`12` background, `9` look-alikes, `3` riders, `2` shadows); of `15` "missed", `13` stood (`92/125`). | Pre-box: **`251/311` (80.7%)**, `141/189` (74.6%) blind, `255/339` (75.2%).<br>Initial `0.3` either-request box rule: **`257/311` (82.6%)**, `282/339` (83.2%, `25/28` live). | All `typesafe/Jev` `311`-grid offline evaluations and two-model `d3-ultra` + `Jev` box studies. |
| **`v2`** *(corrected / stricter)* | 2026-11-17 | Reviewer strictly corrected `42` of `339` grids (`36` labelled + `6` live; `60` squares: `38` sliver→empty, `18` target→sliver, `3` target→empty, `1` empty→sliver). | Pre-box: **`238/339` (70.2%)** (`234/311` = 75.2% + `4/28`).<br>Initial `0.3` either-request box rule: **`271/339` (79.9%)** (`245/311` + `26/28`). | All shipped `vLLM/d3-ultra`, `Liquid/d1.2-ultra`, and `Liquid/d1.2` effort-level tables. |

> **Version collision warning:** The score `257/311` appears in **both** `v1` (the pre-level `0.3` either-request box rule) and `v2` (shipped `vLLM/d3-ultra` at `--effort high`). Never compare raw counts across label versions.

**Strictness sensitivity (`v2`):** `125` sliver squares remain across the `297` uncorrected grids. Re-scoring every remaining sliver as `empty` drops every `d3-ultra` level by roughly `50` grids (`low` `181`, `medium` `221`, `high` `227`, `max` `220` either-request / `229` both-requests) and drops `Liquid/d1.2-ultra` to `188` / `203` / `208` across `low` / `medium` / `high`, while preserving the ordering of effort levels.

### Repeatability & Label-Tilt Caveats

- **Same-day human repeatability:** `40` of the `186` grids were re-labelled blind from scratch on the same day with no model picks and no earlier answers shown. Pass 1 counted as right under Pass 2 labels on **`32/40` (80%)**, and Pass 2 counted as right under Pass 1 labels on **`30/40` (75%)** — an upper bound on human label repeatability. The sample held `24` grids the two-pass build got wrong and `16` it got right; under the second pass, `4/24` flipped right and `1/16` flipped wrong, while `19/24` wrong grids stayed wrong in both, `26/34` wrong picks were blank in both, and `7/9` missed targets were marked in both. Of `8` wrong picks scoring `≥ 0.9`, direct single-square review called `5` empty, `2` look-alikes, and `1` barely target.
- **Label tilt toward `vLLM/d3-ultra`:** Single-square adjudication in `v1` reviewed disagreements around `d3-ultra`'s picks only. Extra squares selected by `Liquid` or `Jev` were never single-square adjudicated, systematically understating those models relative to `d3-ultra`.
- **Selection bias in the `28` live-graded grids:** Those `28` grids were selected solely because `d3-ultra`'s initial `0.3` box rule dropped a square (and were graded non-blind with its picks visible). Pre-box `d3-ultra` scores `4/28` by construction, penalizing any boxless configuration; **`311` labelled** is the fairer comparison between box and boxless levels.
- **Model-generated labels are unreliable:** Model-made labels matched human review on only `~50%` of borderline grids and are not used to adjudicate close comparisons.

---

## 2. Multi-View & Decision Rule Progression (`vLLM/d3-ultra`)

### Population Progression on Older Saved Sets (`v0` Whole-Grid Labels)

Replayed through the Rust server with the harness's own crops and JPEG encoding:

| Grids (`v0` Labels) | Average (50/50) | Logistic | + 2 Close-Up Passes | + Skip Check (`< 0.7`) |
| --- | ---: | ---: | ---: | ---: |
| Static 3×3 (`143`) | 87% | 91% | not used | not used |
| 4×4 (`202`) | 59% | 69% | 74% | 76% |

- On the `18` dynamic 3×3 grids where logistic and average disagreed, logistic was right on **`15/18`** and average on **`2/18`**.
- On the `123` human-labelled 4×4 grids (`v0`), logistic fixed **`23`** grids that equal averaging got wrong and broke **`6`**.

### 4×4 Progression on `v2` Corrected Labels (`339` Graded / `311` Labelled)

| `vLLM/d3-ultra` Stage (`v2` Labels) | Requests (w/ skip) | Input Tokens | Right of 339 | Right of 311 | Wrong Picks | Missed Targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pre-box baseline (2 passes + skip, cutoff `0.50`) | 10 | 10,919 | 238 (70.2%) | 234 (75.2%) | 107 | 21 |
| Initial `0.3` either-request box rule (exempt `95.3%`) | 12 | 16,755 | 271 (79.9%) | 245 (78.8%) | 61 | 22 |
| Shipped `--effort low` (1 pass @ 506 px + gap rule) | 6 | 6,475 | 243 (71.7%) | 233 (74.9%) | 91 | 35 |
| Shipped `--effort medium` (1 pass @ 506 px + 9 whole boxes) | 7 | 7,915 | 274 (80.8%) | 250 (80.4%) | 42 | 46 |
| Shipped `--effort high` (2 passes + 9 whole boxes, inside ≥ 1%) | 11 | 12,351 | 281 (82.9%) | 257 (82.6%) | 39 | 34 |
| Shipped `--effort max` (2 passes + whole AND window boxes) | 12 | 16,755 | 282 (83.2%) | 258 (83.0%) | 39 | 33 |

### Component Breakdown

- **Shipped logistic weights** (fitted on `320` model-labelled live grids over clipped log-odds `clip(p, 1e-4, 1 - 1e-4)`):
  - **3×3:** `[0.571457, 0.939705, -1.719964]` (`whole`, `crop`, `intercept`; cutoff `0.50`).
  - **4×4:** `[0.223152, 1.092908, 0.213385, 0.18743, -1.509533]` (`whole`, `close_up`, `max_4_neighbor_whole`, `max_4_neighbor_close_up`, `intercept`; cutoff `0.50` at `high`/`max`, `0.55` at `low`/`medium`).
- **Two close-up passes** (capture size + `506` px copy, averaged in log-odds):
  - On `186` labelled 4×4 grids (`v0`), adding the `506` px pass moved accuracy from `122` to `129` (`+15` fixed, `−8` broken) and reduced wrong squares from `117` to `99`. Excluding `10` grids whose pre-fills came from larger-size picks (`176` grids): `119 → 123` (`+12/−8`; wrong squares `104 → 93`, holding in `95%` of bootstrap resamples). On the `64` blind grids of that set: `+4/−3`.
  - Second-pass sizes `440`, `506`, `560`, and `622` px each gained `3–7` grids on `186`; `330` px lost `5`; a third pass added nothing. On the `339` graded grids, a single `506` px pass alone beats a single capture pass by **`6–12` grids**.
  - In an interleaved `200`-attempt live A/B test, two-pass 4×4 rounds took a median of `1.13 s` vs. `0.93 s` for one pass and solved `22/35` vs. `17/33` challenges ending on 4×4 (within sampling noise).
- **Skip presence gate** (dedicated parallel request on `"If there are none, click skip"` prompts, `~300` input tokens, select nothing when `< 0.7`):
  - On `133` skippable grids of `186` (`v0`), the `14` no-target grids scored `≤ 0.57` and the `119` with-target grids scored `≥ 0.92`, moving `129 → 133` (`99 → 86` wrong squares, `0` broken; replay median `0.95 → 0.98 s`).
  - Checked on `174` additional live skippable grids: `161` scored `≥ 0.9` and `11` scored `< 0.7`, with four borderline grids at `0.51`, `0.63`, `0.79`, and `0.89` (blind human review confirmed `0.32`/`0.51`/`0.63` had no target and `0.79`/`0.89` contained the target). Across both sets combined, no-target grids scored **`≤ 0.63`** and with-target grids scored **`≥ 0.79`** — narrower headroom around `0.7` than the labelled set alone implied.
  - Removing the skip check costs **`−8` grids** on every `d3-ultra` effort level (`v2`, `339`).
- **Box veto progression** (`9` ordinal whole-grid boxes + `16` seeded 2×3/3×3 windows; veto-only; picks `≥ sigmoid(3.0) ≈ 95.26%` exempt):
  - *Gate 1 (`30` dev grids):* Deterministic; median edge drift `0.03` square across views (p90 `0.09–0.12`, max `0.46`); `0.40 s` wall / `94 ms` compute.
  - *Gate 2 (`123` dev grids, `513` target squares, `≥ 2%` cover, confidence `≥ 0.3 / 0.5`):* `9` ordinal boxes reached `94.2% / 89.7%` of targets (`9%` empty spill; `0.47 s`, `1,400` tokens); `16` seeded windows reached `98.1% / 96.9%` (`17–21%` spill; `0.90 s`, `4,400` tokens); `24` pairs reached `95.5% / 92.8%` (`0.71 s`, `5,400` tokens, `2` requests); ordinal + seeded reached `98.8% / 96.9%` (`107/113` grids had every target square reached).
  - *Initial `0.3` either-request rule (`v1`):* On the preregistered `188` blind grids (`125` skip + `63` `res`), `148 → 150` (`+6/−4`, `p ≈ 0.25`, short of the preregistered `≥ 5/0` or `≥ 7/1` bar). On all `311` labelled (`v1`), `251 → 257` (`+10/−4`; removed `13` empty, `12` slivers, `4` targets). On the `28` live-graded (`v1`), `4 → 25` (`+21/0`; removed `25` empty, `4` slivers, `0` targets; 95% upper bound on true target share `~10%`). Reference pre-box Google verdicts on that capture run: single-grid 4×4 passed `7/11` where the rule changed picks vs. `40/47` unchanged (`p = 0.12`); skip passed `2/15` vs. `14/49` (`p = 0.20`).
  - *Why the `95.26%` exemption matters:* Turning the exemption off (`v1`, `311` / `28`) drops `max` from `257/25` to `249/24` (`+13` missed targets to avoid `1` wrong pick) and drops `high` from `260/22` to `245/16` (`+43` missed targets to avoid `3` wrong picks).
- **Pixel, resize, and JPEG encoder sensitivity (`src/pillow.rs`):**
  - Separate Python JPEG re-encoding shifted 4×4 accuracy by several points (`62% → 72%` Python vs. `59% → 69%` / `58% → 67%` through the server). Across `7` image sizes on `63` labelled grids, roughly `1/3` of grids flipped right/wrong at some size.
  - Bilinear crop enlargement matched nearest-neighbor accuracy on human labels while reducing 4×4 request bytes by `~18%` (`typesafe/Jev` retains nearest-neighbor).
  - Splitting the `16` window boxes across `≥ 2` requests or omitting `4:2:0` chroma subsampling altered box-veto outcomes on `1–2` of `60` grids. `src/pillow.rs` reproduces Pillow's bilinear resize and libjpeg's baseline `4:2:0` quality-90 encoder byte-for-byte (**`5,661/5,661`** pictures across `333` live grids matched identically, reproducing every box and all `29` removals across the `333` grids).

---

## 3. Cross-Model 4×4 Effort-Level Comparison

### A. Shipped Models on `v2` Corrected Labels (`339` Graded / `311` Labelled)

Input tokens are for one 4×4 round with the skip question (`~300` fewer without skip). Liquid figures are **3-pass averages** using the shipped single `boxes` (every-instance) question. In-sample sweep scores flatter chosen parameters (especially `vLLM/d3-ultra` `high` and `max`), and label adjudication tilts toward `d3-ultra` (see [Section 1](#repeatability--label-tilt-caveats)).

| Model | Effort | Requests (w/ skip) | Input Tokens | Right of 339 | Right of 311 | Key 4×4 Decision Settings |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **`vLLM/d3-ultra`** *(pre-box ref)* | — | 10 | 10,919 | 238 (70.2%) | 234 (75.2%) | 2 passes, cutoff `0.50`, skip `< 0.7`, no boxes |
| **`vLLM/d3-ultra`** *(pre-level ref)* | — | 12 | 16,755 | 271 (79.9%) | 245 (78.8%) | 2 passes, cutoff `0.50`, either box `≥ 0.3`, exempt `95.3%` |
| **`vLLM/d3-ultra`** | `low` | 6 | 6,475 | **243 (71.7%)** | **233 (74.9%)** | 1 pass (`506` px), cutoff `0.55`, skip `< 0.7`, gap rule (under `0.60`, crop−whole `> 0.30`) |
| **`vLLM/d3-ultra`** | `medium` | 7 | 7,915 | **274 (80.8%)** | **250 (80.4%)** | 1 pass (`506` px), cutoff `0.55`, skip `< 0.7`, 9 whole boxes `≥ 0.3`, cover `≥ 1%`, exempt `95.3%` |
| **`vLLM/d3-ultra`** | `high` | 11 | 12,351 | **281 (82.9%)** | **257 (82.6%)** | 2 passes (`capture`+`506`), cutoff `0.50`, skip `< 0.7`, 9 whole boxes `≥ 0.3`, cover `≥ 1%`, inside `≥ 1%`, exempt `95.3%` |
| **`vLLM/d3-ultra`** *(default)* | `max` | 12 | 16,755 | **282 (83.2%)** | **258 (83.0%)** | 2 passes (`capture`+`506`), cutoff `0.50`, skip `< 0.7`, **both** whole `≥ 0.02` AND windows `≥ 0.5`, cover `≥ 1%`, exempt `95.3%` |
| **`Liquid/d1.2-ultra`** | `low` | 6 | 7,299 | **232.0 (68.4%)** | **218.3 (70.2%)** | 1 pass (`506` px), cutoff `0.40`, skip `< 0.5`, no boxes |
| **`Liquid/d1.2-ultra`** | `medium` | 7 | 8,668 | **242.0 (71.4%)** | **223.7 (71.9%)** | `low` + 1 `boxes` req `≥ 0.3`, cover `≥ 1%`, exempt `90%` |
| **`Liquid/d1.2-ultra`** | `high` / `max` | 11 | 14,051 | **245.0 (72.3%)** | **225.7 (72.6%)** | 2 passes (`capture`+`506`), cutoff `0.35`, skip `< 0.5`, `boxes` `≥ 0.3`, cover `≥ 1%`, exempt `90%` |
| **`Liquid/d1.2`** | `low` | 6 | 6,777 | **199.7 (58.9%)** | **187.7 (60.4%)** | 1 pass (`capture`), cutoff `0.50`, skip `< 0.5`, no boxes |
| **`Liquid/d1.2`** | `medium` / `high` / `max` | 7 | 8,146 | **236.0 (69.6%)** | **216.7 (69.7%)** | 1 pass (`capture`), cutoff `0.20`, skip `< 0.5`, `boxes` `≥ 0.3`, cover `≥ 1%`, exempt `90%` |

**Parity verification:** `cargo test --locked` replays recorded responses against every shipped effort plan and verifies 100% pick parity on committed image-free, label-free fixtures: `tests/fixtures/vllm_d3_ultra_4x4.json.gz` (`644` grids), `tests/fixtures/liquid_d1_2_ultra_4x4.json.gz` (`339` grids), and `tests/fixtures/liquid_d1_2_4x4.json.gz` (`339` grids). End-to-end live server checks against provider APIs confirmed exact score/pick parity on `20` `d3-ultra` rounds (`6,479–6,486` / `7,901–7,919` / `12,341–12,363` / `16,713–16,783` input tokens) and `34` Liquid rounds (`1,368–1,370` tokens for the `boxes` request).

### B. Estimated Provider Wait Timing by `vLLM/d3-ultra` Effort Level

Derived from request timestamps logged during a `333`-grid offline replay (not timed in live browser runs):

| Effort Level | Provider Queue + Compute Median | Provider p90 | Rounds Waiting on Boxes (of 333) | Slow-Uplink (`16 Mbps`) Wall Median |
| --- | ---: | ---: | ---: | ---: |
| `low` | 276 ms | — | 0 | 2,259 ms |
| `medium` | 276 ms | — | 1 (+83 ms) | 2,259 ms |
| `high` | 282 ms | 348 ms | 1 (+58 ms) | 2,451 ms |
| `max` | 556 ms | 598 ms | **194** (median +285 ms) | 2,516 ms (51 wait; +65 ms vs. `high`) |
| Pre-level either-request *(ref)* | 289 ms | — | 52 | — |

Because `max` requires both box requests to confirm any pick under `95.26%`, `194` of `333` rounds wait for the slowest request (the `16`-window box call, `~0.5 s` compute vs. `~0.26 s` for close-up batches). `--effort high` stays within `1` grid of `max` (`281` vs. `282`) while saving `4,404` tokens and avoiding window-box wait time.

### C. Liquid Model Findings, Non-Determinism & Single `boxes` Question

- **Fitted 4×4 weights matter most:**
  - `Liquid/d1.2-ultra`: `[0.325707, 1.717633, 0.364472, 0.151428, -2.083722]`. Equal-weight average scored `190/339`, whole-grid alone `117`, close-ups alone `221`, fitted OOF `230–232.7` (and transferring `d3-ultra`'s weights scored `232.3`).
  - `Liquid/d1.2`: `[0.02249, 2.075111, 0.598377, 0.516936, -0.64329]`. Equal-weight average scored `132/339`, whole-grid alone `62`, close-ups alone `199`, fitted OOF `199–200`.
  - Adding the whole-grid `boxes` veto gained **`+10` grids** (95% CI `+2` to `+18`) for `d1.2-ultra`, and **`+36` grids** over `low` (**`+48`**, CI `+34` to `+62`, over boxless at the `0.20` cutoff) for `d1.2`. Adding a second close-up pass gained `~3` grids (CI `−4` to `+10`) on `d1.2-ultra` for `+5,383` tokens and `0` grids on `d1.2`.
- **Stochastic pass-to-pass flip rates:** Unlike `vLLM/d3-ultra` (deterministic for identical bytes), Liquid models are non-deterministic. Between passes 1 and 2 over `339` grids:
  - `Liquid/d1.2-ultra` flipped `0.9–1.5%` of squares across `0.50` (`42–61` grids per view; `~40` grids per effort level differed pass-to-pass). Across `6` identical requests, whole-grid scores drifted by median `0.005` (`2/64` crossed `0.50`).
  - `Liquid/d1.2` flipped `2–4%` of squares across `0.50` (`86–154` grids per view; `64–79` grids per effort level; `35–79` grids across levels overall). Across `6` identical requests, whole-grid scores drifted by median `0.03` (`7/64` crossed `0.50`).
  - Effort settings were therefore selected from **plateau centers** across `166M` candidates per model, omitting `inside`, gap, and minimum whole-grid rules that did not generalize out-of-fold.
- **Single `boxes` question vs. 9 ordinal `box` questions:** `vLLM` rejects the multi-instance `boxes` question type; Liquid supports it (up to `16` boxes, averaging `1.9` boxes/grid on `d1.2-ultra` and `1.72` on `d1.2`) at **`1,369` vs. `1,778` input tokens (`−409` tokens)**. Across the same 3 passes (`339` grids), single `boxes` (`conf ≥ 0.3`, flat plateau `0.05–0.4`/`0.5`) matched or edged out `9` ordinal questions (`conf ≥ 0.8`):
  - `d1.2-ultra` `medium`: `242.0` vs. `242.3` (`−0.3`, CI `−7.0` to `+6.3`).
  - `d1.2-ultra` `high`: `245.0` vs. `245.0` (`0.0`, CI `−7.3` to `+7.7`).
  - `d1.2` `medium`: `236.0` vs. `232.7` (`+3.3`, CI `−3.0` to `+10.0`).
  - On `276` non-shrunk grids: `d3-ultra` `9`-box scored `227` vs. largest-only `214–217`, plain `210`, no-rule `186` (adding ordinal prompts `1` through `6` gave `215, 220, 220, 225, 226, 227`); `Liquid` `9`-box scored `204` vs. `boxes` `201–202`, largest-only `192`, plain `191`, no-rule `188`.
- **Wire & protocol constraints:** Liquid's API silently ignores named `assets` (billing `51` text-only tokens), so `src/liquid.rs` translates requests to positional `images` and rejects any reply billed under `1.5` tokens per `32×32` patch. Liquid also rejects gzip request bodies, caps each request at `8` images and `1` state (`--record` sends `27–28` requests / `~20,900` tokens), returns no server timings, and uses `429`/`5xx` retries (up to `3×` honoring `Retry-After`) instead of timer hedging (`40` concurrent requests succeeded). Liquid 3×3 grids remain unmeasured (averaged at `0.60` cutoff).

### D. `typesafe/Jev` (`jev-1.14.0`) on `v1` Earlier Labels (`311` Labelled / `28` Live)

> **Label version note:** All `typesafe/Jev` 4×4 offline experiments were evaluated on **`v1` earlier labels** (not `v2`). On `v1`, `vLLM/d3-ultra` pre-box scored **`251/311` (`4/28` live)** and the initial `0.3` either-request box rule scored **`257/311` (`25/28` live)**. `Jev` is non-deterministic (`1/24` resent window requests returned identical scores), and its extra selections at lower cutoffs were never single-square adjudicated.

| `typesafe/Jev` Configuration (`v1` Labels) | Requests | Input Tokens | Right of 311 | Wrong / Missed | Right of 28 Live | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Committed `--jev-vision scene-and-crop` (cutoff `0.50`) | 1 | 6,983 | **166 (53.4%)** | 73 / 151 | — | Median `522 ms`; no skip check |
| Committed `scene-and-crop` (OOF cutoff `0.32–0.33`) | 1 | 6,983 | **179 (57.6%)** | 139 / 75 | 9 | In-sample @ `0.32`: `181`; skips `10/23` no-target |
| Full pipeline, untuned (`d3-ultra` weights, `0.50`), no boxes | 10 | 21,331 | **132 (42.4%)** | 22 / 264 | 8 | Skip check (`0.7`) skipped `22/23` no-target, `0` with target |
| Full pipeline, untuned + Jev `0.3` box rule | 27 | ~34,700 | **128 (41.2%)** | 20 / 274 | 10 | Box rule removed `10` targets, `8` slivers, `2` empty |
| OOF: `d3-ultra` weights + refit cutoff (`0.21`) | 10 | 21,331 | **176 (56.6%)** | 71 / 126 | — | 136-group LOO CV |
| OOF: close-up alone (AUC `0.991` vs. whole-grid `0.919`) | 9 | — | **188 (60.5%)** | 128 / 60 | — | Cutoff `0.28–0.31` (plain equal average OOF: `144` [46%]) |
| OOF: close-up + neighbor refit (no boxes / + Jev `0.3` boxes) | 10 / 27 | — | **195 (62.7%) / 208 (66.9%)** | 103 / 60 *(no box)* | — | Boxes: `+14` fixed, `−1` broken |
| OOF: all 5 weights refit (no boxes / + Jev `0.3` boxes) | 10 / 27 | 21.3k / 34.7k | **199 (64.0%) / 214 (68.8%)** | 93 / 61 *(no box)* | 12 *(w/ box)* | Whole-grid weight is **negative** (`−0.546`); `d3-ultra` default − OOF 5-wt = **`+58`** (CI `42–74`) |

---

## 4. Key Negative & Surprising Findings

Several seemingly intuitive improvements failed offline validation and were intentionally omitted from shipped configs:

| Hypothesis Tested | Outcome | Key Evidence |
| --- | --- | --- |
| **Cross-model box & score ensembles** (`Liquid` + `d3-ultra`, `Jev` + `d3-ultra`) | **Rejected** | Replacing `d3-ultra` `high`'s 9 boxes with Liquid's 9 boxes dropped `281 → 273.0` (OOF `269.0`; passes `38.4%` vs. `26.9%` empty squares). Using Liquid's `boxes` dropped to `274.7` (OOF `270.7`, saving only `~60` tokens); combining both (`either`) gave `284.0` (OOF `283.3`, `+3` grids — all `3` in the non-blind `28` live set, `0` change on `311` at `257/311` — for `+1,370` tokens). Log-odds score blends of `Liquid/d1.2-ultra` + `d3-ultra` scored `255–274` (mildest `−7.3`, CI `−15.0` to `+0.3`); cross-fitted extra features scored `264.0–266.7` vs. `269.0` `d3-ultra`-only. Two-model `d3-ultra` + `Jev` boxes on `311` (`v1`, LOO) scored `255` (`p = 0.17`, missed `0.025`; best unplanned slice `258`, `p = 0.020`, fails Bonferroni `< 0.006`). |
| **Boxes as logistic features (or boxes alone) vs. veto** | **Rejected** | Cross-fitted on `339` (`v2`): boxes as logistic features scored **`266` vs. `271`** with veto for `d3-ultra` and **`243.3` vs. `244.3`** (`246.0` replacing whole-grid) for `Liquid/d1.2-ultra`. Boxes alone without tile scores achieved only `171/339` (`d3-ultra`) and `176/339` (`Liquid`); `d3-ultra` drew a counted box on **`87%`** of `23` no-target grids vs. **`8.7%`** for Liquid. |
| **Rescuing unselected squares using boxes** | **Rejected** | Gate 3 neighbor-box rescue broke more grids than it fixed. Across `d3-ultra` `high`'s unselected squares (`34` targets, `68` slivers, `3,571` empties), no score/box-cover band contains more targets than empties (e.g., score `30–50%` & cover `≥ 30%`: `1` target vs. `6` empties, even requiring Liquid box agreement). |
| **Window boxes on `Liquid` vs. `vLLM/d3-ultra`** | **Zero gain (Liquid) / Marginal (`d3-ultra`)** | On Liquid, `16` window requests cost `+5,076` tokens and added nothing (`d1.2-ultra`: `247.3` vs. `247.7` without; `d1.2`: `233.3` vs. `232.7`), so Liquid never sends them. On `d3-ultra` (`v2`), the initial either-request `max` (`271`) actually scored **`−8` grids below `high`** (`279`, CI `0` to `+16`; `+6` on the `42` corrected, `+2` on the `297` uncorrected). Requiring a box from *both* requests reaches `282` in-sample (`280` at `0.3/0.3`; OOF `277` vs. `280`) — only **`+1` grid** above shipped `high` (`281`) for `+4,404` tokens and `~2×` median provider compute+queue time (`556 ms` vs. `282 ms`). |
| **Refitting `d3-ultra`'s logistic weights on human labels** | **Worse than shipped weights** | Because binary cross-entropy minimizes per-square loss rather than whole-grid accuracy, refitting missed far more targets (`~52` vs. `23` on `v1`). On `311` (`v1`): default **`251`** vs. OOF 5-weight refit **`232`** (`235` at `0.5`), close-up+neighbor **`239`**, 2-pass-separate **`233`**. On `339` (`v2`, 10-fold group CV, cutoff & exemption tuned): `low` `234` vs. `233` (`+1`, CI `−7` to `+9`), `medium` `272` vs. `269` (`+3`, CI `−5` to `+11`), `medium` 2-pass `230` vs. `238` (`−8`), `high` **`272` vs. `279` (`−7`, CI `−13` to `−2`)**, `max` **`261` vs. `271` (`−10`, CI `−17` to `−3`)**. |
| **Other null results** | **Not adopted** | Prompt compression, `256` px crops, and close-ups without full-scene context failed the `60`-grid screen (`26–42/60` vs. `46/60`). Higher cover (`> 1%`), higher inside-share, whole-grid score floors, and gap rules on box levels hurt. `--device-scale-factor 2` (`389–778` px) gave no accuracy gain over a second pass resized from the default capture. Adding `"pedestrian crossing signals do not count"` to traffic-light prompts fixed `2/7` of `42` traffic-light grids (short of the preregistered `4`). `--use-hints` triggered `6` times in `550` live attempts and yielded only `1` pass. |

---

## 5. Live reCAPTCHA Benchmarks

> **Scope:** All live Google verification benchmarks below predate the 4×4 box rule and `--effort` ladder (except the `297`-attempt capture run that collected `333` grids for offline box replay).

### A. 50-Attempt Live `Jev` vs. `d3-ultra` (2026-11-15)

Headless Camoufox `158.0.1-beta.41`, live localhost reCAPTCHA v2 Enterprise checkbox key, JPEG q90, equal-weight `50/50` averaging (`60%` 3×3 / `50%` 4×4), `6`-round cap; `100/100` attempts opened an image challenge (`145` grids: `67` Jev, `78` `d3-ultra`). An earlier JPEG pilot passed `7/7` Jev and `8/8` `d3-ultra` (+`3` checkbox-only). Full per-attempt failure notes are preserved in [BENCHMARK_REPORT.md](../BENCHMARK_REPORT.md).

| Metric | `typesafe/Jev` (`jev-1.14.0`) | `vLLM/d3-ultra` |
| --- | ---: | ---: |
| **Google-verified attempts** | **34/50 (68.0%)** | **40/50 (80.0%)** |
| 95% Wilson CI | 54.2–79.2% | 67.0–88.8% |
| Starting on 3×3 (all) | 28/34 (82.4%) | 28/34 (82.4%) |
| — Static 3×3 (no tile replacement) | 27/30 (90.0%) | 26/27 (96.3%) |
| — Dynamic 3×3 (with tile replacement) | 1/4 (25.0%) | 2/7 (28.6%) |
| **Starting on 4×4** | **6/16 (37.5%)** | **12/16 (75.0%)** |
| Visible widget rejections / 6-round cap | 15 / 1 | 8 / 2 |
| Median model round (3×3 / 4×4) | 1.16 s / 1.48 s | 1.74 s / 2.35 s |
| Median click → verification (passes; p90) | 3.28 s (6.13 s) | 4.26 s (8.65 s) |
| Median wall time (all attempts; p90) | 4.54 s (8.83 s) | 5.17 s (21.83 s) |

- **`d3-ultra` failure taxonomy (`10` unsuccessful attempts, `28` rounds):** `5/10` began on dynamic 3×3 (`2` hit the `6`-round cap while valid targets kept replacing; `2` had an ambiguous cutoff bus where whole-grid vs. crop disagreed `96–100%` vs. `5–12%`; `1` unresolved hydrants), `4/10` on 4×4 boundary fragments diluted by `50/50` averaging (e.g., bus lower edge `17.3% / 81.8% → 49.6%`; motorcycle rear tire `2.4% / 86.9% → 44.6%`), and `1/10` on static 3×3 cars.
- **Weight-replay follow-up (`78` fresh `d3-ultra` rounds):** `50/50` reproduced `61/61` 3×3 and `17/17` 4×4 live picks; `60/40` on 3×3 changed `5/61` (all from `2` rejected attempts, keeping `100%` of verified 3×3 intact); `35/65` on 4×4 changed `6/17` (`3` rejected, `3` verified — mixed). Replay medians: `1.08 s` (3×3), `1.44 s` (4×4).

### B. 200-Attempt `--max-stages 10` Live Run (`d3-ultra` Logistic + 2-Pass + Skip, Pre-Box)

- **Overall:** `28` passed on checkbox alone (`24/40` in the first 40 attempts, `0/120` in the last 120), `3` never displayed a grid, **`169` challenged → `160` eventually solved (94.7%)** (`104` on the first answer), `7` hit the stage limit (all inside 4×4 skip sequences), and `2` ended on a runner error/timeout.
- **Latency:** Median click → verification **`2.7 s`**, p90 **`> 34.8 s`**. All `43` solves taking `≥ 20 s` included dynamic 3×3 rounds; all `112` non-dynamic solves completed in **`≤ 6.2 s`**.
- **Stage-counting note:** During this run, `--max-stages` counted each submitted grid (so two 5-grid skip sequences consumed all `10`). Current `scripts/chrome_lab.py` counts Google verdicts instead, which has not yet been re-measured live.

| First Challenge Shown | Challenged | Eventually Solved | First-Answer Pass | Median Time to Pass |
| --- | ---: | ---: | ---: | ---: |
| Static 3×3 | 79 | 79 (100%) | 75 | 2.3 s |
| 4×4 | 51 | 45 (88.2%) | 26 *(undercounts skip)* | 3.0 s |
| Dynamic 3×3 | 39 | 36 (92.3%) | 3 | 32.5 s |

- **Fast vs. legacy browser pacing** (`50–100 ms` poll, `0 s` initial pause vs. `200–300 ms` poll, `0.5 s` pause): Static 3×3 first-answer median dropped from **`2.6 s` to `1.7 s`** (`0.74 s` Google load + `0.51 s` model + `0.12 s` click + `0.16 s` verdict); 4×4 dropped from **`2.8 s` to `1.9 s`**. Solve rates (`82/86` vs. `78/83`; first-answer `50` vs. `54`) were statistically indistinguishable.
- **Per-verdict pass rates across this and two subsequent runs** (`≥ 60 ms` post-click gap): static 3×3 **`178/184` (96.7%)**, non-skip 4×4 **`82/94` (87.2%)**, dynamic 3×3 **`77/160` (48.1%)**, 4×4 skip **`17/65` (26.2%)** (5-grid skip **`11/47` [23.4%]**).
- **Later `297`-attempt capture run** (source of the `333` unique 4×4 grids for box replay): overall solve rate **`215/217` (99.1%)**; single-grid 4×4 passed `47/58` (`81.0%`).

### C. Dynamic 3×3 Rejection Gap, Server Pacing & Early-Tile Inference

- **72% human-right vs. 26% Google-accepted gap:** Across `824` live attempts, dynamic 3×3 challenges solved **`34/131` (26.0%)**, with almost every rejection returning generic `"Please try again"`. Blind human review of `36` dynamic challenges (`24` rejected + `12` solved, `123` rounds total) found that `d3-ultra` answered **every round correctly** in **`17/24` (70.8%)** of rejected challenges and **`9/12` (75.0%)** of solved challenges — a weighted **`~72%`** human-verified accuracy vs. **`26%`** Google acceptance (`166` targets picked, `5` missed, `6` wrong across `123` rounds). Tile accuracy is not the bottleneck on dynamic 3×3.
- **Server-controlled tile reveal timer:** After a tile click, the widget fetches the replacement image immediately (median **`23 ms`** after the last click across `56` logged tiles), then waits `1,000 ms + server_value` before displaying and enabling it. Across `225` logged waits, **`9`** fell in a `~1.04 s` band and **`216`** in a `~4.1 s` band (`72/73` multi-wait attempts remained in one band). Among passed attempts, **`37/45`** slow-tile passes carried `TOO_MUCH_TRAFFIC` in Google's reasons, compared to **`0/3`** fast-tile passes, **`0/112`** non-dynamic passes, and **`0/28`** checkbox passes. HyperCaptcha does not modify this timer.
- **Early-tile compositing (on by default in `scripts/chrome_lab.py`; disable with `--no-early-tiles`):** While the widget waits to reveal fetched replacement tiles, the runner sends them to `/model` (`tiles` field) so the server composites them onto the previous screenshot and queries the model ahead of reveal. It falls back to a fresh screenshot if a tile lands in an unexpected cell, fails to load, or leaves any tile score within **`0.25`** of the cutoff.
  - *Offline (`59` rounds):* Server-composited grids matched real-screenshot picks on **`57/59`** overall and **`53/53`** passing the `0.25` margin (Pillow-composited: `58/59` and `52/52`). Only `5/531` scores shifted by `> 0.03` (max `0.20`; one server-drawn tile shifted `0.13` vs. `0.45`, same side of cutoff).
  - *Live randomized `182`-attempt A/B* *(caveat: ran while the runner still composited with Pillow, and two runner copies overlapped by mistake)*: Median delay from tile reveal to next click fell from **`0.77 s` to `0.06 s`**; `84/90` early answers were used (`6` margin fallbacks); 3-round dynamic median fell from **`10.4 s` to `9.1 s`** and 4-round from **`15.4 s` to `13.6 s`**; overall solves were `82/83` vs. `83/85`.
  - *Server-composited follow-up (`30` attempts):* `30` early answers used, `7` fallbacks, `0` failed requests.

### D. 4×4 Skip-Challenge Multi-Grid Sequences

- Across `550` live attempts: non-skip 4×4 solved **`72/74` (97.3%)**, 2-grid skip solved **`8/15` (53.3%)**, and 5-grid skip solved **`1/29` (3.4%)**.
- Across subsequent runs (per judged sequence): all skip challenges passed **`17/65` (26.2%)** and 5-grid sequences passed **`11/47` (23.4%)**.
- On the `125` blind-labelled skip grids grouped into `23` complete 5-grid sequences (`11` passed vs. `12` rejected), passed sequences averaged **`1.00`** wrong grids and rejected sequences **`1.42`** (`p ≈ 0.25`; all 5 grids right in `5/11` passed vs. `2/12` rejected). As with dynamic 3×3, tile accuracy alone does not account for multi-grid skip rejections.

### E. Transport, Fan-Out, Compression, Warming & Hedged Retries (`vLLM/d3-ultra`)

- **Parallel request fan-out** (`--close-ups-per-request` / `--vllm-close-ups-per-request`, default `4`, range `2–9`; provider limit `16` concurrent/key):
  - Provider compute scales with crops per request: `~0.12 s` (2 crops), `~0.26 s` (4 crops), `~0.50 s` (8 crops). Scores are invariant to batch grouping whenever a request carries `≥ 2` crops — verified 100% identical across `258` labelled grids and `30/30` extra grids (`8` vs. `4` crops: `8,216` vs. `9,328` tokens, compute `505 ms` vs. `262 ms`, wall `1,310 ms` vs. `745 ms`).
  - Smaller batches (`3`/req: `0.77 s`; `2`/req across `3` keys: `0.83 s`) increased total upload bytes and tail latency compared to `4`/req (`0.73 s`).

| Round Measurement | Requests Before | Requests Now | Median Before | Median Now | p90 Before | p90 Now |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4×4 (`45` grids × 2, side-by-side alternating) | 5–6 | 9–10 | 1.01 s | **0.73 s** | > 1.53 s | **> 1.14 s** |
| 3×3 (`72` grids, replayed hours apart) | 2 | 4 | 0.91 s | **0.52 s** | > 1.02 s | **> 0.56 s** |
| Live pre-box run (`398` 3×3 / `240` 4×4 rounds) | — | 4 / 9–10 | — | **0.51 s / 0.70 s** | — | — / > 1.40 s |

- **Gzip & HTTP/1.1 vs. HTTP/2:** Fast gzip shrinks base64 JSON payloads by `~25%` on vLLM (rejected by Liquid). Multiplexing over HTTP/2 was no faster than one HTTP/1.1 connection per parallel request: a single shared HTTP/2 connection took `0.87 s` on its fastest quarter of rounds vs. `0.67 s` on HTTP/1.1, and `2–5` HTTP/2 connections gave no speedup while triggering key rate-limit refusals on `20–50%` of rounds. The client therefore stays on HTTP/1.1.
- **Connection warming** (vLLM; disable with `--cold-connections` / `--vllm-cold-connections`): Without warming, the first 4×4 round after startup took `~4 s` and the second `~1.5 s`. Both startup warming and `pool_max_idle_per_host` scale dynamically to `2 + box_reqs + passes * batches` (**`6` / `7` / `11` / `12`** connections for `low` / `medium` / `high` / `max`), priming each socket with one `544×544` noise-JPEG request (`~270 KB`, `394` input tokens). Connections stay warm for `300 s`, ping `GET /models` after `45 s` idle, and re-warm after `90 s` or a slow ping. A round `4 s` after start took `1.4 s` (half waiting on the initial warmup wave), and subsequent rounds — including after `20 s`, `50 s`, and `100 s` idle gaps — completed in **`0.63–0.75 s`** (one outlier at `1.5 s`).
- **Hedged retries** (vLLM; `HEDGE_AT = [1.5 s, 4.0 s, 8.0 s]`, rotated round-robin across `VLLM_SR_EXTRA_API_KEYS`): Roughly `1` in several hundred provider requests stalls `≥ 10 s` while sibling requests finish normally; on degraded networks `~5–6%` remained unanswered at `4 s`. Transient `429`/`5xx`/connection errors also retry once (`150 ms` or `Retry-After` capped at `10 s`).
- **Box-request upload wall cost:** Measured only on a slow **`16 Mbps`** uplink (`50` grids alternating pre-level box rule off/on): median **`+0.6 s`** over a `2.5 s` baseline. Wall cost on a fast uplink has not been measured live.

---

## 6. hCaptcha Status

- **Task scope:** Handles 3×3 grid challenges only (`9` independent pictures + one instruction). Non-grid tasks (drag, point-click, moving object) are skipped via the widget refresh button and counted separately as `passed_over` (up to `--max-refreshes 12`).
- **Canvas grid segmentation (`src/captcha/hcaptcha/grid.rs`):** Because hCaptcha renders pictures on a single `<canvas>` with per-tile nudging and tilt, the server locates the `9`-tile block on its panel, splits it into thirds, and validates bare-panel gutters (`GAP_SHARE ≥ 0.45`, `GAP_OVER_MIDDLE ≥ 0.20`). Earlier gap-cutting missed `2/6` captures; the block-and-thirds method succeeded on **`6/6` real screenshots** and rejects single-image non-grid tasks in unit tests. It composes a square PNG (tile side `64–256` px) and sends `4` parallel requests (`1` whole grid + `3` batches of `3` tiles).
- **Unfitted baseline:** **`0` human-labelled hCaptcha grids** exist. Shipped settings average the two views equally (`50/50`) at cutoff **`0.50`**; logistic weights, skip gate, second close-up pass, and box veto remain off. Example pictures shown beside the instruction are not sent to the model (their count is logged).
- **Hostname & live verification:** hCaptcha refuses `localhost` and `127.0.0.1`, so the lab serves on `hypercaptcha.localhost` by default (fallback `--hostname hypercaptcha.test`). Integration has been verified with hCaptcha's public test key only (which never serves an image challenge). **Live pass rate: unmeasured.**

---

## 7. What Has Not Been Measured Live Yet

To keep empirical claims honest, the following configurations have **not** been evaluated on live CAPTCHA service verdicts:

1. **4×4 box veto rule on live Google verdicts:** No live on/off A/B test exists for any box-veto rule (`9` whole-grid boxes, `boxes`, or `16` windows).
2. **`--effort` ladder (`low` / `medium` / `high` / `max`) live verdicts & fast-uplink latency:** All effort-level accuracy and wait-time numbers come from offline replays on saved grids (and the sole wall-time box measurement used a slow `16 Mbps` uplink).
3. **`Liquid/d1.2` and `Liquid/d1.2-ultra` live browser trials:** Both Liquid models have been measured only on offline saved grids and direct `/model` parity rounds, never against live Google reCAPTCHA or hCaptcha widgets.
4. **hCaptcha live pass rate & labelled grid accuracy:** `0` labelled hCaptcha grids and `0` live challenge trials have been recorded; only canvas segmentation (`6/6` captures) and public test-key integration have been verified.
5. **Verdict-counting `--max-stages` semantics:** The `200`-attempt live run counted each submitted grid toward `--max-stages`; the updated verdict-counting behavior in `scripts/chrome_lab.py` has not yet been measured live.
6. **Late hedged-retry tiers (`4 s` / `8 s`) under real network degradation:** Only the `1.5 s` hedge tier has been observed in live runs; the `4 s` and `8 s` retries are untested against a live degraded link.
