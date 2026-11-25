//! vLLM/d3-ultra on Google's image grids: what was measured with it.
//!
//! Every number here belongs to this model as vLLM serves it. Another model scores the same
//! pictures differently, so it gets a file of its own and inherits nothing from this one.

use crate::{
    captcha::{Logistic, Model, Tuned},
    plan::{BoxRule, Efforts, GapRule, LOG_ODDS_3, Plan},
    providers::{Provider, VllmDecision},
};

pub static MODEL: Model = Model {
    name: "vLLM/d3-ultra",
    provider: Provider::Vllm,
    id: "vllm-sr/d3-ultra",
    tuned: Some(&TUNED),
};

pub static TUNED: Tuned = Tuned {
    vllm_decision: VllmDecision::Logistic,
    // Logistic rules on clipped log-odds. Coefficients were fitted on 320 model-labelled live
    // grids; a refit on human labels gave nearly the same shape, and a refit on the 339 graded
    // grids below was no better at any level.
    logistic_3x3: Some(Logistic {
        weights: [0.571457, 0.939705, -1.719964],
        cutoff: 0.5,
    }),
    logistic_4x4: Some([0.223152, 1.092908, 0.213385, 0.18743, -1.509533]),
    average_4x4: Plan {
        close_ups: BOTH_PASSES,
        select_from: 0.5,
        skip_below: Some(SKIP_BELOW),
        boxes: None,
        gap: None,
    },
    efforts: Some(EFFORTS),
};

/// d3-ultra's close-up scores shift with image scale. The steadier single pass is a copy of
/// the grid 506 pixels on its longer side: alone it scored 6 to 12 grids better than the
/// capture alone. Two passes score the capture as well and average the two.
const ONE_PASS: &[Option<u32>] = &[Some(506)];
const BOTH_PASSES: &[Option<u32>] = &[None, Some(506)];
/// Labelled grids without the target scored at most 0.50 and grids with it at least 0.88.
const SKIP_BELOW: f64 = 0.7;

/// The effort levels, as chosen on 2026-11-18.
///
/// Each level's settings are the best-scoring ones found for the requests it sends, in a sweep
/// of about 18 million settings over 339 graded 4 × 4 grids: 311 labelled square by square and
/// 28 live grids graded afterwards, with the user's stricter corrections of 2026-11-17 on 42 of
/// them. The scores below are on those same grids, so they flatter the settings: selecting
/// from 55% on one pass was also what other grids chose out of fold, and the gains at high and
/// at max are within a few grids. Tokens are for one round on a prompt that allows skipping.
///
/// | level  | tokens | right of 339 | of the 311 labelled |
/// |--------|--------|--------------|---------------------|
/// | low    | 6,475  | 243 (71.7%)  | 233 (74.9%)         |
/// | medium | 7,915  | 274 (80.8%)  | 250 (80.4%)         |
/// | high   | 12,351 | 281 (82.9%)  | 257 (82.6%)         |
/// | max    | 16,755 | 282 (83.2%)  | 258 (83.0%)         |
///
/// The settings in use before the levels (both passes, select from 50%, a box of confidence
/// 0.3 from either request) scored 271. Commit 815afc4 rebuilds them.
const EFFORTS: Efforts = Efforts {
    // One pass scores a little high, so squares are selected from 55%. With no boxes to check
    // a selection against, a square just over the line goes when only its close-up believes
    // in it. Selecting from 50% with no gap rule scored 233.
    low: Plan {
        close_ups: ONE_PASS,
        select_from: 0.55,
        skip_below: Some(SKIP_BELOW),
        boxes: None,
        gap: Some(GapRule {
            under: 0.6,
            more_than: 0.3,
        }),
    },
    // Selecting from 50% scored 269.
    medium: Plan {
        close_ups: ONE_PASS,
        select_from: 0.55,
        skip_below: Some(SKIP_BELOW),
        boxes: Some(BoxRule {
            exempt_from: LOG_ODDS_3,
            whole: Some(0.3),
            every: None,
            windows: None,
            cover: 0.01,
            inside: 0.0,
        }),
        gap: None,
    },
    // With no share of the box asked to lie inside the square it scored 279.
    high: Plan {
        close_ups: BOTH_PASSES,
        select_from: 0.5,
        skip_below: Some(SKIP_BELOW),
        boxes: Some(BoxRule {
            exempt_from: LOG_ODDS_3,
            whole: Some(0.3),
            every: None,
            windows: None,
            cover: 0.01,
            inside: 0.01,
        }),
        gap: None,
    },
    // A selection needs a box from both requests. Taking one from either, however the
    // thresholds were set, scored at most 278, under high. Faint whole-grid boxes count
    // because the window box is the stricter test; with 0.3 for both it scored 280.
    max: Plan {
        close_ups: BOTH_PASSES,
        select_from: 0.5,
        skip_below: Some(SKIP_BELOW),
        boxes: Some(BoxRule {
            exempt_from: LOG_ODDS_3,
            whole: Some(0.02),
            every: None,
            windows: Some(0.5),
            cover: 0.01,
            inside: 0.0,
        }),
        gap: None,
    },
};
