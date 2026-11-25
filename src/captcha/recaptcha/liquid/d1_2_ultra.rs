//! Liquid/d1.2-ultra on Google's image grids: what was measured with it.
//!
//! Every number here belongs to this model as Liquid serves it, and none comes from another
//! model's file. Only 4 × 4 grids have been measured: a 3 × 3 grid's two views are averaged
//! with equal weight.
//!
//! The model does not answer the same request the same way twice, so its answers were
//! recorded three times over, and every figure below is an average over the three passes.

use crate::{
    captcha::{Model, Tuned},
    plan::{BoxRule, Efforts, Plan},
    providers::{Provider, VllmDecision},
};

pub static MODEL: Model = Model {
    name: "Liquid/d1.2-ultra",
    provider: Provider::Liquid,
    id: "d1.2-ultra",
    tuned: Some(&TUNED),
};

pub static TUNED: Tuned = Tuned {
    vllm_decision: VllmDecision::Logistic,
    logistic_3x3: None,
    // Fitted on the 339 graded grids, all three passes, with the close-up pass at 506 pixels.
    // The whole grid counts for little because it is the weak view: by itself it answered
    // 117 grids right, the close-ups 221. With the two views averaged equally the model
    // scored 190, and with these weights 232, or 230 with weights fitted on other grids.
    logistic_4x4: Some([0.325707, 1.717633, 0.364472, 0.151428, -2.083722]),
    average_4x4: Plan::PLAIN,
    efforts: Some(EFFORTS),
};

/// The steadier single pass is the copy of the grid 506 pixels on its longer side: with the
/// capture alone the lowest level scored 220, not 232.
const ONE_PASS: &[Option<u32>] = &[Some(506)];
const BOTH_PASSES: &[Option<u32>] = &[None, Some(506)];
/// On the first pass, grids without the target scored at most 0.33 and grids with it at
/// least 0.69. Any threshold from 0 to 0.7 scored within a grid of this one.
const SKIP_BELOW: f64 = 0.5;
/// The model is asked once for a box around every instance, and a selected square scoring
/// under 90% needs one of confidence 0.3 or more. Confidences from 0.05 to 0.4 scored within
/// two grids of this at either level; 0.6 scored 2 to 3 lower and 0.8 6 to 8 lower.
///
/// Nine questions asking for one box each (the largest, the leftmost and so on), counted
/// from confidence 0.8, scored the same on the same three passes: 242.3 at medium and 245.0
/// at high, a difference of -0.3 (-7.0 to +6.3) and 0.0 (-7.3 to +7.7) grids. The one
/// question costs 1,369 tokens against 1,778.
const BOXES: BoxRule = BoxRule {
    exempt_from: 0.9,
    whole: None,
    every: Some(0.3),
    windows: None,
    cover: 0.01,
    inside: 0.0,
};

/// The effort levels, as chosen on 2026-11-18.
///
/// Chosen from a sweep of 166 million settings over three recorded passes of the 339 graded
/// 4 × 4 grids (311 labelled square by square, 28 graded live), under the labels as corrected
/// on 2026-11-17. Each level takes the middle of a run of settings that scored alike, not the
/// single best one, and uses no inside share, whole-grid minimum or gap rule: none of them
/// held up on grids left out of the choosing. The scores are on the grids the settings were
/// chosen on. Tokens are for one round on a prompt that allows skipping.
///
/// | level  | requests | tokens | right of 339 | of the 311 labelled |
/// |--------|----------|--------|--------------|---------------------|
/// | low    | 6        | 7,299  | 232.0 (68%)  | 218.3 (70%)         |
/// | medium | 7        | 8,668  | 242.0 (71%)  | 223.7 (72%)         |
/// | high   | 11       | 14,051 | 245.0 (72%)  | 225.7 (73%)         |
/// | max    | 11       | 14,051 | 245.0 (72%)  | 225.7 (73%)         |
///
/// The same level answers about 40 of the 339 grids differently from one pass to the next,
/// so differences of a few grids between settings are noise.
const EFFORTS: Efforts = Efforts {
    low: Plan {
        close_ups: ONE_PASS,
        select_from: 0.4,
        skip_below: Some(SKIP_BELOW),
        boxes: None,
        gap: None,
    },
    // The boxes are worth 10 grids here (2 to 18).
    medium: Plan {
        close_ups: ONE_PASS,
        select_from: 0.4,
        skip_below: Some(SKIP_BELOW),
        boxes: Some(BOXES),
        gap: None,
    },
    // The second pass is worth 3 grids (-4 to 10) for 5,383 tokens.
    high: HIGH,
    // Nothing beyond high. A box in a window around every square costs Liquid sixteen
    // requests and 5,076 tokens, and beside the nine-question boxes the best setting with it
    // scored 247.3 against 247.7 without, so it is not sent.
    max: HIGH,
};

const HIGH: Plan = Plan {
    close_ups: BOTH_PASSES,
    select_from: 0.35,
    skip_below: Some(SKIP_BELOW),
    boxes: Some(BOXES),
    gap: None,
};
