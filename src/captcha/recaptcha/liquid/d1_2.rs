//! Liquid/d1.2 on Google's image grids: what was measured with it.
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
    name: "Liquid/d1.2",
    provider: Provider::Liquid,
    id: "d1.2",
    tuned: Some(&TUNED),
};

pub static TUNED: Tuned = Tuned {
    vllm_decision: VllmDecision::Logistic,
    logistic_3x3: None,
    // Fitted on the 339 graded grids, all three passes, with the close-up pass at capture
    // size. The whole grid counts for almost nothing: by itself it answered 62 grids right,
    // the close-ups 199. With the two views averaged equally the model scored 132, and with
    // these weights 200, or 199 with weights fitted on other grids.
    logistic_4x4: Some([0.02249, 2.075111, 0.598377, 0.516936, -0.64329]),
    average_4x4: Plan::PLAIN,
    efforts: Some(EFFORTS),
};

/// One close-up pass, on the grid as captured. With boxes it scored 233 against 228 on a
/// copy 506 pixels wide, and it costs about 500 tokens less.
const ONE_PASS: &[Option<u32>] = &[None];
/// On the first pass, grids without the target scored at most 0.48 and grids with it at
/// least 0.49, all but one of them 0.65 or more. Without the question every level scored 8
/// or 9 grids lower.
const SKIP_BELOW: f64 = 0.5;

/// The effort levels, as chosen on 2026-11-18.
///
/// Chosen from a sweep of 166 million settings over three recorded passes of the 339 graded
/// 4 × 4 grids (311 labelled square by square, 28 graded live), under the labels as corrected
/// on 2026-11-17. Each level takes the middle of a run of settings that scored alike, not the
/// single best one, and uses no inside share, whole-grid minimum or gap rule. The scores are
/// on the grids the settings were chosen on. Tokens are for one round on a prompt that
/// allows skipping.
///
/// | level  | requests | tokens | right of 339 | of the 311 labelled |
/// |--------|----------|--------|--------------|---------------------|
/// | low    | 6        | 6,777  | 199.7 (59%)  | 187.7 (60%)         |
/// | medium | 7        | 8,146  | 236.0 (70%)  | 216.7 (70%)         |
/// | high   | 7        | 8,146  | 236.0 (70%)  | 216.7 (70%)         |
/// | max    | 7        | 8,146  | 236.0 (70%)  | 216.7 (70%)         |
///
/// The same level answers 64 to 79 of the 339 grids differently from one pass to the next,
/// so differences of a few grids between settings are noise.
const EFFORTS: Efforts = Efforts {
    low: Plan {
        close_ups: ONE_PASS,
        select_from: 0.5,
        skip_below: Some(SKIP_BELOW),
        boxes: None,
        gap: None,
    },
    medium: WITH_BOXES,
    // Nothing beyond medium. Beside the nine-question boxes a second close-up pass scored
    // 232.0 against 232.7 without it, and a box in a window around every square 233.3, so
    // neither is sent.
    high: WITH_BOXES,
    max: WITH_BOXES,
};

/// With boxes to check a selection against, far more squares can be selected: from 20%. The
/// model is asked once for a box around every instance, and a square under 90% needs one of
/// confidence 0.3 or more. That is worth 48 grids (34 to 62) over selecting from 20% with no
/// boxes. Cutoffs of 15% to 20% and confidences of 0.05 to 0.5 scored within two grids of this;
/// selecting from 50% scored 217.
///
/// Nine questions asking for one box each, counted from confidence 0.8, scored 232.7 on the
/// same three passes, so the one question is +3.3 grids (-3.0 to +10.0) for 1,369 tokens
/// against 1,778.
const WITH_BOXES: Plan = Plan {
    close_ups: ONE_PASS,
    select_from: 0.2,
    skip_below: Some(SKIP_BELOW),
    boxes: Some(BoxRule {
        exempt_from: 0.9,
        whole: None,
        every: Some(0.3),
        windows: None,
        cover: 0.01,
        inside: 0.0,
    }),
    gap: None,
};
