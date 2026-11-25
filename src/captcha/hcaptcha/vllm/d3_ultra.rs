//! vLLM/d3-ultra on hCaptcha's 3 × 3 grids: starting settings.
//!
//! Nothing here has been fitted: there are no labelled hCaptcha grids, so the two views of a
//! picture are averaged with equal weight. Everything fitted on Google's street photos is off,
//! because it assumes one photo cut into squares, and here every tile is a picture of its own.

use crate::{
    captcha::{Model, Tuned},
    plan::Plan,
    providers::{Provider, VllmDecision},
};

pub static MODEL: Model = Model {
    name: "vLLM/d3-ultra",
    provider: Provider::Vllm,
    id: "vllm-sr/d3-ultra",
    tuned: Some(&TUNED),
};

pub static TUNED: Tuned = Tuned {
    vllm_decision: VllmDecision::Average,
    logistic_3x3: None,
    logistic_4x4: None,
    average_4x4: Plan::PLAIN,
    efforts: None,
};
