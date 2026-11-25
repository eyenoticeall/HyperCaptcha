//! typesafe/Jev on Google's image grids.
//!
//! Jev is asked one question per tile on a single picture of the grid unless `--jev-vision`
//! says otherwise, so it has none of the settings of a model that is asked about every tile
//! twice. What was measured with it still sits in the solver: PNG pictures, and close-ups
//! enlarged without smoothing.

use crate::{captcha::Model, providers::Provider};

pub static MODEL: Model = Model {
    name: "typesafe/Jev",
    provider: Provider::Jev,
    id: "jev-latest",
    tuned: None,
};
