//! The CAPTCHA services the lab can host.
//!
//! Each service has its own module holding everything that is particular to it: how a token is
//! verified, how its challenge text is read, and its solver settings. A setting measured on one
//! service's pictures is not assumed to hold for another's, and one measured with one model is
//! not assumed to hold for another model. So a service's [`Profile`] holds what follows from
//! its challenges alone, and each model used on them has a file of its own with a [`Model`],
//! in a folder named after whoever serves it: `recaptcha/vllm/d3_ultra.rs` is vLLM/d3-ultra
//! on Google's grids.

pub mod hcaptcha;
pub mod recaptcha;

use anyhow::Result;
use clap::ValueEnum;
use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    plan::{Efforts, Plan},
    providers::{Provider, VllmDecision},
};

#[derive(Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Service {
    /// Google reCAPTCHA.
    Recaptcha,
    /// hCaptcha image grids.
    Hcaptcha,
}

impl Service {
    /// The solver settings measured on this service's challenges.
    pub fn profile(self) -> &'static Profile {
        match self {
            Self::Recaptcha => &recaptcha::PROFILE,
            Self::Hcaptcha => &hcaptcha::PROFILE,
        }
    }
}

/// The 3 × 3 logistic rule: weights on clipped log-odds for the whole grid and the individual
/// photo, then the intercept, and the probability from which a tile is selected.
pub struct Logistic {
    pub weights: [f64; 3],
    pub cutoff: f64,
}

/// The question a decision model is asked about one tile shown by itself.
pub struct PhotoQuestion {
    /// The request's state field that carries what to look for.
    pub state_key: &'static str,
    pub task: &'static str,
}

/// What holds for one service's image grids whichever model answers them.
pub struct Profile {
    /// Tiles along one side of the grids this service shows.
    pub layouts: &'static [u8],
    /// What to look for, read from the challenge text of a grid with `side` tiles a side.
    pub target: fn(prompt: &str, side: u8) -> Option<&str>,
    /// Whether the challenge text offers skipping when nothing matches.
    pub allows_skipping: fn(prompt: &str) -> bool,
    pub photo: PhotoQuestion,
    /// Cutoff for the average of two views of a 3 × 3 grid.
    pub average_cutoff_3x3: f64,
    /// The models used on this service's grids. A provider's first entry is its default.
    pub models: &'static [&'static Model],
}

impl Profile {
    /// The entry for the model a provider knows as `id`.
    pub fn model(&self, provider: Provider, id: &str) -> Option<&'static Model> {
        self.of(provider).find(|model| model.id == id)
    }

    /// The entry called `name`, such as "vLLM/d3-ultra", whatever its capitals.
    pub fn named(&self, name: &str) -> Option<&'static Model> {
        self.models
            .iter()
            .copied()
            .find(|model| model.name.eq_ignore_ascii_case(name))
    }

    /// A provider's entries, its default first.
    pub fn of(&self, provider: Provider) -> impl Iterator<Item = &'static Model> {
        self.models
            .iter()
            .copied()
            .filter(move |model| model.provider == provider)
    }
}

/// A model as one service's grids are answered with it.
pub struct Model {
    /// Who serves the model, then the model: "vLLM/d3-ultra". Two providers can serve models
    /// with similar names, so settings, logs and messages use this name.
    pub name: &'static str,
    pub provider: Provider,
    /// What the provider calls the model, as sent in a request.
    pub id: &'static str,
    /// What was measured with the model when it is asked about every tile twice, on the whole
    /// grid and on close-ups. None for a model that is asked another way.
    pub tuned: Option<&'static Tuned>,
}

/// What was measured with one model on one service's image grids, asking about every tile
/// twice.
pub struct Tuned {
    /// How the model's two views of a tile are combined unless a flag says otherwise.
    pub vllm_decision: VllmDecision,
    /// None when no rule has been fitted.
    pub logistic_3x3: Option<Logistic>,
    /// The 4 × 4 logistic rule's weights on clipped log-odds: whole grid, close-up, strongest
    /// edge-adjacent whole grid, strongest edge-adjacent close-up, intercept. A plan's
    /// `select_from` is its cutoff.
    pub logistic_4x4: Option<[f64; 5]>,
    /// How a 4 × 4 round is asked when its two views are averaged instead.
    pub average_4x4: Plan,
    /// How a 4 × 4 round is asked and decided under the logistic rule, at each effort level.
    /// None when no levels have been measured.
    pub efforts: Option<Efforts>,
}

/// A configured service: its keys and how it verifies a token.
pub enum Captcha {
    Recaptcha(recaptcha::Google),
    Hcaptcha(hcaptcha::Hcaptcha),
}

impl Captcha {
    pub fn service(&self) -> Service {
        match self {
            Self::Recaptcha(_) => Service::Recaptcha,
            Self::Hcaptcha(_) => Service::Hcaptcha,
        }
    }

    pub fn profile(&self) -> &'static Profile {
        self.service().profile()
    }

    /// The service's name in messages.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Recaptcha(_) => "Google",
            Self::Hcaptcha(_) => "hCaptcha",
        }
    }

    /// The name the lab page must be opened under. hCaptcha refuses pages on `localhost`.
    pub fn page_host(&self) -> &str {
        match self {
            Self::Recaptcha(_) => "localhost",
            Self::Hcaptcha(hcaptcha) => &hcaptcha.hostname,
        }
    }

    /// Whether the service returns a score for an interaction instead of showing a challenge.
    pub fn score_based(&self) -> bool {
        match self {
            Self::Recaptcha(google) => google.mode.score_based(),
            Self::Hcaptcha(_) => false,
        }
    }

    /// Whether the keys are the service's public test keys, which never show a challenge.
    pub fn test(&self) -> bool {
        match self {
            Self::Recaptcha(google) => google.test,
            Self::Hcaptcha(hcaptcha) => hcaptcha.test,
        }
    }

    pub fn verify(&self, token: &str, user_agent: Option<&str>) -> Result<Value> {
        Ok(match self {
            Self::Recaptcha(google) => serde_json::to_value(google.verify(token, user_agent)?)?,
            Self::Hcaptcha(hcaptcha) => serde_json::to_value(hcaptcha.verify(token)?)?,
        })
    }

    /// The service's part of the lab page's configuration.
    pub fn config(&self) -> Value {
        match self {
            Self::Recaptcha(google) => json!({
                "captcha": Service::Recaptcha,
                "mode": google.mode,
                "site_key": google.site_key,
                "enterprise": google.mode.enterprise(),
                "score_based": google.mode.score_based(),
                "test": google.test,
                "action": recaptcha::ACTION,
            }),
            Self::Hcaptcha(hcaptcha) => json!({
                "captcha": Service::Hcaptcha,
                "mode": "checkbox",
                "site_key": hcaptcha.site_key,
                "score_based": false,
                "test": hcaptcha.test,
            }),
        }
    }
}
