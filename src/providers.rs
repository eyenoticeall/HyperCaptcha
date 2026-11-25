use std::{
    collections::BTreeMap,
    env,
    io::{Cursor, Write},
    ops::Range,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use clap::{Args, ValueEnum};
use flate2::{Compression, write::GzEncoder};
use image::{ImageFormat, imageops};
use reqwest::{Url, blocking::Client};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tiny_skia::{FilterQuality, IntRect, Pixmap, PixmapPaint, Transform};

use crate::{
    boxes,
    captcha::{Model, PhotoQuestion, Profile, Tuned},
    liquid,
    plan::{BoxAsk, BoxRule, Effort, GapRule, Plan},
};

pub enum Input {
    Grid {
        layout: u8,
        prompt: String,
        png: Vec<u8>,
    },
    Trace(Value),
}

impl Input {
    pub fn layout(&self) -> Option<u8> {
        match self {
            Self::Grid { layout, .. } => Some(*layout),
            Self::Trace(_) => None,
        }
    }

    fn max_tile(&self) -> Option<u8> {
        self.layout().map(|side| side * side)
    }
}

#[derive(Args)]
pub struct ProviderOptions {
    /// Backend used to answer cases.
    #[arg(long, value_enum)]
    provider: Option<Provider>,

    /// A model by name, such as vLLM/d3-ultra or typesafe/Jev, which also chooses the provider, or
    /// a provider's own model ID with --provider. Defaults to the provider's first model.
    #[arg(long)]
    model: Option<String>,

    /// API root, including the version prefix. Overrides the provider's base URL.
    #[arg(long)]
    base_url: Option<String>,

    /// Request timeout in seconds.
    #[arg(long, default_value = "60", value_parser = clap::value_parser!(u64).range(1..))]
    timeout: u64,

    /// Maximum completion tokens for OpenAI requests, including reasoning tokens for chat.
    #[arg(long, default_value = "1024", value_parser = clap::value_parser!(u32).range(1..))]
    max_output_tokens: u32,

    /// Decision cutoff for every grid. Without it the cutoff is 0.5, or 0.6 for 3 × 3 grids with averaged scores, and on 4 × 4 grids the effort level sets its own: 0.55 at low and medium.
    #[arg(long, value_parser = crate::parse_probability)]
    tile_threshold: Option<f64>,

    /// vLLM whole-grid weight on 3 × 3 grids. The remainder goes to the crop scores.
    #[arg(long, value_parser = crate::parse_probability)]
    vllm_whole_grid_weight_3x3: Option<f64>,

    /// vLLM whole-grid weight on 4 × 4 grids. The remainder goes to the crop scores.
    #[arg(long, value_parser = crate::parse_probability)]
    vllm_whole_grid_weight_4x4: Option<f64>,

    /// How vLLM combines whole-grid and crop scores for each tile. Defaults to logistic, or average when a whole-grid weight is set.
    #[arg(long, value_enum)]
    vllm_decision: Option<VllmDecision>,

    /// Image input strategy for Jev checkbox challenges.
    #[arg(long, value_enum)]
    jev_vision: Option<JevVision>,

    /// Image format sent to a decision provider. Jev defaults to PNG and vLLM to JPEG.
    #[arg(long, value_enum)]
    decision_image_format: Option<DecisionImageFormat>,

    /// In Jev average mode, send 4 × 4 crop questions as two parallel batches of eight.
    #[arg(long)]
    jev_split_crops: bool,

    /// Shrink 3 × 3 grid screenshots whose longer side exceeds this many pixels before the model sees them.
    #[arg(long, value_parser = clap::value_parser!(u32).range(280..=1568))]
    grid_max_side_3x3: Option<u32>,

    /// Shrink 4 × 4 grid screenshots whose longer side exceeds this many pixels before the model sees them.
    #[arg(long, value_parser = clap::value_parser!(u32).range(280..=1568))]
    grid_max_side_4x4: Option<u32>,

    /// How much vLLM asks about a 4 × 4 grid. Each level sends more requests than the one below it and costs more tokens: low sends the whole grid, one close-up pass and the skip question; medium adds boxes drawn on the whole grid; high adds a second close-up pass; max adds a box drawn around every square. Each level decides with the settings measured for it. Defaults to max. Applies to the logistic decision rule.
    #[arg(long, value_enum)]
    effort: Option<Effort>,

    /// Wait for every box reply in a 4 × 4 round and log every box, instead of answering as soon as the selection is settled. For checking the box rule, at an effort level that has one.
    #[arg(long)]
    vllm_box_wait: bool,

    /// The most close-ups vLLM puts in one request, from 2 to 9. Defaults to 4. Smaller groups mean more parallel requests with the same scores, but a larger upload, and the provider refuses requests beyond 16 at once per key.
    #[arg(long, value_parser = clap::value_parser!(u8).range(2..=9))]
    vllm_close_ups_per_request: Option<u8>,

    /// Do not open vLLM's connections ahead of a round or keep them open between rounds.
    #[arg(long)]
    vllm_cold_connections: bool,

    /// Ask a 4 × 4 round everything any effort level asks, wait for every reply and log it, and decide as a plain round does: both close-up passes, the skip question and both box requests, none of which changes the answer. For recording a model's answers so that settings can be measured for it.
    #[arg(long)]
    record_4x4: bool,
}

// d3-ultra's compute time grows with the close-ups in a request, and its scores are the same
// however the close-ups are grouped, as long as no request holds just one. Fours gave the
// fastest 4 × 4 rounds: smaller groups compute sooner, but every request repeats the scene, and
// the larger upload made slow rounds more frequent. A 4 × 4 round at the highest effort level is
// then twelve requests at once, counting the presence question and the two box requests (the
// provider allows each key 16 at once), and a 3 × 3 round sends three groups of three.
// A request carries at most 16 images, and each 4 × 4 close-up request also carries the scene.
/// The longest a provider's `Retry-After` is obeyed, in seconds. A round is not worth more.
const MOST_RETRY_AFTER: u64 = 10;
const VLLM_CLOSE_UPS_PER_REQUEST: u8 = 4;
const MAX_CLOSE_UPS_WITH_SCENE: u8 = 8;

#[derive(Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Provider {
    OpenaiChat,
    OpenaiCompletions,
    Jev,
    Vllm,
    Liquid,
}

impl Provider {
    pub fn is_decision(self) -> bool {
        matches!(self, Self::Jev | Self::Vllm | Self::Liquid)
    }

    /// Whether the provider's models are asked about every tile twice, on the whole grid and
    /// on close-ups, and so take the settings and flags of that way of asking.
    pub fn asks_twice(self) -> bool {
        matches!(self, Self::Vllm | Self::Liquid)
    }

    /// Whether the provider's models answer one question with a box around every instance.
    /// vLLM refuses that question type.
    pub fn draws_every_instance(self) -> bool {
        self == Self::Liquid
    }

    /// The provider as it is written in a model's name: "vLLM/d3-ultra".
    pub fn label(self) -> &'static str {
        match self {
            Self::OpenaiChat | Self::OpenaiCompletions => "OpenAI",
            Self::Jev => "typesafe",
            Self::Vllm => "vLLM",
            Self::Liquid => "Liquid",
        }
    }
}

/// Rules for combining d3-ultra's whole-grid and crop answers.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum VllmDecision {
    /// Weighted average of the two probabilities.
    Average,
    /// Logistic model on log-odds, fitted to labelled live grids. 4 × 4 adds neighbouring-square evidence.
    Logistic,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum JevVision {
    WholeGrid,
    SceneAndCrop,
    Average,
}

pub struct Backend {
    /// The settings of the CAPTCHA service whose grids this backend answers.
    profile: &'static Profile,
    pub provider: Provider,
    /// What the provider calls the model.
    pub model: String,
    /// The model's name with its provider: "vLLM/d3-ultra".
    pub name: String,
    pub jev_vision: Option<JevVision>,
    decision_image_format: DecisionImageFormat,
    jev_split_crops: bool,
    grid_max_side: [Option<u32>; 2],
    /// What was measured with this model on the service's grids. vLLM only.
    tuned: Option<&'static Tuned>,
    /// The effort level of a 4 × 4 round, where the model has levels and the logistic rule decides.
    pub effort: Option<Effort>,
    /// How a 4 × 4 round is asked and decided.
    plan: Plan,
    vllm_close_ups_per_request: Option<u8>,
    vllm_box_wait: bool,
    warm: Option<Arc<Warm>>,
    tile_threshold_override: Option<f64>,
    vllm_whole_grid_weights: [f64; 2],
    pub vllm_decision: Option<VllmDecision>,
    client: Client,
    url: Url,
    /// One key, or several for vLLM. Requests take them in turn, because the provider
    /// limits how many requests each key may have in flight.
    keys: Arc<Vec<String>>,
    next_key: AtomicUsize,
    max_output_tokens: u32,
    verified_vision: OnceLock<VisionCheck>,
}

#[derive(Clone, Serialize)]
pub struct VisionCheck {
    passed: bool,
    model: String,
    endpoint: String,
    controls: Vec<VisionControl>,
}

#[derive(Clone, Serialize)]
struct VisionControl {
    expected_red: bool,
    probability_red: Option<f64>,
    resolved_model: Option<String>,
    usage: Option<Value>,
    latency_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_meta: Option<Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum Prediction {
    Tiles(Vec<u8>),
    LegitimacyScore(f64),
}

pub enum Output {
    Prediction(Prediction),
    Invalid(String),
    Refusal,
}

pub struct Reply {
    pub output: Output,
    pub resolved_model: Option<String>,
    pub usage: Option<Value>,
    pub tile_probabilities: Option<BTreeMap<u8, f64>>,
    pub decision_components: Option<Value>,
    pub provider_meta: Option<Value>,
}

impl Backend {
    pub fn new(mut options: ProviderOptions, profile: &'static Profile) -> Result<Option<Self>> {
        // A model's name says who serves it, so it can stand in for --provider.
        let named = options
            .model
            .as_deref()
            .and_then(|name| profile.named(name));
        if let (Some(named), Some(provider)) = (named, options.provider) {
            ensure!(
                named.provider == provider,
                "{} is not served by the --provider given",
                named.name
            );
        }
        options.provider = options.provider.or(named.map(|named| named.provider));
        ensure!(
            options.provider.is_some_and(Provider::asks_twice)
                || (options.vllm_whole_grid_weight_3x3.is_none()
                    && options.vllm_whole_grid_weight_4x4.is_none()),
            "--vllm-whole-grid-weight-3x3 and --vllm-whole-grid-weight-4x4 require --provider vllm or liquid"
        );
        ensure!(
            options.vllm_decision.is_none() || options.provider.is_some_and(Provider::asks_twice),
            "--vllm-decision requires --provider vllm"
        );
        ensure!(
            options.vllm_decision != Some(VllmDecision::Logistic)
                || (options.vllm_whole_grid_weight_3x3.is_none()
                    && options.vllm_whole_grid_weight_4x4.is_none()),
            "--vllm-whole-grid-weight-* apply to --vllm-decision average"
        );
        ensure!(
            options.effort.is_none() || options.provider.is_some_and(Provider::asks_twice),
            "--effort requires --provider vllm or liquid"
        );
        ensure!(
            !options.vllm_box_wait || options.provider.is_some_and(Provider::asks_twice),
            "--vllm-box-wait requires --provider vllm or liquid"
        );
        ensure!(
            options.vllm_close_ups_per_request.is_none()
                || options.provider.is_some_and(Provider::asks_twice),
            "--vllm-close-ups-per-request requires --provider vllm or liquid"
        );
        ensure!(
            !options.record_4x4 || options.provider.is_some_and(Provider::asks_twice),
            "--record-4x4 requires --provider vllm or liquid"
        );
        ensure!(
            !options.record_4x4 || options.effort.is_none(),
            "--record-4x4 asks everything an effort level asks, so it takes no --effort"
        );
        ensure!(
            !options.vllm_cold_connections || options.provider.is_some_and(Provider::asks_twice),
            "--vllm-cold-connections requires --provider vllm or liquid"
        );
        ensure!(
            options.jev_vision.is_none() || options.provider == Some(Provider::Jev),
            "--jev-vision requires --provider jev"
        );
        ensure!(
            options.decision_image_format.is_none()
                || options.provider.is_some_and(Provider::is_decision),
            "--decision-image-format requires --provider jev or vllm"
        );
        ensure!(
            !options.jev_split_crops || options.jev_vision == Some(JevVision::Average),
            "--jev-split-crops requires --jev-vision average"
        );
        let Some(provider) = options.provider else {
            ensure!(
                options.model.is_none() && options.base_url.is_none(),
                "--model and --base-url require --provider, or a model named with its provider such as {}",
                profile
                    .models
                    .first()
                    .map_or("vLLM/d3-ultra", |model| model.name)
            );
            return Ok(None);
        };
        ensure!(
            profile.layouts.contains(&4) || provider.asks_twice(),
            "This CAPTCHA service's grids are answered with --provider vllm"
        );

        let (default_url, env_url, endpoint, key_names) = match provider {
            Provider::OpenaiChat => (
                "https://api.openai.com/v1",
                "OPENAI_BASE_URL",
                "chat/completions",
                &["OPENAI_API_KEY"][..],
            ),
            Provider::OpenaiCompletions => (
                "https://api.openai.com/v1",
                "OPENAI_BASE_URL",
                "completions",
                &["OPENAI_API_KEY"][..],
            ),
            Provider::Jev => (
                "https://api.typesafe.ai/v1",
                "JEV_BASE_URL",
                "systemone",
                &["TYPESAFE_API_KEY", "JEV_API_KEY"][..],
            ),
            Provider::Vllm => (
                "https://app.vllm-sr.ai/v1",
                "VLLM_SR_BASE_URL",
                "systemone",
                &["VLLM_SR_API_KEY"][..],
            ),
            Provider::Liquid => (
                "https://api.liquid.ai/decisions/v1",
                "LIQUID_BASE_URL",
                "systemone",
                &["LIQUID_API_KEY"][..],
            ),
        };

        let model = named
            .map(|named| named.id.to_owned())
            .or(options.model)
            .or_else(|| profile.of(provider).next().map(|first| first.id.to_owned()))
            .filter(|model| !model.trim().is_empty())
            .with_context(|| {
                if provider.is_decision() {
                    format!(
                        "this CAPTCHA service has no settings for a {} model",
                        provider.label()
                    )
                } else {
                    "--model must be nonempty and is required for OpenAI providers".into()
                }
            })?;
        let known: Option<&'static Model> = profile.model(provider, &model);
        let name = known.map_or_else(
            || format!("{}/{model}", provider.label()),
            |known| known.name.to_owned(),
        );
        let base = options
            .base_url
            .or_else(|| configured_env(env_url))
            .unwrap_or_else(|| default_url.into());
        let url = endpoint_url(&base, endpoint)?;
        let key = key_names
            .iter()
            .find_map(|name| configured_env(name))
            .with_context(|| format!("set {}", key_names.join(" or ")))?;
        let mut keys = vec![key];
        if provider == Provider::Vllm {
            for extra in configured_env("VLLM_SR_EXTRA_API_KEYS")
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|extra| !extra.is_empty())
            {
                if !keys.iter().any(|known| known == extra) {
                    keys.push(extra.to_owned());
                }
            }
        }
        let build_client = |most_idle: Option<usize>| {
            let builder = Client::builder()
                .user_agent(concat!("HyperCaptcha/", env!("CARGO_PKG_VERSION")))
                .timeout(Duration::from_secs(options.timeout))
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                // A round sends many requests at once and the next round reuses their connections,
                // so idle ones are kept instead of being closed after reqwest's default 90 seconds.
                .pool_idle_timeout(None)
                .tcp_keepalive(Duration::from_secs(30))
                .tcp_nodelay(true);
            match most_idle {
                Some(most) => builder.pool_max_idle_per_host(most),
                None => builder,
            }
            .build()
        };
        let client = build_client(None)?;
        // Settings are measured model by model, so a vLLM model without a file of its own has
        // none to run with.
        let tuned = provider
            .asks_twice()
            .then(|| {
                known.and_then(|known| known.tuned).with_context(|| {
                    let measured: Vec<_> = profile
                        .of(provider)
                        .filter(|model| model.tuned.is_some())
                        .map(|model| model.name)
                        .collect();
                    format!(
                        "this CAPTCHA service has no settings measured with {name}; it has them for {}",
                        measured.join(", ")
                    )
                })
            })
            .transpose()?;
        let vllm_decision = tuned.map(|tuned| {
            options.vllm_decision.unwrap_or(
                if options.vllm_whole_grid_weight_3x3.is_some()
                    || options.vllm_whole_grid_weight_4x4.is_some()
                {
                    VllmDecision::Average
                } else {
                    tuned.vllm_decision
                },
            )
        });
        // The effort levels were measured on the logistic rule's scores. Averaged scores keep
        // one plan, and a provider with nothing measured sends a plain round.
        let (effort, plan) = match (tuned, vllm_decision) {
            _ if options.record_4x4 => (None, Plan::record(provider.draws_every_instance())),
            (Some(tuned), Some(VllmDecision::Logistic)) => match &tuned.efforts {
                Some(efforts) => {
                    let effort = options.effort.unwrap_or_default();
                    (Some(effort), efforts.plan(effort))
                }
                None => {
                    ensure!(
                        options.effort.is_none(),
                        "no effort levels have been measured with {name} on this CAPTCHA service"
                    );
                    (None, Plan::PLAIN)
                }
            },
            (Some(tuned), _) => {
                ensure!(
                    options.effort.is_none(),
                    "--effort applies to --vllm-decision logistic"
                );
                (None, tuned.average_4x4)
            }
            (None, _) => (None, Plan::PLAIN),
        };

        let mut backend = Self {
            profile,
            provider,
            model,
            name,
            tile_threshold_override: options.tile_threshold,
            vllm_whole_grid_weights: [
                options.vllm_whole_grid_weight_3x3.unwrap_or(0.5),
                options.vllm_whole_grid_weight_4x4.unwrap_or(0.5),
            ],
            vllm_decision,
            jev_vision: (provider == Provider::Jev)
                .then(|| options.jev_vision.unwrap_or(JevVision::WholeGrid)),
            decision_image_format: options.decision_image_format.unwrap_or(
                if provider.asks_twice() {
                    DecisionImageFormat::Jpeg
                } else {
                    DecisionImageFormat::Png
                },
            ),
            jev_split_crops: options.jev_split_crops,
            grid_max_side: [options.grid_max_side_3x3, options.grid_max_side_4x4],
            tuned,
            effort,
            plan,
            vllm_close_ups_per_request: options.vllm_close_ups_per_request,
            vllm_box_wait: options.vllm_box_wait || options.record_4x4,
            client,
            url,
            keys: Arc::new(keys),
            next_key: AtomicUsize::new(0),
            max_output_tokens: options.max_output_tokens,
            verified_vision: OnceLock::new(),
            warm: None,
        };
        ensure!(
            backend.vllm_decision != Some(VllmDecision::Logistic)
                || profile.layouts.iter().any(|&side| match side {
                    4 => tuned.is_some_and(|tuned| tuned.logistic_4x4.is_some()),
                    _ => tuned.is_some_and(|tuned| tuned.logistic_3x3.is_some()),
                }),
            "--vllm-decision logistic has not been fitted for this CAPTCHA service"
        );
        ensure!(
            provider.draws_every_instance()
                || backend.plan.boxes.is_none_or(|rule| rule.every.is_none()),
            "{} models do not answer the box question that asks for every instance",
            provider.label()
        );
        ensure!(
            !backend.vllm_box_wait || backend.plan.boxes.is_some(),
            "--vllm-box-wait needs a box rule: --effort medium, high or max with the logistic decision"
        );
        if provider == Provider::Vllm {
            // The busiest round is a 4 × 4 grid with the presence question and its box requests.
            let passes = backend.plan.close_ups.len();
            let boxes = usize::from(backend.plan.boxes.map_or(0, |rule| rule.requests()));
            let connections = 2 + boxes + passes * backend.close_up_batches(4).len();
            // A connection that has not yet carried an upload sends a round's request several
            // round trips slower. Idle connections beyond the warmed set are closed, so a
            // duplicate's new connection cannot displace them.
            backend.client = build_client(Some(connections))?;
            if !options.vllm_cold_connections {
                let (wake, woken) = mpsc::channel();
                let warm = Arc::new(Warm {
                    until: Mutex::new(None),
                    wake,
                    rounds_in_flight: AtomicUsize::new(0),
                    last_full_round: Mutex::new(None),
                    keeper_busy: AtomicBool::new(false),
                });
                let keeper = Keeper {
                    practice: backend.practice_upload()?,
                    models: endpoint_url(&base, "models")?,
                    connections,
                    warm: Arc::clone(&warm),
                };
                thread::spawn(move || keeper.run(woken));
                backend.warm = Some(warm);
            }
        }
        Ok(Some(backend))
    }

    pub fn predict(&self, input: &Input) -> Result<Reply> {
        let vision = self.jev_image_strategy(input);
        let threshold = self.tile_threshold(input.layout());
        if input.layout().is_some()
            && (vision == Some(JevVision::Average) || self.provider.asks_twice())
        {
            return self.predict_decision_grid(input, threshold.expect("decision tile threshold"));
        }
        let (body, contains_jev_image) = match self.provider {
            Provider::OpenaiChat => (
                chat_request(&self.model, input, self.max_output_tokens)?,
                false,
            ),
            Provider::OpenaiCompletions => (
                json!({
                    "model": self.model,
                    "prompt": text_prompt(input)?,
                    "max_tokens": self.max_output_tokens,
                }),
                false,
            ),
            Provider::Jev | Provider::Vllm | Provider::Liquid => (
                jev_request(
                    &self.model,
                    input,
                    vision,
                    self.decision_image_format,
                    self.profile,
                )?,
                self.provider == Provider::Jev && input.max_tile().is_some(),
            ),
        };

        let response = self.send(&body, contains_jev_image)?;
        Ok(response.into_reply(self.provider, input, threshold))
    }

    /// Shrinks a grid screenshot to the configured size for its layout, keeping its aspect ratio.
    /// Returns the PNG to use and its final width and height.
    pub fn prepare_grid(&self, layout: u8, png: Vec<u8>) -> Result<(Vec<u8>, [u32; 2])> {
        let image = image::load_from_memory_with_format(&png, ImageFormat::Png)
            .context("decode the grid screenshot")?;
        let (width, height) = (image.width(), image.height());
        let max_side = match layout {
            3 => self.grid_max_side[0],
            4 => self.grid_max_side[1],
            _ => None,
        };
        let Some(max_side) = max_side.filter(|&max| width.max(height) > max) else {
            return Ok((png, [width, height]));
        };
        resized_png(&image, max_side)
    }

    /// How vLLM asks and decides a 4 × 4 round.
    pub fn plan_4x4(&self) -> Option<Plan> {
        (self.provider.asks_twice() && self.profile.layouts.contains(&4)).then_some(self.plan)
    }

    pub fn decision_image_format(&self) -> Option<DecisionImageFormat> {
        self.provider
            .is_decision()
            .then_some(self.decision_image_format)
    }

    /// The presence cutoff for this round, when vLLM asks whether the target appears at all.
    fn skip_below(&self, input: &Input) -> Option<f64> {
        let Input::Grid { layout, prompt, .. } = input else {
            return None;
        };
        self.plan
            .skip_below
            .filter(|_| *layout == 4 && (self.profile.allows_skipping)(prompt))
    }

    /// Requests sent for one round. Pass the round's input to count its presence request.
    pub fn decision_request_count(&self, layout: u8, input: Option<&Input>) -> Option<u8> {
        if self.provider.asks_twice() || self.jev_vision == Some(JevVision::Average) {
            let presence = u8::from(input.is_some_and(|input| self.skip_below(input).is_some()));
            let passes = if layout == 4 {
                self.plan.close_ups.len() as u8
            } else {
                1
            };
            let boxes = self
                .plan
                .boxes
                .filter(|_| layout == 4)
                .map_or(0, |rule| self.box_request_count(rule));
            // The whole grid, then every close-up request of every pass.
            Some(1 + presence + boxes + passes * self.close_up_batches(layout).len() as u8)
        } else {
            self.provider.is_decision().then_some(1)
        }
    }

    /// The requests a round's box rule sends. Liquid takes one state a request, so each of
    /// its window pictures travels alone.
    fn box_request_count(&self, rule: BoxRule) -> u8 {
        rule.asks()
            .map(|ask| match (ask, self.provider) {
                (BoxAsk::Windows, Provider::Liquid) => boxes::WINDOWS,
                _ => 1,
            })
            .sum()
    }

    /// How many API keys the requests are shared across.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// The tiles of each close-up request, as zero-based ranges.
    fn close_up_batches(&self, layout: u8) -> Vec<Range<u8>> {
        let tiles = layout * layout;
        let per_request = if self.provider.asks_twice() {
            self.vllm_close_ups_per_request
                .unwrap_or(VLLM_CLOSE_UPS_PER_REQUEST)
        } else if layout == 4 && self.jev_split_crops {
            MAX_CLOSE_UPS_WITH_SCENE
        } else {
            tiles
        };
        // Only vLLM's 4 × 4 requests are held to the image limit: Jev sends all sixteen
        // close-ups with the scene unless it is asked to split them.
        let per_request = if layout == 4 && self.provider.asks_twice() {
            per_request.min(if self.provider == Provider::Liquid {
                // The scene takes one of a Liquid request's places.
                liquid::MOST_IMAGES as u8 - 1
            } else {
                MAX_CLOSE_UPS_WITH_SCENE
            })
        } else {
            per_request
        };
        // Spread the tiles evenly over as few requests as that size allows. A request with a
        // single close-up scores it slightly differently, so no request is left with one.
        let mut requests = tiles.div_ceil(per_request);
        if requests > 1 && tiles / requests < 2 {
            requests -= 1;
        }
        let (size, larger) = (tiles / requests, tiles % requests);
        let mut start = 0;

        (0..requests)
            .map(|request| {
                let end = start + size + u8::from(request < larger);
                let batch = start..end;
                start = end;
                batch
            })
            .collect()
    }

    pub fn jev_image_strategy(&self, input: &Input) -> Option<JevVision> {
        let Input::Grid { layout, prompt, .. } = input else {
            return None;
        };
        match self.jev_vision? {
            JevVision::SceneAndCrop
                if *layout != 4 || (self.profile.target)(prompt, 4).is_none() =>
            {
                Some(JevVision::WholeGrid)
            }
            vision => Some(vision),
        }
    }

    pub fn tile_threshold(&self, layout: Option<u8>) -> Option<f64> {
        if !self.provider.is_decision() {
            return None;
        }
        if let Some(cutoff) = self.tile_threshold_override {
            return Some(cutoff);
        }
        // A 4 × 4 round selects from its plan's cutoff, which is one half unless a level sets it.
        if layout == Some(4) {
            return Some(self.plan.select_from);
        }
        if let Some(rule) = self
            .logistic_for(3)
            .and_then(|tuned| tuned.logistic_3x3.as_ref())
        {
            return Some(rule.cutoff);
        }
        let averages = self.jev_vision == Some(JevVision::Average) || self.provider.asks_twice();
        Some(if averages && layout == Some(3) {
            self.profile.average_cutoff_3x3
        } else {
            0.5
        })
    }

    /// The model's settings when the logistic rule decides grids of this size: it is the rule
    /// in use and has been fitted for them. A size it was not fitted for is averaged.
    fn logistic_for(&self, layout: u8) -> Option<&'static Tuned> {
        let tuned = self
            .tuned
            .filter(|_| self.vllm_decision == Some(VllmDecision::Logistic))?;
        match layout {
            4 => tuned.logistic_4x4.is_some(),
            _ => tuned.logistic_3x3.is_some(),
        }
        .then_some(tuned)
    }

    pub fn vllm_whole_grid_weight(&self, layout: u8) -> Option<f64> {
        if self.logistic_for(layout).is_some() {
            return None;
        }
        match layout {
            3 if self.provider.asks_twice() => Some(self.vllm_whole_grid_weights[0]),
            4 if self.provider.asks_twice() => Some(self.vllm_whole_grid_weights[1]),
            _ => None,
        }
    }

    fn predict_decision_grid(&self, input: &Input, threshold: f64) -> Result<Reply> {
        let Input::Grid {
            layout,
            prompt,
            png,
        } = input
        else {
            bail!("decision model averaging requires an image grid");
        };
        let side = *layout;
        let _round = self
            .warm
            .as_deref()
            .map(|warm| RoundInFlight::start(warm, side == 4));
        let target =
            (self.profile.target)(prompt, side).context("missing image challenge target")?;
        let encoding = self.decision_image_format;
        let send_jev_image_header = self.provider == Provider::Jev;
        let scene = grid_asset(png, encoding)?;
        let whole_request = decision_whole_grid(&self.model, side, prompt, scene.clone());
        // A 4 × 4 prompt that allows skipping may show none of the target. vLLM is then also
        // asked, in a small request of its own, whether the target appears at all. Adding the
        // question to the whole-grid request would shift that request's square scores.
        let skip_below = self.skip_below(input);
        // The box requests go out first, each on its own thread and in parallel with the
        // scoring requests, so the round never waits to send one.
        let box_round = self
            .plan
            .boxes
            .filter(|_| side == 4)
            .map(|rule| (rule, self.start_box_requests(target, png, rule)));
        let presence_request =
            skip_below.map(|_| decision_presence(&self.model, side, target, scene.clone()));
        // Close-up passes for 4 × 4 grids, each on its own copy of the grid: the capture, or
        // one rescaled to a set size. Their scores are averaged.
        let mut passes = Vec::new();
        let mut pass_pixels = Vec::new();
        if side != 3 {
            for scale in self.plan.close_ups {
                let (grid, asset) = match *scale {
                    None => (
                        Pixmap::decode_png(png).context("decode grid for tile crops")?,
                        scene.clone(),
                    ),
                    Some(longer_side) => {
                        let image = image::load_from_memory_with_format(png, ImageFormat::Png)
                            .context("decode the grid for a rescaled close-up pass")?;
                        let (rescaled, _) = resized_png(&image, longer_side)?;
                        (
                            Pixmap::decode_png(&rescaled).context("decode the rescaled grid")?,
                            grid_asset(&rescaled, encoding)?,
                        )
                    }
                };
                pass_pixels.push([grid.width(), grid.height()]);
                passes.push((grid, asset));
            }
        }
        // vLLM gets smoothly enlarged squares: the same accuracy on human-labelled grids
        // and about 18% smaller requests. Jev keeps the nearest-neighbour crops it was measured with.
        let filter = if self.provider.asks_twice() {
            FilterQuality::Bilinear
        } else {
            FilterQuality::Nearest
        };
        let send_crops = |grid: &Pixmap, scene: &Value, tiles: Range<u8>| {
            let request = scene_and_crops(
                &self.model,
                target,
                grid,
                side,
                tiles,
                encoding,
                scene.clone(),
                filter,
            )?;
            self.send(&request, send_jev_image_header)
                .context("scene and crops")
        };
        // One list entry is one request.
        let batches = self.close_up_batches(side);
        let batches = &batches;
        let send_crops = &send_crops;
        let send_photos = |tiles: Range<u8>| {
            let request = jev_individual_photos(
                &self.model,
                target,
                png,
                encoding,
                tiles,
                &self.profile.photo,
            )?;
            self.send(&request, send_jev_image_header)
                .context("individual photos")
        };
        let send_photos = &send_photos;
        let (whole, crops, presence) = thread::scope(|scope| -> Result<_> {
            let whole = scope.spawn(|| self.send(&whole_request, send_jev_image_header));
            let presence = presence_request
                .as_ref()
                .map(|request| scope.spawn(|| self.send(request, send_jev_image_header)));
            let crops = if side == 3 {
                let workers: Vec<_> = batches
                    .iter()
                    .cloned()
                    .map(|tiles| scope.spawn(move || send_photos(tiles)))
                    .collect();
                vec![
                    workers
                        .into_iter()
                        .map(|worker| {
                            worker
                                .join()
                                .map_err(|_| anyhow::anyhow!("photo request worker stopped"))?
                        })
                        .collect::<Result<Vec<_>>>()?,
                ]
            } else {
                // Every crop batch of every pass goes out at once, alongside the whole grid.
                let workers: Vec<Vec<_>> = passes
                    .iter()
                    .map(|(grid, scene)| {
                        batches
                            .iter()
                            .cloned()
                            .map(|tiles| scope.spawn(move || send_crops(grid, scene, tiles)))
                            .collect()
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|pass| {
                        pass.into_iter()
                            .map(|worker| {
                                worker
                                    .join()
                                    .map_err(|_| anyhow::anyhow!("crop request worker stopped"))?
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            let whole = whole
                .join()
                .map_err(|_| anyhow::anyhow!("whole-grid request worker stopped"))?;
            // A failed presence request leaves the round's picks as they are.
            let presence = presence.map(|worker| {
                worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("presence request worker stopped"))
                    .and_then(|response| response)
            });
            Ok((whole.context("whole grid")?, crops, presence))
        })?;

        let crop_meta: Vec<_> = crops
            .iter()
            .flatten()
            .map(|response| &response.meta)
            .collect();
        let mut meta = json!({"whole_grid": whole.meta, "crop_requests": crop_meta});
        if let Some(Ok(presence)) = &presence {
            meta["presence"] = json!(presence.meta);
        }
        let passes: Vec<_> = crops.into_iter().map(merge_crop_responses).collect();
        // Record what each 4 × 4 pass scored before they are averaged, so that a logged round
        // can be scored again under another level's plan.
        let pass_log = (side == 4).then(|| {
            passes
                .iter()
                .zip(&pass_pixels)
                .map(|(pass, pixels)| {
                    json!({
                        "grid_pixels": pixels,
                        "tile_probabilities": jev_grid_scores(pass.answers.as_ref(), side * side).ok(),
                    })
                })
                .collect::<Vec<_>>()
        });
        let variant_name = if side == 3 {
            "individual-photos"
        } else {
            "scene-and-crop"
        };
        let combine = match self.logistic_for(side) {
            Some(tuned) => Combine::Logistic(tuned),
            None => Combine::Average(self.vllm_whole_grid_weight(side).unwrap_or(0.5)),
        };
        let close_ups = average_crop_passes(passes);
        // The gap rule compares the two views' own scores for a square.
        let views = self.plan.gap.filter(|_| side == 4).and_then(|rule| {
            let whole = jev_grid_scores(whole.answers.as_ref(), side * side).ok()?;
            let close_ups = jev_grid_scores(close_ups.answers.as_ref(), side * side).ok()?;
            Some((rule, whole, close_ups))
        });
        let mut reply =
            blend_decision_replies(whole, close_ups, variant_name, side, threshold, combine);
        if let Some((components, passes)) = reply.decision_components.as_mut().zip(pass_log) {
            components["variant"]["passes"] = json!(passes);
        }
        match skip_below.zip(presence) {
            Some((minimum, Ok(presence))) => {
                let probability = presence
                    .answers
                    .as_ref()
                    .and_then(|answers| jev_probability(answers, PRESENCE_QUESTION));
                let skipped = skip_when_absent(&mut reply, minimum, probability);
                reply.usage = add_usage(reply.usage.as_ref(), presence.usage.as_ref());
                if let Some(components) = reply.decision_components.as_mut() {
                    components["presence"] = json!({
                        "probability": probability, "minimum": minimum, "skipped": skipped,
                        "usage": presence.usage,
                    });
                }
            }
            Some((minimum, Err(error))) => {
                if let Some(components) = reply.decision_components.as_mut() {
                    components["presence"] = json!({
                        "error": format!("{error:#}"), "minimum": minimum, "skipped": false,
                    });
                }
            }
            None => {}
        }
        if let Some((rule, whole, close_ups)) = views {
            apply_gap_rule(&mut reply, rule, &whole, &close_ups);
        }
        if let Some((rule, replies)) = box_round {
            // Each reply is read as it arrives; the round stops listening once nothing it
            // could say would change the selection.
            let read = replies.into_iter().map(|(ask, result)| {
                let found = result.and_then(|reply| {
                    let found = boxes::reply_boxes(ask, &reply)?;
                    let entry = boxes::log_entry(&found, &reply);
                    Ok((found, entry))
                });
                (ask, found.map_err(|failure| format!("{failure:#}")))
            });
            apply_box_rule(&mut reply, rule, self.vllm_box_wait, read);
        }
        reply.provider_meta = Some(meta);
        Ok(reply)
    }

    /// Makes one box request's pictures and sends it.
    fn box_request(
        wire: Wire,
        warm: Option<&Warm>,
        model: &str,
        target: &str,
        png: &[u8],
        ask: BoxAsk,
        one_state: bool,
    ) -> Result<Value> {
        let start = Instant::now();
        let grid = boxes::decode_grid(png)?;
        let json = match ask {
            BoxAsk::Windows if one_state => {
                return Self::window_requests(wire, model, target, &grid, start);
            }
            BoxAsk::Whole => boxes::whole_request(model, target, &grid)?,
            BoxAsk::Every => boxes::every_request(model, target, &grid)?,
            BoxAsk::Windows => boxes::windows_request(model, target, &grid)?,
        };
        let wire = wire.carrying(json)?;
        let prepared = start.elapsed();
        if let Some(warm) = warm {
            warm.wait_for_keeper();
        }
        let start = Instant::now();
        let (mut reply, attempts): (Value, u8) = wire.send_hedged()?;
        ensure!(reply.is_object(), "the box reply is not an object");
        if !reply["meta"].is_object() {
            reply["meta"] = json!({});
        }
        reply["meta"]["prepare_ms"] = json!(prepared.as_millis());
        reply["meta"]["request_wall_ms"] = json!(start.elapsed().as_millis());
        reply["meta"]["request_bytes"] = json!(wire.body.len());
        if attempts > 1 {
            reply["meta"]["attempts"] = json!(attempts);
        }
        Ok(reply)
    }

    /// Sends the window pictures one to a request, all at once, and puts the replies together
    /// as the reply to a single request holding every window would read.
    fn window_requests(
        wire: Wire,
        model: &str,
        target: &str,
        grid: &crate::pillow::Rgb,
        start: Instant,
    ) -> Result<Value> {
        let requests = boxes::window_requests(model, target, grid)?
            .into_iter()
            .map(|(name, json)| Ok((name, wire.clone().carrying(json)?)))
            .collect::<Result<Vec<_>>>()?;
        let prepared = start.elapsed();
        let start = Instant::now();
        let bytes: usize = requests.iter().map(|(_, wire)| wire.body.len()).sum();
        let replies = thread::scope(|scope| {
            let workers: Vec<_> = requests
                .iter()
                .map(|(name, wire)| (name, scope.spawn(|| wire.send_hedged::<Value>())))
                .collect();
            workers
                .into_iter()
                .map(|(name, worker)| {
                    let (reply, _) = worker
                        .join()
                        .map_err(|_| anyhow::anyhow!("a window request worker stopped"))?
                        .with_context(|| format!("window {name}"))?;
                    Ok((name.clone(), reply))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut usage = None;
        let mut states = serde_json::Map::new();
        for (name, reply) in replies {
            usage = Some(match usage {
                None => reply["usage"].clone(),
                Some(total) => {
                    add_usage(Some(&total), Some(&reply["usage"])).unwrap_or(Value::Null)
                }
            });
            states.insert(name, json!({"answers": reply["answers"]}));
        }
        Ok(json!({
            "states": states,
            "usage": usage,
            "meta": {
                "prepare_ms": prepared.as_millis(), "request_wall_ms": start.elapsed().as_millis(),
                "request_bytes": bytes, "requests": states.len(),
            },
        }))
    }

    /// Sends a 4 × 4 round's box requests, each from its own thread, which also makes its
    /// pictures. Each thread reports once: which request it sent, and the reply.
    fn start_box_requests(
        &self,
        target: &str,
        png: &[u8],
        rule: BoxRule,
    ) -> mpsc::Receiver<(BoxAsk, Result<Value>)> {
        let (results, replies) = mpsc::channel();
        let png: Arc<[u8]> = Arc::from(png);
        for ask in rule.asks() {
            let (wire, warm) = (self.empty_wire(false), self.warm.clone());
            let (results, png) = (results.clone(), Arc::clone(&png));
            let (model, target) = (self.model.clone(), target.to_owned());
            let one_state = self.provider == Provider::Liquid;
            thread::spawn(move || {
                let reply =
                    Self::box_request(wire, warm.as_deref(), &model, &target, &png, ask, one_state);
                let _ = results.send((ask, reply));
            });
        }
        replies
    }

    pub fn check_decision_vision(&self) -> Result<VisionCheck> {
        ensure!(
            self.provider.is_decision(),
            "the vision check requires Jev or vLLM"
        );
        if let Some(check) = self.verified_vision.get() {
            return Ok(check.clone());
        }

        let mut controls = Vec::with_capacity(2);
        for (expected_red, color) in [
            (true, tiny_skia::Color::from_rgba8(255, 0, 0, 255)),
            (false, tiny_skia::Color::from_rgba8(0, 0, 255, 255)),
        ] {
            let mut image = tiny_skia::Pixmap::new(320, 320).context("allocate vision control")?;
            image.fill(color);
            let asset = match self.decision_image_format {
                DecisionImageFormat::Jpeg => pixmap_jpeg(&image)?,
                DecisionImageFormat::Png => {
                    inline_png(&image.encode_png().context("encode vision control")?)
                }
            };
            let request = decision_image_request(
                &self.model,
                asset,
                json!({
                    "red": {
                        "type": "noul",
                    "instructions": "Is the image in `grid` predominantly red?"
                    }
                }),
            );
            let start = Instant::now();
            let response = self
                .send(&request, self.provider == Provider::Jev)
                .context("vision control request failed")?;
            controls.push(VisionControl {
                expected_red,
                probability_red: response
                    .answers
                    .as_ref()
                    .and_then(|answers| jev_probability(answers, "red")),
                resolved_model: response.model,
                usage: response.usage,
                latency_ms: start.elapsed().as_millis(),
                provider_meta: response.meta,
            });
        }
        let check = VisionCheck {
            passed: controls.iter().all(|control| {
                control.probability_red.is_some_and(|probability| {
                    if control.expected_red {
                        probability >= 0.7
                    } else {
                        probability <= 0.3
                    }
                })
            }),
            model: self.model.clone(),
            endpoint: self.url.to_string(),
            controls,
        };
        if check.passed {
            let _ = self.verified_vision.set(check.clone());
        }
        Ok(check)
    }

    fn send(&self, body: &impl Serialize, contains_jev_image: bool) -> Result<ApiResponse> {
        if self.provider == Provider::Vllm {
            return self.send_vllm(body);
        }
        self.send_request(body, contains_jev_image)
    }

    fn send_vllm(&self, body: &impl Serialize) -> Result<ApiResponse> {
        #[derive(Serialize)]
        struct WithMeta<'a, T> {
            #[serde(flatten)]
            request: &'a T,
            options: Value,
        }
        self.send_request(
            &WithMeta {
                request: body,
                options: json!({"return_meta": true}),
            },
            false,
        )
    }

    /// A valid request about the size of a round's largest, for warming a connection: one
    /// noisy image and one question.
    fn practice_upload(&self) -> Result<Wire> {
        let (side, mut state) = (PRACTICE_IMAGE_SIDE, 0x2545_f491_u32);
        let rgb: Vec<u8> = (0..side * side * 3)
            .map(|_| {
                // Noise does not compress, so the upload is as large as the image.
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect();
        #[derive(Serialize)]
        struct WithMeta {
            #[serde(flatten)]
            request: Value,
            options: Value,
        }
        self.wire(
            &WithMeta {
                request: decision_image_request(
                    &self.model,
                    inline_jpeg(&rgb, side, side)?,
                    json!({"red": {
                        "type": "noul",
                        "instructions": "Is the image in `grid` predominantly red?",
                    }}),
                ),
                options: json!({"return_meta": true}),
            },
            false,
        )
    }

    fn wire(&self, body: &impl Serialize, contains_jev_image: bool) -> Result<Wire> {
        let mut wire = self.empty_wire(contains_jev_image);
        let json = if self.provider == Provider::Liquid {
            let named = serde_json::to_value(body).context("encode decision request")?;
            let (listed, tokens) = liquid::request(named)?;
            wire.least_tokens = tokens;
            serde_json::to_vec(&listed)
        } else {
            serde_json::to_vec(body)
        }
        .context("encode decision request")?;
        wire.carrying(json)
    }

    /// A request that has its key and still needs its body.
    fn empty_wire(&self, contains_jev_image: bool) -> Wire {
        let vllm = self.provider == Provider::Vllm;
        Wire {
            client: self.client.clone(),
            url: self.url.clone(),
            keys: Arc::clone(&self.keys),
            first_key: self.next_key.fetch_add(1, Ordering::Relaxed),
            body: Arc::new(Vec::new()),
            gzip: vllm,
            jev_image_header: contains_jev_image,
            names_errors: self.provider.asks_twice(),
            // d3-ultra answers the same request the same way, so a duplicate of a slow request
            // costs only tokens. Liquid does not, and counts every request against one limit
            // for the organisation.
            hedge: vllm,
            least_tokens: 0,
            timeout: None,
        }
    }

    fn send_request(&self, body: &impl Serialize, contains_jev_image: bool) -> Result<ApiResponse> {
        let wire = self.wire(body, contains_jev_image)?;
        if let Some(warm) = &self.warm {
            warm.wait_for_keeper();
        }
        let start = Instant::now();
        let (mut result, attempts): (ApiResponse, u8) = if self.provider.asks_twice() {
            wire.send_hedged()?
        } else {
            (wire.attempt(0).map_err(|failure| failure.error)?, 1)
        };
        // An endpoint that does not read a request's pictures answers from the text alone, and
        // the only sign is how little it billed.
        if wire.least_tokens > 0 {
            let billed = result
                .usage
                .as_ref()
                .and_then(|usage| usage["input_tokens"].as_u64());
            ensure!(
                billed.is_some_and(|billed| billed >= wire.least_tokens),
                "the reply was billed {billed:?} input tokens and its pictures alone cost {}, so they were not read",
                wire.least_tokens
            );
        }
        if self.provider.is_decision() {
            let meta = result.meta.get_or_insert_with(|| json!({}));
            if let Some(fields) = meta.as_object_mut() {
                fields.insert("request_wall_ms".into(), json!(start.elapsed().as_millis()));
                fields.insert("request_bytes".into(), json!(wire.body.len()));
                if attempts > 1 {
                    fields.insert("attempts".into(), json!(attempts));
                }
            }
        }
        Ok(result)
    }

    /// Opens the connections a round needs before the round arrives and keeps them open for
    /// a few minutes. Call it whenever a challenge may be coming, such as when the lab page loads.
    pub fn keep_warm(&self) {
        if let Some(warm) = &self.warm {
            *warm.until.lock().expect("warm lock") = Some(Instant::now() + WARM_FOR);
            let _ = warm.wake.send(());
        }
    }
}

/// How long after its first try an unanswered request is sent again. About one request in
/// several hundred stalls for ten seconds or more at the provider while its neighbours finish
/// normally, and on a poor network a request and its first duplicate can both stall. d3-ultra
/// gives the same answer to the same request, so the first reply is used.
const HEDGE_AT: [Duration; 3] = [
    Duration::from_millis(1500),
    Duration::from_millis(4000),
    Duration::from_millis(8000),
];
const WARM_FOR: Duration = Duration::from_secs(300);
/// Connections left unused this long are pinged. The provider kept an idle one for 60 seconds.
const WARM_PING_AFTER: Duration = Duration::from_secs(45);
/// After this long unused the connections may have been closed, so they are warmed again.
const WARM_LAPSE: Duration = Duration::from_secs(90);
const PRACTICE_TIMEOUT: Duration = Duration::from_secs(6);
/// How long a request waits for the keeper to hand the connections back.
const KEEPER_PATIENCE: Duration = Duration::from_millis(2500);
/// Noise of this size makes a JPEG about twice the size of a round's largest scoring request,
/// which leaves the connection able to send that request in one go.
const PRACTICE_IMAGE_SIDE: u32 = 544;

/// Everything one HTTP attempt needs, so that a duplicate can run on its own thread.
#[derive(Clone)]
struct Wire {
    client: Client,
    url: Url,
    keys: Arc<Vec<String>>,
    /// Which key the first try uses. A second try takes the next one.
    first_key: usize,
    body: Arc<Vec<u8>>,
    gzip: bool,
    jev_image_header: bool,
    /// Whether a refusal's error code is read into the message.
    names_errors: bool,
    /// Whether a request with no reply yet is sent again at the times in `HEDGE_AT`.
    hedge: bool,
    /// The fewest input tokens a reply can have been billed if the pictures were read.
    least_tokens: u64,
    /// Replaces the client's timeout for this request.
    timeout: Option<Duration>,
}

struct Failure {
    error: anyhow::Error,
    then: AfterFailure,
    /// How long the provider asked to be left alone, from its `Retry-After`.
    retry_after: Option<Duration>,
}

#[derive(Clone, Copy, PartialEq)]
enum AfterFailure {
    /// The provider was busy or the connection failed: send the request again.
    Retry,
    /// This try ran out of time. Others still running may answer.
    Wait,
    /// The provider rejected the request or its reply was unreadable: give up.
    Stop,
}

impl Wire {
    /// Gives the request its body, compressed when the provider takes that.
    fn carrying(mut self, json: Vec<u8>) -> Result<Self> {
        self.body = Arc::new(if self.gzip {
            // Base64 images shrink by about a quarter, which matters because a round's
            // requests share the uplink.
            let mut gzip = GzEncoder::new(Vec::new(), Compression::fast());
            gzip.write_all(&json).context("compress decision request")?;
            gzip.finish().context("compress decision request")?
        } else {
            json
        });
        Ok(self)
    }

    fn attempt<T: DeserializeOwned>(&self, try_number: usize) -> Result<T, Failure> {
        let fatal = |error: anyhow::Error| Failure {
            error,
            then: AfterFailure::Stop,
            retry_after: None,
        };
        let key = &self.keys[(self.first_key + try_number) % self.keys.len()];
        let mut request = self
            .client
            .post(self.url.clone())
            .bearer_auth(key)
            .header("Content-Type", "application/json")
            .body(self.body.as_ref().clone());
        if self.gzip {
            request = request.header("Content-Encoding", "gzip");
        }
        if self.jev_image_header {
            request = request.header("TypeSafe-Danger-Zone-Images", "true");
        }
        if let Some(timeout) = self.timeout {
            request = request.timeout(timeout);
        }
        let response = request.send().map_err(|error| {
            let message = if error.is_timeout() {
                "request timed out"
            } else {
                "request failed"
            };
            Failure {
                then: if error.is_timeout() {
                    AfterFailure::Wait
                } else {
                    AfterFailure::Retry
                },
                error: anyhow::Error::new(error.without_url()).context(message),
                retry_after: None,
            }
        })?;
        if !response.status().is_success() {
            let status = response.status();
            let then = if status.as_u16() == 429 || status.is_server_error() {
                AfterFailure::Retry
            } else {
                AfterFailure::Stop
            };
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok()?.trim().parse().ok())
                .map(|seconds: u64| Duration::from_secs(seconds.min(MOST_RETRY_AFTER)));
            if self.names_errors
                && let Ok(body) = response.json::<Value>()
                && let Some(code) = body["error"]["code"].as_str()
                && code.len() <= 64
                && code.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            {
                return Err(Failure {
                    error: anyhow::anyhow!("HTTP {status} ({code})"),
                    then,
                    retry_after,
                });
            }
            return Err(Failure {
                error: anyhow::anyhow!("HTTP {status}"),
                then,
                retry_after,
            });
        }
        response
            .json()
            .map_err(|error| error.without_url())
            .context("response did not match the API JSON format")
            .map_err(fatal)
    }

    /// Sends the request, and sends it again while no try has answered: at each time in
    /// `HEDGE_AT`, and once after the first try that fails because the provider was busy or
    /// the connection broke. Each try uses the next key. Returns the first good reply and how
    /// many tries were started.
    fn send_hedged<T: DeserializeOwned + Send + 'static>(&self) -> Result<(T, u8)> {
        let (results, replies) = mpsc::channel();
        let start = |try_number: usize, delay: Duration| {
            let (wire, results) = (self.clone(), results.clone());
            thread::spawn(move || {
                thread::sleep(delay);
                let _ = results.send(wire.attempt(try_number));
            });
        };
        let most_tries = 1 + HEDGE_AT.len();
        let began = Instant::now();
        start(0, Duration::ZERO);
        let (mut started, mut finished) = (1, 0);
        let mut error = None;
        while finished < started {
            // The next duplicate is due at its time in the list, whatever started the last one.
            let due = HEDGE_AT
                .get(started - 1)
                .filter(|_| self.hedge)
                .map(|at| at.saturating_sub(began.elapsed()));
            let reply = match due {
                Some(wait) => match replies.recv_timeout(wait) {
                    Ok(reply) => reply,
                    Err(_) => {
                        start(started, Duration::ZERO);
                        started += 1;
                        continue;
                    }
                },
                None => match replies.recv() {
                    Ok(reply) => reply,
                    Err(_) => break,
                },
            };
            finished += 1;
            let failure = match reply {
                Ok(response) => return Ok((response, started as u8)),
                Err(failure) => failure,
            };
            if failure.then == AfterFailure::Stop {
                return Err(failure.error);
            }
            // With duplicates on a timer, only the first failure is answered with another try.
            // Without them every busy reply is, until the tries run out.
            if failure.then == AfterFailure::Retry
                && (error.is_none() || !self.hedge)
                && started < most_tries
            {
                // A busy provider gets a moment, or as long as it asked for.
                start(
                    started,
                    failure.retry_after.unwrap_or(Duration::from_millis(150)),
                );
                started += 1;
            }
            error.get_or_insert(failure.error);
        }
        Err(error.unwrap_or_else(|| anyhow::anyhow!("request failed")))
    }
}

/// Counts a round while it runs, so the keeper does not ping in the middle of it.
struct RoundInFlight<'a> {
    warm: &'a Warm,
    /// A 4 × 4 round uses nearly every connection, so none of them needs a ping soon after.
    full: bool,
}

impl<'a> RoundInFlight<'a> {
    fn start(warm: &'a Warm, full: bool) -> Self {
        warm.rounds_in_flight.fetch_add(1, Ordering::SeqCst);
        Self { warm, full }
    }
}

impl Drop for RoundInFlight<'_> {
    fn drop(&mut self) {
        if self.full {
            *self.warm.last_full_round.lock().expect("warm lock") = Some(Instant::now());
        }
        self.warm.rounds_in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Shared with the thread that keeps a provider's connections open.
struct Warm {
    until: Mutex<Option<Instant>>,
    wake: mpsc::Sender<()>,
    rounds_in_flight: AtomicUsize,
    last_full_round: Mutex<Option<Instant>>,
    /// Set while the keeper is using every connection.
    keeper_busy: AtomicBool,
}

impl Warm {
    /// Lets the keeper finish with the connections instead of opening new ones beside it.
    fn wait_for_keeper(&self) {
        let patience = Instant::now() + KEEPER_PATIENCE;
        while self.keeper_busy.load(Ordering::SeqCst) && Instant::now() < patience {
            thread::sleep(Duration::from_millis(5));
        }
    }
}

/// Holds `connections` connections open and warm while warmth is requested.
struct Keeper {
    practice: Wire,
    models: Url,
    connections: usize,
    warm: Arc<Warm>,
}

impl Keeper {
    fn run(self, wake: mpsc::Receiver<()>) {
        let mut last_wave: Option<Instant> = None;
        loop {
            let woken = wake.recv_timeout(Duration::from_secs(1));
            if woken == Err(mpsc::RecvTimeoutError::Disconnected) {
                return;
            }
            let wanted = self
                .warm
                .until
                .lock()
                .expect("warm lock")
                .is_some_and(|until| Instant::now() < until);
            // The connections were last used by the keeper or by a 4 × 4 round.
            let last_use = last_wave.max(*self.warm.last_full_round.lock().expect("warm lock"));
            if !wanted || last_use.is_some_and(|last| last.elapsed() < WARM_PING_AFTER) {
                continue;
            }
            self.warm.keeper_busy.store(true, Ordering::SeqCst);
            // A round in flight is already using the connections.
            if self.warm.rounds_in_flight.load(Ordering::SeqCst) == 0 {
                // New connections need an upload before they are as fast as used ones.
                let new = last_use.is_none_or(|last| last.elapsed() > WARM_LAPSE) || self.ping();
                if new {
                    self.practise();
                }
                last_wave = Some(Instant::now());
            }
            self.warm.keeper_busy.store(false, Ordering::SeqCst);
        }
    }

    /// Sends a cheap request on every connection at once, so that each travels on its own.
    /// Returns whether any connection looks new: a request that had to open one takes
    /// three round trips instead of one.
    fn ping(&self) -> bool {
        let (client, key) = (&self.practice.client, &self.practice.keys[0]);
        let times: Vec<Option<Duration>> = thread::scope(|scope| {
            let pings: Vec<_> = (0..self.connections)
                .map(|_| {
                    scope.spawn(|| {
                        let start = Instant::now();
                        client
                            .get(self.models.clone())
                            .bearer_auth(key)
                            .timeout(Duration::from_secs(3))
                            .send()
                            // An unread reply would close the connection.
                            .and_then(|reply| reply.bytes())
                            .ok()
                            .map(|_| start.elapsed())
                    })
                })
                .collect();
            pings
                .into_iter()
                .map(|ping| ping.join().ok().flatten())
                .collect()
        });
        let fastest = times.iter().flatten().min().copied();
        times.iter().any(|time| match (time, fastest) {
            (Some(time), Some(fastest)) => *time > fastest * 2 + Duration::from_millis(50),
            _ => true,
        })
    }

    /// Uploads a round-sized request on every connection at once.
    fn practise(&self) {
        thread::scope(|scope| {
            for index in 0..self.connections {
                let mut wire = self.practice.clone();
                wire.first_key += index;
                // Requests wait for the keeper, so a stalled upload must not hold them long.
                wire.timeout = Some(PRACTICE_TIMEOUT);
                scope.spawn(move || {
                    let _ = wire.attempt::<ApiResponse>(0);
                });
            }
        });
    }
}

pub fn configured_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn endpoint_url(base: &str, path: &str) -> Result<Url> {
    let mut url = Url::parse(base).context("invalid API base URL")?;
    ensure!(
        url.host().is_some() && matches!(url.scheme(), "http" | "https"),
        "the API base URL must use HTTP or HTTPS"
    );
    ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "the API base URL cannot contain credentials, a query, or a fragment"
    );
    url.path_segments_mut()
        .map_err(|_| anyhow::anyhow!("invalid API base URL path"))?
        .pop_if_empty()
        .extend(path.split('/'));
    Ok(url)
}

fn text_prompt(input: &Input) -> Result<String> {
    let Input::Trace(trace) = input else {
        bail!("legacy Completions requires a text fixture");
    };
    Ok(format!(
        "Estimate the probability that this browser interaction was initiated by a human. Consider the telemetry jointly.\n\nBrowser interaction:\n{}\n\nReturn only a JSON object with one numeric field, \"score\" (0 to 1).",
        serde_json::to_string_pretty(trace)?
    ))
}

fn chat_request(model: &str, input: &Input, max_output_tokens: u32) -> Result<Value> {
    let content = match input {
        Input::Grid {
            layout,
            prompt,
            png,
        } => json!([
            {
                "type": "text",
                "text": format!("Challenge instructions: {prompt}\nThe image is a {layout} by {layout} grid. Number tiles from 1, left to right and top to bottom. Return only JSON with a \"tiles\" array of all matching tiles. Use an empty array if none match.")
            },
            {
                "type": "image_url",
                "image_url": {"url": format!("data:image/png;base64,{}", STANDARD.encode(png))}
            }
        ]),
        Input::Trace(_) => json!(text_prompt(input)?),
    };
    Ok(json!({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_output_tokens,
    }))
}

fn jev_request(
    model: &str,
    input: &Input,
    vision: Option<JevVision>,
    encoding: DecisionImageFormat,
    profile: &Profile,
) -> Result<Value> {
    match input {
        Input::Grid {
            layout,
            prompt,
            png,
        } => {
            let side = *layout;
            let scene = grid_asset(png, encoding)?;
            if vision == Some(JevVision::SceneAndCrop) {
                let target = (profile.target)(prompt, side)
                    .context("missing split-photo challenge target")?;
                let grid = Pixmap::decode_png(png).context("decode grid for tile crops")?;
                return scene_and_crops(
                    model,
                    target,
                    &grid,
                    side,
                    0..side * side,
                    encoding,
                    scene,
                    FilterQuality::Nearest,
                );
            }
            Ok(decision_whole_grid(model, side, prompt, scene))
        }
        Input::Trace(trace) => Ok(json!({
            "model": model,
            "state": trace,
            "questions": {"legitimate": {
                "type": "noul",
                "instructions": "Was the interaction initiated by a legitimate human user? Consider the evidence jointly. Return a probability near 0 for automation and near 1 for a legitimate human."
            }},
        })),
    }
}

/// Resizes a grid so its longer side is `longer_side` pixels, keeping its aspect ratio.
/// Returns the PNG and its width and height.
fn resized_png(image: &image::DynamicImage, longer_side: u32) -> Result<(Vec<u8>, [u32; 2])> {
    let (width, height) = (image.width(), image.height());
    let scale = f64::from(longer_side) / f64::from(width.max(height));
    let size = |side: u32| ((f64::from(side) * scale).round() as u32).max(1);
    let resized = image.resize_exact(size(width), size(height), imageops::FilterType::Lanczos3);
    let mut out = Cursor::new(Vec::new());
    resized
        .write_to(&mut out, ImageFormat::Png)
        .context("encode the resized grid")?;
    Ok((out.into_inner(), [resized.width(), resized.height()]))
}

const PRESENCE_QUESTION: &str = "present";

/// One question about the whole grid: does the target appear anywhere in it?
fn decision_presence(model: &str, side: u8, target: &str, scene: Value) -> Value {
    let mut request = decision_image_request(
        model,
        scene,
        json!({PRESENCE_QUESTION: {
            "type": "noul",
            "instructions": "Does the photo in `grid` show at least one object of the type in `target`? Count a partly visible one. Do not count a different kind of object that looks similar.",
        }}),
    );
    request["state"]["target"] = json!(target);
    request["state"]["grid_layout"] = json!({"rows": side, "columns": side});
    request
}

fn decision_whole_grid(model: &str, side: u8, prompt: &str, scene: Value) -> Value {
    let questions: BTreeMap<_, _> = (0..side * side).map(|index| {
        let row = index / side + 1;
        let column = index % side + 1;
        (
            format!("tile_{:02}", index + 1),
            json!({
                "type": "noul",
                "instructions": format!("Does the tile at row {row}, column {column} of `grid` match `challenge_instructions`? Count from the top left."),
            }),
        )
    }).collect();
    let mut request = decision_image_request(model, scene, json!(questions));
    request["state"]["challenge_instructions"] = json!(prompt);
    request["state"]["grid_layout"] = json!({"rows": side, "columns": side});
    request
}

fn grid_asset(png: &[u8], encoding: DecisionImageFormat) -> Result<Value> {
    match encoding {
        DecisionImageFormat::Png => Ok(inline_png(png)),
        DecisionImageFormat::Jpeg => {
            pixmap_jpeg(&Pixmap::decode_png(png).context("decode grid for JPEG encoding")?)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scene_and_crops(
    model: &str,
    target: &str,
    grid: &Pixmap,
    side: u8,
    tiles: Range<u8>,
    encoding: DecisionImageFormat,
    scene: Value,
    filter: FilterQuality,
) -> Result<Value> {
    let mut request = decision_image_request(model, scene, json!({}));
    request["state"]["target"] = json!(target);
    request["state"]["grid_layout"] = json!({ "rows": side, "columns": side });
    for index in tiles {
        let row = index / side;
        let column = index % side;
        let tile = format!("tile_{:02}", index + 1);
        let asset_ref = format!("hypercaptcha-tile-crop-{:02}", index + 1);
        let crop = enlarged_tile(grid, side, row, column, filter)?;
        request["assets"][&asset_ref] = match encoding {
            DecisionImageFormat::Png => {
                inline_png(&crop.encode_png().context("encode enlarged tile")?)
            }
            DecisionImageFormat::Jpeg => pixmap_jpeg(&crop)?,
        };
        // Keep the crop reference after the text that tells Jev what to look for.
        request["questions"][&tile] = json!({
            "type": "noul",
            "instructions": [
                format!(
                    "PHOTO is an enlarged crop of row {}, column {} from the upper left of `grid`. Does any visible part of an object of the requested type in `target` fall inside PHOTO? Use `grid` to connect small fragments to the rest of the object. Include small fragments such as part of a wheel. Exclude shadows, nearby objects, people, and support structures.",
                    row + 1,
                    column + 1,
                ),
                asset_ref,
            ]
        });
    }
    Ok(request)
}

fn jev_individual_photos<'a>(
    model: &'a str,
    target: &'a str,
    png: &[u8],
    encoding: DecisionImageFormat,
    tiles: Range<u8>,
    question: &'a PhotoQuestion,
) -> Result<impl Serialize + use<'a>> {
    #[derive(Serialize)]
    struct PhotoRequest<'a> {
        model: &'a str,
        state: Value,
        questions: BTreeMap<String, Question>,
        assets: BTreeMap<String, Value>,
    }

    #[derive(Serialize)]
    struct Question {
        #[serde(rename = "type")]
        kind: &'static str,
        instructions: PhotoInstructions,
    }

    #[derive(Serialize)]
    struct PhotoInstructions {
        task: &'static str,
        #[serde(rename = "PHOTO")]
        photo: String,
    }

    let grid = image::load_from_memory_with_format(png, ImageFormat::Png)?.into_rgb8();
    let mut request = PhotoRequest {
        model,
        state: json!({question.state_key: target}),
        questions: BTreeMap::new(),
        assets: BTreeMap::new(),
    };
    for index in tiles {
        let reference = format!("hcap-comparison-photo-{:02}", index + 1);
        let bounds = tile_bounds(grid.width(), grid.height(), 3, index / 3, index % 3, 2)?;
        let crop = imageops::crop_imm(
            &grid,
            bounds.x() as u32,
            bounds.y() as u32,
            bounds.width(),
            bounds.height(),
        );
        let enlarged = imageops::resize(
            &crop.to_image(),
            CROP_SIZE,
            CROP_SIZE,
            imageops::FilterType::Lanczos3,
        );
        let asset = match encoding {
            DecisionImageFormat::Png => {
                let mut bytes = Cursor::new(Vec::new());
                enlarged.write_to(&mut bytes, ImageFormat::Png)?;
                inline_png(bytes.get_ref())
            }
            DecisionImageFormat::Jpeg => {
                inline_jpeg(enlarged.as_raw(), enlarged.width(), enlarged.height())?
            }
        };
        request.assets.insert(reference.clone(), asset);
        request.questions.insert(
            format!("tile_{:02}", index + 1),
            Question {
                kind: "noul",
                instructions: PhotoInstructions {
                    task: question.task,
                    photo: reference,
                },
            },
        );
    }
    Ok(request)
}

// TypeSafe accepts still images of at least 280 pixels per side.
const CROP_SIZE: u32 = 320;

fn tile_bounds(
    width: u32,
    height: u32,
    side: u8,
    row: u8,
    column: u8,
    inset: i32,
) -> Result<IntRect> {
    let edge = |length: u32, index: u8| {
        (f64::from(length) * f64::from(index) / f64::from(side)).round_ties_even() as i32
    };
    IntRect::from_ltrb(
        edge(width, column) + inset,
        edge(height, row) + inset,
        edge(width, column + 1) - inset,
        edge(height, row + 1) - inset,
    )
    .context("invalid grid tile bounds")
}

fn enlarged_tile(
    grid: &Pixmap,
    side: u8,
    row: u8,
    column: u8,
    filter: FilterQuality,
) -> Result<Pixmap> {
    let bounds = tile_bounds(grid.width(), grid.height(), side, row, column, 0)?;
    let tile = grid.clone_rect(bounds).context("crop grid tile")?;
    let mut enlarged = Pixmap::new(CROP_SIZE, CROP_SIZE).context("allocate enlarged tile")?;
    let scale = Transform::from_scale(
        CROP_SIZE as f32 / tile.width() as f32,
        CROP_SIZE as f32 / tile.height() as f32,
    );
    let paint = PixmapPaint {
        quality: filter,
        ..PixmapPaint::default()
    };
    enlarged.draw_pixmap(0, 0, tile.as_ref(), &paint, scale, None);
    Ok(enlarged)
}

fn decision_image_request(model: &str, asset: Value, questions: Value) -> Value {
    json!({
        "model": model,
        "state": {"grid": "hypercaptcha-grid-asset"},
        "assets": {"hypercaptcha-grid-asset": asset},
        "questions": questions,
    })
}

fn inline_png(png: &[u8]) -> Value {
    json!({"type": "image", "format": "png", "data_base64": STANDARD.encode(png)})
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum DecisionImageFormat {
    Png,
    Jpeg,
}

fn pixmap_jpeg(image: &Pixmap) -> Result<Value> {
    let rgb: Vec<_> = image
        .data()
        .chunks_exact(4)
        .flat_map(|pixel| pixel[..3].iter().copied())
        .collect();
    inline_jpeg(&rgb, image.width(), image.height())
}

fn inline_jpeg(rgb: &[u8], width: u32, height: u32) -> Result<Value> {
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90).encode(
        rgb,
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(json!({"type": "image", "format": "jpeg", "data_base64": STANDARD.encode(bytes)}))
}

#[derive(Deserialize)]
struct ApiResponse {
    model: Option<String>,
    usage: Option<Value>,
    choices: Option<Vec<OpenAiChoice>>,
    answers: Option<BTreeMap<String, JevAnswer>>,
    meta: Option<Value>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: Option<OpenAiMessage>,
    text: Option<String>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
    refusal: Option<String>,
}

#[derive(Deserialize)]
struct JevAnswer {
    noul: Option<f64>,
}

fn jev_probability(answers: &BTreeMap<String, JevAnswer>, id: &str) -> Option<f64> {
    answers
        .get(id)
        .and_then(|answer| answer.noul)
        .filter(|value| (0.0..=1.0).contains(value))
}

impl ApiResponse {
    fn into_reply(self, provider: Provider, input: &Input, tile_threshold: Option<f64>) -> Reply {
        let (output, tile_probabilities) = if provider.is_decision() {
            parse_jev(
                self.answers,
                input,
                tile_threshold.expect("decision tile threshold"),
            )
        } else {
            (parse_openai(self.choices, provider, input), None)
        };
        Reply {
            output,
            resolved_model: self.model,
            usage: self.usage,
            tile_probabilities,
            decision_components: None,
            provider_meta: self.meta,
        }
    }
}

/// Selects nothing when the model says the target is probably absent. The squares that scored
/// highest are then usually a look-alike, such as a moped on a bicycle prompt.
fn skip_when_absent(reply: &mut Reply, minimum: f64, probability: Option<f64>) -> bool {
    let skipped = probability.is_some_and(|probability| probability < minimum)
        && matches!(reply.output, Output::Prediction(_));
    if skipped {
        reply.output = Output::Prediction(Prediction::Tiles(Vec::new()));
    }
    skipped
}

/// Drops the selections that only the close-up believes in: see [`GapRule`].
fn apply_gap_rule(
    reply: &mut Reply,
    rule: GapRule,
    whole: &BTreeMap<u8, f64>,
    close_ups: &BTreeMap<u8, f64>,
) {
    let (Output::Prediction(Prediction::Tiles(picks)), Some(scores)) =
        (&reply.output, &reply.tile_probabilities)
    else {
        return;
    };
    let (kept, removed): (Vec<u8>, Vec<u8>) = picks.iter().partition(|tile| {
        scores[tile] >= rule.under || close_ups[tile] - whole[tile] <= rule.more_than
    });
    if !removed.is_empty() {
        reply.output = Output::Prediction(Prediction::Tiles(kept));
    }
    if let Some(components) = reply.decision_components.as_mut() {
        components["gap"] =
            json!({"under": rule.under, "more_than": rule.more_than, "removed": removed});
    }
}

/// One box request's outcome: which request it was, then its boxes with their log entry, or
/// why it failed.
type BoxReply = (BoxAsk, Result<(Vec<boxes::Found>, Value), String>);

/// Drops the selected squares that scored under the rule's exemption and that the boxes do
/// not reach: see [`boxes::Vote`].
///
/// `replies` yields each box request's outcome as it arrives. The requests all left when the
/// round began. Listening stops once every square is decided, unless `wait` asks for
/// every reply, and at the first failure, which leaves the squares still undecided selected.
fn apply_box_rule(
    reply: &mut Reply,
    rule: BoxRule,
    wait: bool,
    replies: impl Iterator<Item = BoxReply>,
) {
    let (Output::Prediction(Prediction::Tiles(picks)), Some(scores)) =
        (&reply.output, &reply.tile_probabilities)
    else {
        return;
    };
    let picks = picks.clone();
    let at_risk = boxes::at_risk(&picks, scores, &rule);
    let mut vote = boxes::Vote::new(rule, &at_risk);
    let mut log = json!({
        "confidence": rule.whole, "every_confidence": rule.every,
        "window_confidence": rule.windows, "cover": rule.cover,
        "inside": rule.inside, "risk_below": rule.exempt_from, "at_risk": at_risk,
    });
    let (mut heard, mut error) = (0, None);
    let mut replies = replies;
    while heard < rule.requests() && error.is_none() && (wait || !vote.settled()) {
        let Some((ask, result)) = replies.next() else {
            error = Some("a box request worker stopped".to_owned());
            break;
        };
        let name = ask.name();
        match result {
            Ok((found, entry)) => {
                vote.hear(ask, &found);
                log[name] = entry;
                heard += 1;
            }
            Err(failure) => error = Some(format!("{name}: {failure}")),
        }
    }
    let removed = vote.dropped();
    if !removed.is_empty() {
        let kept = picks
            .iter()
            .copied()
            .filter(|tile| !removed.contains(tile))
            .collect();
        reply.output = Output::Prediction(Prediction::Tiles(kept));
    }
    log["picks_before"] = json!(picks);
    log["removed"] = json!(removed);
    if let Some(error) = error {
        log["error"] = json!(error);
    }
    if let Some(components) = reply.decision_components.as_mut() {
        components["boxes"] = log;
    }
}

#[derive(Clone, Copy)]
enum Combine {
    /// Whole-grid weight; the crop view gets the remainder.
    Average(f64),
    /// The logistic rules fitted for one model on a CAPTCHA service.
    Logistic(&'static Tuned),
}

fn log_odds(probability: f64) -> f64 {
    let p = probability.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}

fn edge_neighbors(tile: u8, side: u8) -> impl Iterator<Item = u8> {
    let (row, column) = ((tile - 1) / side, (tile - 1) % side);
    [
        (row > 0).then(|| tile - side),
        (row + 1 < side).then(|| tile + side),
        (column > 0).then(|| tile - 1),
        (column + 1 < side).then(|| tile + 1),
    ]
    .into_iter()
    .flatten()
}

fn logistic_scores(
    whole: &BTreeMap<u8, f64>,
    crops: &BTreeMap<u8, f64>,
    side: u8,
    tuned: &Tuned,
) -> Result<BTreeMap<u8, f64>, String> {
    let unfitted = || format!("no logistic rule is fitted for {side} × {side} grids");
    let rule_4x4 = (side == 4)
        .then(|| tuned.logistic_4x4.as_ref().ok_or_else(unfitted))
        .transpose()?;
    let rule_3x3 = (side != 4)
        .then(|| tuned.logistic_3x3.as_ref().ok_or_else(unfitted))
        .transpose()?;
    let strongest = |scores: &BTreeMap<u8, f64>, tile| {
        edge_neighbors(tile, side)
            .map(|neighbor| log_odds(scores[&neighbor]))
            .fold(f64::NEG_INFINITY, f64::max)
    };
    Ok((1..=side * side)
        .map(|tile| {
            let (w, c) = (log_odds(whole[&tile]), log_odds(crops[&tile]));
            let z = if let Some(&[a, b, nw, nc, bias]) = rule_4x4 {
                a * w + b * c + nw * strongest(whole, tile) + nc * strongest(crops, tile) + bias
            } else {
                let [a, b, bias] = rule_3x3.expect("a 3 × 3 rule when not 4 × 4").weights;
                a * w + b * c + bias
            };
            (tile, 1.0 / (1.0 + (-z).exp()))
        })
        .collect())
}

fn blend_decision_replies(
    whole: ApiResponse,
    variant: ApiResponse,
    variant_name: &str,
    side: u8,
    threshold: f64,
    combine: Combine,
) -> Reply {
    let max_tile = side * side;
    let whole_scores = jev_grid_scores(whole.answers.as_ref(), max_tile)
        .map_err(|error| format!("whole grid: {error}"));
    let variant_scores = jev_grid_scores(variant.answers.as_ref(), max_tile)
        .map_err(|error| format!("{variant_name}: {error}"));
    let (output, tile_probabilities) = match (&whole_scores, &variant_scores) {
        (Ok(whole), Ok(variant)) => {
            let blended = match combine {
                Combine::Average(whole_weight) => Ok(whole
                    .iter()
                    .map(|(&tile, &score)| {
                        (
                            tile,
                            score * whole_weight + variant[&tile] * (1.0 - whole_weight),
                        )
                    })
                    .collect()),
                Combine::Logistic(tuned) => logistic_scores(whole, variant, side, tuned),
            };
            match blended {
                Ok(blended) => (select_jev_tiles(&blended, threshold), Some(blended)),
                Err(error) => (Output::Invalid(error), None),
            }
        }
        (Err(error), _) | (_, Err(error)) => (Output::Invalid(error.clone()), None),
    };

    let resolved_model = if whole.model == variant.model {
        whole.model.clone()
    } else {
        None
    };
    let usage = add_usage(whole.usage.as_ref(), variant.usage.as_ref());
    Reply {
        output,
        resolved_model,
        usage,
        tile_probabilities,
        decision_components: Some(json!({
            "rule": match combine {
                Combine::Average(_) => "average",
                Combine::Logistic(_) => "logistic",
            },
            "whole_grid": {
                "strategy": "whole-grid", "tile_probabilities": whole_scores.ok(),
                "resolved_model": whole.model, "usage": whole.usage,
            },
            "variant": {
                "strategy": variant_name, "tile_probabilities": variant_scores.ok(),
                "resolved_model": variant.model, "usage": variant.usage,
            },
        })),
        provider_meta: None,
    }
}

fn add_usage(first: Option<&Value>, second: Option<&Value>) -> Option<Value> {
    let (first, second) = first.zip(second)?;
    let total = |key: &str| first[key].as_u64()?.checked_add(second[key].as_u64()?);
    Some(json!({"input_tokens": total("input_tokens"), "output_tokens": total("output_tokens")}))
}

fn merge_crop_responses(mut crops: Vec<ApiResponse>) -> ApiResponse {
    let mut first = crops.remove(0);
    for other in crops {
        first.answers = first.answers.zip(other.answers).map(|(mut first, other)| {
            first.extend(other);
            first
        });
        first.usage = add_usage(first.usage.as_ref(), other.usage.as_ref());
        if first.model != other.model {
            first.model = None;
        }
    }
    first
}

/// Averages each tile's close-up score across passes in log-odds. One pass is returned as it is.
fn average_crop_passes(passes: Vec<ApiResponse>) -> ApiResponse {
    let count = passes.len() as f64;
    let mut passes = passes.into_iter();
    let mut first = passes.next().expect("at least one close-up pass");
    if count == 1.0 {
        return first;
    }
    let scores = |answers: &BTreeMap<String, JevAnswer>| -> BTreeMap<String, f64> {
        answers
            .keys()
            .filter_map(|id| Some((id.clone(), log_odds(jev_probability(answers, id)?))))
            .collect()
    };
    let mut sums = first.answers.as_ref().map(scores);
    for other in passes {
        // A tile that any pass failed to score drops out, and is reported as missing later.
        sums = sums.zip(other.answers.as_ref()).map(|(sums, answers)| {
            sums.into_iter()
                .filter_map(|(id, sum)| {
                    let score = log_odds(jev_probability(answers, &id)?);
                    Some((id, sum + score))
                })
                .collect()
        });
        first.usage = add_usage(first.usage.as_ref(), other.usage.as_ref());
        if first.model != other.model {
            first.model = None;
        }
    }
    first.answers = sums.map(|sums| {
        sums.into_iter()
            .map(|(id, sum)| {
                let noul = Some(1.0 / (1.0 + (-sum / count).exp()));
                (id, JevAnswer { noul })
            })
            .collect()
    });
    first
}

fn parse_jev(
    answers: Option<BTreeMap<String, JevAnswer>>,
    input: &Input,
    threshold: f64,
) -> (Output, Option<BTreeMap<u8, f64>>) {
    if let Some(max_tile) = input.max_tile() {
        match jev_grid_scores(answers.as_ref(), max_tile) {
            Ok(scores) => (select_jev_tiles(&scores, threshold), Some(scores)),
            Err(error) => (Output::Invalid(error), None),
        }
    } else {
        let Some(answers) = answers else {
            return (Output::Invalid("missing Jev answers".into()), None);
        };
        let output = match jev_probability(&answers, "legitimate") {
            Some(value) => Output::Prediction(Prediction::LegitimacyScore(value)),
            None => {
                Output::Invalid("missing or invalid Jev Noul probability for legitimate".into())
            }
        };
        (output, None)
    }
}

fn jev_grid_scores(
    answers: Option<&BTreeMap<String, JevAnswer>>,
    max_tile: u8,
) -> Result<BTreeMap<u8, f64>, String> {
    let answers = answers.ok_or("missing Jev answers")?;
    (1..=max_tile)
        .map(|tile| {
            let id = format!("tile_{tile:02}");
            jev_probability(answers, &id)
                .map(|value| (tile, value))
                .ok_or_else(|| format!("missing or invalid Jev Noul probability for {id}"))
        })
        .collect()
}

fn select_jev_tiles(probabilities: &BTreeMap<u8, f64>, threshold: f64) -> Output {
    let tiles = probabilities
        .iter()
        .filter_map(|(&tile, &score)| (score >= threshold).then_some(tile))
        .collect();
    Output::Prediction(Prediction::Tiles(tiles))
}

fn parse_openai(choices: Option<Vec<OpenAiChoice>>, provider: Provider, input: &Input) -> Output {
    let Some(choice) = choices.and_then(|choices| choices.into_iter().next()) else {
        return Output::Invalid("missing completion choice".into());
    };
    if choice.finish_reason.as_deref() == Some("content_filter")
        || choice
            .message
            .as_ref()
            .and_then(|message| message.refusal.as_deref())
            .is_some_and(|refusal| !refusal.is_empty())
    {
        return Output::Refusal;
    }
    if choice.finish_reason.as_deref() == Some("length") {
        return Output::Invalid("completion hit its token limit".into());
    }
    let content = match provider {
        Provider::OpenaiChat => choice.message.and_then(|message| message.content),
        Provider::OpenaiCompletions => choice.text,
        _ => unreachable!(),
    };
    let Some(content) = content else {
        return Output::Invalid("missing completion text".into());
    };
    let fence = "```";
    let content = content.trim();
    let payload = content
        .strip_prefix(fence)
        .and_then(|body| body.strip_suffix(fence))
        .map_or(content, |body| body.strip_prefix("json").unwrap_or(body));

    if let Some(max_tile) = input.max_tile() {
        #[derive(Deserialize)]
        struct Tiles {
            tiles: Vec<u8>,
        }

        if let Ok(Tiles { mut tiles }) = serde_json::from_str(payload) {
            tiles.sort_unstable();
            if tiles.iter().all(|tile| (1..=max_tile).contains(tile))
                && !tiles.windows(2).any(|pair| pair[0] == pair[1])
            {
                return Output::Prediction(Prediction::Tiles(tiles));
            }
        }
        Output::Invalid(format!(
            "expected JSON with unique integer tile IDs from 1 through {max_tile}"
        ))
    } else {
        #[derive(Deserialize)]
        struct Score {
            score: f64,
        }

        if let Ok(Score { score }) = serde_json::from_str(payload)
            && (0.0..=1.0).contains(&score)
        {
            return Output::Prediction(Prediction::LegitimacyScore(score));
        }
        Output::Invalid("expected JSON with a numeric score from 0 to 1".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECAPTCHA: &Tuned = &crate::captcha::recaptcha::vllm::d3_ultra::TUNED;

    fn tiles(values: &[f64]) -> BTreeMap<u8, f64> {
        (1..).zip(values.iter().copied()).collect()
    }

    fn assert_close(actual: &BTreeMap<u8, f64>, expected: &[f64]) {
        for (tile, want) in (1..).zip(expected) {
            let got = actual[&tile];
            assert!((got - want).abs() < 1e-5, "tile {tile}: {got} != {want}");
        }
    }

    // Reference values computed independently in Python from logged d3-ultra answers.
    #[test]
    fn logistic_rule_matches_the_fitted_reference_on_3x3() {
        let whole = tiles(&[
            0.910962,
            0.0212042,
            0.996278,
            0.999845,
            0.999755,
            0.00590522,
            0.00108685,
            0.0034534,
            0.000995191,
        ]);
        let photos = tiles(&[
            0.497478,
            0.0124854,
            0.999416,
            0.999385,
            0.999831,
            0.0157936,
            0.00317473,
            0.00289398,
            0.000319073,
        ]);
        let scores = logistic_scores(&whole, &photos, 3, RECAPTCHA).unwrap();
        assert_close(
            &scores,
            &[
                0.401178,
                0.000329711,
                0.999791,
                0.999964,
                0.999986,
                0.000196957,
                1.63392e-05,
                2.90287e-05,
                1.78865e-06,
            ],
        );
        let Output::Prediction(Prediction::Tiles(picks)) =
            select_jev_tiles(&scores, RECAPTCHA.logistic_3x3.as_ref().unwrap().cutoff)
        else {
            panic!("expected tiles")
        };
        assert_eq!(picks, [3, 4, 5]);
    }

    #[test]
    fn grid_resize_applies_only_above_the_layout_limit() {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            options: ProviderOptions,
        }
        let Cli { options } = <Cli as clap::Parser>::parse_from([
            "test",
            "--provider",
            "vllm",
            "--grid-max-side-4x4",
            "467",
        ]);
        // SAFETY: tests run single-threaded with respect to this variable's readers below.
        unsafe { env::set_var("VLLM_SR_API_KEY", "test-key") };
        let backend = Backend::new(options, &crate::captcha::recaptcha::PROFILE)
            .unwrap()
            .unwrap();
        let png = Pixmap::new(772, 778).unwrap().encode_png().unwrap();
        let (_, size) = backend.prepare_grid(4, png.clone()).unwrap();
        assert_eq!(size, [463, 467]);
        let (same, size) = backend.prepare_grid(3, png.clone()).unwrap();
        assert_eq!((same, size), (png, [772, 778]));
        let small = Pixmap::new(388, 388).unwrap().encode_png().unwrap();
        assert_eq!(backend.prepare_grid(4, small).unwrap().1, [388, 388]);
    }

    #[test]
    fn close_up_passes_are_averaged_in_log_odds() {
        let pass = |scores: &[(&str, f64)]| ApiResponse {
            model: Some("d3".into()),
            usage: Some(json!({"input_tokens": 5, "output_tokens": 0})),
            choices: None,
            answers: Some(
                scores
                    .iter()
                    .map(|&(id, noul)| (id.to_owned(), JevAnswer { noul: Some(noul) }))
                    .collect(),
            ),
            meta: None,
        };
        let single = average_crop_passes(vec![pass(&[("tile_01", 0.3)])]);
        assert_eq!(single.answers.unwrap()["tile_01"].noul, Some(0.3));

        // tile_02 is missing from the second pass, so it is left out instead of guessed.
        let averaged = average_crop_passes(vec![
            pass(&[("tile_01", 0.9), ("tile_02", 0.4)]),
            pass(&[("tile_01", 0.5)]),
        ]);
        let answers = averaged.answers.unwrap();
        assert_eq!(answers.len(), 1);
        // The mean of log-odds ln 9 and 0 is ln 3, which is a probability of 0.75.
        assert!((answers["tile_01"].noul.unwrap() - 0.75).abs() < 1e-12);
        assert_eq!(averaged.usage.unwrap()["input_tokens"], 10);
        assert_eq!(averaged.model.as_deref(), Some("d3"));
    }

    /// d3-ultra's answers on one recorded 4 × 4 grid, and the squares each effort level should
    /// select from them. Boxes are left, top, right, bottom in fractions of the grid, then
    /// the confidence.
    #[derive(Deserialize)]
    struct Recorded {
        id: String,
        presence: Option<f64>,
        whole: Vec<f64>,
        captured: Vec<f64>,
        at_506: Vec<f64>,
        whole_boxes: Vec<[f64; 5]>,
        /// Only recorded for the models that answer the every-instance question.
        #[serde(default)]
        every_boxes: Vec<[f64; 5]>,
        window_boxes: Vec<[f64; 5]>,
        picks: BTreeMap<String, Vec<u8>>,
    }

    fn answers(scores: &[f64]) -> ApiResponse {
        ApiResponse {
            model: None,
            usage: None,
            choices: None,
            answers: Some(
                (1..)
                    .zip(scores)
                    .map(|(tile, &noul)| {
                        (format!("tile_{tile:02}"), JevAnswer { noul: Some(noul) })
                    })
                    .collect(),
            ),
            meta: None,
        }
    }

    /// A recorded grid decided under a plan, with the same steps in the same order as a live
    /// round. `whole_first` sets which box reply is heard first.
    fn decide(plan: &Plan, tuned: &'static Tuned, grid: &Recorded, whole_first: bool) -> Vec<u8> {
        let passes = plan
            .close_ups
            .iter()
            .map(|scale| match scale {
                None => answers(&grid.captured),
                Some(506) => answers(&grid.at_506),
                Some(other) => panic!("no recorded pass at {other} pixels"),
            })
            .collect();
        let close_ups = average_crop_passes(passes);
        let whole = answers(&grid.whole);
        let views = plan.gap.map(|rule| {
            (
                rule,
                jev_grid_scores(whole.answers.as_ref(), 16).unwrap(),
                jev_grid_scores(close_ups.answers.as_ref(), 16).unwrap(),
            )
        });
        let mut reply = blend_decision_replies(
            whole,
            close_ups,
            "scene-and-crop",
            4,
            plan.select_from,
            Combine::Logistic(tuned),
        );
        // The question was asked exactly where a score was recorded.
        if let Some((minimum, probability)) = plan.skip_below.zip(grid.presence) {
            skip_when_absent(&mut reply, minimum, Some(probability));
        }
        if let Some((rule, whole, close_ups)) = views {
            apply_gap_rule(&mut reply, rule, &whole, &close_ups);
        }
        if let Some(rule) = plan.boxes {
            let found = |boxes: &[[f64; 5]]| -> Vec<boxes::Found> {
                boxes
                    .iter()
                    .map(|&[left, top, right, bottom, confidence]| boxes::Found {
                        name: String::new(),
                        rect: [left, top, right, bottom],
                        confidence,
                    })
                    .collect()
            };
            let mut replies: Vec<BoxReply> = rule
                .asks()
                .map(|ask| {
                    let recorded = match ask {
                        BoxAsk::Whole => &grid.whole_boxes,
                        BoxAsk::Every => &grid.every_boxes,
                        BoxAsk::Windows => &grid.window_boxes,
                    };
                    (ask, Ok((found(recorded), Value::Null)))
                })
                .collect();
            if !whole_first {
                replies.reverse();
            }
            apply_box_rule(&mut reply, rule, false, replies.into_iter());
        }
        match reply.output {
            Output::Prediction(Prediction::Tiles(picks)) => picks,
            _ => panic!("{}: no selection", grid.id),
        }
    }

    // A fixture holds a model's recorded answers on graded grids and the squares each level
    // should select on them, worked out in Python by the sweep that chose the levels' settings.
    // So this fails if a level's plan, the model's weights or any rule drifts from what was
    // measured. `same` names neighbouring levels that are one plan on purpose.
    fn levels_select_what_was_measured(
        packed: &[u8],
        grid_count: usize,
        tuned: &'static Tuned,
        same: &[(&str, &str)],
    ) {
        use std::io::Read;
        let mut json = String::new();
        flate2::read::GzDecoder::new(packed)
            .read_to_string(&mut json)
            .unwrap();
        let grids: Vec<Recorded> = serde_json::from_str(&json).unwrap();
        assert_eq!(grids.len(), grid_count);
        let efforts = tuned.efforts.as_ref().unwrap();
        let mut answers_by_level = Vec::new();
        for (name, effort) in [
            ("low", Effort::Low),
            ("medium", Effort::Medium),
            ("high", Effort::High),
            ("max", Effort::Max),
        ] {
            let plan = efforts.plan(effort);
            let picked: Vec<_> = grids
                .iter()
                .map(|grid| decide(&plan, tuned, grid, true))
                .collect();
            let wrong: Vec<_> = grids
                .iter()
                .zip(&picked)
                .filter(|(grid, picks)| grid.picks[name] != **picks)
                .map(|(grid, picks)| format!("{}: {picks:?}, not {:?}", grid.id, grid.picks[name]))
                .collect();
            assert!(
                wrong.is_empty(),
                "{name}: {} grids differ: {:?}",
                wrong.len(),
                &wrong[..wrong.len().min(5)]
            );
            // Whichever box reply is heard first, the answer is the same.
            for (grid, picks) in grids.iter().zip(&picked) {
                assert_eq!(
                    &decide(&plan, tuned, grid, false),
                    picks,
                    "{name}, {}",
                    grid.id
                );
            }
            answers_by_level.push((name, picked));
        }
        // The levels are different plans, not one plan under four names, except where two are
        // meant to be the same.
        for pair in answers_by_level.windows(2) {
            let ((below, lower), (above, upper)) = (&pair[0], &pair[1]);
            let differ = lower.iter().zip(upper).filter(|(a, b)| a != b).count();
            if same.contains(&(below, above)) {
                assert_eq!(differ, 0, "{below} and {above} should be one plan");
            } else {
                assert!(
                    differ > 20,
                    "{below} and {above} differ on only {differ} grids"
                );
            }
        }
    }

    // vLLM/d3-ultra on 644 grids: the 311 labelled, and 333 live ones
    // (reports/labels/next/boxgrid/tok_v2_fixture.py writes the file).
    #[test]
    fn each_effort_level_selects_what_was_measured_on_the_recorded_grids() {
        levels_select_what_was_measured(
            include_bytes!("../tests/fixtures/vllm_d3_ultra_4x4.json.gz"),
            644,
            RECAPTCHA,
            &[],
        );
    }

    // Liquid's models on the first of three recorded passes over the 339 graded grids
    // (reports/labels/next/boxgrid/liquid/chosen.py writes the files). Liquid does not answer a
    // request the same way twice, so these recordings are the only way to check its levels.
    #[test]
    fn liquid_d1_2_levels_select_what_was_measured() {
        levels_select_what_was_measured(
            include_bytes!("../tests/fixtures/liquid_d1_2_4x4.json.gz"),
            339,
            &crate::captcha::recaptcha::liquid::d1_2::TUNED,
            &[("medium", "high"), ("high", "max")],
        );
    }

    // vLLM refuses the question type, so none of its plans may send it, and Liquid's levels
    // send it in place of the nine questions.
    #[test]
    fn only_liquid_plans_ask_for_every_instance() {
        let rules = |tuned: &Tuned| {
            let efforts = tuned.efforts.as_ref().unwrap();
            [Effort::Low, Effort::Medium, Effort::High, Effort::Max]
                .into_iter()
                .filter_map(|effort| efforts.plan(effort).boxes)
                .collect::<Vec<_>>()
        };
        assert!(rules(RECAPTCHA).iter().all(|rule| rule.every.is_none()));
        assert!(
            Plan::record(Provider::Vllm.draws_every_instance())
                .boxes
                .unwrap()
                .every
                .is_none()
        );
        for tuned in [
            &crate::captcha::recaptcha::liquid::d1_2::TUNED,
            &crate::captcha::recaptcha::liquid::d1_2_ultra::TUNED,
        ] {
            let rules = rules(tuned);
            assert!(!rules.is_empty());
            for rule in rules {
                assert_eq!(rule.asks().collect::<Vec<_>>(), [BoxAsk::Every]);
            }
        }
    }

    #[test]
    fn liquid_d1_2_ultra_levels_select_what_was_measured() {
        levels_select_what_was_measured(
            include_bytes!("../tests/fixtures/liquid_d1_2_ultra_4x4.json.gz"),
            339,
            &crate::captcha::recaptcha::liquid::d1_2_ultra::TUNED,
            &[("high", "max")],
        );
    }

    // A failed box request leaves the squares it would have decided selected, while a square
    // that a complete reply did not reach is dropped all the same.
    #[test]
    fn a_failed_box_request_leaves_undecided_squares_selected() {
        let scores: BTreeMap<u8, f64> = [(6, 0.9), (7, 0.9), (11, 0.99)].into();
        let reply = || Reply {
            output: Output::Prediction(Prediction::Tiles(vec![6, 7, 11])),
            resolved_model: None,
            usage: None,
            tile_probabilities: Some(scores.clone()),
            decision_components: Some(json!({})),
            provider_meta: None,
        };
        let rule = RECAPTCHA.efforts.as_ref().unwrap().max.boxes.unwrap();
        let on = |tile: u8| {
            let (row, column) = (f64::from((tile - 1) / 4), f64::from((tile - 1) % 4));
            boxes::Found {
                name: String::new(),
                rect: [
                    column / 4.0,
                    row / 4.0,
                    (column + 1.0) / 4.0,
                    (row + 1.0) / 4.0,
                ],
                confidence: 0.9,
            }
        };
        let picks = |reply: &Reply| match &reply.output {
            Output::Prediction(Prediction::Tiles(picks)) => picks.clone(),
            _ => unreachable!(),
        };
        // The window reply reaches square 6 only, then the whole-grid request fails.
        let mut round = reply();
        let heard = vec![
            (BoxAsk::Windows, Ok((vec![on(6)], Value::Null))),
            (BoxAsk::Whole, Err("timed out".to_owned())),
        ];
        apply_box_rule(&mut round, rule, false, heard.into_iter());
        assert_eq!(picks(&round), [6, 11]);
        let log = &round.decision_components.as_ref().unwrap()["boxes"];
        assert_eq!(log["removed"], json!([7]));
        assert_eq!(log["error"], "whole_grid: timed out");
        // With both replies in, square 6 needed the whole-grid box it did not get.
        let mut round = reply();
        let heard = vec![
            (BoxAsk::Windows, Ok((vec![on(6)], Value::Null))),
            (BoxAsk::Whole, Ok((vec![on(7)], Value::Null))),
        ];
        apply_box_rule(&mut round, rule, false, heard.into_iter());
        assert_eq!(picks(&round), [11]);
        // Once every square is decided the round stops listening: the second reply is not read.
        let mut round = reply();
        let mut heard = vec![(BoxAsk::Windows, Ok((vec![], Value::Null)))]
            .into_iter()
            .chain(std::iter::from_fn(|| -> Option<BoxReply> {
                panic!("listened for a reply that could change nothing")
            }));
        apply_box_rule(&mut round, rule, false, &mut heard);
        assert_eq!(picks(&round), [11]);
    }

    #[test]
    fn logistic_rule_uses_edge_neighbors_on_4x4() {
        let whole = tiles(&[
            0.240337, 0.750125, 0.69387, 0.613134, 0.957105, 0.995155, 0.991341, 0.993727,
            0.182279, 0.910848, 0.821835, 0.848625, 0.931801, 0.985958, 0.988057, 0.992624,
        ]);
        let crops = tiles(&[
            0.463749, 0.983213, 0.942063, 0.669614, 0.998127, 0.999454, 0.996333, 0.998662,
            0.019002, 0.972076, 0.725535, 0.974846, 0.996249, 0.988316, 0.991891, 0.991116,
        ]);
        let scores = logistic_scores(&whole, &crops, 4, RECAPTCHA).unwrap();
        assert_close(
            &scores,
            &[
                0.478572, 0.996758, 0.97775, 0.843709, 0.999814, 0.999958, 0.99973, 0.999865,
                0.0131823, 0.995649, 0.876105, 0.994475, 0.999007, 0.998126, 0.998719, 0.998614,
            ],
        );
        assert_eq!(edge_neighbors(1, 4).collect::<Vec<_>>(), [5, 2]);
        assert_eq!(edge_neighbors(6, 4).collect::<Vec<_>>(), [2, 10, 5, 7]);
        assert_eq!(edge_neighbors(16, 4).collect::<Vec<_>>(), [12, 15]);
    }
}
