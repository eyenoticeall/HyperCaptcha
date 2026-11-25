//! The CAPTCHA services the lab can host and declarative TOML model configurations.
//!
//! Each service has its own module holding everything that is particular to it: how a token is
//! verified, how its challenge text is read, and its built-in solver configs. A setting measured
//! on one service's pictures is not assumed to hold for another's, and one measured with one
//! model is not assumed to hold for another model. So a service's [`Profile`] holds what follows
//! from its challenges alone, and each built-in model lives in a TOML file under
//! `models/<service>/<provider>/<model>.toml`, embedded via `include_str!`.

pub mod hcaptcha;
pub mod recaptcha;

use std::{collections::BTreeMap, fs, path::Path, sync::LazyLock};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use reqwest::header::{HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tiny_skia::FilterQuality;

use crate::{
    liquid,
    plan::{Efforts, Plan, validate_probability},
    providers::{DecisionImageFormat, Provider, VllmDecision, validate_base_url},
};

pub const MODEL_CONFIG_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
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

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Recaptcha => "recaptcha",
            Self::Hcaptcha => "hcaptcha",
        }
    }
}

/// Wire format used to carry pictures in System One decision requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireDialect {
    /// Named pictures in `assets`, referenced by name in `state` and `questions` (vLLM, TypeSafe).
    #[serde(rename = "systemone-assets")]
    SystemOneAssets,
    /// Ordered `data:image/...` URLs in `images`, referenced by position (`"image 1"`, etc.; Liquid).
    #[serde(rename = "systemone-images")]
    SystemOneImages,
}

/// Resampling filter used when enlarging 4 × 4 tile close-ups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CropFilter {
    Bilinear,
    Nearest,
}

impl CropFilter {
    pub fn filter_quality(self) -> FilterQuality {
        match self {
            Self::Bilinear => FilterQuality::Bilinear,
            Self::Nearest => FilterQuality::Nearest,
        }
    }
}

/// Optional endpoint configuration table (`[endpoint]`) in a TOML model config.
///
/// Optional for built-in providers (inherits provider defaults; specified fields override).
/// Required when `provider = "custom"`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointConfig {
    pub dialect: Option<WireDialect>,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub extra_api_keys_env: Option<String>,
    pub image_format: Option<DecisionImageFormat>,
    pub crop_filter: Option<CropFilter>,
    pub gzip: Option<bool>,
    pub max_images: Option<usize>,
    pub multi_state_windows: Option<bool>,
    pub supports_every_box: Option<bool>,
    pub return_meta: Option<bool>,
    pub hedge_on_stall: Option<bool>,
    pub warm_connections: Option<bool>,
    pub min_patch_tokens: Option<bool>,
    pub extra_headers: Option<BTreeMap<String, String>>,
}

/// Fully resolved HTTP and wire capabilities for a decision provider endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireProfile {
    pub dialect: WireDialect,
    pub base_url: String,
    /// Environment variable that overrides `base_url` for built-in providers when not
    /// overridden by CLI `--base-url` or an external `--model-config`.
    pub base_url_env: Option<String>,
    /// Primary API key environment variable names checked in order.
    pub api_key_envs: Vec<String>,
    /// Optional environment variable holding comma-separated extra API keys for rotation.
    pub extra_api_keys_env: Option<String>,
    pub image_format: DecisionImageFormat,
    pub crop_filter: CropFilter,
    pub gzip: bool,
    pub max_images: usize,
    pub multi_state_windows: bool,
    pub supports_every_box: bool,
    pub return_meta: bool,
    pub hedge_on_stall: bool,
    pub warm_connections: bool,
    pub min_patch_tokens: bool,
    pub extra_headers: BTreeMap<String, String>,
    pub names_errors: bool,
}

impl WireProfile {
    /// Built-in wire defaults for a built-in decision provider.
    pub fn for_provider(provider: Provider) -> Result<Self> {
        Ok(match provider {
            Provider::Vllm => Self {
                dialect: WireDialect::SystemOneAssets,
                base_url: "https://app.vllm-sr.ai/v1".into(),
                base_url_env: Some("VLLM_SR_BASE_URL".into()),
                api_key_envs: vec!["VLLM_SR_API_KEY".into()],
                extra_api_keys_env: Some("VLLM_SR_EXTRA_API_KEYS".into()),
                image_format: DecisionImageFormat::Jpeg,
                crop_filter: CropFilter::Bilinear,
                gzip: true,
                max_images: 16,
                multi_state_windows: true,
                supports_every_box: false,
                return_meta: true,
                hedge_on_stall: true,
                warm_connections: true,
                min_patch_tokens: false,
                extra_headers: BTreeMap::new(),
                names_errors: true,
            },
            Provider::Liquid => Self {
                dialect: WireDialect::SystemOneImages,
                base_url: "https://api.liquid.ai/decisions/v1".into(),
                base_url_env: Some("LIQUID_BASE_URL".into()),
                api_key_envs: vec!["LIQUID_API_KEY".into()],
                extra_api_keys_env: None,
                image_format: DecisionImageFormat::Jpeg,
                crop_filter: CropFilter::Bilinear,
                gzip: false,
                max_images: liquid::MOST_IMAGES,
                multi_state_windows: false,
                supports_every_box: true,
                return_meta: false,
                hedge_on_stall: false,
                warm_connections: false,
                min_patch_tokens: true,
                extra_headers: BTreeMap::new(),
                names_errors: true,
            },
            Provider::Jev => Self {
                dialect: WireDialect::SystemOneAssets,
                base_url: "https://api.typesafe.ai/v1".into(),
                base_url_env: Some("JEV_BASE_URL".into()),
                api_key_envs: vec!["TYPESAFE_API_KEY".into(), "JEV_API_KEY".into()],
                extra_api_keys_env: None,
                image_format: DecisionImageFormat::Png,
                crop_filter: CropFilter::Nearest,
                gzip: false,
                max_images: 16,
                multi_state_windows: true,
                supports_every_box: false,
                return_meta: false,
                hedge_on_stall: false,
                warm_connections: false,
                min_patch_tokens: false,
                extra_headers: BTreeMap::from([(
                    "TypeSafe-Danger-Zone-Images".into(),
                    "true".into(),
                )]),
                names_errors: false,
            },
            Provider::Custom => {
                bail!("provider \"custom\" requires an [endpoint] section in --model-config");
            }
            Provider::OpenaiChat | Provider::OpenaiCompletions => {
                bail!("OpenAI providers are not decision endpoints");
            }
        })
    }

    /// Resolves a provider's wire profile, inheriting defaults for built-in providers and
    /// requiring `[endpoint]` fields for `Provider::Custom`.
    pub fn resolve(provider: Provider, overrides: Option<&EndpointConfig>) -> Result<Self> {
        let profile = if provider == Provider::Custom {
            let endpoint = overrides.context(
                "provider \"custom\" requires an [endpoint] section in the model config",
            )?;
            let dialect = endpoint
                .dialect
                .context("provider \"custom\" requires endpoint.dialect")?;
            let base_url = endpoint
                .base_url
                .clone()
                .context("provider \"custom\" requires endpoint.base_url")?;
            let api_key_env = endpoint
                .api_key_env
                .clone()
                .context("provider \"custom\" requires endpoint.api_key_env")?;
            Self {
                dialect,
                base_url,
                base_url_env: None,
                api_key_envs: vec![api_key_env],
                extra_api_keys_env: endpoint.extra_api_keys_env.clone(),
                image_format: endpoint.image_format.unwrap_or(DecisionImageFormat::Jpeg),
                crop_filter: endpoint.crop_filter.unwrap_or(CropFilter::Bilinear),
                gzip: endpoint.gzip.unwrap_or(false),
                max_images: endpoint.max_images.unwrap_or(match dialect {
                    WireDialect::SystemOneAssets => 16,
                    WireDialect::SystemOneImages => liquid::MOST_IMAGES,
                }),
                multi_state_windows: endpoint
                    .multi_state_windows
                    .unwrap_or(dialect == WireDialect::SystemOneAssets),
                supports_every_box: endpoint.supports_every_box.unwrap_or(false),
                return_meta: endpoint.return_meta.unwrap_or(false),
                hedge_on_stall: endpoint.hedge_on_stall.unwrap_or(false),
                warm_connections: endpoint.warm_connections.unwrap_or(false),
                min_patch_tokens: endpoint.min_patch_tokens.unwrap_or(false),
                extra_headers: endpoint.extra_headers.clone().unwrap_or_default(),
                names_errors: true,
            }
        } else {
            let mut base = Self::for_provider(provider)?;
            if let Some(endpoint) = overrides {
                if let Some(dialect) = endpoint.dialect {
                    base.dialect = dialect;
                }
                if let Some(base_url) = &endpoint.base_url {
                    base.base_url = base_url.clone();
                }
                if let Some(api_key_env) = &endpoint.api_key_env {
                    base.api_key_envs = vec![api_key_env.clone()];
                }
                if let Some(extra_api_keys_env) = &endpoint.extra_api_keys_env {
                    base.extra_api_keys_env = Some(extra_api_keys_env.clone());
                }
                if let Some(image_format) = endpoint.image_format {
                    base.image_format = image_format;
                }
                if let Some(crop_filter) = endpoint.crop_filter {
                    base.crop_filter = crop_filter;
                }
                if let Some(gzip) = endpoint.gzip {
                    base.gzip = gzip;
                }
                if let Some(max_images) = endpoint.max_images {
                    base.max_images = max_images;
                }
                if let Some(multi_state_windows) = endpoint.multi_state_windows {
                    base.multi_state_windows = multi_state_windows;
                }
                if let Some(supports_every_box) = endpoint.supports_every_box {
                    base.supports_every_box = supports_every_box;
                }
                if let Some(return_meta) = endpoint.return_meta {
                    base.return_meta = return_meta;
                }
                if let Some(hedge_on_stall) = endpoint.hedge_on_stall {
                    base.hedge_on_stall = hedge_on_stall;
                }
                if let Some(warm_connections) = endpoint.warm_connections {
                    base.warm_connections = warm_connections;
                }
                if let Some(min_patch_tokens) = endpoint.min_patch_tokens {
                    base.min_patch_tokens = min_patch_tokens;
                }
                if let Some(extra_headers) = &endpoint.extra_headers {
                    base.extra_headers = extra_headers.clone();
                }
            }
            base
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<()> {
        validate_base_url(&self.base_url)?;
        ensure!(
            !self.api_key_envs.is_empty()
                && self.api_key_envs.iter().all(|name| !name.trim().is_empty()),
            "endpoint.api_key_env must be non-empty"
        );
        if let Some(extra) = &self.extra_api_keys_env {
            ensure!(
                !extra.trim().is_empty(),
                "endpoint.extra_api_keys_env must be non-empty when set"
            );
        }
        ensure!(
            self.max_images >= 1,
            "endpoint.max_images must be at least 1"
        );
        for (name, value) in &self.extra_headers {
            HeaderName::from_bytes(name.as_bytes())
                .with_context(|| format!("invalid endpoint.extra_headers name {name:?}"))?;
            HeaderValue::from_str(value)
                .with_context(|| format!("invalid endpoint.extra_headers value for {name:?}"))?;
        }
        Ok(())
    }
}

/// The 3 × 3 logistic rule: weights on clipped log-odds for the whole grid and the individual
/// photo, then the intercept, and the probability from which a tile is selected.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Logistic {
    pub weights: [f64; 3],
    pub cutoff: f64,
}

impl Logistic {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.weights.iter().all(|weight| weight.is_finite()),
            "logistic_3x3.weights must all be finite"
        );
        validate_probability("logistic_3x3.cutoff", self.cutoff)?;
        Ok(())
    }
}

/// The 4 × 4 logistic rule's weights on clipped log-odds: whole grid, close-up, strongest
/// edge-adjacent whole grid, strongest edge-adjacent close-up, intercept. A plan's
/// `select_from` is its cutoff.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Logistic4x4 {
    pub weights: [f64; 5],
}

impl Logistic4x4 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.weights.iter().all(|weight| weight.is_finite()),
            "logistic_4x4.weights must all be finite"
        );
        Ok(())
    }
}

fn default_decision_average() -> VllmDecision {
    VllmDecision::Average
}

/// What was measured with one model on one service's image grids, asking about every tile
/// twice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tuned {
    /// How the model's two views of a tile are combined unless a flag says otherwise.
    #[serde(default = "default_decision_average")]
    pub default_decision: VllmDecision,
    /// None when no rule has been fitted.
    #[serde(default)]
    pub logistic_3x3: Option<Logistic>,
    /// None when no 4 × 4 rule has been fitted.
    #[serde(default)]
    pub logistic_4x4: Option<Logistic4x4>,
    /// How a 4 × 4 round is asked when its two views are averaged instead.
    #[serde(default = "Plan::plain")]
    pub average_4x4: Plan,
    /// How a 4 × 4 round is asked and decided under the logistic rule, at each effort level.
    /// None when no levels have been measured.
    #[serde(default)]
    pub efforts: Option<Efforts>,
}

impl Tuned {
    pub fn validate(&self, supports_every_box: bool) -> Result<()> {
        if let Some(logistic_3x3) = &self.logistic_3x3 {
            logistic_3x3.validate()?;
        }
        if let Some(logistic_4x4) = &self.logistic_4x4 {
            logistic_4x4.validate()?;
        }
        self.average_4x4
            .validate(supports_every_box)
            .map_err(|err| anyhow::anyhow!("tuning.average_4x4: {err}"))?;
        if let Some(efforts) = &self.efforts {
            efforts
                .validate(supports_every_box)
                .map_err(|err| anyhow::anyhow!("tuning.{err}"))?;
        }
        Ok(())
    }
}

/// Declarative TOML model configuration schema (`schema_version = 1`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub schema_version: u32,
    pub service: Service,
    pub name: String,
    pub provider: Provider,
    pub model_id: String,
    #[serde(default)]
    pub endpoint: Option<EndpointConfig>,
    #[serde(default)]
    pub tuning: Option<Tuned>,
}

impl ModelConfig {
    pub fn from_toml(raw: &str) -> Result<Self> {
        let config: Self = toml::from_str(raw).context("parse TOML model config")?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("read model config {}", path.display()))?;
        Self::from_toml(&raw).with_context(|| format!("invalid model config {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == MODEL_CONFIG_SCHEMA_VERSION,
            "unsupported model config schema_version {} (expected {MODEL_CONFIG_SCHEMA_VERSION})",
            self.schema_version
        );
        ensure!(
            !self.name.trim().is_empty(),
            "model config `name` must be non-empty"
        );
        ensure!(
            !self.model_id.trim().is_empty(),
            "model config `model_id` must be non-empty"
        );
        ensure!(
            self.provider.is_decision(),
            "model config `provider` must be a decision provider (vllm, liquid, jev, or custom)"
        );
        let wire = WireProfile::resolve(self.provider, self.endpoint.as_ref())?;
        if let Some(tuning) = &self.tuning {
            tuning.validate(wire.supports_every_box)?;
        }
        Ok(())
    }

    pub fn into_model(self) -> Result<Model> {
        self.validate()?;
        let has_explicit_base_url = self
            .endpoint
            .as_ref()
            .is_some_and(|endpoint| endpoint.base_url.is_some());
        let wire = WireProfile::resolve(self.provider, self.endpoint.as_ref())?;
        Ok(Model {
            name: self.name,
            provider: self.provider,
            id: self.model_id,
            wire,
            has_explicit_base_url,
            tuned: self.tuning,
        })
    }
}

/// The question a decision model is asked about one tile shown by itself.
pub struct PhotoQuestion {
    /// The request's state field that carries what to look for.
    pub state_key: &'static str,
    pub task: &'static str,
}

/// What holds for one service's image grids whichever model answers them.
pub struct Profile {
    pub service: Service,
    /// Tiles along one side of the grids this service shows.
    pub layouts: &'static [u8],
    /// What to look for, read from the challenge text of a grid with `side` tiles a side.
    pub target: fn(prompt: &str, side: u8) -> Option<&str>,
    /// Whether the challenge text offers skipping when nothing matches.
    pub allows_skipping: fn(prompt: &str) -> bool,
    pub photo: PhotoQuestion,
    /// Cutoff for the average of two views of a 3 × 3 grid.
    pub average_cutoff_3x3: f64,
    /// Built-in models used on this service's grids. A provider's first entry is its default.
    models: &'static LazyLock<Vec<Model>>,
}

impl Profile {
    pub fn models(&self) -> &'static [Model] {
        self.models.as_slice()
    }

    /// The entry for the model a provider knows as `id`.
    pub fn model(&self, provider: Provider, id: &str) -> Option<&'static Model> {
        self.of(provider).find(|model| model.id == id)
    }

    /// The entry called `name`, such as "vLLM/d3-ultra", whatever its capitals.
    pub fn named(&self, name: &str) -> Option<&'static Model> {
        self.models()
            .iter()
            .find(|model| model.name.eq_ignore_ascii_case(name))
    }

    /// A provider's entries, its default first.
    pub fn of(&self, provider: Provider) -> impl Iterator<Item = &'static Model> {
        self.models()
            .iter()
            .filter(move |model| model.provider == provider)
    }
}

/// A validated model as one service's grids are answered with it.
#[derive(Clone, Debug)]
pub struct Model {
    /// Who serves the model, then the model: "vLLM/d3-ultra". Two providers can serve models
    /// with similar names, so settings, logs and messages use this name.
    pub name: String,
    pub provider: Provider,
    /// What the provider calls the model, as sent in a request.
    pub id: String,
    pub wire: WireProfile,
    /// Whether the config's `[endpoint]` explicitly provided a `base_url`.
    pub has_explicit_base_url: bool,
    /// What was measured with the model when it is asked about every tile twice, on the whole
    /// grid and on close-ups. None for a model that is asked another way or not yet tuned.
    pub tuned: Option<Tuned>,
}

impl Model {
    pub fn from_embedded_toml(raw: &str, expected_service: Service) -> Result<Self> {
        let config = ModelConfig::from_toml(raw)?;
        ensure!(
            config.service == expected_service,
            "embedded model config {} is for {}, expected {}",
            config.name,
            config.service.as_str(),
            expected_service.as_str()
        );
        config.into_model()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_builtin_and_example_toml_configs_parse_and_validate() {
        assert_eq!(recaptcha::PROFILE.models().len(), 4);
        assert_eq!(hcaptcha::PROFILE.models().len(), 1);
        assert!(recaptcha::PROFILE.named("vLLM/d3-ultra").is_some());
        assert!(recaptcha::PROFILE.named("Liquid/d1.2").is_some());
        assert!(recaptcha::PROFILE.named("Liquid/d1.2-ultra").is_some());
        assert!(recaptcha::PROFILE.named("typesafe/Jev").is_some());
        assert!(hcaptcha::PROFILE.named("vLLM/d3-ultra").is_some());

        let example = include_str!("../../models/examples/custom-decision-model.toml");
        let parsed = ModelConfig::from_toml(example).unwrap();
        assert_eq!(parsed.provider, Provider::Custom);
        assert_eq!(parsed.service, Service::Recaptcha);
        let model = parsed.into_model().unwrap();
        assert_eq!(model.wire.dialect, WireDialect::SystemOneAssets);
        assert_eq!(model.wire.api_key_envs, vec!["MY_DECISION_API_KEY"]);

        // Built-in Jev inherits both primary/fallback API key env vars and image header.
        let jev = recaptcha::PROFILE.named("typesafe/Jev").unwrap();
        assert_eq!(
            jev.wire.api_key_envs,
            vec!["TYPESAFE_API_KEY", "JEV_API_KEY"]
        );
        assert_eq!(
            jev.wire
                .extra_headers
                .get("TypeSafe-Danger-Zone-Images")
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(jev.wire.crop_filter, CropFilter::Nearest);
        assert_eq!(jev.wire.image_format, DecisionImageFormat::Png);

        // Built-in provider partial [endpoint] override inherits unspecified defaults.
        let partial = ModelConfig::from_toml(
            r#"
            schema_version = 1
            service = "recaptcha"
            name = "vLLM/custom-override"
            provider = "vllm"
            model_id = "vllm-sr/custom"

            [endpoint]
            gzip = false
            max_images = 10
            "#,
        )
        .unwrap()
        .into_model()
        .unwrap();
        assert_eq!(partial.wire.dialect, WireDialect::SystemOneAssets);
        assert_eq!(partial.wire.base_url, "https://app.vllm-sr.ai/v1");
        assert_eq!(partial.wire.api_key_envs, vec!["VLLM_SR_API_KEY"]);
        assert!(!partial.wire.gzip);
        assert_eq!(partial.wire.max_images, 10);
        assert!(partial.wire.return_meta);
        assert!(!partial.has_explicit_base_url);
    }

    #[test]
    fn invalid_model_configs_are_rejected() {
        // Unsupported schema version.
        let bad_version = r#"
            schema_version = 2
            service = "recaptcha"
            name = "vLLM/test"
            provider = "vllm"
            model_id = "test"
        "#;
        assert!(
            ModelConfig::from_toml(bad_version)
                .unwrap_err()
                .to_string()
                .contains("schema_version")
        );

        // Custom provider without [endpoint].
        let missing_endpoint = r#"
            schema_version = 1
            service = "recaptcha"
            name = "Custom/test"
            provider = "custom"
            model_id = "test"
        "#;
        assert!(
            ModelConfig::from_toml(missing_endpoint)
                .unwrap_err()
                .to_string()
                .contains("[endpoint]")
        );

        // Box rule asking for `every` when supports_every_box is false.
        let bad_every = r#"
            schema_version = 1
            service = "recaptcha"
            name = "vLLM/test"
            provider = "vllm"
            model_id = "test"

            [tuning]
            default_decision = "logistic"

            [tuning.average_4x4]
            close_ups = ["capture"]
            select_from = 0.5

            [tuning.average_4x4.boxes]
            exempt_from = 0.9
            every = 0.3
            cover = 0.01
            inside = 0.0
        "#;
        assert!(
            ModelConfig::from_toml(bad_every)
                .unwrap_err()
                .to_string()
                .contains("supports_every_box")
        );

        // Probability out of range.
        let bad_prob = r#"
            schema_version = 1
            service = "recaptcha"
            name = "vLLM/test"
            provider = "vllm"
            model_id = "test"

            [tuning]
            default_decision = "average"

            [tuning.average_4x4]
            close_ups = ["capture"]
            select_from = 1.5
        "#;
        assert!(
            ModelConfig::from_toml(bad_prob)
                .unwrap_err()
                .to_string()
                .contains("0..=1")
        );

        // Empty close_ups.
        let empty_close_ups = r#"
            schema_version = 1
            service = "recaptcha"
            name = "vLLM/test"
            provider = "vllm"
            model_id = "test"

            [tuning]
            default_decision = "average"

            [tuning.average_4x4]
            close_ups = []
            select_from = 0.5
        "#;
        assert!(
            ModelConfig::from_toml(empty_close_ups)
                .unwrap_err()
                .to_string()
                .contains("close_ups")
        );

        // Zero max_images.
        let bad_max_images = r#"
            schema_version = 1
            service = "recaptcha"
            name = "vLLM/test"
            provider = "vllm"
            model_id = "test"

            [endpoint]
            max_images = 0
        "#;
        assert!(
            ModelConfig::from_toml(bad_max_images)
                .unwrap_err()
                .to_string()
                .contains("max_images")
        );
    }
}
