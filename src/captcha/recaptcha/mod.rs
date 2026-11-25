use std::{sync::LazyLock, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use reqwest::{Url, blocking::Client, blocking::RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{Model, PhotoQuestion, Profile, Service};
use crate::providers::configured_env;

pub const ACTION: &str = "hypercaptcha_trial";
const TEST_SITE: &str = "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI";
const TEST_SECRET: &str = "6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe";

static MODELS: LazyLock<Vec<Model>> = LazyLock::new(|| {
    [
        include_str!("../../../models/recaptcha/vllm/d3-ultra.toml"),
        include_str!("../../../models/recaptcha/typesafe/jev.toml"),
        include_str!("../../../models/recaptcha/liquid/d1.2.toml"),
        include_str!("../../../models/recaptcha/liquid/d1.2-ultra.toml"),
    ]
    .into_iter()
    .map(|raw| Model::from_embedded_toml(raw, Service::Recaptcha).expect("valid built-in config"))
    .collect()
});

/// What holds for Google's image grids whichever model answers them.
pub static PROFILE: Profile = Profile {
    service: Service::Recaptcha,
    layouts: &[3, 4],
    target: challenge_target,
    allows_skipping,
    photo: PhotoQuestion {
        state_key: "target",
        task: "Does PHOTO contain at least one visible object of the type specified in `target`? Count a recognizable partial or distant object. The answer depends only on the contents of PHOTO. Other photo tiles are unrelated.",
    },
    average_cutoff_3x3: 0.6,
    models: &MODELS,
};

/// The object named on the second line of "Select all images with" or "Select all squares
/// with", falling back to a single-line prompt when no newline is present.
fn challenge_target(prompt: &str, side: u8) -> Option<&str> {
    let mut lines = prompt.lines();
    let instruction = lines.next()?.trim();
    let prefix = match side {
        3 => "Select all images with",
        4 => "Select all squares with",
        _ => return None,
    };
    let suffix = instruction.strip_prefix(prefix)?;
    if ["", " a", " an"].contains(&suffix) {
        return lines
            .next()
            .map(str::trim)
            .filter(|target| !target.is_empty());
    }
    if prompt.contains(['\n', '\r']) {
        return None;
    }
    let rest = suffix.strip_prefix(char::is_whitespace)?.trim_start();
    let target = ["a", "an"]
        .into_iter()
        .find_map(|article| {
            rest.strip_prefix(article)?
                .strip_prefix(char::is_whitespace)
        })
        .unwrap_or(rest)
        .trim()
        .trim_end_matches('.');
    (!target.is_empty()).then_some(target)
}

/// Whether the challenge text offers skipping, as in "If there are none, click skip".
fn allows_skipping(prompt: &str) -> bool {
    prompt
        .lines()
        .skip(2)
        .any(|line| line.to_lowercase().contains("skip"))
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    V2,
    V2Enterprise,
    V3,
    V3Enterprise,
    PolicyChallenge,
}

impl Mode {
    pub fn enterprise(self) -> bool {
        matches!(
            self,
            Self::V2Enterprise | Self::V3Enterprise | Self::PolicyChallenge
        )
    }

    pub fn score_based(self) -> bool {
        matches!(self, Self::V3 | Self::V3Enterprise)
    }

    fn expects_action(self) -> bool {
        self.score_based() || self == Self::PolicyChallenge
    }
}

pub struct Google {
    pub mode: Mode,
    pub site_key: String,
    pub test: bool,
    hostname: String,
    client: Client,
    auth: Auth,
}

enum Auth {
    SiteVerify(String),
    Enterprise {
        endpoint: Url,
        credentials: CloudCredentials,
    },
}

enum CloudCredentials {
    ApiKey(String),
    AccessToken(String),
}

impl CloudCredentials {
    fn from_env() -> Result<Self> {
        if let Some(key) = configured_env("GOOGLE_CLOUD_API_KEY") {
            return Ok(Self::ApiKey(key));
        }
        configured_env("GOOGLE_CLOUD_ACCESS_TOKEN")
            .map(Self::AccessToken)
            .context("set GOOGLE_CLOUD_API_KEY or GOOGLE_CLOUD_ACCESS_TOKEN")
    }

    fn attach(&self, request: RequestBuilder) -> RequestBuilder {
        match self {
            Self::ApiKey(key) => request.header("X-Goog-Api-Key", key),
            Self::AccessToken(token) => request.bearer_auth(token),
        }
    }
}

#[derive(Default, Serialize)]
pub struct Verdict {
    pub google_valid: bool,
    pub valid: bool,
    pub hostname: Option<String>,
    pub action: Option<String>,
    pub score: Option<f64>,
    pub challenge: Option<String>,
    pub reasons: Vec<String>,
    pub errors: Vec<String>,
    pub assessment: Option<String>,
}

#[derive(Deserialize)]
struct SiteVerification {
    success: bool,
    hostname: Option<String>,
    action: Option<String>,
    score: Option<f64>,
    #[serde(rename = "error-codes", default)]
    errors: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Assessment {
    name: Option<String>,
    token_properties: TokenProperties,
    risk_analysis: Option<RiskAnalysis>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TokenProperties {
    valid: bool,
    action: Option<String>,
    hostname: Option<String>,
    invalid_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct RiskAnalysis {
    score: Option<f64>,
    challenge: Option<String>,
    #[serde(default)]
    reasons: Vec<String>,
}

impl Google {
    pub fn new(mode: Mode, test: bool, hostname: String) -> Result<Self> {
        ensure!(
            !test || mode == Mode::V2,
            "--google-test-key requires --mode v2"
        );
        let site_key = if test {
            TEST_SITE.into()
        } else {
            required("RECAPTCHA_SITE_KEY")?
        };
        let auth = if test {
            Auth::SiteVerify(TEST_SECRET.into())
        } else if mode.enterprise() {
            let project = required("GOOGLE_CLOUD_PROJECT")?;
            let mut endpoint =
                Url::parse("https://recaptchaenterprise.googleapis.com/v1/projects/")?;
            endpoint
                .path_segments_mut()
                .expect("fixed HTTP endpoint")
                .pop_if_empty()
                .extend([&project, "assessments"]);
            Auth::Enterprise {
                endpoint,
                credentials: CloudCredentials::from_env()?,
            }
        } else {
            Auth::SiteVerify(required("RECAPTCHA_SECRET_KEY")?)
        };
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()?;
        Ok(Self {
            mode,
            site_key,
            test,
            hostname,
            client,
            auth,
        })
    }

    pub fn verify(&self, token: &str, user_agent: Option<&str>) -> Result<Verdict> {
        let request = match &self.auth {
            Auth::SiteVerify(secret) => self
                .client
                .post("https://www.google.com/recaptcha/api/siteverify")
                .form(&[("secret", secret.as_str()), ("response", token)]),
            Auth::Enterprise {
                endpoint,
                credentials,
            } => {
                let mut event = json!({"token": token, "siteKey": self.site_key});
                if let Some(user_agent) = user_agent {
                    event["userAgent"] = json!(user_agent);
                }
                if self.mode.expects_action() {
                    event["expectedAction"] = json!(ACTION);
                }
                credentials
                    .attach(self.client.post(endpoint.clone()))
                    .json(&json!({"event": event}))
            }
        };
        let response = request.send().map_err(|error| error.without_url())?;
        if !response.status().is_success() {
            bail!("Google returned HTTP {}", response.status());
        }
        let verdict = if self.mode.enterprise() {
            let assessment: Assessment = response.json().map_err(|error| error.without_url())?;
            let token = assessment.token_properties;
            let risk = assessment.risk_analysis.unwrap_or_default();
            Verdict {
                google_valid: token.valid,
                hostname: token.hostname,
                action: token.action,
                score: risk.score,
                challenge: risk.challenge,
                reasons: risk.reasons,
                errors: token
                    .invalid_reason
                    .filter(|reason| reason != "INVALID_REASON_UNSPECIFIED")
                    .into_iter()
                    .collect(),
                assessment: assessment.name,
                ..Verdict::default()
            }
        } else {
            let result: SiteVerification = response.json().map_err(|error| error.without_url())?;
            Verdict {
                google_valid: result.success,
                hostname: result.hostname,
                action: result.action,
                score: result.score,
                errors: result.errors,
                ..Verdict::default()
            }
        };
        Ok(validate(
            verdict,
            self.mode,
            (!self.test).then_some(self.hostname.as_str()),
        ))
    }
}

fn validate(mut result: Verdict, mode: Mode, hostname: Option<&str>) -> Verdict {
    if result.google_valid {
        if let Some(hostname) = hostname
            && result.hostname.as_deref() != Some(hostname)
        {
            result.errors.push("unexpected_hostname".into());
        }
        if mode.expects_action() && result.action.as_deref() != Some(ACTION) {
            result.errors.push("unexpected_action".into());
        }
        if mode.score_based()
            && !result
                .score
                .is_some_and(|score| (0.0..=1.0).contains(&score))
        {
            result.errors.push("missing_or_invalid_score".into());
        }
        if mode == Mode::PolicyChallenge && result.challenge.as_deref() == Some("FAILED") {
            result.errors.push("challenge_failed".into());
        }
    }
    result.valid = result.google_valid && result.errors.is_empty();
    result
}

fn required(name: &str) -> Result<String> {
    configured_env(name).with_context(|| format!("set {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_fail_open_v3_and_binds_action_and_host() {
        let result = validate(
            Verdict {
                google_valid: true,
                hostname: Some("localhost".into()),
                action: Some("some_other_action".into()),
                score: Some(0.9),
                errors: vec!["Over free quota.".into()],
                ..Verdict::default()
            },
            Mode::V3,
            Some("research.example"),
        );
        assert!(!result.valid);
        assert_eq!(
            result.errors,
            [
                "Over free quota.",
                "unexpected_hostname",
                "unexpected_action"
            ]
        );
    }

    #[test]
    fn accepts_realistic_checkbox_and_score_responses() {
        let checkbox = Verdict {
            google_valid: true,
            ..Verdict::default()
        };
        assert!(validate(checkbox, Mode::V2, None).valid);
        let score = Verdict {
            google_valid: true,
            hostname: Some("localhost".into()),
            action: Some(ACTION.into()),
            score: Some(0.0),
            ..Verdict::default()
        };
        assert!(validate(score, Mode::V3Enterprise, Some("localhost")).valid);
    }

    #[test]
    fn policy_challenge_verifies_the_action_and_records_the_challenge_outcome() {
        for (action, challenge, expected) in [
            (ACTION, "PASSED", true),
            (ACTION, "FAILED", false),
            (ACTION, "NOCAPTCHA", true),
            ("another_action", "PASSED", false),
        ] {
            let verdict = Verdict {
                google_valid: true,
                hostname: Some("localhost".into()),
                action: Some(action.into()),
                challenge: Some(challenge.into()),
                ..Verdict::default()
            };
            let result = validate(verdict, Mode::PolicyChallenge, Some("localhost"));
            assert_eq!(result.valid, expected);
            assert_eq!(result.challenge.as_deref(), Some(challenge));
        }
    }

    #[test]
    fn challenge_target_supports_multiline_and_single_line_prompts() {
        // Standard multi-line reCAPTCHA prompts keep top priority.
        assert_eq!(
            challenge_target("Select all images with\nbicycles", 3),
            Some("bicycles")
        );
        assert_eq!(
            challenge_target("Select all images with a\nfire hydrant", 3),
            Some("fire hydrant")
        );
        assert_eq!(
            challenge_target("Select all images with an\nairplane", 3),
            Some("airplane")
        );
        assert_eq!(
            challenge_target(
                "Select all squares with\ntraffic lights\nIf there are none, click skip",
                4
            ),
            Some("traffic lights")
        );
        assert!(allows_skipping(
            "Select all squares with\ntraffic lights\nIf there are none, click skip"
        ));

        // Single-line manual web UI fallback when no newline is present.
        assert_eq!(
            challenge_target("Select all images with bicycles", 3),
            Some("bicycles")
        );
        assert_eq!(
            challenge_target("Select all images with bicycles.", 3),
            Some("bicycles")
        );
        assert_eq!(
            challenge_target("Select all images with a fire hydrant", 3),
            Some("fire hydrant")
        );
        assert_eq!(
            challenge_target("Select all images with an airplane", 3),
            Some("airplane")
        );
        assert_eq!(
            challenge_target("Select all images with apples", 3),
            Some("apples")
        );
        assert_eq!(
            challenge_target("Select all images with antlers", 3),
            Some("antlers")
        );
        assert_eq!(
            challenge_target("Select all squares with traffic lights", 4),
            Some("traffic lights")
        );
        assert_eq!(
            challenge_target("Select all squares with a bus.", 4),
            Some("bus")
        );

        // Reject missing/empty targets, mismatched layouts, non-whitespace suffixes, and
        // multi-line prompts whose first line is not the exact prefix.
        for (prompt, side) in [
            ("", 3),
            ("Select all images with", 3),
            ("Select all images with   ", 3),
            ("Select all images with a", 3),
            ("Select all images with an", 3),
            ("Select all images with .", 3),
            ("Select all images with a .", 3),
            ("Select all images with\n", 3),
            ("Select all images with\n   \nClick verify", 3),
            ("Select all images without bicycles", 3),
            ("Select all bicycles", 3),
            ("Select all images with bicycles", 4),
            ("Select all squares with traffic lights", 3),
            ("Select all images with bicycles\ncars", 3),
            (
                "Select all squares with traffic lights\nIf there are none, click skip",
                4,
            ),
            ("Select all images with\nbicycles", 5),
        ] {
            assert_eq!(
                challenge_target(prompt, side),
                None,
                "expected None for ({prompt:?}, {side})"
            );
        }
    }
}
