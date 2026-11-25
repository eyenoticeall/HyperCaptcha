//! hCaptcha: token verification and the solver settings for its image grids.
//!
//! Only the grid task is handled: nine separate pictures and one instruction, such as "Pick the
//! flying animal". hCaptcha's other tasks (dragging a piece, clicking a point, following a
//! moving object) are left alone.

pub mod grid;
pub mod vllm {
    pub mod d3_ultra;
}

use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use super::{PhotoQuestion, Profile};
use crate::providers::configured_env;

/// hCaptcha's published test pair. It never shows a challenge and always verifies.
const TEST_SITE: &str = "10000000-ffff-ffff-ffff-000000000001";
const TEST_SECRET: &str = "0x0000000000000000000000000000000000000000";
/// hCaptcha refuses pages on `localhost` and `127.0.0.1`. Browsers resolve every name under
/// `.localhost` to this machine, so this one needs no hosts entry.
pub const DEFAULT_HOSTNAME: &str = "hypercaptcha.localhost";

/// What holds for hCaptcha's 3 × 3 grids whichever model answers them. Nothing has been
/// fitted: there are no labelled hCaptcha grids, so the cutoff is one half.
pub static PROFILE: Profile = Profile {
    layouts: &[3],
    target: instruction,
    allows_skipping: |_| false,
    photo: PhotoQuestion {
        state_key: "challenge_instructions",
        task: "PHOTO is one of nine separate pictures in an image challenge. Is PHOTO a picture that `challenge_instructions` asks the user to select? Judge what PHOTO shows. Ignore faint overlaid patterns, text and noise.",
    },
    average_cutoff_3x3: 0.5,
    models: &[&vllm::d3_ultra::MODEL],
};

/// hCaptcha's instructions do not name one object in a fixed place, so the model is given the
/// whole instruction.
fn instruction(prompt: &str, side: u8) -> Option<&str> {
    let prompt = prompt.trim();
    (side == 3 && !prompt.is_empty()).then_some(prompt)
}

pub struct Hcaptcha {
    pub site_key: String,
    pub test: bool,
    /// The name the lab page is served under, which hCaptcha reports back for a token.
    pub hostname: String,
    secret: String,
    client: Client,
}

#[derive(Default, Serialize)]
pub struct Verdict {
    pub hcaptcha_valid: bool,
    pub valid: bool,
    pub hostname: Option<String>,
    pub challenge_ts: Option<String>,
    pub score: Option<f64>,
    pub errors: Vec<String>,
}

#[derive(Deserialize)]
struct SiteVerification {
    success: bool,
    hostname: Option<String>,
    challenge_ts: Option<String>,
    score: Option<f64>,
    #[serde(rename = "error-codes", default)]
    errors: Vec<String>,
}

impl Hcaptcha {
    pub fn new(test: bool, hostname: String) -> Result<Self> {
        let key = |name: &str| configured_env(name).with_context(|| format!("set {name}"));
        let (site_key, secret) = if test {
            (TEST_SITE.into(), TEST_SECRET.into())
        } else {
            (key("HCAPTCHA_SITE_KEY")?, key("HCAPTCHA_SECRET_KEY")?)
        };
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()?;
        Ok(Self {
            site_key,
            test,
            hostname,
            secret,
            client,
        })
    }

    pub fn verify(&self, token: &str) -> Result<Verdict> {
        let response = self
            .client
            .post("https://api.hcaptcha.com/siteverify")
            .form(&[
                ("secret", self.secret.as_str()),
                ("response", token),
                ("sitekey", self.site_key.as_str()),
            ])
            .send()
            .map_err(|error| error.without_url())?;
        if !response.status().is_success() {
            bail!("hCaptcha returned HTTP {}", response.status());
        }
        let result: SiteVerification = response.json().map_err(|error| error.without_url())?;
        Ok(validate(
            Verdict {
                hcaptcha_valid: result.success,
                hostname: result.hostname,
                challenge_ts: result.challenge_ts,
                score: result.score,
                errors: result.errors,
                ..Verdict::default()
            },
            (!self.test).then_some(self.hostname.as_str()),
        ))
    }
}

fn validate(mut result: Verdict, hostname: Option<&str>) -> Verdict {
    // hCaptcha documents that it may answer "not-provided" in place of the hostname when
    // challenge traffic is unusually high, so that answer is not held against a token.
    if result.hcaptcha_valid
        && let Some(expected) = hostname
        && !matches!(result.hostname.as_deref(), Some(found) if found == expected || found == "not-provided")
    {
        result.errors.push("unexpected_hostname".into());
    }
    result.valid = result.hcaptcha_valid && result.errors.is_empty();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(success: bool, hostname: Option<&str>, errors: &[&str]) -> Verdict {
        Verdict {
            hcaptcha_valid: success,
            hostname: hostname.map(str::to_owned),
            errors: errors.iter().map(|&error| error.to_owned()).collect(),
            ..Verdict::default()
        }
    }

    #[test]
    fn a_token_is_valid_only_for_the_lab_hostname() {
        let host = Some(DEFAULT_HOSTNAME);
        assert!(validate(verdict(true, host, &[]), host).valid);
        assert!(validate(verdict(true, Some("not-provided"), &[]), host).valid);
        let elsewhere = validate(verdict(true, Some("example.com"), &[]), host);
        assert!(!elsewhere.valid);
        assert_eq!(elsewhere.errors, ["unexpected_hostname"]);
        assert!(!validate(verdict(true, None, &[]), host).valid);
        // The test keys verify without a hostname check.
        assert!(validate(verdict(true, Some("dummy-key-pass"), &[]), None).valid);
    }

    #[test]
    fn a_rejected_token_keeps_hcaptchas_error_codes() {
        let rejected = validate(
            verdict(false, None, &["invalid-input-response"]),
            Some(DEFAULT_HOSTNAME),
        );
        assert!(!rejected.valid);
        assert_eq!(rejected.errors, ["invalid-input-response"]);
    }

    #[test]
    fn the_whole_instruction_is_what_to_look_for() {
        assert_eq!(
            (PROFILE.target)(" Pick the flying animal \n", 3),
            Some("Pick the flying animal")
        );
        assert_eq!((PROFILE.target)("  ", 3), None);
        assert_eq!((PROFILE.target)("Pick the flying animal", 4), None);
    }
}
