mod boxes;
mod captcha;
mod liquid;
mod pillow;
mod plan;
mod providers;
mod server;

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};

use crate::{
    captcha::{Captcha, Service, hcaptcha, recaptcha},
    providers::ProviderOptions,
};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Open a local page for a real reCAPTCHA or hCaptcha integration.
    Serve {
        /// CAPTCHA service the page hosts. Each has its own keys and its own solver settings.
        #[arg(long, value_enum, default_value = "recaptcha")]
        captcha: Service,

        /// Google integration and key type, required for reCAPTCHA. Use policy-challenge for a visual challenge key.
        #[arg(long, value_enum)]
        mode: Option<recaptcha::Mode>,

        /// Use Google's public v2 integration test keys.
        #[arg(long)]
        google_test_key: bool,

        /// Use hCaptcha's public test keys, which never show a challenge.
        #[arg(long)]
        hcaptcha_test_key: bool,

        /// Port on localhost.
        #[arg(long, default_value_t = 8765)]
        port: u16,

        /// Hostname the service reports for your own keys. Defaults to localhost for reCAPTCHA. hCaptcha refuses localhost, so its page is served as hypercaptcha.localhost unless this names a hosts-file entry for 127.0.0.1.
        #[arg(long)]
        hostname: Option<String>,

        /// Optional JSON Lines file for model and verification results.
        #[arg(long)]
        log: Option<PathBuf>,

        #[command(flatten)]
        provider: ProviderOptions,
    },
}

fn parse_probability(value: &str) -> Result<f64, String> {
    match value.parse() {
        Ok(value) if (0.0..=1.0).contains(&value) => Ok(value),
        _ => Err("expected a finite number between 0 and 1".into()),
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Serve {
            captcha,
            mode,
            google_test_key,
            hcaptcha_test_key,
            port,
            hostname,
            log,
            provider,
        } => {
            let captcha = match captcha {
                Service::Recaptcha => {
                    ensure!(
                        !hcaptcha_test_key,
                        "--hcaptcha-test-key requires --captcha hcaptcha"
                    );
                    let mode = mode.context("--mode is required for reCAPTCHA")?;
                    let hostname = hostname.unwrap_or_else(|| "localhost".into());
                    Captcha::Recaptcha(recaptcha::Google::new(mode, google_test_key, hostname)?)
                }
                Service::Hcaptcha => {
                    ensure!(
                        mode.is_none() && !google_test_key,
                        "--mode and --google-test-key apply to reCAPTCHA"
                    );
                    let hostname = hostname.unwrap_or_else(|| hcaptcha::DEFAULT_HOSTNAME.into());
                    ensure!(
                        !matches!(hostname.as_str(), "localhost" | "127.0.0.1"),
                        "hCaptcha refuses pages on localhost. Leave --hostname out, or name a hosts-file entry for 127.0.0.1."
                    );
                    Captcha::Hcaptcha(hcaptcha::Hcaptcha::new(hcaptcha_test_key, hostname)?)
                }
            };
            let backend = providers::Backend::new(provider, captcha.profile())?;
            server::serve(port, captcha, backend, log.as_deref())
        }
    }
}
