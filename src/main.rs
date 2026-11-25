mod boxes;
mod captcha;
mod envfile;
mod liquid;
mod pillow;
mod plan;
mod providers;
mod server;

use std::path::PathBuf;

use anyhow::{Result, bail, ensure};
use clap::{Parser, Subcommand};
use serde::Serialize;

use crate::{
    captcha::{Captcha, ModelConfig, Service, hcaptcha, recaptcha},
    plan::Effort,
    providers::{Provider, ProviderOptions, VllmDecision},
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
        /// CAPTCHA service the page hosts. Defaults to hcaptcha when --hcaptcha-test-key is passed, otherwise recaptcha.
        #[arg(long, visible_alias = "service", value_enum)]
        captcha: Option<Service>,

        /// Google integration and key type, required for reCAPTCHA unless --google-test-key is set (which defaults to v2). Use policy-challenge for a visual challenge key.
        #[arg(long, value_enum)]
        mode: Option<recaptcha::Mode>,

        /// Use Google's public v2 integration test keys (defaults --mode to v2).
        #[arg(long)]
        google_test_key: bool,

        /// Use hCaptcha's public test keys, which never show a challenge (defaults --captcha to hcaptcha).
        #[arg(long)]
        hcaptcha_test_key: bool,

        /// Load environment variables from one or more .env files (repeatable; process environment takes precedence and later files override earlier ones).
        #[arg(long = "env-file", value_name = "PATH")]
        env_files: Vec<PathBuf>,

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
        provider: Box<ProviderOptions>,
    },

    /// List built-in models by service, provider, required env vars, decision rule, and effort tiers.
    Models {
        /// Filter to a single CAPTCHA service.
        #[arg(long, visible_alias = "captcha", value_enum)]
        service: Option<Service>,

        /// Output machine-readable JSON.
        #[arg(long)]
        json: bool,
    },

    /// Validate one or more TOML model configuration files offline.
    ValidateConfig {
        /// Paths to TOML model configuration files (`schema_version = 1`).
        #[arg(required = true, value_name = "PATH")]
        paths: Vec<PathBuf>,
    },
}

#[derive(Serialize)]
struct ModelEntry {
    service: Service,
    name: String,
    provider: Provider,
    model_id: String,
    required_env: Vec<String>,
    optional_env: Vec<String>,
    base_url: String,
    default_decision: Option<VllmDecision>,
    efforts: Vec<Effort>,
    default_effort: Option<Effort>,
}

fn parse_probability(value: &str) -> Result<f64, String> {
    match value.parse() {
        Ok(value) if (0.0..=1.0).contains(&value) => Ok(value),
        _ => Err("expected a finite number between 0 and 1".into()),
    }
}

fn builtin_model_entries(filter: Option<Service>) -> Vec<ModelEntry> {
    let services: &[Service] = match filter {
        Some(Service::Recaptcha) => &[Service::Recaptcha],
        Some(Service::Hcaptcha) => &[Service::Hcaptcha],
        None => &[Service::Recaptcha, Service::Hcaptcha],
    };
    let mut entries = Vec::new();
    for &service in services {
        for model in service.profile().models() {
            let default_decision = model.provider.asks_twice().then(|| {
                model
                    .tuned
                    .as_ref()
                    .map_or(VllmDecision::Average, |tuned| tuned.default_decision)
            });
            let has_efforts = model
                .tuned
                .as_ref()
                .is_some_and(|tuned| tuned.efforts.is_some());
            let efforts = if has_efforts {
                Effort::ALL.to_vec()
            } else {
                Vec::new()
            };
            let default_effort = (has_efforts && default_decision == Some(VllmDecision::Logistic))
                .then(Effort::default);
            let optional_env = model
                .wire
                .extra_api_keys_env
                .iter()
                .cloned()
                .chain(model.wire.base_url_envs.iter().cloned())
                .collect();
            entries.push(ModelEntry {
                service,
                name: model.name.clone(),
                provider: model.provider,
                model_id: model.id.clone(),
                required_env: model.wire.api_key_envs.clone(),
                optional_env,
                base_url: model.wire.base_url.clone(),
                default_decision,
                efforts,
                default_effort,
            });
        }
    }
    entries
}

fn print_models(filter: Option<Service>, json: bool) -> Result<()> {
    let entries = builtin_model_entries(filter);
    if json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
        return Ok(());
    }

    let rows: Vec<[String; 7]> = entries
        .iter()
        .map(|entry| {
            let decision = entry
                .default_decision
                .map_or("-", VllmDecision::as_str)
                .to_owned();
            let efforts = if entry.efforts.is_empty() {
                "-".to_owned()
            } else {
                entry
                    .efforts
                    .iter()
                    .map(|&effort| {
                        if entry.default_effort == Some(effort) {
                            format!("{} (default)", effort.as_str())
                        } else {
                            effort.as_str().to_owned()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            [
                entry.service.as_str().to_owned(),
                entry.name.clone(),
                entry.provider.as_str().to_owned(),
                entry.model_id.clone(),
                decision,
                efforts,
                entry.required_env.join(" or "),
            ]
        })
        .collect();

    let headers = [
        "SERVICE",
        "NAME",
        "PROVIDER",
        "MODEL ID",
        "DECISION",
        "EFFORTS",
        "REQUIRED ENV",
    ];
    let mut widths: [usize; 7] = headers.map(str::len);
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.len());
        }
    }

    let format_row = |cells: [&str; 7]| {
        format!(
            "{:<w0$}  {:<w1$}  {:<w2$}  {:<w3$}  {:<w4$}  {:<w5$}  {}",
            cells[0],
            cells[1],
            cells[2],
            cells[3],
            cells[4],
            cells[5],
            cells[6],
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
            w3 = widths[3],
            w4 = widths[4],
            w5 = widths[5],
        )
    };
    println!("{}", format_row(headers));
    for row in &rows {
        println!(
            "{}",
            format_row([
                row[0].as_str(),
                row[1].as_str(),
                row[2].as_str(),
                row[3].as_str(),
                row[4].as_str(),
                row[5].as_str(),
                row[6].as_str(),
            ])
        );
    }
    Ok(())
}

fn validate_configs(paths: &[PathBuf]) -> Result<()> {
    let mut failures = 0usize;
    for path in paths {
        match ModelConfig::from_path(path) {
            Ok(config) => {
                let decision = config.provider.asks_twice().then(|| {
                    config
                        .tuning
                        .as_ref()
                        .map_or(VllmDecision::Average, |tuned| tuned.default_decision)
                });
                let has_efforts = config
                    .tuning
                    .as_ref()
                    .is_some_and(|tuned| tuned.efforts.is_some());
                let mut details = format!(
                    "service={}, provider={}, model_id={}",
                    config.service.as_str(),
                    config.provider.as_str(),
                    config.model_id,
                );
                if let Some(decision) = decision {
                    details.push_str(&format!(", decision={}", decision.as_str()));
                }
                if has_efforts {
                    details.push_str(", efforts=low,medium,high,max");
                }
                println!("OK {}: {} ({details})", path.display(), config.name);
            }
            Err(error) => {
                eprintln!("ERR {}: {error:#}", path.display());
                failures += 1;
            }
        }
    }
    ensure!(
        failures == 0,
        "{failures} of {} model config(s) failed validation",
        paths.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Serve {
            captcha,
            mode,
            google_test_key,
            hcaptcha_test_key,
            env_files,
            port,
            hostname,
            log,
            provider,
        } => {
            let service = captcha.unwrap_or(if hcaptcha_test_key {
                Service::Hcaptcha
            } else {
                Service::Recaptcha
            });
            match service {
                Service::Recaptcha => {
                    ensure!(
                        !hcaptcha_test_key,
                        "--hcaptcha-test-key requires --captcha hcaptcha"
                    );
                }
                Service::Hcaptcha => {
                    ensure!(
                        mode.is_none() && !google_test_key,
                        "--mode and --google-test-key apply to reCAPTCHA"
                    );
                }
            }

            envfile::load_files(&env_files)?;

            let captcha = match service {
                Service::Recaptcha => {
                    let mode = match (mode, google_test_key) {
                        (Some(mode), _) => mode,
                        (None, true) => recaptcha::Mode::V2,
                        (None, false) => bail!("--mode is required for reCAPTCHA"),
                    };
                    let hostname = hostname.unwrap_or_else(|| "localhost".into());
                    Captcha::Recaptcha(recaptcha::Google::new(mode, google_test_key, hostname)?)
                }
                Service::Hcaptcha => {
                    let hostname = hostname.unwrap_or_else(|| hcaptcha::DEFAULT_HOSTNAME.into());
                    ensure!(
                        !matches!(hostname.as_str(), "localhost" | "127.0.0.1"),
                        "hCaptcha refuses pages on localhost. Leave --hostname out, or name a hosts-file entry for 127.0.0.1."
                    );
                    Captcha::Hcaptcha(hcaptcha::Hcaptcha::new(hcaptcha_test_key, hostname)?)
                }
            };
            let backend = providers::Backend::new(*provider, captcha.profile())?;
            server::serve(port, captcha, backend, log.as_deref())
        }
        Command::Models { service, json } => print_models(service, json),
        Command::ValidateConfig { paths } => validate_configs(&paths),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_service_and_captcha_aliases_and_repeatable_env_files() {
        let parsed = Cli::try_parse_from([
            "hypercaptcha",
            "serve",
            "--service",
            "hcaptcha",
            "--hcaptcha-test-key",
            "--env-file",
            "a.env",
            "--env-file",
            "b.env",
        ])
        .unwrap();
        match parsed.command {
            Command::Serve {
                captcha,
                hcaptcha_test_key,
                env_files,
                ..
            } => {
                assert_eq!(captcha, Some(Service::Hcaptcha));
                assert!(hcaptcha_test_key);
                assert_eq!(
                    env_files,
                    vec![PathBuf::from("a.env"), PathBuf::from("b.env")]
                );
            }
            _ => panic!("expected serve command"),
        }

        for flag in ["--service", "--captcha"] {
            let parsed =
                Cli::try_parse_from(["hypercaptcha", "models", flag, "recaptcha", "--json"])
                    .unwrap();
            match parsed.command {
                Command::Models { service, json } => {
                    assert_eq!(service, Some(Service::Recaptcha));
                    assert!(json);
                }
                _ => panic!("expected models command"),
            }
        }

        let parsed = Cli::try_parse_from([
            "hypercaptcha",
            "validate-config",
            "models/recaptcha/vllm/d3-ultra.toml",
            "models/hcaptcha/vllm/d3-ultra.toml",
        ])
        .unwrap();
        match parsed.command {
            Command::ValidateConfig { paths } => assert_eq!(paths.len(), 2),
            _ => panic!("expected validate-config command"),
        }
    }

    #[test]
    fn builtin_model_entries_cover_all_shipped_models() {
        let all = builtin_model_entries(None);
        assert_eq!(all.len(), 5);
        assert_eq!(builtin_model_entries(Some(Service::Recaptcha)).len(), 4);
        assert_eq!(builtin_model_entries(Some(Service::Hcaptcha)).len(), 1);
        let vllm = all
            .iter()
            .find(|entry| entry.service == Service::Recaptcha && entry.name == "vLLM/d3-ultra")
            .unwrap();
        assert_eq!(vllm.default_decision, Some(VllmDecision::Logistic));
        assert_eq!(vllm.default_effort, Some(Effort::High));
        assert_eq!(vllm.efforts, Effort::ALL);
    }
}
