use std::{
    fs::{File, OpenOptions},
    io::{self, Cursor, Read, Write},
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ImageFormat, imageops};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::{
    captcha::{Captcha, Service, hcaptcha::grid, recaptcha},
    providers::{Backend, Input, Output, Provider},
};

const MAX_BODY: u64 = 16 * 1024 * 1024;
/// Google's replacement tiles are a few kilobytes each.
const MAX_TILE_BASE64: usize = 1024 * 1024;

struct Lab {
    captcha: Captcha,
    backend: Option<Backend>,
    log: Box<dyn Write>,
    port: u16,
}

#[derive(Deserialize)]
struct VerifyRequest {
    trial_id: String,
    source: Source,
    token: String,
}

#[derive(Deserialize)]
struct Trial {
    trial_id: String,
    source: Source,
}

#[derive(Deserialize)]
struct ModelRequest {
    trial_id: String,
    source: Source,
    prompt: Option<String>,
    layout: Option<u8>,
    image: Option<String>,
    /// Pictures to draw into `image` before the model sees it.
    tiles: Option<Vec<NewTile>>,
    /// hCaptcha: where the nine pictures are in `image`, when the page could tell. Each is
    /// left, top, width and height as fractions of the image.
    pictures: Option<Vec<[f64; 4]>>,
    trace: Option<Value>,
}

/// A screenshot of an hCaptcha challenge whose pictures are to be found.
#[derive(Deserialize)]
struct PicturesRequest {
    trial_id: String,
    source: Source,
    image: String,
}

/// A replacement tile the widget has fetched and not yet shown.
#[derive(Deserialize)]
struct NewTile {
    /// Left, top, width and height of the tile's place, as fractions of the grid.
    #[serde(rename = "box")]
    place: [f64; 4],
    /// The tile's picture as the browser received it, base64-encoded.
    image: String,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Source {
    Browser,
    Camoufox,
    Chrome,
}

#[derive(Deserialize)]
struct ObservationRequest {
    trial_id: String,
    source: Source,
    outcome: Observation,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Observation {
    NoImageChallenge,
    Captured,
    ModelSolved,
    ModelFailed,
    HumanSolved,
    Abandoned,
}

pub fn serve(
    port: u16,
    captcha: Captcha,
    backend: Option<Backend>,
    log: Option<&Path>,
) -> Result<()> {
    let log: Box<dyn Write> = match log {
        Some(path) => Box::new(open_log(path)?),
        None => Box::new(io::sink()),
    };
    let server = Server::http(("127.0.0.1", port))
        .map_err(|error| anyhow::anyhow!("could not start the localhost server: {error}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .context("expected TCP server")?
        .port();
    let mut lab = Lab {
        captcha,
        backend,
        log,
        port,
    };
    println!("Open http://{}:{port}", lab.captcha.page_host());
    if let Some(backend) = &lab.backend {
        backend.keep_warm();
    }
    for mut request in server.incoming_requests() {
        // Any request to the lab means a challenge may follow, so the model's connections
        // are opened now instead of when the first grid arrives.
        if let Some(backend) = &lab.backend {
            backend.keep_warm();
        }
        let (status, body, content_type) = lab.route(&mut request);
        let response = Response::from_string(body)
            .with_status_code(StatusCode(status))
            .with_header(Header::from_bytes("Content-Type", content_type).expect("static header"))
            .with_header(Header::from_bytes("Cache-Control", "no-store").expect("static header"))
            .with_header(
                Header::from_bytes("X-Content-Type-Options", "nosniff").expect("static header"),
            );
        if let Err(error) = request.respond(response) {
            eprintln!("Failed to send localhost response: {error}");
        }
    }
    Ok(())
}

impl Lab {
    fn route(&mut self, request: &mut Request) -> (u16, String, &'static str) {
        // Only the printed name is answered, so another site cannot reach the lab by pointing
        // a name of its own at this machine.
        let host = format!("{}:{}", self.captcha.page_host(), self.port);
        if header(request, "Host") != Some(host.as_str()) {
            let error = match self.captcha {
                Captcha::Recaptcha(_) => "Open the localhost URL printed by the server.",
                Captcha::Hcaptcha(_) => "Open the URL printed by the server.",
            };
            return json_response(403, json!({"error": error}));
        }
        if request.method() == &Method::Post {
            let origin = format!("http://{host}");
            if header(request, "Origin") != Some(origin.as_str()) {
                return json_response(
                    403,
                    json!({"error": "Requests must come from the lab page."}),
                );
            }
        }
        let result = match (request.method(), request.url()) {
            (&Method::Get, "/") => {
                return (
                    200,
                    match self.captcha {
                        Captcha::Recaptcha(_) => include_str!("../web/recaptcha/index.html"),
                        Captcha::Hcaptcha(_) => include_str!("../web/hcaptcha/index.html"),
                    }
                    .into(),
                    "text/html; charset=utf-8",
                );
            }
            (&Method::Get, "/lab.js") => {
                return (
                    200,
                    match self.captcha {
                        Captcha::Recaptcha(_) => include_str!("../web/recaptcha/lab.js"),
                        Captcha::Hcaptcha(_) => include_str!("../web/hcaptcha/lab.js"),
                    }
                    .into(),
                    "text/javascript; charset=utf-8",
                );
            }
            (&Method::Get, "/lab.css") => {
                return (
                    200,
                    include_str!("../web/lab.css").into(),
                    "text/css; charset=utf-8",
                );
            }
            (&Method::Get, "/config") => {
                let mut config = self.captcha.config();
                let backend = self.backend.as_ref();
                for (name, value) in [
                    ("provider", json!(backend.map(|backend| backend.provider))),
                    (
                        "model",
                        json!(backend.map(|backend| backend.model.as_str())),
                    ),
                    (
                        "model_name",
                        json!(backend.map(|backend| backend.name.as_str())),
                    ),
                    (
                        "jev_vision",
                        json!(backend.and_then(|backend| backend.jev_vision)),
                    ),
                    (
                        "decision_image_format",
                        json!(backend.and_then(Backend::decision_image_format)),
                    ),
                    (
                        "decision_requests_3x3",
                        json!(backend.and_then(|backend| backend.decision_request_count(3, None))),
                    ),
                    (
                        "decision_requests_4x4",
                        json!(backend.and_then(|backend| backend.decision_request_count(4, None))),
                    ),
                    (
                        "tile_threshold_3x3",
                        json!(backend.and_then(|backend| backend.tile_threshold(Some(3)))),
                    ),
                    (
                        "tile_threshold_4x4",
                        json!(backend.and_then(|backend| backend.tile_threshold(Some(4)))),
                    ),
                    (
                        "vllm_whole_grid_weight_3x3",
                        json!(backend.and_then(|backend| backend.vllm_whole_grid_weight(3))),
                    ),
                    (
                        "vllm_whole_grid_weight_4x4",
                        json!(backend.and_then(|backend| backend.vllm_whole_grid_weight(4))),
                    ),
                    (
                        "vllm_decision",
                        json!(backend.and_then(|backend| backend.vllm_decision)),
                    ),
                    ("effort", json!(backend.and_then(|backend| backend.effort))),
                    ("plan_4x4", json!(backend.and_then(Backend::plan_4x4))),
                    (
                        "supports_vision",
                        json!(backend.is_some_and(|backend| backend.provider.supports_vision())),
                    ),
                    (
                        "requires_vision_check",
                        json!(
                            backend.is_some_and(|backend| backend.provider.requires_vision_check())
                        ),
                    ),
                    (
                        "is_decision",
                        json!(backend.is_some_and(|backend| backend.provider.is_decision())),
                    ),
                    ("api_keys", json!(backend.map(Backend::key_count))),
                ] {
                    config[name] = value;
                }
                return json_response(200, config);
            }
            (&Method::Post, "/verify") => self.verify(request),
            (&Method::Post, "/model") => self.predict(request),
            (&Method::Post, "/vision-check") => self.check_vision(request),
            (&Method::Post, "/observe") => self.observe(request),
            (&Method::Post, "/pictures") => self.find_pictures(request),
            _ => return json_response(404, json!({"error": "Route not found."})),
        };
        match result {
            Ok(value) => json_response(200, value),
            Err(error) => json_response(400, json!({"error": error.to_string()})),
        }
    }

    fn verify(&mut self, request: &mut Request) -> Result<Value> {
        let submission: VerifyRequest = read_json(request)?;
        validate_id(&submission.trial_id)?;
        ensure!(
            !submission.token.is_empty() && submission.token.len() <= 16384,
            "Invalid {} response token.",
            self.captcha.name()
        );
        let start = Instant::now();
        let outcome = self
            .captcha
            .verify(&submission.token, header(request, "User-Agent"));
        let result = match outcome {
            Ok(verdict) => verdict,
            Err(error) => {
                json!({"error": format!("{} verification failed: {error}", self.captcha.name())})
            }
        };
        let event = match self.captcha {
            Captcha::Recaptcha(_) => "google",
            Captcha::Hcaptcha(_) => "hcaptcha",
        };
        self.record(
            &submission.trial_id,
            submission.source,
            event,
            Some(start),
            None,
            result,
        )
    }

    fn check_vision(&mut self, request: &mut Request) -> Result<Value> {
        let backend = self
            .backend
            .as_ref()
            .context("Start the lab with a decision provider (--provider or --model-config).")?;
        let trial: Trial = read_json(request)?;
        validate_id(&trial.trial_id)?;
        let start = Instant::now();
        let tile_threshold = backend.tile_threshold(None);
        let result = decision_vision_result(backend);
        self.record(
            &trial.trial_id,
            trial.source,
            "vision_check",
            Some(start),
            tile_threshold,
            result,
        )
    }

    fn predict(&mut self, request: &mut Request) -> Result<Value> {
        let backend = self
            .backend
            .as_ref()
            .context("Start the lab with --provider to call a model.")?;
        let submission: ModelRequest = read_json(request)?;
        validate_id(&submission.trial_id)?;
        let mut grid_pixels = None;
        // The grid the model was shown, when it is not the screenshot as sent.
        let mut drawn_grid = None;
        // hCaptcha: where each picture is in the screenshot, for the caller to click.
        let mut pictures = None;
        let input = if self.captcha.score_based() {
            let trace = submission
                .trace
                .filter(Value::is_object)
                .context("Missing browser interaction telemetry.")?;
            ensure!(
                trace.to_string().len() <= 4096,
                "Browser interaction telemetry is too large."
            );
            Input::Trace(trace)
        } else {
            ensure!(
                backend.provider != Provider::OpenaiCompletions,
                "Legacy Completions accepts text. Use an image-capable provider for grids."
            );
            let layouts = self.captcha.profile().layouts;
            let layout = submission
                .layout
                .or_else(|| (layouts.len() == 1).then_some(layouts[0]))
                .filter(|layout| layouts.contains(layout))
                .with_context(|| match layouts {
                    [3] => "hCaptcha grids are 3 × 3.",
                    _ => "Choose a 3 × 3 or 4 × 4 layout.",
                })?;
            let prompt = submission
                .prompt
                .filter(|prompt| !prompt.trim().is_empty() && prompt.len() <= 300)
                .context("Enter the challenge instructions (300 bytes maximum).")?;
            let encoded = submission.image.with_context(|| {
                format!(
                    "Add a cropped PNG of the {} image grid.",
                    self.captcha.name()
                )
            })?;
            let png = STANDARD
                .decode(encoded)
                .context("Could not decode the PNG.")?;
            validate_png(&png)?;
            let png = match (&self.captcha, submission.tiles) {
                (Captcha::Hcaptcha(_), tiles) => {
                    ensure!(tiles.is_none(), "hCaptcha grids do not replace tiles.");
                    // The pictures are set side by side as the grid the model is shown.
                    let (png, places) = hcaptcha_grid(&png, submission.pictures.as_deref())?;
                    pictures = Some(places);
                    drawn_grid = Some(STANDARD.encode(&png));
                    png
                }
                (Captcha::Recaptcha(_), Some(tiles)) => {
                    let png = grid_with_new_tiles(&png, &tiles)?;
                    drawn_grid = Some(STANDARD.encode(&png));
                    png
                }
                (Captcha::Recaptcha(_), None) => png,
            };
            let (png, pixels) = backend.prepare_grid(layout, png)?;
            grid_pixels = Some(pixels);
            Input::Grid {
                layout,
                prompt,
                png,
            }
        };
        // The drawn grid goes back to the caller and stays out of the log, which would grow by
        // the size of a screenshot per round.
        let with_drawn_grid = |mut reply: Value| {
            if let Some(png) = &drawn_grid {
                reply["grid_png"] = json!(png);
            }
            if let Some(pictures) = &pictures {
                reply["pictures"] = json!(pictures);
            }
            reply
        };
        let start = Instant::now();
        let decision_grid = backend.provider.is_decision() && matches!(input, Input::Grid { .. });
        let tile_threshold = backend.tile_threshold(input.layout());
        if decision_grid {
            let check = decision_vision_result(backend);
            if check["passed"] != true {
                return self
                    .record(
                    &submission.trial_id,
                    submission.source,
                    "model",
                    Some(start),
                    tile_threshold,
                    json!({
                        "status": "vision_unverified",
                        "detail": "The decision model did not pass the image controls for this endpoint.",
                        "vision_check": check,
                    }),
                )
                    .map(with_drawn_grid);
            }
        }
        let result = match backend.predict(&input) {
            Ok(reply) => {
                let mut result = match reply.output {
                    Output::Prediction(prediction) => {
                        json!({"status": "answered", "prediction": prediction})
                    }
                    Output::Invalid(detail) => {
                        json!({"status": "invalid_output", "detail": detail})
                    }
                    Output::Refusal => json!({"status": "refused"}),
                };
                result["resolved_model"] = json!(reply.resolved_model);
                result["usage"] = json!(reply.usage);
                result["tile_probabilities"] = json!(reply.tile_probabilities);
                if let Some(components) = reply.decision_components {
                    let key = if backend.provider == Provider::Jev {
                        "jev_components"
                    } else {
                        "vllm_components"
                    };
                    result[key] = components;
                }
                if let Some(meta) = reply.provider_meta {
                    result["provider_meta"] = meta;
                }
                if let Some(pixels) = grid_pixels {
                    result["grid_pixels"] = json!(pixels);
                }
                if decision_grid {
                    result["vision_verified"] = json!(true);
                    result["decision_image_format"] = json!(backend.decision_image_format());
                    result["decision_requests"] = json!(
                        input
                            .layout()
                            .and_then(|side| backend.decision_request_count(side, Some(&input)))
                    );
                    if let Some(weight) = input
                        .layout()
                        .and_then(|side| backend.vllm_whole_grid_weight(side))
                    {
                        result["whole_grid_weight"] = json!(weight);
                    }
                    if let Some(rule) = backend.vllm_decision {
                        result["vllm_decision"] = json!(rule);
                    }
                    if let Some(effort) = backend.effort.filter(|_| input.layout() == Some(4)) {
                        result["effort"] = json!(effort);
                    }
                }
                if backend.provider == Provider::Jev && decision_grid {
                    result["jev_vision"] = json!(backend.jev_image_strategy(&input));
                }
                result
            }
            Err(error) => json!({"status": "api_error", "error": format!("{error:#}")}),
        };
        self.record(
            &submission.trial_id,
            submission.source,
            "model",
            Some(start),
            tile_threshold,
            result,
        )
        .map(with_drawn_grid)
    }

    /// Finds the nine pictures of an hCaptcha grid without asking a model, so a runner can
    /// tell a grid from hCaptcha's other tasks and save what it was shown.
    fn find_pictures(&mut self, request: &mut Request) -> Result<Value> {
        ensure!(
            matches!(self.captcha, Captcha::Hcaptcha(_)),
            "Pictures are found for hCaptcha grids."
        );
        let submission: PicturesRequest = read_json(request)?;
        validate_id(&submission.trial_id)?;
        let png = STANDARD
            .decode(submission.image)
            .context("Could not decode the PNG.")?;
        validate_png(&png)?;
        let start = Instant::now();
        let found = hcaptcha_grid(&png, None);
        let result = match &found {
            Ok((_, places)) => json!({"status": "grid", "pictures": places}),
            Err(error) => json!({"status": "no_grid", "detail": error.to_string()}),
        };
        let mut reply = self.record(
            &submission.trial_id,
            submission.source,
            "pictures",
            Some(start),
            None,
            result,
        )?;
        if let Ok((grid, _)) = found {
            reply["grid_png"] = json!(STANDARD.encode(grid));
        }
        Ok(reply)
    }

    fn observe(&mut self, request: &mut Request) -> Result<Value> {
        ensure!(
            !self.captcha.test() && !self.captcha.score_based(),
            "Observed outcomes apply to live image challenge trials."
        );
        let submission: ObservationRequest = read_json(request)?;
        validate_id(&submission.trial_id)?;
        self.record(
            &submission.trial_id,
            submission.source,
            "observation",
            None,
            None,
            json!({"outcome": submission.outcome}),
        )
    }

    fn record(
        &mut self,
        trial_id: &str,
        source: Source,
        event: &str,
        start: Option<Instant>,
        tile_threshold: Option<f64>,
        result: Value,
    ) -> Result<Value> {
        #[derive(Serialize)]
        struct Record<'a> {
            trial_id: &'a str,
            source: Source,
            event: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            captcha: Option<Service>,
            #[serde(skip_serializing_if = "Option::is_none")]
            mode: Option<recaptcha::Mode>,
            #[serde(skip_serializing_if = "Option::is_none")]
            google_test_key: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            hcaptcha_test_key: Option<bool>,
            timestamp_ms: u128,
            #[serde(skip_serializing_if = "Option::is_none")]
            latency_ms: Option<u128>,
            #[serde(skip_serializing_if = "Option::is_none")]
            requested_model: Option<&'a str>,
            /// The model with its provider, as in "vLLM/d3-ultra".
            model_name: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            provider: Option<Provider>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tile_threshold: Option<f64>,
            result: Value,
        }

        let model_event = self
            .backend
            .as_ref()
            .filter(|_| matches!(event, "model" | "vision_check"));
        let entry = Record {
            trial_id,
            source,
            event,
            // reCAPTCHA's lines keep the fields they have always had.
            captcha: matches!(self.captcha, Captcha::Hcaptcha(_)).then_some(Service::Hcaptcha),
            mode: match &self.captcha {
                Captcha::Recaptcha(google) => Some(google.mode),
                Captcha::Hcaptcha(_) => None,
            },
            google_test_key: match &self.captcha {
                Captcha::Recaptcha(google) => Some(google.test),
                Captcha::Hcaptcha(_) => None,
            },
            hcaptcha_test_key: match &self.captcha {
                Captcha::Recaptcha(_) => None,
                Captcha::Hcaptcha(hcaptcha) => Some(hcaptcha.test),
            },
            timestamp_ms: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            latency_ms: start.map(|start| start.elapsed().as_millis()),
            requested_model: model_event.map(|backend| backend.model.as_str()),
            model_name: model_event.map(|backend| backend.name.as_str()),
            provider: model_event.map(|backend| backend.provider),
            tile_threshold,
            result,
        };
        let mut line = serde_json::to_vec(&entry)?;
        line.push(b'\n');
        self.log
            .write_all(&line)
            .context("Could not append the trial log.")?;
        Ok(serde_json::to_value(entry)?)
    }
}

fn decision_vision_result(backend: &Backend) -> Value {
    match backend.check_decision_vision() {
        Ok(check) => json!(check),
        Err(error) => json!({"passed": false, "error": format!("{error:#}")}),
    }
}

fn open_log(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("could not open log {}", path.display()))
}

fn header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request
        .headers()
        .iter()
        .find(|header| header.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|header| header.value.as_str())
}

fn read_json<T: DeserializeOwned>(request: &mut Request) -> Result<T> {
    let mut data = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY + 1)
        .read_to_end(&mut data)?;
    ensure!(data.len() as u64 <= MAX_BODY, "Request body is too large.");
    serde_json::from_slice(&data).context("Invalid JSON request.")
}

fn validate_id(id: &str) -> Result<()> {
    ensure!(
        !id.is_empty()
            && id.len() <= 64
            && id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "Invalid trial ID."
    );
    Ok(())
}

fn validate_png(png: &[u8]) -> Result<()> {
    if !png.starts_with(b"\x89PNG\r\n\x1a\n") || png.len() < 24 || &png[12..16] != b"IHDR" {
        bail!("The screenshot must be a PNG.");
    }
    let width = u32::from_be_bytes(png[16..20].try_into()?);
    let height = u32::from_be_bytes(png[20..24].try_into()?);
    ensure!(
        [width, height]
            .iter()
            .all(|side| (280..=1568).contains(side)),
        "Crop the grid to a PNG between 280 and 1568 pixels on each side."
    );
    ensure!(
        tiny_skia::Pixmap::decode_png(png).is_ok(),
        "The screenshot is not a readable PNG."
    );
    Ok(())
}

/// The grid screenshot with each tile's picture drawn over its place, as a PNG. A tile is
/// stretched to its place the way a browser does it, by linear interpolation.
/// The grid of an hCaptcha screenshot's nine pictures, and where each picture is in the
/// screenshot as left, top, width and height fractions. `given` are the places the page
/// reported; without them the pictures are found in the screenshot.
fn hcaptcha_grid(png: &[u8], given: Option<&[[f64; 4]]>) -> Result<(Vec<u8>, Vec<[f64; 4]>)> {
    let screenshot = image::load_from_memory_with_format(png, ImageFormat::Png)
        .context("decode the challenge screenshot")?
        .into_rgb8();
    let (width, height) = screenshot.dimensions();
    let places = match given {
        Some(given) => {
            let places = given
                .iter()
                .map(|&place| grid::Place::from_fractions(place, width, height))
                .collect::<Result<Vec<_>>>()?;
            places
                .try_into()
                .map_err(|_| anyhow::anyhow!("An hCaptcha grid has nine pictures."))?
        }
        None => grid::find_pictures(&screenshot)?,
    };
    let fractions = places
        .iter()
        .map(|place| place.fractions(width, height))
        .collect();
    Ok((grid::compose(&screenshot, &places)?, fractions))
}

fn grid_with_new_tiles(png: &[u8], tiles: &[NewTile]) -> Result<Vec<u8>> {
    ensure!(
        (1..=16).contains(&tiles.len()),
        "Send between 1 and 16 tiles."
    );
    let mut grid = image::load_from_memory_with_format(png, ImageFormat::Png)
        .context("Could not decode the PNG.")?
        .into_rgb8();
    let (width, height) = (f64::from(grid.width()), f64::from(grid.height()));
    for tile in tiles {
        let [left, top, wide, high] = tile.place;
        // A box may reach a pixel past the edge of the screenshot; the picture is clipped there.
        ensure!(
            tile.place.iter().all(|part| (0.0..=1.0).contains(part)),
            "Give a tile's box as fractions of the grid."
        );
        let size = [
            (wide * width).round() as u32,
            (high * height).round() as u32,
        ];
        ensure!(size.iter().all(|&side| side > 0), "A tile's box is empty.");
        ensure!(
            tile.image.len() <= MAX_TILE_BASE64,
            "A tile's picture is too large."
        );
        let bytes = STANDARD
            .decode(&tile.image)
            .context("Could not decode a tile's picture.")?;
        let picture = image::load_from_memory(&bytes)
            .context("A tile's picture is not a readable JPEG or PNG.")?
            .into_rgb8();
        let picture = imageops::resize(&picture, size[0], size[1], imageops::FilterType::Triangle);
        imageops::replace(
            &mut grid,
            &picture,
            (left * width).round() as i64,
            (top * height).round() as i64,
        );
    }
    let mut out = Cursor::new(Vec::new());
    grid.write_to(&mut out, ImageFormat::Png)
        .context("Could not encode the grid.")?;
    Ok(out.into_inner())
}

fn json_response(status: u16, result: Value) -> (u16, String, &'static str) {
    (
        status,
        result.to_string(),
        "application/json; charset=utf-8",
    )
}
