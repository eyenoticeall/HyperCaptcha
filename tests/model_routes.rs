use std::{
    io::{BufRead, BufReader, Read},
    process::{Child, Command, Stdio},
    thread,
    time::Duration,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::blocking::Client;
use serde_json::{Value, json};
use tiny_http::{Header, Response, Server};

/// The JSON body of a provider request. vLLM requests arrive gzip-compressed.
fn json_body(request: &mut tiny_http::Request) -> Value {
    let gzip = request
        .headers()
        .iter()
        .any(|header| header.field.equiv("Content-Encoding") && header.value.as_str() == "gzip");
    let mut bytes = Vec::new();
    request.as_reader().read_to_end(&mut bytes).unwrap();
    if gzip {
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(bytes.as_slice())
            .read_to_end(&mut json)
            .unwrap();
        bytes = json;
    }
    serde_json::from_slice(&bytes).unwrap()
}

struct LabProcess(Child);

impl Drop for LabProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

const EFFORT: &str = "--effort";

fn start_lab(mode: &str, provider: &str, api_base: &str, extra: &[&str]) -> (LabProcess, String) {
    // The mocks expect an exact sequence of requests, so vLLM's connections are not warmed, and
    // a 4 × 4 round is asked at the lowest effort level, which sends no box requests, unless
    // the test names a level or averages the scores, which has no levels.
    let mut extra = extra.to_vec();
    let averaged = extra.iter().any(|flag| {
        flag.starts_with("--vllm-decision") || flag.starts_with("--vllm-whole-grid-weight")
    });
    if provider == "vllm" && !averaged && !extra.contains(&EFFORT) {
        extra.extend([EFFORT, "low"]);
    }
    spawn_lab(mode, provider, api_base, &extra, provider == "vllm")
}

fn spawn_lab(
    mode: &str,
    provider: &str,
    api_base: &str,
    extra: &[&str],
    cold: bool,
) -> (LabProcess, String) {
    let mut process = LabProcess(
        Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
            .args([
                "serve",
                "--mode",
                mode,
                "--port",
                "0",
                "--provider",
                provider,
                "--base-url",
                api_base,
                "--timeout",
                "3",
            ])
            .args(
                (provider != "vllm")
                    .then_some(["--model", "test-model"])
                    .into_iter()
                    .flatten(),
            )
            .args(cold.then_some("--vllm-cold-connections"))
            .args(extra)
            .env("RECAPTCHA_SITE_KEY", "test-site")
            .env("RECAPTCHA_SECRET_KEY", "test-secret")
            .env("GOOGLE_CLOUD_PROJECT", "test-project")
            .env("GOOGLE_CLOUD_API_KEY", "test-google-key")
            .env("OPENAI_API_KEY", "test-model-key")
            .env("TYPESAFE_API_KEY", "test-model-key")
            .env("VLLM_SR_API_KEY", "test-vllm-key")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut stdout = BufReader::new(process.0.stdout.take().unwrap());
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    let url = line.trim().strip_prefix("Open ").unwrap().to_string();
    (process, url)
}

#[test]
fn model_routes_use_the_correct_wire_formats() {
    for (mode, provider, endpoint, layout, prompt, split) in [
        (
            "v2",
            "openai-chat",
            "/v1/chat/completions",
            3,
            "Select all images with bicycles.",
            false,
        ),
        ("v3", "openai-completions", "/v1/completions", 3, "", false),
        (
            "v2-enterprise",
            "jev",
            "/v1/systemone",
            3,
            "Select all images with bicycles.",
            false,
        ),
        ("v3-enterprise", "jev", "/v1/systemone", 3, "", false),
        (
            "policy-challenge",
            "jev",
            "/v1/systemone",
            3,
            "Select all images with bicycles.",
            false,
        ),
        (
            "v2-enterprise",
            "jev",
            "/v1/systemone",
            4,
            "Select all squares with\nbicycles\nIf there are none, click skip",
            true,
        ),
        (
            "v2-enterprise",
            "jev",
            "/v1/systemone",
            4,
            "Select all images with bicycles.",
            false,
        ),
    ] {
        let mock = Server::http("127.0.0.1:0").unwrap();
        let api_base = format!("http://{}/v1", mock.server_addr());
        let jev_grid = provider == "jev" && mode != "v3-enterprise";
        let server = thread::spawn(move || {
            if jev_grid {
                for (color, probability) in [([255, 0, 0], 0.95), ([0, 0, 255], 0.05)] {
                    let mut request = mock.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
                    assert_eq!(request.url(), endpoint);
                    assert!(request.headers().iter().any(|header| {
                        header.field.equiv("TypeSafe-Danger-Zone-Images")
                            && header.value.as_str() == "true"
                    }));
                    let body: Value = serde_json::from_reader(request.as_reader()).unwrap();
                    let reference = body["state"]["grid"].as_str().unwrap();
                    let encoded = body["assets"][reference]["data_base64"].as_str().unwrap();
                    let png = STANDARD.decode(encoded).unwrap();
                    let image = tiny_skia::Pixmap::decode_png(&png).unwrap();
                    assert_eq!(image.data()[..3], color);
                    assert_eq!(body["questions"]["red"]["type"], "noul");
                    request
                        .respond(Response::from_string(
                            json!({"answers":{"red":{"type":"noul","noul":probability}}})
                                .to_string(),
                        ))
                        .unwrap();
                }
            }
            let mut request = mock.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
            assert_eq!(request.url(), endpoint);
            assert!(
                request
                    .headers()
                    .iter()
                    .any(|header| header.field.equiv("Authorization")
                        && header.value.as_str() == "Bearer test-model-key")
            );
            let body: Value = serde_json::from_reader(request.as_reader()).unwrap();
            assert_eq!(body["model"], "test-model");
            let reply = match (mode, provider) {
                ("v2", "openai-chat") => {
                    let parts = &body["messages"][0]["content"];
                    assert!(parts[0]["text"].as_str().unwrap().contains("bicycles"));
                    assert!(
                        parts[1]["image_url"]["url"]
                            .as_str()
                            .unwrap()
                            .starts_with("data:image/png;base64,")
                    );
                    json!({"choices":[{"message":{"content":"{\"tiles\":[9,1]}"}}]})
                }
                ("v2-enterprise" | "policy-challenge", "jev") => {
                    assert!(
                        request
                            .headers()
                            .iter()
                            .any(|header| header.field.equiv("TypeSafe-Danger-Zone-Images")
                                && header.value.as_str() == "true")
                    );
                    let reference = body["state"]["grid"].as_str().unwrap();
                    assert_eq!(body["assets"][reference]["format"], "png");
                    assert_eq!(
                        body["state"]["grid_layout"],
                        json!({"rows": layout, "columns": layout})
                    );
                    let questions = body["questions"].as_object().unwrap();
                    assert_eq!(questions.len(), layout * layout);
                    if split {
                        assert_eq!(body["state"]["target"], "bicycles");
                        assert_eq!(body["assets"].as_object().unwrap().len(), 17);
                        for (index, question) in questions.values().enumerate() {
                            let reference = question["instructions"][1].as_str().unwrap();
                            let encoded =
                                body["assets"][reference]["data_base64"].as_str().unwrap();
                            let crop =
                                tiny_skia::Pixmap::decode_png(&STANDARD.decode(encoded).unwrap())
                                    .unwrap();
                            assert_eq!((crop.width(), crop.height()), (320, 320));
                            let center = 4 * (160 * 320 + 160);
                            assert_eq!(crop.data()[center], (index + 1) as u8 * 10);
                        }
                    } else {
                        assert_eq!(body["state"]["challenge_instructions"], prompt);
                        assert_eq!(body["assets"].as_object().unwrap().len(), 1);
                    }
                    let answers: serde_json::Map<_, _> = questions
                        .iter()
                        .map(|(id, question)| {
                            assert_eq!(question["type"], "noul");
                            let probability = if ["tile_01", "tile_09"].contains(&id.as_str()) {
                                0.9
                            } else {
                                0.1
                            };
                            (id.clone(), json!({"type":"noul", "noul":probability}))
                        })
                        .collect();
                    json!({"answers":answers})
                }
                ("v3", "openai-completions") => {
                    assert!(body["prompt"].as_str().unwrap().contains("pointer_moves"));
                    assert!(body.get("messages").is_none());
                    json!({"choices":[{"text":"{\"score\":0.3}"}]})
                }
                ("v3-enterprise", "jev") => {
                    assert_eq!(body["state"]["pointer_moves"], 3);
                    assert!(body.get("assets").is_none());
                    json!({"answers":{"legitimate":{"type":"noul", "noul":0.3}}})
                }
                _ => unreachable!(),
            };
            request
                .respond(
                    Response::from_string(reply.to_string()).with_header(
                        Header::from_bytes("Content-Type", "application/json").unwrap(),
                    ),
                )
                .unwrap();
        });

        let extra: &[&str] = if jev_grid {
            &["--jev-vision", "scene-and-crop"]
        } else {
            &[]
        };
        let (_process, url) = start_lab(mode, provider, &api_base, extra);
        let client = Client::builder()
            .timeout(Duration::from_secs(6))
            .build()
            .unwrap();
        let config = client
            .get(format!("{url}/config"))
            .send()
            .unwrap()
            .json::<Value>()
            .unwrap();
        assert_eq!(config["site_key"], "test-site");
        assert!(!config.to_string().contains("test-secret"));
        if mode == "policy-challenge" {
            assert_eq!(config["mode"], "policy-challenge");
            assert_eq!(config["enterprise"], true);
            assert_eq!(config["score_based"], false);
        }
        let mut input = json!({"source":"camoufox", "trial_id":"test-trial"});
        let is_grid = !mode.starts_with("v3");
        if is_grid {
            let mut grid = tiny_skia::Pixmap::new(320, 320).unwrap();
            if split {
                for (index, pixel) in grid.data_mut().chunks_exact_mut(4).enumerate() {
                    let tile = ((index / 320) / 80) * 4 + ((index % 320) / 80) + 1;
                    pixel.copy_from_slice(&[(tile * 10) as u8, 0, 0, 255]);
                }
            }
            input["image"] = json!(STANDARD.encode(grid.encode_png().unwrap()));
            input["prompt"] = json!(prompt);
            input["layout"] = json!(layout);
        } else {
            input["trace"] = json!({"pointer_moves":3});
        }
        let reply: Value = client
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&input)
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        server.join().unwrap();
        assert_eq!(reply["source"], "camoufox");
        assert_eq!(reply["trial_id"], "test-trial");
        assert_eq!(reply["result"]["status"], "answered");
        if jev_grid {
            assert_eq!(reply["result"]["vision_verified"], true);
            assert_eq!(
                reply["result"]["jev_vision"],
                if split {
                    "scene-and-crop"
                } else {
                    "whole-grid"
                }
            );
        }
        assert_eq!(
            reply["result"]["prediction"]["value"],
            if is_grid { json!([1, 9]) } else { json!(0.3) }
        );
    }
}

#[test]
fn jev_rejects_grids_when_the_endpoint_ignores_images() {
    let mock = Server::http("127.0.0.1:0").unwrap();
    let api_base = format!("http://{}/v1", mock.server_addr());
    let server = thread::spawn(move || {
        for _ in 0..2 {
            let mut request = mock.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
            let body: Value = serde_json::from_reader(request.as_reader()).unwrap();
            assert!(body["questions"].get("red").is_some());
            request
                .respond(Response::from_string(
                    json!({"answers":{"red":{"noul":0.9}}}).to_string(),
                ))
                .unwrap();
        }
        assert!(
            mock.recv_timeout(Duration::from_millis(150))
                .unwrap()
                .is_none()
        );
    });
    let (_process, url) = start_lab("v2", "jev", &api_base, &[]);
    let png = tiny_skia::Pixmap::new(320, 320)
        .unwrap()
        .encode_png()
        .unwrap();
    let result: Value = Client::new()
        .post(format!("{url}/model"))
        .header("Origin", &url)
        .json(&json!({
            "source": "chrome",
            "trial_id": "unverified-image",
            "layout": 3,
            "prompt": "Select all bicycles",
            "image": STANDARD.encode(png),
        }))
        .send()
        .unwrap()
        .json()
        .unwrap();
    server.join().unwrap();
    assert_eq!(result["result"]["status"], "vision_unverified");
    assert_eq!(result["result"]["vision_check"]["passed"], false);
    assert!(result["result"].get("prediction").is_none());
}

#[test]
fn jev_averages_two_views_before_applying_the_layout_cutoff() {
    let mock = Server::http("127.0.0.1:0").unwrap();
    let api_base = format!("http://{}/v1", mock.server_addr());
    let server = thread::spawn(move || {
        for probability in [0.95, 0.05] {
            let mut request = mock.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
            let body: Value = serde_json::from_reader(request.as_reader()).unwrap();
            assert!(body["questions"].get("red").is_some());
            request
                .respond(Response::from_string(
                    json!({"answers":{"red":{"type":"noul", "noul":probability}}}).to_string(),
                ))
                .unwrap();
        }
        for (side, missing) in [(3, false), (4, false), (3, true)] {
            let mut whole_and_variant = [false; 2];
            for _ in 0..2 {
                let mut request = mock.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap();
                let fields: Value = serde_json::from_str(&body).unwrap();
                let is_variant = fields["state"].get("target").is_some();
                whole_and_variant[usize::from(is_variant)] = true;
                let asset_count = match (side, is_variant) {
                    (3, true) => {
                        assert!(fields["state"].get("grid").is_none());
                        let instructions = &fields["questions"]["tile_01"]["instructions"];
                        assert!(
                            instructions["task"]
                                .as_str()
                                .unwrap()
                                .starts_with("Does PHOTO")
                        );
                        assert_eq!(instructions["PHOTO"], "hcap-comparison-photo-01");
                        assert!(body.find("\"task\"").unwrap() < body.find("\"PHOTO\"").unwrap());
                        9
                    }
                    (4, true) => {
                        assert!(fields["state"].get("grid").is_some());
                        assert!(fields["questions"]["tile_01"]["instructions"].is_array());
                        17
                    }
                    _ => 1,
                };
                assert_eq!(fields["assets"].as_object().unwrap().len(), asset_count);
                let answers: serde_json::Map<_, _> = (1..=side * side)
                    .filter(|&tile| !(is_variant && missing && tile == 9))
                    .map(|tile| {
                        let probability = match (tile, is_variant) {
                            (1, false) => 0.3,
                            (1, true) => 0.9,
                            (2, false) => 0.1,
                            (2, true) => 0.9,
                            _ => 0.0,
                        };
                        (
                            format!("tile_{tile:02}"),
                            json!({"type":"noul", "noul":probability}),
                        )
                    })
                    .collect();
                request
                    .respond(Response::from_string(
                        json!({"model":"test-model", "answers":answers, "usage":{"input_tokens":100,"output_tokens":10}}).to_string(),
                    ))
                    .unwrap();
            }
            assert_eq!(whole_and_variant, [true, true]);
        }
    });

    let (_process, url) = start_lab(
        "v2-enterprise",
        "jev",
        &api_base,
        &["--jev-vision", "average"],
    );
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let config: Value = client
        .get(format!("{url}/config"))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(config["jev_vision"], "average");
    assert_eq!(config["tile_threshold_3x3"], 0.6);
    assert_eq!(config["tile_threshold_4x4"], 0.5);
    let image = STANDARD.encode(
        tiny_skia::Pixmap::new(320, 320)
            .unwrap()
            .encode_png()
            .unwrap(),
    );
    for (side, missing) in [(3, false), (4, false), (3, true)] {
        let noun = if side == 3 { "images" } else { "squares" };
        let reply: Value = client
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(
                &json!({"source":"chrome", "trial_id":"average", "layout":side,
                "prompt":format!("Select all {noun} with\nbicycles"), "image":image}),
            )
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        let result = &reply["result"];
        assert_eq!(reply["tile_threshold"], if side == 3 { 0.6 } else { 0.5 });
        assert_eq!(result["jev_vision"], "average");
        assert_eq!(result["usage"]["input_tokens"], 200);
        assert_eq!(
            result["jev_components"]["whole_grid"]["tile_probabilities"]["1"],
            0.3
        );
        if missing {
            assert_eq!(result["status"], "invalid_output");
            assert!(result.get("prediction").is_none());
        } else {
            assert_eq!(
                result["jev_components"]["variant"]["tile_probabilities"]["1"],
                0.9
            );
            assert_eq!(result["tile_probabilities"]["1"], 0.6);
            assert_eq!(result["tile_probabilities"]["2"], 0.5);
            assert_eq!(
                result["prediction"]["value"],
                if side == 3 { json!([1]) } else { json!([1, 2]) }
            );
        }
    }
    server.join().unwrap();
}

#[test]
fn decision_providers_preserve_scene_context_and_merge_crop_batches() {
    for (provider, model, key, extra, rule) in [
        (
            "vllm",
            "vllm-sr/d3-ultra",
            "test-vllm-key",
            &[][..],
            "logistic",
        ),
        (
            "vllm",
            "vllm-sr/d3-ultra",
            "test-vllm-key",
            &["--vllm-decision", "average"][..],
            "average",
        ),
        (
            "vllm",
            "vllm-sr/d3-ultra",
            "test-vllm-key",
            &[
                "--vllm-whole-grid-weight-3x3",
                "0.60",
                "--vllm-whole-grid-weight-4x4",
                "0.35",
            ][..],
            "weighted",
        ),
        (
            "jev",
            "test-model",
            "test-model-key",
            &[
                "--jev-vision",
                "average",
                "--jev-split-crops",
                "--decision-image-format",
                "jpeg",
            ][..],
            "average",
        ),
    ] {
        assert_jpeg_crop_batches(provider, model, key, extra, rule);
    }
}

#[test]
fn vllm_warms_a_rounds_connections_when_the_server_starts() {
    // One practice request for each connection of the busiest round: the whole grid, the
    // presence question, the two box requests and eight close-up requests. None is answered
    // until all have arrived, so each must have a connection of its own.
    // The mock's thread pool sometimes leaves one of a burst of connections waiting for a free
    // thread, which never comes while the others are held. A fresh server is then tried.
    let mut attempt = 0;
    let (mock, _process, requests) = loop {
        attempt += 1;
        let mock = Server::http("127.0.0.1:0").unwrap();
        let api_base = format!("http://{}/v1", mock.server_addr());
        let (process, _url) = spawn_lab("v2-enterprise", "vllm", &api_base, &[], false);
        let requests: Vec<_> = (0..12)
            .map_while(|_| mock.recv_timeout(Duration::from_secs(3)).unwrap())
            .collect();
        if requests.len() == 12 {
            break (mock, process, requests);
        }
        assert!(
            attempt < 4,
            "only {} practice requests arrived",
            requests.len()
        );
    };
    for mut request in requests {
        assert_eq!(request.url(), "/v1/systemone");
        let body = json_body(&mut request);
        assert_eq!(body["questions"].as_object().unwrap().len(), 1);
        let image = body["assets"]["hypercaptcha-grid-asset"]["data_base64"]
            .as_str()
            .unwrap();
        // About twice a round's largest scoring request, so the connection can send that in
        // one go.
        assert!(STANDARD.decode(image).unwrap().len() > 200_000);
        request
            .respond(Response::from_string(
                json!({
                    "model":"vllm-sr/d3-ultra", "answers":{"red":{"noul":0.5}},
                    "usage":{"input_tokens":10,"output_tokens":0},
                })
                .to_string(),
            ))
            .unwrap();
    }
    // The connections were just used, so no ping follows.
    assert!(
        mock.recv_timeout(Duration::from_millis(500))
            .unwrap()
            .is_none()
    );
}

#[test]
fn vllm_selects_nothing_when_a_skippable_grid_lacks_the_target() {
    // (presence score from the model, extra flags, whether the question is asked, expected picks)
    // A presence score of -1 makes the mock fail that request.
    for (present, extra, asked, expected) in [
        (0.3, &[][..], true, json!([])),
        (0.9, &[][..], true, json!([15, 16])),
        (-1.0, &[][..], true, json!([15, 16])),
        // Averaged scores keep the question too, with both close-up passes.
        (0.3, &["--vllm-decision", "average"][..], true, json!([])),
    ] {
        // The lowest effort level, the default here, sends one close-up pass.
        let passes = if extra.is_empty() { 1 } else { 2 };
        let mock = Server::http("127.0.0.1:0").unwrap();
        let api_base = format!("http://{}/v1", mock.server_addr());
        let server = thread::spawn(move || {
            let mut presence_questions = 0;
            // Two vision checks, the whole grid, four requests of four close-ups in every
            // pass, and the presence request, which is tried twice when it fails.
            let presence_tries = usize::from(asked) * if present < 0.0 { 2 } else { 1 };
            for index in 0..3 + 4 * passes + presence_tries {
                let mut request = mock.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
                assert!(
                    request
                        .headers()
                        .iter()
                        .any(|header| header.field.equiv("Content-Encoding"))
                );
                let body = json_body(&mut request);
                let questions = body["questions"].as_object().unwrap();
                let answers: Value = if index < 2 {
                    json!({"red":{"noul":if index == 0 { 0.99 } else { 0.01 }}})
                } else {
                    if let Some(question) = questions.get("present") {
                        presence_questions += 1;
                        // It travels alone with the grid, so the other requests are unchanged.
                        assert_eq!(questions.len(), 1);
                        assert_eq!(body["assets"].as_object().unwrap().len(), 1);
                        assert_eq!(body["state"]["target"], "bicycles");
                        assert!(
                            question["instructions"]
                                .as_str()
                                .unwrap()
                                .contains("at least one object of the type in `target`")
                        );
                    }
                    Value::Object(
                        questions
                            .keys()
                            .map(|id| {
                                let score = match id.as_str() {
                                    "present" => present,
                                    "tile_15" | "tile_16" => 0.95,
                                    _ => 0.0,
                                };
                                (id.clone(), json!({"noul":score}))
                            })
                            .collect(),
                    )
                };
                if present < 0.0 && questions.contains_key("present") {
                    request
                        .respond(Response::from_string("{}").with_status_code(500))
                        .unwrap();
                    continue;
                }
                request
                    .respond(Response::from_string(
                        json!({
                            "model":"vllm-sr/d3-ultra", "answers":answers,
                            "usage":{"input_tokens":10,"output_tokens":0},
                        })
                        .to_string(),
                    ))
                    .unwrap();
            }
            assert_eq!(presence_questions, presence_tries);
        });
        let (_process, url) = start_lab("v2-enterprise", "vllm", &api_base, extra);
        let img = STANDARD.encode(
            tiny_skia::Pixmap::new(320, 320)
                .unwrap()
                .encode_png()
                .unwrap(),
        );
        let result: Value = Client::new()
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id":"vllm-skip", "source":"camoufox", "layout":4,
                "prompt":"Select all squares with\nbicycles\nIf there are none, click skip",
                "image":img,
            }))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        server.join().unwrap();
        let result = &result["result"];
        assert_eq!(result["prediction"]["value"], expected);
        let presence = &result["vllm_components"]["presence"];
        let scoring = 1 + 4 * passes as u64;
        if present < 0.0 {
            // The failed request is logged and the square scores decide alone.
            assert!(presence["error"].is_string());
            assert_eq!(presence["skipped"], false);
            assert_eq!(result["usage"]["input_tokens"], 10 * scoring);
        } else {
            assert_eq!(presence["probability"], present);
            assert_eq!(presence["minimum"], 0.7);
            assert_eq!(presence["skipped"], present < 0.7);
            assert_eq!(result["decision_requests"], scoring + 1);
            assert_eq!(result["usage"]["input_tokens"], 10 * (scoring + 1));
            // The squares' own scores are still reported when the grid is skipped.
            assert!(result["tile_probabilities"]["16"].as_f64().unwrap() > 0.9);
        }
        let level = if passes == 1 {
            json!("low")
        } else {
            Value::Null
        };
        assert_eq!(result["effort"], level);
    }
}

#[test]
fn each_effort_level_sends_its_own_requests_and_checks_picks_against_its_boxes() {
    // Squares 15 and 16 are selected with scores low enough to need a box at the levels that
    // draw them. Each case gives the level, the squares the whole-grid boxes cover, the
    // confidence of a window box on square 15 (0 for none) and the expected picks. A whole-grid
    // entry of 0 makes the mock fail that request.
    for (effort, whole, window_confidence, expected) in [
        // Low draws no boxes, so nothing is checked.
        ("low", &[][..], 0.0, json!([15, 16])),
        // Medium and high need a whole-grid box.
        ("medium", &[16][..], 0.6, json!([16])),
        ("high", &[15, 16][..], 0.0, json!([15, 16])),
        ("high", &[16][..], 0.6, json!([16])),
        ("high", &[][..], 0.0, json!([])),
        ("high", &[0][..], 0.0, json!([15, 16])),
        // Max needs a box from each request: only square 15 has a window box.
        ("max", &[15, 16][..], 0.6, json!([15])),
        ("max", &[16][..], 0.6, json!([])),
        // A window box under 0.5 does not count.
        ("max", &[15, 16][..], 0.4, json!([])),
        // With the whole-grid request failed, square 15 is undecided and stays; square 16,
        // which the window reply did not reach, goes all the same.
        ("max", &[0][..], 0.6, json!([15])),
    ] {
        let (rule_on, windows_on) = (effort != "low", effort == "max");
        // Low and medium score one close-up pass, the others two.
        let scoring_requests = if matches!(effort, "low" | "medium") {
            5
        } else {
            9
        };
        let whole_fails = whole == [0];
        let square = |tile: u8| {
            let (row, column) = (f64::from((tile - 1) / 4), f64::from((tile - 1) % 4));
            json!({"left": column / 4.0, "top": row / 4.0, "right": (column + 1.0) / 4.0, "bottom": (row + 1.0) / 4.0})
        };
        let mock = Server::http("127.0.0.1:0").unwrap();
        let api_base = format!("http://{}/v1", mock.server_addr());
        let (answered, round_answered) = std::sync::mpsc::channel::<()>();
        let server = thread::spawn(move || {
            let (mut scoring, mut whole_requests, mut window_requests) = (0, 0, 0);
            let (mut index, mut answered) = (0, None);
            // Two vision checks, then the round's requests in whatever order they arrive. The
            // lab may answer before a box request has left, so the mock goes on until both
            // have come, and then until nothing more does.
            loop {
                let Some(mut request) = mock.recv_timeout(Duration::from_millis(50)).unwrap()
                else {
                    if answered.is_none() && round_answered.try_recv().is_ok() {
                        answered = Some(std::time::Instant::now());
                    }
                    let all_came =
                        (!rule_on || whole_requests > 0) && (!windows_on || window_requests > 0);
                    let waited = answered.map_or(Duration::ZERO, |at| at.elapsed());
                    if waited > Duration::from_millis(if all_came { 400 } else { 5000 }) {
                        break;
                    }
                    continue;
                };
                index += 1;
                let body = json_body(&mut request);
                let reply = if index <= 2 {
                    json!({"answers": {"red": {"noul": if index == 1 { 0.99 } else { 0.01 }}}})
                } else if body["states"].is_object() {
                    window_requests += 1;
                    let states = body["states"].as_object().unwrap();
                    assert_eq!(states.len(), 16);
                    assert_eq!(
                        states["s15"]["questions"]["q"]["instructions"],
                        "the bicycle in the bottom of the image, including every visible part of it"
                    );
                    assert_eq!(
                        states["s15"]["state"],
                        "Part of a street photo. Target: bicycles."
                    );
                    assert!(
                        states["s15"]["images"][0]
                            .as_str()
                            .unwrap()
                            .starts_with("data:image/jpeg;base64,/9j/")
                    );
                    let answers: serde_json::Map<String, Value> = states
                        .keys()
                        .map(|name| {
                            // Square 15's window is the bottom half without its first column,
                            // so the square is its middle third.
                            let confidence = if name == "s15" { window_confidence } else { 0.0 };
                            let rect = json!({"left": 0.34, "top": 0.5, "right": 0.66, "bottom": 1.0});
                            (name.clone(), json!({"answers": {"q": {"type": "box", "box": rect, "confidence": confidence}}}))
                        })
                        .collect();
                    json!({"states": answers})
                } else if body["images"].is_array() {
                    whole_requests += 1;
                    assert_eq!(body["state"], "A street photo. Locate the bicycles.");
                    assert_eq!(body["questions"].as_object().unwrap().len(), 9);
                    assert_eq!(
                        body["questions"]["nearest"],
                        json!({"type": "box", "instructions": "the bicycle nearest to the camera"})
                    );
                    assert_eq!(body["options"]["return_meta"], true);
                    if whole_fails {
                        request
                            .respond(Response::from_string("{}").with_status_code(500))
                            .unwrap();
                        continue;
                    }
                    let answers: serde_json::Map<String, Value> = body["questions"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .zip(whole.iter().chain(std::iter::repeat(&0)))
                        .map(|(name, &tile)| {
                            let answer = if tile == 0 {
                                json!({"type": "box", "box": square(1), "confidence": 0.0})
                            } else {
                                json!({"type": "box", "box": square(tile), "confidence": 0.8})
                            };
                            (name.clone(), answer)
                        })
                        .collect();
                    json!({"answers": answers})
                } else {
                    scoring += 1;
                    let answers: serde_json::Map<String, Value> = body["questions"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .map(|id| {
                            let score = if ["tile_15", "tile_16"].contains(&id.as_str()) {
                                0.9
                            } else {
                                0.0
                            };
                            (id.clone(), json!({"noul": score}))
                        })
                        .collect();
                    json!({"answers": answers})
                };
                let mut reply = reply;
                reply["model"] = json!("vllm-sr/d3-ultra");
                reply["usage"] = json!({"input_tokens": 10, "output_tokens": 0});
                request
                    .respond(Response::from_string(reply.to_string()))
                    .unwrap();
            }
            assert_eq!(scoring, scoring_requests);
            // A failed request is tried once more.
            assert_eq!(
                whole_requests,
                usize::from(rule_on) * if whole_fails { 2 } else { 1 }
            );
            // A level's box requests are sent in every round, whatever the scores turn out to be.
            assert_eq!(window_requests, usize::from(windows_on));
        });
        let (_process, url) = start_lab("v2-enterprise", "vllm", &api_base, &[EFFORT, effort]);
        let config: Value = Client::new()
            .get(format!("{url}/config"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(config["effort"], effort);
        let one_pass = scoring_requests == 5;
        assert_eq!(
            config["plan_4x4"]["close_ups"],
            if one_pass {
                json!([506])
            } else {
                json!([null, 506])
            }
        );
        assert_eq!(config["plan_4x4"]["boxes"].is_object(), rule_on);
        assert_eq!(config["plan_4x4"]["gap"].is_object(), effort == "low");
        assert_eq!(
            config["tile_threshold_4x4"],
            if one_pass { 0.55 } else { 0.5 }
        );
        let img = STANDARD.encode(
            tiny_skia::Pixmap::new(320, 320)
                .unwrap()
                .encode_png()
                .unwrap(),
        );
        let result: Value = Client::new()
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id":"vllm-boxes", "source":"camoufox", "layout":4,
                "prompt":"Select all squares with\nbicycles", "image":img,
            }))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        answered.send(()).unwrap();
        server.join().unwrap();
        let result = &result["result"];
        assert_eq!(
            result["prediction"]["value"], expected,
            "{effort}: whole {whole:?}, window {window_confidence}"
        );
        assert_eq!(result["effort"], effort);
        // The scores are reported as the model gave them, whatever the boxes removed.
        assert!(result["tile_probabilities"]["15"].as_f64().unwrap() > 0.5);
        // Each pass's scores are logged, so a round can be scored again under another level.
        let passes = &result["vllm_components"]["variant"]["passes"];
        assert_eq!(passes.as_array().unwrap().len(), (scoring_requests - 1) / 4);
        let box_requests = u64::from(rule_on) + u64::from(windows_on);
        assert_eq!(
            result["decision_requests"],
            scoring_requests as u64 + box_requests
        );
        // The scoring requests' token count is unchanged by the box requests.
        assert_eq!(
            result["usage"]["input_tokens"],
            10 * scoring_requests as u64
        );
        let boxes = &result["vllm_components"]["boxes"];
        if !rule_on {
            assert!(boxes.is_null());
            // The gap rule ran and found the two views in agreement.
            assert_eq!(result["vllm_components"]["gap"]["removed"], json!([]));
            continue;
        }
        assert_eq!(boxes["picks_before"], json!([15, 16]));
        assert_eq!(boxes["at_risk"], json!([15, 16]));
        let removed: Vec<u64> = [15, 16]
            .into_iter()
            .filter(|tile| !expected.as_array().unwrap().contains(&json!(tile)))
            .collect();
        assert_eq!(boxes["removed"], json!(removed));
        assert_eq!(boxes["error"].is_string(), whole_fails);
    }
}

/// One request to the mock provider below, and where its reply goes.
struct Asked {
    headers: Vec<(String, String)>,
    body: Value,
    reply: std::sync::mpsc::Sender<String>,
}

impl Asked {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(field, _)| field.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// A mock provider that gives every connection a thread of its own. tiny_http hands
/// connections to a pool that can leave a new one waiting behind those still open, which a
/// round that opens two dozen at once, as a Liquid round with window boxes does, runs into.
fn many_at_once() -> (String, std::sync::mpsc::Receiver<Asked>) {
    use std::io::Write;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let api_base = format!("http://{}/v1", listener.local_addr().unwrap());
    let (asked, requests) = std::sync::mpsc::channel();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let (Ok(mut stream), asked) = (stream, asked.clone()) else {
                break;
            };
            thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                // A connection carries one request after another until the lab closes it.
                loop {
                    let mut headers = Vec::new();
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    loop {
                        line.clear();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 {
                            return;
                        }
                        let Some((field, value)) = line.trim_end().split_once(':') else {
                            break;
                        };
                        headers.push((field.to_owned(), value.trim().to_owned()));
                    }
                    let length = headers
                        .iter()
                        .find(|(field, _)| field.eq_ignore_ascii_case("content-length"))
                        .map_or(0, |(_, value)| value.parse().unwrap());
                    let mut body = vec![0; length];
                    if reader.read_exact(&mut body).is_err() {
                        return;
                    }
                    let (reply, replied) = std::sync::mpsc::channel();
                    let request = Asked {
                        headers,
                        body: serde_json::from_slice(&body).unwrap(),
                        reply,
                    };
                    let (Ok(()), Ok(reply)) = (asked.send(request), replied.recv()) else {
                        return;
                    };
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                        reply.len()
                    );
                    if stream.write_all((head + &reply).as_bytes()).is_err() {
                        return;
                    }
                }
            });
        }
    });
    (api_base, requests)
}

/// Starts a lab that asks Liquid/d1.2-ultra at `api_base`.
fn start_liquid_lab(api_base: &str, extra: &[&str]) -> (LabProcess, String) {
    let mut process = LabProcess(
        Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
            .args(["serve", "--mode", "v2", "--port", "0", "--timeout", "20"])
            .args(["--model", "Liquid/d1.2-ultra", "--base-url", api_base])
            .args(extra)
            .env("RECAPTCHA_SITE_KEY", "test-site")
            .env("RECAPTCHA_SECRET_KEY", "test-secret")
            .env("LIQUID_API_KEY", "test-liquid-key")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut line = String::new();
    BufReader::new(process.0.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let url = line.trim().strip_prefix("Open ").unwrap().to_string();
    (process, url)
}

#[test]
fn liquid_gets_its_pictures_as_a_list_and_each_window_in_a_request_of_its_own() {
    // Liquid ignores named assets, refuses compressed bodies and takes one state a request.
    // In record mode a 4 × 4 round asks everything: the whole grid, two close-up passes of
    // four requests, the skip question, nine boxes on the whole grid, a box around every
    // instance on the whole grid and a box in each of sixteen windows.
    let (api_base, mock) = many_at_once();
    let server = thread::spawn(move || {
        let (mut checks, mut whole, mut close_ups, mut present) = (0, 0, 0, 0);
        let (mut whole_boxes, mut every, mut windows) = (0, 0, 0);
        for _ in 0..2 + 1 + 8 + 1 + 2 + 16 {
            let request = mock.recv_timeout(Duration::from_secs(30)).unwrap();
            assert_eq!(request.header("Content-Encoding"), None);
            assert_eq!(
                request.header("Authorization"),
                Some("Bearer test-liquid-key")
            );
            let body = &request.body;
            assert_eq!(body["model"], "d1.2-ultra");
            assert!(body.get("assets").is_none() && body.get("states").is_none());
            let images = body["images"].as_array().unwrap();
            assert!(images.iter().all(|image| {
                image
                    .as_str()
                    .unwrap()
                    .starts_with("data:image/jpeg;base64,/9j/")
            }));
            let questions = body["questions"].as_object().unwrap();
            let boxes = questions.values().all(|question| question["type"] == "box");
            let answers: serde_json::Map<String, Value> = if questions.contains_key("red") {
                checks += 1;
                assert_eq!(body["state"]["grid"], "image 1");
                json!({"red": {"noul": if checks == 1 { 0.99 } else { 0.01 }}})
                    .as_object()
                    .unwrap()
                    .clone()
            } else if questions
                .values()
                .all(|question| question["type"] == "boxes")
            {
                // One question for every instance, on the nine-box request's picture.
                every += 1;
                assert_eq!(images.len(), 1);
                assert_eq!(body["state"], "A street photo. Locate the bicycles.");
                assert_eq!(
                    body["questions"],
                    json!({"q": {"type": "boxes", "instructions": "every bicycle", "max": 16}})
                );
                let rect = json!({"left": 0.5, "top": 0.75, "right": 1.0, "bottom": 1.0});
                json!({"q": {
                    "type": "boxes", "count": 1, "at_least": false,
                    "boxes": [{"box": rect, "found": 0.9, "confidence": 0.8}],
                }})
                .as_object()
                .unwrap()
                .clone()
            } else if boxes {
                assert_eq!(images.len(), 1);
                assert!(
                    questions
                        .values()
                        .all(|question| question.get("max").is_none())
                );
                if questions.len() == 9 {
                    whole_boxes += 1;
                } else {
                    windows += 1;
                    assert_eq!(questions.len(), 1);
                    assert_eq!(body["state"], "Part of a street photo. Target: bicycles.");
                }
                questions
                    .keys()
                    .map(|name| {
                        let rect = json!({"left": 0.5, "top": 0.75, "right": 1.0, "bottom": 1.0});
                        (
                            name.clone(),
                            json!({"type": "box", "box": rect, "found": 0.9, "confidence": 0.8}),
                        )
                    })
                    .collect()
            } else {
                assert_eq!(body["state"]["grid"], "image 1");
                if questions.contains_key("present") {
                    present += 1;
                } else if images.len() == 1 {
                    whole += 1;
                    assert_eq!(questions.len(), 16);
                } else {
                    // The scene, then four close-ups in the order of their questions.
                    close_ups += 1;
                    assert_eq!(images.len(), 5);
                    for (place, question) in questions.values().enumerate() {
                        assert_eq!(
                            question["instructions"][1],
                            format!("PHOTO is image {}.", place + 2)
                        );
                    }
                }
                questions
                    .keys()
                    .map(|id| {
                        let score = if ["tile_15", "tile_16", "present"].contains(&id.as_str()) {
                            0.9
                        } else {
                            0.0
                        };
                        (id.clone(), json!({"type": "noul", "noul": score}))
                    })
                    .collect()
            };
            let reply = json!({
                "model": "d1.2-ultra", "answers": answers,
                "usage": {"input_tokens": 5000, "output_tokens": 0, "cost": 0.0},
            });
            request.reply.send(reply.to_string()).unwrap();
        }
        assert_eq!(
            (
                checks,
                whole,
                close_ups,
                present,
                whole_boxes,
                every,
                windows
            ),
            (2, 1, 8, 1, 1, 1, 16)
        );
    });
    let (_process, url) = start_liquid_lab(&api_base, &["--record-4x4"]);
    let client = Client::new();
    let config: Value = client
        .get(format!("{url}/config"))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(config["provider"], "liquid");
    assert_eq!(config["model_name"], "Liquid/d1.2-ultra");
    assert_eq!(config["effort"], Value::Null);
    assert_eq!(config["plan_4x4"]["close_ups"], json!([null, 506]));
    let image = STANDARD.encode(
        tiny_skia::Pixmap::new(320, 320)
            .unwrap()
            .encode_png()
            .unwrap(),
    );
    let result: Value = client
        .post(format!("{url}/model"))
        .header("Origin", &url)
        .json(&json!({
            "trial_id": "liquid-record", "source": "camoufox", "layout": 4,
            "prompt": "Select all squares with\nbicycles\nIf there are none, click skip",
            "image": image,
        }))
        .send()
        .unwrap()
        .json()
        .unwrap();
    server.join().unwrap();
    let result = &result["result"];
    assert_eq!(result["status"], "answered", "{result}");
    assert_eq!(result["vllm_components"]["boxes"]["error"], Value::Null);
    // Record mode asks for boxes and lets them change nothing.
    assert_eq!(result["prediction"]["value"], json!([15, 16]));
    assert_eq!(result["decision_requests"], 28);
    let boxes = &result["vllm_components"]["boxes"];
    assert_eq!(boxes["removed"], json!([]));
    assert_eq!(boxes["whole_grid"]["boxes"].as_array().unwrap().len(), 9);
    assert_eq!(boxes["every"]["boxes"][0]["name"], "e01");
    assert_eq!(
        boxes["every"]["boxes"][0]["rect"],
        json!([0.5, 0.75, 1.0, 1.0])
    );
    assert_eq!(boxes["every"]["count"], 1);
    assert_eq!(boxes["every"]["found"]["e01"], 0.9);
    assert_eq!(boxes["windows"]["boxes"].as_array().unwrap().len(), 16);
    assert_eq!(boxes["windows"]["meta"]["requests"], 16);
    assert_eq!(boxes["windows"]["found"]["s07"], 0.9);
    assert_eq!(
        result["vllm_components"]["presence"]["probability"],
        json!(0.9)
    );
}

#[test]
fn liquid_decides_4x4_grids_with_its_fitted_rule_and_averages_3x3_grids() {
    // Only 4 × 4 grids have been measured with Liquid's models. So a 4 × 4 round runs the
    // model's fitted weights at an effort level, and a 3 × 3 round averages the two views and
    // selects from the service's 60%. The highest level sends what high sends: no windows.
    // Its one box request asks once for every instance, not nine times for one box each.
    let (api_base, mock) = many_at_once();
    let server = thread::spawn(move || {
        let (mut checks, mut every, mut one_box) = (0, 0, 0);
        // Two picture checks, a 3 × 3 round of four requests and a 4 × 4 round of eleven.
        for _ in 0..2 + 4 + 11 {
            let request = mock.recv_timeout(Duration::from_secs(30)).unwrap();
            let body = &request.body;
            assert!(body["images"].is_array());
            let questions = body["questions"].as_object().unwrap();
            let answers: serde_json::Map<String, Value> = questions
                .iter()
                .map(|(id, question)| {
                    let answer = if id == "red" {
                        checks += 1;
                        json!({"type": "noul", "noul": if checks == 1 { 0.99 } else { 0.01 }})
                    } else if question["type"] == "boxes" {
                        every += 1;
                        assert_eq!(body["images"].as_array().unwrap().len(), 1);
                        assert_eq!(
                            *question,
                            json!({"type": "boxes", "instructions": "every bicycle", "max": 16})
                        );
                        // A confident box over squares 15 and 16, and one too faint to count
                        // over square 14.
                        let drawn = |left: f64, confidence: f64| {
                            let rect =
                                json!({"left": left, "top": 0.75, "right": 1.0, "bottom": 1.0});
                            json!({"box": rect, "found": 0.9, "confidence": confidence})
                        };
                        json!({
                            "type": "boxes", "count": 2, "at_least": false,
                            "boxes": [drawn(0.5, 0.8), drawn(0.25, 0.2)],
                        })
                    } else if question["type"] == "box" {
                        one_box += 1;
                        json!({"type": "box", "box": null, "found": 0.1, "confidence": 0.0})
                    } else {
                        let score = match (questions.len(), id.as_str()) {
                            // 3 × 3: the first photo averages 0.65 and the second 0.55.
                            (9, "tile_01" | "tile_02") => 0.7,
                            (3, "tile_01") => 0.6,
                            (3, "tile_02") => 0.4,
                            (9 | 3, _) => 0.1,
                            (_, "tile_15" | "tile_16" | "present") => 0.9,
                            // 4 × 4: selected, but not surely enough to go unchecked.
                            (_, "tile_14") => 0.6,
                            _ => 0.0,
                        };
                        json!({"type": "noul", "noul": score})
                    };
                    (id.clone(), answer)
                })
                .collect();
            let reply = json!({
                "model": "d1.2-ultra", "answers": answers,
                "usage": {"input_tokens": 5000, "output_tokens": 0},
            });
            request.reply.send(reply.to_string()).unwrap();
        }
        assert_eq!((checks, every, one_box), (2, 1, 0));
        assert!(mock.recv_timeout(Duration::from_millis(300)).is_err());
    });
    let (_process, url) = start_liquid_lab(&api_base, &[]);
    let client = Client::new();
    let config: Value = client
        .get(format!("{url}/config"))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(config["vllm_decision"], "logistic");
    assert_eq!(config["effort"], "max");
    assert_eq!(config["plan_4x4"]["close_ups"], json!([null, 506]));
    assert_eq!(config["plan_4x4"]["boxes"]["every"], 0.3);
    assert_eq!(config["plan_4x4"]["boxes"]["whole"], Value::Null);
    assert_eq!(config["plan_4x4"]["boxes"]["windows"], Value::Null);
    assert_eq!(config["tile_threshold_4x4"], 0.35);
    assert_eq!(config["tile_threshold_3x3"], 0.6);
    assert_eq!(config["decision_requests_4x4"], 10);
    let image = STANDARD.encode(
        tiny_skia::Pixmap::new(320, 320)
            .unwrap()
            .encode_png()
            .unwrap(),
    );
    let ask = |layout: u8, prompt: &str| -> Value {
        let reply: Value = client
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id": "liquid-levels", "source": "camoufox", "layout": layout,
                "prompt": prompt, "image": image,
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        reply["result"].clone()
    };
    let small = ask(3, "Select all images with\nbicycles");
    assert_eq!(small["status"], "answered", "{small}");
    assert_eq!(small["prediction"]["value"], json!([1]));
    assert_eq!(small["vllm_components"]["rule"], "average");
    assert_eq!(small["decision_requests"], 4);
    assert_eq!(small["effort"], Value::Null);
    let large = ask(
        4,
        "Select all squares with\nbicycles\nIf there are none, click skip",
    );
    server.join().unwrap();
    assert_eq!(large["status"], "answered", "{large}");
    assert_eq!(large["prediction"]["value"], json!([15, 16]));
    assert_eq!(large["vllm_components"]["rule"], "logistic");
    assert_eq!(large["decision_requests"], 11);
    assert_eq!(large["effort"], "max");
    // Squares 15 and 16 score over 90% and need no box. Square 14 does, and only the faint
    // box reaches it, so it is dropped.
    let boxes = &large["vllm_components"]["boxes"];
    assert_eq!(boxes["picks_before"], json!([14, 15, 16]));
    assert_eq!(boxes["at_risk"], json!([14]));
    assert_eq!(boxes["removed"], json!([14]));
    assert_eq!(boxes["error"], Value::Null);
    assert_eq!(boxes["every"]["boxes"].as_array().unwrap().len(), 2);
    assert_eq!(boxes["every"]["count"], 2);
    assert_eq!(boxes["whole_grid"], Value::Null);
}

#[test]
fn liquid_refuses_a_reply_billed_less_than_its_pictures_cost() {
    // An endpoint that does not read the pictures still answers, from the text alone. The
    // bill is the only sign: a 320 × 320 picture costs 150 input tokens.
    let mock = Server::http("127.0.0.1:0").unwrap();
    let api_base = format!("http://{}/v1", mock.server_addr());
    let server = thread::spawn(move || {
        let mut request = mock.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        let body: Value = serde_json::from_reader(request.as_reader()).unwrap();
        assert!(body["questions"].get("red").is_some());
        let reply = json!({
            "model": "d1.2-ultra", "answers": {"red": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 51, "output_tokens": 0},
        });
        request
            .respond(Response::from_string(reply.to_string()))
            .unwrap();
    });
    let (_process, url) = start_liquid_lab(&api_base, &[]);
    let image = STANDARD.encode(
        tiny_skia::Pixmap::new(320, 320)
            .unwrap()
            .encode_png()
            .unwrap(),
    );
    let result: Value = Client::new()
        .post(format!("{url}/model"))
        .header("Origin", &url)
        .json(&json!({
            "trial_id": "liquid-unread", "source": "camoufox", "layout": 4,
            "prompt": "Select all squares with\nbicycles", "image": image,
        }))
        .send()
        .unwrap()
        .json()
        .unwrap();
    server.join().unwrap();
    assert_eq!(result["result"]["status"], "vision_unverified");
    let said = result["result"]["vision_check"]["error"].as_str().unwrap();
    assert!(said.contains("so they were not read"), "{said}");
}

#[test]
fn a_model_named_with_its_provider_chooses_the_provider() {
    // Two providers can serve models with similar names, so a model is known by both:
    // "vLLM/d3-ultra". The provider's own ID is what a request carries.
    for (args, provider, id, name) in [
        (
            &["--model", "vLLM/d3-ultra"][..],
            "vllm",
            "vllm-sr/d3-ultra",
            "vLLM/d3-ultra",
        ),
        (
            &["--model", "vllm/D3-Ultra"][..],
            "vllm",
            "vllm-sr/d3-ultra",
            "vLLM/d3-ultra",
        ),
        (
            &["--provider", "vllm"][..],
            "vllm",
            "vllm-sr/d3-ultra",
            "vLLM/d3-ultra",
        ),
        (
            &["--model", "typesafe/Jev"][..],
            "jev",
            "jev-latest",
            "typesafe/Jev",
        ),
        (
            &["--provider", "jev"][..],
            "jev",
            "jev-latest",
            "typesafe/Jev",
        ),
        (
            &["--provider", "jev", "--model", "jev-2"][..],
            "jev",
            "jev-2",
            "typesafe/jev-2",
        ),
    ] {
        let mut lab = LabProcess(
            Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
                .args(["serve", "--mode", "v2", "--port", "0"])
                .args(["--base-url", "http://127.0.0.1:9/v1"])
                .args((provider == "vllm").then_some("--vllm-cold-connections"))
                .args(args)
                .env("RECAPTCHA_SITE_KEY", "test-site")
                .env("RECAPTCHA_SECRET_KEY", "test-secret")
                .env("TYPESAFE_API_KEY", "test-model-key")
                .env("VLLM_SR_API_KEY", "test-vllm-key")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut line = String::new();
        BufReader::new(lab.0.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let url = line.trim().strip_prefix("Open ").unwrap();
        let config: Value = Client::new()
            .get(format!("{url}/config"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(config["provider"], provider, "{args:?}");
        assert_eq!(config["model"], id, "{args:?}");
        assert_eq!(config["model_name"], name, "{args:?}");
    }
}

#[test]
fn effort_and_tuned_settings_are_refused_where_nothing_was_measured() {
    // The levels were measured with d3-ultra's logistic rule on Google's grids. Anywhere else
    // there is nothing to choose between, and a model with no settings file has none to run on.
    let recaptcha = ["--mode", "v2", "--provider"];
    for (args, message) in [
        (
            [&recaptcha[..], &["jev", "--effort", "low"]].concat(),
            "--effort requires --provider vllm",
        ),
        (
            [
                &recaptcha[..],
                &["vllm", "--vllm-decision", "average", "--effort", "high"],
            ]
            .concat(),
            "--effort applies to --vllm-decision logistic",
        ),
        (
            [
                "--captcha",
                "hcaptcha",
                "--hcaptcha-test-key",
                "--provider",
                "vllm",
                "--effort",
                "max",
            ]
            .to_vec(),
            "--effort applies to --vllm-decision logistic",
        ),
        (
            [
                &recaptcha[..],
                &["liquid", "--vllm-decision", "average", "--effort", "low"],
            ]
            .concat(),
            "--effort applies to --vllm-decision logistic",
        ),
        (
            [&recaptcha[..], &["liquid", "--model", "d1"]].concat(),
            "no settings measured with Liquid/d1; it has them for Liquid/d1.2, Liquid/d1.2-ultra",
        ),
        (
            [&recaptcha[..], &["vllm", "--record-4x4", "--effort", "max"]].concat(),
            "--record-4x4 asks everything an effort level asks",
        ),
        (
            [
                "--captcha",
                "hcaptcha",
                "--hcaptcha-test-key",
                "--provider",
                "liquid",
            ]
            .to_vec(),
            "this CAPTCHA service has no settings for a Liquid model",
        ),
        (
            [&recaptcha[..], &["vllm", "--model", "typesafe/Jev"]].concat(),
            "typesafe/Jev is not served by the --provider given",
        ),
        (
            [&recaptcha[..], &["vllm", "--model", "vllm-sr/other"]].concat(),
            "no settings measured with vLLM/vllm-sr/other; it has them for vLLM/d3-ultra",
        ),
        (
            [
                &recaptcha[..],
                &["vllm", "--effort", "low", "--vllm-box-wait"],
            ]
            .concat(),
            "--vllm-box-wait needs a box rule",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
            .args([
                "serve",
                "--port",
                "0",
                "--base-url",
                "http://127.0.0.1:9/v1",
            ])
            .arg("--vllm-cold-connections")
            .args(&args)
            .env("RECAPTCHA_SITE_KEY", "test-site")
            .env("RECAPTCHA_SECRET_KEY", "test-secret")
            .env("TYPESAFE_API_KEY", "test-model-key")
            .env("VLLM_SR_API_KEY", "test-vllm-key")
            .env("LIQUID_API_KEY", "test-liquid-key")
            .output()
            .unwrap();
        let said = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{args:?} was accepted");
        assert!(said.contains(message), "{args:?}: {said}");
    }
}

#[test]
fn model_route_draws_new_tiles_into_the_grid_before_asking() {
    let mock = Server::http("127.0.0.1:0").unwrap();
    let api_base = format!("http://{}/v1", mock.server_addr());
    let server = thread::spawn(move || {
        let mut grids_checked = 0;
        // Two vision checks, the whole grid, and three close-up requests of three squares.
        for index in 0..6 {
            let mut request = mock.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
            let body = json_body(&mut request);
            let questions = body["questions"].as_object().unwrap();
            let answers: Value = if index < 2 {
                json!({"red":{"noul":if index == 0 { 0.99 } else { 0.01 }}})
            } else {
                // The whole-grid request shows the model the tile in its place.
                if let Some(scene) = body["state"]["grid"].as_str() {
                    let grid = image::load_from_memory(
                        &STANDARD
                            .decode(body["assets"][scene]["data_base64"].as_str().unwrap())
                            .unwrap(),
                    )
                    .unwrap()
                    .into_rgb8();
                    let [red, green, blue] = grid.get_pixel(200, 120).0;
                    assert!(red > 200 && green < 60 && blue < 60);
                    let [red, green, blue] = grid.get_pixel(80, 120).0;
                    assert!(red < 60 && green < 60 && blue < 60);
                    grids_checked += 1;
                }
                Value::Object(
                    questions
                        .keys()
                        .map(|id| (id.clone(), json!({"noul":0.0})))
                        .collect(),
                )
            };
            request
                .respond(Response::from_string(
                    json!({
                        "model":"vllm-sr/d3-ultra", "answers":answers,
                        "usage":{"input_tokens":10,"output_tokens":0},
                    })
                    .to_string(),
                ))
                .unwrap();
        }
        assert!(grids_checked > 0);
    });
    let (_process, url) = start_lab("v2-enterprise", "vllm", &api_base, &[]);
    let png = |image: image::RgbImage| {
        let mut out = std::io::Cursor::new(Vec::new());
        image.write_to(&mut out, image::ImageFormat::Png).unwrap();
        STANDARD.encode(out.into_inner())
    };
    let reply: Value = Client::new()
        .post(format!("{url}/model"))
        .header("Origin", &url)
        .json(&json!({
            "trial_id":"new-tiles", "source":"camoufox", "layout":3,
            "prompt":"Select all images with\ncars\nClick verify once there are none left",
            "image":png(image::RgbImage::new(320, 320)),
            // A 10 x 10 red picture for the box 160..240 across and 80..160 down.
            "tiles":[{
                "box":[0.5, 0.25, 0.25, 0.25],
                "image":png(image::RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0]))),
            }],
        }))
        .send()
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .unwrap();
    server.join().unwrap();
    assert_eq!(reply["result"]["prediction"]["value"], json!([]));
    // The drawn grid comes back beside the logged record, not inside it.
    assert!(reply["result"].get("grid_png").is_none());
    let drawn = image::load_from_memory(
        &STANDARD
            .decode(reply["grid_png"].as_str().unwrap())
            .unwrap(),
    )
    .unwrap()
    .into_rgb8();
    assert_eq!(drawn.dimensions(), (320, 320));
    for (x, y, pixel) in drawn.enumerate_pixels() {
        let inside = (160..240).contains(&x) && (80..160).contains(&y);
        assert_eq!(pixel.0, if inside { [255, 0, 0] } else { [0, 0, 0] });
    }
}

#[test]
fn vllm_sends_a_request_a_third_time_when_two_tries_hang() {
    let mock = Server::http("127.0.0.1:0").unwrap();
    let api_base = format!("http://{}/v1", mock.server_addr());
    let server = thread::spawn(move || {
        // The first two tries of the whole-grid request are received and never answered.
        let mut held = Vec::new();
        // Two vision checks, three close-up requests, and three tries of the whole grid.
        for index in 0..8 {
            let mut request = mock.recv_timeout(Duration::from_secs(8)).unwrap().unwrap();
            let body = json_body(&mut request);
            let questions = body["questions"].as_object().unwrap();
            let answers: Value = if index < 2 {
                json!({"red":{"noul":if index == 0 { 0.99 } else { 0.01 }}})
            } else {
                if body["state"]["grid"].is_string() && held.len() < 2 {
                    held.push(request);
                    continue;
                }
                Value::Object(
                    questions
                        .keys()
                        .map(|id| (id.clone(), json!({"noul":0.0})))
                        .collect(),
                )
            };
            request
                .respond(Response::from_string(
                    json!({
                        "model":"vllm-sr/d3-ultra", "answers":answers,
                        "usage":{"input_tokens":10,"output_tokens":0},
                    })
                    .to_string(),
                ))
                .unwrap();
        }
        assert_eq!(held.len(), 2);
    });
    // The lab gives each try three seconds, so the first has timed out and the second is
    // still unanswered when the third goes out.
    let (_process, url) = start_lab("v2-enterprise", "vllm", &api_base, &[]);
    let started = std::time::Instant::now();
    let reply: Value = Client::new()
        .post(format!("{url}/model"))
        .header("Origin", &url)
        .json(&json!({
            "trial_id":"third-try", "source":"camoufox", "layout":3,
            "prompt":"Select all images with\ncars",
            "image":STANDARD.encode(
                tiny_skia::Pixmap::new(320, 320).unwrap().encode_png().unwrap(),
            ),
        }))
        .send()
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .unwrap();
    let took = started.elapsed();
    server.join().unwrap();
    assert_eq!(reply["result"]["prediction"]["value"], json!([]));
    assert_eq!(
        reply["result"]["provider_meta"]["whole_grid"]["attempts"],
        3
    );
    // The second try goes out at 1.5 seconds and the third at 4.
    assert!(
        took > Duration::from_millis(3900) && took < Duration::from_secs(7),
        "{took:?}"
    );
}

fn assert_jpeg_crop_batches(
    provider: &'static str,
    model: &'static str,
    key: &'static str,
    extra: &[&str],
    rule: &str,
) {
    // The width of the grid copy each close-up pass is cut from. Averaged vLLM scores use the
    // capture and a 506-pixel copy, the lowest effort level the copy alone, and Jev the capture.
    let pass_widths: &'static [u32] = match (provider, rule) {
        ("vllm", "logistic") => &[506],
        ("vllm", _) => &[320, 506],
        _ => &[320],
    };
    let passes = pass_widths.len();
    // vLLM spreads the sixteen close-ups over four requests and Jev, asked to split, over two.
    let pass_batches: Vec<Vec<u8>> = if provider == "vllm" {
        (0..4)
            .map(|group| (group * 4 + 1..=group * 4 + 4).collect())
            .collect()
    } else {
        vec![(1..=8).collect(), (9..=16).collect()]
    };
    let requests_per_pass = pass_batches.len();
    let mock = Server::http("127.0.0.1:0").unwrap();
    let api_base = format!("http://{}/v1", mock.server_addr());
    let server = thread::spawn(move || {
        let mut batches = Vec::new();
        for index in 0..3 + requests_per_pass * passes {
            let mut request = mock.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
            assert_eq!(request.url(), "/v1/systemone");
            assert!(request.headers().iter().any(|header| {
                header.field.equiv("Authorization")
                    && header.value.as_str() == format!("Bearer {key}")
            }));
            assert_eq!(
                request
                    .headers()
                    .iter()
                    .any(|header| header.field.equiv("TypeSafe-Danger-Zone-Images")),
                provider == "jev",
            );
            let body = json_body(&mut request);
            assert_eq!(body["model"], model);
            if provider == "vllm" {
                assert_eq!(body["options"]["return_meta"], true);
            } else {
                assert!(body.get("options").is_none());
            }
            let questions = body["questions"].as_object().unwrap();
            let assets = body["assets"].as_object().unwrap();
            let scene = body["state"]["grid"].as_str().unwrap();
            assert_eq!(assets[scene]["format"], "jpeg");
            let answer = if index < 2 {
                assert!(questions.contains_key("red"));
                json!({"red":{"noul":if index == 0 { 0.99 } else { 0.01 }}})
            } else {
                let mut batch: Vec<_> = questions
                    .keys()
                    .map(|id| id[5..].parse::<u8>().unwrap())
                    .collect();
                batch.sort_unstable();
                batches.push(batch);
                let crop_pass = body["state"].get("target").is_some();
                assert_eq!(
                    assets.len(),
                    if crop_pass { 1 + questions.len() } else { 1 }
                );
                let scene_width = image::load_from_memory(
                    &STANDARD
                        .decode(assets[scene]["data_base64"].as_str().unwrap())
                        .unwrap(),
                )
                .unwrap()
                .width();
                if crop_pass {
                    assert!(pass_widths.contains(&scene_width));
                } else {
                    assert_eq!(scene_width, 320);
                }
                if crop_pass {
                    for question in questions.values() {
                        let reference = question["instructions"][1].as_str().unwrap();
                        let asset = &assets[reference];
                        assert_eq!(asset["format"], "jpeg");
                        assert!(
                            STANDARD
                                .decode(asset["data_base64"].as_str().unwrap())
                                .unwrap()
                                .starts_with(&[0xff, 0xd8])
                        );
                    }
                }
                Value::Object(
                    questions
                        .keys()
                        .map(|tile| {
                            let score = match (tile.as_str(), crop_pass) {
                                ("tile_16", true) if scene_width == 506 => 0.95,
                                ("tile_16", true) => 0.9,
                                ("tile_16", false) => 0.2,
                                ("tile_15", true) => 0.8,
                                _ => 0.0,
                            };
                            (tile.clone(), json!({"noul":score}))
                        })
                        .collect(),
                )
            };
            request
                .respond(Response::from_string(
                    json!({
                        "model":model, "answers":answer,
                        "usage":{"input_tokens":10,"output_tokens":0},
                        "meta":{"queue_ms":1.5,"compute_ms":30}
                    })
                    .to_string(),
                ))
                .unwrap();
        }
        batches.sort();
        let mut expected = vec![(1..=16).collect::<Vec<_>>()];
        for _ in 0..passes {
            expected.extend(pass_batches.iter().cloned());
        }
        expected.sort();
        assert_eq!(batches, expected);
    });
    let (_process, url) = start_lab("v2-enterprise", provider, &api_base, extra);
    let img = STANDARD.encode(
        tiny_skia::Pixmap::new(320, 320)
            .unwrap()
            .encode_png()
            .unwrap(),
    );
    let result: Value = Client::new()
        .post(format!("{url}/model"))
        .header("Origin", &url)
        .json(&json!({
            "trial_id":"vllm-grid", "source":"camoufox", "layout":4,
            "prompt":"Select all squares with\nbicycles", "image":img,
        }))
        .send()
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .unwrap();
    server.join().unwrap();
    assert_eq!(result["provider"], provider);
    // The lowest effort level selects from 55%.
    let low = pass_widths == [506];
    assert_eq!(result["tile_threshold"], if low { 0.55 } else { 0.5 });
    let result = &result["result"];
    // Crops say yes to 15 (0.8) and 16 (0.9, or 0.95 on the rescaled copy); the whole grid gives
    // 16 only 0.2 and its neighbours nothing, so the logistic rule declines both while averaging keeps 16.
    let expected_tiles = match rule {
        "weighted" => json!([15, 16]),
        "logistic" => json!([]),
        _ => json!([16]),
    };
    assert_eq!(result["prediction"]["value"], expected_tiles);
    if provider == "vllm" {
        match rule {
            "logistic" => assert!(result.get("whole_grid_weight").is_none()),
            "weighted" => assert_eq!(result["whole_grid_weight"], 0.35),
            _ => assert_eq!(result["whole_grid_weight"], 0.5),
        }
        let decision = if rule == "logistic" {
            "logistic"
        } else {
            "average"
        };
        assert_eq!(result["vllm_decision"], decision);
        assert_eq!(result["vllm_components"]["rule"], decision);
    }
    assert_eq!(result["vision_verified"], true);
    assert_eq!(result["decision_image_format"], "jpeg");
    assert_eq!(result["decision_requests"], 1 + requests_per_pass * passes);
    assert_eq!(
        result["usage"]["input_tokens"],
        10 + 10 * requests_per_pass * passes
    );
    assert_eq!(
        result["provider_meta"]["crop_requests"]
            .as_array()
            .unwrap()
            .len(),
        requests_per_pass * passes
    );
    assert_eq!(result["provider_meta"]["whole_grid"]["queue_ms"], 1.5);
    let key = if provider == "vllm" {
        "vllm_components"
    } else {
        "jev_components"
    };
    let close_up = &result[key]["variant"];
    // Every pass is logged with the size of its grid copy.
    for (pass, &width) in close_up["passes"]
        .as_array()
        .unwrap()
        .iter()
        .zip(pass_widths)
    {
        assert_eq!(pass["grid_pixels"], json!([width, width]));
        let score = if width == 506 { 0.95 } else { 0.9 };
        assert_eq!(pass["tile_probabilities"]["16"], score);
    }
    assert_eq!(close_up["passes"].as_array().unwrap().len(), passes);
    if passes == 2 {
        // The two passes are averaged in log-odds: 0.9 and 0.95 give about 0.929.
        let averaged = close_up["tile_probabilities"]["16"].as_f64().unwrap();
        assert!((averaged - 0.928_961).abs() < 1e-5, "{averaged}");
        let unchanged = close_up["tile_probabilities"]["15"].as_f64().unwrap();
        assert!((unchanged - 0.8).abs() < 1e-9, "{unchanged}");
    } else {
        let score = if low { 0.95 } else { 0.9 };
        assert_eq!(close_up["tile_probabilities"]["16"], score);
    }
}

/// A screenshot like hCaptcha's canvas below the instruction: a frame, a grey panel, and nine
/// textured pictures with gaps between them.
fn hcaptcha_screenshot() -> String {
    let mut image = image::RgbImage::from_pixel(1000, 720, image::Rgb([235, 235, 235]));
    for y in 20..700 {
        for x in 20..980 {
            image.put_pixel(x, y, image::Rgb([216, 216, 216]));
        }
    }
    for index in 0..9u32 {
        let (left, top) = (170 + index % 3 * 226, 30 + index / 3 * 222);
        for y in top..top + 208 {
            for x in left..left + 210 {
                let colour = [
                    (40 + index * 20 + (x * 3) % 60) as u8,
                    (90 + (y * 2) % 80) as u8,
                    (30 + (x + y) % 120) as u8,
                ];
                image.put_pixel(x, y, image::Rgb(colour));
            }
        }
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image.write_to(&mut png, image::ImageFormat::Png).unwrap();
    STANDARD.encode(png.into_inner())
}

#[test]
fn hcaptcha_grids_are_found_in_a_screenshot_and_answered_picture_by_picture() {
    const PROMPT: &str = "Pick the flying animal";
    let mock = Server::http("127.0.0.1:0").unwrap();
    let api_base = format!("http://{}/v1", mock.server_addr());
    let server = thread::spawn(move || {
        let mut photo_batches = Vec::new();
        let mut whole_grids = 0;
        for index in 0..6 {
            let mut request = mock.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
            assert_eq!(request.url(), "/v1/systemone");
            let body = json_body(&mut request);
            let questions = body["questions"].as_object().unwrap();
            let assets = body["assets"].as_object().unwrap();
            let answer = if index < 2 {
                assert!(questions.contains_key("red"));
                json!({"red":{"noul":if index == 0 { 0.99 } else { 0.01 }}})
            } else if let Some(scene) = body["state"]["grid"].as_str() {
                // The whole view: the nine pictures side by side, without the gaps.
                whole_grids += 1;
                assert_eq!(body["state"]["challenge_instructions"], PROMPT);
                assert_eq!(questions.len(), 9);
                let grid = image::load_from_memory(
                    &STANDARD
                        .decode(assets[scene]["data_base64"].as_str().unwrap())
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(grid.width(), grid.height());
                assert!((620..=660).contains(&grid.width()), "{}", grid.width());
                Value::Object(
                    questions
                        .keys()
                        .map(|tile| {
                            let score = match tile.as_str() {
                                "tile_02" => 0.9,
                                "tile_05" => 0.4,
                                _ => 0.05,
                            };
                            (tile.clone(), json!({"noul":score}))
                        })
                        .collect(),
                )
            } else {
                // Each picture by itself, asked the whole instruction: hCaptcha's text names
                // no single object the way Google's does.
                assert_eq!(body["state"], json!({"challenge_instructions": PROMPT}));
                assert_eq!(assets.len(), questions.len());
                let mut batch = Vec::new();
                for (tile, question) in questions {
                    batch.push(tile[5..].parse::<u8>().unwrap());
                    let instructions = &question["instructions"];
                    assert!(
                        instructions["task"]
                            .as_str()
                            .unwrap()
                            .starts_with("PHOTO is one of nine separate pictures")
                    );
                    assert!(assets.contains_key(instructions["PHOTO"].as_str().unwrap()));
                }
                batch.sort_unstable();
                photo_batches.push(batch);
                Value::Object(
                    questions
                        .keys()
                        .map(|tile| {
                            let score = match tile.as_str() {
                                "tile_02" | "tile_05" => 0.8,
                                "tile_07" => 0.9,
                                _ => 0.05,
                            };
                            (tile.clone(), json!({"noul":score}))
                        })
                        .collect(),
                )
            };
            request
                .respond(Response::from_string(
                    json!({
                        "model":"vllm-sr/d3-ultra", "answers":answer,
                        "usage":{"input_tokens":10,"output_tokens":0},
                    })
                    .to_string(),
                ))
                .unwrap();
        }
        assert_eq!(whole_grids, 1);
        photo_batches.sort();
        assert_eq!(photo_batches, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    });
    let mut process = LabProcess(
        Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
            .args(["serve", "--captcha", "hcaptcha", "--hcaptcha-test-key"])
            .args(["--port", "0", "--provider", "vllm", "--base-url", &api_base])
            .args(["--timeout", "3", "--vllm-cold-connections"])
            .env("VLLM_SR_API_KEY", "test-vllm-key")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut line = String::new();
    BufReader::new(process.0.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let url = line.trim().strip_prefix("Open ").unwrap().to_string();
    // hCaptcha refuses pages on localhost, so the lab is opened under a name of its own.
    let port: u16 = url
        .strip_prefix("http://hypercaptcha.localhost:")
        .unwrap()
        .parse()
        .unwrap();
    let client = Client::builder()
        .resolve("hypercaptcha.localhost", ([127, 0, 0, 1], port).into())
        .build()
        .unwrap();
    let refused = Client::new()
        .get(format!("http://localhost:{port}/config"))
        .send()
        .unwrap();
    assert_eq!(refused.status(), 403);
    let config: Value = client
        .get(format!("{url}/config"))
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(config["captcha"], "hcaptcha");
    assert_eq!(config["vllm_decision"], "average");
    assert_eq!(config["decision_requests_3x3"], 4);
    // Nothing has been measured on hCaptcha's grids, so there are no levels to choose from.
    assert_eq!(config["effort"], Value::Null);
    assert_eq!(config["plan_4x4"], Value::Null);

    let post = |path: &str, body: Value| {
        client
            .post(format!("{url}/{path}"))
            .header("Origin", &url)
            .json(&body)
            .send()
            .unwrap()
    };
    let image = hcaptcha_screenshot();
    let found: Value = post(
        "pictures",
        json!({"trial_id":"hcaptcha-grid", "source":"chrome", "image":image}),
    )
    .json()
    .unwrap();
    assert_eq!(found["captcha"], "hcaptcha");
    assert_eq!(found["result"]["status"], "grid");
    let pictures = found["result"]["pictures"].as_array().unwrap();
    assert_eq!(pictures.len(), 9);
    // The middle place, as fractions of the screenshot: the middle picture and up to half of
    // the gap around it.
    let middle: Vec<f64> = pictures[4]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let (picture, slack) = ([0.396, 0.35, 0.21, 0.2889], 0.012);
    assert!(middle[0] <= picture[0] && middle[0] >= picture[0] - slack);
    assert!(middle[1] <= picture[1] && middle[1] >= picture[1] - slack);
    assert!(middle[0] + middle[2] >= picture[0] + picture[2] - 0.002);
    assert!(middle[1] + middle[3] >= picture[1] + picture[3] - 0.002);
    assert!(middle[2] <= picture[2] + 2.0 * slack && middle[3] <= picture[3] + 2.0 * slack);

    let wrong_layout = post(
        "model",
        json!({"trial_id":"hcaptcha-grid", "source":"chrome", "layout":4, "prompt":PROMPT, "image":image}),
    );
    assert_eq!(wrong_layout.status(), 400);
    let reply: Value = post(
        "model",
        json!({"trial_id":"hcaptcha-grid", "source":"chrome", "prompt":PROMPT, "image":image}),
    )
    .error_for_status()
    .unwrap()
    .json()
    .unwrap();
    server.join().unwrap();
    assert_eq!(reply["captcha"], "hcaptcha");
    assert!(reply.get("mode").is_none());
    assert_eq!(reply["tile_threshold"], 0.5);
    assert_eq!(reply["pictures"], found["result"]["pictures"]);
    let grid = image::load_from_memory(
        &STANDARD
            .decode(reply["grid_png"].as_str().unwrap())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(grid.width(), grid.height());
    assert!((620..=660).contains(&grid.width()));
    let result = &reply["result"];
    // The two views are averaged: 2 scores 0.85 and 5 scores 0.6. Picture 7 scores 0.9 by
    // itself but 0.05 in the whole view, which leaves it under one half.
    assert_eq!(result["prediction"]["value"], json!([2, 5]));
    assert_eq!(result["vllm_decision"], "average");
    assert_eq!(result["whole_grid_weight"], 0.5);
    assert_eq!(result["decision_requests"], 4);
    assert!(result["vllm_components"].get("boxes").is_none());
    assert!(result["vllm_components"].get("presence").is_none());
}

struct TempFile(std::path::PathBuf);

impl TempFile {
    fn write(name: &str, content: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("hypercaptcha-test-{}-{name}", std::process::id()));
        std::fs::write(&path, content).unwrap();
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn custom_model_config_drives_assets_and_images_dialects_end_to_end() {
    // Part 1: Custom `systemone-assets` endpoint with gzip, return_meta, custom header, custom
    // env var, and custom fitted logistic weights + effort levels.
    {
        let mock = Server::http("127.0.0.1:0").unwrap();
        let api_base = format!("http://{}/v1", mock.server_addr());
        let config_file = TempFile::write(
            "custom-assets.toml",
            &format!(
                r#"
schema_version = 1
service = "recaptcha"
name = "Acme/vision-alpha"
provider = "custom"
model_id = "acme/vision-alpha-v1"

[endpoint]
dialect = "systemone-assets"
base_url = "{api_base}"
api_key_env = "ACME_ALPHA_KEY"
image_format = "jpeg"
crop_filter = "bilinear"
gzip = true
max_images = 16
multi_state_windows = true
supports_every_box = false
return_meta = true
hedge_on_stall = false
warm_connections = false
min_patch_tokens = false
extra_headers = {{ "X-Acme-Header" = "alpha-42" }}

[tuning]
default_decision = "logistic"

[tuning.logistic_4x4]
weights = [0.223152, 1.092908, 0.213385, 0.18743, -1.509533]

[tuning.efforts.low]
close_ups = ["capture"]
select_from = 0.55

[tuning.efforts.medium]
close_ups = ["capture"]
select_from = 0.55

[tuning.efforts.medium.boxes]
exempt_from = 0.9525741268224334
whole = 0.3
cover = 0.01
inside = 0.0

[tuning.efforts.high]
close_ups = ["capture", 506]
select_from = 0.5

[tuning.efforts.max]
close_ups = ["capture", 506]
select_from = 0.5
"#
            ),
        );

        let server =
            thread::spawn(move || {
                // 2 vision checks + 1 whole grid + 4 close-up requests + 1 whole-grid box request.
                let mut saw_boxes = 0;
                for index in 0..8 {
                    let mut request = mock.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
                    assert_eq!(request.url(), "/v1/systemone");
                    assert!(request.headers().iter().any(|h| {
                        h.field.equiv("Authorization") && h.value.as_str() == "Bearer secret-alpha"
                    }));
                    assert!(request.headers().iter().any(|h| {
                        h.field.equiv("X-Acme-Header") && h.value.as_str() == "alpha-42"
                    }));
                    assert!(request.headers().iter().any(|h| {
                        h.field.equiv("Content-Encoding") && h.value.as_str() == "gzip"
                    }));
                    let body = json_body(&mut request);
                    assert_eq!(body["model"], "acme/vision-alpha-v1");
                    assert_eq!(body["options"]["return_meta"], true);
                    let reply = if index < 2 {
                        assert!(body["assets"].is_object());
                        json!({
                            "model": "acme/vision-alpha-v1",
                            "answers": {"red": {"noul": if index == 0 { 0.99 } else { 0.01 }}},
                            "usage": {"input_tokens": 10, "output_tokens": 0}
                        })
                    } else if body["images"].is_array() {
                        saw_boxes += 1;
                        // Cover only square 16 so square 15 gets vetoed by the box rule.
                        let square16 =
                            json!({"left": 0.75, "top": 0.75, "right": 1.0, "bottom": 1.0});
                        let answers: serde_json::Map<String, Value> = body["questions"]
                            .as_object()
                            .unwrap()
                            .keys()
                            .enumerate()
                            .map(|(i, name)| {
                                let answer = if i == 0 {
                                    json!({"type": "box", "box": square16, "confidence": 0.8})
                                } else {
                                    json!({"type": "box", "box": null, "confidence": 0.0})
                                };
                                (name.clone(), answer)
                            })
                            .collect();
                        json!({
                            "model": "acme/vision-alpha-v1",
                            "answers": answers,
                            "usage": {"input_tokens": 10, "output_tokens": 0}
                        })
                    } else {
                        assert!(body["assets"].is_object());
                        let answers: serde_json::Map<String, Value> = body["questions"]
                            .as_object()
                            .unwrap()
                            .keys()
                            .map(|id| {
                                let score = if ["tile_15", "tile_16"].contains(&id.as_str()) {
                                    0.9
                                } else {
                                    0.0
                                };
                                (id.clone(), json!({"noul": score}))
                            })
                            .collect();
                        json!({
                            "model": "acme/vision-alpha-v1",
                            "answers": answers,
                            "usage": {"input_tokens": 10, "output_tokens": 0}
                        })
                    };
                    request
                        .respond(Response::from_string(reply.to_string()))
                        .unwrap();
                }
                assert_eq!(saw_boxes, 1);
            });

        let mut lab = LabProcess(
            Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
                .args([
                    "serve",
                    "--mode",
                    "v2",
                    "--port",
                    "0",
                    "--timeout",
                    "5",
                    "--model-config",
                    config_file.0.to_str().unwrap(),
                    "--effort",
                    "medium",
                    "--box-wait",
                ])
                .env("RECAPTCHA_SITE_KEY", "test-site")
                .env("RECAPTCHA_SECRET_KEY", "test-secret")
                .env("ACME_ALPHA_KEY", "secret-alpha")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut line = String::new();
        BufReader::new(lab.0.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let url = line.trim().strip_prefix("Open ").unwrap().to_string();

        let client = Client::new();
        let config: Value = client
            .get(format!("{url}/config"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(config["provider"], "custom");
        assert_eq!(config["model"], "acme/vision-alpha-v1");
        assert_eq!(config["model_name"], "Acme/vision-alpha");
        assert_eq!(config["effort"], "medium");
        assert_eq!(config["supports_vision"], true);
        assert_eq!(config["requires_vision_check"], true);
        assert_eq!(config["is_decision"], true);
        assert_eq!(config["plan_4x4"]["close_ups"], json!([null]));
        assert_eq!(config["tile_threshold_4x4"], 0.55);

        let image = STANDARD.encode(
            tiny_skia::Pixmap::new(320, 320)
                .unwrap()
                .encode_png()
                .unwrap(),
        );
        let reply: Value = client
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id": "custom-assets",
                "source": "chrome",
                "layout": 4,
                "prompt": "Select all squares with\nbicycles",
                "image": image,
            }))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        server.join().unwrap();
        assert_eq!(reply["result"]["status"], "answered");
        assert_eq!(reply["result"]["prediction"]["value"], json!([16]));
        assert_eq!(
            reply["result"]["vllm_components"]["boxes"]["removed"],
            json!([15])
        );
    }

    // Part 2: Custom `systemone-images` endpoint with positional images, custom max_images,
    // supports_every_box, min_patch_tokens, and custom header.
    {
        let (api_base, mock) = many_at_once();
        let config_file = TempFile::write(
            "custom-images.toml",
            &format!(
                r#"
schema_version = 1
service = "recaptcha"
name = "Beta/img-v2"
provider = "custom"
model_id = "beta-img-v2"

[endpoint]
dialect = "systemone-images"
base_url = "{api_base}"
api_key_env = "BETA_IMG_KEY"
image_format = "jpeg"
crop_filter = "bilinear"
gzip = false
max_images = 6
multi_state_windows = false
supports_every_box = true
return_meta = false
hedge_on_stall = false
warm_connections = false
min_patch_tokens = true
extra_headers = {{ "X-Beta-Dialect" = "positional" }}

[tuning]
default_decision = "logistic"

[tuning.logistic_4x4]
weights = [0.325707, 1.717633, 0.364472, 0.151428, -2.083722]

[tuning.efforts.low]
close_ups = ["capture"]
select_from = 0.4

[tuning.efforts.medium]
close_ups = ["capture"]
select_from = 0.4

[tuning.efforts.medium.boxes]
exempt_from = 0.9
every = 0.3
cover = 0.01
inside = 0.0

[tuning.efforts.high]
close_ups = ["capture"]
select_from = 0.4

[tuning.efforts.high.boxes]
exempt_from = 0.9
every = 0.3
cover = 0.01
inside = 0.0

[tuning.efforts.max]
close_ups = ["capture"]
select_from = 0.4

[tuning.efforts.max.boxes]
exempt_from = 0.9
every = 0.3
cover = 0.01
inside = 0.0
"#
            ),
        );

        let server = thread::spawn(move || {
            let mut saw_every = 0;
            // 2 vision checks + 1 whole grid + 4 close-up requests (max_images=6 caps 4x4 batch
            // to 5 crops -> 4 batches of 4) + 1 every-instance box request.
            for index in 0..8 {
                let request = mock.recv_timeout(Duration::from_secs(10)).unwrap();
                assert_eq!(request.header("Authorization"), Some("Bearer secret-beta"));
                assert_eq!(request.header("X-Beta-Dialect"), Some("positional"));
                assert_eq!(request.header("Content-Encoding"), None);
                let body = &request.body;
                assert_eq!(body["model"], "beta-img-v2");
                assert!(body.get("assets").is_none());
                let images = body["images"].as_array().unwrap();
                assert!(images.len() <= 6, "exceeded max_images=6: {}", images.len());
                let questions = body["questions"].as_object().unwrap();
                let answers: serde_json::Map<String, Value> = if index < 2 {
                    assert!(body.get("options").is_none());
                    json!({"red": {"noul": if index == 0 { 0.99 } else { 0.01 }}})
                        .as_object()
                        .unwrap()
                        .clone()
                } else if questions.values().all(|q| q["type"] == "boxes") {
                    saw_every += 1;
                    let rect = json!({"left": 0.5, "top": 0.75, "right": 1.0, "bottom": 1.0});
                    json!({"q": {
                        "type": "boxes", "count": 1, "at_least": false,
                        "boxes": [{"box": rect, "found": 0.9, "confidence": 0.8}],
                    }})
                    .as_object()
                    .unwrap()
                    .clone()
                } else {
                    assert!(body.get("options").is_none());
                    questions
                        .keys()
                        .map(|id| {
                            let score = match id.as_str() {
                                "tile_15" | "tile_16" => 0.9,
                                "tile_14" => 0.6,
                                _ => 0.0,
                            };
                            (id.clone(), json!({"type": "noul", "noul": score}))
                        })
                        .collect()
                };
                let reply = json!({
                    "model": "beta-img-v2",
                    "answers": answers,
                    "usage": {"input_tokens": 5000, "output_tokens": 0},
                });
                request.reply.send(reply.to_string()).unwrap();
            }
            assert_eq!(saw_every, 1);
        });

        let mut lab = LabProcess(
            Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
                .args([
                    "serve",
                    "--mode",
                    "v2",
                    "--port",
                    "0",
                    "--timeout",
                    "10",
                    "--model-config",
                    config_file.0.to_str().unwrap(),
                    "--close-ups-per-request",
                    "9",
                    "--box-wait",
                ])
                .env("RECAPTCHA_SITE_KEY", "test-site")
                .env("RECAPTCHA_SECRET_KEY", "test-secret")
                .env("BETA_IMG_KEY", "secret-beta")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut line = String::new();
        BufReader::new(lab.0.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let url = line.trim().strip_prefix("Open ").unwrap().to_string();

        let client = Client::new();
        let image = STANDARD.encode(
            tiny_skia::Pixmap::new(320, 320)
                .unwrap()
                .encode_png()
                .unwrap(),
        );
        let reply: Value = client
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id": "custom-images",
                "source": "chrome",
                "layout": 4,
                "prompt": "Select all squares with\nbicycles",
                "image": image,
            }))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        server.join().unwrap();
        assert_eq!(reply["result"]["status"], "answered");
        assert_eq!(reply["result"]["prediction"]["value"], json!([15, 16]));
        assert_eq!(
            reply["result"]["vllm_components"]["boxes"]["removed"],
            json!([14])
        );
    }
}

#[test]
fn unmeasured_models_can_be_recorded_and_evaluated_with_provider_agnostic_flags() {
    // 1. Unmeasured vLLM model with `--record` and `--cold-connections` works on both 3 × 3 and 4 × 4.
    {
        let mock = Server::http("127.0.0.1:0").unwrap();
        let api_base = format!("http://{}/v1", mock.server_addr());
        let server = thread::spawn(move || {
            // 2 vision checks + 3 × 3 (1 whole + 3 photo batches)
            // + 4 × 4 record (1 whole + 8 close-up batches + 1 presence + 1 whole box + 1 windows box).
            for index in 0..2 + 4 + 12 {
                let mut request = mock.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
                let body = json_body(&mut request);
                assert_eq!(body["model"], "vllm-sr/brand-new");
                let reply = if index < 2 {
                    json!({"answers": {"red": {"noul": if index == 0 { 0.99 } else { 0.01 }}}})
                } else if body["states"].is_object() {
                    let states: serde_json::Map<String, Value> = body["states"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .map(|name| {
                            (
                                name.clone(),
                                json!({"answers": {"q": {"type": "box", "box": null, "confidence": 0.0}}}),
                            )
                        })
                        .collect();
                    json!({"states": states})
                } else if body["images"].is_array() {
                    let answers: serde_json::Map<String, Value> = body["questions"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .map(|name| {
                            (
                                name.clone(),
                                json!({"type": "box", "box": null, "confidence": 0.0}),
                            )
                        })
                        .collect();
                    json!({"answers": answers})
                } else {
                    let answers: serde_json::Map<String, Value> = body["questions"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .map(|id| {
                            let score = if matches!(id.as_str(), "tile_01" | "tile_16" | "present")
                            {
                                0.9
                            } else {
                                0.0
                            };
                            (id.clone(), json!({"noul": score}))
                        })
                        .collect();
                    json!({"answers": answers})
                };
                let mut reply = reply;
                reply["model"] = json!("vllm-sr/brand-new");
                reply["usage"] = json!({"input_tokens": 10, "output_tokens": 0});
                request
                    .respond(Response::from_string(reply.to_string()))
                    .unwrap();
            }
        });

        let mut lab = LabProcess(
            Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
                .args([
                    "serve",
                    "--mode",
                    "v2",
                    "--port",
                    "0",
                    "--provider",
                    "vllm",
                    "--model",
                    "vllm-sr/brand-new",
                    "--base-url",
                    &api_base,
                    "--record",
                    "--cold-connections",
                ])
                .env("RECAPTCHA_SITE_KEY", "test-site")
                .env("RECAPTCHA_SECRET_KEY", "test-secret")
                .env("VLLM_SR_API_KEY", "test-vllm-key")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut line = String::new();
        BufReader::new(lab.0.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let url = line.trim().strip_prefix("Open ").unwrap().to_string();
        let client = Client::new();
        let config: Value = client
            .get(format!("{url}/config"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(config["model"], "vllm-sr/brand-new");
        assert_eq!(config["vllm_decision"], "average");
        assert_eq!(config["effort"], Value::Null);
        assert_eq!(config["plan_4x4"]["close_ups"], json!([null, 506]));
        assert_eq!(config["plan_4x4"]["boxes"]["every"], Value::Null);

        let image = STANDARD.encode(
            tiny_skia::Pixmap::new(320, 320)
                .unwrap()
                .encode_png()
                .unwrap(),
        );
        let r3: Value = client
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id": "unmeasured-3x3",
                "source": "chrome",
                "layout": 3,
                "prompt": "Select all images with\nbicycles",
                "image": image,
            }))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(r3["result"]["prediction"]["value"], json!([1]));
        assert_eq!(r3["result"]["vllm_components"]["rule"], "average");

        let r4: Value = client
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id": "unmeasured-4x4",
                "source": "chrome",
                "layout": 4,
                "prompt": "Select all squares with\nbicycles\nIf there are none, click skip",
                "image": image,
            }))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        server.join().unwrap();
        assert_eq!(r4["result"]["prediction"]["value"], json!([1, 16]));
        assert_eq!(
            r4["result"]["vllm_components"]["boxes"]["removed"],
            json!([])
        );
        assert_eq!(
            r4["result"]["vllm_components"]["presence"]["probability"],
            0.9
        );
    }

    // 1b. Unmeasured Liquid model with `--record` uses positional images, `every` box, and 16
    // single-state window boxes without needing a pre-existing settings file.
    {
        let (api_base, mock) = many_at_once();
        let server = thread::spawn(move || {
            let mut saw_every = 0;
            let mut saw_windows = 0;
            // 2 vision checks + 1 whole + 8 close-up batches + 1 presence + 1 whole box + 1 every box + 16 window boxes.
            for index in 0..2 + 1 + 8 + 1 + 2 + 16 {
                let request = mock.recv_timeout(Duration::from_secs(30)).unwrap();
                let body = &request.body;
                assert_eq!(body["model"], "d1.9-unmeasured");
                assert!(body["images"].is_array());
                let questions = body["questions"].as_object().unwrap();
                let answers: serde_json::Map<String, Value> = if index < 2 {
                    json!({"red": {"noul": if index == 0 { 0.99 } else { 0.01 }}})
                        .as_object()
                        .unwrap()
                        .clone()
                } else if questions.values().all(|q| q["type"] == "boxes") {
                    saw_every += 1;
                    json!({"q": {"type": "boxes", "count": 0, "at_least": false, "boxes": []}})
                        .as_object()
                        .unwrap()
                        .clone()
                } else if questions.values().all(|q| q["type"] == "box") {
                    if questions.len() == 1 {
                        saw_windows += 1;
                    }
                    questions
                        .keys()
                        .map(|name| {
                            (
                                name.clone(),
                                json!({"type": "box", "box": null, "confidence": 0.0}),
                            )
                        })
                        .collect()
                } else {
                    questions
                        .keys()
                        .map(|id| {
                            let score = if matches!(id.as_str(), "tile_16" | "present") {
                                0.9
                            } else {
                                0.0
                            };
                            (id.clone(), json!({"type": "noul", "noul": score}))
                        })
                        .collect()
                };
                let reply = json!({
                    "model": "d1.9-unmeasured",
                    "answers": answers,
                    "usage": {"input_tokens": 5000, "output_tokens": 0},
                });
                request.reply.send(reply.to_string()).unwrap();
            }
            assert_eq!((saw_every, saw_windows), (1, 16));
        });

        let mut lab = LabProcess(
            Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
                .args([
                    "serve",
                    "--mode",
                    "v2",
                    "--port",
                    "0",
                    "--timeout",
                    "20",
                    "--provider",
                    "liquid",
                    "--model",
                    "d1.9-unmeasured",
                    "--base-url",
                    &api_base,
                    "--record",
                ])
                .env("RECAPTCHA_SITE_KEY", "test-site")
                .env("RECAPTCHA_SECRET_KEY", "test-secret")
                .env("LIQUID_API_KEY", "test-liquid-key")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut line = String::new();
        BufReader::new(lab.0.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let url = line.trim().strip_prefix("Open ").unwrap().to_string();
        let image = STANDARD.encode(
            tiny_skia::Pixmap::new(320, 320)
                .unwrap()
                .encode_png()
                .unwrap(),
        );
        let reply: Value = Client::new()
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id": "liquid-unmeasured-record",
                "source": "chrome",
                "layout": 4,
                "prompt": "Select all squares with\nbicycles\nIf there are none, click skip",
                "image": image,
            }))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        server.join().unwrap();
        assert_eq!(reply["result"]["status"], "answered");
        assert_eq!(reply["result"]["prediction"]["value"], json!([16]));
        assert_eq!(reply["result"]["decision_requests"], 28);
        assert_eq!(reply["result"]["vllm_components"]["rule"], "average");
        assert_eq!(
            reply["result"]["vllm_components"]["boxes"]["removed"],
            json!([])
        );
    }

    // 2. Unmeasured Liquid model on hCaptcha works when `--decision-rule average` is passed.
    {
        let (api_base, mock) = many_at_once();
        let server = thread::spawn(move || {
            // 2 vision checks + 1 whole grid + 3 photo batches.
            for index in 0..6 {
                let request = mock.recv_timeout(Duration::from_secs(10)).unwrap();
                let body = &request.body;
                assert_eq!(body["model"], "d1.2");
                assert!(body["images"].is_array());
                let questions = body["questions"].as_object().unwrap();
                let answers: serde_json::Map<String, Value> = if index < 2 {
                    json!({"red": {"noul": if index == 0 { 0.99 } else { 0.01 }}})
                        .as_object()
                        .unwrap()
                        .clone()
                } else {
                    questions
                        .keys()
                        .map(|id| {
                            let score = if id == "tile_03" { 0.85 } else { 0.1 };
                            (id.clone(), json!({"type": "noul", "noul": score}))
                        })
                        .collect()
                };
                let reply = json!({
                    "model": "d1.2",
                    "answers": answers,
                    "usage": {"input_tokens": 5000, "output_tokens": 0},
                });
                request.reply.send(reply.to_string()).unwrap();
            }
        });

        let mut lab = LabProcess(
            Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
                .args([
                    "serve",
                    "--captcha",
                    "hcaptcha",
                    "--hcaptcha-test-key",
                    "--port",
                    "0",
                    "--provider",
                    "liquid",
                    "--model",
                    "d1.2",
                    "--decision-rule",
                    "average",
                    "--base-url",
                    &api_base,
                ])
                .env("LIQUID_API_KEY", "test-liquid-key")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut line = String::new();
        BufReader::new(lab.0.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let url = line.trim().strip_prefix("Open ").unwrap().to_string();
        let port: u16 = url
            .strip_prefix("http://hypercaptcha.localhost:")
            .unwrap()
            .parse()
            .unwrap();
        let client = Client::builder()
            .resolve("hypercaptcha.localhost", ([127, 0, 0, 1], port).into())
            .build()
            .unwrap();
        let config: Value = client
            .get(format!("{url}/config"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(config["captcha"], "hcaptcha");
        assert_eq!(config["provider"], "liquid");
        assert_eq!(config["supports_vision"], true);
        assert_eq!(config["requires_vision_check"], true);
        assert_eq!(config["is_decision"], true);

        let reply: Value = client
            .post(format!("{url}/model"))
            .header("Origin", &url)
            .json(&json!({
                "trial_id": "hcaptcha-liquid",
                "source": "chrome",
                "prompt": "Pick the flying animal",
                "image": hcaptcha_screenshot(),
            }))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        server.join().unwrap();
        assert_eq!(reply["result"]["status"], "answered");
        assert_eq!(reply["result"]["prediction"]["value"], json!([3]));
    }
}

#[test]
fn config_endpoint_exposes_vision_and_decision_capabilities() {
    let cases: &[(&[&str], bool, bool, bool)] = &[
        (&[], false, false, false),
        (
            &[
                "--provider",
                "openai-completions",
                "--model",
                "gpt-3.5-turbo-instruct",
            ],
            false,
            false,
            false,
        ),
        (
            &["--provider", "openai-chat", "--model", "gpt-4o"],
            true,
            false,
            false,
        ),
        (&["--provider", "jev"], true, true, true),
        (
            &["--provider", "vllm", "--cold-connections"],
            true,
            true,
            true,
        ),
        (&["--provider", "liquid"], true, true, true),
    ];
    for &(args, supports_vision, requires_vision_check, is_decision) in cases {
        let mut lab = LabProcess(
            Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
                .args(["serve", "--mode", "v2", "--port", "0"])
                .args(args)
                .env("RECAPTCHA_SITE_KEY", "test-site")
                .env("RECAPTCHA_SECRET_KEY", "test-secret")
                .env("OPENAI_API_KEY", "test-openai-key")
                .env("TYPESAFE_API_KEY", "test-jev-key")
                .env("VLLM_SR_API_KEY", "test-vllm-key")
                .env("LIQUID_API_KEY", "test-liquid-key")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut line = String::new();
        BufReader::new(lab.0.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let url = line.trim().strip_prefix("Open ").unwrap();
        let config: Value = Client::new()
            .get(format!("{url}/config"))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(config["supports_vision"], supports_vision, "{args:?}");
        assert_eq!(
            config["requires_vision_check"], requires_vision_check,
            "{args:?}"
        );
        assert_eq!(config["is_decision"], is_decision, "{args:?}");
    }

    // Service mismatch and missing --model-config for --provider custom are rejected.
    let hcaptcha_cfg = TempFile::write(
        "hcaptcha-only.toml",
        r#"
schema_version = 1
service = "hcaptcha"
name = "vLLM/d3-ultra"
provider = "vllm"
model_id = "vllm-sr/d3-ultra"
"#,
    );
    let mismatch = Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
        .args([
            "serve",
            "--mode",
            "v2",
            "--port",
            "0",
            "--model-config",
            hcaptcha_cfg.0.to_str().unwrap(),
        ])
        .env("RECAPTCHA_SITE_KEY", "test-site")
        .env("RECAPTCHA_SECRET_KEY", "test-secret")
        .env("VLLM_SR_API_KEY", "test-vllm-key")
        .output()
        .unwrap();
    assert!(!mismatch.status.success());
    assert!(
        String::from_utf8_lossy(&mismatch.stderr).contains("is for --captcha hcaptcha"),
        "{}",
        String::from_utf8_lossy(&mismatch.stderr)
    );

    let custom_no_cfg = Command::new(env!("CARGO_BIN_EXE_hypercaptcha"))
        .args([
            "serve",
            "--mode",
            "v2",
            "--port",
            "0",
            "--provider",
            "custom",
        ])
        .env("RECAPTCHA_SITE_KEY", "test-site")
        .env("RECAPTCHA_SECRET_KEY", "test-secret")
        .output()
        .unwrap();
    assert!(!custom_no_cfg.status.success());
    assert!(
        String::from_utf8_lossy(&custom_no_cfg.stderr)
            .contains("--provider custom requires --model-config"),
        "{}",
        String::from_utf8_lossy(&custom_no_cfg.stderr)
    );
}
