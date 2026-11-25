//! Liquid's decision endpoint.
//!
//! It answers the same typed questions as the other decision providers, with these
//! differences, each found by sending it a request:
//!
//! - Pictures go in an ordered `images` list and are referred to by position. The named
//!   `assets` the other providers take are ignored without an error: the request is answered
//!   from its text alone.
//! - A request body may not be compressed.
//! - A request holds one state, so the sixteen window pictures of a box round go in sixteen
//!   requests.
//! - A reply carries no timings, and the same request is not answered the same way twice.
//!
//! So a request is built as for the others and rewritten here, and a reply is held to the
//! tokens its pictures must have cost.

use std::io::Cursor;

use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

/// The most pictures one request may carry.
pub const MOST_IMAGES: usize = 8;

/// Rewrites a request that names its pictures in `assets` into one that lists them in
/// `images`. Each mention of a picture's name in the state or the questions becomes its
/// place in the list. Also returns the fewest input tokens the pictures can cost.
pub fn request(mut request: Value) -> Result<(Value, u64)> {
    let Some(Value::Object(assets)) = request
        .as_object_mut()
        .and_then(|fields| fields.remove("assets"))
    else {
        return Ok((request, 0));
    };
    // Pictures are listed in the order the request first mentions them: the state's, then
    // each question's.
    let mut order = Vec::new();
    for part in ["state", "questions"] {
        mentions(&request[part], &assets, &mut order);
    }
    for name in assets.keys() {
        if !order.contains(name) {
            order.push(name.clone());
        }
    }
    ensure!(
        order.len() <= MOST_IMAGES,
        "a Liquid request takes at most {MOST_IMAGES} pictures, and this one has {}",
        order.len()
    );
    for part in ["state", "questions"] {
        point_at(&mut request[part], &order, false);
    }
    let mut tokens = 0;
    let mut images = Vec::with_capacity(order.len());
    for name in &order {
        let asset = &assets[name];
        let (format, data) = asset["format"]
            .as_str()
            .zip(asset["data_base64"].as_str())
            .with_context(|| format!("the picture {name} has no format or data"))?;
        tokens += picture_tokens(data).with_context(|| format!("measure the picture {name}"))?;
        images.push(json!(format!("data:image/{format};base64,{data}")));
    }
    request["images"] = json!(images);
    Ok((request, tokens))
}

/// What a picture costs: one and a half input tokens for each 32-pixel patch that covers it.
fn picture_tokens(base64: &str) -> Result<u64> {
    let bytes = STANDARD.decode(base64).context("decode the picture")?;
    let (width, height) = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .into_dimensions()?;
    let patches = u64::from(width.div_ceil(32)) * u64::from(height.div_ceil(32));
    Ok((patches * 3).div_ceil(2))
}

/// Collects the picture names a value mentions, in order and once each.
fn mentions(value: &Value, assets: &Map<String, Value>, order: &mut Vec<String>) {
    match value {
        Value::String(text) if assets.contains_key(text) && !order.contains(text) => {
            order.push(text.clone());
        }
        Value::Array(items) => items.iter().for_each(|item| mentions(item, assets, order)),
        Value::Object(fields) => fields
            .values()
            .for_each(|item| mentions(item, assets, order)),
        _ => {}
    }
}

/// Replaces each picture's name with its place in the list. Under a key, "image 2" reads as
/// that key's value. Loose in a list of instructions it would not, so there it is spelt out.
fn point_at(value: &mut Value, order: &[String], in_list: bool) {
    match value {
        Value::String(text) => {
            if let Some(place) = order.iter().position(|name| name == text) {
                *text = if in_list {
                    format!("PHOTO is image {}.", place + 1)
                } else {
                    format!("image {}", place + 1)
                };
            }
        }
        Value::Array(items) => items
            .iter_mut()
            .for_each(|item| point_at(item, order, true)),
        Value::Object(fields) => fields
            .values_mut()
            .for_each(|item| point_at(item, order, false)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(width: u32, height: u32) -> Value {
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut bytes)
            .encode(
                &vec![0; (width * height * 3) as usize],
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        json!({"type": "image", "format": "jpeg", "data_base64": STANDARD.encode(bytes)})
    }

    #[test]
    fn named_pictures_become_a_list_referred_to_by_place() {
        let (sent, tokens) = request(json!({
            "model": "d1.2",
            "state": {"grid": "scene", "target": "bus"},
            "assets": {"crop-b": picture(64, 64), "crop-a": picture(33, 32), "scene": picture(386, 391)},
            "questions": {
                "tile_02": {"type": "noul", "instructions": ["Is it in PHOTO?", "crop-b"]},
                "tile_01": {"type": "noul", "instructions": {"task": "Is it in PHOTO?", "PHOTO": "crop-a"}},
            },
        }))
        .unwrap();
        assert!(sent.get("assets").is_none());
        assert_eq!(sent["state"], json!({"grid": "image 1", "target": "bus"}));
        // Questions are read in the order of their names, so the first tile's picture is second.
        assert_eq!(
            sent["questions"]["tile_01"]["instructions"]["PHOTO"],
            "image 2"
        );
        assert_eq!(
            sent["questions"]["tile_02"]["instructions"],
            json!(["Is it in PHOTO?", "PHOTO is image 3."])
        );
        let images = sent["images"].as_array().unwrap();
        assert_eq!(images.len(), 3);
        assert!(images.iter().all(|image| {
            image
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        }));
        assert_eq!(
            images[0],
            format!(
                "data:image/jpeg;base64,{}",
                picture(386, 391)["data_base64"].as_str().unwrap()
            )
        );
        // 13 × 13 patches, 2 × 1 and 2 × 2, at one and a half tokens each, rounded up a picture.
        assert_eq!(tokens, 254 + 3 + 6);
    }

    #[test]
    fn a_request_without_pictures_is_left_alone() {
        let text = json!({"model": "d1.2", "state": "A message.", "questions": {}});
        assert_eq!(request(text.clone()).unwrap(), (text, 0));
    }

    #[test]
    fn more_pictures_than_a_request_takes_are_refused() {
        let assets: Map<_, _> = (0..9).map(|n| (format!("p{n}"), picture(32, 32))).collect();
        let error =
            request(json!({"model": "d1.2", "state": {}, "assets": assets, "questions": {}}));
        assert!(
            error
                .unwrap_err()
                .to_string()
                .contains("at most 8 pictures")
        );
    }
}
