//! The box rule for 4 × 4 grids.
//!
//! The model is asked to draw boxes around the target. d3-ultra draws nine on the whole grid,
//! one to a question, and at the highest effort level one in each of sixteen windows, a square
//! with its neighbours. Liquid's models are asked once for a box around every instance on the
//! whole grid. A selected square that scored under the rule's exemption is dropped unless a box
//! from every request the round sent reaches it. Boxes only ever remove selections. The
//! thresholds are a [`BoxRule`], set by the round's plan.
//!
//! The requests are the ones the rule was measured with, down to the bytes of each picture and
//! the order of the questions, because d3-ultra's boxes shift when the pictures are encoded
//! differently or the windows are spread over several requests. Change nothing here without
//! measuring the rule again.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::ImageFormat;
use serde::{Serialize, Serializer};
use serde_json::{Value, json};

use crate::{
    pillow::Rgb,
    plan::{BoxAsk, BoxRule},
};

/// Longer side, in pixels, of the whole-grid picture and of each window.
const WHOLE_SIDE: f64 = 778.0;
const WINDOW_SIDE: f64 = 386.0;
const SIDE: u8 = 4;

/// The nine whole-grid questions, in the order they are sent.
const WHOLE_QUESTIONS: [(&str, &str, &str); 9] = [
    (
        "largest",
        "the largest ",
        " in the image, including every visible part of it",
    ),
    ("second", "the second largest ", " in the image"),
    ("third", "the third largest ", " in the image"),
    ("leftmost", "the leftmost ", ""),
    ("rightmost", "the rightmost ", ""),
    ("topmost", "the ", " highest up in the image"),
    ("bottommost", "the ", " lowest down in the image"),
    ("nearest", "the ", " nearest to the camera"),
    ("farthest", "the ", " farthest from the camera"),
];
const WINDOW_QUESTION: &str = "q";
const EVERY_QUESTION: &str = "q";
/// The most boxes the every-instance question may draw: one a square.
const MOST_INSTANCES: u8 = SIDE * SIDE;

/// Left, top, right, bottom as fractions of the grid.
type Rect = [f64; 4];

fn tile_rect(tile: u8) -> Rect {
    let (row, column) = ((tile - 1) / SIDE, (tile - 1) % SIDE);
    let side = f64::from(SIDE);
    [
        f64::from(column) / side,
        f64::from(row) / side,
        f64::from(column + 1) / side,
        f64::from(row + 1) / side,
    ]
}

/// The square and the squares around it.
fn window(tile: u8) -> Rect {
    let [left, top, right, bottom] = tile_rect(tile);
    [
        (left - 0.25).max(0.0),
        (top - 0.25).max(0.0),
        (right + 0.25).min(1.0),
        (bottom + 0.25).min(1.0),
    ]
}

/// Where the square sits in its window, as the question words it.
fn place_in_window(tile: u8) -> &'static str {
    let (square, window) = (tile_rect(tile), window(tile));
    // A square touching one side of its window is on that side. One touching both or neither
    // is in the middle.
    let side = |low: usize, high: usize| {
        let (at_low, at_high) = (square[low] == window[low], square[high] == window[high]);
        if at_low && !at_high {
            0
        } else if at_high && !at_low {
            2
        } else {
            1
        }
    };
    [
        ["top left", "top", "top right"],
        ["left", "centre", "right"],
        ["bottom left", "bottom", "bottom right"],
    ][side(1, 3)][side(0, 2)]
}

/// The target as one object: "motorcycles" becomes "motorcycle".
fn singular(target: &str) -> String {
    match target {
        "buses" => "bus".into(),
        "stairs" => "set of stairs".into(),
        _ => target.strip_suffix('s').unwrap_or(target).into(),
    }
}

/// The part of the grid inside `window`, enlarged so its longer side is `longer_side` pixels.
fn picture(grid: &Rgb, window: Rect, longer_side: f64) -> Rgb {
    let edge = |fraction: f64, length: usize| (fraction * length as f64).round_ties_even() as usize;
    let part = grid.crop(
        edge(window[0], grid.width),
        edge(window[1], grid.height),
        edge(window[2], grid.width),
        edge(window[3], grid.height),
    );
    let scale = longer_side / part.width.max(part.height) as f64;
    let size = |length: usize| (length as f64 * scale).round_ties_even() as usize;
    part.resize_bilinear(size(part.width), size(part.height))
}

fn data_url(picture: &Rgb) -> String {
    format!(
        "data:image/jpeg;base64,{}",
        STANDARD.encode(picture.jpeg_quality_90())
    )
}

pub fn decode_grid(png: &[u8]) -> Result<Rgb> {
    let image = image::load_from_memory_with_format(png, ImageFormat::Png)
        .context("decode the grid for the box requests")?
        .to_rgb8();
    Ok(Rgb {
        width: image.width() as usize,
        height: image.height() as usize,
        data: image.into_raw(),
    })
}

/// A JSON object whose keys keep the order given. serde_json's own maps sort them.
struct Ordered<K, V>(Vec<(K, V)>);

impl<K: Serialize, V: Serialize> Serialize for Ordered<K, V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(key, value)| (key, value)))
    }
}

#[derive(Serialize)]
struct Question {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: String,
    /// The most boxes to draw, for the question that asks for every instance. Left out of
    /// every other question, whose bytes the box rule was measured with.
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<u8>,
}

#[derive(Serialize)]
struct State {
    state: String,
    images: [String; 1],
    questions: Ordered<&'static str, Question>,
}

#[derive(Serialize)]
struct Options {
    return_meta: bool,
}

#[derive(Serialize)]
struct WholeRequest<'a> {
    #[serde(flatten)]
    state: State,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Options>,
}

#[derive(Serialize)]
struct WindowsRequest<'a> {
    states: Ordered<String, State>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Options>,
}

/// An encoded box request and the fewest input tokens its pictures cost.
pub struct BoxPayload {
    pub json: Vec<u8>,
    pub patch_tokens: u64,
}

fn request_options(return_meta: bool) -> Option<Options> {
    return_meta.then_some(Options { return_meta: true })
}

fn picture_patch_tokens(pic: &Rgb) -> u64 {
    crate::liquid::patch_tokens(pic.width as u32, pic.height as u32)
}

fn ask_for_box(instructions: String) -> Question {
    Question {
        kind: "box",
        instructions,
        max: None,
    }
}

/// The request for nine boxes on the whole grid, as JSON.
pub fn whole_request(
    model: &str,
    target: &str,
    grid: &Rgb,
    return_meta: bool,
) -> Result<BoxPayload> {
    let one = singular(target);
    let pic = picture(grid, [0.0, 0.0, 1.0, 1.0], WHOLE_SIDE);
    let patch_tokens = picture_patch_tokens(&pic);
    let request = WholeRequest {
        state: State {
            state: format!("A street photo. Locate the {target}."),
            images: [data_url(&pic)],
            questions: Ordered(
                WHOLE_QUESTIONS
                    .iter()
                    .map(|(name, before, after)| {
                        (*name, ask_for_box(format!("{before}{one}{after}")))
                    })
                    .collect(),
            ),
        },
        model,
        options: request_options(return_meta),
    };
    let json = serde_json::to_vec(&request).context("encode the whole-grid box request")?;
    Ok(BoxPayload { json, patch_tokens })
}

/// The request for a box around every instance on the whole grid, as JSON: the nine-box
/// request's picture and state with one question. Only Liquid's models answer its type.
pub fn every_request(
    model: &str,
    target: &str,
    grid: &Rgb,
    return_meta: bool,
) -> Result<BoxPayload> {
    let pic = picture(grid, [0.0, 0.0, 1.0, 1.0], WHOLE_SIDE);
    let patch_tokens = picture_patch_tokens(&pic);
    let request = WholeRequest {
        state: State {
            state: format!("A street photo. Locate the {target}."),
            images: [data_url(&pic)],
            questions: Ordered(vec![(
                EVERY_QUESTION,
                Question {
                    kind: "boxes",
                    instructions: format!("every {}", singular(target)),
                    max: Some(MOST_INSTANCES),
                },
            )]),
        },
        model,
        options: request_options(return_meta),
    };
    let json = serde_json::to_vec(&request).context("encode the every-instance box request")?;
    Ok(BoxPayload { json, patch_tokens })
}

/// How many windows a grid has: one around each square.
pub const WINDOWS: u8 = SIDE * SIDE;

/// Each window's name, picture, question, and patch-token floor.
fn window_states(target: &str, grid: &Rgb) -> Vec<(String, State, u64)> {
    let one = singular(target);
    (1..=WINDOWS)
        .map(|tile| {
            let place = place_in_window(tile);
            let pic = picture(grid, window(tile), WINDOW_SIDE);
            let patch_tokens = picture_patch_tokens(&pic);
            let state = State {
                state: format!("Part of a street photo. Target: {target}."),
                images: [data_url(&pic)],
                questions: Ordered(vec![(
                    WINDOW_QUESTION,
                    ask_for_box(format!(
                        "the {one} in the {place} of the image, including every visible part of it"
                    )),
                )]),
            };
            (format!("s{tile:02}"), state, patch_tokens)
        })
        .collect()
}

/// The request for one box in each of the sixteen windows, as JSON. The windows travel
/// together: d3-ultra answers a window differently when it shares a request with fewer others.
pub fn windows_request(
    model: &str,
    target: &str,
    grid: &Rgb,
    return_meta: bool,
) -> Result<BoxPayload> {
    let mut patch_tokens = 0u64;
    let states = window_states(target, grid)
        .into_iter()
        .map(|(name, state, tokens)| {
            patch_tokens += tokens;
            (name, state)
        })
        .collect();
    let request = WindowsRequest {
        states: Ordered(states),
        model,
        options: request_options(return_meta),
    };
    let json = serde_json::to_vec(&request).context("encode the window box request")?;
    Ok(BoxPayload { json, patch_tokens })
}

/// The same sixteen windows as a request each, with the window's name, for a provider whose
/// requests hold one state.
pub fn window_requests(
    model: &str,
    target: &str,
    grid: &Rgb,
    return_meta: bool,
) -> Result<Vec<(String, BoxPayload)>> {
    window_states(target, grid)
        .into_iter()
        .map(|(name, state, patch_tokens)| {
            let request = WholeRequest {
                state,
                model,
                options: request_options(return_meta),
            };
            let json = serde_json::to_vec(&request).context("encode a window's box request")?;
            Ok((name, BoxPayload { json, patch_tokens }))
        })
        .collect()
}

/// One box from a reply, in fractions of the whole grid.
#[derive(Clone, Serialize)]
pub struct Found {
    pub name: String,
    pub rect: Rect,
    pub confidence: f64,
}

/// A box answer's rectangle in fractions of its picture, and its confidence. An answer with
/// no rectangle draws no box.
fn answer_box(answer: &Value) -> Result<Option<(Rect, f64)>> {
    ensure!(answer.is_object(), "a box answer is not an object");
    let rect = &answer["box"];
    if rect.is_null() {
        return Ok(None);
    }
    let edge = |name: &str| {
        rect[name]
            .as_f64()
            .with_context(|| format!("a box has no {name} edge"))
    };
    Ok(Some((
        [edge("left")?, edge("top")?, edge("right")?, edge("bottom")?],
        answer["confidence"].as_f64().unwrap_or(0.0),
    )))
}

/// The boxes in the whole-grid reply. Fails unless all nine questions were answered.
pub fn whole_boxes(reply: &Value) -> Result<Vec<Found>> {
    let mut found = Vec::new();
    for (name, ..) in WHOLE_QUESTIONS {
        let answer = &reply["answers"][name];
        ensure!(!answer.is_null(), "the whole-grid reply has no {name} box");
        if let Some((rect, confidence)) = answer_box(answer)? {
            found.push(Found {
                name: name.into(),
                rect,
                confidence,
            });
        }
    }
    Ok(found)
}

/// The boxes in the reply to the every-instance question, named in the order drawn. A reply
/// that draws none is complete: the model found no instance.
pub fn every_boxes(reply: &Value) -> Result<Vec<Found>> {
    let drawn = reply["answers"][EVERY_QUESTION]["boxes"]
        .as_array()
        .context("the every-instance reply has no list of boxes")?;
    let mut found = Vec::new();
    for (place, answer) in drawn.iter().enumerate() {
        if let Some((rect, confidence)) = answer_box(answer)? {
            found.push(Found {
                name: format!("e{:02}", place + 1),
                rect,
                confidence,
            });
        }
    }
    Ok(found)
}

/// The boxes in one box request's reply.
pub fn reply_boxes(ask: BoxAsk, reply: &Value) -> Result<Vec<Found>> {
    match ask {
        BoxAsk::Whole => whole_boxes(reply),
        BoxAsk::Every => every_boxes(reply),
        BoxAsk::Windows => window_boxes(reply),
    }
}

/// The boxes in the window reply, moved from each window onto the grid. Fails unless all
/// sixteen windows were answered.
pub fn window_boxes(reply: &Value) -> Result<Vec<Found>> {
    let mut found = Vec::new();
    for tile in 1..=SIDE * SIDE {
        let name = format!("s{tile:02}");
        let answer = &reply["states"][&name]["answers"][WINDOW_QUESTION];
        ensure!(!answer.is_null(), "the window reply has no box for {name}");
        if let Some((rect, confidence)) = answer_box(answer)? {
            let window = window(tile);
            let (width, height) = (window[2] - window[0], window[3] - window[1]);
            found.push(Found {
                name,
                rect: [
                    window[0] + rect[0] * width,
                    window[1] + rect[1] * height,
                    window[0] + rect[2] * width,
                    window[1] + rect[3] * height,
                ],
                confidence,
            });
        }
    }
    Ok(found)
}

/// Whether a box of at least `confidence` reaches the square: it covers enough of the square,
/// and enough of the box lies inside it.
fn reached(boxes: &[Found], tile: u8, confidence: f64, rule: &BoxRule) -> bool {
    let square = tile_rect(tile);
    boxes
        .iter()
        .filter(|found| found.confidence >= confidence)
        .any(|found| {
            let width = found.rect[2].min(square[2]) - found.rect[0].max(square[0]);
            let height = found.rect[3].min(square[3]) - found.rect[1].max(square[1]);
            let shared = width.max(0.0) * height.max(0.0);
            let area = (found.rect[2] - found.rect[0]) * (found.rect[3] - found.rect[1]);
            // The square is a sixteenth of the grid.
            shared * 16.0 >= rule.cover && area > 0.0 && shared / area >= rule.inside
        })
}

/// The selections the rule could drop: those scoring under its exemption.
pub fn at_risk(picks: &[u8], scores: &BTreeMap<u8, f64>, rule: &BoxRule) -> BTreeSet<u8> {
    picks
        .iter()
        .copied()
        .filter(|tile| {
            scores
                .get(tile)
                .is_some_and(|&score| score < rule.exempt_from)
        })
        .collect()
}

/// Decides which at-risk selections stay, one box reply at a time.
///
/// A selection stays once a box from every request has reached it, and is dropped as soon as
/// one complete reply holds no box that does. Which reply comes first makes no difference to
/// the answer, so a round can stop listening when nothing is left undecided.
pub struct Vote {
    rule: BoxRule,
    /// Each undecided selection, with the replies that have reached it so far.
    undecided: BTreeMap<u8, u8>,
    dropped: BTreeSet<u8>,
}

impl Vote {
    pub fn new(rule: BoxRule, at_risk: &BTreeSet<u8>) -> Self {
        Self {
            rule,
            undecided: at_risk.iter().map(|&tile| (tile, 0)).collect(),
            dropped: BTreeSet::new(),
        }
    }

    /// Takes one request's complete reply.
    pub fn hear(&mut self, ask: BoxAsk, boxes: &[Found]) {
        let confidence = self.rule.confidence(ask).unwrap_or(f64::INFINITY);
        let needed = self.rule.requests();
        let (rule, dropped) = (&self.rule, &mut self.dropped);
        self.undecided.retain(|&tile, reached_by| {
            if !reached(boxes, tile, confidence, rule) {
                dropped.insert(tile);
                return false;
            }
            *reached_by += 1;
            *reached_by < needed
        });
    }

    /// Whether every selection has been decided, so no further reply can change the answer.
    pub fn settled(&self) -> bool {
        self.undecided.is_empty()
    }

    /// The selections to drop. Those still undecided, because a request failed, are kept.
    pub fn dropped(self) -> BTreeSet<u8> {
        self.dropped
    }
}

/// What the round's log keeps about one box reply. Where the model also says how likely the
/// object is to be in the picture at all, that is kept for every question, drawn or not, and
/// for every box of the every-instance question, with how many instances it counted.
pub fn log_entry(boxes: &[Found], reply: &Value) -> Value {
    let mut entry = json!({"boxes": boxes, "usage": reply["usage"], "meta": reply["meta"]});
    let every = &reply["answers"][EVERY_QUESTION];
    let drawn = every["boxes"].as_array();
    if drawn.is_some() {
        entry["count"] = every["count"].clone();
        entry["at_least"] = every["at_least"].clone();
    }
    let names: Vec<String> = (1..=drawn.map_or(0, Vec::len))
        .map(|place| format!("e{place:02}"))
        .collect();
    let answers: Vec<(&String, &Value)> = match (drawn, reply["states"].as_object()) {
        (Some(drawn), _) => names.iter().zip(drawn).collect(),
        (None, Some(states)) => states
            .iter()
            .map(|(name, state)| (name, &state["answers"][WINDOW_QUESTION]))
            .collect(),
        (None, None) => reply["answers"].as_object().into_iter().flatten().collect(),
    };
    let found: serde_json::Map<String, Value> = answers
        .into_iter()
        .filter_map(|(name, answer)| Some((name.clone(), json!(answer["found"].as_f64()?))))
        .collect();
    if !found.is_empty() {
        entry["found"] = Value::Object(found);
    }
    entry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pillow::fnv1a;

    #[test]
    fn windows_and_wording_match_the_measured_requests() {
        // (window, words) for squares 1 to 16, from the Python that the rule was measured with.
        let expected: [(Rect, &str); 16] = [
            ([0.0, 0.0, 0.5, 0.5], "top left"),
            ([0.0, 0.0, 0.75, 0.5], "top"),
            ([0.25, 0.0, 1.0, 0.5], "top"),
            ([0.5, 0.0, 1.0, 0.5], "top right"),
            ([0.0, 0.0, 0.5, 0.75], "left"),
            ([0.0, 0.0, 0.75, 0.75], "centre"),
            ([0.25, 0.0, 1.0, 0.75], "centre"),
            ([0.5, 0.0, 1.0, 0.75], "right"),
            ([0.0, 0.25, 0.5, 1.0], "left"),
            ([0.0, 0.25, 0.75, 1.0], "centre"),
            ([0.25, 0.25, 1.0, 1.0], "centre"),
            ([0.5, 0.25, 1.0, 1.0], "right"),
            ([0.0, 0.5, 0.5, 1.0], "bottom left"),
            ([0.0, 0.5, 0.75, 1.0], "bottom"),
            ([0.25, 0.5, 1.0, 1.0], "bottom"),
            ([0.5, 0.5, 1.0, 1.0], "bottom right"),
        ];
        for (tile, (rect, words)) in (1..=16).zip(expected) {
            assert_eq!(window(tile), rect, "window of square {tile}");
            assert_eq!(place_in_window(tile), words, "words for square {tile}");
        }
        for (plural, one) in [
            ("motorcycles", "motorcycle"),
            ("buses", "bus"),
            ("stairs", "set of stairs"),
            ("crosswalks", "crosswalk"),
            ("a fire hydrant", "a fire hydrant"),
        ] {
            assert_eq!(singular(plural), one);
        }
    }

    #[test]
    fn requests_keep_the_measured_shape_and_order() {
        let grid = Rgb {
            width: 16,
            height: 16,
            data: vec![90; 16 * 16 * 3],
        };
        let whole_payload = whole_request("m", "buses", &grid, true).unwrap();
        assert_eq!(whole_payload.patch_tokens, 938);
        let whole = String::from_utf8(whole_payload.json).unwrap();
        assert!(whole.starts_with(
            r#"{"state":"A street photo. Locate the buses.","images":["data:image/jpeg;base64,"#
        ));
        let names: Vec<_> = WHOLE_QUESTIONS.iter().map(|(name, ..)| *name).collect();
        let mut from = 0;
        for name in names {
            let at = whole[from..]
                .find(&format!(r#""{name}":{{"type":"box","instructions":"#))
                .unwrap_or_else(|| panic!("{name} is missing or out of order"));
            from += at;
        }
        assert!(whole.contains(r#""largest":{"type":"box","instructions":"the largest bus in the image, including every visible part of it"}"#));
        assert!(whole.contains(
            r#""topmost":{"type":"box","instructions":"the bus highest up in the image"}"#
        ));
        assert!(whole.ends_with(r#""model":"m","options":{"return_meta":true}}"#));

        let whole_no_meta =
            String::from_utf8(whole_request("m", "buses", &grid, false).unwrap().json).unwrap();
        assert!(whole_no_meta.ends_with(r#""model":"m"}"#));
        assert!(!whole_no_meta.contains("return_meta"));

        let windows_payload = windows_request("m", "stairs", &grid, true).unwrap();
        assert!(windows_payload.patch_tokens > 0);
        let windows = String::from_utf8(windows_payload.json).unwrap();
        assert!(windows.starts_with(
            r#"{"states":{"s01":{"state":"Part of a street photo. Target: stairs.","images":["data:image/jpeg;base64,"#
        ));
        assert!(windows.contains(r#""questions":{"q":{"type":"box","instructions":"the set of stairs in the top left of the image, including every visible part of it"}}}"#));
        assert_eq!(windows.matches(r#""state":"Part of"#).count(), 16);
        assert!(windows.find(r#""s02":"#).unwrap() < windows.find(r#""s16":"#).unwrap());
        assert!(windows.ends_with(r#""model":"m","options":{"return_meta":true}}"#));

        let windows_no_meta =
            String::from_utf8(windows_request("m", "stairs", &grid, false).unwrap().json).unwrap();
        assert!(windows_no_meta.ends_with(r#""model":"m"}"#));
        assert!(!windows_no_meta.contains("return_meta"));

        let single_windows = window_requests("m", "stairs", &grid, false).unwrap();
        assert_eq!(single_windows.len(), 16);
        assert_eq!(
            single_windows
                .iter()
                .map(|(_, payload)| payload.patch_tokens)
                .sum::<u64>(),
            windows_payload.patch_tokens
        );
        for (_, payload) in &single_windows {
            let text = String::from_utf8(payload.json.clone()).unwrap();
            assert!(text.ends_with(r#""model":"m"}"#));
            assert!(!text.contains("return_meta"));
            assert!(payload.patch_tokens > 0);
        }
    }

    #[test]
    fn the_every_instance_request_is_one_question_on_the_whole_grid_picture() {
        let grid = Rgb {
            width: 16,
            height: 16,
            data: vec![90; 16 * 16 * 3],
        };
        let whole_payload = whole_request("m", "stairs", &grid, true).unwrap();
        let every_payload = every_request("m", "stairs", &grid, true).unwrap();
        assert_eq!(every_payload.patch_tokens, whole_payload.patch_tokens);
        let whole = String::from_utf8(whole_payload.json).unwrap();
        let every = String::from_utf8(every_payload.json).unwrap();
        // The same state and picture as the nine-box request, then the one question.
        let questions = r#""],"questions":{"#;
        let shared = whole.find(questions).unwrap() + questions.len();
        assert_eq!(every[..shared], whole[..shared]);
        assert_eq!(
            &every[shared..],
            r#""q":{"type":"boxes","instructions":"every set of stairs","max":16}},"model":"m","options":{"return_meta":true}}"#
        );
        let every_no_meta =
            String::from_utf8(every_request("m", "stairs", &grid, false).unwrap().json).unwrap();
        assert_eq!(
            &every_no_meta[shared..],
            r#""q":{"type":"boxes","instructions":"every set of stairs","max":16}},"model":"m"}"#
        );
        // No other question says how many boxes to draw.
        assert!(!whole.contains("max"));
        let windows =
            String::from_utf8(windows_request("m", "stairs", &grid, true).unwrap().json).unwrap();
        assert!(!windows.contains("max"));
    }

    #[test]
    fn an_every_instance_reply_may_draw_any_number_of_boxes_or_none() {
        let drawn = |confidence: f64| json!({"box": {"left": 0.0, "top": 0.25, "right": 0.5, "bottom": 0.5}, "found": 0.7, "confidence": confidence});
        let reply = json!({"usage": {"input_tokens": 1369}, "answers": {"q": {
            "type": "boxes", "boxes": [drawn(0.9), drawn(0.1)], "count": 2, "at_least": false,
        }}});
        let boxes = reply_boxes(BoxAsk::Every, &reply).unwrap();
        assert_eq!(boxes.len(), 2);
        assert_eq!((boxes[1].name.as_str(), boxes[1].confidence), ("e02", 0.1));
        assert_eq!(boxes[0].rect, [0.0, 0.25, 0.5, 0.5]);
        let entry = log_entry(&boxes, &reply);
        assert_eq!(entry["count"], 2);
        assert_eq!(entry["at_least"], false);
        assert_eq!(entry["found"], json!({"e01": 0.7, "e02": 0.7}));
        // No instance found is an answer, and drops every square that needed a box.
        let none = json!({"answers": {"q": {"type": "boxes", "boxes": [], "count": 0}}});
        assert!(every_boxes(&none).unwrap().is_empty());
        let rule = BoxRule {
            exempt_from: 0.9,
            whole: None,
            every: Some(0.3),
            windows: None,
            cover: 0.01,
            inside: 0.0,
        };
        let mut vote = Vote::new(rule, &BTreeSet::from([5, 6]));
        vote.hear(BoxAsk::Every, &[]);
        assert!(vote.settled());
        assert_eq!(vote.dropped(), BTreeSet::from([5, 6]));
        // The faint box does not count, the confident one reaches squares 5 and 6.
        let mut vote = Vote::new(rule, &BTreeSet::from([5, 6, 7]));
        vote.hear(BoxAsk::Every, &boxes);
        assert_eq!(vote.dropped(), BTreeSet::from([7]));
        // A reply without the list is not an answer.
        assert!(every_boxes(&json!({"answers": {"q": {"type": "boxes"}}})).is_err());
        assert!(every_boxes(&json!({"error": {"message": "no"}})).is_err());
    }

    fn found(rect: Rect, confidence: f64) -> Found {
        Found {
            name: "test".into(),
            rect,
            confidence,
        }
    }

    fn rule(whole: f64, windows: Option<f64>, inside: f64) -> BoxRule {
        BoxRule {
            exempt_from: 0.95,
            whole: Some(whole),
            every: None,
            windows,
            cover: 0.01,
            inside,
        }
    }

    #[test]
    fn a_pick_is_dropped_when_no_confident_box_reaches_it() {
        let scores: BTreeMap<u8, f64> = [(1, 0.99), (2, 0.9), (6, 0.6), (7, 0.4)].into();
        // Square 1 scored too high to drop, and square 7 is not a pick.
        let at_risk = at_risk(&[1, 2, 6], &scores, &rule(0.3, None, 0.0));
        assert_eq!(at_risk, BTreeSet::from([2, 6]));
        // A box over a tenth of square 2's width and height covers 1% of it. A faint box over
        // square 6 does not count.
        let boxes = [
            found([0.25, 0.0, 0.275, 0.025], 0.3),
            found([0.25, 0.25, 0.5, 0.5], 0.29),
        ];
        let mut vote = Vote::new(rule(0.3, None, 0.0), &at_risk);
        vote.hear(BoxAsk::Whole, &boxes);
        assert!(vote.settled());
        assert_eq!(vote.dropped(), BTreeSet::from([6]));
        // The same faint box counts at a lower setting.
        let mut vote = Vote::new(rule(0.2, None, 0.0), &at_risk);
        vote.hear(BoxAsk::Whole, &boxes);
        assert!(vote.dropped().is_empty());
        // A box that only touches the square's edge does not reach it.
        let mut vote = Vote::new(rule(0.3, None, 0.0), &BTreeSet::from([6]));
        vote.hear(BoxAsk::Whole, &[found([0.0, 0.0, 0.25, 0.25], 0.9)]);
        assert_eq!(vote.dropped(), BTreeSet::from([6]));
    }

    #[test]
    fn a_box_must_lie_far_enough_inside_the_square() {
        // A box over the whole grid covers all of square 6, with a sixteenth of itself in it.
        // A box over the grid's left half and a sliver of square 6's column has under 1% there.
        let wide = [found([0.0, 0.0, 1.0, 1.0], 0.9)];
        let sliver = [found([0.0, 0.0, 0.254, 1.0], 0.9)];
        for (boxes, inside, kept) in [
            (&wide, 0.05, true),
            (&wide, 0.07, false),
            (&sliver, 0.0, true),
            (&sliver, 0.01, false),
        ] {
            let mut vote = Vote::new(rule(0.3, None, inside), &BTreeSet::from([6]));
            vote.hear(BoxAsk::Whole, boxes);
            assert_eq!(vote.dropped().is_empty(), kept, "inside {inside}");
        }
    }

    #[test]
    fn with_two_requests_a_pick_needs_a_box_from_each_whichever_answers_first() {
        // Whole-grid boxes reach squares 2 and 6, window boxes squares 6 and 7.
        let whole = [
            found([0.25, 0.0, 0.5, 0.25], 0.1),
            found([0.25, 0.25, 0.5, 0.5], 0.1),
        ];
        let windows = [
            found([0.25, 0.25, 0.5, 0.5], 0.6),
            found([0.5, 0.25, 0.75, 0.5], 0.6),
        ];
        let at_risk = BTreeSet::from([2, 6, 7, 11]);
        let mut first = Vote::new(rule(0.02, Some(0.5), 0.0), &at_risk);
        first.hear(BoxAsk::Whole, &whole);
        // Squares 7 and 11 are already gone; 2 and 6 wait for the window boxes.
        assert!(!first.settled());
        first.hear(BoxAsk::Windows, &windows);
        assert!(first.settled());
        let mut second = Vote::new(rule(0.02, Some(0.5), 0.0), &at_risk);
        second.hear(BoxAsk::Windows, &windows);
        assert!(!second.settled());
        second.hear(BoxAsk::Whole, &whole);
        assert_eq!(first.dropped(), BTreeSet::from([2, 7, 11]));
        assert_eq!(second.dropped(), BTreeSet::from([2, 7, 11]));

        // Each request has its own confidence: this window box is too faint to count.
        let mut vote = Vote::new(rule(0.02, Some(0.7), 0.0), &BTreeSet::from([6]));
        vote.hear(BoxAsk::Whole, &whole);
        vote.hear(BoxAsk::Windows, &windows);
        assert_eq!(vote.dropped(), BTreeSet::from([6]));
        // When the other request fails, a square the first reply reached stays undecided,
        // and so selected, while one it did not reach is dropped all the same.
        let mut vote = Vote::new(rule(0.02, Some(0.5), 0.0), &BTreeSet::from([6, 11]));
        vote.hear(BoxAsk::Whole, &whole);
        assert!(!vote.settled());
        assert_eq!(vote.dropped(), BTreeSet::from([11]));
    }

    #[test]
    fn replies_are_read_onto_the_grid_and_must_be_complete() {
        let answer = |confidence: f64| json!({"type": "box", "confidence": confidence, "box": {"left": 0.5, "top": 0.0, "right": 1.0, "bottom": 0.5}});
        let mut whole = json!({"answers": {}});
        for (name, ..) in WHOLE_QUESTIONS {
            whole["answers"][name] = answer(0.4);
        }
        assert_eq!(whole_boxes(&whole).unwrap().len(), 9);
        whole["answers"]["third"] = json!({"type": "box", "box": null, "confidence": 0.0});
        assert_eq!(whole_boxes(&whole).unwrap().len(), 8);
        whole["answers"].as_object_mut().unwrap().remove("nearest");
        assert!(whole_boxes(&whole).is_err());

        let mut windows = json!({"states": {}});
        for tile in 1..=16 {
            windows["states"][format!("s{tile:02}")] = json!({"answers": {"q": answer(0.8)}});
        }
        let boxes = window_boxes(&windows).unwrap();
        // Square 16's window is the bottom-right quarter, so the top-right quarter of it is
        // squares 12's right half and so on.
        assert_eq!(boxes[15].rect, [0.75, 0.5, 1.0, 0.75]);
        // Square 6's window is the top-left nine squares.
        assert_eq!(boxes[5].rect, [0.375, 0.0, 0.75, 0.375]);
        windows["states"].as_object_mut().unwrap().remove("s09");
        assert!(window_boxes(&windows).is_err());
    }

    /// Checks every picture against the lengths and hashes that Pillow produced for the same
    /// grids. Needs a reference JSON file, so it runs only when asked:
    /// `HYPERCAPTCHA_BOX_REFERENCE=path/to/box-reference.json cargo test -- --ignored`
    #[test]
    #[ignore]
    fn pictures_match_pillow_byte_for_byte() {
        let path =
            std::env::var("HYPERCAPTCHA_BOX_REFERENCE").expect("set HYPERCAPTCHA_BOX_REFERENCE");
        let reference: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let (mut pictures, mut wrong_pixels, mut wrong_jpegs) = (0, Vec::new(), Vec::new());
        for (id, grid) in reference["grids"].as_object().unwrap() {
            let png = std::fs::read(grid["image"].as_str().unwrap()).unwrap();
            let image = decode_grid(&png).unwrap();
            let mut views = vec![("ord".to_string(), [0.0, 0.0, 1.0, 1.0], WHOLE_SIDE)];
            views.extend((1..=16).map(|tile| (format!("s{tile:02}"), window(tile), WINDOW_SIDE)));
            for (name, rect, side) in views {
                let expected = &grid[&name];
                let made = picture(&image, rect, side);
                let jpeg = made.jpeg_quality_90();
                pictures += 1;
                if json!([made.width, made.height]) != expected["size"]
                    || format!("{:016x}", fnv1a(&made.data)) != expected["rgb"].as_str().unwrap()
                {
                    wrong_pixels.push(format!("{id}/{name}"));
                } else if json!(jpeg.len()) != expected["jpeg_len"]
                    || format!("{:016x}", fnv1a(&jpeg)) != expected["jpeg"].as_str().unwrap()
                {
                    wrong_jpegs.push(format!("{id}/{name}"));
                }
            }
        }
        println!(
            "{pictures} pictures: {} with different pixels, {} with a different JPEG",
            wrong_pixels.len(),
            wrong_jpegs.len()
        );
        assert!(
            wrong_pixels.is_empty(),
            "pixels differ: {:?}",
            &wrong_pixels[..wrong_pixels.len().min(8)]
        );
        assert!(
            wrong_jpegs.is_empty(),
            "JPEGs differ: {:?}",
            &wrong_jpegs[..wrong_jpegs.len().min(8)]
        );
    }
}
