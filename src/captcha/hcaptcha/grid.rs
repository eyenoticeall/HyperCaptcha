//! Reads hCaptcha's grid task out of a screenshot.
//!
//! hCaptcha draws its nine pictures on a canvas, so the page has no element per picture to ask
//! for a position. The pictures sit on a flat panel with a gap between them, and are found
//! here as the block they form on it. They are then set side by side as the 3 × 3 grid the
//! model is shown.

use std::{collections::HashMap, io::Cursor};

use anyhow::{Context, Result, bail, ensure};
use image::{ImageFormat, RgbImage, imageops};

/// A picture's place in the screenshot, in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

impl Place {
    /// Left, top, width and height as fractions of the screenshot.
    pub fn fractions(&self, width: u32, height: u32) -> [f64; 4] {
        let (w, h) = (f64::from(width), f64::from(height));
        [
            f64::from(self.left) / w,
            f64::from(self.top) / h,
            f64::from(self.width) / w,
            f64::from(self.height) / h,
        ]
    }

    /// The place given as fractions of a screenshot of that size.
    pub fn from_fractions(place: [f64; 4], width: u32, height: u32) -> Result<Self> {
        let [left, top, w, h] = place;
        ensure!(
            place.iter().all(|value| value.is_finite())
                && left >= 0.0
                && top >= 0.0
                && w > 0.0
                && h > 0.0
                && left + w <= 1.0 + 1e-9
                && top + h <= 1.0 + 1e-9,
            "A picture's place must lie inside the screenshot."
        );
        let scale = |value: f64, side: u32| (value * f64::from(side)).round() as u32;
        let (left, top) = (scale(left, width), scale(top, height));
        Ok(Self {
            left: left.min(width - 1),
            top: top.min(height - 1),
            width: scale(w, width).clamp(1, width - left.min(width - 1)),
            height: scale(h, height).clamp(1, height - top.min(height - 1)),
        })
    }
}

/// How far a channel may be from the panel's colour and still count as panel. hCaptcha lays
/// faint noise over the canvas.
const PANEL_TOLERANCE: i16 = 12;
/// The same for the flat colours along the screenshot's edge, which carry no noise.
const EDGE_TOLERANCE: i16 = 6;
/// Panel colours to try, most common first: the pictures' panel is not always the largest
/// flat area of a screenshot.
const PANEL_CANDIDATES: usize = 4;
/// A line of pixels crosses the pictures when at least this share of it is not flat. Three
/// pictures fill most of a line, so one or two pale ones still leave it well above this.
const PICTURE_SHARE: f64 = 0.15;
/// The least share of flat pixels along each of the four gaps between the pictures.
const GAP_SHARE: f64 = 0.45;
/// How much flatter the gaps must be than lines through the middle of the pictures. A single
/// large picture, as in hCaptcha's other tasks, is no flatter along one line than another.
const GAP_OVER_MIDDLE: f64 = 0.2;

/// Finds the nine pictures, in reading order.
///
/// hCaptcha nudges and tilts each picture a little, so the gaps between them are not straight
/// lines to cut along. The nine are found as one block instead: the stretch of columns, and
/// then of rows, that pictures cross. The block is divided into thirds each way, and counts as
/// a grid only if the lines between the thirds are mostly flat panel. Each place therefore
/// holds its picture and up to half of the gap around it.
pub fn find_pictures(screenshot: &RgbImage) -> Result<[Place; 9]> {
    let (width, height) = screenshot.dimensions();
    ensure!(
        width >= 150 && height >= 150,
        "The screenshot is too small to hold nine pictures."
    );
    let edge = edge_colours(screenshot);
    for panel in panel_colours(screenshot) {
        let flat = flat_pixels(screenshot, panel, &edge);
        if let Some(places) = block_of_nine(&flat, width, height) {
            return Ok(places);
        }
    }
    bail!("No 3 × 3 grid of pictures was found in the screenshot.")
}

fn block_of_nine(flat: &[bool], width: u32, height: u32) -> Option<[Place; 9]> {
    // The share of flat pixels in the rectangle from (left, top) to (right, bottom).
    let flat_share = |left: u32, top: u32, right: u32, bottom: u32| {
        let (right, bottom) = (right.min(width), bottom.min(height));
        if left >= right || top >= bottom {
            return 0.0;
        }
        let count: usize = (top..bottom)
            .map(|y| {
                let row = (y * width) as usize;
                flat[row + left as usize..row + right as usize]
                    .iter()
                    .filter(|&&flat| flat)
                    .count()
            })
            .sum();
        count as f64 / f64::from((right - left) * (bottom - top))
    };
    // Runs shorter than a small picture are outlines and noise. Runs a gap apart are joined.
    let (least, join) = (width * 4 / 100, width * 5 / 100);
    let columns: Vec<bool> = (0..width)
        .map(|x| 1.0 - flat_share(x, 0, x + 1, height) >= PICTURE_SHARE)
        .collect();
    let (left, right) = longest_stretch(&columns, least, join)?;
    let rows: Vec<bool> = (0..height)
        .map(|y| 1.0 - flat_share(left, y, right, y + 1) >= PICTURE_SHARE)
        .collect();
    let (top, bottom) = longest_stretch(&rows, least, join)?;
    let (wide, tall) = (right - left, bottom - top);
    // Nine square pictures make a square block, and it is most of what the screenshot shows.
    if wide * 100 < tall * 80 || tall * 100 < wide * 80 {
        return None;
    }
    if wide.min(tall) * 100 < width.min(height) * 45 {
        return None;
    }
    // A gap may lie a little to either side of the line between two thirds.
    let band = (wide * 15 / 1000).max(2);
    let reach = (wide * 4 / 100) as i64;
    let line = |start: u32, length: u32, thirds: u32| start + (length * thirds + 3) / 6;
    let across = |x: i64| {
        flat_share(
            (x - i64::from(band)).max(0) as u32,
            top,
            x as u32 + band,
            bottom,
        )
    };
    let down = |y: i64| {
        flat_share(
            left,
            (y - i64::from(band)).max(0) as u32,
            right,
            y as u32 + band,
        )
    };
    let flattest = |share: &dyn Fn(i64) -> f64, at: u32| {
        (-reach..=reach)
            .map(|offset| share(i64::from(at) + offset))
            .fold(0.0, f64::max)
    };
    let gaps = [
        flattest(&across, line(left, wide, 2)),
        flattest(&across, line(left, wide, 4)),
        flattest(&down, line(top, tall, 2)),
        flattest(&down, line(top, tall, 4)),
    ];
    let middles = [1, 3, 5].map(|sixths| {
        (across(i64::from(line(left, wide, sixths))) + down(i64::from(line(top, tall, sixths))))
            / 2.0
    });
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    if gaps.iter().any(|&gap| gap < GAP_SHARE) || mean(&gaps) < mean(&middles) + GAP_OVER_MIDDLE {
        return None;
    }
    let third = |start: u32, length: u32, index: u32| {
        let from = start + length * index / 3;
        (from, start + length * (index + 1) / 3 - from)
    };
    Some(std::array::from_fn(|index| {
        let (x, w) = third(left, wide, index as u32 % 3);
        let (y, h) = third(top, tall, index as u32 / 3);
        Place {
            left: x,
            top: y,
            width: w,
            height: h,
        }
    }))
}

/// The longest stretch of set runs, each at least `least` long, with at most `join` unset
/// between one run and the next.
fn longest_stretch(set: &[bool], least: u32, join: u32) -> Option<(u32, u32)> {
    let mut stretches: Vec<(u32, u32)> = Vec::new();
    let mut start = None;
    for (index, &flag) in set.iter().chain([&false]).enumerate() {
        let index = index as u32;
        match (flag, start) {
            (true, None) => start = Some(index),
            (false, Some(from)) => {
                start = None;
                if index - from < least {
                    continue;
                }
                match stretches.last_mut() {
                    Some(last) if from - last.1 <= join => last.1 = index,
                    _ => stretches.push((from, index)),
                }
            }
            _ => {}
        }
    }
    stretches.into_iter().max_by_key(|&(from, to)| to - from)
}

fn near(pixel: [u8; 3], colour: [u8; 3], tolerance: i16) -> bool {
    pixel
        .iter()
        .zip(colour)
        .all(|(&found, expected)| (i16::from(found) - i16::from(expected)).abs() <= tolerance)
}

fn rounded(pixel: [u8; 3]) -> [u8; 3] {
    pixel.map(|channel| (channel & !3) + 2)
}

/// The most common near-grey colours, as the panel behind the pictures is white or grey.
fn panel_colours(image: &RgbImage) -> Vec<[u8; 3]> {
    let mut counts = HashMap::<[u8; 3], u32>::new();
    for pixel in image.pixels() {
        let [r, g, b] = pixel.0;
        let (low, high) = (r.min(g).min(b), r.max(g).max(b));
        if high - low <= 8 && low >= 150 {
            *counts.entry(rounded(pixel.0)).or_default() += 1;
        }
    }
    let mut colours: Vec<_> = counts.into_iter().collect();
    colours.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    // Neighbouring shades of one panel would otherwise fill the list.
    let mut chosen: Vec<[u8; 3]> = Vec::new();
    for (colour, _) in colours {
        if !chosen
            .iter()
            .any(|&other| near(other, colour, PANEL_TOLERANCE))
        {
            chosen.push(colour);
            if chosen.len() == PANEL_CANDIDATES {
                break;
            }
        }
    }
    chosen
}

/// The colours that make up much of the screenshot's outer margin: whatever frames the panel.
/// Without them a frame above the top row would read as part of its pictures.
fn edge_colours(image: &RgbImage) -> Vec<[u8; 3]> {
    let (width, height) = image.dimensions();
    let margin = (width.min(height) / 20).max(2);
    let mut counts = HashMap::<[u8; 3], u32>::new();
    let mut total = 0;
    for y in 0..height {
        for x in 0..width {
            if x < margin || y < margin || x >= width - margin || y >= height - margin {
                *counts.entry(rounded(image.get_pixel(x, y).0)).or_default() += 1;
                total += 1;
            }
        }
    }
    counts
        .into_iter()
        .filter(|&(_, count)| count * 100 >= total * 15)
        .map(|(colour, _)| colour)
        .collect()
}

fn flat_pixels(image: &RgbImage, panel: [u8; 3], edge: &[[u8; 3]]) -> Vec<bool> {
    image
        .pixels()
        .map(|pixel| {
            near(pixel.0, panel, PANEL_TOLERANCE)
                || edge
                    .iter()
                    .any(|&colour| near(pixel.0, colour, EDGE_TOLERANCE))
        })
        .collect()
}

/// The side, in pixels, of each picture in the grid the model is shown. hCaptcha's pictures
/// are about 210 pixels on a canvas drawn at twice the page's size.
const LARGEST_PICTURE: u32 = 256;

/// Sets the nine pictures side by side as one square PNG, each the same size.
pub fn compose(screenshot: &RgbImage, places: &[Place; 9]) -> Result<Vec<u8>> {
    let mut sides: Vec<u32> = places
        .iter()
        .map(|place| place.width.min(place.height))
        .collect();
    sides.sort_unstable();
    let side = sides[4].clamp(64, LARGEST_PICTURE);
    let mut grid = RgbImage::new(side * 3, side * 3);
    for (index, place) in places.iter().enumerate() {
        ensure!(
            place.width > 0
                && place.height > 0
                && place.left + place.width <= screenshot.width()
                && place.top + place.height <= screenshot.height(),
            "A picture's place must lie inside the screenshot."
        );
        let picture =
            imageops::crop_imm(screenshot, place.left, place.top, place.width, place.height)
                .to_image();
        let picture = if picture.dimensions() == (side, side) {
            picture
        } else {
            imageops::resize(&picture, side, side, imageops::FilterType::Lanczos3)
        };
        let (column, row) = (index as u32 % 3, index as u32 / 3);
        imageops::replace(
            &mut grid,
            &picture,
            i64::from(column * side),
            i64::from(row * side),
        );
    }
    let mut png = Cursor::new(Vec::new());
    grid.write_to(&mut png, ImageFormat::Png)
        .context("encode the grid of pictures")?;
    Ok(png.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    /// A panel with nine textured squares, each nudged a little as hCaptcha's are.
    fn challenge(panel: [u8; 3], pale: &[usize]) -> (RgbImage, Vec<Place>) {
        let mut image = RgbImage::from_pixel(1000, 760, Rgb([235, 235, 235]));
        for y in 30..730 {
            for x in 20..980 {
                // Faint noise, within what hCaptcha overlays.
                let noise = ((x * 7 + y * 13) % 9) as i16 - 4;
                image.put_pixel(
                    x,
                    y,
                    Rgb(panel.map(|channel| (i16::from(channel) + noise).clamp(0, 255) as u8)),
                );
            }
        }
        let mut places = Vec::new();
        for index in 0..9u32 {
            let (column, row) = (index % 3, index / 3);
            let nudge = (index * 5) % 7;
            let place = Place {
                left: 180 + column * 226 + nudge,
                top: 40 + row * 222 + (6 - nudge),
                width: 210,
                height: 208,
            };
            for y in place.top..place.top + place.height {
                for x in place.left..place.left + place.width {
                    let colour = if pale.contains(&(index as usize)) {
                        // Nearly the panel's own colour: this picture does not stand out.
                        panel.map(|channel| channel.saturating_add(((x + y) % 5) as u8))
                    } else {
                        [
                            (40 + index * 20 + (x * 3) % 60) as u8,
                            (90 + (y * 2) % 80) as u8,
                            (30 + (x + y) % 120) as u8,
                        ]
                    };
                    image.put_pixel(x, y, Rgb(colour));
                }
            }
            places.push(place);
        }
        (image, places)
    }

    /// Each place found must hold its picture: all but a sliver of it, and little else.
    fn assert_holds(found: &[Place; 9], pictures: &[Place]) {
        for (index, (found, picture)) in found.iter().zip(pictures).enumerate() {
            let overlap = |a: u32, a_len: u32, b: u32, b_len: u32| {
                (a + a_len).min(b + b_len).saturating_sub(a.max(b))
            };
            let shared = overlap(found.left, found.width, picture.left, picture.width)
                * overlap(found.top, found.height, picture.top, picture.height);
            assert!(
                shared * 100 >= picture.width * picture.height * 95,
                "picture {} is cut: {found:?} for {picture:?}",
                index + 1
            );
            assert!(
                found.width * found.height * 100 <= picture.width * picture.height * 125,
                "picture {} has too much around it: {found:?} for {picture:?}",
                index + 1
            );
        }
    }

    #[test]
    fn finds_nine_nudged_pictures_on_a_grey_or_white_panel() {
        for panel in [[216, 216, 216], [255, 255, 255]] {
            let (image, expected) = challenge(panel, &[]);
            let found = find_pictures(&image).unwrap();
            assert_holds(&found, &expected);
        }
    }

    #[test]
    fn places_a_picture_that_does_not_stand_out_from_the_panel() {
        let (image, expected) = challenge([255, 255, 255], &[4, 8]);
        let found = find_pictures(&image).unwrap();
        // A pale picture shares its column and its row with pictures that do stand out.
        assert_holds(&found, &expected);
    }

    #[test]
    fn refuses_a_screenshot_without_a_grid() {
        // One large picture, as in hCaptcha's drag and click tasks.
        let mut image = RgbImage::from_pixel(1000, 720, Rgb([216, 216, 216]));
        for y in 40..680 {
            for x in 60..940 {
                image.put_pixel(x, y, Rgb([(x % 200) as u8, (y % 200) as u8, 90]));
            }
        }
        assert!(find_pictures(&image).is_err());
        // The same when the picture is square, as a block of nine would be: no gaps cross it.
        let mut image = RgbImage::from_pixel(1000, 720, Rgb([216, 216, 216]));
        for y in 40..680 {
            for x in 180..820 {
                image.put_pixel(x, y, Rgb([(x % 200) as u8, (y % 200) as u8, 90]));
            }
        }
        assert!(find_pictures(&image).is_err());
        // A whole row that does not stand out leaves two rows, which is not a grid.
        let (image, _) = challenge([216, 216, 216], &[0, 1, 2]);
        assert!(find_pictures(&image).is_err());
    }

    #[test]
    fn composes_the_pictures_without_their_gaps() {
        let (image, places) = challenge([216, 216, 216], &[]);
        let places: [Place; 9] = places.try_into().unwrap();
        let png = compose(&image, &places).unwrap();
        let grid = image::load_from_memory(&png).unwrap().into_rgb8();
        assert_eq!(grid.dimensions(), (624, 624));
        // No panel grey is left between or around the pictures.
        let grey = grid
            .pixels()
            .filter(|pixel| pixel.0.iter().all(|&channel| channel.abs_diff(216) <= 12))
            .count();
        assert!(grey < 2000, "{grey} panel pixels in the composed grid");
        // The middle of each cell is the middle of its picture.
        for index in 0..9u32 {
            let place = places[index as usize];
            let from = image.get_pixel(place.left + place.width / 2, place.top + place.height / 2);
            let to = grid.get_pixel(index % 3 * 208 + 104, index / 3 * 208 + 104);
            for channel in 0..3 {
                assert!(from.0[channel].abs_diff(to.0[channel]) <= 40);
            }
        }
    }

    #[test]
    fn a_place_survives_the_trip_through_fractions() {
        let place = Place {
            left: 180,
            top: 40,
            width: 210,
            height: 208,
        };
        let fractions = place.fractions(1000, 720);
        assert_eq!(Place::from_fractions(fractions, 1000, 720).unwrap(), place);
        assert!(Place::from_fractions([0.9, 0.0, 0.2, 0.1], 1000, 720).is_err());
        assert!(Place::from_fractions([0.1, 0.1, 0.0, 0.1], 1000, 720).is_err());
    }
}
