//! How one 4 × 4 round is asked and decided.
//!
//! A plan names the requests a round sends beside the whole-grid request, and the rules that
//! turn their answers into a selection. Plans are data: the values are measured with one model
//! on one CAPTCHA service's grids and live in that model's TOML configuration file.

use std::fmt;

use anyhow::{Result, ensure};
use clap::ValueEnum;
use serde::{Deserialize, Deserializer, Serialize, de};

/// How much the model is asked about a 4 × 4 grid. Each level sends more than the one below
/// it, and so costs more tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Effort {
    /// The whole grid, one close-up pass and the skip question.
    Low,
    /// Low, and boxes drawn on the whole grid.
    Medium,
    /// Medium, and a second close-up pass.
    #[default]
    High,
    /// High, and a box drawn in a window around every square.
    Max,
}

impl Effort {
    pub const ALL: [Self; 4] = [Self::Low, Self::Medium, Self::High, Self::Max];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

/// One of the box requests a round can send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxAsk {
    /// Nine questions on a picture of the whole grid, one box each: the largest, the second
    /// largest, the leftmost and so on.
    Whole,
    /// One question on the same picture, answered with a box around every instance. Only
    /// models with `supports_every_box = true` answer it.
    Every,
    /// One question on each of sixteen pictures, a square with its neighbours.
    Windows,
}

impl BoxAsk {
    /// The request's name in the round's log.
    pub fn name(self) -> &'static str {
        match self {
            Self::Whole => "whole_grid",
            Self::Every => "every",
            Self::Windows => "windows",
        }
    }
}

/// One plan for each effort level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Efforts {
    pub low: Plan,
    pub medium: Plan,
    pub high: Plan,
    pub max: Plan,
}

impl Efforts {
    pub fn plan(&self, effort: Effort) -> &Plan {
        match effort {
            Effort::Low => &self.low,
            Effort::Medium => &self.medium,
            Effort::High => &self.high,
            Effort::Max => &self.max,
        }
    }

    pub fn validate(&self, supports_every_box: bool) -> Result<()> {
        for (name, plan) in [
            ("low", &self.low),
            ("medium", &self.medium),
            ("high", &self.high),
            ("max", &self.max),
        ] {
            plan.validate(supports_every_box)
                .map_err(|err| anyhow::anyhow!("efforts.{name}: {err}"))?;
        }
        Ok(())
    }
}

/// The probability at log-odds 3, a little over 95%.
#[cfg(test)]
pub const LOG_ODDS_3: f64 = 0.9525741268224334;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    /// One close-up pass per entry: every square is enlarged from a copy of the grid and
    /// scored. `None` is the grid as captured (`"capture"` in TOML), a number the longer side
    /// in pixels of a rescaled copy. The passes' scores are averaged in log-odds.
    #[serde(deserialize_with = "deserialize_close_ups")]
    pub close_ups: Vec<Option<u32>>,
    /// A square is selected when its combined score is at least this.
    pub select_from: f64,
    /// On a prompt that allows skipping, the round also asks whether the target appears at
    /// all and selects nothing when that probability is below this. `None` does not ask.
    #[serde(default)]
    pub skip_below: Option<f64>,
    /// The box rule, with the box requests it needs. `None` sends none.
    #[serde(default)]
    pub boxes: Option<BoxRule>,
    /// The gap rule. It needs no request of its own.
    #[serde(default)]
    pub gap: Option<GapRule>,
}

/// The model is asked to draw boxes around the target, and a selected square is dropped
/// unless a box from every box request the round sends reaches it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxRule {
    /// A selection scoring at least this is never dropped.
    pub exempt_from: f64,
    /// The least confident box that counts among the nine drawn on the whole grid, one to a
    /// question. `None` does not send that request.
    #[serde(default)]
    pub whole: Option<f64>,
    /// The least confident box that counts among those drawn on the whole grid in answer to
    /// one question asking for every instance. `None` does not send that request.
    #[serde(default)]
    pub every: Option<f64>,
    /// The least confident box that counts among those drawn in a window around each square.
    /// `None` does not send that request.
    #[serde(default)]
    pub windows: Option<f64>,
    /// A box reaches a square when it covers at least this share of the square
    pub cover: f64,
    /// and at least this share of the box lies inside the square.
    pub inside: f64,
}

impl BoxRule {
    /// The least confident box that counts from a request, or `None` when the rule does not
    /// send that request.
    pub fn confidence(&self, ask: BoxAsk) -> Option<f64> {
        match ask {
            BoxAsk::Whole => self.whole,
            BoxAsk::Every => self.every,
            BoxAsk::Windows => self.windows,
        }
    }

    /// The box requests a round sends for this rule.
    pub fn asks(&self) -> impl Iterator<Item = BoxAsk> + use<> {
        let rule = *self;
        [BoxAsk::Whole, BoxAsk::Every, BoxAsk::Windows]
            .into_iter()
            .filter(move |&ask| rule.confidence(ask).is_some())
    }

    /// How many box requests that is.
    pub fn requests(&self) -> u8 {
        self.asks().count() as u8
    }

    pub fn validate(&self, supports_every_box: bool) -> Result<()> {
        validate_probability("boxes.exempt_from", self.exempt_from)?;
        validate_probability("boxes.cover", self.cover)?;
        validate_probability("boxes.inside", self.inside)?;
        if let Some(whole) = self.whole {
            validate_probability("boxes.whole", whole)?;
        }
        if let Some(every) = self.every {
            validate_probability("boxes.every", every)?;
            ensure!(
                supports_every_box,
                "boxes.every requires endpoint.supports_every_box = true"
            );
        }
        if let Some(windows) = self.windows {
            validate_probability("boxes.windows", windows)?;
        }
        ensure!(
            self.requests() > 0,
            "boxes must specify at least one of whole, every, or windows"
        );
        Ok(())
    }
}

/// A selected square scoring under `under` is dropped when its close-up score exceeds its
/// whole-grid score by more than `more_than`: only the close-up believes in it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GapRule {
    pub under: f64,
    pub more_than: f64,
}

impl GapRule {
    pub fn validate(&self) -> Result<()> {
        validate_probability("gap.under", self.under)?;
        validate_probability("gap.more_than", self.more_than)?;
        Ok(())
    }
}

impl Plan {
    /// One close-up pass on the grid as captured and nothing else: a round with nothing tuned.
    pub fn plain() -> Self {
        Self {
            close_ups: vec![None],
            select_from: 0.5,
            skip_below: None,
            boxes: None,
            gap: None,
        }
    }

    /// Asks everything any plan asks and decides as a plain round does: both close-up passes,
    /// the skip question, and every box request, with thresholds that can change no answer.
    /// For recording a model's answers before anything has been measured with it. `every`
    /// says whether the model answers the question asking for every instance.
    pub fn record(every: bool) -> Self {
        Self {
            close_ups: vec![None, Some(506)],
            select_from: 0.5,
            skip_below: Some(0.0),
            boxes: Some(BoxRule {
                exempt_from: 0.0,
                whole: Some(0.0),
                every: if every { Some(0.0) } else { None },
                windows: Some(0.0),
                cover: 0.0,
                inside: 0.0,
            }),
            gap: None,
        }
    }

    pub fn validate(&self, supports_every_box: bool) -> Result<()> {
        ensure!(!self.close_ups.is_empty(), "close_ups must not be empty");
        for px in self.close_ups.iter().flatten() {
            ensure!(*px > 0, "close_ups pixel size must be positive");
        }
        validate_probability("select_from", self.select_from)?;
        if let Some(skip_below) = self.skip_below {
            validate_probability("skip_below", skip_below)?;
        }
        if let Some(boxes) = &self.boxes {
            boxes.validate(supports_every_box)?;
        }
        if let Some(gap) = &self.gap {
            gap.validate()?;
        }
        Ok(())
    }
}

pub(crate) fn validate_probability(name: &str, value: f64) -> Result<()> {
    ensure!(
        value.is_finite() && (0.0..=1.0).contains(&value),
        "{name} must be a finite probability in 0..=1 (got {value})"
    );
    Ok(())
}

fn deserialize_close_ups<'de, D>(deserializer: D) -> Result<Vec<Option<u32>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct Pass(Option<u32>);

    impl<'de> Deserialize<'de> for Pass {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            struct PassVisitor;

            impl de::Visitor<'_> for PassVisitor {
                type Value = Pass;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("\"capture\" or a positive integer pixel size")
                }

                fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                    if value == "capture" {
                        Ok(Pass(None))
                    } else {
                        Err(E::invalid_value(de::Unexpected::Str(value), &self))
                    }
                }

                fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                    let px = u32::try_from(value)
                        .ok()
                        .filter(|&px| px > 0)
                        .ok_or_else(|| E::invalid_value(de::Unexpected::Unsigned(value), &self))?;
                    Ok(Pass(Some(px)))
                }

                fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                    let px = u32::try_from(value)
                        .ok()
                        .filter(|&px| px > 0)
                        .ok_or_else(|| E::invalid_value(de::Unexpected::Signed(value), &self))?;
                    Ok(Pass(Some(px)))
                }

                fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                    Ok(Pass(None))
                }

                fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                    Ok(Pass(None))
                }
            }

            deserializer.deserialize_any(PassVisitor)
        }
    }

    let passes = Vec::<Pass>::deserialize(deserializer)?;
    Ok(passes.into_iter().map(|pass| pass.0).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rule_sends_the_requests_it_has_a_confidence_for() {
        let rule = |whole, every, windows| BoxRule {
            exempt_from: 0.9,
            whole,
            every,
            windows,
            cover: 0.01,
            inside: 0.0,
        };
        let asks = |rule: BoxRule| rule.asks().collect::<Vec<_>>();
        assert_eq!(asks(rule(Some(0.3), None, None)), [BoxAsk::Whole]);
        assert_eq!(asks(rule(None, Some(0.3), None)), [BoxAsk::Every]);
        assert_eq!(
            asks(rule(Some(0.3), None, Some(0.6))),
            [BoxAsk::Whole, BoxAsk::Windows]
        );
        assert_eq!(Plan::record(false).boxes.unwrap().requests(), 2);
        assert_eq!(Plan::record(true).boxes.unwrap().requests(), 3);
    }

    #[test]
    fn the_exemption_constant_is_log_odds_three() {
        assert_eq!(LOG_ODDS_3, 1.0 / (1.0 + (-3.0f64).exp()));
    }

    #[test]
    fn plan_deserializes_from_toml_and_serializes_capture_as_null_in_json() {
        let plan: Plan = toml::from_str(
            r#"
            close_ups = ["capture", 506]
            select_from = 0.5
            skip_below = 0.7
            "#,
        )
        .unwrap();
        assert_eq!(plan.close_ups, vec![None, Some(506)]);
        assert!(plan.validate(false).is_ok());
        let json = serde_json::to_value(&plan).unwrap();
        assert_eq!(json["close_ups"], serde_json::json!([null, 506]));
    }

    #[test]
    fn default_effort_is_high() {
        assert_eq!(Effort::default(), Effort::High);
        assert_eq!(
            Effort::ALL.map(Effort::as_str),
            ["low", "medium", "high", "max"]
        );
    }
}
