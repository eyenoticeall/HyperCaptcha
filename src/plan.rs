//! How one 4 × 4 round is asked and decided.
//!
//! A plan names the requests a round sends beside the whole-grid request, and the rules that
//! turn their answers into a selection. Plans are data: the values are measured with one model
//! on one CAPTCHA service's grids and live in that model's settings file.

use clap::ValueEnum;
use serde::Serialize;

/// How much the model is asked about a 4 × 4 grid. Each level sends more than the one below
/// it, and so costs more tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Effort {
    /// The whole grid, one close-up pass and the skip question.
    Low,
    /// Low, and boxes drawn on the whole grid.
    Medium,
    /// Medium, and a second close-up pass.
    High,
    /// High, and a box drawn in a window around every square.
    #[default]
    Max,
}

/// One of the box requests a round can send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxAsk {
    /// Nine questions on a picture of the whole grid, one box each: the largest, the second
    /// largest, the leftmost and so on.
    Whole,
    /// One question on the same picture, answered with a box around every instance. Only
    /// Liquid's models answer it.
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
pub struct Efforts {
    pub low: Plan,
    pub medium: Plan,
    pub high: Plan,
    pub max: Plan,
}

impl Efforts {
    pub fn plan(&self, effort: Effort) -> Plan {
        match effort {
            Effort::Low => self.low,
            Effort::Medium => self.medium,
            Effort::High => self.high,
            Effort::Max => self.max,
        }
    }
}

/// The probability at log-odds 3, a little over 95%.
pub const LOG_ODDS_3: f64 = 0.9525741268224334;

#[derive(Clone, Copy, Serialize)]
pub struct Plan {
    /// One close-up pass per entry: every square is enlarged from a copy of the grid and
    /// scored. `None` is the grid as captured, a number the longer side in pixels of a
    /// rescaled copy. The passes' scores are averaged in log-odds.
    pub close_ups: &'static [Option<u32>],
    /// A square is selected when its combined score is at least this.
    pub select_from: f64,
    /// On a prompt that allows skipping, the round also asks whether the target appears at
    /// all and selects nothing when that probability is below this. `None` does not ask.
    pub skip_below: Option<f64>,
    /// The box rule, with the box requests it needs. `None` sends none.
    pub boxes: Option<BoxRule>,
    /// The gap rule. It needs no request of its own.
    pub gap: Option<GapRule>,
}

/// The model is asked to draw boxes around the target, and a selected square is dropped
/// unless a box from every box request the round sends reaches it.
#[derive(Clone, Copy, Serialize)]
pub struct BoxRule {
    /// A selection scoring at least this is never dropped.
    pub exempt_from: f64,
    /// The least confident box that counts among the nine drawn on the whole grid, one to a
    /// question. `None` does not send that request.
    pub whole: Option<f64>,
    /// The least confident box that counts among those drawn on the whole grid in answer to
    /// one question asking for every instance. `None` does not send that request.
    pub every: Option<f64>,
    /// The least confident box that counts among those drawn in a window around each square.
    /// `None` does not send that request.
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
}

/// A selected square scoring under `under` is dropped when its close-up score exceeds its
/// whole-grid score by more than `more_than`: only the close-up believes in it.
#[derive(Clone, Copy, Serialize)]
pub struct GapRule {
    pub under: f64,
    pub more_than: f64,
}

impl Plan {
    /// One close-up pass on the grid as captured and nothing else: a round with nothing tuned.
    pub const PLAIN: Self = Self {
        close_ups: &[None],
        select_from: 0.5,
        skip_below: None,
        boxes: None,
        gap: None,
    };

    /// Asks everything any plan asks and decides as a plain round does: both close-up passes,
    /// the skip question, and every box request, with thresholds that can change no answer.
    /// For recording a model's answers before anything has been measured with it. `every`
    /// says whether the model answers the question asking for every instance.
    pub const fn record(every: bool) -> Self {
        Self {
            close_ups: &[None, Some(506)],
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
}
