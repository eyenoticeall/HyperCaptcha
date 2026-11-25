//! Opt-in `.env` file parser and process-wide read-only overlay.
//!
//! Files are only loaded when the user explicitly passes `--env-file <PATH>`.
//! Existing process environment variables always take precedence over values from
//! `--env-file`; when multiple `--env-file` arguments are given, later files
//! override earlier ones for keys not set in the process environment.

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, bail};

static OVERLAY: OnceLock<BTreeMap<String, String>> = OnceLock::new();

/// Loads one or more `.env` files in order and installs them as the fallback overlay
/// used by [`lookup`]. Does nothing when `paths` is empty.
pub fn load_files(paths: &[PathBuf]) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }
    let merged = read_files(paths)?;
    OVERLAY
        .set(merged)
        .map_err(|_| anyhow::anyhow!("env-file overlay was already initialized"))?;
    Ok(())
}

/// Reads and merges one or more `.env` files in order (later files override earlier ones).
pub fn read_files(paths: &[PathBuf]) -> Result<BTreeMap<String, String>> {
    let mut merged = BTreeMap::new();
    for path in paths {
        let parsed = read_file(path)?;
        merged.extend(parsed);
    }
    Ok(merged)
}

/// Reads and parses a single `.env` file.
pub fn read_file(path: &Path) -> Result<BTreeMap<String, String>> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("read env file {}", path.display()))?;
    parse_str(&contents).with_context(|| format!("parse env file {}", path.display()))
}

/// Looks up an environment variable with process-environment precedence over any
/// loaded `--env-file` overlay, returning `None` if unset or whitespace-only.
pub fn lookup(name: &str) -> Option<String> {
    match env::var(name) {
        Ok(value) => (!value.trim().is_empty()).then_some(value),
        Err(env::VarError::NotPresent) => OVERLAY
            .get()
            .and_then(|map| map.get(name))
            .filter(|value| !value.trim().is_empty())
            .cloned(),
        Err(env::VarError::NotUnicode(_)) => None,
    }
}

/// Pure precedence helper for testing without mutating process environment state.
#[cfg(test)]
fn lookup_with_overlay(
    name: &str,
    process_value: Option<&str>,
    overlay: &BTreeMap<String, String>,
) -> Option<String> {
    if let Some(value) = process_value {
        return (!value.trim().is_empty()).then(|| value.to_owned());
    }
    overlay
        .get(name)
        .filter(|value| !value.trim().is_empty())
        .cloned()
}

/// Parses `.env` text supporting `KEY=VALUE`, optional `export ` prefix, `#` comments,
/// and single- or double-quoted values. Error messages intentionally omit line contents
/// so secrets are never printed.
pub fn parse_str(contents: &str) -> Result<BTreeMap<String, String>> {
    let contents = contents.strip_prefix('\u{feff}').unwrap_or(contents);
    let mut entries = BTreeMap::new();

    for (index, raw_line) in contents.lines().enumerate() {
        let line_no = index + 1;
        let line = raw_line.trim_start();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = strip_export_prefix(line);
        let Some((raw_key, raw_value)) = line.split_once('=') else {
            bail!("line {line_no}: expected KEY=VALUE");
        };
        let key = raw_key.trim();
        if !is_valid_key(key) {
            bail!("line {line_no}: invalid variable name");
        }
        let value = parse_value(raw_value, line_no)?;
        entries.insert(key.to_owned(), value);
    }

    Ok(entries)
}

fn strip_export_prefix(line: &str) -> &str {
    if let Some(rest) = line.strip_prefix("export")
        && let Some(first) = rest.chars().next()
        && matches!(first, ' ' | '\t')
    {
        return rest.trim_start_matches([' ', '\t']);
    }
    line
}

fn is_valid_key(key: &str) -> bool {
    let mut bytes = key.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn parse_value(raw_value: &str, line_no: usize) -> Result<String> {
    let trimmed_start = raw_value.trim_start_matches([' ', '\t']);
    if let Some(rest) = trimmed_start.strip_prefix('\'') {
        let Some((value, trailing)) = rest.split_once('\'') else {
            bail!("line {line_no}: unterminated single-quoted value");
        };
        check_trailing_comment(trailing, line_no)?;
        return Ok(value.to_owned());
    }
    if let Some(rest) = trimmed_start.strip_prefix('"') {
        let mut value = String::new();
        let mut chars = rest.chars();
        loop {
            match chars.next() {
                Some('"') => {
                    check_trailing_comment(chars.as_str(), line_no)?;
                    return Ok(value);
                }
                Some('\\') => match chars.next() {
                    Some('n') => value.push('\n'),
                    Some('r') => value.push('\r'),
                    Some('t') => value.push('\t'),
                    Some('"') => value.push('"'),
                    Some('\\') => value.push('\\'),
                    Some('$') => value.push('$'),
                    Some(other) => {
                        value.push('\\');
                        value.push(other);
                    }
                    None => bail!("line {line_no}: unterminated double-quoted value"),
                },
                Some(ch) => value.push(ch),
                None => bail!("line {line_no}: unterminated double-quoted value"),
            }
        }
    }

    // Unquoted value: an inline `#` starts a comment only when preceded by whitespace.
    let mut prev_ws = false;
    let mut end = raw_value.len();
    for (idx, ch) in raw_value.char_indices() {
        if ch == '#' && prev_ws {
            end = idx;
            break;
        }
        prev_ws = matches!(ch, ' ' | '\t');
    }
    Ok(raw_value[..end].trim_matches([' ', '\t']).to_owned())
}

fn check_trailing_comment(trailing: &str, line_no: usize) -> Result<()> {
    let rest = trailing.trim_matches([' ', '\t']);
    if rest.is_empty() || rest.starts_with('#') {
        return Ok(());
    }
    bail!("line {line_no}: unexpected trailing characters after quoted value");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_dotenv_syntax() {
        let input = "\u{feff}\
# Full-line comment
   # Indented comment

PLAIN=hello-world
export EXPORTED=from-export
export\tTAB_EXPORT=tab-val
export_key=not-a-keyword
EMPTY=
EMPTY_COMMENT=   # trailing comment
HASH_INSIDE=abc#123
HASH_LEADING=#not-a-comment
INLINE_COMMENT=keep-me # ignore this
SINGLE='literal # value \\n stays raw' # comment
DOUBLE=\"escaped\\nline\\t\\\"quote\\\"\\\\slash\\$dollar\\q\" # comment
SPACED =  value with spaces  \r
DUPLICATE=first
DUPLICATE=second
";
        let parsed = parse_str(input).unwrap();
        assert_eq!(parsed["PLAIN"], "hello-world");
        assert_eq!(parsed["EXPORTED"], "from-export");
        assert_eq!(parsed["TAB_EXPORT"], "tab-val");
        assert_eq!(parsed["export_key"], "not-a-keyword");
        assert_eq!(parsed["EMPTY"], "");
        assert_eq!(parsed["EMPTY_COMMENT"], "");
        assert_eq!(parsed["HASH_INSIDE"], "abc#123");
        assert_eq!(parsed["HASH_LEADING"], "#not-a-comment");
        assert_eq!(parsed["INLINE_COMMENT"], "keep-me");
        assert_eq!(parsed["SINGLE"], "literal # value \\n stays raw");
        assert_eq!(
            parsed["DOUBLE"],
            "escaped\nline\t\"quote\"\\slash$dollar\\q"
        );
        assert_eq!(parsed["SPACED"], "value with spaces");
        assert_eq!(parsed["DUPLICATE"], "second");
    }

    #[test]
    fn malformed_lines_fail_without_leaking_secret_values() {
        for bad in [
            "super-secret-token-without-equals",
            "123BAD=super-secret-token",
            "KEY='unterminated-secret",
            "KEY=\"unterminated-secret",
            "KEY=\"secret\" trailing-junk",
            "KEY='secret' trailing-junk",
        ] {
            let err = parse_str(bad).unwrap_err().to_string();
            assert!(!err.contains("secret"), "leaked secret in error: {err}");
        }
    }

    #[test]
    fn precedence_prefers_process_env_then_later_env_files() {
        let root =
            std::env::temp_dir().join(format!("hypercaptcha-envfile-unit-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let first = root.join("first.env");
        let second = root.join("second.env");
        fs::write(&first, "A=from-first\nB=from-first\nC=from-first\n").unwrap();
        fs::write(&second, "B=from-second\nC=\nD=from-second\n").unwrap();

        let overlay = read_files(&[first, second]).unwrap();
        // Process env wins over overlay (even if empty).
        assert_eq!(
            lookup_with_overlay("A", Some("from-proc"), &overlay).as_deref(),
            Some("from-proc")
        );
        assert_eq!(lookup_with_overlay("A", Some(""), &overlay), None);
        // Unset in process env falls back to merged overlay (later file wins).
        assert_eq!(
            lookup_with_overlay("A", None, &overlay).as_deref(),
            Some("from-first")
        );
        assert_eq!(
            lookup_with_overlay("B", None, &overlay).as_deref(),
            Some("from-second")
        );
        // Empty value in later file clears earlier value.
        assert_eq!(lookup_with_overlay("C", None, &overlay), None);
        assert_eq!(
            lookup_with_overlay("D", None, &overlay).as_deref(),
            Some("from-second")
        );
        assert_eq!(lookup_with_overlay("MISSING", None, &overlay), None);

        let _ = fs::remove_dir_all(&root);
    }
}
