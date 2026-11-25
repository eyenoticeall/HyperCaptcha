# HyperCaptcha

HyperCaptcha is a Rust research harness for real CAPTCHA widgets on a local page: Google reCAPTCHA, and hCaptcha's image grids. It logs model responses and the service's verification results under the same trial ID. Run automated image trials in Chrome or Camoufox. Camoufox also supports manual image trials and browser-driven score trials.

## CAPTCHA services

`--captcha` picks the service the lab hosts: `recaptcha`, the default, or `hcaptcha`. Each service has its own code, page, runner and solver settings, so tuning one leaves the other as it was measured.

| | reCAPTCHA | hCaptcha |
| --- | --- | --- |
| Token verification and solver settings | `src/captcha/recaptcha.rs` | `src/captcha/hcaptcha/` |
| Lab page | `web/recaptcha/` | `web/hcaptcha/` |
| Runner | `scripts/chrome_lab.py` | `scripts/hcaptcha_lab.py` |

A service's `Profile` in `src/captcha/` holds what was measured on its challenges: how its instruction is read, the question asked about one tile, the weights and cutoffs that combine d3-ultra's two views of a tile, and the 4 × 4 skip check, second pass and box rule. The sections from here to [hCaptcha](#hcaptcha) describe reCAPTCHA. Its answers did not change when the settings moved into its profile: the 333 saved 4 × 4 grids and 61 saved 3 × 3 grids gave identical scores and selections before and after.

## Start the lab

Install Rust 1.88 or later. Run the public-key integration check without credentials.

```sh
cargo run --locked -- serve --mode v2 --google-test-key
```

Open [localhost on port 8765](http://localhost:8765) and check the Google checkbox. Google's public key always skips the image challenge and returns a passing verification response. Google's Cloud testing options can also force fixed scores or placeholder challenges. Use regular keys to collect research results. [Google public keys](https://developers.google.com/recaptcha/docs/faq/#id-like-to-run-automated-tests-with-recaptcha-what-should-i-do) · [Google Cloud test-key behavior](https://docs.cloud.google.com/sdk/gcloud/reference/recaptcha/keys/create)

## Configure live keys

Create a checkbox or score-based key for each integration you want to study. Allow `localhost` on each dedicated research key and keep the verification credentials in your server environment. Restart the lab with the matching mode and environment variables when changing integrations. [Create Google keys](https://docs.cloud.google.com/recaptcha/docs/create-key-website) · [Localhost setup](https://developers.google.com/recaptcha/docs/faq/#im-getting-an-error-localhost-is-not-in-the-list-of-supported-domains-what-should-i-do)

| Mode | Key type | Server environment |
| --- | --- | --- |
| `v2` | Checkbox | `RECAPTCHA_SITE_KEY`, `RECAPTCHA_SECRET_KEY` |
| `v2-enterprise` | Checkbox | `RECAPTCHA_SITE_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_API_KEY` |
| `v3` | Score | `RECAPTCHA_SITE_KEY`, `RECAPTCHA_SECRET_KEY` |
| `v3-enterprise` | Score | `RECAPTCHA_SITE_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_API_KEY` |
| `policy-challenge` | Policy-based challenge | `RECAPTCHA_SITE_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_API_KEY` |

The legacy modes use Google's `api.js` and SiteVerify with a legacy secret. The Enterprise modes use `enterprise.js` and the Assessments API, which requires reCAPTCHA enabled in the Google Cloud project. These names identify the integration interfaces. Google's migration can put keys used via the legacy interface under Cloud management, so record the key configuration and tier alongside experimental results. [Classic migration](https://docs.cloud.google.com/recaptcha/docs/migration-overview) · [SiteVerify](https://docs.cloud.google.com/recaptcha/docs/verify) · [Assessments API](https://docs.cloud.google.com/recaptcha/docs/create-assessment-website)

## Request real image challenges

Use a dedicated **policy-based challenge** key on a Google Cloud project with billing enabled and the reCAPTCHA Enterprise API enabled. Set its default threshold to `1.0` for a challenge on every request. The example uses Google's `security` difficulty, which can produce harder challenges. Keep this policy on a dedicated lab key so results can be compared with naturally triggered v2 challenges. [Google policy-based challenge setup](https://docs.cloud.google.com/recaptcha/docs/instrument-web-pages-policy) · [Key creation flags](https://docs.cloud.google.com/sdk/gcloud/reference/recaptcha/keys/create)

```sh
gcloud auth login
export GOOGLE_CLOUD_PROJECT="your-research-project"
gcloud services enable recaptchaenterprise.googleapis.com --project="$GOOGLE_CLOUD_PROJECT"
export RECAPTCHA_SITE_KEY="$(gcloud recaptcha keys create \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --display-name="HyperCaptcha localhost image trials" \
  --web --domains=localhost --integration-type=policy-based-challenge \
  --default-score-threshold=1.0 --security-preference=security \
  --format='value(name.basename())')"
```

Supply an existing `GOOGLE_CLOUD_API_KEY` or use a short-lived `GOOGLE_CLOUD_ACCESS_TOKEN`. For the token option, sign in with `gcloud auth login` and use an account that can create reCAPTCHA assessments. Restart the lab with a fresh token when it expires. [Authenticate assessment requests](https://docs.cloud.google.com/recaptcha/docs/create-assessment-website)

```sh
export GOOGLE_CLOUD_ACCESS_TOKEN="$(gcloud auth print-access-token)"
mkdir -p reports
cargo run --locked -- serve --mode policy-challenge --log reports/image-challenges.jsonl
```

Open the lab in Camoufox and click **Request image challenge**. The existing screenshot and model workflow applies if you start the server with a vision provider. Assessments include Google's challenge result when available.

For naturally triggered trials, use a regular v2 checkbox key with its security preference set to `security`. Google still decides when to show the challenge. v3 score-only keys return scores for the interaction. [Google key types](https://docs.cloud.google.com/recaptcha/docs/choose-key-type)

## Connect a model

```sh
mkdir -p reports
cargo run --locked -- serve --mode v2 --provider openai-chat \
  --model "$OPENAI_MODEL" --log reports/v2-chat.jsonl

cargo run --locked -- serve --mode v2-enterprise --provider jev \
  --model "$JEV_MODEL" --log reports/v2-enterprise-jev.jsonl

cargo run --locked -- serve --mode v3 --provider openai-completions \
  --model "$COMPLETIONS_MODEL" --base-url "$COMPLETIONS_BASE_URL" \
  --log reports/v3-completions.jsonl
```

- **Chat Completions** uses `OPENAI_API_KEY`. Choose a model with image input for v2.
- **typesafe/Jev** (`--provider jev`) uses `TYPESAFE_API_KEY` or `JEV_API_KEY` and defaults to `jev-latest`. It asks a yes/no question per tile and selects tiles with probability at or above `--tile-threshold`, which defaults to 0.5.
- **vLLM/d3-ultra** (`--provider vllm`) uses `VLLM_SR_API_KEY` and sends `vllm-sr/d3-ultra` on System One. It supports image grids and score fixtures. Further keys, separated by commas in `VLLM_SR_EXTRA_API_KEYS`, share the requests in turn, which raises the rate limit because the provider counts it per key. [System One API](https://app.vllm-sr.ai/docs)
- **Liquid/d1.2** and **Liquid/d1.2-ultra** (`--provider liquid`, which defaults to `d1.2`) use `LIQUID_API_KEY`. They are asked the way vLLM/d3-ultra is, with weights and effort levels measured for each on 4 × 4 grids: see [Liquid](#liquid). [Liquid decisions](https://docs.liquid.ai/api/guides/decisions)
- **Legacy Completions** uses `OPENAI_API_KEY` and accepts text. Use an OpenAI-compatible backend that serves `/completions`. OpenAI retired its documented legacy models.
- **Google alone** works with no `--provider`.

A model is named with whoever serves it, because two providers can serve models with similar names: `vLLM/d3-ultra`, `typesafe/Jev`. `--model vLLM/d3-ultra` chooses the provider as well, so it needs no `--provider`, and capitals do not matter. `/config` and every log line give that name as `model_name`, beside the ID the provider itself uses in `model`. A provider's own ID still works with `--provider`, and is then shown as, for example, `typesafe/jev-2`. The sections below say "Jev" and "vLLM" for short, and the `--jev-*` and `--vllm-*` flags and the `vllm_components` log field keep their names.

Custom API roots include the version prefix, usually `/v1`. Use `--base-url` or the provider's `OPENAI_BASE_URL`, `JEV_BASE_URL`, `VLLM_SR_BASE_URL` or `LIQUID_BASE_URL` variable. Model requests use `--timeout` with a 60-second default. OpenAI requests also support `--max-output-tokens`. [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) · [Legacy Completions](https://developers.openai.com/api/docs/guides/completions) · [Jev images](https://docs.typesafe.ai/images)

## Run a trial

### Automated checkbox trials in Chrome

Start the Rust lab with a live `v2` or `v2-enterprise` checkbox key. Use the `security` preference for the dedicated research key. With Google Chrome installed, run:

```sh
uv run --with playwright==1.62.0 scripts/chrome_lab.py --trials 3
```

Each trial opens a fresh Chrome context and clicks the checkbox once. Google decides whether to issue a token immediately or show a challenge. Capture mode saves the first supported 3 × 3 or 4 × 4 image grid and its exact instructions. The trial log is `reports/chrome/trials.jsonl`. A checkbox-only pass is counted separately from an image trial.

To evaluate a model, start the lab with `--provider openai-chat`, `--provider jev`, or `--provider vllm` and the corresponding credentials, then add `--submit`.

```sh
uv run --with playwright==1.62.0 scripts/chrome_lab.py --trials 3 --submit
```

The browser sends the grid and instructions to Rust, clicks the returned tile numbers, and takes another screenshot if Google replaces tiles. The runner verifies when the model is finished selecting tiles and records the Google token assessment, a visible rejection, or the reason the trial stopped. The trials log includes model answers and references to the PNGs, grouped by trial ID. Use `--max-rounds` to change the six-round limit. By default a trial ends at Google's first rejection. With `--max-stages 10` it keeps going until Google passes it or has rejected ten times: after "Please try again" it answers the new challenge Google loads, and when Google keeps the grid and asks for more tiles it presses the widget's reload button for a different one. Replaced tiles in a dynamic grid belong to the same answer, and grids that Google serves in a row before giving one verdict count as one rejection at most. An answer counts as rejected when the widget sent its verify request and another challenge followed, or when an error message appeared. Each round then logs its `stage`, the number of the answer it belongs to, and any `rejection` message, and a trial that is rejected that many times ends as `stage_limit`. One that reaches eight rounds per allowed rejection without a pass ends as `round_limit`, since Google can keep serving grids without a verdict. With `--use-hints`, when Google keeps the grid and asks to select all matching images, the runner adds the highest-scoring tile not yet selected and verifies again, up to twice per grid, and logs each hint with the round. Across 550 live attempts Google asked for more tiles 6 times, and the added tile led to a pass once, so the option is off by default. Each round records `timeline_ms`: the time since the checkbox click at which the grid finished loading, the screenshot was taken, the model answered, the tiles were clicked and verification was requested. The runner checks the page every 50 to 100 milliseconds and reads the first grid as soon as it has loaded. `--no-fast` restores the earlier pacing: a check every 200 to 300 milliseconds after a half-second pause. Use `--url` for a different localhost port and `--output` for a different run directory.

Before sending a grid to a decision provider, the lab sends red and blue control images through the provider's endpoint using its image encoding. It requires a probability of at least 0.7 that the red image is red and at most 0.3 for the blue image. The lab reuses a passing control result until the server restarts. The runner saves the result, token usage, and latency to `vision-checks.jsonl` and exits before opening a challenge if either control fails. Trial records link to the control through `vision_check_id` and include a SHA-256 hash for each captured grid. [TypeSafe image format](https://docs.typesafe.ai/images)

Command Code's Jev endpoint answered text requests in testing, but returned similar answers and identical token usage with different PNGs, an invalid PNG, and an omitted image. Use a vision-capable System One endpoint for image trials. With credentials for TypeSafe, start the server with `--provider jev --model jev-1.14.0 --base-url https://api.typesafe.ai/v1` and set `TYPESAFE_API_KEY`. [Command Code Provider API](https://commandcode.ai/docs/provider#decision-models-typesafejev)

### Compare Jev image inputs

The default `--jev-vision whole-grid` uses a single PNG per round. Add `--jev-vision scene-and-crop` to test the full scene plus one enlarged crop per square for 4 × 4 challenges with the English “Select all squares with” prompt. Jev gets both the scene and the crop in one request and answers whether any visible part of the target falls inside the crop. The mode uses the full grid for 3 × 3 rounds and other prompt formats. The server logs the image strategy used for each result. Use separate output directories and the same tile threshold to compare runs. [TypeSafe image inputs](https://docs.typesafe.ai/images)

Add `--jev-vision average` to score each supported English grid twice. The first request uses the whole grid. The second uses individual photos for 3 × 3 grids and the full scene with per-square close-ups for 4 × 4 grids. Rust averages the two probabilities for every tile and selects tiles at or above 60% on 3 × 3 grids and 50% on 4 × 4 grids. An explicit `--tile-threshold` overrides both cutoffs. Each round logs both sets of scores, the average, and the cutoff used.

For a matched comparison with vLLM, run Jev with `--jev-vision average --jev-split-crops --decision-image-format jpeg`. Both providers send the full grid and crop batch in parallel on 3 × 3 grids. On 4 × 4 grids Jev sends the whole grid plus two crop batches of eight, in parallel. vLLM's nearest round is `--vllm-decision average`, which also scores a second close-up pass and asks the skip question. Use `--decision-image-format png` to compare image encodings with the same request layout. [TypeSafe image formats](https://docs.typesafe.ai/images)

### Use d3-ultra as an alternative

Load `VLLM_SR_API_KEY` and start a separate lab with a configured live checkbox key.

```sh
source .env.vllm
mkdir -p reports
cargo run --release --locked -- serve --mode v2-enterprise --port 8776 \
  --provider vllm --log reports/vllm.jsonl
```

vLLM asks d3-ultra about every tile twice, once on the whole grid and once on close-ups, and combines the two answers with a logistic rule. It uses JPEG for the full grid and crops and sends every request of a round at once: four for a 3 × 3 grid, the whole grid and three requests of three photos each. For 4 × 4 grids, what it sends depends on `--effort`, described under [Effort levels on 4 × 4 grids](#effort-levels-on-4--4-grids). The two higher levels send nine scoring requests: the whole grid, then two close-up passes of four crop requests each. The two lower levels send the second of those passes alone. Each crop request gets the full scene and four crops, enlarged with bilinear smoothing. The first pass uses the grid as captured and the second a copy resized to 506 pixels on its longer side. The two passes' scores for each square are averaged in log-odds before the logistic rule runs. When the prompt allows skipping, a tenth small request asks whether the target appears anywhere in the grid. Smoothing kept accuracy the same on human-labelled grids and made 4 × 4 requests about 18% smaller than nearest-neighbour enlargement. Jev keeps nearest-neighbour crops. Use `--tile-threshold` to override the cutoffs. The server records each request's client wall time, provider queue time, and model compute time in `provider_meta`. [System One API](https://app.vllm-sr.ai/docs)

A round takes as long as its slowest request, so the close-ups are spread over several small requests. d3-ultra's compute time grows with the close-ups in a request, from about 0.12 seconds for two to 0.26 for four and 0.5 for eight, and its scores are the same however the close-ups are grouped, as long as no request holds just one. Request bodies are gzip-compressed. All 258 labelled grids were replayed through the server with eight close-ups per 4 × 4 request and nine photos per 3 × 3 request, as before, and with the new grouping. Every score and every pick was identical. The API was 0.28 seconds away per round trip.

| Round | Requests before | Requests now | Median before | Median now | Slowest tenth before | Slowest tenth now |
|---|---:|---:|---:|---:|---:|---:|
| 4 × 4, 45 grids twice, builds timed in turn | 5 or 6 | 9 or 10 | 1.01 s | 0.73 s | over 1.53 s | over 1.14 s |
| 3 × 3, 72 grids, replayed hours apart | 2 | 4 | 0.91 s | 0.52 s | over 1.02 s | over 0.56 s |

Only the 4 × 4 row is a side-by-side measurement. Round times drifted by a quarter between passes on the same connection, so the 3 × 3 row shows the direction more than the size. In a live run the model took a median of 0.51 seconds on 398 3 × 3 rounds and 0.70 on 240 4 × 4 rounds, with one 4 × 4 round in ten over 1.4 seconds.

A 4 × 4 round now uses about 10,900 input tokens instead of 9,800, because every close-up request repeats the scene, and uploads 1.07 MB. Smaller groups were not faster. On 45 grids timed in turn against each other, three close-ups per request had a median of 0.77 seconds against 0.73 for four, and pairs spread over three keys 0.83. Both also had more slow rounds in that pass, in which one or two requests arrive about a second late while the model's compute time is normal. That pass ran before idle connections were limited to a round's worth, so part of the difference may come from duplicates opening unused connections. HTTP/2 was no faster than one HTTP/1.1 connection per request, so the client stays on HTTP/1.1. With every request on one shared connection, its fastest quarter of rounds took 0.87 seconds against 0.67 in the same pass. A pass over two to five HTTP/2 connections showed no gain either, but the key's rate limit refused a fifth to a half of its rounds. `--vllm-close-ups-per-request` sets the group size, from 2 to 9. The provider allows each key 16 requests at once.

A connection that has not yet carried an upload sends a round's request several round trips slower. Without preparation the first 4 × 4 round after the server starts took about 4 seconds and the second 1.5. So when the server starts, and whenever the lab page is opened after a quiet spell, it opens the twelve connections a 4 × 4 round needs and sends one practice request on each: a noise image with one question, about 270 KB and 394 input tokens. With that, a round sent four seconds after start took 1.4 seconds, half of it waiting for the practice requests to finish, and the next ones 0.64 to 0.74. Connections left unused for 45 seconds get a small `GET /models` request each, and a practice request if that shows a new connection was opened, for five minutes after the last request to the lab. Rounds sent after idle gaps of 20, 50 and 100 seconds took 0.63 to 0.75 seconds, apart from one of 1.5 seconds. Idle connections beyond those ten are closed. `--vllm-cold-connections` turns the preparation off.

If a request has no reply after 1.5 seconds, the server sends it again, and again at 4 and 8 seconds if there is still none, each time on the next key. It uses whichever reply comes first, since d3-ultra gives the same answer to the same request. About one request in several hundred stalls for ten seconds at the provider while its neighbours finish normally. On a poor network a request and its first duplicate can both stall: in two runs on such a network about 5% and 6% of requests were still unanswered after 4 seconds, and in one of them two rounds failed when a request ran out its 40 seconds. The later tries are untested against a real poor network. A request that the provider refuses as busy, or that fails with a server or connection error, is sent once more, on the next key when there is one. Each request's `provider_meta` records `request_wall_ms`, `request_bytes` and, when it was sent more than once, `attempts`.

The default `--vllm-decision logistic` works on the log-odds of the two answers. On 4 × 4 grids it also uses the strongest whole-grid and close-up evidence from the squares above, below, left and right, because objects run across neighbouring squares. It selects tiles at or above 50%. Accuracy was measured against human labels on 582 saved grids. A grid counts as right when every square holding the target is selected and no empty square is. Squares the target only barely enters are optional. The two rules pick different tiles on 97 grids, and a person reviewed all of them without seeing either rule's picks. A random sample of 99 of the remaining grids, where both rules pick the same tiles, was also reviewed. Combining the two gives each rule's accuracy across all saved grids of that kind:

| Grids | Average | Logistic | Logistic, two close-up passes | With the skip check |
| --- | ---: | ---: | ---: | ---: |
| Static 3 × 3 (143) | 87% | 91% | not used | not used |
| 4 × 4 (202) | 59% | 69% | 74% | 76% |

These figures come from sending the saved grids back through the lab server, so they reflect the harness's own crops and JPEG encoding. d3-ultra's answers shift with small pixel changes, and a separate re-encoding of the same grids moved 4 × 4 accuracy by several points, so measure changes through the server. On the 123 labelled 4 × 4 grids, logistic fixed 23 that averaging got wrong and broke 6. On 63 more labelled 4 × 4 grids, sent at seven image sizes, about a third of grids were right at some sizes and wrong at others, so single-grid differences of a few points are noise. Averaging nearest-neighbour and bilinear crops of the same size did not help. On the 18 dynamic 3 × 3 grids where the rules differ, logistic was right on 15 and averaging on 2. Most remaining 4 × 4 errors are squares selected without the target in them. A refit on the human labels gave nearly the same coefficients, so further gains need better answers from the model rather than a different combination rule. Model-made labels matched the human review on only about half of the borderline grids, so use human labels to settle close comparisons. Each round logs both views' scores, the combined probability, and the rule in `vllm_components`.

The second close-up pass exists because of that sensitivity to size: averaging the close-up scores from two sizes of the same capture steadies them. On all 186 labelled 4 × 4 grids, sent through the server, the second pass raised the grids answered right from 122 to 129, fixing 15 and breaking 8, and cut wrongly judged squares from 117 to 99. Ten of those grids had labels pre-filled from larger-size picks, which may favour the second pass. Without them the second pass fixed 12 grids and broke 8, taking 119 right of 176 to 123, and cut wrong squares from 104 to 93. That drop in wrong squares holds in 95% of resamples of the grids, while the grid count alone does not separate from chance. On the 64 grids labelled blind it fixed 4 and broke 3. The 74% in the table comes from the older labelled grids only, none of which had such pre-filled labels. Second-pass sizes of 440, 506, 560 and 622 pixels each gained between 3 and 7 grids of 186, a 330-pixel second pass lost 5, and a third pass added nothing. The requests run in parallel, so a round still takes about a second, with a median of 0.95 seconds across the 186 grids. In an interleaved live run of 200 attempts per setting, two-pass 4 × 4 rounds took a median of 1.13 seconds against 0.93 for one pass. Two passes solved 22 of 35 challenges that ended on a 4 × 4 grid and one pass 17 of 33, a difference well within chance. A 4 × 4 round uses about 9,500 input tokens, up from 5,300, and uploads 1.1 MB, up from 0.55 MB. The low and medium effort levels send the 506-pixel pass alone, which by itself scored 6 to 12 grids better than the capture alone on 339 graded grids. `vllm_components.variant.passes` logs each pass's size and scores.

The labels themselves only repeat so far. Forty of the 186 grids were labelled a second time from scratch, on the same day, with no model picks and no earlier answers shown. The first answer would count as right under the second label on 32 of the 40 grids, and the second under the first on 30, and same-day agreement is an upper bound. The sample held 24 grids the two-pass build got wrong and 16 it got right. Under the second label 4 of the 24 became right and 1 of the 16 became wrong, so there is no sign that pre-filling labels from model picks inflated the figures, though 16 grids cannot rule out a few points either way. Most of the remaining errors are the model's and not the labels': 19 of the 24 wrong grids had an error under both labels, 26 of the 34 wrongly selected squares were blank in both rounds, and 7 of the 9 missed squares were marked in both. Eight of those wrongly selected squares had scores of 0.9 or higher. Asked about each one directly, the reviewer called five empty, two a look-alike and one barely the target.

The whole-grid labels above turned out to be stricter than the reviewer's own judgement of single squares, so they understate 4 × 4 accuracy. Every square where the current build and the labels disagreed was shown again, one outlined square at a time, mixed with squares where they agreed. Of the 59 squares counted as wrongly selected, the reviewer called 15 the target and 21 a sliver at the edge that could go either way, leaving 23 real ones: 12 background, 7 look-alikes and 4 riders. Of the 27 squares counted as missed, 14 were slivers and 3 were not the target, leaving 10. None of the 25 agreed squares changed between target and blank. With those rulings applied the current build is right on 159 of the 186 grids, against 133 under the original labels. The first set of rulings was made knowing the model had picked each square and the second was not, and only this build's disagreements were reviewed, so 159 is an upper estimate and other builds need their own disagreements ruled on before they are compared on these labels. Against the corrected labels a different cutoff, the class answers and a close-up of the middle of each square all failed to improve on the current rule.

A second set was labelled blind afterwards: 125 grids from 4 × 4 skip challenges in the live runs, with the model's picks and Google's verdicts hidden. The 66 squares where the build and the labels disagreed were then ruled on one at a time among 24 agreed ones. Of the 51 squares counted as wrongly selected, the reviewer called 6 the target and 19 a sliver, leaving 26: 12 background, 9 look-alikes, 3 riders and 2 shadows. Of the 15 counted as missed, 13 stood. None of the 24 agreed squares changed. The build is right on 92 of the 125 grids. On blind labels alone, these grids and the 64 of the first set, it is right on 141 of 189, or 75%. That is the fairer figure: the 159 of 186 includes 122 grids whose labels started from model picks, and the build is right on 110 of those. The decision rule held on the fresh labels. The pooled blind grids gave 141 right at a cutoff of 0.5, 142 at 0.6 and 139 at 0.7, and rules refitted on them and tested on held-out grids averaged between 135 and 141.5. Most remaining wrong squares are single squares at an object's edge. The one repeated mistake is a pedestrian signal selected on a traffic-light prompt, at scores of 0.79 to 0.99, in 7 of the 42 traffic-light grids; that reading of the squares is the author's, and the reviewer ruled them look-alikes. A sentence saying that pedestrian crossing signals do not count was added to the questions for that target and replayed once on the 42 grids. It removed the pick in 2 of the 7 grids and broke none, short of the 4 set beforehand, so it was not adopted. Google had also passed one answer that included such a square.

Grouped into the challenges Google judged, the second set covers 11 five-grid challenges that passed and 12 that were rejected. The passed ones averaged 1.00 wrong grids and the rejected ones 1.42, a gap that chance produces about one time in four, so these labels do not show that accuracy decides a five-grid challenge. All five grids were right in 5 of the 11 passed and 2 of the 12 rejected.

Some 4 × 4 prompts end with "If there are none, click skip", and some of those grids show none of the target. The square-by-square questions then tend to select a look-alike, such as a moped on a bicycle prompt. For these prompts vLLM also asks one question about the whole grid, whether it shows at least one object of the target type, and selects nothing when the answer is below 0.7. The question goes in a request of its own, in parallel with the others, because adding it to the whole-grid request shifted that request's square scores. Of the 186 labelled grids, 133 allow skipping. The 14 of those without the target scored 0.57 or lower and the 119 with it scored 0.92 or higher, so the check skipped exactly the 14. Four of them had squares selected before, which took the grids answered right from 129 to 133 and wrong squares from 99 to 86, with no grid made worse and no square score changed. The cutoff was chosen after seeing those scores, so 174 more skippable grids saved from live runs were scored as a check. There, 161 scored 0.9 or higher and 11 scored below 0.7, but the gap was not empty: four grids scored 0.51, 0.63, 0.79 and 0.89. All 9 that Google had passed with squares selected scored 0.9 or higher. The check would have changed the answer on 3 grids, all bicycle prompts, which scored 0.32, 0.51 and 0.63. A person then labelled those three and the two at 0.79 and 0.89 without seeing any picks: the three show no bicycle and the other two do show the target. Across both sets, grids without the target scored 0.63 or lower and grids with it 0.79 or higher, so the 0.7 cutoff has less room than the labelled set alone suggests. The request adds about 300 input tokens and one more copy of the grid to the round, and the median replay time went from 0.95 to 0.98 seconds. Each round logs the score in `vllm_components.presence`.

On 4 × 4 grids vLLM also checks its selection against boxes, since most remaining errors are squares selected without the target in them. d3-ultra is asked to draw boxes around the target: nine on the whole grid, for the largest, second and third largest, leftmost, rightmost, topmost, bottommost, nearest and farthest one, and one in each of sixteen windows, a square together with the squares around it. As first measured, a selected square that scored under 95% was dropped when no box with a confidence of at least 0.3, from either request, covered 1% or more of it. Each effort level now sets the rule's thresholds and which of the two requests it sends, as described under [Effort levels on 4 × 4 grids](#effort-levels-on-4--4-grids); the measurements in this section and the next three paragraphs are of the rule as first measured. Boxes only ever remove selections. The rule applies to the logistic rule only. Each round logs the squares at risk, the boxes it used and the squares removed in `vllm_components.boxes`, and the square scores are reported unchanged. The accuracy tables, solve times and request counts elsewhere in this README were measured before the rule existed.

The rule was measured twice. A live run saved 333 new 4 × 4 grids, and the rule changed the selection on 28 of them. A person graded those 28 on a page that showed the rule's picks and flagged the removed squares, so the grading was not blind. Of the 29 squares removed, 25 were ruled empty, 4 a sliver and none the target. That fixed 21 grids and broke none, and three stayed wrong because another wrongly selected square remained. On the 311 grids labelled before the rule existed it fixed 10 grids and broke 4, removing 13 empty squares, 12 slivers and 4 targets. A confidence of 0.2 fixed 9 and broke 1 there and fixed 19 and broke none on the new grids. The earlier set favours 0.2, which was chosen after looking at it, and the new set favours 0.3, whose picks its grading page showed. Google's verdicts have not been compared with the rule on and off: in the run that saved the new grids, challenges were already solved 215 times of 217, so the rule can only reduce rejections along the way.

d3-ultra's boxes shift with the bytes it is sent. Asking for the sixteen windows in two or more requests instead of one changed the rule's result on 1 or 2 of 60 grids, and so did a JPEG without chroma subsampling. The rule was measured with pictures that Python cut, enlarged and saved with Pillow, so the server reproduces those exactly: `src/pillow.rs` holds Pillow's bilinear resize and libjpeg's baseline encoder, and its output matched Pillow byte for byte on all 5,661 pictures of the 333 new grids. The questions also go in the order they were measured in. Sent back through the server, all 333 grids gave the same boxes from both requests as the Python had, and the same 29 squares were removed on the same 28 grids.

Both box requests leave with the round's scoring requests, each on its own thread and connection. The whole-grid request is about 100 KB and 1,400 input tokens, and the window request about 460 KB and 4,400, on top of the 1.07 MB and 10,900 tokens of the scoring requests. As first measured, the round answered as soon as its selection could no longer change: at once when no selected square scored under 95%, and otherwise when a box had reached every such square or both replies were in. Of the 333 grids, 119 had no such square, the first box reply to arrive settled 165, and 49 used both. How long each effort level's rule waits is under [Effort levels on 4 × 4 grids](#effort-levels-on-4--4-grids). As first measured, a square was dropped only once both replies were in and complete. Now a square needs a box from every box request its level sends, so it is dropped as soon as one complete reply holds no box that reaches it, and a failed request leaves the squares it would have decided selected, with the error logged. `--vllm-box-wait` makes every round wait for every box reply and log every box, for checking the rule. The two requests add about half again to a round's upload, so their cost depends on the uplink. It was measured only on a slow one, 16 Mbps on a day when a 4 × 4 round took 2.5 seconds with the rule off: 50 saved grids sent in turn to a server with the rule off and one with it on took a median 0.6 seconds longer with it on. The cost on a fast connection has not been measured.

### Effort levels on 4 × 4 grids

`--effort` sets how much vLLM asks d3-ultra about a 4 × 4 grid. Each level sends everything the level below it sends and something more, so it costs more tokens. Every request of a round still leaves at once. The default is `max`.

| Level | A round sends | Input tokens | Right of 339 | Right of the 311 labelled |
| --- | --- | ---: | ---: | ---: |
| `low` | the whole grid, one close-up pass on a 506-pixel copy, the skip question | 6,475 | 243 (71.7%) | 233 (74.9%) |
| `medium` | low, and nine boxes drawn on the whole grid | 7,915 | 274 (80.8%) | 250 (80.4%) |
| `high` | medium, and a second close-up pass on the grid as captured | 12,351 | 281 (82.9%) | 257 (82.6%) |
| `max` | high, and a box drawn in a window around every square | 16,755 | 282 (83.2%) | 258 (83.0%) |

High scored within one grid of max for 4,400 fewer tokens, and it answers sooner, so `--effort high` is the cheaper choice if the difference does not show up live. Tokens are for a prompt that allows skipping; without the skip question a round uses about 300 fewer.

Each level decides with settings of its own:

- **Low** selects squares from 55%, because one pass scores a little high. It draws no boxes, so a square scored under 60% is dropped when its close-up score is more than 30 points above its whole-grid score: only the close-up believes in it.
- **Medium** selects from 55%. A selected square under 95% needs a whole-grid box of confidence 0.3 or more that covers 1% of it.
- **High** selects from 50%, with medium's box rule, and at least 1% of the box must lie inside the square.
- **Max** selects from 50%. A selected square under 95% needs a box from each request: a whole-grid box of confidence 0.02 or more and a window box of 0.5 or more.

All four ask the skip question on prompts that allow skipping and select nothing under 0.7.

These settings are the best-scoring ones found for each level's requests on 339 graded grids: the 311 labelled square by square and the 28 live grids graded afterwards, after a stricter correction of 42 of them on 2026-11-17, mostly edge slivers re-marked as empty. The search covered about 18 million combinations of which close-up scores to use, the box thresholds, the selection cutoff, the exemption, a minimum whole-grid score, the gap rule and the skip threshold. The scores above are on those same grids, so they flatter the settings. Selecting from 55% on one pass is the one change that other grids also chose, in nearly every left-out group, for about 5 grids at either level; the gains at high and max are within a few grids. The settings in use before the levels (both passes, select from 50%, a box of confidence 0.3 from either request) scored 271 on the same grids, and no setting of that either-request rule reached high, which is why max asks for both. Commit `815afc4` rebuilds the earlier settings. A refit of the logistic weights on the corrected labels was no better at any level. No level has been compared on Google's verdicts yet.

A round ends when its scoring requests are back and its box rule has nothing left to decide. Max waits longest, because a square that the whole-grid boxes reach still needs a window box, and the window request is the slowest of a round: about 0.5 seconds of compute against 0.26 for a close-up request. On the request times logged for the 333 replayed live rounds, counting only the provider's queue and compute time, max needed the window reply in 194 rounds and waited a median of 285 milliseconds for it after the scoring requests, taking the median round from 282 to 556 milliseconds. The settings before the levels waited in 52 rounds. High and medium needed a box reply after scoring in 1 round each, and low never waits. On that run's own network, which was slow enough for the close-up uploads to dominate, max waited in 51 rounds and its median round was 65 milliseconds longer than high's. These are estimates from one replay; no level has been timed live.

The 28 live grids are the ones where the first box rule removed a selection, so a level without boxes scores poorly on them by construction; the 311 are the better guide to a level's accuracy.

The settings live in `src/captcha/recaptcha/vllm/d3_ultra.rs` and belong to vLLM/d3-ultra: another model scores the same pictures differently, so it needs a settings file of its own, in a folder named after its provider, and vLLM refuses a `--model` that has none. `--effort` applies to the logistic rule. `--vllm-decision average` keeps one plan, with both close-up passes and the skip question.

`cargo test` decides d3-ultra's recorded answers on 644 grids again under each level's plan and compares the selections with the search's own, so a level cannot drift from what was measured. The answers are in `tests/fixtures/vllm_d3_ultra_4x4.json.gz`, without pictures or labels. Twenty rounds were also sent through the built server to d3-ultra, four grids at each level and four more chosen so that the gap rule and the inside share each removed a square: every score matched the recording exactly, every selection matched, and rounds with the skip question used 6,479 to 6,486, 7,901 to 7,919, 12,341 to 12,363 and 16,713 to 16,783 input tokens.

`/config` reports the level as `effort` and its settings as `plan_4x4`. Each round logs `effort`, every pass's scores, and the squares each rule removed in `vllm_components.gap` and `vllm_components.boxes`. A round run with `--effort max --vllm-box-wait` logs everything any level uses, so it can be decided again offline under each of them.

### Liquid

`--model Liquid/d1.2` or `--model Liquid/d1.2-ultra` asks one of Liquid's decision models. Put the key in `.env.liquid` as `LIQUID_API_KEY`; the file is ignored by git.

```sh
source .env.liquid
cargo run --release --locked -- serve --mode v2-enterprise --port 8777 \
  --model Liquid/d1.2-ultra --log reports/liquid.jsonl
```

A Liquid model is asked about every tile twice, on the whole grid and on close-ups, as vLLM/d3-ultra is, so the `--vllm-*` flags and the `vllm_components` log field apply to it under those names. Its settings are in `src/captcha/recaptcha/liquid/`, one file a model, and none of d3-ultra's numbers is used. Each model has its own fitted weights for 4 × 4 grids and its own effort levels. Its 3 × 3 grids have not been measured: their two views are averaged with equal weight and selected from 60%.

Each model's answers were recorded three times over the 339 graded 4 × 4 grids, since it does not answer a request the same way twice, and the figures are averages over the three passes:

| Level | Liquid/d1.2-ultra sends | Tokens | Right of 339 | Liquid/d1.2 sends | Tokens | Right of 339 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `low` | whole grid, one close-up pass at 506 pixels, skip question | 7,299 | 232.0 (68%) | whole grid, one close-up pass on the capture, skip question | 6,777 | 199.7 (59%) |
| `medium` | low, and one question for a box around every instance on the whole grid | 8,668 | 242.0 (71%) | low, and one question for a box around every instance on the whole grid | 8,146 | 236.0 (70%) |
| `high` | medium, and a second close-up pass | 14,051 | 245.0 (72%) | what medium sends | 8,146 | 236.0 (70%) |
| `max` | what high sends | 14,051 | 245.0 (72%) | what medium sends | 8,146 | 236.0 (70%) |

vLLM/d3-ultra scores 243, 274, 281 and 282 on the same grids. The labels lean its way: squares were ruled on one at a time where its picks disagreed with the labels, and Liquid selects squares that were never ruled on. Counting every unruled sliver as empty, d1.2-ultra's levels score 188, 203 and 208.

- **The weights matter most.** With the two views averaged equally the models scored 190 and 132. The whole grid is their weak view: alone it answers 117 and 62 grids right, the close-ups 221 and 199. The fitted weights, which lean almost wholly on the close-ups, score 232 and 200, and about the same when fitted on other grids.
- **Boxes help, windows do not.** A selected square under 90% needs a whole-grid box of confidence 0.3 or more. That is worth 10 grids for d1.2-ultra and 36 for d1.2, which then selects from 20%. A box in a window around every square costs Liquid sixteen requests and 5,076 tokens and added nothing for either model, so no level sends it.
- **One box question, not nine.** vLLM/d3-ultra draws one box a question, so it is asked nine times: for the largest, the leftmost and so on. Liquid has a question that draws a box around every instance (`boxes`, here "every motorcycle" with at most 16), and its levels ask that once. On the same three passes the nine questions, counted from confidence 0.8, scored 242.3, 245.0 and 232.7 where the one question scores 242.0, 245.0 and 236.0: differences of −0.3 (−7 to +6), 0.0 (−7 to +8) and +3.3 (−3 to +10) grids. The one question costs 1,369 input tokens against 1,778. It draws 1.9 boxes a grid, and about two in five of them are under the 0.8 the nine were counted from, so they count from 0.3: 0.05 to 0.4 scored alike.
- **A second close-up pass** is worth about 3 grids for d1.2-ultra (−4 to +10) and nothing for d1.2.
- **The settings are from plateaus.** The sweep covered 166 million settings a model. The same level answers 35 to 79 of the 339 grids differently from one pass to the next, so each level takes the middle of a run of settings that scored alike, with one weight vector a model and no inside share, minimum whole-grid score or gap rule: none of those held up on grids left out of the choosing.

`cargo test` decides each model's first recorded pass again at every level and compares the selections with the sweep's (`tests/fixtures/liquid_*_4x4.json.gz`). At the final settings 34 live rounds were sent through the server, at medium and high for d1.2-ultra and medium for d1.2: each sent the requests its level names, the box request was billed 1,368 to 1,370 tokens, and the selection was what the round's own logged scores and boxes give. No level has been run against Google.

Liquid's endpoint answers the same typed questions, with differences that were each found by sending it a request:

- Pictures go in an ordered `images` list and are referred to by position. The named `assets` the other two providers take are ignored without an error, and the request is answered from its text alone: a grid sent that way was billed 51 input tokens. So every request is rewritten in `src/liquid.rs`, and a reply billed fewer tokens than its pictures cost (1.5 a 32-pixel patch) is refused.
- A compressed request body is refused, so nothing is compressed.
- A request holds one state and at most eight pictures. The sixteen window pictures of a box round go in sixteen requests, and a close-up request carries the scene and at most seven close-ups.
- A reply carries no timings, so `provider_meta` holds only what the lab measures itself.
- The same request is not answered the same way twice. Sent six times, a whole-grid question's score moved by a median of 0.03 on d1.2 and 0.005 on d1.2-ultra, and 7 and 2 of 64 questions crossed 50%. So a recorded round cannot be checked by sending it again, and comparisons between settings carry that noise.

No request is sent twice on a timer, because a duplicate would be answered differently and Liquid counts every request against one limit for the organisation. A request refused as busy is sent again, after the `Retry-After` it names, up to three times. Forty requests at once were all answered. Connections are not opened ahead of a round.

`--record-4x4` asks a 4 × 4 round everything any effort level asks, waits for every reply and logs it, and decides as a plain round does: both close-up passes, the skip question, nine boxes on the whole grid, a box in each window and, on Liquid, the question for a box around every instance, none of which changes the answer. That is 27 or 28 requests and about 20,900 input tokens on Liquid, 8,200 of them for the box requests. It is how a model's answers are recorded so that settings can be measured for it.

Skippable prompts are where live 4 × 4 failures come from. In 550 live attempts, challenges that ended on a 4 × 4 grid without the skip line were solved 72 times of 74. Challenges with the skip line ran two or five grids in a row. The two-grid ones were solved 8 times of 15 and the five-grid ones once in 29. In older runs, five-grid sequences failed even when the first two grids were answered right by the labels, so their failures are not explained by wrong answers alone.

Dynamic 3 × 3 challenges, the ones that say "click verify once there are none left", fail for a different reason than wrong answers. Across 824 live attempts they were solved 34 times of 131, and nearly every rejection was the generic "Please try again". A person labelled every round of 36 of them without seeing the model's picks or the outcome: 24 drawn from the rejected challenges and 12 from the solved ones. The model had every round right in 17 of the 24 rejected challenges and in 9 of the 12 solved ones, so a right answer was rejected about as often as a wrong one. Weighted by how many of each kind there were, about 72% of dynamic challenges were answered right in every round, against 26% solved. Better tile accuracy would therefore change little here. Across the 123 labelled rounds the model picked 166 photos that held the target, left 5 such photos unpicked and picked 6 that did not.

A later run of 200 attempts let each attempt keep answering after a rejection, with `--max-stages 10`. Of those, 28 passed on the checkbox alone and 3 never showed a grid. Of the 169 that were challenged, 160 eventually passed, 104 of them on the first answer. Seven used all ten answers, every one of them inside a run of 4 × 4 skip grids, and 2 ended on a runner error or timeout. A solved challenge took a median of 2.7 seconds from the checkbox click to the pass, and one in ten took over 34.8 seconds.

| First challenge shown | Challenged | Eventually solved | Solved on the first answer | Median time to pass |
|---|---:|---:|---:|---:|
| Static 3 × 3 | 79 | 79 | 75 | 2.3 s |
| 4 × 4 | 51 | 45 | 26 | 3.0 s |
| Dynamic 3 × 3 | 39 | 36 | 3 | 32.5 s |

Google judges a 4 × 4 skip challenge as several grids with one verdict at the end, usually five, so the first-answer column undercounts it: the widget moves from one grid to the next by itself, about 20 milliseconds after the click, and asks Google only after the last. Counted per answer that Google judged, across that run and two later ones, static 3 × 3 grids passed 178 of 184 times, 4 × 4 grids without a skip line 82 of 94, dynamic challenges 77 of 160 and skip challenges 17 of 65, of which the five-grid ones 11 of 47. Here an answer counts as judged when the next grid or message came at least 60 milliseconds after the click. A rejection is not evidence of a wrong answer: on labelled dynamic challenges Google rejected fully right answers about as often as wrong ones. The runner logged Google's error message for only 63 of the 149 rejections, because it moved on as soon as the next grid appeared and the message tends to appear a moment later. When this run was made `--max-stages` counted each submitted grid, so two skip challenges used up all ten. It now counts Google's verdicts, which has not been run live yet.

All 43 solves that took 20 seconds or more included dynamic rounds, and the 112 solves without one took at most 6.2 seconds. Half the attempts used the earlier browser pacing, a half-second pause and a page check every 200 to 300 milliseconds, and half the current one. First-answer passes on a static 3 × 3 grid took a median of 2.6 seconds with the earlier pacing and 1.7 with the current: 0.74 seconds for Google to load the grid, 0.51 for the model, 0.12 to click the tiles and 0.16 for Google's verdict. On a 4 × 4 grid they took 2.8 and 1.9 seconds. The current pacing solved 82 of 86 challenged attempts and the earlier one 78 of 83, and the first answer was accepted 50 and 54 times, differences this run cannot tell from chance. Google challenged more as the run went on: 24 of the first 40 attempts passed on the checkbox alone and none of the last 120.

The slow part of a dynamic challenge is a wait that Google sets. When a tile is clicked the widget fetches its replacement at once, a median of 23 milliseconds after the last click in 56 logged tiles, but it shows the new tile, and lets it be clicked, only after a timer of one second plus a number that arrives with the challenge. In 225 recorded rounds the tiles appeared about 1.04 seconds after the last click in 9 and about 4.1 seconds after it in the rest, and 72 of 73 attempts with several such waits kept one speed throughout. Among passed attempts, 37 of the 45 with slow tiles had `TOO_MUCH_TRAFFIC` among Google's reasons, and none of the 3 with fast tiles, the 112 without dynamic rounds or the 28 checkbox passes did. The harness does not change that timer.

The runner uses the gap before the new tiles appear, unless it is started with `--no-early-tiles`. It listens to the browser's own image responses and requests nothing itself. It sends each fetched tile and its tile's place to the lab server with the previous screenshot, as the `tiles` field of a `/model` request. The server draws the tiles in, asks the model while the widget is still waiting, and returns the grid it showed the model as `grid_png`. When the tiles appear it checks that each went where it assumed and clicks at once. It takes a fresh screenshot instead when a tile went elsewhere, when a picture has not loaded, or when any score is within 0.25 of the cutoff, because a drawn-in tile differs slightly from the browser's rendering and a close score can flip. On 59 recorded rounds the drawn-in grid got the same picks as the real screenshot in 57, and in all 53 that the margin lets through. The margin was chosen on the first 29 of those rounds, when the runner drew the tiles itself with Pillow (58 of 59 the same, 52 let through), and was kept unchanged for the server's drawing. The two drawings differ by under half a brightness level on average. That moved 5 of 531 scores by more than 0.03, the most by 0.2, which is why the margin stays. The margin lowers the chance of a wrong early pick and does not rule it out: one server-drawn tile scored 0.13 where the real screenshot scored 0.45, a gap wider than the margin, though on the same side of the cutoff.

In a live run of 182 attempts, with the order of the two settings drawn at random, the next click followed the new tiles by a median of 0.06 seconds with early answers and 0.77 without. Of 90 early answers 84 were used and 6 gave way to a fresh screenshot over a close score. A dynamic challenge of three rounds took a median of 9.1 seconds from its first grid to Verify against 10.4, and one of four rounds 13.6 against 15.4. The wait Google sets is the same in both. Early answers solved 82 of 83 challenged attempts and the plain path 83 of 85, and too few dynamic answers were submitted to compare how often Google accepted them. Two copies of that run overlapped by mistake, so its attempts ran two at a time, and it was made while the runner still drew the tiles itself. A later run of 30 attempts with the server's drawing used 30 early answers and fell back on 7, with no failed requests. The option is off by default. `--log-tile-network` records when each of the widget's image requests starts and finishes and saves the replacement tiles.

Google serves 4 × 4 images at 450 × 450 pixels, while a default screenshot of the grid is about 388 pixels across. To test sharper input, run the browser runner with `--device-scale-factor 2` to capture grids at twice the resolution. Then start the lab with `--grid-max-side-4x4 467 --grid-max-side-3x3 389` to shrink each grid before the model sees it. The model log records the size used in `grid_pixels`. Scale 2 fixes the browser window at 1280 × 800. On 63 labelled grids, sizes from 389 to 778 pixels gave the same accuracy within noise as a single pass, and a second pass resized from the default capture gained as much as one from a sharper capture, so both options are off by default.

Use `--vllm-decision average` for the previous rule, which averages the two probabilities and selects tiles at or above 60% on 3 × 3 grids and 50% on 4 × 4 grids. Add `--vllm-whole-grid-weight-3x3 0.60 --vllm-whole-grid-weight-4x4 0.35` to test other whole-grid weights, with the remaining weight going to the crop scores. Each flag defaults to `0.5` and implies the average rule. The model log records the weight applied to each round.

Google rejects some correct answers in sessions it rates as high risk. That is most visible on 3 × 3 challenges that replace tiles, where the saved grids show correct selections followed by "Please try again". Compare decision rules on labelled grids as well as live pass rates.

### Manual image and score trials

For **v2**, Google decides whether the checkbox opens an image challenge. When an image challenge appears, copy its instructions and capture the grid as a cropped PNG between 280 and 1568 pixels on each side. Use the model panel in a separate tab while the challenge is open. Select the returned tiles in the Google widget, capture again if Google replaces tiles, and record the observed outcome on the original tab. A failed or abandoned challenge may never issue a token, so record that outcome before resetting. Use the checkbox-only outcome for trials where Google let the interaction through without an image challenge.

For **v3**, click **Request assessment**. The Rust server immediately verifies the Google token. When a model is configured, it also gives the model timing and input-event counts from the lab page. The Google score and model estimate measure different things. The Google score is the result for that browser interaction. The model estimate is a classification based on the limited telemetry provided. [Google v3 scoring](https://developers.google.com/recaptcha/docs/v3)

The JSON Lines log records the trial ID, Google mode, browser source, and result of each event. Match events by `trial_id`. Google and model calls each include latency. Model events include the resolved model and usage when the provider returns them. Screenshots, raw telemetry, tokens, and API keys are omitted from the log. Keep human baselines and automated trials in distinct log files when comparing them.

## Camoufox

With the Rust server running, install Camoufox's browser build and launch a visible browser. `--url` can target a different localhost port.

```sh
uv run --with camoufox==0.5.9 python -m camoufox fetch
uv run --with camoufox==0.5.9 scripts/camoufox_lab.py
```

Use `--smoke` to click and verify the public v2 test widget. For v3 or v3 Enterprise, use `--score` to run one automated score trial and print its result.

```sh
uv run --with camoufox==0.5.9 scripts/camoufox_lab.py --smoke
uv run --with camoufox==0.5.9 scripts/camoufox_lab.py --score
```

For manual v2 trials, interact with Google's checkbox and challenge in the visible browser. For automated v2 trials, run the shared checkbox runner with `--browser camoufox`.

```sh
uv run --with camoufox==0.5.9 python scripts/chrome_lab.py --browser camoufox \
  --submit --trials 5 --output reports/camoufox
```

Events from both Camoufox workflows are marked `camoufox` in the log. [Camoufox installation](https://camoufox.com/python/installation/)

## hCaptcha

Only hCaptcha's grid task is handled: nine separate pictures and one instruction, such as "Pick the flying animal". hCaptcha also serves tasks of other kinds, such as dragging a piece, clicking a point or following a moving object. The runner passes over those with the widget's own refresh button and counts them apart from wrong answers.

Create a sitekey of your own and set it to **Always Challenge**, because hCaptcha's public test keys never show a challenge. Put `HCAPTCHA_SITE_KEY` and `HCAPTCHA_SECRET_KEY` in the server's environment. hCaptcha refuses pages on `localhost` and `127.0.0.1`, so the lab serves its hCaptcha page as `hypercaptcha.localhost`, a name browsers resolve to this machine without a hosts entry. That name has been checked with the public test key only. If hCaptcha refuses it for a live key, add `127.0.0.1 hypercaptcha.test` to your hosts file, as hCaptcha's documentation suggests, and start the lab with `--hostname hypercaptcha.test`. [hCaptcha local development](https://docs.hcaptcha.com/#local-development) · [hCaptcha test keys](https://docs.hcaptcha.com/#integration-testing-test-keys)

```sh
cargo run --locked -- serve --captcha hcaptcha --hcaptcha-test-key
cargo run --release --locked -- serve --captcha hcaptcha --provider vllm \
  --log reports/hcaptcha.jsonl
```

The first command is the integration check with hCaptcha's public keys. With the second running, capture grids without answering them, then answer them:

```sh
uv run --with playwright==1.62.0 scripts/hcaptcha_lab.py --trials 3
uv run --with playwright==1.62.0 scripts/hcaptcha_lab.py --trials 3 --submit --max-stages 10
```

hCaptcha draws its pictures on a canvas, so the page has no element per picture. The runner sends a screenshot of the canvas below the instruction to the lab, which finds the nine pictures and returns where each one is (`src/captcha/hcaptcha/grid.rs`). hCaptcha nudges and tilts each picture, so the lab finds the block the nine form on their panel, divides it into thirds, and takes it for a grid only if the lines between the thirds are mostly bare panel. That found the grid in all six real screenshots saved so far, and it is how a grid is told from hCaptcha's other tasks. The lab sets the pictures side by side as one 3 × 3 grid and sends d3-ultra four requests at once: the whole grid with the full instruction, and the nine pictures by themselves in three groups of three. The runner clicks the middle of each selected picture and then the widget's button. A challenge has several pages and hCaptcha judges it once, after the last. `--max-stages` keeps a trial going after a rejection, as it does for reCAPTCHA.

Nothing in hCaptcha's profile has been fitted. There are no labelled hCaptcha grids, so the two views of a picture are averaged with equal weight and the cutoff is one half. The logistic rules, the skip check, the second close-up pass and the box rule stay off: they were fitted on Google's street photos and assume one photo cut into squares, whereas each hCaptcha tile is a picture of its own. The pass rate on live challenges has not been measured yet. An example picture shown beside the instruction is not given to the model; the runner records how many there were.
