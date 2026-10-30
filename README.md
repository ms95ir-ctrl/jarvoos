# Jarvoos

A terminal agent in the spirit of Hermes and Goose, built for a cheap setup: a **local model first**
(Strata, llama-server, Ollama, LM Studio, any OpenAI-compatible server), a **cloud fallback** (OpenRouter) that
takes over the moment the local server fails, keyless web search, and markdown **skill files**.

## Install

You need `git` and Python 3.11+. Then:

```bash
curl -fsSL https://raw.githubusercontent.com/ms95ir-ctrl/jarvoos/main/install.sh | bash
```

The installer ends by offering to run `jarvoos setup` (answer `n` to skip, or set `JARVOOS_NO_SETUP=1`).
Update later with `jarvoos update`.

## First run

Just type `jarvoos`. If nothing is configured it offers to run setup, then drops you into the chat.
Setup takes about a minute and writes nothing until you confirm the summary (Ctrl+C at any point = nothing saved):

1. **Looking for local servers.** Probes localhost ports 8090 (Strata), 8080/8081 (llama-server), 11434 (Ollama)
   and 1234 (LM Studio) in parallel, plus any hosts in `JARVOOS_DETECT_HOSTS` (comma separated, `host`, `host:port` or URL).
2. **Pick a backend.** Detected servers come first and are pre-selected. Otherwise you get presets for Strata,
   llama-server, Ollama, LM Studio, OpenRouter and a custom URL, each with a default address.
3. **Local server.** The URL is checked (`/v1` is added or removed automatically if needed); "connection refused" and
   "reachable but 401/404" are told apart; a model that is still loading is waited for (up to 30 s). If a key is needed
   it is asked for, hidden, and stored under a per-server variable such as `JARVOOS_KEY_127_0_0_1_8090`.
   The model id and context size are read from the server. Thinking mode (default off) is asked for Strata and llama-server.
4. **OpenRouter.** Get a key at https://openrouter.ai/keys and paste it (stray whitespace and paste markers are stripped).
   It is validated, then you pick a model from a searchable list showing context and $ per million tokens
   (only models that support tools; default `deepseek/deepseek-v4-flash`). A local choice then asks
   "Also add OpenRouter as fallback?".
5. **Real test.** Every provider gets a 1-token streaming chat (time to first token is shown) and a tiny tool call.
   If tools fail you can continue chat-only; if chat fails you see the exact URL tried and can retry, edit or skip.
6. **Permission mode**, asked once (see below).
7. **Summary and save.** Config is written atomically with mode 600 (the old one is kept as `config.toml.bak`),
   then a quick health check runs.

Change one part later: `jarvoos setup model | fallback | search | mode`. Check everything: `jarvoos doctor` (add `--fix`
to repair file permissions and missing folders). Doctor reports ok/warn/fail per check with the exact command that fixes it
and exits non-zero on any failure.

## Job radar (0.3)

Build your own job radar: a web page of scored postings that refreshes on a schedule.

```bash
jarvoos radar setup
```

Setup asks about your background, the work you want, places, salary and languages, proposes a search profile (you can correct it
in plain words), then offers to build your CV fact bank, connect Telegram and schedule updates. Without a working model it falls
back to your answers as typed and keyword scoring.

| Command | What it does |
|---|---|
| `jarvoos radar run` | fetch, score, notify, rewrite the page now (what cron runs) |
| `jarvoos radar serve` | live page at http://localhost:8800 with status buttons (interested, applied, interview...) |
| `jarvoos radar status` | counts, schedule, Telegram state |
| `jarvoos radar cron on --every 6` / `off` | manage the crontab block (only its own lines are touched) |
| `jarvoos radar telegram` | connect a bot: token from @BotFather, then send the bot any message |
| `jarvoos radar cv facts` | questions that build your CV fact bank (or import existing CV text during setup) |
| `jarvoos radar cv review` | what is vague, unverifiable or missing |
| `jarvoos radar cv build` | HTML and PDF (PDF needs chromium, weasyprint or wkhtmltopdf) |
| `jarvoos radar cv tailor ID` | re-orders your own bullets for one posting; it never rewrites or invents them |

Sources are free and keyless: arbeitnow (German searches) and remotive and remoteok (remote work). Postings are scored by your
model (local first, cloud fallback) against your profile and fact bank; most should score under 50. Strong matches (default 70+) go to
Telegram once. A static `index.html` is rewritten after every run, so cron alone keeps the page current. Files live in
`~/.local/share/jarvoos/radar` (`JARVOOS_RADAR` overrides). The Telegram token is stored in `secrets.env`, mode 600.

## Skills and their status

`jarvoos skills` (or `/skills` in chat) lists skills with a status: **tested** (done for real and checked), **experimental**
(a plan that has not been run here) or **untested** (user skills without a status). Bundled: `job-radar`, `openscad`, `build123d`,
`video-reel` (frames + voiceover + ffmpeg), `web-research` (tested) and `ai-media` (experimental: local AI image/video generation).
Add your own as markdown files in `~/.config/jarvoos/skills/` with `name`, `description` and optional `status` frontmatter.

## Pointing Jarvoos at a backend

Setup does this for you; by hand, the same settings go in `~/.config/jarvoos/config.toml` (see `config.example.toml`).

| Backend | Start it | `kind` | `base_url` |
|---|---|---|---|
| Strata | (your Strata install) | `strata` | `http://127.0.0.1:8090/v1` |
| llama-server | `llama-server -m model.gguf --jinja -c 32768 --port 8080` | `llamacpp` | `http://127.0.0.1:8080/v1` |
| OpenRouter | needs a key from openrouter.ai/keys | `openrouter` | `https://openrouter.ai/api/v1` |
| Ollama | `OLLAMA_CONTEXT_LENGTH=32768 ollama serve` | `ollama` | `http://127.0.0.1:11434/v1` |
| LM Studio | start the local server in the app | `lmstudio` | `http://127.0.0.1:1234/v1` |

- **llama-server needs `--jinja`**, otherwise tool calls do not work (setup and doctor tell you).
- **Ollama defaults to a 4096-token context**, far too small for an agent; raise it as above.
- **Strata** never validates the model name, so Jarvoos takes the id from `/v1/models`.
- The `chain` setting orders providers: first is the primary, the rest are fallbacks, e.g. `chain = ["local", "openrouter"]`.
  If the primary is unreachable, times out, or returns 5xx/429, the next provider takes over immediately and the chat shows it.

## Permission modes

| `mode` | Meaning |
|---|---|
| `approve` (default) | Reading, listing and searching run freely; writing files and running commands ask first |
| `auto` | Nothing asks (a hard-deny list such as `rm -rf /` still applies). Use in throwaway environments |
| `chat` | No tools at all, just conversation |

Set it in setup, with `jarvoos config set mode auto`, or per run with `--auto-approve`.

## Use

```
jarvoos                  interactive chat (/help lists the slash commands; Alt+Enter = newline)
jarvoos -c               continue the last session
jarvoos -z "question"    one shot: print the answer and exit
jarvoos setup [model|fallback|search|mode]
jarvoos doctor [--fix]   check install, config and every provider
jarvoos config get|set|path|edit
```

With no terminal (cron, CI) an unconfigured Jarvoos prints how to configure it by hand and exits 1.

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter key (wins over `secrets.env`) |
| `JARVOOS_KEY_<HOST>_<PORT>` | key for a local server that needs one, e.g. `JARVOOS_KEY_127_0_0_1_8090` |
| `EXA_API_KEY` | only if `search.provider = "exa"` |
| `JARVOOS_HOME` | config folder instead of `~/.config/jarvoos` (state then lives in `$JARVOOS_HOME/state`) |
| `JARVOOS_STATE` | state folder (sessions, history) |
| `JARVOOS_DETECT_HOSTS` | extra hosts or URLs for setup to scan, comma separated |

## Failover

Every request goes to the first provider in `chain`. Connection refused, a timeout, HTTP 5xx or 429, or a stalled
stream sends the request to the next provider immediately; the primary is skipped for a cooldown and retried later.
Each switch is announced in the chat and the toolbar marks that a fallback is serving.

## Files and secrets

| Path | Contents |
|---|---|
| `~/.config/jarvoos/config.toml` | providers, chain, mode, search (old `[primary]`/`[fallback]` files still load and are rewritten on the next save) |
| `~/.config/jarvoos/config.toml.bak` | the previous config, written before every save |
| `~/.config/jarvoos/secrets.env` | API keys, mode 600; environment variables override it |
| `~/.config/jarvoos/skills/*.md` | your skills (`name` / `description` frontmatter); same name overrides a bundled one |

Keys are never copied into the environment of tools the agent runs, and shell tools see an environment with
`*KEY*`, `*TOKEN*`, `*SECRET*` and `*PASSWORD*` variables removed.

## Safety notes

- Search results and fetched pages are marked as untrusted data in the tool output, and the system prompt
  tells the model never to act on instructions found there.
- `fetch_url` refuses private and loopback addresses unless allowed in the config.
- The fallback sends your conversation to OpenRouter. Leave it unconfigured if that is not acceptable.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest
```

Not built yet: Telegram client and debug bots, tool search/installer, speech, spend cap.
