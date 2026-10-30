# Jarvoos

A stripped-down terminal agent in the spirit of Hermes / Claude Code, built for a cheap setup:
a **local model first** (Strata, llama.cpp, anything OpenAI-compatible), a **cloud fallback** (DeepSeek via
OpenRouter) that takes over the moment the local server fails, web search (Exa, optionally SearXNG),
and markdown **skill files** loaded on demand.

## Install (private repo)

You need a GitHub login that can read the repo (`gh auth login`), `git`, and Python 3.11+:

```bash
gh api repos/ms95ir-ctrl/jarvoos/contents/install.sh -H "Accept: application/vnd.github.raw" | bash
jarvoos setup     # endpoints, models, API keys
jarvoos           # chat
```

Update later with `jarvoos update`.

## Use

```
jarvoos                  interactive chat (history, /help, /clear, /think, /model, /events, /exit)
jarvoos -z "question"    one shot: print the answer and exit
jarvoos setup [primary|fallback|search]
jarvoos doctor           test endpoints and keys
```

Reading files, listing folders, web search and page fetches run freely. Writing files and running shell
commands ask first (`y` / `n` / `a` = always this session). `--auto-approve` skips the questions; use it only
in a throwaway environment.

## Failover

Every request goes to the primary first. Connection refused, an HTTP error, a stalled stream (no data
within a timeout that grows with prompt size, because local prefill is slow), or an empty answer sends the
same request to the fallback immediately. The primary is then skipped for `cooldown` seconds (default 300)
and probed again afterwards. Each switch and each recovery is written to `~/.local/state/jarvoos/events.log`
(`/events` shows the latest); notifier hooks for Telegram are planned.

## Files and secrets

| Path | Contents |
|---|---|
| `~/.config/jarvoos/config.toml` | endpoints, models, timeouts |
| `~/.config/jarvoos/secrets.env` | API keys, mode 600; environment variables override it |
| `~/.config/jarvoos/skills/*.md` | your skills (`name` / `description` frontmatter); same name overrides a bundled one |

Keys are never copied into the environment of tools the agent runs, and shell tools see an environment with
`*KEY*`, `*TOKEN*`, `*SECRET*` and `*PASSWORD*` variables removed.

## Safety notes

- Search results and fetched pages are marked as untrusted data in the tool output, and the system prompt
  tells the model never to act on instructions found there.
- `fetch_url` refuses private and loopback addresses unless `tools.allow_private_fetch = true`.
- The fallback sends your conversation to OpenRouter. Leave it unconfigured if that is not acceptable.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest
```

Not built yet: Telegram client and debug bots, tool search/installer, speech, spend cap.
