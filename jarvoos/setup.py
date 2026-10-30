"""Interactive setup wizard: endpoints, models and API keys. Keys go to secrets.env (0600)."""
from __future__ import annotations

import httpx
from rich.console import Console
from rich.prompt import Confirm, Prompt

from .config import CONFIG_FILE, SECRETS_FILE, get_secret, load_config, mask, save_config, set_secret

console = Console()


def list_models(base_url: str, api_key: str = "") -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=8)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def ask_key(label: str, env_name: str) -> None:
    current = get_secret(env_name)
    console.print(f"{label}: [dim]{mask(current)}[/dim]")
    value = Prompt.ask("  paste new key (Enter keeps the current one, '-' removes it)",
                       password=True, default="", show_default=False)
    if value == "-":
        set_secret(env_name, "")
    elif value:
        set_secret(env_name, value)


def setup_primary(cfg: dict) -> None:
    p = cfg["primary"]
    console.rule("Primary model (local server)")
    p["base_url"] = Prompt.ask("OpenAI-compatible base URL", default=p["base_url"])
    models: list[str] = []
    try:
        models = list_models(p["base_url"], get_secret(p["api_key_env"]))
        console.print(f"[green]reachable[/green], models: {', '.join(models[:5]) or '(none listed)'}")
    except Exception as e:
        console.print(f"[yellow]could not reach it ({e}); you can still continue[/yellow]")
    p["model"] = Prompt.ask("Model name", default=p["model"] or (models[0] if models else "local"))
    p["thinking"] = Prompt.ask("Thinking mode (off is faster; on reasons before answering)",
                               choices=["off", "on", "default"], default=p["thinking"])


def setup_fallback(cfg: dict) -> None:
    f = cfg["fallback"]
    console.rule("Fallback model (OpenRouter)")
    if not Confirm.ask("Configure a cloud fallback?", default=True):
        f["model"] = ""
        return
    ask_key("OpenRouter API key", f["api_key_env"])
    ids: list[str] = []
    try:
        ids = sorted(i for i in list_models(f["base_url"]) if "deepseek" in i.lower())
    except Exception:
        pass
    if ids:
        console.print("DeepSeek models on OpenRouter:")
        for n, i in enumerate(ids, 1):
            console.print(f"  {n:>2}. {i}")
    choice = Prompt.ask("Model id (or a number from the list)", default=f["model"] or "")
    if choice.isdigit() and 0 < int(choice) <= len(ids):
        choice = ids[int(choice) - 1]
    f["model"] = choice


def setup_search(cfg: dict) -> None:
    console.rule("Web search")
    ask_key("Exa API key (neural search)", "EXA_API_KEY")
    url = Prompt.ask("SearXNG URL for free search, optional (blank = none)", default=cfg["search"]["searxng_url"])
    cfg["search"]["searxng_url"] = url.strip()
    cfg["search"]["provider"] = "exa" if get_secret("EXA_API_KEY") else ("searxng" if url.strip() else "exa")


SECTIONS = {"primary": setup_primary, "fallback": setup_fallback, "search": setup_search}


def run(section: str | None = None) -> None:
    cfg = load_config()
    console.print("[bold]Jarvoos setup[/bold]  (Ctrl+C to abort; nothing is saved until the end)")
    for name in [section] if section else SECTIONS:
        SECTIONS[name](cfg)
    save_config(cfg)
    console.print(f"\nSaved settings to {CONFIG_FILE} and keys to {SECRETS_FILE} (mode 600).")
    console.print("Run [bold]jarvoos doctor[/bold] to test everything, or just [bold]jarvoos[/bold] to start.")
