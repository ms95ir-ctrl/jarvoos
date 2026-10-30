#!/usr/bin/env bash
# Jarvoos installer. Needs: git, python3 >= 3.11, and access to the private repo (GitHub login via `gh auth login`).
#   gh api repos/ms95ir-ctrl/jarvoos/contents/install.sh -H "Accept: application/vnd.github.raw" | bash
# Installs into ~/.local/share/jarvoos (code + venv) and links ~/.local/bin/jarvoos. No sudo, nothing else touched.
set -euo pipefail

REPO="${JARVOOS_REPO:-ms95ir-ctrl/jarvoos}"
DEST="${JARVOOS_DEST:-$HOME/.local/share/jarvoos}"
BIN="${JARVOOS_BIN:-$HOME/.local/bin}"

die() { echo "jarvoos installer: $*" >&2; exit 1; }
command -v git >/dev/null || die "git is required"
PY=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3 || true)
[ -n "$PY" ] || die "python3 is required"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "Python 3.11+ is required (found $("$PY" -V))"
"$PY" -c 'import venv, ensurepip' 2>/dev/null || die "python venv support is missing (Debian/Ubuntu: sudo apt install python3-venv)"

mkdir -p "$DEST" "$BIN"
if [ -d "$DEST/src/.git" ]; then
  echo "Updating existing checkout…"
  git -C "$DEST/src" pull --ff-only
else
  echo "Fetching $REPO…"
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh repo clone "$REPO" "$DEST/src" -- --quiet
  else
    git clone --quiet "https://github.com/$REPO.git" "$DEST/src"
  fi
fi

[ -d "$DEST/venv" ] || "$PY" -m venv "$DEST/venv"
"$DEST/venv/bin/pip" install --quiet "$DEST/src"
ln -sf "$DEST/venv/bin/jarvoos" "$BIN/jarvoos"

echo
echo "Installed: $BIN/jarvoos"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "Add $BIN to your PATH (e.g. echo 'export PATH=\"$BIN:\$PATH\"' >> ~/.bashrc)";; esac

# This script usually runs as `curl ... | bash`, so stdin is the script itself; questions go through /dev/tty.
if ( : </dev/tty ) 2>/dev/null && [ -z "${JARVOOS_NO_SETUP:-}" ]; then
  printf '\nSet up Jarvoos now (finds your local server, asks for an OpenRouter key, tests everything)? [Y/n] ' >/dev/tty
  read -r ans </dev/tty || ans=n
  case "${ans:-y}" in
    [Nn]*) echo "Run  jarvoos setup  whenever you are ready." ;;
    *) "$DEST/venv/bin/jarvoos" setup </dev/tty >/dev/tty 2>&1 && echo "Start chatting with:  jarvoos" || echo "Setup did not finish. Run  jarvoos setup  to try again." ;;
  esac
else
  echo "Next: run  jarvoos setup  then  jarvoos"
fi
