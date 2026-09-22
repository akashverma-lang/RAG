#!/usr/bin/env bash
# Local RAG - one-line install for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/akashverma-lang/RAG/main/install.sh | bash
#
# Downloads the app, sets up a private Python environment and starts it. Everything
# lands in one folder that can be deleted to uninstall.
set -euo pipefail

# ---- edit these to your repository ----------------------------------------------
OWNER="${RAG_OWNER:-akashverma-lang}"
REPO="${RAG_REPO:-RAG}"
BRANCH="${RAG_BRANCH:-main}"
# ---------------------------------------------------------------------------------

TARGET="${RAG_HOME:-$HOME/LocalRAG}"

say()  { printf '  %s\n' "$1"; }
step() { printf '\n>> %s\n' "$1"; }

echo
echo "=============================================="
echo "  Local RAG - installing"
echo "=============================================="

step "Looking for Python"
PY=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")
        major=${v%%.*}; minor=${v##*.}
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then PY="$c"; break; fi
        say "found Python $v, which is too old"
    fi
done
if [ -z "$PY" ]; then
    echo
    echo "  Python 3.10 or newer is required."
    echo "    macOS:  brew install python@3.12"
    echo "    Ubuntu: sudo apt install python3 python3-venv python3-pip"
    echo "  Then run this command again."
    exit 1
fi
say "using $($PY --version)"

step "Downloading Local RAG"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
URL="https://github.com/$OWNER/$REPO/archive/refs/heads/$BRANCH.tar.gz"
if ! curl -fsSL "$URL" -o "$TMP/app.tar.gz"; then
    echo "  Could not download $URL"
    echo "  Check the repository name, or that it is public."
    exit 1
fi
mkdir -p "$TARGET"
# --strip-components drops the owner-repo-branch wrapper directory GitHub adds.
tar -xzf "$TMP/app.tar.gz" -C "$TARGET" --strip-components=1
say "installed to $TARGET"

step "Setting up a private Python environment"
cd "$TARGET"
[ -x ".venv/bin/python" ] || "$PY" -m venv .venv
say "installing packages - this takes a few minutes the first time"
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt --quiet

# A small launcher, so it can be started again without retyping any of this.
cat > "$TARGET/start.sh" <<'LAUNCH'
#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/python launcher.py
LAUNCH
chmod +x "$TARGET/start.sh"

echo
echo "=============================================="
echo "  Installed. Starting it now."
echo "=============================================="
echo
echo "  A setup screen will ask for your documents folder"
echo "  and a free API key:"
echo "     Groq    console.groq.com/keys"
echo "     Gemini  aistudio.google.com/apikey"
echo
echo "  Start it again later with:  $TARGET/start.sh"
echo

exec .venv/bin/python launcher.py
