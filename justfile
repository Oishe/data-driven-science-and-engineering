# Default notebook for edit/run/check recipes.
nb := "notebooks/sparsity.py"

# Sandbox mode. Default off; flip on with `just sandbox=--sandbox <recipe>`.
sandbox := "--no-sandbox"

# Port for the LAN-facing share recipe.
port := "2718"

# List all recipes.
default:
    @just --list

# Edit a notebook, pairing-ready (--no-token). Defaults to sparsity.
edit nb=nb:
    uv run marimo edit --no-token {{sandbox}} --watch {{nb}}

# Run a notebook as a read-only app.
run nb=nb:
    uv run marimo run {{sandbox}} {{nb}}

# Binds 0.0.0.0 (marimo defaults to localhost, unreachable from other devices)
# and drops the token so the printed URL opens directly — trusted networks only.
# Serve a notebook read-only to other LAN devices, in sandbox mode.
share nb=nb:
    @echo "On another device, open:  http://$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname):{{port}}"
    uv run marimo edit --sandbox --no-token --host 0.0.0.0 --port {{port}} {{nb}}

# Lint + marimo formatting/pitfall checks.
check nb=nb:
    uv run ruff check {{nb}}
    uv run marimo check --fix {{nb}}

# Preview the landing page only (live reload; no WASM apps).
serve:
    uv run mkdocs serve

# Full deploy pipeline into _site/ (landing + every notebook; CI runs this too).
build:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf _site
    uv run mkdocs build -d _site
    for f in notebooks/*.py; do
        name=$(basename "$f" .py)
        uvx marimo export html-wasm --sandbox "$f" -o "_site/app/$name" --mode run --no-show-code
        uvx marimo export html-wasm --sandbox "$f" -o "_site/notebook/$name" --mode run --show-code
        test -f "_site/app/$name/index.html" && test -f "_site/notebook/$name/index.html"
    done
    touch _site/.nojekyll
    test -f _site/index.html

# Build the full site, then serve it exactly as deployed at localhost:8000.
preview: build
    uv run python -m http.server -d _site 8000

# Agent server
agent:
    npx stdio-to-ws "npx @zed-industries/claude-code-acp" --port 3017
