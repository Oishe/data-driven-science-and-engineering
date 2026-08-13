# Default notebook for edit/run/check recipes.
nb := "notebooks/02_compression_via_sparsity.py"

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

# Lint + marimo formatting/pitfall checks on one notebook.
check nb=nb:
    uv run ruff check {{nb}}
    uv run marimo check --fix {{nb}}

# Non-mutating lint + tests across the repo (CI runs this exact recipe).
check-all:
    uv run ruff check .
    uv run marimo check notebooks
    uv run pytest -q

# Preview the landing page only at :8000 (no WASM apps — they 404).
serve:
    uv run python -m http.server -d docs 8000

# Full deploy pipeline into _site/ (landing + every notebook; CI runs this too).
build:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf _site
    mkdir -p _site
    cp -R docs/. _site/
    for f in notebooks/*.py; do
        name=$(basename "$f" .py)
        uvx marimo export html-wasm --sandbox "$f" -o "_site/app/$name" --mode run --no-show-code
        uvx marimo export html-wasm --sandbox "$f" -o "_site/notebook/$name" --mode run --show-code
        test -f "_site/app/$name/index.html" && test -f "_site/notebook/$name/index.html"
    done
    touch _site/.nojekyll
    test -f _site/index.html

# Serve it exactly as deployed at localhost:8000.
preview:
    uv run python -m http.server -d _site 8000

# Agent server
agent:
    npx stdio-to-ws "npx @zed-industries/claude-code-acp" --port 3017
