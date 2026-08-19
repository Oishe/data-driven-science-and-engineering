# Data-Driven Science &amp; Engineering

**Interactive notebooks on the mathematics behind modern data analysis. They run in your browser.**

[![Site](https://img.shields.io/badge/live-oishe.github.io-2f5fd0)](https://oishe.github.io/data-driven-science-and-engineering/)
[![Deploy](https://github.com/Oishe/data-driven-science-and-engineering/actions/workflows/deploy.yml/badge.svg)](https://github.com/Oishe/data-driven-science-and-engineering/actions/workflows/deploy.yml)
[![Checks](https://github.com/Oishe/data-driven-science-and-engineering/actions/workflows/check.yml/badge.svg)](https://github.com/Oishe/data-driven-science-and-engineering/actions/workflows/check.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Each notebook is a short chapter you can read, then take apart yourself. No install, no
server. [marimo](https://marimo.io) notebooks compile to WebAssembly and ship as static
files on GitHub Pages.

**→ [Start here: the landing page](https://oishe.github.io/data-driven-science-and-engineering/)**

![Sparsity Exploration: an image and its full FFT spectrum above; below, the reconstruction from the largest 5% of coefficients and the thinned spectrum it came from](docs/assets/hero.png)

---

## Chapters

Each chapter ships twice: a code-hidden **app** for exploring, and a code-visible
**notebook** that puts the implementation beside the derivation.

| # | Chapter | Explore | Read |
| --- | --- | --- | --- |
| 01 | **A Signal Is a Vector**, one dimension, small enough to see every number | [Launch app ▶](https://oishe.github.io/data-driven-science-and-engineering/app/01_signal_is_a_vector/) | [Notebook 📖](https://oishe.github.io/data-driven-science-and-engineering/notebook/01_signal_is_a_vector/) |
| 02 | **Compression via Sparsity**, the same move in 2-D, the idea behind JPEG and JPEG 2000 | [Launch app ▶](https://oishe.github.io/data-driven-science-and-engineering/app/02_compression_via_sparsity/) | [Notebook 📖](https://oishe.github.io/data-driven-science-and-engineering/notebook/02_compression_via_sparsity/) |
| 03 | **Sparsity Exploration**, chapter 02 stripped to the controls, for a quick look with no derivation | [Launch app ▶](https://oishe.github.io/data-driven-science-and-engineering/app/03_compression_simple_app/) | [Notebook 📖](https://oishe.github.io/data-driven-science-and-engineering/notebook/03_compression_simple_app/) |

More chapters are in progress, following the arc of *Data-Driven Science and Engineering:
Machine Learning, Dynamical Systems, and Control* by Steven L. Brunton and J. Nathan Kutz:
singular value decomposition, Fourier and wavelet analysis, regression and model selection,
and dynamical systems.

## The idea these chapters build

An image is *dense* in pixel space. Rotate it into an orthonormal basis (FFT, DCT, or
wavelet) and it becomes *sparse*: a few large coefficients hold almost all the energy.
Keep the largest, zero the rest, invert, and you have lossy compression. Parseval's
theorem guarantees that dropping the smallest coefficients does the least damage available
in that basis, and PSNR measures what damage remains.

```
                     transform
   Dense pixels  ───────────────▶  Coefficients
        ▲                                │
        │ inverse transform              │ threshold (keep the big ones)
        │                                ▼
   Reconstruction ◀───────────────  Sparse coefficients
```

The core is plain functions, no class hierarchy:

- `TRANSFORMS` maps a name to a `(forward, inverse)` pair for FFT, DCT, and DWT (db4), each an exact inverse of the other
- `threshold_keep(coeffs, keep_frac)` keeps the top-k coefficients by magnitude
- `energy_curve(coeffs)` returns sorted magnitudes with the cumulative energy fraction
- `psnr(x, x_hat)` and `mse(x, x_hat)` measure the distortion
- `coeff_display(coeffs)` renders the shared log-magnitude coefficient view

## How it works

Each notebook is a single `.py` file, a Python module you can diff in git, with no JSON
cell soup. marimo runs the cells as a dataflow graph instead of top to bottom, so when you
move a slider it re-runs the cells that depend on that slider and nothing else.

`marimo export html-wasm` compiles a notebook into a Pyodide bundle that runs Python in
the browser. The build exports each chapter twice from one source:

| Export | Path | For |
| --- | --- | --- |
| `--mode run --no-show-code` | `/app/<name>/` | Exploring the ideas |
| `--mode run --show-code` | `/notebook/<name>/` | Reading the implementation |

The landing page is a hand-written `docs/index.html`, with no static-site generator behind
a one-page site. `just build` copies `docs/` into `_site/`, then loops over `notebooks/*.py`
exporting both modes. CI runs that same recipe.

## Repository layout

```
notebooks/        One .py file per chapter (marimo notebooks)
docs/             The landing page (index.html), copied verbatim into the build
tests/            Smoke test: every discovered notebook imports as a module
justfile          Every command; the single source of truth for the build
.github/workflows/
  check.yml       Lint + tests on every push and PR
  deploy.yml      Build the site and publish to GitHub Pages on main
```

## Local development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```bash
just              # list every recipe
just edit         # edit a notebook in marimo
just run          # run a notebook as a read-only app
just check        # ruff + marimo checks + pytest
just serve        # preview the landing page alone at :8000 (WASM apps 404)
just build        # the full deploy pipeline into _site/
just preview      # serve the last build at :8000, as deployed
```

`nb` defaults to `notebooks/02_compression_via_sparsity.py`; pass a path to target another
chapter, e.g. `just edit notebooks/01_signal_is_a_vector.py`.

**Screenshots.** To refresh the images above, run `just build` then `just preview`, open
`http://localhost:8000/app/<name>/`, wait for the WASM kernel to boot, then capture the
viewport into `docs/assets/`. Anything in `docs/` ships with the site.

## Deploy

Pushing to `main` triggers [`deploy.yml`](.github/workflows/deploy.yml). It runs
`just build`, the same recipe you run locally, then publishes `_site/` to GitHub Pages.

## License

[MIT](LICENSE) © Oishe Farhan
