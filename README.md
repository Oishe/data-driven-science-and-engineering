# Compression via Sparsity

*From dense pixel space to sparse transform space — the idea behind JPEG, JPEG 2000, and most lossy image compression.*

Interactive [marimo](https://marimo.io) notebooks that run entirely in the browser via WebAssembly — no install, no server. Each ships two ways: a code-hidden **playground** and a code-visible **notebook**.

| Notebook | Playground | Notebook |
| --- | --- | --- |
| **Compression via Sparsity** — the full image story | [Launch ▶](https://oishe.github.io/sparsity/app/sparsity/) | [Read 📖](https://oishe.github.io/sparsity/notebook/sparsity/) |
| **A Signal Is a Vector (1-D)** — the companion, every number visible | [Launch ▶](https://oishe.github.io/sparsity/app/sparsity_1d/) | [Read 📖](https://oishe.github.io/sparsity/notebook/sparsity_1d/) |

**[Full landing page →](https://oishe.github.io/sparsity/)**

---

## What it is

One notebook, `notebooks/sparsity.py`, in two acts:

- **The article** — the narrative inline, each concept followed by the clean function that implements it and a demo. Rigorous and reproducible; this is the code-visible **notebook** export.
- **The playground** — one control panel (image URL · transform · keep-%) reusing the article's functions, with grouped live outputs. This is the code-hidden **app** export.

The whole story is one loop between two spaces:

```
                     transform
   Dense pixels  ───────────────▶  Coefficients
        ▲                                │
        │ inverse transform              │ threshold (keep the big ones)
        │                                ▼
   Reconstruction ◀───────────────  Sparse coefficients
```

## The idea

An image is *dense* in pixel space. Rotate it into an orthonormal basis (FFT, DCT, or wavelet) and it becomes *sparse*: a few large coefficients hold almost all the energy. Keep the largest, zero the rest, invert — that is lossy compression. Because the basis is orthonormal, discarding the smallest coefficients is provably the least-damaging approximation (Parseval), and the distortion is measured with PSNR/MSE.

The core is plain functions, no class hierarchy:

- `TRANSFORMS` — `{name: (forward, inverse)}` for FFT · DCT · DWT (db4), each an exactly-invertible pair
- `threshold_keep(coeffs, keep_frac)` — keep the top-k coefficients by magnitude
- `energy_curve(coeffs)` — sorted magnitudes + cumulative energy fraction
- `psnr(x, x_hat)` / `mse(x, x_hat)` — the distortion axis
- `coeff_display(coeffs)` — the shared log-magnitude coefficient view

## Run it locally

Recipes live in the [`justfile`](justfile) (`just` lists them; `nb` defaults to the sparsity notebook):

```bash
just edit      # edit the notebook (pairing-ready)
just run       # run as a read-only app
just check     # ruff + marimo checks
```

## Build the site

The deployed site is a Markdown landing page (MkDocs Material, from `docs/index.md`) plus, for each notebook, a code-hidden app and a code-visible notebook. One recipe reproduces the whole deploy pipeline into `_site/`:

```bash
just build     # full pipeline: landing + every notebook, both modes
just preview   # build, then serve the whole site at localhost:8000
just serve     # landing page only, live reload (WASM apps not included)
```

## Deploy

Pushing to `main` triggers [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which installs `just` and runs `just build` — the same recipe you run locally — then publishes `_site/` to GitHub Pages.
