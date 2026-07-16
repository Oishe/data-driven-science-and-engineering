# Interactive Notebooks

Small explorable explanations, each running entirely in your browser via
WebAssembly. Every notebook ships two ways:

- **Playground** — code hidden, just the interactive UI.
- **Notebook** — every function on show, rigorous and reproducible.

---

## Compression via Sparsity

From dense pixel space to sparse transform space — the idea behind JPEG,
JPEG 2000, and most lossy image compression. Load any image, pick a transform
(FFT · DCT · wavelet), drag the keep-% and watch the reconstruction,
coefficients and PSNR update live.



|[Playground](app/sparsity)| [Notebook](notebook/sparsity/)|
|--|--|
|Interactive App| Full notebook with code|

---

## Intuition: A Signal Is a Vector (1-D)

The one-dimensional companion — small enough to see every number. A signal is a
point in ℝⁿ; compression is just choosing better coordinates. Watch a smooth
signal go from 64 dense samples to a handful of Fourier coefficients.

|[Playground](app/sparsity_1d)| [Notebook](notebook/sparsity_1d/)|
|--|--|
|Interactive App| Full notebook with code|
