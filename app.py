# /// script
# requires-python = ">=3.14"
# dependencies = [
#       "bokeh",
#       "marimo",
#       "numpy",
#       "pillow",
#       "pywavelets",
#       "scipy"
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App()

with app.setup:
    import io
    import urllib.request

    from bokeh.layouts import column
    from bokeh.models import Range1d
    import bokeh.plotting as bk
    import numpy as np
    from PIL import Image
    import pywt
    from scipy.fft import dctn, idctn, fft2, ifft2, fftshift

    import marimo as mo


@app.cell(hide_code=True)
def title():
    mo.md(r"""
    #️Sparsity Exploration
    """)
    return


@app.cell
def cell_image_ui():
    url_input = mo.ui.text(
        value="https://picsum.photos/1500/1000",
        kind="url",
        full_width=True,
    )

    load_image = mo.ui.run_button(
        label="Load Image",
        full_width=True,
    )

    mo.hstack(
        [url_input, load_image],
        align="stretch",
        justify="center",
        widths=[4, 1],
    )
    return load_image, url_input


@app.cell
def cell_get_image(load_image, url_input):
    load_image

    def get_image(url):
        with urllib.request.urlopen(url) as response:
            return Image.open(io.BytesIO(response.read()))

    image = get_image(url_input.value)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.
    return image, rgb


@app.cell
def cell_plot_image(image):
    mo.image(image)
    return


@app.cell
def cell_transform_options():
    class Transform:
        """A 2D analysis/synthesis transform over spatial axes (0, 1).

        Subclasses implement forward/backward; everything else operates on a
        coefficient array of the same shape as the image, so the pipeline is
        uniform across FFT/DCT/DWT and works for grayscale (H, W) or color
        (H, W, C) (the channel axis rides along as a batch dimension).
        """

        def forward(self, img):
            raise NotImplementedError

        def backward(self, coeffs):
            raise NotImplementedError

        def display(self, mag):
            # how to arrange coefficient magnitudes for viewing
            return mag

        def basis_image(self, shape, index):
            # a basis vector is the inverse transform of a single unit coefficient
            e = np.zeros(shape)
            e[index] = 1.0
            return self.backward(e).real

        def basis_indices(self, shape, k):
            # lowest-frequency k x k block of coefficient space
            return [(i, j) for i in range(k) for j in range(k)]

        def fresh(self):
            # stateless clone with the same config (used by the basis grid so
            # priming layout state never touches the pipeline's transform)
            return type(self)()


    class FFT(Transform):
        def forward(self, img):
            return fft2(img, axes=(0, 1), norm="ortho")

        def backward(self, coeffs):
            return ifft2(coeffs, axes=(0, 1), norm="ortho")

        def display(self, mag):
            # center the spectrum so low frequencies sit in the middle
            return fftshift(mag, axes=(0, 1))


    class DCT(Transform):
        def forward(self, img):
            return dctn(img, axes=(0, 1), norm="ortho")

        def backward(self, coeffs):
            return idctn(coeffs, axes=(0, 1), norm="ortho")


    class DWT(Transform):
        # level fixed at 2: the deepest non-redundant (orthonormal) depth for the
        # default 1500x1000 image. TODO: derive from image size so dyadic-sized
        # inputs can use deeper wavelets without leaking energy.
        def __init__(self, wavelet="db4", mode="periodization", level=2):
            self.wavelet = wavelet
            self.mode = mode
            self.level = level
            self._layout = None  # forward stashes the coeff layout for backward

        def forward(self, img):
            coeffs = pywt.wavedec2(
                img, self.wavelet, mode=self.mode, level=self.level, axes=(0, 1)
            )
            arr, self._layout = pywt.coeffs_to_array(coeffs, axes=(0, 1))
            return arr

        def backward(self, arr):
            coeffs = pywt.array_to_coeffs(arr, self._layout, output_format="wavedec2")
            return pywt.waverec2(coeffs, self.wavelet, mode=self.mode, axes=(0, 1))

        def basis_indices(self, shape, k):
            # spread across the coeff array so the grid spans coarse -> fine scales
            rows = np.linspace(0, shape[0] - 1, k).round().astype(int)
            cols = np.linspace(0, shape[1] - 1, k).round().astype(int)
            return [(int(r), int(c)) for r in rows for c in cols]

        def fresh(self):
            return DWT(self.wavelet, self.mode, self.level)


    transforms = {
        "FFT": FFT(),
        "DCT": DCT(),
        "DWT": DWT(),
    }


    def plot_transform_space(coeffs, transform, colorspace):
        view = colorspace.coeff_view(transform.display(coeffs))
        view_log = np.log1p(view)
        vmin, vmax = np.percentile(view_log, [0.1, 99.9])
        return mo.image(view_log, vmin=vmin, vmax=vmax)


    return DWT, plot_transform_space, transforms


@app.cell
def cell_colorspace_options():
    class ColorSpace:
        """Reversible per-pixel color transform wrapped around the pipeline:
        rgb -> encode -> transform/threshold/inverse -> decode -> rgb."""

        def encode(self, rgb):
            return rgb

        def decode(self, arr):
            return arr

        def coeff_view(self, mag):
            # which coefficient channels to show in the transform-space plot
            return mag


    class YCbCr(ColorSpace):
        # JFIF full-range BT.601. Decorrelates luma (Y) from chroma (Cb, Cr) so a
        # single global magnitude threshold naturally spends its budget on luma.
        _M = np.array(
            [[0.299, 0.587, 0.114],
             [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]],
            dtype=np.float32,
        )
        _OFF = np.array([0.0, 0.5, 0.5], dtype=np.float32)
        _MINV = np.linalg.inv(_M)

        def encode(self, rgb):
            return rgb @ self._M.T + self._OFF

        def decode(self, ycc):
            return (ycc - self._OFF) @ self._MINV.T

        def coeff_view(self, mag):
            # chroma carries little structure; show luma magnitudes only
            return mag[..., 0]


    color_spaces = {
        "RGB": ColorSpace(),
        "YCbCr": YCbCr(),
    }
    return (color_spaces,)


@app.cell
def cell_transform_ui(transforms):
    transform_choice = mo.ui.dropdown(
        options=list(transforms),
        value="FFT",
        label="Transform",
    )
    transform_choice
    return (transform_choice,)


@app.cell
def cell_wavelet_ui():
    wavelet_choice = mo.ui.dropdown(
        options=["db4", "sym4", "haar"],
        value="db4",
        label="Wavelet",
    )
    return (wavelet_choice,)


@app.cell
def cell_wavelet_display(transform_choice, wavelet_choice):
    # Show the wavelet selector only when DWT is the active transform.
    mo.stop(transform_choice.value != "DWT", mo.md(""))
    wavelet_choice
    return


@app.cell
def cell_colorspace_ui(color_spaces):
    color_space_choice = mo.ui.dropdown(
        options=list(color_spaces),
        value="RGB",
        label="Color space",
    )
    color_space_choice
    return (color_space_choice,)


@app.cell
def cell_transform_image(
    DWT,
    color_space_choice,
    color_spaces,
    rgb,
    transform_choice,
    transforms,
    wavelet_choice,
):
    transform = transforms[transform_choice.value]
    if transform_choice.value == "DWT":
        transform = DWT(wavelet=wavelet_choice.value)
    colorspace = color_spaces[color_space_choice.value]

    # Encode into the working color space, then into a sparse coefficient space
    source = colorspace.encode(rgb)
    y = transform.forward(source)
    y_mag = np.abs(y)
    return colorspace, transform, y, y_mag


@app.cell
def plot_transform(colorspace, plot_transform_space, transform, y_mag):
    plot_transform_space(y_mag, transform, colorspace)
    return


@app.cell
def cell_basis_grid(transform):
    # Basis vectors of the selected transform: inverse-transform of each unit
    # coefficient, cropped to the function's support, shown as a K x K montage.
    # Cropping is a no-op for global bases (FFT/DCT) and zooms into the localized
    # DWT wavelets so their shape is visible. A fresh instance is used so priming
    # the DWT layout doesn't disturb the pipeline's transform.
    _N, _K = 64, 8
    _t = transform.fresh()  # same config, no shared state
    _t.forward(np.zeros((_N, _N), dtype=np.float32))


    def _crop_to_support(img, pad=2):
        mag = np.abs(img)
        if mag.max() == 0:
            return img
        _ys, _xs = np.where(mag > 0.02 * mag.max())
        _y0, _y1 = max(_ys.min() - pad, 0), min(_ys.max() + 1 + pad, img.shape[0])
        _x0, _x1 = max(_xs.min() - pad, 0), min(_xs.max() + 1 + pad, img.shape[1])
        return img[_y0:_y1, _x0:_x1]


    _gap = 1
    _side = _K * _N + (_K + 1) * _gap
    _canvas = np.full((_side, _side), 0.15)
    for _n, _idx in enumerate(_t.basis_indices((_N, _N), _K)):
        _img = _crop_to_support(_t.basis_image((_N, _N), _idx))
        _lo, _hi = float(_img.min()), float(_img.max())
        _tile = (_img - _lo) / (_hi - _lo) if _hi > _lo else np.full_like(_img, 0.5)
        _tile = (
            np.asarray(
                Image.fromarray((_tile * 255).astype(np.uint8)).resize((_N, _N), Image.NEAREST)
            )
            / 255.0
        )
        _r, _c = divmod(_n, _K)
        _y = _gap + _r * (_N + _gap)
        _x = _gap + _c * (_N + _gap)
        _canvas[_y:_y + _N, _x:_x + _N] = _tile
    mo.accordion({"Transform basis vectors": mo.image(_canvas, width=520)})
    return


@app.cell
def cell_sort_mags(y_mag):
    # Coefficient magnitudes sorted high->low, with cumulative energy.
    def downsample_geom(values, n=3000):
        # Log-spaced sample of indices into a 1-D array (keeps head detail on log-x).
        idx = np.unique(np.round(np.geomspace(1, values.size, n)).astype(int)) - 1
        return idx, values[idx]


    sorted_mag = np.sort(y_mag.ravel())[::-1]
    dist_total = sorted_mag.size
    cum_energy = np.cumsum(sorted_mag.astype(np.float64) ** 2)
    energy_frac = cum_energy / cum_energy[-1]  # fraction of total energy in the top-k

    dist_rank, dist_mag = downsample_geom(sorted_mag)
    dist_efrac = energy_frac[dist_rank]
    return cum_energy, dist_efrac, dist_mag, dist_rank, dist_total, energy_frac


@app.cell
def cell_keep_pct_ui():
    keep_pct = mo.ui.slider(
        steps=np.unique([np.arange(1, 11) / scale for scale in [10, 1, 0.1]]).tolist(),
        value=5,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Keep percentage",
    )
    keep_pct
    return (keep_pct,)


@app.cell
def cell_threshold(cum_energy, keep_pct, y, y_mag):
    # Keep only the largest-magnitude coefficients (top keep_pct%).
    threshold = float(np.percentile(y_mag, 100 - keep_pct.value))

    _keep = y_mag >= threshold
    y_sparse = np.where(_keep, y, 0.0)
    y_sparse_mag = np.where(_keep, y_mag, 0.0)

    kept_count = int(_keep.sum())
    kept_energy_frac = float(np.sum(y_mag[_keep].astype(np.float64) ** 2) / cum_energy[-1])
    return kept_count, kept_energy_frac, threshold, y_sparse, y_sparse_mag


@app.cell
def cell_energy_readout(dist_total, kept_count, kept_energy_frac, threshold):
    mo.hstack(
        [
            mo.stat(f"{kept_count / dist_total:.2%}", label="coefficients kept"),
            mo.stat(f"{kept_energy_frac:.2%}", label="energy retained"),
            mo.stat(f"{threshold:.3g}", label="threshold |coeff|"),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def cell_compaction(dist_total, energy_frac):
    def _coeffs_for_energy(target):
        return (int(np.searchsorted(energy_frac, target)) + 1) / dist_total


    mo.hstack(
        [
            mo.stat(f"{_coeffs_for_energy(0.99):.3%}", label="coeffs for 99% energy"),
            mo.stat(f"{_coeffs_for_energy(0.999):.3%}", label="coeffs for 99.9% energy"),
            mo.stat(f"{_coeffs_for_energy(0.9999):.3%}", label="coeffs for 99.99% energy"),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def cell_plot_threshold(
    colorspace,
    plot_transform_space,
    transform,
    y_sparse_mag,
):
    plot_transform_space(y_sparse_mag, transform, colorspace)
    return


@app.cell
def cell_inverse_transform(colorspace, transform, y_sparse):
    # Reconstruct: invert the transform, decode back to RGB, clip for display.
    # (.real: the inverse FFT returns complex; DCT/DWT are already real.)
    reconstruct = np.clip(
        colorspace.decode(
            transform.backward(y_sparse).real,
        ),
        0.0,
        1.0,
    )
    return (reconstruct,)


@app.cell
def cell_plot_reconstruction(reconstruct):
    mo.image(reconstruct)
    return


@app.cell
def cell_plot_distribution(
    dist_efrac,
    dist_mag,
    dist_rank,
    kept_count,
    threshold,
):
    _x = dist_rank + 1  # 1-based ranks so the log x-axis can show the head

    _p_mag = bk.figure(
        height=230,
        sizing_mode="stretch_width",
        x_axis_type="log",
        y_axis_type="log",
        y_axis_label="|coefficient|",
    )
    _p_mag.line(_x, dist_mag, line_width=2, color="steelblue")
    _p_mag.hspan(threshold, line_dash="dashed", line_color="red")
    _p_mag.vspan(kept_count, line_dash="dashed", line_color="green")

    _p_energy = bk.figure(
        height=180,
        sizing_mode="stretch_width",
        x_axis_type="log",
        x_range=_p_mag.x_range,
        x_axis_label="coefficient rank (descending)",
        y_axis_label="cumulative energy",
    )
    _p_energy.y_range = Range1d(start=0, end=1.02)
    _p_energy.line(_x, dist_efrac, line_width=2, color="orange")
    _p_energy.vspan(kept_count, line_dash="dashed", line_color="green")

    mo.as_html(column(_p_mag, _p_energy, sizing_mode="stretch_width"))
    return


if __name__ == "__main__":
    app.run()
