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
app = marimo.App(width="medium")

with app.setup:
    import copy
    import io
    import urllib.request

    import bokeh.plotting as bk
    import marimo as mo
    import numpy as np
    import pywt
    from bokeh.layouts import column
    from bokeh.models.ranges import Range1d
    from PIL import Image
    from scipy.fft import dctn, fft2, fftshift, idctn, ifft2, ifftshift


@app.cell(hide_code=True)
def md_title():
    mo.md(r"""
    #️Sparsity Exploration
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Input Image
    """)
    return


@app.cell
def ui_in_image():
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
def compute_get_image(load_image, url_input):
    load_image

    def get_image(url):
        with urllib.request.urlopen(url) as response:
            return Image.open(io.BytesIO(response.read()))

    image_rgb = np.asarray(get_image(url_input.value).convert("RGB"), dtype=np.float32) / 255.0
    return (image_rgb,)


@app.cell
def ui_out_image(image_rgb):
    mo.image(image_rgb)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Image Transformation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1. Color Space
    """)
    return


@app.class_definition
class ColorSpace:
    def encode(self, arr):
        return NotImplementedError

    def decode(self, arr):
        return NotImplementedError

    def coeff_view(self, arr):
        return NotImplementedError

    def channel_views(self, arr):
        # One [w,h,3] RGB image per channel, colored so the channel reads correctly.
        return NotImplementedError

    def channel_colors(self):
        # RGB tint per channel: the direction it pushes RGB, peak-normalized.
        base = self.decode(np.zeros(3, np.float32))
        cols = np.stack([self.decode(e) - base for e in np.eye(3, dtype=np.float32)])
        cols = np.clip(cols, 0, None)
        return cols / cols.max(axis=1, keepdims=True)


@app.class_definition
class RGB(ColorSpace):
    channels = ("R", "G", "B")

    def encode(self, arr):
        return arr

    def decode(self, arr):
        return arr

    def coeff_view(self, arr):
        return arr

    def channel_views(self, arr):
        return [arr * (np.arange(3) == i) for i in range(3)]


@app.class_definition
class YCbCr(ColorSpace):
    channels = ("Y", "Cb", "Cr")

    # JFIF full-range BT.601. Decorrelates luma (Y) from chroma (Cb, Cr) so a
    # single global magnitude threshold naturally spends its budget on luma.
    _M = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        dtype=np.float32,
    )
    _OFF = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    _MINV = np.linalg.inv(_M)

    def encode(self, rgb):
        return rgb @ self._M.T + self._OFF

    def decode(self, ycc):
        return (ycc - self._OFF) @ self._MINV.T

    def coeff_view(self, arr):
        return arr[0]

    def channel_views(self, ycc):
        def view(i):
            v = np.full_like(ycc, 0.5)  # neutral luma + chroma
            v[..., i] = ycc[..., i]
            return np.clip(self.decode(v), 0, 1)
        return [view(i) for i in range(3)]


@app.cell
def ui_in_color_space():
    color_space_choice = mo.ui.tabs(
        tabs={
            "RGB": mo.md(""),
            "YCbCr": mo.md(""),
        },
        value="RGB",
        label="Color Space",
    )
    color_space_choice
    return (color_space_choice,)


@app.cell
def compute_color_space(color_space_choice, image_rgb):
    color_spaces = {
        "RGB": RGB(),
        "YCbCr": YCbCr(),
    }
    color_space = color_spaces[color_space_choice.value]

    image_color_space = color_space.encode(image_rgb)
    return color_space, image_color_space


@app.cell
def ui_out_color_space(color_space, image_color_space):
    plot_channel_strip(color_space.channel_views(image_color_space), color_space.channels)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2. Frequency Space
    """)
    return


@app.class_definition
class Transform:
    def forward(self, img):
        raise NotImplementedError

    def backward(self, coeffs):
        raise NotImplementedError


@app.class_definition
class FFT(Transform):
    def forward(self, img):
        return fftshift(fft2(img, axes=(0, 1), norm="ortho"), axes=(0, 1))

    def backward(self, coeffs):
        return ifft2(ifftshift(coeffs, axes=(0, 1)), axes=(0, 1), norm="ortho")


@app.class_definition
class DCT(Transform):
    def forward(self, img):
        return dctn(img, axes=(0, 1), norm="ortho")

    def backward(self, coeffs):
        return idctn(coeffs, axes=(0, 1), norm="ortho")


@app.class_definition
class DWT(Transform):
    # level=2 fixed: deepest orthonormal depth for 1500x1000 image.
    # TODO: derive from image size so dyadic-sized
    # inputs can use deeper wavelets without leaking energy.

    def __init__(self, wavelet="db4", mode="periodization", level=2):
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self._layout = None  # forward stashes the coeff layout for backward

    def forward(self, img):
        coeffs = pywt.wavedec2(img, self.wavelet, mode=self.mode, level=self.level, axes=(0, 1))
        arr, self._layout = pywt.coeffs_to_array(coeffs, axes=(0, 1))
        return arr

    def backward(self, arr):
        coeffs = pywt.array_to_coeffs(arr, self._layout, output_format="wavedec2")
        return pywt.waverec2(coeffs, self.wavelet, mode=self.mode, axes=(0, 1))


@app.cell
def ui_in_controls():
    wavelet_choice = mo.ui.dropdown(
        options=["haar", "db4", "sym4"],
        value="haar",
        label="Wavelet",
    )

    transform_choice = mo.ui.tabs(
        tabs={
            "FFT": mo.md(""),
            "DCT": mo.md(""),
            "DWT": wavelet_choice,
        },
        value="FFT",
        label="Transform",
    )

    transform_choice
    return transform_choice, wavelet_choice


@app.cell
def compute_transform_image(
    image_color_space,
    transform_choice,
    wavelet_choice,
):
    transforms = {
        "FFT": FFT(),
        "DCT": DCT(),
        "DWT": DWT(wavelet=wavelet_choice.value),
    }
    transform = transforms[transform_choice.value]

    y = transform.forward(image_color_space)
    y_mag = np.abs(y)
    return transform, y, y_mag


@app.function
def plot_transform_space(coeffs):
    view_log = np.log1p(coeffs.sum(axis=-1))
    vmin, vmax = np.percentile(view_log, [0.1, 99.9])
    return mo.image(view_log, vmin=vmin, vmax=vmax)


@app.function
def plot_channel_strip(images, names):
    # Shared layout: one labeled, colored image per channel, side by side.
    return mo.hstack(
        [
            mo.vstack([mo.md(f"**{_name}**"), mo.image(_img)], align="center")
            for _name, _img in zip(names, images)
        ]
    )


@app.cell
def ui_out_plot_transform(y_mag):
    plot_transform_space(y_mag)
    return


@app.function
def plot_all_transform_spaces(coeffs, color_space):
    colors = color_space.channel_colors()
    images = []
    for _i in range(len(color_space.channels)):
        _chan = np.log1p(coeffs[..., _i])
        _vmin, _vmax = np.percentile(_chan, [0.1, 99.9])
        _norm = np.clip((_chan - _vmin) / (_vmax - _vmin), 0, 1)
        images.append(_norm[..., None] * colors[_i])
    return plot_channel_strip(images, color_space.channels)


@app.cell
def ui_out_plot_all_transforms(color_space, y_mag):
    plot_all_transform_spaces(y_mag, color_space)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Sparse Thresholding
    """)
    return


@app.cell
def compute_sort_mags(y_mag):
    # Coefficient magnitudes sorted high->low, with cumulative energy.
    sorted_mag = np.sort(y_mag.ravel())[::-1]
    dist_total = sorted_mag.size
    cum_energy = np.cumsum(sorted_mag.astype(np.float64) ** 2)
    energy_frac = cum_energy / cum_energy[-1]  # fraction of total energy in the top-k
    return cum_energy, dist_total, energy_frac, sorted_mag


@app.function
def plot_distribution(sorted_mag, energy_frac, threshold, kept_count):
    def downsample_geom(values, n=3000):
        # Log-spaced sample of indices into a 1-D array (keeps head detail on log-x).
        idx = np.unique(np.round(np.geomspace(1, values.size, n)).astype(int)) - 1
        return idx, values[idx]

    rank, mag = downsample_geom(sorted_mag)
    efrac = energy_frac[rank]
    x = rank + 1  # 1-based ranks so the log x-axis can show the head

    p_mag = bk.figure(
        height=230,
        sizing_mode="stretch_width",
        x_axis_type="log",
        y_axis_type="log",
        y_axis_label="|coefficient|",
    )
    p_mag.line(x, mag, line_width=2, color="steelblue")
    p_mag.hspan(threshold, line_dash="dashed", line_color="red")
    p_mag.vspan(kept_count, line_dash="dashed", line_color="green")

    p_energy = bk.figure(
        height=180,
        sizing_mode="stretch_width",
        x_axis_type="log",
        x_range=p_mag.x_range,
        x_axis_label="coefficient rank (descending)",
        y_axis_label="cumulative energy",
    )
    p_energy.y_range = Range1d(start=0, end=1.02)
    p_energy.line(x, efrac, line_width=2, color="orange")
    p_energy.vspan(kept_count, line_dash="dashed", line_color="green")

    return mo.as_html(column(p_mag, p_energy, sizing_mode="stretch_width"))


@app.cell
def ui_in_keep_pct():
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
def ui_out_plot_distribution(energy_frac, kept_count, sorted_mag, threshold):
    plot_distribution(sorted_mag, energy_frac, threshold, kept_count)
    return


@app.cell
def compute_threshold(cum_energy, keep_pct, y, y_mag):
    # Keep only the largest-magnitude coefficients (top keep_pct%).
    threshold = float(np.percentile(y_mag, 100 - keep_pct.value))

    _keep = y_mag >= threshold
    y_sparse = np.where(_keep, y, 0.0)
    y_sparse_mag = np.where(_keep, y_mag, 0.0)

    kept_count = int(_keep.sum())
    kept_energy_frac = float(np.sum(y_mag[_keep].astype(np.float64) ** 2) / cum_energy[-1])
    return kept_count, kept_energy_frac, threshold, y_sparse, y_sparse_mag


@app.cell
def ui_out_energy_readout(dist_total, kept_count, kept_energy_frac, threshold):
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
def ui_out_channel_keep(color_space, threshold, y_mag):
    _kept = (y_mag >= threshold).mean(axis=(0, 1))  # fraction of each channel's coeffs kept
    mo.hstack(
        [
            mo.stat(f"{_frac:.2%}", label=f"{_name} kept")
            for _name, _frac in zip(color_space.channels, _kept)
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def ui_out_compaction(dist_total, energy_frac):
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
def ui_out_plot_threshold(y_sparse_mag):
    plot_transform_space(y_sparse_mag)
    return


@app.cell
def ui_out_plot_all_thresholds(color_space, y_sparse_mag):
    plot_all_transform_spaces(y_sparse_mag, color_space)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Inverse Transform Reconstruction
    """)
    return


@app.cell
def compute_inverse_transform(color_space, transform, y_sparse):
    # Reconstruct: invert the transform, decode back to RGB, clip for display.
    # (.real: the inverse FFT returns complex; DCT/DWT are already real.)
    reconstruct = np.clip(
        color_space.decode(
            transform.backward(y_sparse).real,
        ),
        0.0,
        1.0,
    )
    return (reconstruct,)


@app.cell
def ui_out_reconstruction(reconstruct):
    mo.image(reconstruct)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Additional Context
    """)
    return


@app.function
def plot_basis_grid(transform, n=64, k=8):
    # Basis vectors of the transform: inverse-transform each unit coefficient,
    # cropped to its support, as a K x K montage. Cropping is a no-op for the
    # global FFT/DCT bases and zooms into the localized DWT wavelets so their
    # shape is visible. Work on a deep copy so priming the DWT layout never
    # touches the pipeline's transform.
    t = copy.deepcopy(transform)
    t.forward(np.zeros((n, n), dtype=np.float32))

    if isinstance(transform, DWT):
        # spread across the coeff array so the grid spans coarse -> fine scales
        rows = np.linspace(0, n - 1, k).round().astype(int)
        cols = np.linspace(0, n - 1, k).round().astype(int)
        indices = [(int(r), int(c)) for r in rows for c in cols]
    elif isinstance(transform, FFT):
        # lowest-frequency k x k block, centered (FFT puts DC at the middle)
        c = n // 2
        indices = [(c + i, c + j) for i in range(k) for j in range(k)]
    else:
        # lowest-frequency k x k block of coefficient space (DCT: DC at corner)
        indices = [(i, j) for i in range(k) for j in range(k)]

    def basis_image(index):
        # a basis vector is the inverse transform of a single unit coefficient
        e = np.zeros((n, n))
        e[index] = 1.0
        return t.backward(e).real

    def crop_to_support(img, pad=2):
        mag = np.abs(img)
        if mag.max() == 0:
            return img
        ys, xs = np.where(mag > 0.02 * mag.max())
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, img.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, img.shape[1])
        return img[y0:y1, x0:x1]

    gap = 1
    side = k * n + (k + 1) * gap
    canvas = np.full((side, side), 0.15)
    for pos, index in enumerate(indices):
        img = crop_to_support(basis_image(index))
        lo, hi = float(img.min()), float(img.max())
        tile = (img - lo) / (hi - lo) if hi > lo else np.full_like(img, 0.5)
        tile = (
            np.asarray(Image.fromarray((tile * 255).astype(np.uint8)).resize((n, n), Image.NEAREST))
            / 255.0
        )
        r, c = divmod(pos, k)
        y = gap + r * (n + gap)
        x = gap + c * (n + gap)
        canvas[y : y + n, x : x + n] = tile
    return mo.accordion({"Transform basis vectors": mo.image(canvas, width=520)})


@app.cell
def ui_out_basis_grid(transform):
    plot_basis_grid(transform)
    return


if __name__ == "__main__":
    app.run()
