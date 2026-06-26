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
app = marimo.App(width="medium", layout_file="layouts/app.grid.json")


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    #️Sparsity Exploration
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def cell_imports():
    import io
    import urllib.request

    import numpy as np
    import bokeh.plotting as bk
    import pywt
    from PIL import Image
    from scipy.fft import dctn, idctn, fft2, ifft2, fftshift

    return Image, bk, dctn, fft2, fftshift, idctn, ifft2, io, np, pywt, urllib


@app.cell
def cell_image_ui(mo):
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
def cell_get_image(Image, io, load_image, np, url_input, urllib):
    load_image

    def get_image(url):
        with urllib.request.urlopen(url) as response:
            return Image.open(io.BytesIO(response.read()))

    image = get_image(url_input.value)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    # rgb = np.asarray(image, dtype=np.float32) / 255.0
    return gray, image


@app.cell
def cell_plot_image(image, mo):
    mo.image(image)
    return


@app.cell
def _(gray, mo):
    mo.image(gray)
    return


@app.cell
def cell_transform_options(dctn, fft2, fftshift, idctn, ifft2, mo, np, pywt):
    def wavelet_pair(wavelet="db4", mode="periodization"):
        _slices = {}  # forward stashes the coeff layout for backward to reuse

        def forward(img):
            arr, _slices["layout"] = pywt.coeffs_to_array(
                pywt.wavedec2(img, wavelet, mode=mode)
            )
            return arr

        def backward(arr):
            coeffs = pywt.array_to_coeffs(
                arr, _slices["layout"], output_format="wavedec2"
            )
            return pywt.waverec2(coeffs, wavelet, mode=mode)

        return forward, backward

    transforms = {
        "FFT": (fft2, ifft2),
        "DCT": (dctn, idctn),
        "DWT": wavelet_pair(),
    }

    def plot_transform_space(coeffs, choice):
        # Todo: rename coeffs
        # center the FFT spectrum for display
        if choice == "FFT":
            coeffs = fftshift(coeffs)
        coeffs_log = np.log1p(coeffs)
        vmin, vmax = np.percentile(coeffs_log, [0.1, 99.9])
        return mo.image(coeffs_log, vmin=vmin, vmax=vmax)

    return plot_transform_space, transforms


@app.cell
def cell_transform_ui(mo, transforms):
    transform_choice = mo.ui.dropdown(
        options=list(transforms),
        value="FFT",
        label="Transform",
    )
    transform_choice
    return (transform_choice,)


@app.cell
def cell_transform_image(gray, np, transform_choice, transforms):
    forward, backward = transforms[transform_choice.value]

    # Transform into a sparse coefficient space
    y = forward(gray)
    y_mag = np.abs(y)
    return backward, y, y_mag


@app.cell
def plot_transform(plot_transform_space, transform_choice, y_mag):
    plot_transform_space(y_mag, transform_choice.value)
    return


@app.cell
def cell_keep_pct_ui(mo, np):
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
def cell_threshold(keep_pct, np, y, y_mag):
    # Keep only the largest-magnitude coefficients
    threshold = np.percentile(y_mag, 100 - keep_pct.value)
    y_sparse = np.where(y_mag >= threshold, y, 0.0)
    y_sparse_mag = np.where(y_mag >= threshold, y_mag, 0.0)
    return threshold, y_sparse, y_sparse_mag


@app.cell
def cell_plot_threshold(plot_transform_space, transform_choice, y_sparse_mag):
    plot_transform_space(y_sparse_mag, transform_choice.value)
    return


@app.cell
def cell_inverse_transform(backward, y_sparse):
    # Reconstruct the image from the reduced coefficient set
    # (.real: the inverse FFT returns complex; DCT is already real)
    reconstruct = backward(y_sparse).real
    return (reconstruct,)


@app.cell
def cell_plot_reconstruction(mo, reconstruct):
    mo.image(reconstruct)
    return


@app.cell
def cell_sort_mags(np, y_mag):
    # Sorted, downsampled coefficient distribution for the plot.
    # ~1.5M sorted magnitudes -> ~3k log-spaced ranks (keeps head detail on log-y).
    _mags = np.sort(y_mag.ravel())[::-1]
    dist_total = _mags.size
    _idx = np.unique(np.round(np.geomspace(1, dist_total, 3000)).astype(int)) - 1
    dist_rank = _idx
    dist_mag = _mags[_idx]
    return dist_mag, dist_rank, dist_total


@app.cell
def cell_plot_distribution(
    bk,
    dist_mag,
    dist_rank,
    dist_total,
    keep_pct,
    mo,
    threshold,
):
    dist_keep_count = int(round(keep_pct.value / 100 * dist_total))

    p = bk.figure(
        height=300,
        sizing_mode="stretch_width",
        y_axis_type="log",
        x_axis_label="coefficient rank (descending)",
        y_axis_label="|coefficient|",
    )
    p.line(dist_rank, dist_mag, line_width=2)
    p.hspan(threshold, line_dash="dashed", line_color="red")
    p.vspan(dist_keep_count, line_dash="dashed", line_color="green")
    mo.as_html(p)
    return


if __name__ == "__main__":
    app.run()
