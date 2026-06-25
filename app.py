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


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    #️Sparsity Exploration
    """)
    return


@app.cell
def cell_marimo():
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
def _(mo):
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
def cell_get_image(Image, io, load_image, mo, np, url_input, urllib):
    load_image

    def get_image(url):
        with urllib.request.urlopen(url) as response:
            return Image.open(io.BytesIO(response.read()))

    image = get_image(url_input.value)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    # rgb = np.asarray(image, dtype=np.float32) / 255.0

    mo.image(image)
    return (gray,)


@app.cell
def cell_transform_choice(dctn, fft2, idctn, ifft2, mo, pywt):
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
    transform_choice = mo.ui.dropdown(
        options=list(transforms),
        value="FFT",
        label="Transform",
    )
    transform_choice
    return transform_choice, transforms


@app.cell
def cell_keep_pct(mo, np):
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
def cell_forward(gray, np, transform_choice, transforms):
    t_choice = transform_choice.value
    forward, backward = transforms[t_choice]

    # Transform into a sparse coefficient space
    y = forward(gray)
    y_mag = np.abs(y)
    return backward, t_choice, y, y_mag


@app.cell
def cell_sparse(backward, keep_pct, np, y, y_mag):
    # Keep only the largest-magnitude coefficients
    threshold = np.percentile(y_mag, 100 - keep_pct.value)
    y_sparse = np.where(y_mag >= threshold, y, 0.0)

    # Reconstruct the image from the reduced coefficient set
    # (.real: the inverse FFT returns complex; DCT is already real)
    reconstruct = backward(y_sparse).real
    return reconstruct, threshold, y_sparse


@app.cell
def cell_dist_data(np, y_mag):
    # Sorted, downsampled coefficient distribution for the plot.
    # ~1.5M sorted magnitudes -> ~3k log-spaced ranks (keeps head detail on log-y).
    _mags = np.sort(y_mag.ravel())[::-1]
    dist_total = _mags.size
    _idx = np.unique(np.round(np.geomspace(1, dist_total, 3000)).astype(int)) - 1
    dist_rank = _idx
    dist_mag = _mags[_idx]
    return dist_mag, dist_rank, dist_total


@app.cell
def cell_plot_compare(
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


@app.cell
def cell_space_original(fftshift, mo, np, t_choice, y, y_sparse):
    def image_transform_space(coeffs):
        # center the FFT spectrum for display
        if t_choice == "FFT":
            coeffs = fftshift(coeffs)
        coeffs_log = np.log1p(np.abs(coeffs))
        vmin, vmax = np.percentile(coeffs_log, [0.1, 99.9])
        return mo.image(coeffs_log, vmin=vmin, vmax=vmax)

    original = image_transform_space(y)
    sparse = image_transform_space(y_sparse)
    return original, sparse


@app.cell
def cell_view_all(gray, mo, original, reconstruct, sparse):
    mo.vstack(
        [
            mo.hstack(
                [mo.image(gray), original],
                align="center",
                widths=[1, 1],
            ),
            mo.hstack(
                [mo.image(reconstruct), sparse],
                align="center",
                widths=[1, 1],
            ),
        ],
        align="stretch",
        gap=2,
    )
    return


if __name__ == "__main__":
    app.run()
