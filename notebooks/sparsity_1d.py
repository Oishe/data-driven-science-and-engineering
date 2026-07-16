# /// script
# requires-python = ">=3.14"
# dependencies = [
#       "marimo",
#       "matplotlib",
#       "numpy",
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Compressing a Signal: Sparsity in the Right Basis

    A signal is usually stored as a long list of samples, but that list is rarely its most economical description. Choose the **right basis** — the right set of axes to measure it against — and nearly all of those numbers collapse to zero, leaving a **sparse** handful that compresses the signal with almost no loss. This notebook builds that idea from first principles, on a small signal.
    """)
    return


@app.cell(hide_code=True)
def md_signal_intro():
    mo.md(r"""
    ## 1. A signal is a high dimensional vector

    Lets start with an example of a real world signal: a sound wave, a temperature trace, a pixel intensity, etc. Almost any measurement of the signal unfolds over time or space.
    """)
    return


@app.cell
def compute_signal():
    def build_signal(n):
        """A smooth periodic signal built from a few pure cosines, so its Fourier expansion is exactly sparse."""
        t = np.linspace(0, 1, n, endpoint=False)
        return (
            0.5
            + 4.0 * np.cos(2 * np.pi * 1 * t)
            + 2.0 * np.cos(2 * np.pi * 3 * t)
            + 1.2 * np.cos(2 * np.pi * 6 * t)
        )


    n = 64
    x = build_signal(n)
    return n, x


@app.cell
def demo_signal(n, x):
    def plot_signal(x, n):
        t = np.linspace(0, 1, n, endpoint=False)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(t, x, color="blue", linewidth=1.8)
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.set_title("a signal", fontsize=11)
        style_signal_axes(ax)
        return theme(fig)


    plot_signal(x, n)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When we measure the signal, we sample its values at evenly spaced intervals.
    In this case we are taking $n = 64$ samples.
    """)
    return


@app.cell
def demo_signal_samples(n, x):
    def plot_signal_samples(x, n):
        i = np.arange(n)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        draw_stem(ax, i, x, "blue")
        ax.set_xlabel("sample index $i$")
        ax.set_ylabel("$x_i$")
        ax.set_title(rf"the same signal, sampled at n = {n} points", fontsize=11)
        style_signal_axes(ax)
        return theme(fig)


    plot_signal_samples(x, n)
    return


@app.cell(hide_code=True)
def md_signal_vector():
    mo.md(r"""
    If we stack each of the samples into a column, the signal becomes a single vector.

    $$
    \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^{n}.
    $$

    The entire signal is a **single point in an $n$-dimensional space**. Each sample is the value along each dimension. In this case our signal vector $x$ is a point in $\mathbb{R}^{64}$.
    """)
    return


@app.cell(hide_code=True)
def md_basis_intro():
    mo.md(r"""
    ## 2. Changing coordinates or basis

    Lets see what happens when we decide to change our coordinates, the basis vectors.
    We have a 2-dimensional vector $\mathbf{V}$. We can choose to measure it with any coordinate basis. Below, we have two sets of axes: the standard basis in blue and a rotated basis in red.
    """)
    return


@app.cell
def ui_rotate():
    rotate = mo.ui.slider(
        -90,
        90,
        value=30,
        step=1,
        label=r"rotate the basis $\theta$ (degrees)",
        show_value=True,
        full_width=True,
    )

    mo.vstack(
        [
            mo.md(
                r"**Drag the slider** to turn the red axes and watch the vector's coordinates change — the vector itself is fixed."
            ),
            rotate,
        ]
    )
    return (rotate,)


@app.cell
def demo_rotate(rotate):
    @mo.cache
    def plot_rotate_coords(degrees):
        theta = np.deg2rad(degrees)
        v_std = np.array([1.0, 1.6])
        B = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x_prime = B.T @ v_std

        fg = plt.rcParams.get("text.color", "black")
        e1p, e2p = B[:, 0], B[:, 1]
        foot1, foot2 = x_prime[0] * e1p, x_prime[1] * e2p
        L = 2.2
        fig, ax = plt.subplots(figsize=(5.4, 5.4))

        def axis(vec, color):
            ax.annotate(
                "", xy=vec * L, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=1.6)
            )
            ax.plot([0, -vec[0] * L], [0, -vec[1] * L], color=color, lw=0.8, alpha=0.4)

        axis(np.array([1.0, 0.0]), "blue")
        axis(np.array([0.0, 1.0]), "blue")
        axis(e1p, "red")
        axis(e2p, "red")

        ax.plot([v_std[0], v_std[0]], [v_std[1], 0], color="blue", lw=1.0, ls="--", alpha=0.7)
        ax.plot([v_std[0], 0], [v_std[1], v_std[1]], color="blue", lw=1.0, ls="--", alpha=0.7)
        ax.plot([v_std[0], foot1[0]], [v_std[1], foot1[1]], color="red", lw=1.0, ls="--", alpha=0.7)
        ax.plot([v_std[0], foot2[0]], [v_std[1], foot2[1]], color="red", lw=1.0, ls="--", alpha=0.7)

        ax.annotate("", xy=v_std, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=fg, lw=2.6))

        ax.text(*(np.array([1.0, 0.0]) * L * 1.05), r"$\hat{e}_1$", color="blue", fontsize=11)
        ax.text(*(np.array([0.0, 1.0]) * L * 1.05), r"$\hat{e}_2$", color="blue", fontsize=11)
        ax.text(*(e1p * L * 1.05), r"$\hat{e}_1'$", color="red", fontsize=11)
        ax.text(*(e2p * L * 1.05), r"$\hat{e}_2'$", color="red", fontsize=11)
        ax.text(v_std[0] + 0.08, v_std[1] + 0.08, r"$\mathbf{V}$", color=fg, fontsize=13)

        ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.3)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

        eq = mo.md(
            r"$$\underbrace{\textcolor{blue}{\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}}}_{\textcolor{blue}{[\,\hat{e}_1\ \ \hat{e}_2\,]}}\,"
            r"\underbrace{\textcolor{blue}{\begin{bmatrix}"
            rf" {v_std[0]:.2f} \\ {v_std[1]:.2f}"
            r" \end{bmatrix}}}_{\textcolor{blue}{\mathbf{x}}} = "
            r"\underbrace{\begin{bmatrix}"
            rf" {v_std[0]:.2f} \\ {v_std[1]:.2f}"
            r" \end{bmatrix}}_{\mathbf{v}} = "
            r"\underbrace{\textcolor{red}{\begin{bmatrix}"
            rf" {B[0, 0]:.2f} & {B[0, 1]:.2f} \\ {B[1, 0]:.2f} & {B[1, 1]:.2f}"
            r" \end{bmatrix}}}_{\textcolor{red}{[\,\hat{e}_1'\ \ \hat{e}_2'\,]}}\,"
            r"\underbrace{\textcolor{red}{\begin{bmatrix}"
            rf" {x_prime[0]:.2f} \\ {x_prime[1]:.2f}"
            r" \end{bmatrix}}}_{\textcolor{red}{\mathbf{x}'}}$$"
        )
        return mo.vstack([theme(fig), eq], align="center")


    plot_rotate_coords(rotate.value)
    return


@app.cell(hide_code=True)
def md_basis_2d():
    mo.md(r"""
    A vector is a geometric object — an arrow that exists before any axes are chosen. Selecting a **basis** only fixes the rulers we measure it against: the arrow never moves, only its **coordinates** do. Collect two orthonormal basis vectors $\hat{\mathbf{e}}_1',\ \hat{\mathbf{e}}_2'$ as the columns of a matrix,

    $$
    B =
    \begin{bmatrix}
    \big| & \big| \\
    \boldsymbol{\hat{\mathbf{e}}_1'} & \boldsymbol{\hat{\mathbf{e}}_2'} \\ \big| & \big|
    \end{bmatrix}
    $$

    and that matrix *is* the transformation — here a **rotation** of the axes. The same vector $\mathbf{v}$ acquires new coordinates $\mathbf{x}'$:

    $$
    \mathbf{v} = B\,\mathbf{x}', \qquad \mathbf{x}' = B^{\top}\mathbf{v}.
    $$

    Because the basis is orthonormal, $B^{\top} = B^{-1}$: the change of basis is a reversible rotation that discards nothing.
    """)
    return


@app.cell(hide_code=True)
def md_basis_nd():
    mo.md(r"""
    ### The same move in $n$ dimensions

    Nothing about this is special to two dimensions. Choose any **orthonormal basis** of $\mathbb{R}^n$ — unit vectors $\boldsymbol{\psi}_1, \dots, \boldsymbol{\psi}_n$ meeting at right angles — and collect them as the columns of a matrix $\Psi$. Just as $B$ turned a vector's standard coordinates into rotated ones, $\Psi$ gives the same vector $\mathbf{x}$ new coordinates $\mathbf{s} = \Psi^{\top}\mathbf{x}$:

    $$
    \mathbf{x}
    = \Psi\,\mathbf{s}
    = \underbrace{\begin{bmatrix} \big| & \big| & & \big| \\ \boldsymbol{\psi}_1 & \boldsymbol{\psi}_2 & \cdots & \boldsymbol{\psi}_n \\ \big| & \big| & & \big| \end{bmatrix}}_{\Psi}\,
    \underbrace{\begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_n \end{bmatrix}}_{\mathbf{s}}
    = \sum_{k=1}^{n} s_k\, \boldsymbol{\psi}_k.
    $$

    Reading the product column by column, the signal is just a weighted sum of the basis vectors $\boldsymbol{\psi}_k$, each scaled by its coordinate $s_k$. Orthonormality ($\Psi^{\top}\Psi = I$) keeps this a pure **rotation**: nothing is lost, and it reverses exactly through $\Psi$.

    > **Parseval's theorem.** An orthonormal rotation preserves length, $\lVert\mathbf{x}\rVert = \lVert\mathbf{s}\rVert$ — the signal carries the same energy in every such basis.
    """)
    return


@app.cell(hide_code=True)
def md_waves():
    mo.md(r"""
    ## 3. Bases of waves: Fourier and the DCT

    For natural signals the most useful rotations use **waves** of increasing frequency as their axes. This is just another orthonormal basis, so its coordinates describe the *same* signal vector $\mathbf{x}$ as faithfully as the raw samples do — we have only rotated $\mathbf{x}$ into a more revealing frame.

    ### Fourier

    The Fourier basis writes a signal as a sum of **complex exponentials** of increasing frequency,

    $$
    \boldsymbol{\psi}_k \;\propto\; e^{\,2\pi i k t} = \cos(2\pi k t) + i\,\sin(2\pi k t),
    $$

    where Euler's formula unpacks each exponential into a cosine and a sine. Paired with the fast Fourier transform, this single decomposition became the backbone of **modern signal processing** — audio and image coding, communications, filtering, and spectral analysis all live in it.
    """)
    return


@app.cell(hide_code=True)
def md_dct():
    mo.md(r"""
    ### The DCT

    Splitting each complex exponential into its cosine and sine parts gives a real Fourier basis: a constant **DC** term, then a normalized cosine/sine pair at each frequency $k$,

    $$
    \boldsymbol{\psi}_1 = \tfrac{1}{\sqrt{n}}\,\mathbf{1}, \qquad
    \boldsymbol{\psi}_{k}^{\cos} = \sqrt{\tfrac{2}{n}}\,\cos(2\pi k t), \qquad
    \boldsymbol{\psi}_{k}^{\sin} = \sqrt{\tfrac{2}{n}}\,\sin(2\pi k t).
    $$

    The **discrete cosine transform (DCT)**, introduced by Nasir Ahmed in 1974, goes one step further and keeps *only* the cosines. An even reflection of the signal at its edges avoids the artificial jump a periodic basis imposes at the boundary, and its energy-compacting cosines made it the workhorse behind **JPEG**, MPEG, and MP3 — among the most widely used transforms in computing. Each column is a single cosine of rising frequency,

    $$
    \boldsymbol{\psi}_k(i) \;\propto\; \cos\!\Big(\tfrac{\pi\,(2i + 1)\,k}{2n}\Big),
    $$

    with the $k = 0$ column normalized by $\sqrt{1/n}$ and the rest by $\sqrt{2/n}$.

    A coordinate $s_k = \boldsymbol{\psi}_k^{\top} \mathbf{x}$ asks *how much of wave $k$ is present in the signal.* Each column of $\Psi$ — Fourier or DCT — is one wave; the first few are plotted below.
    """)
    return


@app.cell
def demo_basis_vectors(n):
    def plot_basis_vectors(n):
        Psi_f, _ = fourier_basis(n)
        Psi_d = dct_basis(n)
        cols = 6
        t = np.arange(n)
        fig, axes = plt.subplots(2, cols, figsize=(11, 3.0), sharex=True, sharey=True)
        for k in range(cols):
            axes[0, k].plot(t, Psi_f[:, k], color="green", linewidth=1.4)
            axes[1, k].plot(t, Psi_d[:, k], color="green", linewidth=1.4)
            axes[0, k].set_title(rf"$\boldsymbol{{\psi}}_{{{k + 1}}}$", fontsize=10)
        axes[0, 0].set_ylabel("Fourier", fontsize=10)
        axes[1, 0].set_ylabel("DCT", fontsize=10)
        for ax in axes.ravel():
            ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
        return theme(fig)


    plot_basis_vectors(n)
    return


@app.cell(hide_code=True)
def md_build():
    mo.md(r"""
    ## 4. Building the signal one wave at a time

    A change of basis is a **recipe** for rebuilding the signal: stack the basis waves as the columns of $\Psi$, weight each by its coordinate $s_k$, and sum.

    $$
    \mathbf{x} = \Psi\,\mathbf{s} = \sum_{k} s_k\, \boldsymbol{\psi}_k.
    $$
    """)
    return


@app.cell
def demo_sum_equation(n, x):
    def plot_sum_equation(x, n):
        Psi, _ = fourier_basis(n)
        s = Psi.T @ x
        order = np.argsort(np.abs(s))[::-1]
        k_terms = 3
        idx = np.arange(n)
        fig, axes = plt.subplots(
            1, 1 + k_terms, figsize=(9, 3.4), sharey=True, gridspec_kw={"wspace": 0.7}
        )

        def vplot(ax, vals, color):
            ax.plot(vals, idx, color=color, linewidth=1.6)
            ax.axvline(0, color="gray", linewidth=0.6, alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "bottom"):
                ax.spines[side].set_visible(False)

        vplot(axes[0], x, "blue")
        axes[0].invert_yaxis()
        axes[0].set_title(r"$\mathbf{x}$", fontsize=13)

        for j in range(k_terms):
            k = order[j]
            vplot(axes[j + 1], s[k] * Psi[:, k], "green")
            axes[j + 1].set_title(rf"$s_{{{j + 1}}}\,\boldsymbol{{\psi}}_{{{j + 1}}}$", fontsize=13)
            op = "=" if j == 0 else "+"
            axes[j + 1].text(
                -0.35,
                0.5,
                op,
                transform=axes[j + 1].transAxes,
                ha="center",
                va="center",
                fontsize=17,
                clip_on=False,
            )
        axes[-1].text(
            1.25,
            0.5,
            r"$+\ \cdots$",
            transform=axes[-1].transAxes,
            ha="center",
            va="center",
            fontsize=15,
            clip_on=False,
        )
        return theme(fig)


    plot_sum_equation(x, n)
    return


@app.cell
def demo_matrix_equation(n, x):
    def plot_matrix_equation(x, n):
        Psi, _ = fourier_basis(n)
        s = Psi.T @ x
        fig, (ax_x, ax_p, ax_s) = plt.subplots(
            1,
            3,
            figsize=(6.8, 4.2),
            gridspec_kw={"width_ratios": [1, 9, 1], "wspace": 0.55},
        )

        def strip(ax, m, title):
            peak = np.abs(m).max()
            ax.imshow(m, cmap="RdBu_r", vmin=-peak, vmax=peak, aspect="auto")
            ax.set_title(title, fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])

        strip(ax_x, x[:, None], r"$\mathbf{x}$")
        strip(ax_p, Psi, r"$\Psi$")
        strip(ax_s, s[:, None], r"$\mathbf{s}$")
        ax_p.text(
            -0.10,
            0.5,
            "=",
            transform=ax_p.transAxes,
            ha="center",
            va="center",
            fontsize=17,
            clip_on=False,
        )
        ax_p.text(
            1.06,
            0.5,
            r"$\times$",
            transform=ax_p.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            clip_on=False,
        )
        return theme(fig)


    plot_matrix_equation(x, n)
    return


@app.cell(hide_code=True)
def md_sparsity():
    mo.md(r"""
    ## 5. The payoff: the right basis is sparse

    The figure below shows the same $64$-dimensional signal expressed in three bases. In the **standard** basis all $64$ coordinates matter. Rotated into the **DCT**, the energy concentrates in a few low-frequency coordinates. Rotated into the **Fourier** basis — exactly matched to this periodic signal — all but a handful of coordinates are **zero**:

    $$
    \mathbf{s}_{\text{Fourier}} \approx
    \big[\ \underbrace{s_1, \dots, s_r}_{\text{a few}},\ 0,\ \dots,\ 0\ \big]^{\top}.
    $$

    Same vector, same information, now described by $r \ll n$ numbers. **That is sparsity.** Beneath each panel is the number of coordinates needed to hold $99\%$ of the signal's energy.
    """)
    return


@app.cell
def demo_sparsity(n, x):
    def plot_sparsity(x, n):
        coords = {
            "standard": x,
            "DCT": dct_basis(n).T @ x,
            "Fourier": fourier_basis(n)[0].T @ x,
        }
        colors = {"standard": "blue", "DCT": "green", "Fourier": "red"}
        idx = np.arange(n)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
        for ax, (name, s) in zip(axes, coords.items()):
            draw_stem(ax, idx, s, colors[name], markersize=3)
            ax.set_title(
                rf"$\mathbf{{s}}$ in {name} — {coeffs_for_energy(s)} coords for 99% energy",
                fontsize=10,
            )
            ax.set_xlabel("coordinate $k$")
            style_signal_axes(ax)
        axes[0].set_ylabel("value")
        return theme(fig)


    plot_sparsity(x, n)
    return


@app.cell(hide_code=True)
def md_reconstruct():
    mo.md(r"""
    Those few nonzero coordinates are all it takes. Adding the Fourier waves back **in order of importance**, and because this signal is only a few Fourier cosines, a handful of terms does not merely approximate it — it reproduces it **exactly**:
    """)
    return


@app.cell
def demo_partial_sums(n, x):
    def plot_partial_sums(x, n):
        Psi, _ = fourier_basis(n)
        s = Psi.T @ x
        order = np.argsort(np.abs(s))[::-1]
        i = np.arange(n)
        steps = [1, 2, 3, 4]
        fig, axes = plt.subplots(1, len(steps), figsize=(11, 2.6), sharey=True)
        for ax, r in zip(axes, steps):
            s_r = np.zeros_like(s)
            s_r[order[:r]] = s[order[:r]]
            recon = Psi @ s_r
            err = np.linalg.norm(x - recon) / np.linalg.norm(x)
            ax.plot(i, x, color="gray", linewidth=1.0, alpha=0.6, zorder=1)
            ax.plot(i, recon, color="green", linewidth=1.8, zorder=2)
            ax.set_title(rf"$r={r}$   (error {err:.0%})", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            style_signal_axes(ax)
        return theme(fig)


    plot_partial_sums(x, n)
    return


@app.cell(hide_code=True)
def md_compression():
    mo.md(r"""
    ## 6. Keep the top $r$

    When most coordinates are near zero, discard them. Keep the $r$ largest, set the rest to zero to form $\mathbf{s}_r$, and rotate back:

    $$
    \hat{\mathbf{x}} = \Psi\,\mathbf{s}_r.
    $$

    Vary $r$ below and watch how few waves it takes to rebuild the signal. In the **Fourier** basis the error collapses to zero at $r = 4$; in the **DCT** it fades more gradually — the basis that *matches* the signal wins.
    """)
    return


@app.cell
def ui_controls(n):
    keep_r = mo.ui.slider(1, n, value=4, label=r"terms kept $r$", show_value=True, full_width=True)
    basis_pick = mo.ui.radio(options=["Fourier", "DCT"], value="Fourier", label="basis", inline=True)
    mo.vstack([basis_pick, keep_r])
    return basis_pick, keep_r


@app.cell
def compute_keep(basis_pick, keep_r, n, x):
    def compress(basis_name, r):
        Psi = dct_basis(n) if basis_name == "DCT" else fourier_basis(n)[0]
        s = Psi.T @ x
        s_kept = keep_top(s, r)
        recon = Psi @ s_kept
        error = float(np.linalg.norm(x - recon) / np.linalg.norm(x))
        energy = float(np.sum(s_kept**2) / np.sum(s**2))
        return s, s_kept, recon, error, energy


    s, s_kept, recon, keep_error, kept_energy = compress(basis_pick.value, keep_r.value)
    return keep_error, kept_energy, recon, s, s_kept


@app.cell
def demo_keep(basis_pick, keep_r, n, recon, s, s_kept, x):
    def plot_keep(x, n, recon, s, s_kept, basis_name, r):
        i = np.arange(n)
        fig, (ax_sig, ax_coef) = plt.subplots(1, 2, figsize=(11, 3.4))

        ax_sig.plot(i, x, color="gray", linewidth=1.2, alpha=0.6, label="original", zorder=1)
        ax_sig.plot(i, recon, color="green", linewidth=1.9, label="reconstruction", zorder=2)
        ax_sig.set_title("signal", fontsize=11)
        ax_sig.set_xlabel("sample index $i$")
        ax_sig.legend(frameon=False, fontsize=9)
        style_signal_axes(ax_sig)

        kept_mask = s_kept != 0
        draw_stem(ax_coef, i[~kept_mask], s[~kept_mask], "gray", markersize=3)
        draw_stem(ax_coef, i[kept_mask], s[kept_mask], "red", markersize=4)
        ax_coef.set_title(rf"{basis_name} coordinates — top {r} kept", fontsize=11)
        ax_coef.set_xlabel("coordinate $k$")
        style_signal_axes(ax_coef)
        return theme(fig)


    plot_keep(x, n, recon, s, s_kept, basis_pick.value, keep_r.value)
    return


@app.cell
def out_keep_stats(keep_error, keep_r, kept_energy, n):
    def show_keep_stats(r, n, kept_energy, keep_error):
        return mo.hstack(
            [
                mo.stat(f"{r} / {n}", label="terms kept"),
                mo.stat(f"{kept_energy:.2%}", label="energy retained"),
                mo.stat(f"{keep_error:.2%}", label="reconstruction error"),
            ],
            justify="start",
            gap=2,
        )


    show_keep_stats(keep_r.value, n, kept_energy, keep_error)
    return


@app.cell(hide_code=True)
def md_appendix():
    mo.md(r"""
    ## Appendix

    Helper functions supporting the figures above: figure theming, stem and axis styling, the Fourier and DCT bases, and the two operations behind compression — ranking coordinates by energy and keeping the largest.
    """)
    return


@app.function
def theme(fig):
    # Transparent so marimo's light/dark theme shows through and styles the text.
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.patch.set_alpha(0)
    fig.tight_layout()
    return fig


@app.function
def style_signal_axes(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


@app.function
def draw_stem(ax, idx, vals, color, markersize=4):
    markerline, stemline, baseline = ax.stem(idx, vals)
    plt.setp(markerline, color=color, markersize=markersize)
    plt.setp(stemline, color=color, linewidth=1.1)
    plt.setp(baseline, color=color, linewidth=0.8, alpha=0.35)


@app.function
def dct_basis(n):
    # Orthonormal DCT-II synthesis matrix; each column is a cosine of increasing
    # frequency. x = Psi @ s and s = Psi.T @ x, exactly.
    i = np.arange(n)
    k = np.arange(n)
    Psi = np.cos(np.pi * (2 * i[:, None] + 1) * k[None, :] / (2 * n))
    Psi[:, 0] *= np.sqrt(1 / n)
    Psi[:, 1:] *= np.sqrt(2 / n)
    return Psi


@app.function
def fourier_basis(n):
    # Real orthonormal Fourier basis: DC, then cos/sin pairs by rising frequency,
    # closing on the Nyquist wave. Columns are orthonormal for even n.
    t = np.arange(n) / n
    columns = [np.ones(n) / np.sqrt(n)]
    labels = ["DC"]
    for k in range(1, n // 2):
        columns.append(np.sqrt(2 / n) * np.cos(2 * np.pi * k * t))
        columns.append(np.sqrt(2 / n) * np.sin(2 * np.pi * k * t))
        labels += [rf"$\cos {k}$", rf"$\sin {k}$"]
    columns.append(np.cos(np.pi * np.arange(n)) / np.sqrt(n))
    labels.append("Nyq")
    return np.column_stack(columns), labels


@app.function
def coeffs_for_energy(s, frac=0.99):
    # How many of the largest-magnitude coordinates capture `frac` of the total energy.
    energy = np.sort(np.asarray(s, np.float64) ** 2)[::-1]
    total = energy.sum()
    if total == 0:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, frac) + 1)


@app.function
def keep_top(s, r):
    kept = np.zeros_like(s)
    order = np.argsort(np.abs(s))[::-1][:r]
    kept[order] = s[order]
    return kept


if __name__ == "__main__":
    app.run()
