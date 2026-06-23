# /// script
# requires-python = ">=3.14"
# dependencies = [
#       "marimo",
#       "numpy",
#       "plotly"
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates.default = "plotly_dark"
    return go, np


@app.cell
def _(mo):
    freq = mo.ui.slider(1, 20, value=3, label="frequency")
    freq
    return (freq,)


@app.cell
def _(freq, go, np):
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(freq.value * x)
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(title=f"sin({freq.value} * 2π x)")
    fig
    return


if __name__ == "__main__":
    app.run()
