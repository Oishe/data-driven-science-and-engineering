"""Visualize a marimo notebook's cell dependency DAG as standalone HTML.

uv run python depgraph.py app.py -o depgraph.html
uv run python depgraph.py app.py -o depgraph.html --nodrop-setup
uv run python depgraph.py app.py -o depgraph.html --drop-types ui_out
uv run python depgraph.py app.py -o depgraph.html --nodrop-hidden

Node `type` is derived from the cell: the setup cell from `setup_cell_id`, every
other cell from its name prefix (md_ / ui_in_ / ui_out_ / compute_). Nodes are
styled per type via a pyvis group, with a matching HTML legend; filter nodes out
by type or by hidden-code config.
"""

import importlib.util
from pathlib import Path

from pyvis.network import Network
import fire

# Role -> (background, border) node colors. Roles come from the cell-name prefix
# convention in app.py (setup is special-cased); unrecognized -> "compute".
TYPE_COLOR = {
    "setup": ("#9e9e9e", "#616161"),  # gray
    "md": ("#81c784", "#388e3c"),  # green
    "ui_in": ("#64b5f6", "#1976d2"),  # blue
    "ui_out": ("#b39ddb", "#7e57c2"),  # lavender
    "*plot": ("#b39ddb", "#7e57c2"),  # lavender
    "compute": ("#ffb74d", "#f57c00"),  # amber
}
ROLES = tuple(t for t in TYPE_COLOR if t != "setup")


def load_app(notebook="app.py"):
    path = Path(notebook).resolve()
    spec = importlib.util.spec_from_file_location(f"_nb_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = module.app
    app._maybe_initialize()
    return app


def node_type(app, cid):
    if cid == app._cell_manager.setup_cell_id:
        return "setup"
    name = app._cell_manager.cell_name(cid)
    for role in ROLES:
        if name.startswith(role + "_"):
            return role
    return "compute"  # fallback for unconventional names


def extract(
    notebook="app.py",
    drop_setup=True,
    drop_types=(),
    drop_hidden=True,
):
    app = load_app(notebook)
    g = app._graph
    name = app._cell_manager.cell_name

    drop_types = set(drop_types.split(",")) if isinstance(drop_types, str) else set(drop_types)
    types = {cid: node_type(app, cid) for cid in g.cells}

    def keep_cell(cid):
        if drop_setup and types[cid] == "setup":
            return False
        if types[cid] in drop_types:
            return False
        if drop_hidden and g.cells[cid].config.hide_code:
            return False
        return True

    keep = {cid for cid in g.cells if keep_cell(cid)}
    nodes = [{"id": str(cid), "label": name(cid), "type": types[cid]} for cid in keep]

    edges = [
        {
            "source": str(cid),
            "target": str(child),
            "label": ", ".join(sorted(cell.defs & g.cells[child].refs)),
        }
        for cid, cell in g.cells.items()
        if cid in keep
        for child in g.children.get(cid, set())
        if child in keep
    ]
    return nodes, edges


def to_pyvis(nodes, edges, height="100vh", width="100%"):

    def _group_color(bg, bd):
        c = {"background": bg, "border": bd}
        return {"color": {**c, "highlight": c, "hover": c}}

    net = Network(height=height, width=width, directed=True, bgcolor="#222222")
    net.options.groups = {t: _group_color(bg, bd) for t, (bg, bd) in TYPE_COLOR.items()}
    for n in nodes:
        net.add_node(
            n["id"],
            label=n["label"],
            shape="box",
            size=50,
            group=n["type"],
        )
    for e in edges:
        net.add_edge(
            e["source"],
            e["target"],
            title=e["label"],
            width=2,
        )
    net.barnes_hut(
        gravity=-8000,  # 10x less repulsion
        central_gravity=0.5,  # pull harder toward center
        spring_length=110,  # shorter edges
        spring_strength=0.04,  # stiffer springs hold neighbors close
    )
    # net.show_buttons(filter_=["layout", "physics"])
    return net


def export_html(
    notebook="app.py",
    out="depgraph.html",
    drop_setup=True,
    drop_types=(),
    drop_hidden=True,
):
    nodes, edges = extract(
        notebook,
        drop_setup=drop_setup,
        drop_types=drop_types,
        drop_hidden=drop_hidden,
    )
    to_pyvis(nodes, edges).save_graph(out)

    html = Path(out).read_text()
    Path(out).write_text(html)
    return f"wrote {out}  ({len(nodes)} cells, {len(edges)} edges)"


if __name__ == "__main__":
    fire.Fire(export_html)
