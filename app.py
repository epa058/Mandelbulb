"""
Mandelbulb Explorer – a Plotly Dash front end for the Numba raymarcher.

    python app.py            ->  http://127.0.0.1:8050

Click the image to fly toward that point (or recentre / zoom out, depending on
the click mode). Sliders control the fractal, camera and shading; "Download
PNG" renders a supersampled high-resolution still.
"""

import base64
import io
import threading
import time

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import plotly.graph_objects as go
from PIL import Image

from mandelbulb import PALETTES, DEFAULT_PALETTE, Camera, pick, render, warmup

DEFAULT_VIEW = {"target": [0.0, 0.0, 0.0], "dist": 3.6}
DEFAULT_AZ, DEFAULT_EL = 30.0, 25.0
BASE_DIST = 3.6

PRESETS = {
    "Classic (power 8)": dict(power=8, iters=12, julia=[], palette="Skytopia"),
    "Power 2 – Nylander": dict(power=2, iters=16, julia=[], palette="Bone"),
    "Power 4": dict(power=4, iters=12, julia=[], palette="Ember"),
    "Power 12": dict(power=12, iters=10, julia=[], palette="Amethyst"),
    "Julia bulb": dict(power=8, iters=12, julia=["on"], palette="Ocean"),
}

# Numba's default thread pool is not re-entrant; Dash serves requests on
# several threads, so serialise renders.
RENDER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_data_uri(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_camera(view, az, el, fov):
    return Camera(target=view["target"], azimuth=az, elevation=el,
                  distance=view["dist"], fov=fov)


def fractal_kwargs(p):
    return dict(
        power=float(p["power"]),
        iterations=int(p["iters"]),
        julia=bool(p["julia"]),
        julia_c=(float(p["cx"]), float(p["cy"]), float(p["cz"])),
        detail=float(p["detail"]),
    )


def shading_kwargs(p):
    flags = p["flags"] or []
    return dict(
        shadows="shadows" in flags,
        ao="ao" in flags,
        glow=float(p["glow"]) if "glow" in flags else 0.0,
        color_freq=float(p["cfreq"]),
        color_offset=float(p["coffset"]),
        fog=0.6 if "fog" in flags else 0.0,
    )


def empty_figure(uri, size):
    fig = go.Figure(go.Image(source=uri, colormodel="rgba", hoverinfo="none"))
    fig.update_layout(
        width=size, height=size,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[-0.5, size - 0.5]),
        yaxis=dict(visible=False, range=[size - 0.5, -0.5], scaleanchor="x"),
        dragmode=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def section(title, *children):
    return html.Div(className="section", children=[html.H3(title), *children])


def labeled(label, component):
    return html.Div(className="control", children=[html.Label(label), component])


def slider(id_, lo, hi, step, value, marks=None):
    return dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value,
                      marks=marks or {lo: str(lo), hi: str(hi)},
                      tooltip={"placement": "bottom", "always_visible": False},
                      updatemode="mouseup")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app = Dash(__name__, title="Mandelbulb Explorer")

controls = html.Div(className="panel", children=[
    html.H1("Mandelbulb Explorer"),
    html.P("Click the image to fly toward a point.", className="hint"),

    section(
        "Fractal",
        labeled("Preset", dcc.Dropdown(id="preset", options=list(PRESETS),
                                       placeholder="Choose a preset…", clearable=True)),
        labeled("Power", slider("power", 2, 16, 0.5, 8, {2: "2", 8: "8", 16: "16"})),
        labeled("Iterations", slider("iters", 3, 40, 1, 12, {3: "3", 12: "12", 40: "40"})),
        labeled("Detail (lower = finer, slower)",
                slider("detail", 0.25, 4, 0.25, 1.0, {0.25: "¼", 1: "1", 4: "4"})),
        dcc.Checklist(id="julia", options=[{"label": " Julia mode", "value": "on"}],
                      value=[], className="check"),
        html.Div(id="julia-box", style={"display": "none"}, children=[
            labeled("c.x", slider("cx", -1.2, 1.2, 0.01, -0.6)),
            labeled("c.y", slider("cy", -1.2, 1.2, 0.01, -0.4)),
            labeled("c.z", slider("cz", -1.2, 1.2, 0.01, 0.4)),
        ]),
    ),

    section(
        "Camera",
        labeled("Click action", dcc.RadioItems(
            id="click-mode", value="zoom", inline=True, className="radio",
            options=[{"label": " Zoom in ×2", "value": "zoom"},
                     {"label": " Recentre", "value": "center"},
                     {"label": " Zoom out", "value": "out"}])),
        labeled("Azimuth °", slider("az", 0, 360, 0.5, DEFAULT_AZ, {0: "0", 180: "180", 360: "360"})),
        labeled("Elevation °", slider("el", -89, 89, 0.5, DEFAULT_EL, {-89: "-89", 0: "0", 89: "89"})),
        labeled("Field of view °", slider("fov", 15, 90, 1, 40, {15: "15", 40: "40", 90: "90"})),
        html.Div(className="buttons", children=[
            html.Button("Zoom in", id="btn-in"),
            html.Button("Zoom out", id="btn-out"),
            html.Button("Reset view", id="btn-reset"),
        ]),
    ),

    section(
        "Look",
        labeled("Palette", dcc.Dropdown(id="palette", options=list(PALETTES),
                                        value=DEFAULT_PALETTE, clearable=False)),
        labeled("Colour frequency", slider("cfreq", 0.0, 4.0, 0.05, 1.4, {0: "0", 4: "4"})),
        labeled("Colour offset", slider("coffset", 0.0, 1.0, 0.01, 0.1, {0: "0", 1: "1"})),
        labeled("Glow", slider("glow", 0.0, 1.5, 0.05, 0.4, {0: "0", 1.5: "1.5"})),
        dcc.Checklist(
            id="flags", value=["shadows", "ao", "glow"], inline=True, className="check",
            options=[{"label": " Shadows", "value": "shadows"},
                     {"label": " Ambient occlusion", "value": "ao"},
                     {"label": " Glow", "value": "glow"},
                     {"label": " Fog", "value": "fog"}]),
        labeled("Preview size", dcc.RadioItems(
            id="res", value=480, inline=True, className="radio",
            options=[{"label": f" {r}", "value": r} for r in (320, 480, 640, 800)])),
    ),

    section(
        "Export",
        labeled("Size (px)", dcc.Dropdown(
            id="export-size", value=1600, clearable=False,
            options=[{"label": f"{s} × {s}", "value": s} for s in (1024, 1600, 2048, 3072)])),
        labeled("Anti-aliasing", dcc.RadioItems(
            id="export-ss", value=2, inline=True, className="radio",
            options=[{"label": " off", "value": 1}, {"label": " 2×2", "value": 2},
                     {"label": " 3×3", "value": 3}])),
        html.Button("Download PNG", id="btn-export", className="primary"),
        dcc.Download(id="download"),
    ),
])

viewer = html.Div(className="viewer", children=[
    dcc.Loading(type="circle", color="#d9a066", delay_show=250, children=dcc.Graph(
        id="graph", config={"displayModeBar": False, "scrollZoom": False},
        figure=empty_figure("", 480))),
    html.Div(id="status", className="status"),
    html.Div(id="readout", className="readout"),
])

app.layout = html.Div(className="app", children=[
    dcc.Store(id="view", data=DEFAULT_VIEW),
    viewer,
    controls,
])

# Shared input groups --------------------------------------------------------
FRACTAL_INPUTS = dict(power=Input("power", "value"), iters=Input("iters", "value"),
                      detail=Input("detail", "value"), julia=Input("julia", "value"),
                      cx=Input("cx", "value"), cy=Input("cy", "value"), cz=Input("cz", "value"))
SHADING_INPUTS = dict(palette=Input("palette", "value"), cfreq=Input("cfreq", "value"),
                      coffset=Input("coffset", "value"), glow=Input("glow", "value"),
                      flags=Input("flags", "value"))
CAMERA_INPUTS = dict(view=Input("view", "data"), az=Input("az", "value"),
                     el=Input("el", "value"), fov=Input("fov", "value"))


def as_states(group):
    return {k: State(v.component_id, v.component_property) for k, v in group.items()}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(Output("julia-box", "style"), Input("julia", "value"))
def toggle_julia(julia):
    return {"display": "block"} if julia else {"display": "none"}


@app.callback(
    Output("power", "value"), Output("iters", "value"),
    Output("julia", "value"), Output("palette", "value"),
    Input("preset", "value"), prevent_initial_call=True)
def apply_preset(name):
    if not name:
        return no_update, no_update, no_update, no_update
    p = PRESETS[name]
    return p["power"], p["iters"], p["julia"], p["palette"]


@app.callback(
    Output("view", "data"), Output("az", "value"), Output("el", "value"),
    Output("graph", "clickData"),
    inputs=dict(
        click=Input("graph", "clickData"),
        n_in=Input("btn-in", "n_clicks"),
        n_out=Input("btn-out", "n_clicks"),
        n_reset=Input("btn-reset", "n_clicks"),
        mode=State("click-mode", "value"),
        res=State("res", "value"),
        cam=as_states(CAMERA_INPUTS),
        frac=as_states(FRACTAL_INPUTS),
    ),
    prevent_initial_call=True,
)
def navigate(click, n_in, n_out, n_reset, mode, res, cam, frac):
    trig = ctx.triggered_id
    view = dict(cam["view"])
    az, el = cam["az"], cam["el"]

    if trig == "btn-reset":
        return DEFAULT_VIEW, DEFAULT_AZ, DEFAULT_EL, None
    if trig == "btn-out" or (trig == "graph" and click and mode == "out"):
        view["dist"] = min(view["dist"] * 2.0, BASE_DIST * 2.0)
        return view, no_update, no_update, None

    # Zoom in / recentre: cast a ray through the chosen pixel
    if trig == "btn-in":
        x = y = (res - 1) / 2.0
        mode = "zoom"
    elif trig == "graph" and click:
        pt = click["points"][0]
        x, y = float(pt["x"]), float(pt["y"])
    else:
        return no_update, no_update, no_update, None

    camera = make_camera(view, az, el, cam["fov"])
    with RENDER_LOCK:
        hit = pick(camera, x, y, res, res, **fractal_kwargs(frac))
    if hit is None:
        return no_update, no_update, no_update, None
    point, t, ray_dir = hit

    # New camera sits on the ray we just marched (guaranteed empty space),
    # looking straight at the hit point.
    new_az, new_el = Camera.angles_from_direction(ray_dir)
    new_dist = t * (0.5 if mode == "zoom" else 1.0)
    return {"target": point.tolist(), "dist": new_dist}, round(new_az, 2), round(new_el, 2), None


@app.callback(
    Output("graph", "figure"), Output("status", "children"), Output("readout", "children"),
    inputs=dict(res=Input("res", "value"), cam=CAMERA_INPUTS,
                frac=FRACTAL_INPUTS, shade=SHADING_INPUTS),
)
def update_image(res, cam, frac, shade):
    camera = make_camera(cam["view"], cam["az"], cam["el"], cam["fov"])
    t0 = time.perf_counter()
    with RENDER_LOCK:
        img = render(camera, res, res, palette=shade["palette"],
                     **fractal_kwargs(frac), **shading_kwargs(shade))
    dt = time.perf_counter() - t0
    zoom = BASE_DIST / cam["view"]["dist"]
    tx, ty, tz = cam["view"]["target"]
    status = f"Rendered {res}×{res} in {dt:.2f} s"
    readout = [
        html.Span(f"zoom ×{zoom:,.1f}" if zoom < 1e4 else f"zoom ×{zoom:.2e}"),
        html.Span(f"target ({tx:.6g}, {ty:.6g}, {tz:.6g})"),
        html.Span(f"az {cam['az']:.1f}°  el {cam['el']:.1f}°"),
    ]
    if zoom > 64 and frac["iters"] < 16:
        readout.append(html.Span("tip: raise iterations when zoomed in", className="tip"))
    return empty_figure(to_data_uri(img), res), status, readout


@app.callback(
    Output("download", "data"),
    inputs=dict(n=Input("btn-export", "n_clicks"),
                size=State("export-size", "value"), ss=State("export-ss", "value"),
                cam=as_states(CAMERA_INPUTS), frac=as_states(FRACTAL_INPUTS),
                shade=as_states(SHADING_INPUTS)),
    prevent_initial_call=True,
)
def export(n, size, ss, cam, frac, shade):
    camera = make_camera(cam["view"], cam["az"], cam["el"], cam["fov"])
    with RENDER_LOCK:
        img = render(camera, size, size, palette=shade["palette"], supersample=ss,
                     **fractal_kwargs(frac), **shading_kwargs(shade))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    zoom = BASE_DIST / cam["view"]["dist"]
    name = f"mandelbulb_p{frac['power']:g}_z{zoom:.3g}_{size}.png"
    return dcc.send_bytes(buf.getvalue(), name)


if __name__ == "__main__":
    print("Compiling renderer (first run only takes ~30 s; cached afterwards)…")
    warmup()
    app.run(debug=False, threaded=True)
