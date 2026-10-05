# Mandelbulb

A Python Mandelbulb raymarcher with an interactive Plotly Dash explorer and a [**real-time WebGL viewer**](https://epa058.github.io/Mandelbulb/web/). This project was inspired by [Dash-Fractal-Explorer](https://github.com/SterlingButters/Dash-Fractal-Explorer) and aimed to reproduce the Mandelbulb renders on [Skytopia](https://www.skytopia.com/project/fractal/mandelbulb.html).

| Classic power 8         | Zoom ×4.9            | Power 4                |
| ----------------------- | -------------------- | ---------------------- |
| ![](images/classic.png) | ![](images/zoom.png) | ![](images/power4.png) |

## Quick start

Install the dependencies and launch the Dash explorer:

```bash
pip install -r requirements.txt
python app.py
```

---

## Interactive Explorer

The main explorer is implemented in `app.py` using Plotly Dash.

### Navigation

* **Click the image** to fly toward a point on the fractal.
* Choose the click action:

  * **Zoom in ×2**
  * **Recentre**
  * **Zoom out**
* The camera is placed along the ray used to reach the clicked point, so it remains outside the fractal (supposedly).

### Fractal settings

* **Power:** 2–16
* **Iterations:** controls the number of distance-estimator iterations
* **Detail:** controls the hit tolerance in pixels
* **Julia mode:** switches from the Mandelbulb to Julia variants and adjusts the parameter `c`
* **Presets:** provides useful starting points for exploring different regions

### Camera

The camera controls include:

* Azimuth
* Elevation
* Field of view
* Zoom in
* Zoom out
* Reset

The readout below the image shows the current zoom, target, azimuth, and elevation. These values can be copied directly into `render.py` to reproduce the view.

---

## Real-time WebGL Explorer

The project also includes a standalone WebGL 2 renderer in `web/index.html`. Open the file directly in any browser with WebGL 2 support.

### Orbit mode

* **Drag:** orbit around the fractal
* **Scroll / pinch:** zoom
* **Double-click / double-tap:** fly toward a point
* **H:** hide/show the control panel
* **R:** reset the camera

### Fly mode

Press **F** to enter first-person fly mode. Click the view to capture the mouse, then use:

| Key               | Action                                |
| ----------------- | ------------------------------------- |
| `W` / `S`         | Move toward / away from the crosshair |
| `A` / `D`         | Strafe left / right                   |
| `Space` / `Shift` | Move up / down                        |
| `Esc`             | Release the cursor                    |

Scrolling changes the movement speed. Switching back to orbit mode centres the orbit on whatever point the crosshair is currently targeting.

### Animation

The WebGL explorer supports:

* **Auto-rotate:** ... auto-rotates
* **Breathe power:** continuously varies the Mandelbulb power between 4.5 and 11.5
* **Orbit Julia c:** continuously changes the Julia parameter and morphs the resulting fractal

### Snapshots and reproducible renders

* **Snapshot:** saves the current converged frame as a PNG.
* **Copy render.py command:** generates the exact `render.py` command needed to reproduce the current view at print resolution.

The WebGL renderer uses 32-bit floating-point arithmetic. Fine detail begins to break down at roughly ×3000 zoom. For deeper zooms, use the Python renderer.

---

## Command-line Renderer

`render.py` provides a command-line interface for producing high-resolution images and animations.

### Render a still image

```bash
python render.py still -o bulb.png --size 2048 --ss 2
```

Render a specific camera position:

```bash
python render.py still \
    --target 0.5443 0.1203 0.5253 \
    --az 33.5 \
    --el 20 \
    --zoom 4.9 \
    --palette Ember
```

### Create a turntable animation

```bash
python render.py turntable \
    --frames 120 \
    --size 512 \
    --gif spin.gif
```

### Morph between powers

```bash
python render.py morph \
    --from-power 2 \
    --to-power 12 \
    --frames 90 \
    --gif morph.gif
```

### Create a zoom animation

```bash
python render.py zoom \
    --target 0.5443 0.1203 0.5253 \
    --az 33.5 \
    --el 20 \
    --zoom-to 500 \
    --gif dive.gif
```

Run the following to see all available options:

```bash
python render.py <command> -h
```

For `zoom`, use the target, azimuth, and elevation reported by the Dash explorer after clicking a point. This ensures that the camera follows a path through empty space rather than entering the fractal.

---

## How It Works

The core renderer is implemented in `mandelbulb/renderer.py`. It uses a sphere-tracing raymarcher compiled with Numba:

### 1. Distance estimator

The Mandelbulb is evaluated using the White–Nylander triplex power formula together with a running derivative. For each iteration:

1. Convert `z` to spherical coordinates `(r, θ, φ)`.

2. Apply the power transformation:

```math
   z \leftarrow r^n
   \begin{pmatrix}
   \sin(n\theta)\cos(n\phi) \\
   \sin(n\theta)\sin(n\phi) \\
   \cos(n\theta)
   \end{pmatrix} + c.
```

3. Update the derivative:

```math
   dr \leftarrow n\,r^{n-1}\,dr + 1.
```

4. Estimate the distance to the surface using

```math
   DE = \frac{1}{2}\,\frac{r \, \ln r}{dr}.
```

### 2. Raymarching

Each camera ray is first clipped to a bounding sphere. The renderer then advances the ray using the distance estimator:

```text
step = DE × fudge
```

The hit tolerance is scaled according to the distance and pixel footprint. This keeps the apparent level of detail approximately constant as the camera zooms in and out.

### 3. Shading

Surface normals are calculated using a tetrahedral gradient.

The lighting model combines:

* Soft shadows
* A key light
* Sky ambient light
* Fill lighting
* Blinn–Phong specular highlights
* Fresnel rim lighting

Ambient occlusion is computed from five samples along the surface normal, together with a term based on the raymarch step count.

### 4. Colour

The surface colour is based on an orbit trap using the minimum value of `|z|` encountered during the distance-estimator iterations. This value is mapped through an [Inigo Quilez cosine palette](https://iquilezles.org/articles/palettes/) to produce the final colour.

### 5. Background and glow

Rays that do not hit the fractal receive a background gradient. A glow effect is added according to how closely the ray passed to the fractal surface.

---

## Project Structure

```text
app.py                  Dash interactive explorer
web/index.html          Standalone WebGL 2 explorer
render.py               Command-line renderer
mandelbulb/
    renderer.py         Raymarcher, distance estimator, camera and picking
    palettes.py         Cosine colour palettes
assets/
    style.css           Dash application styling
images/                 Example renders
```

