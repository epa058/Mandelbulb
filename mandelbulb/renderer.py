"""
Mandelbulb raymarcher (CPU, Numba-parallel).

Rendering pipeline per pixel
----------------------------
1. Build a primary ray from the camera.
2. Clip it against a bounding sphere, then sphere-trace the fractal using the
   analytic distance estimator  DE = 0.5 * log(r) * r / dr  (White/Nylander
   triplex power formula with running derivative dr).
3. On a hit: tetrahedral-gradient normal, soft shadow, ambient occlusion,
   Blinn-Phong + sky + rim lighting, orbit-trap colouring through a cosine
   palette.
4. On a miss: background gradient plus a glow driven by how close the ray
   passed to the surface.

All world-space tolerances scale with the camera distance / pixel footprint so
the same code works from the overview down to deep zooms (float64 throughout).
"""

import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Parameter vector layout (keeps Numba signatures small)
# ---------------------------------------------------------------------------
P_POWER, P_ITERS, P_BAILOUT, P_JULIA, P_JX, P_JY, P_JZ = 0, 1, 2, 3, 4, 5, 6
P_STEPS, P_DETAIL, P_FUDGE, P_SHADOWS, P_AO, P_GLOW = 7, 8, 9, 10, 11, 12
P_CFREQ, P_COFFSET, P_BOUND, P_SPEC, P_SCALE, P_FOG = 13, 14, 15, 16, 17, 18
N_PARAMS = 19


# ---------------------------------------------------------------------------
# Distance estimator
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def distance_estimate(x, y, z, P):
    """Return (distance, orbit_trap) for the power-n Mandelbulb / Julia bulb."""
    power = P[P_POWER]
    max_iter = int(P[P_ITERS])
    bailout = P[P_BAILOUT]
    if P[P_JULIA] > 0.5:
        cx, cy, cz = P[P_JX], P[P_JY], P[P_JZ]
        plus = 0.0
    else:
        cx, cy, cz = x, y, z
        plus = 1.0

    zx, zy, zz = x, y, z
    dr = 1.0
    trap = 1e20
    for _ in range(max_iter):
        r = math.sqrt(zx * zx + zy * zy + zz * zz)
        if r > bailout:
            break
        if r < trap:
            trap = r
        if r < 1e-300:
            theta = 0.0
            phi = 0.0
        else:
            theta = math.acos(max(-1.0, min(1.0, zz / r)))
            phi = math.atan2(zy, zx)
        rp1 = r ** (power - 1.0)
        dr = rp1 * power * dr + plus
        zr = rp1 * r
        theta *= power
        phi *= power
        st = math.sin(theta)
        zx = zr * st * math.cos(phi) + cx
        zy = zr * st * math.sin(phi) + cy
        zz = zr * math.cos(theta) + cz

    r = math.sqrt(zx * zx + zy * zy + zz * zz)
    if r < 1e-300 or dr == 0.0:
        return 0.0, trap
    return 0.5 * math.log(r) * r / dr, trap


@njit(cache=True, fastmath=True)
def _de(x, y, z, P):
    d, _ = distance_estimate(x, y, z, P)
    return d


# ---------------------------------------------------------------------------
# Ray marching
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def march(ox, oy, oz, dx, dy, dz, pix, P):
    """Sphere-trace one ray.

    Returns (t_hit or -1, steps, min_distance_ratio, trap).
    `pix` is the angular size of a pixel (radians) – sets the hit tolerance.
    """
    R = P[P_BOUND]
    b = ox * dx + oy * dy + oz * dz
    c = ox * ox + oy * oy + oz * oz - R * R
    disc = b * b - c
    sq = math.sqrt(max(disc, 0.0))
    t1 = -b + sq
    if disc < 0.0 or t1 < 0.0:
        # Missed the bounding volume: estimate glow at closest approach so the
        # halo stays continuous across the sphere's silhouette.
        tc = max(-b, 1e-12)
        d = _de(ox + dx * tc, oy + dy * tc, oz + dz * tc, P)
        return -1.0, 0, d / tc, 0.0
    t0 = max(0.0, -b - sq)

    max_steps = int(P[P_STEPS])
    detail = P[P_DETAIL]
    fudge = P[P_FUDGE]
    t = t0
    min_ratio = 1e20
    for s in range(max_steps):
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        d, trap = distance_estimate(px, py, pz, P)
        eps = max(t * pix * detail, 1e-14)
        if d < eps:
            return t, s, 0.0, trap
        ratio = d / max(t, 1e-12)
        if ratio < min_ratio:
            min_ratio = ratio
        t += d * fudge
        if t > t1:
            return -1.0, s, min_ratio, 0.0
    # Ran out of steps while still approaching -> treat as a (fuzzy) hit
    return t, max_steps, 0.0, trap


@njit(cache=True, fastmath=True)
def _normal(px, py, pz, h, P):
    # Tetrahedral central differences: 4 DE evaluations
    d1 = _de(px + h, py - h, pz - h, P)
    d2 = _de(px - h, py - h, pz + h, P)
    d3 = _de(px - h, py + h, pz - h, P)
    d4 = _de(px + h, py + h, pz + h, P)
    nx = d1 - d2 - d3 + d4
    ny = -d1 - d2 + d3 + d4
    nz = -d1 + d2 - d3 + d4
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln < 1e-300:
        return 0.0, 0.0, 1.0
    return nx / ln, ny / ln, nz / ln


@njit(cache=True, fastmath=True)
def _soft_shadow(px, py, pz, lx, ly, lz, eps, P):
    k = 10.0
    res = 1.0
    t = eps * 4.0
    limit = 2.0 * P[P_BOUND]
    for _ in range(96):
        h = _de(px + lx * t, py + ly * t, pz + lz * t, P)
        if h < eps * 0.5:
            return 0.0
        res = min(res, k * h / t)
        t += max(h * 0.9, eps)
        if t > limit:
            break
    return max(0.0, min(1.0, res))


@njit(cache=True, fastmath=True)
def _ambient_occlusion(px, py, pz, nx, ny, nz, scale, P):
    occ = 0.0
    w = 1.0
    for i in range(5):
        h = scale * (0.01 + 0.12 * i / 4.0)
        d = _de(px + nx * h, py + ny * h, pz + nz * h, P)
        occ += (h - d) * w
        w *= 0.75
    return max(0.0, min(1.0, 1.0 - 3.0 * occ / scale))


@njit(cache=True, fastmath=True)
def _palette(t, pal):
    # Inigo Quilez cosine palette: a + b*cos(2pi(c*t + d))
    r = pal[0, 0] + pal[1, 0] * math.cos(6.283185307 * (pal[2, 0] * t + pal[3, 0]))
    g = pal[0, 1] + pal[1, 1] * math.cos(6.283185307 * (pal[2, 1] * t + pal[3, 1]))
    b = pal[0, 2] + pal[1, 2] * math.cos(6.283185307 * (pal[2, 2] * t + pal[3, 2]))
    return max(r, 0.0), max(g, 0.0), max(b, 0.0)


@njit(cache=True, fastmath=True)
def _shade(ox, oy, oz, dx, dy, dz, pix, cam, P, pal):
    """Colour (linear RGB) for one primary ray."""
    fx, fy, fz = cam[3], cam[4], cam[5]
    rx, ry, rz = cam[6], cam[7], cam[8]
    ux, uy, uz = cam[9], cam[10], cam[11]
    scale = P[P_SCALE]

    # Background gradient (by ray elevation relative to camera up)
    v = 0.5 + 0.5 * (dx * ux + dy * uy + dz * uz)
    bg_r = 0.015 + 0.05 * v
    bg_g = 0.017 + 0.055 * v
    bg_b = 0.030 + 0.090 * v

    t, steps, min_ratio, trap = march(ox, oy, oz, dx, dy, dz, pix, P)

    if t < 0.0:
        g = P[P_GLOW] * math.exp(-min_ratio / (12.0 * pix * P[P_DETAIL] + 0.004))
        gr, gg, gb = _palette(P[P_COFFSET] + 0.15, pal)
        return bg_r + g * gr * 0.6, bg_g + g * gg * 0.6, bg_b + g * gb * 0.6

    px = ox + dx * t
    py = oy + dy * t
    pz = oz + dz * t
    eps = max(t * pix * P[P_DETAIL], 1e-14)
    nx, ny, nz = _normal(px, py, pz, eps * 0.5, P)
    # Make sure the normal faces the viewer (numerical safety)
    if nx * dx + ny * dy + nz * dz > 0.0:
        nx, ny, nz = -nx, -ny, -nz

    # Key light: above-left, slightly behind the camera (camera-relative)
    lx = -0.55 * rx + 0.75 * ux - 0.45 * fx
    ly = -0.55 * ry + 0.75 * uy - 0.45 * fy
    lz = -0.55 * rz + 0.75 * uz - 0.45 * fz
    ll = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx /= ll
    ly /= ll
    lz /= ll

    # Lift the shading point off the surface a little
    sx = px + nx * eps * 2.0
    sy = py + ny * eps * 2.0
    sz = pz + nz * eps * 2.0

    ndl = nx * lx + ny * ly + nz * lz
    shadow = 1.0
    if P[P_SHADOWS] > 0.5 and ndl > 0.0:
        shadow = _soft_shadow(sx, sy, sz, lx, ly, lz, eps, P)

    ao = 1.0
    if P[P_AO] > 0.5:
        ao = _ambient_occlusion(sx, sy, sz, nx, ny, nz, scale, P)
    # Cheap "step-count" occlusion adds crevice darkening
    ao *= 1.0 - 0.6 * (steps / P[P_STEPS])

    # Albedo from orbit trap
    ct = P[P_CFREQ] * math.log(trap + 1e-6) + P[P_COFFSET]
    ar, ag, ab = _palette(ct, pal)

    diff = max(ndl, 0.0) * shadow
    hx, hy, hz = lx - dx, ly - dy, lz - dz
    hl = math.sqrt(hx * hx + hy * hy + hz * hz)
    spec = 0.0
    if hl > 0.0:
        ndh = max(0.0, (nx * hx + ny * hy + nz * hz) / hl)
        spec = P[P_SPEC] * (ndh ** 40.0) * shadow
    sky = 0.5 + 0.5 * (nx * ux + ny * uy + nz * uz)
    fill = max(0.0, -(nx * lx + ny * ly + nz * lz)) * 0.25
    fres = (1.0 + (nx * dx + ny * dy + nz * dz)) ** 3

    kr = 1.15 * diff * 1.00 + (0.28 * sky * 0.55 + fill * 0.6) * ao
    kg = 1.15 * diff * 0.95 + (0.28 * sky * 0.65 + fill * 0.5) * ao
    kb = 1.15 * diff * 0.85 + (0.28 * sky * 0.90 + fill * 0.4) * ao

    cr = ar * kr * (0.35 + 0.65 * ao) + spec + 0.25 * fres * ao * ar
    cg = ag * kg * (0.35 + 0.65 * ao) + spec + 0.25 * fres * ao * ag
    cb = ab * kb * (0.35 + 0.65 * ao) + spec + 0.25 * fres * ao * ab

    # Optional depth fog relative to the view scale
    fog = P[P_FOG]
    if fog > 0.0:
        f = 1.0 - math.exp(-fog * max(0.0, t / (scale * 10.0) - 0.6))
        cr = cr * (1.0 - f) + bg_r * f
        cg = cg * (1.0 - f) + bg_g * f
        cb = cb * (1.0 - f) + bg_b * f
    return cr, cg, cb


@njit(cache=True, parallel=True, fastmath=True)
def _render_kernel(width, height, cam, tan_half_fov, P, pal, ss):
    out = np.empty((height, width, 3), dtype=np.uint8)
    ox, oy, oz = cam[0], cam[1], cam[2]
    fx, fy, fz = cam[3], cam[4], cam[5]
    rx, ry, rz = cam[6], cam[7], cam[8]
    ux, uy, uz = cam[9], cam[10], cam[11]
    aspect = width / height
    pix = 2.0 * tan_half_fov / height
    inv = 1.0 / (ss * ss)
    for i in prange(height):
        for j in range(width):
            ar = 0.0
            ag = 0.0
            ab = 0.0
            for si in range(ss):
                for sj in range(ss):
                    u = (2.0 * (j + (sj + 0.5) / ss) / width - 1.0) * tan_half_fov * aspect
                    v = (1.0 - 2.0 * (i + (si + 0.5) / ss) / height) * tan_half_fov
                    dx = fx + u * rx + v * ux
                    dy = fy + u * ry + v * uy
                    dz = fz + u * rz + v * uz
                    dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                    dx /= dl
                    dy /= dl
                    dz /= dl
                    cr, cg, cb = _shade(ox, oy, oz, dx, dy, dz, pix / ss, cam, P, pal)
                    ar += cr
                    ag += cg
                    ab += cb
            # Exposure tone map + gamma
            r = 1.0 - math.exp(-1.6 * ar * inv)
            g = 1.0 - math.exp(-1.6 * ag * inv)
            b = 1.0 - math.exp(-1.6 * ab * inv)
            out[i, j, 0] = int(255.0 * min(1.0, r) ** 0.4545 + 0.5)
            out[i, j, 1] = int(255.0 * min(1.0, g) ** 0.4545 + 0.5)
            out[i, j, 2] = int(255.0 * min(1.0, b) ** 0.4545 + 0.5)
    return out


# ---------------------------------------------------------------------------
# Python-side API
# ---------------------------------------------------------------------------
class Camera:
    """Orbit camera: looks at `target` from `distance`, azimuth/elevation in degrees (z up)."""

    # Default view = the project's signature close-up (zoom x4.9 into the power-8 bulb)
    def __init__(self, target=(0.54433, 0.120321, 0.525335), azimuth=33.5, elevation=20.0,
                 distance=3.6 / 4.9, fov=40.0):
        self.target = np.asarray(target, dtype=np.float64)
        self.azimuth = float(azimuth)
        self.elevation = float(np.clip(elevation, -89.9, 89.9))
        self.distance = float(distance)
        self.fov = float(fov)

    @property
    def tan_half_fov(self):
        return math.tan(math.radians(self.fov) / 2.0)

    def basis(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        back = np.array([math.cos(el) * math.cos(az),
                         math.cos(el) * math.sin(az),
                         math.sin(el)])
        pos = self.target + self.distance * back
        fwd = -back
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return pos, fwd, right, up

    def packed(self):
        pos, fwd, right, up = self.basis()
        return np.concatenate([pos, fwd, right, up]).astype(np.float64)

    def ray(self, x, y, width, height):
        """World-space ray through pixel (x, y) (column, row; may be fractional)."""
        pos, fwd, right, up = self.basis()
        t = self.tan_half_fov
        u = (2.0 * (x + 0.5) / width - 1.0) * t * (width / height)
        v = (1.0 - 2.0 * (y + 0.5) / height) * t
        d = fwd + u * right + v * up
        return pos, d / np.linalg.norm(d)

    @staticmethod
    def angles_from_direction(d):
        """Azimuth/elevation (deg) of a camera whose view direction is `d`."""
        b = -np.asarray(d) / np.linalg.norm(d)
        el = math.degrees(math.asin(max(-1.0, min(1.0, b[2]))))
        az = math.degrees(math.atan2(b[1], b[0])) % 360.0
        return az, el


DEFAULTS = dict(
    power=8.0, iterations=14, bailout=4.0,
    julia=False, julia_c=(0.35, 0.35, -0.35),
    max_steps=None, detail=1.0, fudge=None,
    shadows=True, ao=True, glow=0.0,
    color_freq=1.4, color_offset=0.1, bound=None, specular=0.35, fog=0.0,
)


def build_params(camera, **kw):
    o = dict(DEFAULTS)
    o.update({k: v for k, v in kw.items() if v is not None})
    zoom = max(0.0, math.log2(3.6 / camera.distance))
    P = np.zeros(N_PARAMS, dtype=np.float64)
    P[P_POWER] = o["power"]
    P[P_ITERS] = o["iterations"]
    P[P_BAILOUT] = o["bailout"]
    P[P_JULIA] = 1.0 if o["julia"] else 0.0
    P[P_JX:P_JZ + 1] = o["julia_c"]
    P[P_DETAIL] = o["detail"]
    # Julia DEs overestimate more -> smaller default step
    P[P_FUDGE] = o["fudge"] or (0.55 if o["julia"] else 0.9)
    P[P_STEPS] = o["max_steps"] or min(2000, int((200 + 45 * zoom) * 0.9 / P[P_FUDGE]))
    P[P_SHADOWS] = 1.0 if o["shadows"] else 0.0
    P[P_AO] = 1.0 if o["ao"] else 0.0
    P[P_GLOW] = o["glow"]
    P[P_CFREQ] = o["color_freq"]
    P[P_COFFSET] = o["color_offset"]
    # Low powers extend further out; Julia sets can too.
    P[P_BOUND] = o["bound"] or (2.0 if (o["power"] < 4 or o["julia"]) else 1.5)
    P[P_SPEC] = o["specular"]
    P[P_SCALE] = 0.35 * camera.distance  # world size of "local" features
    P[P_FOG] = o["fog"]
    return P


def render(camera, width=480, height=480, palette=None, supersample=1, **kw):
    """Render the fractal and return an (H, W, 3) uint8 RGB array."""
    from .palettes import get_palette
    pal = get_palette(palette)
    P = build_params(camera, **kw)
    return _render_kernel(int(width), int(height), camera.packed(),
                          camera.tan_half_fov, P, pal, int(supersample))


def pick(camera, x, y, width, height, **kw):
    """Cast the ray through pixel (x, y). Returns (hit_point, t, ray_dir) or None."""
    P = build_params(camera, **kw)
    o, d = camera.ray(x, y, width, height)
    pix = 2.0 * camera.tan_half_fov / height
    t, _, _, _ = march(o[0], o[1], o[2], d[0], d[1], d[2], pix, P)
    if t < 0:
        return None
    return o + d * t, t, d


def warmup():
    """Trigger Numba compilation with a tiny render."""
    render(Camera(), 8, 8)
