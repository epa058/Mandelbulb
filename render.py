"""
Command-line renderer for stills and animations.

Examples
--------
    # A 2048px anti-aliased still of the classic power-8 bulb
    python render.py still -o bulb.png --size 2048 --ss 2

    # Same view the Dash app shows (copy target / az / el / zoom from its readout)
    python render.py still --target 0.5443 0.1203 0.5253 --az 33.5 --el 20 --zoom 4.9

    # 120-frame turntable as PNG frames + GIF
    python render.py turntable -o frames/ --frames 120 --size 512 --gif spin.gif

    # Power morph from 2 to 12
    python render.py morph -o morph/ --frames 90 --from-power 2 --to-power 12 --gif morph.gif

    # Zoom flight toward a point (exponential zoom)
    python render.py zoom -o dive/ --frames 150 --target 0.5443 0.1203 0.5253 --zoom-to 500 --gif dive.gif
"""

import argparse
import os
import time

import numpy as np
from PIL import Image

from mandelbulb import PALETTES, Camera, render, warmup

BASE_DIST = 3.6


def common_args(p):
    p.add_argument("-o", "--out", default="mandelbulb.png", help="output file (still) or folder (animations)")
    p.add_argument("--size", type=int, default=1024, help="image size in pixels (square)")
    p.add_argument("--width", type=int, help="override width")
    p.add_argument("--height", type=int, help="override height")
    p.add_argument("--ss", type=int, default=1, help="supersampling factor per axis (1-4)")
    p.add_argument("--power", type=float, default=8.0)
    p.add_argument("--iters", type=int, default=14)
    p.add_argument("--detail", type=float, default=1.0, help="hit tolerance in pixels (lower = finer)")
    p.add_argument("--julia", type=float, nargs=3, metavar=("CX", "CY", "CZ"), help="render a Julia bulb with this c")
    p.add_argument("--palette", default="Ember", choices=list(PALETTES))
    p.add_argument("--color-freq", type=float, default=1.4)
    p.add_argument("--color-offset", type=float, default=0.1)
    p.add_argument("--glow", type=float, default=0.0)
    p.add_argument("--fog", type=float, default=0.0)
    p.add_argument("--no-shadows", action="store_true")
    p.add_argument("--no-ao", action="store_true")
    p.add_argument("--target", type=float, nargs=3, default=(0.54433, 0.120321, 0.525335))
    p.add_argument("--az", type=float, default=33.5, help="azimuth (deg)")
    p.add_argument("--el", type=float, default=20.0, help="elevation (deg)")
    p.add_argument("--zoom", type=float, default=4.9, help="zoom factor (camera distance = 3.6 / zoom)")
    p.add_argument("--fov", type=float, default=40.0)
    p.add_argument("--gif", help="also write an animated GIF (animations only)")
    p.add_argument("--fps", type=int, default=30)


def render_kwargs(a, **over):
    kw = dict(
        power=a.power, iterations=a.iters, detail=a.detail,
        julia=a.julia is not None, julia_c=tuple(a.julia) if a.julia else None,
        palette=a.palette, color_freq=a.color_freq, color_offset=a.color_offset,
        glow=a.glow, fog=a.fog, shadows=not a.no_shadows, ao=not a.no_ao,
        supersample=a.ss,
    )
    kw.update(over)
    return kw


def size_of(a):
    return a.width or a.size, a.height or a.size


def camera_of(a, **over):
    c = dict(target=a.target, azimuth=a.az, elevation=a.el,
             distance=BASE_DIST / a.zoom, fov=a.fov)
    c.update(over)
    return Camera(**c)


def save_frames(frames_iter, a, n):
    os.makedirs(a.out, exist_ok=True)
    gif = []
    t0 = time.perf_counter()
    for i, img in enumerate(frames_iter):
        Image.fromarray(img).save(os.path.join(a.out, f"frame_{i:04d}.png"))
        if a.gif:
            gif.append(Image.fromarray(img))
        el = time.perf_counter() - t0
        print(f"\r  frame {i + 1}/{n}  ({el:.0f}s elapsed, ~{el / (i + 1) * (n - i - 1):.0f}s left)",
              end="", flush=True)
    print()
    if a.gif and gif:
        gif[0].save(a.gif, save_all=True, append_images=gif[1:],
                    duration=int(1000 / a.fps), loop=0, optimize=True)
        print(f"  wrote {a.gif}")


def cmd_still(a):
    w, h = size_of(a)
    t0 = time.perf_counter()
    img = render(camera_of(a), w, h, **render_kwargs(a))
    Image.fromarray(img).save(a.out)
    print(f"wrote {a.out}  ({w}x{h}, {time.perf_counter() - t0:.1f}s)")


def cmd_turntable(a):
    w, h = size_of(a)
    n = a.frames
    frames = (render(camera_of(a, azimuth=a.az + 360.0 * i / n), w, h, **render_kwargs(a))
              for i in range(n))
    save_frames(frames, a, n)


def cmd_morph(a):
    w, h = size_of(a)
    n = a.frames
    powers = np.linspace(a.from_power, a.to_power, n)
    frames = (render(camera_of(a, azimuth=a.az + a.spin * i / n), w, h,
                     **render_kwargs(a, power=float(p)))
              for i, p in enumerate(powers))
    save_frames(frames, a, n)


def cmd_zoom(a):
    w, h = size_of(a)
    n = a.frames
    zooms = a.zoom * (a.zoom_to / a.zoom) ** (np.arange(n) / max(n - 1, 1))
    frames = (render(camera_of(a, distance=BASE_DIST / z), w, h,
                     **render_kwargs(a, iterations=a.iters + int(2 * np.log2(z))))
              for z in zooms)
    save_frames(frames, a, n)


def main():
    ap = argparse.ArgumentParser(description="Mandelbulb raymarcher (Numba)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("still", help="render one image")
    common_args(p)
    p.set_defaults(fn=cmd_still)

    p = sub.add_parser("turntable", help="orbit 360 degrees around the target")
    common_args(p)
    p.add_argument("--frames", type=int, default=120)
    p.set_defaults(fn=cmd_turntable, out="turntable")

    p = sub.add_parser("morph", help="animate the power parameter")
    common_args(p)
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--from-power", type=float, default=2.0)
    p.add_argument("--to-power", type=float, default=12.0)
    p.add_argument("--spin", type=float, default=90.0, help="degrees of rotation over the clip")
    p.set_defaults(fn=cmd_morph, out="morph")

    p = sub.add_parser("zoom", help="exponential zoom toward --target (keeps az/el)")
    common_args(p)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--zoom-to", type=float, default=200.0)
    p.set_defaults(fn=cmd_zoom, out="zoom")

    a = ap.parse_args()
    print("compiling (cached after first run)...")
    warmup()
    a.fn(a)


if __name__ == "__main__":
    main()
