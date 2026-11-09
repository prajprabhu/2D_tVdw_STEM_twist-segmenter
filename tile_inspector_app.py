"""
tile_inspector_app.py — Interactive tile inspector, served via Panel/Bokeh.

Run with (or use run_tile_inspector_app.sh / .bat):
    panel serve tile_inspector_app.py --port 5009 --show

Nothing is loaded at startup. The "Source path" box is pre-filled with
the demo image (examples/data/heating_500C_frame357.png, frame 357 of a
500 C heating video); click "Load" to load it, or enter any other path
(e.g. the Gomb-Net sample examples/data/moire.npz, fetched by
`python examples/download_sample_data.py`).

Enter a video, image, or .npz array-stack path in the "Source path" box
and click "Load" to switch what's loaded — a video decodes to its normal
multi-frame tensor; a still image is loaded as a one-frame video
(Frames=1); an .npz archive (tried under the key 'im_array' first, then
'video'/'images'/'data'/'array', then whatever's there if it's the only
array) loads as an N-frame stack. Either way TiledVideo and everything
downstream needs no changes (see load_video_tensor()). The "Material"
dropdown (twist_mapper_GPU.materials registry) prefills the "d-spacing
(Å)" box with that material's default first-order plane spacing
(default_d_spacing_ang) — edit the d-spacing box afterward if the prefill
isn't the plane you actually want; either way, whatever value ends up in
that box (from the dropdown or typed by hand) drives the 1st-order Bragg
ring search live (TileFFTAnalysis.expected_ring_radius converts it to a
pixel radius via tile_size/pixel_size_ang), not just on Load. The "Pixel
size (Å/pixel)" box works the same way but is auto-populated from the
source file's own metadata when available (currently only .npz archives
carry this reliably -- see extract_pixel_size_ang) rather than from a
dropdown; falls back to manual entry (keeping whatever was already in the
box) for videos/images, which don't have a dependable place to store it.

Drag the red handle on the video frame to pick a tile — it snaps to the
nearest tile on the TiledVideo grid. Hover the frame to read the pixel
(X, Y) under the cursor (top-down image coordinates, matching row/col
math elsewhere in this app) — handy for characterizing border/edge
pixels. Scrub the frame slider to change frames (locked to frame 0 for a
single-image source). Edit the "Tile size"/"Stride" text boxes to rebuild
the tile grid at a different size — small tile sizes or large strides are
exactly how to get tiles that reach all the way to the frame border,
which the default 128/32 grid may not reach. Pick which of the four
TileFFTAnalysis panels to display. The "subtract mean before FFT"
checkbox toggles TileFFTAnalysis/TiledVideo's subtract_mean option live,
for comparing the FFT/radial-profile with and without the DC bin reduced
(see TileFFTAnalysis._compute_fft_magnitude's docstring — it changes
DC-adjacent contrast, not Bragg peak sharpness). Two separate CLAHE
checkboxes: "local" (TiledVideo.apply_clahe_to_tile) runs CLAHE on just
the selected tile, live, before Hann windowing/FFT; "global" runs CLAHE
once on every frame of the whole video (twist_mapper_GPU.
apply_clahe_to_video), before tiling — a real preprocessing step, not
free, so toggling it retiles from the original loaded video rather than
recomputing on every render like the other live controls.

Unlike the earlier static-HTML comparison demo (matplotlib/Plotly/Bokeh
hover-and-drag previews, client-side only), this is a real Bokeh server
app: every drag, frame change, or tile-size/stride edit fires an actual
Python callback on the server, which calls straight into
TiledVideo/TileFFTAnalysis for whichever tile is now selected. There is
no precomputed neighborhood limitation — any tile on the current grid
works, computed on demand.

Known simplification: module-level state (`state`, the data sources, the
analysis cache) is shared across all sessions rather than isolated per
browser tab — fine for one researcher using this locally, but two people
hitting the same served instance at once would see each other's
selections. Wrap the body below in a function returning a fresh
`pn.Column` per session (Panel's usual pattern for multi-user apps) if
that ever matters.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PointDrawTool, HoverTool, CustomJSHover

import torch
from twist_mapper_GPU import TiledVideo, TileFFTAnalysis, MATERIALS, get_material, apply_clahe_to_video
from twist_mapper_GPU.source_loading import (
    load_video_tensor as _load_video_tensor,
    extract_pixel_size_ang,
    default_d_spacing_ang,
)

pn.extension()

# ----------------------------------------------------------------------
# Config / one-time setup
# ----------------------------------------------------------------------

# Pre-filled in the "Source path" box but NOT loaded at startup -- the app
# starts empty until the user clicks "Load".
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'data', 'moire.npz')
SAMPLE_HINT = "Run `python examples/download_sample_data.py` to fetch the sample data."
# What the "Source path" box is pre-filled with: frame 357 of the 500 C
# heating video, saved as a grayscale PNG.
DEMO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'data',
                         'heating_500C_frame357.png')
DEFAULT_SOURCE_PATH = DEMO_PATH
DEFAULT_TILE_SIZE = 128
DEFAULT_STRIDE = 20
PIXEL_SIZE_ANG = 0.25  # fallback default when a source has no pixel-size metadata
DEFAULT_MATERIAL = 'MoS2'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

D_SPACING_ANG = default_d_spacing_ang(get_material(DEFAULT_MATERIAL))


def load_video_tensor(path: str) -> torch.Tensor:
    """Thin wrapper binding twist_mapper_GPU.source_loading.load_video_tensor
    to this app's `device` global (see that function for the full
    video/image/.npz loading behavior)."""
    return _load_video_tensor(path, device)


# The as-loaded video, before any global CLAHE -- kept around so toggling
# global CLAHE on/off can retile from here instead of reloading the file.
# Both stay None until the first successful load_source(); every callback
# below no-ops while `tiled is None`.
_raw_video_tensor = None
tiled = None

N_FRAMES, H, W = 0, 0, 0

# Mutable "current grid" globals — rebuild_tiling()/load_source() reassign
# these when the user edits tile size / stride or loads a new source;
# everything downstream reads them fresh.
TILE_SIZE = DEFAULT_TILE_SIZE
STRIDE = DEFAULT_STRIDE
N_ROWS, N_COLS = 0, 0
EXPECTED_RADIUS = TileFFTAnalysis.expected_ring_radius(TILE_SIZE, PIXEL_SIZE_ANG, D_SPACING_ANG)


def get_frame_np(frame_idx):
    return tiled.video[frame_idx].detach().cpu().numpy()


# ----------------------------------------------------------------------
# App state
# ----------------------------------------------------------------------

state = {
    'frame': 0,
    'row': 0,
    'col': 0,
    'panel': '1',
    'subtract_mean': False,
}
_analysis_cache = {'key': None, 'analysis': None}
_snapping = {'flag': False}
# Guards rebuild_tiling from firing (via tile_size/stride/d_spacing/pixel_size
# widgets' watchers) mid-load_source(), while load_source is itself setting
# those same widgets' values from freshly-loaded file metadata.
_loading = {'flag': False}

# ----------------------------------------------------------------------
# Bokeh frame view: image + tile grid + draggable handle/box
# ----------------------------------------------------------------------

def _handle_xy(row, col):
    x = col * STRIDE + TILE_SIZE / 2
    y = H - (row * STRIDE + TILE_SIZE / 2)
    return x, y


def _grid_lines():
    """(xs, ys) for multi_line — every other tile boundary, current TILE_SIZE/STRIDE/N_ROWS/N_COLS."""
    xs_v, ys_v = [], []
    for r in range(0, N_ROWS + 1, 2):
        y = r * STRIDE
        xs_v.append([0, W]); ys_v.append([y, y])
    for c in range(0, N_COLS + 1, 2):
        x = c * STRIDE
        xs_v.append([x, x]); ys_v.append([0, H])
    return xs_v, ys_v


def build_frame_figure():
    """(Re)builds the Bokeh frame view -- image + tile grid + draggable
    handle/box -- reading the current H/W/TILE_SIZE/STRIDE/N_ROWS/N_COLS
    globals. Called once at startup and again from load_source() whenever
    a new source may have changed frame size, since the hover tool's
    pixel-Y flip and the figure's x_range/y_range are baked in at
    construction time and can't just be patched in place.

    Returns (fig, img_source, handle_source, box_source, grid_source).
    """
    img_source = ColumnDataSource(data=dict(image=[get_frame_np(state['frame'])[::-1, :]]))

    x0, y0 = _handle_xy(state['row'], state['col'])
    # handle_source: the single draggable point (PointDrawTool only ever touches
    # x/y here). box_source: the visible red rectangle, kept in sync from Python
    # on every drag/rebuild — kept separate from handle_source so resizing the
    # box on a tile-size change never depends on PointDrawTool preserving extra
    # data-source columns across a point edit.
    handle_source = ColumnDataSource(data=dict(x=[x0], y=[y0]))
    box_source = ColumnDataSource(data=dict(x=[x0], y=[y0], width=[TILE_SIZE], height=[TILE_SIZE]))
    grid_source = ColumnDataSource(data=dict(zip(('xs', 'ys'), _grid_lines())))

    fig = figure(width=480, height=480, tools="", toolbar_location="above",
                x_range=(0, W), y_range=(0, H))
    fig.axis.visible = False
    fig.grid.visible = False
    fig.outline_line_color = None

    image_renderer = fig.image(image='image', source=img_source, x=0, y=0, dw=W, dh=H,
                               palette="Greys256", level="image")

    # Pixel (X, Y) hover readout — purely client-side. $x/$y are Bokeh's
    # cursor-position-in-data-space special vars; Y needs a CustomJSHover flip
    # since the image was stored row-reversed (img[::-1]) to display top-down
    # while Bokeh's own y-axis is bottom-up — this recovers the same top-down
    # row convention used everywhere else in this app (see _handle_xy/snap_to_tile).
    # H is baked into this JS string at build time, which is exactly why this
    # whole figure gets rebuilt (not patched) when a new source changes H.
    y_to_row = CustomJSHover(code=f"return Math.round({H} - value).toString();")
    fig.add_tools(HoverTool(
        renderers=[image_renderer],
        tooltips=[("Pixel X", "$x{0}"), ("Pixel Y", "$y{custom}")],
        formatters={"$y": y_to_row},
    ))

    fig.multi_line('xs', 'ys', source=grid_source, line_color="white", line_alpha=0.15, line_width=1)

    fig.rect(x='x', y='y', width='width', height='height', source=box_source,
            fill_alpha=0.08, fill_color="#ef4444", line_color="#ef4444", line_width=2.5)
    handle_renderer = fig.scatter(x='x', y='y', source=handle_source, size=14, fill_alpha=0,
                                  line_color="#ef4444", line_width=2)

    point_draw = PointDrawTool(renderers=[handle_renderer], num_objects=1)
    fig.add_tools(point_draw)
    fig.toolbar.active_tap = point_draw
    fig.toolbar.active_drag = point_draw

    return fig, img_source, handle_source, box_source, grid_source


# Built by load_source() on the first successful load.
frame_fig = img_source = handle_source = box_source = grid_source = None
bokeh_pane = pn.pane.Bokeh(None)

# ----------------------------------------------------------------------
# Analysis panel (matplotlib, one of 4 TileFFTAnalysis/TiledVideo plots)
# ----------------------------------------------------------------------

mpl_pane = pn.pane.Matplotlib(tight=True, dpi=100, sizing_mode='stretch_width')
coord_label = pn.pane.Markdown("")
status_label = pn.pane.Markdown("")

path_input = pn.widgets.TextInput(name='Source path (video, image, or .npz)', value=DEFAULT_SOURCE_PATH, width=440)
material_selector = pn.widgets.Select(name='Material', options=sorted(MATERIALS.keys()),
                                      value=DEFAULT_MATERIAL, width=140)
d_spacing_input = pn.widgets.FloatInput(
    name='d-spacing (Å)', value=D_SPACING_ANG,
    start=0.001, step=0.01, width=140,
)
pixel_size_input = pn.widgets.FloatInput(
    name='Pixel size (Å/pixel)', value=PIXEL_SIZE_ANG,
    start=0.0001, step=0.001, width=140,
)
load_button = pn.widgets.Button(name='Load', button_type='primary', width=80)
load_status = pn.pane.Markdown(
    "No source loaded yet. Click **Load** to open the file shown above, or enter another path."
    + ("" if os.path.isfile(DEFAULT_SOURCE_PATH) else
       (f" (Sample not downloaded yet. {SAMPLE_HINT})" if DEFAULT_SOURCE_PATH == SAMPLE_PATH
        else " (That file doesn't exist on this machine.)"))
)

frame_slider = pn.widgets.IntSlider(name='Frame', start=0, end=1, value=0, disabled=True)
frame_input = pn.widgets.IntInput(name='Go to frame', value=0, start=0, end=0, width=110,
                                  disabled=True)
panel_selector = pn.widgets.RadioButtonGroup(
    options={
        '1 · Tile / Hann / FFT': '1',
        '2 · Radial profile + ring': '2',
        '3 · Angular profile': '3',
        '4 · Correlogram': '4',
    },
    button_type='default',
)
panel_selector.value = '1'

subtract_mean_toggle = pn.widgets.Checkbox(
    name='Subtract mean before FFT (reduces DC, does not sharpen Bragg peaks)',
    value=state['subtract_mean'],
)
clahe_toggle = pn.widgets.Checkbox(
    name='Apply local CLAHE (selected tile, before FFT)',
    value=False,
)
global_clahe_toggle = pn.widgets.Checkbox(
    name='Apply global CLAHE (whole video, before tiling)',
    value=False,
)

tile_size_input = pn.widgets.IntInput(name='Tile size (px)', value=DEFAULT_TILE_SIZE,
                                      start=8, step=8, width=140)
stride_input = pn.widgets.IntInput(name='Stride (px)', value=DEFAULT_STRIDE,
                                   start=1, step=1, width=140)


def compute_analysis(frame_idx, row_idx, col_idx, subtract_mean):
    tile = tiled.get_tile(frame_idx, row_idx, col_idx)
    if tiled.apply_clahe:
        tile = tiled.apply_clahe_to_tile(tile)
    analysis = TileFFTAnalysis(tile, frame_idx, row_idx, col_idx, apply_hann=True,
                               subtract_mean=subtract_mean)
    analysis.compute_ring_radius(EXPECTED_RADIUS)
    analysis.compute_angular_profile()
    analysis.compute_twist()
    return analysis


def compute_pca_style_correlogram(analysis):
    """Same math as CorrelogramDataset.__getitem__ (correlogram_dataset.py),
    on this one on-demand tile: circular autocorrelation (Wiener-Khinchin)
    of the RAW UNFOLDED angular profile -- no folding into sectors, no
    smoothing, no zero-lag normalization -- unlike TileFFTAnalysis.
    compute_twist()'s folded/smoothed/normalized correlogram above. Plain
    numpy here (not torch) since it's a single small (n_bins,) profile
    plotted on demand, not the batched-across-every-tile path
    CorrelogramDataset is built for."""
    profile = np.asarray(analysis.angular_intensity, dtype=np.float64)
    profile_dc = profile - profile.mean()

    fft_1d = np.fft.fft(profile_dc)
    power_spectrum = np.abs(fft_1d) ** 2
    correlogram = np.fft.ifft(power_spectrum).real
    correlogram = np.fft.fftshift(correlogram)

    n_bins = len(profile)
    bin_width = 360.0 / n_bins
    lags = (np.arange(n_bins) - n_bins // 2) * bin_width
    return lags, correlogram


def plot_correlogram_comparison(analysis):
    """Panel 4: TileFFTAnalysis's own folded/smoothed/normalized
    correlogram plot (plot_correlogram) plus a third panel showing what
    the PCA pipeline (CorrelogramDataset) would compute for this same
    tile -- see compute_pca_style_correlogram's docstring for the math
    differences."""
    analysis.plot_correlogram()  # builds its own 2-panel figure, doesn't call plt.show() under Agg
    fig = plt.gcf()
    fig.set_size_inches(11, 9)
    ax1, ax2 = fig.axes

    # Re-layout the existing 2 axes onto the top row of a 2x2 grid, and add
    # the 3rd panel spanning the full bottom row, rather than duplicating
    # plot_correlogram's folded-profile/correlogram plotting code here.
    new_gs = fig.add_gridspec(2, 2, top=0.90, hspace=0.5, wspace=0.3)
    ax1.set_position(new_gs[0, 0].get_position(fig))
    ax2.set_position(new_gs[0, 1].get_position(fig))
    ax3 = fig.add_subplot(new_gs[1, :])

    lags_pca, corr_pca = compute_pca_style_correlogram(analysis)
    ax3.plot(lags_pca, corr_pca, color='#6D28D9', lw=1.0)
    ax3.axvline(0, color='gray', ls=':', lw=1)
    ax3.set_xlabel('Lag (degrees)')
    ax3.set_ylabel('Autocorrelation (raw, unnormalized)')
    ax3.set_title('PCA-Pipeline Correlogram\n(unfolded, unsmoothed, unnormalized)')
    ax3.grid(True, alpha=0.3)


def render_panel():
    if tiled is None:
        return
    f, r, c, sm = state['frame'], state['row'], state['col'], state['subtract_mean']
    key = (f, r, c, sm, TILE_SIZE, STRIDE, tiled.apply_clahe)
    if _analysis_cache['key'] != key:
        _analysis_cache['key'] = key
        _analysis_cache['analysis'] = compute_analysis(f, r, c, sm)
    analysis = _analysis_cache['analysis']

    plt.close('all')
    panel_type = state['panel']
    if panel_type == '1':
        tiled.plot_tile_fft(f, r, c, hann=True, subtract_mean=sm)
    elif panel_type == '2':
        analysis.plot_ring_radius()
    elif panel_type == '3':
        analysis.plot_angular_profile()
    else:
        plot_correlogram_comparison(analysis)

    mpl_pane.object = plt.gcf()
    coord_label.object = f"**Frame {f} · Row {r} · Col {c}**"


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------

def snap_to_tile(x, y):
    col = round((x - TILE_SIZE / 2) / STRIDE)
    row = round((H - y - TILE_SIZE / 2) / STRIDE)
    col = max(0, min(N_COLS - 1, col))
    row = max(0, min(N_ROWS - 1, row))
    return row, col


def on_handle_change(attr, old, new):
    if _snapping['flag']:
        return
    data = handle_source.data
    if len(data['x']) == 0:
        return
    x, y = data['x'][-1], data['y'][-1]
    row, col = snap_to_tile(x, y)
    snap_x, snap_y = _handle_xy(row, col)

    _snapping['flag'] = True
    handle_source.data = dict(x=[snap_x], y=[snap_y])
    box_source.data = dict(x=[snap_x], y=[snap_y], width=[TILE_SIZE], height=[TILE_SIZE])
    _snapping['flag'] = False

    if (row, col) != (state['row'], state['col']):
        state['row'], state['col'] = row, col
        render_panel()


def on_frame_change(event):
    if tiled is None:
        return
    state['frame'] = event.new
    frame_input.value = event.new  # no-op (no event) when it already matches
    img_source.data = dict(image=[get_frame_np(state['frame'])[::-1, :]])
    render_panel()


def on_frame_input_change(event):
    """Typed frame number -> clamp to the loaded range and move the slider,
    which then redraws via on_frame_change."""
    if tiled is None:
        return
    clamped = min(max(int(event.new), 0), N_FRAMES - 1)
    if clamped != event.new:
        frame_input.value = clamped
        return
    frame_slider.value = clamped


frame_slider.param.watch(on_frame_change, 'value')
frame_input.param.watch(on_frame_input_change, 'value')


def on_panel_change(event):
    state['panel'] = event.new
    render_panel()


panel_selector.param.watch(on_panel_change, 'value')


def on_subtract_mean_change(event):
    state['subtract_mean'] = event.new
    render_panel()


subtract_mean_toggle.param.watch(on_subtract_mean_change, 'value')


def on_clahe_change(event):
    if tiled is None:
        return  # load_source() reads clahe_toggle.value directly
    tiled.apply_clahe = event.new
    render_panel()


clahe_toggle.param.watch(on_clahe_change, 'value')


def on_global_clahe_change(event):
    """Global CLAHE is a real preprocessing pass over every frame (~6.5s
    for the reference 606-frame video -- see apply_clahe_to_video), not
    free like the other live controls, so this retiles from
    _raw_video_tensor rather than recomputing per-render. Rebuilds the
    Bokeh figure only if H/W actually changed (CLAHE could in principle
    shift what crop_black_margins considers "content" at the border,
    though in practice it usually won't) -- otherwise just refreshes the
    displayed frame in place."""
    global tiled, N_FRAMES, H, W, N_ROWS, N_COLS
    global frame_fig, img_source, handle_source, box_source, grid_source

    if tiled is None:
        return  # load_source() reads global_clahe_toggle.value directly

    old_H, old_W = H, W
    video = apply_clahe_to_video(_raw_video_tensor) if event.new else _raw_video_tensor
    tiled = TiledVideo(video, tile_size=TILE_SIZE, stride=STRIDE, crop_black_margins=True,
                       apply_clahe=tiled.apply_clahe)
    N_FRAMES, H, W = tiled.video.shape
    N_ROWS, N_COLS = tiled.n_rows, tiled.n_cols
    _analysis_cache['key'] = None

    if (H, W) != (old_H, old_W):
        frame_fig, img_source, handle_source, box_source, grid_source = build_frame_figure()
        handle_source.on_change('data', on_handle_change)
        bokeh_pane.object = frame_fig
        frame_slider.end = max(1, N_FRAMES - 1)
        frame_input.end = max(0, N_FRAMES - 1)
    else:
        img_source.data = dict(image=[get_frame_np(state['frame'])[::-1, :]])

    status_label.object = f"Grid: {N_ROWS} x {N_COLS} tiles (tile_size={TILE_SIZE}, stride={STRIDE})"
    render_panel()


global_clahe_toggle.param.watch(on_global_clahe_change, 'value')


def rebuild_tiling(_event=None):
    """Rebuilds the tile grid at the current tile_size_input/stride_input
    values — reuses the already-loaded (and already black-margin-cropped)
    video tensor, so this is just a cheap as_strided view rebuild, not a
    video reload. No-ops while load_source() is mid-flight (see _loading)."""
    global tiled, TILE_SIZE, STRIDE, N_ROWS, N_COLS, EXPECTED_RADIUS

    if _loading['flag'] or tiled is None:
        return

    new_tile_size = tile_size_input.value
    new_stride = stride_input.value

    if new_tile_size <= 0 or new_stride <= 0:
        status_label.object = "**Error:** tile size and stride must be positive."
        return
    if new_tile_size > min(H, W):
        status_label.object = f"**Error:** tile size {new_tile_size} exceeds frame size ({H}x{W})."
        return
    if d_spacing_input.value <= 0:
        status_label.object = "**Error:** d-spacing must be positive."
        return
    if pixel_size_input.value <= 0:
        status_label.object = "**Error:** pixel size must be positive."
        return

    new_tiled = TiledVideo(tiled.video, tile_size=new_tile_size, stride=new_stride,
                           apply_clahe=tiled.apply_clahe)

    tiled = new_tiled
    TILE_SIZE, STRIDE = new_tile_size, new_stride
    N_ROWS, N_COLS = tiled.n_rows, tiled.n_cols
    EXPECTED_RADIUS = TileFFTAnalysis.expected_ring_radius(TILE_SIZE, pixel_size_input.value, d_spacing_input.value)

    # Clamp the current selection onto the new (possibly smaller) grid
    state['row'] = min(state['row'], N_ROWS - 1)
    state['col'] = min(state['col'], N_COLS - 1)

    grid_source.data = dict(zip(('xs', 'ys'), _grid_lines()))
    snap_x, snap_y = _handle_xy(state['row'], state['col'])
    handle_source.data = dict(x=[snap_x], y=[snap_y])
    box_source.data = dict(x=[snap_x], y=[snap_y], width=[TILE_SIZE], height=[TILE_SIZE])

    status_label.object = f"Grid: {N_ROWS} x {N_COLS} tiles (tile_size={TILE_SIZE}, stride={STRIDE})"
    render_panel()


tile_size_input.param.watch(rebuild_tiling, 'value')
stride_input.param.watch(rebuild_tiling, 'value')
d_spacing_input.param.watch(rebuild_tiling, 'value')
pixel_size_input.param.watch(rebuild_tiling, 'value')


def on_material_change(event):
    """Prefills d_spacing_input with the newly picked material's default
    plane spacing -- this itself fires d_spacing_input's own watcher
    (rebuild_tiling) to recompute EXPECTED_RADIUS and re-render, so no
    separate recompute is needed here. Overwrites whatever was in the
    box, including a prior manual edit -- that's the point of picking a
    different material."""
    material = get_material(event.new)
    d_spacing_input.value = default_d_spacing_ang(material)


material_selector.param.watch(on_material_change, 'value')


def load_source(path):
    """Loads `path` (video or image, see load_video_tensor) as the new
    source, rebuilding the tile grid, Bokeh frame view, and analysis
    panel. Keeps whatever tile_size/stride are currently set in the input
    boxes (validated against the new frame size) rather than resetting
    them -- consistent with rebuild_tiling's own validation. If the file
    carries its own pixel-size metadata (currently only .npz -- see
    extract_pixel_size_ang), pixel_size_input is updated to match;
    otherwise whatever's already in that box (the PIXEL_SIZE_ANG default,
    or a previous manual entry) is left alone, for the user to set by
    hand."""
    global tiled, TILE_SIZE, STRIDE, N_ROWS, N_COLS, EXPECTED_RADIUS, N_FRAMES, H, W
    global frame_fig, img_source, handle_source, box_source, grid_source, _raw_video_tensor

    if not os.path.isfile(path):
        hint = f" {SAMPLE_HINT}" if os.path.abspath(path) == SAMPLE_PATH else ""
        load_status.object = f"**Error:** file not found: `{path}`.{hint}"
        return

    try:
        video_tensor = load_video_tensor(path)
    except Exception as exc:
        load_status.object = f"**Error loading** `{path}`: {exc}"
        return

    _, new_h, new_w = video_tensor.shape
    new_tile_size, new_stride = tile_size_input.value, stride_input.value

    if new_tile_size <= 0 or new_stride <= 0:
        load_status.object = "**Error:** tile size and stride must be positive."
        return
    if new_tile_size > min(new_h, new_w):
        load_status.object = (f"**Error:** tile size {new_tile_size} exceeds this source's "
                              f"frame size ({new_h}x{new_w}).")
        return
    if d_spacing_input.value <= 0:
        load_status.object = "**Error:** d-spacing must be positive."
        return

    found_pixel_size = extract_pixel_size_ang(path)
    pixel_size_ang = found_pixel_size if found_pixel_size and found_pixel_size > 0 else pixel_size_input.value
    if pixel_size_ang <= 0:
        load_status.object = "**Error:** pixel size must be positive."
        return

    _loading['flag'] = True  # suppress rebuild_tiling while setting widgets below
    try:
        _raw_video_tensor = video_tensor
        video_for_tiling = apply_clahe_to_video(video_tensor) if global_clahe_toggle.value else video_tensor
        tiled = TiledVideo(video_for_tiling, tile_size=new_tile_size, stride=new_stride,
                           crop_black_margins=True, apply_clahe=clahe_toggle.value)
        TILE_SIZE, STRIDE = new_tile_size, new_stride
        N_FRAMES, H, W = tiled.video.shape
        N_ROWS, N_COLS = tiled.n_rows, tiled.n_cols
        EXPECTED_RADIUS = TileFFTAnalysis.expected_ring_radius(TILE_SIZE, pixel_size_ang, d_spacing_input.value)

        if found_pixel_size and found_pixel_size > 0:
            pixel_size_input.value = found_pixel_size

        state['frame'] = N_FRAMES // 2
        state['row'] = N_ROWS // 2
        state['col'] = N_COLS // 2
        _analysis_cache['key'] = None
        _analysis_cache['analysis'] = None

        # H/W (hence the hover tool's baked-in flip and the figure's x/y_range)
        # may have changed, so the whole Bokeh figure gets rebuilt rather than
        # patched in place — see build_frame_figure()'s docstring.
        frame_fig, img_source, handle_source, box_source, grid_source = build_frame_figure()
        handle_source.on_change('data', on_handle_change)
        bokeh_pane.object = frame_fig

        # IntSlider needs end > start, so a 1-frame source keeps end=1 and is just disabled
        frame_slider.end = max(1, N_FRAMES - 1)
        frame_slider.value = state['frame']
        frame_slider.disabled = N_FRAMES <= 1
        frame_input.end = max(0, N_FRAMES - 1)
        frame_input.value = state['frame']
        frame_input.disabled = N_FRAMES <= 1
    finally:
        _loading['flag'] = False

    kind = 'image (1 frame)' if N_FRAMES == 1 else f'video ({N_FRAMES} frames)'
    pixel_note = " · pixel size read from file metadata" if found_pixel_size and found_pixel_size > 0 else ""
    status_label.object = f"Grid: {N_ROWS} x {N_COLS} tiles (tile_size={TILE_SIZE}, stride={STRIDE})"
    load_status.object = f"Loaded {kind}: `{path}` ({H}x{W}){pixel_note}"

    render_panel()


def on_load_click(event):
    load_source(path_input.value.strip())


load_button.on_click(on_load_click)

render_panel()  # initial render

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

app = pn.Column(
    "# Tile Inspector",
    pn.Row(path_input, material_selector, d_spacing_input, pixel_size_input, load_button),
    load_status,
    pn.Row(
        pn.Column(bokeh_pane, pn.Row(frame_slider, frame_input), pn.Row(tile_size_input, stride_input), status_label),
        pn.Column(panel_selector, subtract_mean_toggle, clahe_toggle, global_clahe_toggle,
                 mpl_pane, coord_label),
    ),
)

app.servable(title="Tile Inspector")
