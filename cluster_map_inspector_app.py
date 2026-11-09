"""
cluster_map_inspector_app.py — Interactive pixel-resolution cluster map
viewer, served via Panel/Bokeh.

Run with (a different port than tile_inspector_app.py's default, so both
can be served at the same time without clashing; or use
run_cluster_map_inspector_app.sh / .bat):
    panel serve cluster_map_inspector_app.py --port 5006 --show

Nothing is loaded or computed at startup. The "Source path" box is
pre-filled with the demo image (examples/data/heating_500C_frame357.png,
frame 357 of a 500 C heating video); click "Load & Recompute" to run the
pipeline on it, or enter any other path (e.g. the Gomb-Net sample
examples/data/moire.npz, fetched by `python examples/download_sample_data.py`). PCA and
K-means run on the GPU via RAPIDS (cuML) when it's installed, and on the
CPU via scikit-learn otherwise (see dynamic_pipeline.HAS_RAPIDS).

Shows the video frame with the Hann-weighted-blend pixel-resolution
cluster map (`twist_mapper_GPU.pixel_cluster_map`) overlaid, with an alpha
slider to fade between the raw frame and the cluster mask, and a frame
slider to scrub through the video. Clusters are numbered only (`Cluster
0`..`Cluster {N-1}`) — no semantic void/amorphous/monolayer/bilayer
labeling.

Unlike the earlier cache-reading version of this app, everything here is
computed **dynamically**: the "Source path" / "Material" / "d-spacing" /
"Pixel size" / "Tile size" / "Stride" / "Decomposition" (PCA or NMF) /
"Apply CLAHE" controls (same core set tile_inspector_app.py has, via the
shared twist_mapper_GPU.source_loading module), plus a "Number of
clusters" control specific to this app, feed straight into
`twist_mapper_GPU.dynamic_pipeline.run_dynamic_pipeline` on "Load &
Recompute" — batched GPU correlogram generation + PCA/NMF + K-means, ~10s
end to end for the full 606-frame/218,766-tile reference video (see that
module's docstring for why this is now fast enough to redo on demand,
where the notebook pipeline this mirrors takes 9-24 minutes). Unlike the
per-tile controls in tile_inspector_app.py (cheap enough to recompute on
every keystroke), changing any of these boxes here does NOT recompute
live — the whole decomposition/K-means fit is expensive enough that
recomputation is only triggered by the explicit "Load & Recompute" click.
The "PCs used for clustering" checkboxes and "Number of clusters" are
separate: clustering doesn't depend on tiling or the decomposition, so
their "Compute clustering" button re-fits only K-means on the
already-computed decomposition (PCA or NMF), with no reload. Component
names in the UI are 1-based (PC1..PC10).
"""

import os
import threading
import numpy as np
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt

import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

from twist_mapper_GPU import (MATERIALS, get_material, TileFFTAnalysis, build_symmetric_hann2d,
                              blend_and_classify_frame)
from twist_mapper_GPU.source_loading import load_video_tensor, extract_pixel_size_ang, default_d_spacing_ang
from twist_mapper_GPU.dynamic_pipeline import (
    run_dynamic_pipeline, refit_kmeans, find_dominant_frequencies, plot_pc_spectra, HAS_RAPIDS,
    compute_cluster_dominant_components,
)

pn.extension()

# ----------------------------------------------------------------------
# Config / one-time setup
# ----------------------------------------------------------------------

# Pre-filled in the "Source path" box but NOT loaded at startup -- the app
# starts empty until the user clicks "Load & Recompute".
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
N_COMPONENTS = 10
N_CLUSTERS = 5
# Components checked by default, as 0-based indices (displayed 1-based: PC1-PC4, PC6)
DEFAULT_SELECTED_PCS = [0, 1, 2, 3, 5]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
D_SPACING_ANG = default_d_spacing_ang(get_material(DEFAULT_MATERIAL))


def run_pipeline_for(path, tile_size, stride, pixel_size_ang, d_spacing_ang, apply_clahe,
                     n_clusters=N_CLUSTERS, method='pca'):
    """Loads `path` and runs the full dynamic pipeline (tiling -> batched
    correlograms -> decomposition (PCA or NMF, see `method`) -> K-means)
    against it. Returns a DynamicPipelineResult (see dynamic_pipeline.py)."""
    video_tensor = load_video_tensor(path, device)
    expected_radius = TileFFTAnalysis.expected_ring_radius(tile_size, pixel_size_ang, d_spacing_ang)
    return run_dynamic_pipeline(
        video_tensor, tile_size=tile_size, stride=stride, expected_radius=expected_radius,
        n_components=N_COMPONENTS, n_clusters=n_clusters, apply_clahe=apply_clahe,
        crop_black_margins=True, verbose=True, method=method,
    )


# Everything below stays empty until the first successful load_source();
# every callback no-ops while `tiled is None`.
tiled = None
N_FRAMES, H, W = 0, 0, 0
N_ROWS, N_COLS = 0, 0


def _standardize(x_pca_np):
    """z-score per PC, mirroring run_dynamic_pipeline's own internal
    standardization exactly -- the K-means centroids were fit in this
    standardized space, so the pixel-blend field must be too, or nearest-
    centroid classification compares mismatched scales (raw PC1 has much
    larger magnitude than raw PC8, since PCA orders components by
    variance)."""
    return (x_pca_np - x_pca_np.mean(axis=0)) / (x_pca_np.std(axis=0) + 1e-8)


def _build_pca_grid(result):
    x_pca_np = result.X_pca.detach().cpu().numpy()
    x_pca_z = _standardize(x_pca_np)
    n_components = x_pca_z.shape[1]
    return torch.as_tensor(x_pca_z, device=device, dtype=torch.float32).reshape(
        result.Frames, result.Rows, result.Cols, n_components)


X_pca_grid = None          # (Frames, Rows, Cols, n_components), z-scored
centroids = None           # (n_clusters, len(selected_pcs))
N_CLUSTERS_ACTUAL = 0
hann2d = None

pca_components = None      # (N_COMPONENTS, n_bins) -- PCA loadings or NMF basis
explained_variance_ratio = None  # None when method='nmf'
n_bins = 0
selected_pcs = list(DEFAULT_SELECTED_PCS)   # 0-based components feeding K-means + the pixel blend
current_method = 'pca'     # 'pca' or 'nmf' -- which decomposition is currently active

CLUSTER_COLORS = np.zeros((0, 3), dtype=np.uint8)


def _cluster_colors(n):
    tab10 = matplotlib.colormaps['tab10'].resampled(max(n, 1))
    return (np.array([tab10(i)[:3] for i in range(n)]) * 255).astype(np.uint8).reshape(-1, 3)


def get_frame_np(frame_idx):
    return tiled.video[frame_idx].detach().cpu().numpy()


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------

state = {'frame': 0, 'alpha': 0.5}


def compute_overlay_rgb(frame_idx, alpha):
    """Grayscale frame blended with the cluster-color overlay at `alpha`
    (0 = raw frame only, 1 = cluster colors only). Pixels the tile grid
    never covers (see pixel_cluster_map.blend_pc_field's `covered` return)
    always show the raw frame, regardless of alpha. Row 0 = top, matching
    the source video tensor (NOT Bokeh's bottom-up image convention --
    see compute_rgba)."""
    frame = get_frame_np(frame_idx)  # (H, W), float32 in [0, 1]
    gray = np.clip(frame * 255, 0, 255).astype(np.float32)
    rgb = np.stack([gray, gray, gray], axis=-1)

    labels, covered = blend_and_classify_frame(
        X_pca_grid[frame_idx][..., selected_pcs], centroids, tiled.stride, H, W, hann2d
    )
    labels_np = labels.detach().cpu().numpy()
    covered_np = covered.detach().cpu().numpy()

    overlay = CLUSTER_COLORS[labels_np].astype(np.float32)  # (H, W, 3)
    blended = rgb.copy()
    blended[covered_np] = (1 - alpha) * rgb[covered_np] + alpha * overlay[covered_np]
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_rgba(frame_idx, alpha):
    """compute_overlay_rgb() as RGBA, flipped for Bokeh's image_rgba glyph
    (row 0 at the bottom)."""
    rgb = compute_overlay_rgb(frame_idx, alpha)
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    # force contiguity so the uint8->uint32 view in rgba_to_bokeh_image is valid
    return np.ascontiguousarray(rgba[::-1, :, :])


def rgba_to_bokeh_image(rgba):
    """(H, W, 4) uint8 -> (H, W) uint32, the packed format Bokeh's
    image_rgba glyph expects."""
    return rgba.view(dtype=np.uint32).reshape(rgba.shape[0], rgba.shape[1])


# ----------------------------------------------------------------------
# Bokeh figure
# ----------------------------------------------------------------------

def build_frame_figure():
    """(Re)builds the Bokeh frame view -- reads the current H/W globals.
    Called once at startup and again from load_source() whenever a new
    source may have changed frame size (the figure's x_range/y_range are
    baked in at construction time)."""
    img_source = ColumnDataSource(data=dict(image=[rgba_to_bokeh_image(compute_rgba(state['frame'], state['alpha']))]))

    fig = figure(width=520, height=520, tools="pan,wheel_zoom,reset", toolbar_location="above",
                x_range=(0, W), y_range=(0, H))
    fig.axis.visible = False
    fig.grid.visible = False
    fig.outline_line_color = None
    fig.image_rgba(image='image', source=img_source, x=0, y=0, dw=W, dh=H)

    return fig, img_source


# Built by load_source() on the first successful load.
frame_fig = img_source = None
bokeh_pane = pn.pane.Bokeh(None)

# ----------------------------------------------------------------------
# Widgets + callbacks
# ----------------------------------------------------------------------

path_input = pn.widgets.TextInput(name='Source path (video, image, or .npz)', value=DEFAULT_SOURCE_PATH, width=380)
material_selector = pn.widgets.Select(name='Material', options=sorted(MATERIALS.keys()),
                                      value=DEFAULT_MATERIAL, width=130)
d_spacing_input = pn.widgets.FloatInput(name='d-spacing (Å)', value=D_SPACING_ANG,
                                        start=0.001, step=0.01, width=120)
pixel_size_input = pn.widgets.FloatInput(name='Pixel size (Å/pixel)', value=PIXEL_SIZE_ANG,
                                         start=0.0001, step=0.001, width=130)
tile_size_input = pn.widgets.IntInput(name='Tile size (px)', value=DEFAULT_TILE_SIZE,
                                      start=8, step=8, width=110)
stride_input = pn.widgets.IntInput(name='Stride (px)', value=DEFAULT_STRIDE,
                                   start=1, step=1, width=110)
n_clusters_input = pn.widgets.IntInput(name='Number of clusters', value=N_CLUSTERS,
                                       start=2, step=1, width=130)
method_selector = pn.widgets.Select(name='Decomposition', options=['PCA', 'NMF'], value='PCA', width=100)
clahe_toggle = pn.widgets.Checkbox(name='Apply CLAHE (global, before tiling)', value=False)

# Labels are 1-based (PC1..PC10, matching the PC spectra plot); values are
# the 0-based column indices into X_pca_grid.
pc_checkbox = pn.widgets.CheckBoxGroup(
    name='PCs used for clustering', options={f'PC{i + 1}': i for i in range(N_COMPONENTS)},
    value=list(DEFAULT_SELECTED_PCS), inline=True)
cluster_button = pn.widgets.Button(name='Compute clustering', button_type='primary', width=160)


def _pc_names(indices):
    return ", ".join(f'PC{i + 1}' for i in indices)


pc_selection_status = pn.pane.Markdown(
    f"Selected {len(DEFAULT_SELECTED_PCS)}/{N_COMPONENTS} PCs: {_pc_names(DEFAULT_SELECTED_PCS)}",
    width=520)
show_spectra_toggle = pn.widgets.Toggle(name='Show PC spectra', value=False, width=160)
spectra_pane = pn.pane.Matplotlib(visible=False, tight=True)
load_button = pn.widgets.Button(name='Load & Recompute', button_type='primary', width=160)
_backend_note = ("PCA/K-means backend: **RAPIDS (GPU)**" if HAS_RAPIDS
                 else "PCA/K-means backend: **scikit-learn (CPU)** -- install RAPIDS for GPU speed-up")
load_status = pn.pane.Markdown(
    "No source loaded yet. Click **Load & Recompute** to run on the file shown above, "
    "or enter another path."
    + ("" if os.path.isfile(DEFAULT_SOURCE_PATH) else
       (f" (Sample not downloaded yet. {SAMPLE_HINT})" if DEFAULT_SOURCE_PATH == SAMPLE_PATH
        else " (That file doesn't exist on this machine.)"))
    + f"<br>{_backend_note}"
)

frame_slider = pn.widgets.Player(name='Frame', start=0, end=1, value=0, disabled=True,
                                 step=1, interval=100, loop_policy='loop', show_value=True,
                                 width=520)
frame_input = pn.widgets.IntInput(name='Go to frame', value=0, start=0, end=0, width=110,
                                  disabled=True)
alpha_slider = pn.widgets.FloatSlider(name='Cluster overlay opacity', start=0.0, end=1.0,
                                     step=0.05, value=state['alpha'])
legend_md = pn.pane.Markdown("", width=520)


def _legend_markdown():
    """Cluster legend: color swatch + cluster number, followed in brackets by
    the cluster's dominant correlogram component and that component's
    dominant angular periodicity, e.g. "0 (PC3, 60.0°)" -- see
    dynamic_pipeline.compute_cluster_dominant_components. Component names
    are 1-based to match the "PCs used for clustering" checkboxes."""
    if centroids is None or pca_components is None:
        return ""
    label = 'PC' if current_method == 'pca' else 'NMF'
    dominant = compute_cluster_dominant_components(centroids, selected_pcs, pca_components, n_bins)
    items = []
    for i, ((r, g, b), d) in enumerate(zip(CLUSTER_COLORS, dominant)):
        items.append(
            f'<span style="white-space:nowrap;">'
            f'<span style="color:rgb({r},{g},{b});font-size:16px;">&#9632;</span> '
            f'{i} ({label}{d["component"] + 1}, {d["angle_deg"]:.1f}°)</span>'
        )
    return ("**Clusters** (dominant component, its dominant angle):<br>"
            + " &nbsp;&nbsp; ".join(items))


def refresh():
    if tiled is None or img_source is None:
        return
    rgba = compute_rgba(state['frame'], state['alpha'])
    img_source.data = dict(image=[rgba_to_bokeh_image(rgba)])


def on_frame_change(event):
    state['frame'] = event.new
    frame_input.value = event.new  # no-op (no event) when it already matches
    refresh()


def on_frame_input_change(event):
    """Typed frame number -> clamp to the loaded range and move the player,
    which then redraws via on_frame_change."""
    if tiled is None:
        return
    clamped = min(max(int(event.new), 0), N_FRAMES - 1)
    if clamped != event.new:
        frame_input.value = clamped
        return
    frame_slider.value = clamped


def on_alpha_change(event):
    state['alpha'] = event.new
    refresh()


frame_slider.param.watch(on_frame_change, 'value')
frame_input.param.watch(on_frame_input_change, 'value')
alpha_slider.param.watch(on_alpha_change, 'value')


def _selected_pc_indices():
    """pc_checkbox's checked values: sorted 0-based column indices into
    X_pca_grid's last dimension (the labels shown are 1-based)."""
    return sorted(pc_checkbox.value)


def _refit_clustering():
    """Re-fits K-means on the currently selected subset of PCA columns of
    the already-computed X_pca_grid, with the current "Number of
    clusters". Run by the "Compute clustering" button (and at the end of
    every Load & Recompute): clustering is independent of CLAHE/tiling/
    correlograms/PCA, so changing the PC subset or cluster count never
    needs a reload -- unlike tile_size/stride/d-spacing/etc., which change
    those upstream stages. See dynamic_pipeline.refit_kmeans."""
    global centroids, N_CLUSTERS_ACTUAL, CLUSTER_COLORS, selected_pcs

    if X_pca_grid is None:
        return
    idx = _selected_pc_indices()
    if len(idx) < 1:
        pc_selection_status.object = "**Error:** select at least 1 PC."
        return
    n_clusters = n_clusters_input.value
    if n_clusters < 2:
        pc_selection_status.object = "**Error:** number of clusters must be at least 2."
        return

    selected_pcs = idx
    X_sel = X_pca_grid[..., idx].reshape(-1, len(idx))
    _, new_centroids = refit_kmeans(X_sel, n_clusters)

    centroids = torch.as_tensor(new_centroids, device=device, dtype=torch.float32)
    N_CLUSTERS_ACTUAL = centroids.shape[0]

    CLUSTER_COLORS = _cluster_colors(N_CLUSTERS_ACTUAL)
    legend_md.object = _legend_markdown()
    pc_selection_status.object = (f"Clustered into {N_CLUSTERS_ACTUAL} clusters on "
                                  f"{len(idx)}/{X_pca_grid.shape[-1]} PCs: {_pc_names(idx)}")
    refresh()


def on_cluster_click(event):
    _refit_clustering()


def on_clustering_settings_change(event):
    """PC subset / cluster count edited: nothing is recomputed until
    "Compute clustering" is clicked, so just flag the pending change."""
    if X_pca_grid is None:
        return
    pc_selection_status.object = (f"Selected {len(pc_checkbox.value)} PCs "
                                  f"({_pc_names(sorted(pc_checkbox.value))}), "
                                  f"{n_clusters_input.value} clusters -- click "
                                  "**Compute clustering** to apply.")


cluster_button.on_click(on_cluster_click)
n_clusters_input.param.watch(on_clustering_settings_change, 'value')
pc_checkbox.param.watch(on_clustering_settings_change, 'value')

_spectra_fig = None


def _refresh_pc_spectra():
    """(Re)builds the PC-loadings-and-spectra figure from the current
    pca_components/n_bins/explained_variance_ratio globals and pushes it
    into spectra_pane. Closes the previous figure first so repeated
    toggling/reloading doesn't leak matplotlib figures."""
    global _spectra_fig
    if pca_components is None:
        return
    if _spectra_fig is not None:
        plt.close(_spectra_fig)
    dominant_freqs = find_dominant_frequencies(pca_components, n_bins)
    _spectra_fig = plot_pc_spectra(pca_components, n_bins, dominant_freqs,
                                   explained_variance_ratio=explained_variance_ratio,
                                   component_label=current_method.upper())
    spectra_pane.object = _spectra_fig


def on_show_spectra_toggle(event):
    if event.new:
        _refresh_pc_spectra()
    spectra_pane.visible = event.new


show_spectra_toggle.param.watch(on_show_spectra_toggle, 'value')


def on_material_change(event):
    """Prefills d_spacing_input with the newly picked material's default
    plane spacing -- unlike tile_inspector_app.py, this does NOT trigger
    a live recompute (the whole PCA/K-means fit is too expensive for
    that); it only takes effect on the next "Load & Recompute" click."""
    material = get_material(event.new)
    d_spacing_input.value = default_d_spacing_ang(material)


material_selector.param.watch(on_material_change, 'value')


def load_source(path):
    """Runs the full dynamic pipeline against `path` with whatever
    tile-size/stride/d-spacing/pixel-size/CLAHE settings are currently in
    the widgets, and rebuilds everything downstream (tiled, X_pca_grid,
    centroids, the Bokeh figure, frame slider bounds, recording-range
    bounds). Synchronous (blocks the server callback for the pipeline's
    runtime, ~10s for the reference video) -- deliberately not
    backgrounded like the recording feature, since this mutates shared
    display state (X_pca_grid/centroids/tiled) that concurrent frame
    scrubbing would otherwise race against.
    """
    global tiled, N_FRAMES, H, W, N_ROWS, N_COLS, X_pca_grid, hann2d, frame_fig, img_source
    global pca_components, explained_variance_ratio, n_bins, current_method
    global centroids, N_CLUSTERS_ACTUAL, CLUSTER_COLORS, selected_pcs

    if not os.path.isfile(path):
        hint = f" {SAMPLE_HINT}" if os.path.abspath(path) == SAMPLE_PATH else ""
        load_status.object = f"**Error:** file not found: `{path}`.{hint}"
        return

    new_tile_size, new_stride = tile_size_input.value, stride_input.value
    if new_tile_size <= 0 or new_stride <= 0:
        load_status.object = "**Error:** tile size and stride must be positive."
        return
    if d_spacing_input.value <= 0:
        load_status.object = "**Error:** d-spacing must be positive."
        return
    new_n_clusters = n_clusters_input.value
    if new_n_clusters < 2:
        load_status.object = "**Error:** number of clusters must be at least 2."
        return

    found_pixel_size = extract_pixel_size_ang(path)
    pixel_size_ang = found_pixel_size if found_pixel_size and found_pixel_size > 0 else pixel_size_input.value
    if pixel_size_ang <= 0:
        load_status.object = "**Error:** pixel size must be positive."
        return

    load_button.disabled = True
    load_status.object = f"Recomputing from `{path}`..."

    try:
        result = run_pipeline_for(path, new_tile_size, new_stride, pixel_size_ang,
                                  d_spacing_input.value, clahe_toggle.value, new_n_clusters,
                                  method=method_selector.value.lower())
    except Exception as exc:
        load_status.object = f"**Error loading** `{path}`: {exc}"
        load_button.disabled = False
        return
    finally:
        load_button.disabled = False

    if found_pixel_size and found_pixel_size > 0:
        pixel_size_input.value = found_pixel_size

    tiled = result.tiled
    N_FRAMES, H, W = tiled.video.shape
    N_ROWS, N_COLS = tiled.n_rows, tiled.n_cols
    X_pca_grid = _build_pca_grid(result)
    hann2d = build_symmetric_hann2d(tiled.tile_size, device=device)
    pca_components = result.pca_components
    explained_variance_ratio = result.explained_variance_ratio
    n_bins = result.correlograms.shape[1]
    current_method = result.method
    # Provisional clustering on all PCs, so the figure built below has
    # centroids whose width matches selected_pcs; _refit_clustering() at the
    # end then re-applies whatever PC subset / cluster count is selected.
    selected_pcs = list(range(X_pca_grid.shape[-1]))
    centroids = torch.as_tensor(result.centroids, device=device, dtype=torch.float32)
    N_CLUSTERS_ACTUAL = centroids.shape[0]
    CLUSTER_COLORS = _cluster_colors(N_CLUSTERS_ACTUAL)

    state['frame'] = N_FRAMES // 2
    frame_fig, img_source = build_frame_figure()
    bokeh_pane.object = frame_fig

    frame_slider.end = max(1, N_FRAMES - 1)  # Player needs end > start; 1-frame sources just disable it
    frame_slider.value = state['frame']
    frame_slider.disabled = N_FRAMES <= 1
    frame_input.end = max(0, N_FRAMES - 1)
    frame_input.value = state['frame']
    frame_input.disabled = N_FRAMES <= 1
    record_start_input.end = max(0, N_FRAMES - 1)
    record_end_input.end = max(0, N_FRAMES - 1)
    record_end_input.value = N_FRAMES - 1

    # Re-fit clustering under whatever "Number of clusters" / "PCs used for
    # clustering" selections are currently set, rather than blindly trusting
    # result.centroids (which run_dynamic_pipeline always fits on ALL PCs
    # internally) -- keeps behavior consistent if a non-default PC subset
    # was already selected before this Load & Recompute click. This also
    # calls refresh(), so it must run after frame_fig/img_source above.
    _refit_clustering()
    if show_spectra_toggle.value:
        _refresh_pc_spectra()

    kind = 'image (1 frame)' if N_FRAMES == 1 else f'video ({N_FRAMES} frames)'
    pixel_note = " · pixel size read from file metadata" if found_pixel_size and found_pixel_size > 0 else ""
    timings_note = ", ".join(f"{k}={v:.1f}s" for k, v in result.timings.items())
    load_status.object = (f"Loaded {kind}: `{path}` ({H}x{W}){pixel_note} — {timings_note}"
                          f"<br>{_backend_note}")


def on_load_click(event):
    load_source(path_input.value.strip())


load_button.on_click(on_load_click)

# ----------------------------------------------------------------------
# Recording: writes compute_overlay_rgb() for a frame range to an mp4,
# at whatever the current opacity slider is set to, on a background
# thread (recording the full 606-frame video takes ~15-20s, so it must
# not block the server's websocket loop).
# ----------------------------------------------------------------------

record_dir_input = pn.widgets.TextInput(name='Save folder', value=os.getcwd(), width=340)
record_filename_input = pn.widgets.TextInput(name='Filename', value='cluster_map_recording.mp4', width=260)
record_start_input = pn.widgets.IntInput(name='Start frame', value=0, start=0, end=0, width=110)
record_end_input = pn.widgets.IntInput(name='End frame', value=0, start=0, end=0, width=110)
record_fps_input = pn.widgets.IntInput(name='FPS', value=10, start=1, end=60, width=80)
record_button = pn.widgets.Button(name='● Record video', button_type='danger', width=160)
record_status = pn.pane.Markdown('', width=520)


def _record_worker(save_path, fps, start_f, end_f, alpha, doc):
    def set_status(text):
        if doc is not None:
            doc.add_next_tick_callback(lambda: setattr(record_status, 'object', text))
        else:
            record_status.object = text

    def set_button_disabled(disabled):
        if doc is not None:
            doc.add_next_tick_callback(lambda: setattr(record_button, 'disabled', disabled))
        else:
            record_button.disabled = disabled

    n = end_f - start_f + 1
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    try:
        if not writer.isOpened():
            set_status(f"**Failed to open** `{save_path}` for writing.")
            return
        for i, f in enumerate(range(start_f, end_f + 1)):
            rgb = compute_overlay_rgb(f, alpha)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            if i % 20 == 0 or i == n - 1:
                set_status(f"Recording... frame {i + 1}/{n}")
    finally:
        writer.release()
    set_status(f"Saved {n} frames ({start_f}-{end_f} @ {fps} fps) to `{save_path}`")
    set_button_disabled(False)


def on_record_click(event):
    if tiled is None:
        record_status.object = "**Load a source first.**"
        return
    save_dir = record_dir_input.value.strip()
    filename = record_filename_input.value.strip()
    if not filename.lower().endswith('.mp4'):
        filename += '.mp4'
    start_f = min(record_start_input.value, record_end_input.value)
    end_f = max(record_start_input.value, record_end_input.value)
    fps = record_fps_input.value
    alpha = state['alpha']

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as exc:
        record_status.object = f"**Could not create** `{save_dir}`: {exc}"
        return
    save_path = os.path.join(save_dir, filename)

    record_button.disabled = True
    record_status.object = f"Recording frames {start_f}-{end_f} to `{save_path}`..."

    doc = pn.state.curdoc
    threading.Thread(target=_record_worker, args=(save_path, fps, start_f, end_f, alpha, doc),
                     daemon=True).start()


record_button.on_click(on_record_click)

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

record_controls = pn.Column(
    "### Record video",
    "Records the cluster-overlay video (at the current opacity setting) to an mp4 file.",
    record_dir_input,
    record_filename_input,
    pn.Row(record_start_input, record_end_input, record_fps_input),
    record_button,
    record_status,
)

source_controls = pn.Column(
    "### Source",
    pn.Row(path_input, material_selector),
    pn.Row(d_spacing_input, pixel_size_input, tile_size_input, stride_input),
    pn.Row(method_selector, clahe_toggle, load_button),
    load_status,
)

clustering_controls = pn.Column(
    "### PCs used for clustering",
    "Restricts K-means (and the pixel-blend classification) to a subset of the "
    f"{N_COMPONENTS} PCA components. Clustering is independent of tiling and PCA: change "
    "the PCs or the number of clusters, then click **Compute clustering** (no reload needed).",
    pc_checkbox,
    pn.Row(n_clusters_input, cluster_button),
    pc_selection_status,
    show_spectra_toggle,
    spectra_pane,
)

app = pn.Column(
    "# Cluster Map Inspector",
    "Pixel-resolution K-means cluster map (Hann-weighted PC blend), overlaid on the video frame -- "
    "computed dynamically via twist_mapper_GPU.dynamic_pipeline, not read from a cache.",
    source_controls,
    pn.layout.Divider(),
    clustering_controls,
    pn.layout.Divider(),
    bokeh_pane,
    pn.Row(frame_slider, frame_input),
    alpha_slider,
    legend_md,
    pn.layout.Divider(),
    record_controls,
)

app.servable(title="Cluster Map Inspector")
