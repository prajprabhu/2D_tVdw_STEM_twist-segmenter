"""
dynamic_pipeline.py — Batched, GPU-vectorized replacement for
CorrelogramDataset's per-tile Python loop, plus an end-to-end
tiling -> correlograms -> PCA -> K-means orchestration function, for
recomputing a full clustering result on demand (new video, new CLAHE
setting, ...) instead of only ever reading a cache built once offline.

Why this exists
----------------
CorrelogramDataset benchmarks at ~2.5ms/tile because every __getitem__ is
one Python-loop iteration doing its own Hann window + FFT + radial/
angular-profile scatter_add + a GPU->CPU->GPU round trip (see its
docstring). For 218,766 tiles that's 9-24 minutes — not because the GPU
is slow (a 128x128 FFT is trivial work), but because thousands of tiny
kernel launches and syncs dominate over actual compute. PCA
(cuml.decomposition.PCA) and K-means (cuml.cluster.KMeans) are already
GPU and already fast (seconds) once a feature matrix exists — dataset
generation is the only stage that needed to change shape to make a full
recompute fast enough to trigger interactively.

The decomposition stage between correlograms and K-means is pluggable
(run_dynamic_pipeline's `method` parameter): 'pca' (default) or 'nmf' --
see compute_nmf's docstring for why NMF needs a different (non-negative,
not z-scored) preprocessing and what that changes about the resulting
components. NMF runs on scikit-learn/CPU (cuML has no NMF as of this
writing), so it's the slowest stage in that mode, but still on the order
of a few seconds for the reference video, not minutes.

BatchedCorrelogramPipeline computes the same math as
CorrelogramDataset + TileFFTAnalysis (verified stage-by-stage against
them), just as a handful of large batched tensor ops (chunked by frame
range to bound VRAM) instead of one Python-loop iteration per tile:

    tiling (zero-copy, already free)
      -> Hann window + batched torch.fft.fft2               (GPU)
      -> batched radial-profile scatter_add                 (GPU)
      -> batched ring-radius (argmax in a shared, physics-   (GPU)
         bounded window -- see class docstring for the
         accuracy tradeoff vs TileFFTAnalysis's
         scipy.signal.find_peaks-based per-tile search)
      -> batched angular-profile scatter_add                (GPU)
      -> batched correlogram (Wiener-Khinchin via FFT)       (GPU)

Unlike TileFFTAnalysis/CorrelogramDataset (kept as-is, unchanged — they
remain the right tool for interactive single-tile inspection in
tile_inspector_app.py), this module chains straight through into a live
PCA + K-means fit. It uses cuML/cuPy (RAPIDS) for that fit when they're
installed, and falls back to scikit-learn on the CPU when they aren't
(see HAS_RAPIDS) -- RAPIDS is Linux/WSL2-only, so the fallback is what
makes this module, and cluster_map_inspector_app.py, run on native
Windows.
"""

from dataclasses import dataclass, field
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import NMF as skNMF

# RAPIDS (cupy + cuML) is optional: it only runs on Linux/WSL2 with an
# NVIDIA GPU. When it's missing (e.g. native Windows, or a CPU-only
# install), PCA and K-means fall back to scikit-learn on the CPU -- same
# algorithms, slower on large videos. HAS_RAPIDS tells callers which
# backend is active.
try:
    import cupy as cp
    from cuml.decomposition import PCA as cumlPCA
    from cuml.cluster import KMeans as cumlKMeans
    HAS_RAPIDS = True
except ImportError:
    cp = None
    from sklearn.decomposition import PCA as skPCA
    from sklearn.cluster import KMeans as skKMeans
    HAS_RAPIDS = False

from .tiled_video import TiledVideo
from .video_preprocessing import apply_clahe_to_video


class BatchedCorrelogramPipeline:
    """Vectorized, batched Bragg-ring correlogram computation for every
    tile in a TiledVideo's grid — see module docstring for why this
    exists and how it differs from CorrelogramDataset.

    Attributes
    ----------
    tiled : TiledVideo
    expected_radius, tolerance, dr, n_bins, apply_hann, subtract_mean
        Same meaning as CorrelogramDataset's constructor arguments.
    frame_chunk_size : int
        Number of frames processed per batch (each batch holds
        frame_chunk_size * Rows * Cols tiles). Bounds peak VRAM use —
        lower this if a chunk doesn't fit. Chunking by frame (not raw
        tile count) keeps chunk boundaries aligned with TiledVideo's
        memory layout, so each chunk is a single contiguous slice of
        view_5d.
    Frames, Rows, Cols, total_tiles, tile_size, device
        Derived from `tiled`.

    Ring-radius accuracy tradeoff
    ------------------------------
    TileFFTAnalysis.compute_ring_radius uses scipy.signal.find_peaks
    (prominence-filtered local maxima, per tile, CPU) and prefers the
    detected peak closest to expected_radius over the tallest one.
    Vectorizing that exactly would require a per-tile Python loop again
    — defeating the purpose. Instead, since every tile shares the same
    physics-bounded search window [r_min, r_max] (expected_radius is one
    constant for the whole run, not per-tile), this class uses a single
    batched argmax within that shared window for all tiles at once. This
    is NOT an approximation of the "no peak found" fallback path —
    TileFFTAnalysis's own code already does exactly this
    (`np.argmax(window) + r_min`) whenever find_peaks finds nothing. The
    one real behavior difference is in windows containing multiple
    comparable peaks: this class always picks the tallest, while
    TileFFTAnalysis picks whichever detected peak sits closest to
    expected_radius. Single-bin noise spikes fool both approaches
    equally (scipy's `distance` parameter only dedupes between accepted
    peaks, it doesn't smooth away an isolated spike).
    """

    def __init__(self, tiled: TiledVideo, expected_radius: float, tolerance: float = 0.15,
                dr: float = None, n_bins: int = 360, apply_hann: bool = True,
                subtract_mean: bool = False, frame_chunk_size: int = 32):
        self.tiled = tiled
        self.expected_radius = expected_radius
        self.tolerance = tolerance
        self.dr = dr if dr is not None else expected_radius * tolerance
        self.n_bins = n_bins
        self.apply_hann = apply_hann
        self.subtract_mean = subtract_mean
        self.frame_chunk_size = frame_chunk_size

        self.Frames, self.Rows, self.Cols = tiled.view_5d.shape[:3]
        self.total_tiles = self.Frames * self.Rows * self.Cols
        self.tile_size = tiled.tile_size
        self.device = tiled.video.device
        self.dtype = tiled.video.dtype

        H = W = self.tile_size
        self.H, self.W = H, W
        self._hann2d = self._build_hann2d() if apply_hann else None
        self._radius_map, self._angle_map, self._radius_idx, self._max_radius = \
            self._build_radial_geometry()
        self._angle_bin_idx = self._build_angle_bins()
        self._r_min, self._r_max = self._search_window()
        # Radial bin *counts* don't depend on tile content, only on
        # geometry (same for every tile) -- compute once, reuse for all.
        self._radial_count = self._build_radial_count()

    # ------------------------------------------------------------------
    # Shared geometry (identical for every tile since tile_size is fixed)
    # ------------------------------------------------------------------

    def _build_hann2d(self) -> torch.Tensor:
        """periodic=True Hann window -- matches TileFFTAnalysis's own
        FFT windowing exactly (NOT pixel_cluster_map.build_symmetric_hann2d's
        periodic=False variant, which is for spatial blending, a
        different use case)."""
        hann_h = torch.hann_window(self.H, periodic=True, device=self.device, dtype=self.dtype)
        hann_w = torch.hann_window(self.W, periodic=True, device=self.device, dtype=self.dtype)
        return torch.outer(hann_h, hann_w)

    def _build_radial_geometry(self):
        cy, cx = self.H // 2, self.W // 2
        yy, xx = torch.meshgrid(
            torch.arange(self.H, device=self.device) - cy,
            torch.arange(self.W, device=self.device) - cx,
            indexing='ij'
        )
        yy, xx = yy.float(), xx.float()
        radius_map = torch.sqrt(yy ** 2 + xx ** 2)
        angle_map = torch.rad2deg(torch.atan2(yy, xx)) % 360.0
        radius_idx = radius_map.round().long().flatten()  # (H*W,)
        max_radius = int(radius_idx.max().item()) + 1
        return radius_map, angle_map, radius_idx, max_radius

    def _build_angle_bins(self) -> torch.Tensor:
        bin_width = 360.0 / self.n_bins
        return (self._angle_map / bin_width).long().clamp(max=self.n_bins - 1).flatten()  # (H*W,)

    def _search_window(self) -> tuple:
        r_min = max(1, int(self.expected_radius * (1.0 - self.tolerance)))
        r_max = min(self._max_radius - 1, int(self.expected_radius * (1.0 + self.tolerance)) + 1)
        if r_min >= r_max:
            raise ValueError(f"Search window empty around expected_radius={self.expected_radius:.1f}")
        return r_min, r_max

    def _build_radial_count(self) -> torch.Tensor:
        count = torch.zeros(self._max_radius, device=self.device, dtype=self.dtype)
        count.scatter_add_(0, self._radius_idx, torch.ones_like(self._radius_idx, dtype=self.dtype))
        return count.clamp(min=1)

    # ------------------------------------------------------------------
    # Batched pipeline
    # ------------------------------------------------------------------

    def compute_correlograms(self, verbose: bool = True) -> tuple:
        """Runs the batched pipeline over every tile.

        Returns
        -------
        correlograms : torch.Tensor (total_tiles, n_bins)
        spatial_coords : torch.Tensor (total_tiles, 3), dtype long
            [frame_idx, row_idx, col_idx] per tile, same row-major
            (frame, row, col) order CorrelogramDataset._unravel_index
            uses, so X_pca reshapes to (Frames, Rows, Cols, ...) the
            same way either path produces it.
        """
        correlogram_blocks = []
        coord_blocks = []
        t0 = time.time()

        for f0 in range(0, self.Frames, self.frame_chunk_size):
            f1 = min(f0 + self.frame_chunk_size, self.Frames)
            chunk_frames = f1 - f0

            tiles_batch = self.tiled.view_5d[f0:f1].reshape(chunk_frames * self.Rows * self.Cols,
                                                             self.H, self.W)
            correlogram_blocks.append(self._process_batch(tiles_batch))

            frame_idx = torch.arange(f0, f1, device=self.device).repeat_interleave(self.Rows * self.Cols)
            row_idx = torch.arange(self.Rows, device=self.device).repeat_interleave(self.Cols).repeat(chunk_frames)
            col_idx = torch.arange(self.Cols, device=self.device).repeat(chunk_frames * self.Rows)
            coord_blocks.append(torch.stack([frame_idx, row_idx, col_idx], dim=1))

            if verbose:
                done = f1 * self.Rows * self.Cols
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else float('nan')
                eta = (self.total_tiles - done) / rate if rate > 0 else float('nan')
                print(f"  {done}/{self.total_tiles} tiles ({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")

        correlograms = torch.cat(correlogram_blocks, dim=0)
        spatial_coords = torch.cat(coord_blocks, dim=0)
        if verbose:
            print(f"Computed {self.total_tiles} correlograms in {time.time() - t0:.1f}s "
                 f"(batched, {self.frame_chunk_size} frames/chunk)")
        return correlograms, spatial_coords

    def _process_batch(self, tiles_batch: torch.Tensor) -> torch.Tensor:
        """tiles_batch: (N, H, W) -> correlograms (N, n_bins)."""
        N = tiles_batch.shape[0]

        if self.subtract_mean:
            tiles_batch = tiles_batch - tiles_batch.mean(dim=(-2, -1), keepdim=True)
        if self.apply_hann:
            tiles_batch = tiles_batch * self._hann2d.unsqueeze(0)

        fft_complex = torch.fft.fft2(tiles_batch)
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        fft_magnitude = torch.abs(fft_shifted)  # (N, H, W)
        mag_flat = fft_magnitude.reshape(N, -1)  # (N, H*W)

        # Radial profile: same bin assignment for every tile (shared
        # geometry) -- one scatter_add call for the whole batch, and
        # radial bin counts (self._radial_count) were already
        # precomputed once in __init__.
        radial_idx_batch = self._radius_idx.unsqueeze(0).expand(N, -1)
        radial_sum = torch.zeros(N, self._max_radius, device=self.device, dtype=self.dtype)
        radial_sum.scatter_add_(1, radial_idx_batch, mag_flat)
        radial_profile = radial_sum / self._radial_count.unsqueeze(0)

        # Ring radius: batched argmax within the shared physics-bounded
        # window (see class docstring for the tradeoff vs
        # TileFFTAnalysis's per-tile scipy.find_peaks search).
        window = radial_profile[:, self._r_min:self._r_max]
        ring_radius = window.argmax(dim=1).to(self.dtype) + self._r_min  # (N,)

        # Angular profile: annulus mask differs per tile (depends on
        # each tile's own ring_radius), angle-bin assignment is shared.
        radius_map_flat = self._radius_map.flatten().unsqueeze(0)  # (1, H*W)
        lo = (ring_radius - self.dr).unsqueeze(1)
        hi = (ring_radius + self.dr).unsqueeze(1)
        annulus_mask = (radius_map_flat >= lo) & (radius_map_flat <= hi)  # (N, H*W)

        angle_idx_batch = self._angle_bin_idx.unsqueeze(0).expand(N, -1)
        masked_vals = mag_flat * annulus_mask
        bin_sum = torch.zeros(N, self.n_bins, device=self.device, dtype=self.dtype)
        bin_sum.scatter_add_(1, angle_idx_batch, masked_vals)
        bin_count = torch.zeros(N, self.n_bins, device=self.device, dtype=self.dtype)
        bin_count.scatter_add_(1, angle_idx_batch, annulus_mask.to(self.dtype))
        angular_profile = bin_sum / bin_count.clamp(min=1)

        # Correlogram: Wiener-Khinchin, same math as
        # CorrelogramDataset.__getitem__, batched along dim=1.
        profile_dc = angular_profile - angular_profile.mean(dim=1, keepdim=True)
        fft_1d = torch.fft.fft(profile_dc, dim=1)
        power_spectrum = torch.abs(fft_1d) ** 2
        correlogram = torch.fft.ifft(power_spectrum, dim=1).real
        correlogram = torch.fft.fftshift(correlogram, dim=1)
        return correlogram


def find_dominant_frequencies(components: np.ndarray, n_bins: int, top_k: int = 3) -> list:
    """For every PCA loading vector (one row of `components`, a periodic
    function over n_bins samples spanning exactly one 360-degree
    revolution), finds the top_k dominant non-DC frequencies by FFT
    magnitude -- see plot_pc_spectra's docstring for what this says about
    a PC physically.

    Parameters
    ----------
    components : np.ndarray (n_components, n_bins)
        PCA loading vectors, e.g. DynamicPipelineResult.pca_components.
    n_bins : int
    top_k : int

    Returns
    -------
    list (length n_components) of list of (cycles_per_rev, magnitude, period_deg)
        tuples, sorted by descending magnitude.
    """
    n_components = components.shape[0]
    cycles = np.arange(n_bins // 2 + 1)

    results = []
    for i in range(n_components):
        spectrum = np.abs(np.fft.rfft(components[i]))
        search = spectrum.copy()
        search[0] = 0  # exclude DC (mean offset carries no periodicity information)
        order = np.argsort(search)[::-1][:top_k]
        top = [(int(cycles[k]), float(spectrum[k]),
               360.0 / cycles[k] if cycles[k] > 0 else float('inf')) for k in order]
        results.append(top)
    return results


def plot_pc_spectra(components: np.ndarray, n_bins: int, dominant_freqs: list = None,
                    explained_variance_ratio: np.ndarray = None, min_cycles_shown: int = 30,
                    component_label: str = 'PC'):
    """For every component: its loading curve (angular-lag domain, left
    column) next to the magnitude spectrum of that loading's own FFT
    (frequency domain, right column), with the dominant non-DC frequency
    marked. Works on either PCA loadings or NMF basis vectors -- both are
    just a (n_components, n_bins) array of per-component weights over the
    same angular-lag axis; `component_label` only changes plot titles
    ('PC' vs 'NMF', say).

    Each component's loading is itself a periodic function over the
    correlogram's n_bins angular-lag bins, spanning exactly one full
    360-degree revolution -- so its own FFT bin index k is directly "k
    cycles per revolution," i.e. a periodicity of 360/k degrees. A
    component dominated by k=6 encodes 60-degree periodicity (hexagonal
    lattice symmetry); k=1 is a single broad lobe; higher k means finer
    angular structure. See dynamic_pipeline module docstring / the
    Correlogram_PCA notebook for the motivation (checking whether
    components correspond to individual Fourier harmonics of the
    underlying angular signal).

    Returns the Figure -- does NOT call plt.show(), so callers decide how
    to display it (plt.show() in a notebook, pn.pane.Matplotlib in a
    Panel app, ...). Callers should plt.close(fig) once they're done with
    it if they don't need the live figure kept open.
    """
    if dominant_freqs is None:
        dominant_freqs = find_dominant_frequencies(components, n_bins)

    n_components = components.shape[0]
    lags = np.linspace(-180, 180, n_bins, endpoint=False)
    cycles = np.arange(n_bins // 2 + 1)

    fig, axes = plt.subplots(n_components, 2, figsize=(13, 2.6 * n_components),
                             gridspec_kw={'width_ratios': [1.15, 1]}, squeeze=False)
    fig.suptitle(f'{component_label} loadings and their angular-frequency spectra',
                fontsize=14, weight='bold', y=0.995)

    for i in range(n_components):
        top1_cycles = dominant_freqs[i][0][0]

        ax_time = axes[i, 0]
        ax_time.plot(lags, components[i], lw=1.2, color='#1f77b4')
        ax_time.axvline(0, color='gray', ls=':', lw=1)
        title = f'{component_label}{i + 1} loading'
        if explained_variance_ratio is not None:
            title += f' ({explained_variance_ratio[i] * 100:.1f}% var)'
        ax_time.set_title(title, fontsize=10, weight='bold')
        ax_time.set_ylabel('Weight')
        ax_time.grid(True, alpha=0.3)
        if i == n_components - 1:
            ax_time.set_xlabel('Angular lag (degrees)')

        spectrum = np.abs(np.fft.rfft(components[i]))
        xmax = max(min_cycles_shown, int(top1_cycles * 1.25) + 5)
        ax_freq = axes[i, 1]
        ax_freq.plot(cycles[:xmax], spectrum[:xmax], lw=1.2, color='#d1483c')
        ax_freq.axvline(top1_cycles, color='gray', ls='--', lw=1)
        ax_freq.set_xlim(0, xmax)
        ax_freq.set_title(f'Spectrum — dominant: {top1_cycles} cyc/rev '
                          f'({360 / top1_cycles:.0f} deg)', fontsize=10, weight='bold')
        ax_freq.set_ylabel('|FFT|')
        ax_freq.grid(True, alpha=0.3)
        if i == n_components - 1:
            ax_freq.set_xlabel('Cycles per 360-degree revolution')

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def compute_nmf(X_nonneg, n_components: int, random_state: int = 0, max_iter: int = 200) -> tuple:
    """Fits Non-negative Matrix Factorization on an already-non-negative
    feature matrix -- the drop-in alternative to cuML's PCA in
    run_dynamic_pipeline's decomposition stage (see that function's
    `method` parameter). scikit-learn, CPU-only: cuML has no NMF
    implementation (checked as of cuml 25.12).

    Unlike PCA, NMF requires X >= 0, so callers must NOT feed it the
    z-scored/standardized correlograms PCA uses (z-scoring introduces
    negative values that would violate NMF's constraint) -- shift the raw
    correlograms to be non-negative instead (run_dynamic_pipeline's
    method='nmf' branch subtracts the global minimum). This also means
    NMF's decomposition is not directly comparable to PCA's: NMF finds an
    additive, parts-based basis (every component's contribution only ever
    adds, never cancels), whereas PCA's components can trade off against
    each other with either sign -- for correlograms specifically, an NMF
    basis vector is a directly interpretable "always-present" angular
    pattern rather than a variance-maximizing direction.

    Everything downstream of this call (z-scoring the returned W before
    K-means, find_dominant_frequencies/plot_pc_spectra on the returned
    components, the pixel-blend classification) is unchanged from the PCA
    path -- see run_dynamic_pipeline and the module docstring.

    Parameters
    ----------
    X_nonneg : cupy.ndarray or np.ndarray, shape (N, n_bins), all entries >= 0
    n_components : int
    random_state : int
    max_iter : int
        scikit-learn's own default (200); this data's low intrinsic rank
        converges well within that on the reference video, but raise it
        if a ConvergenceWarning shows up for a different dataset.

    Returns
    -------
    W : np.ndarray (N, n_components)
        Per-tile NMF scores -- analogous to PCA's X_pca (non-negative,
        NOT yet standardized; callers z-score this the same way as X_pca
        before K-means).
    components : np.ndarray (n_components, n_bins)
        NMF basis vectors -- analogous to PCA's components_.
    """
    if torch.is_tensor(X_nonneg):
        X_np = X_nonneg.detach().cpu().numpy()
    elif cp is not None and isinstance(X_nonneg, cp.ndarray):
        X_np = cp.asnumpy(X_nonneg)
    else:
        X_np = np.asarray(X_nonneg)
    nmf = skNMF(n_components=n_components, init='nndsvda', random_state=random_state,
               max_iter=max_iter)
    W = nmf.fit_transform(X_np)
    return W.astype(np.float32), nmf.components_.astype(np.float32)


def refit_kmeans(X_pca_z, n_clusters: int, random_state: int = 0) -> tuple:
    """Re-fits K-means on an already PCA-transformed + standardized feature
    matrix, without redoing CLAHE/tiling/correlograms/PCA/standardization --
    none of those depend on n_clusters, so changing cluster count only ever
    needs this step. Cheap enough (seconds) to call live, e.g. from a
    "Number of clusters" widget, unlike a full run_dynamic_pipeline() rerun.

    Parameters
    ----------
    X_pca_z : torch.Tensor or np.ndarray, shape (N, n_components)
        Standardized PCA scores -- e.g. DynamicPipelineResult.X_pca after
        the same z-score used internally here, or cluster_map_inspector_app's
        X_pca_grid reshaped to (N, n_components) (already standardized by
        _build_pca_grid).
    n_clusters : int
    random_state : int

    Returns
    -------
    cluster_labels : np.ndarray (N,)
    centroids : np.ndarray (n_clusters, n_components)
    """
    X_np = X_pca_z.detach().cpu().numpy() if torch.is_tensor(X_pca_z) else np.asarray(X_pca_z)
    if HAS_RAPIDS:
        kmeans = cumlKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = cp.asnumpy(kmeans.fit_predict(cp.asarray(X_np)))
        centroids = cp.asnumpy(kmeans.cluster_centers_)
    else:
        kmeans = skKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_np)
        centroids = kmeans.cluster_centers_
    return cluster_labels, centroids


def _fit_pca(X_standardized: torch.Tensor, n_components: int) -> tuple:
    """PCA on the z-scored correlogram matrix: cuML on the GPU when RAPIDS
    is installed, scikit-learn on the CPU otherwise. Returns
    (X_pca (N, n_components), explained_variance_ratio, components), all
    np.ndarray."""
    if HAS_RAPIDS:
        pca = cumlPCA(n_components=n_components)
        X_pca = cp.asnumpy(pca.fit_transform(cp.asarray(X_standardized)))
        return X_pca, cp.asnumpy(pca.explained_variance_ratio_), cp.asnumpy(pca.components_)
    pca = skPCA(n_components=n_components)
    X_pca = pca.fit_transform(X_standardized.detach().cpu().numpy())
    return (X_pca.astype(np.float32), pca.explained_variance_ratio_.astype(np.float32),
            pca.components_.astype(np.float32))


@dataclass
class DynamicPipelineResult:
    """Bundles everything a live app / notebook needs after one dynamic
    pipeline run, plus per-stage timings for judging whether a given
    video/settings combination is actually fast enough to recompute
    on demand."""
    tiled: TiledVideo
    correlograms: torch.Tensor              # (N, n_bins), raw (unstandardized)
    spatial_coords: torch.Tensor            # (N, 3), dtype long, [frame, row, col]
    X_standardized: torch.Tensor            # (N, n_bins) -- z-scored correlograms fed to PCA;
                                            # meaningless (not built) when method='nmf', see below
    X_pca: torch.Tensor                     # (N, n_components) -- PCA scores or NMF's W, per `method`
    explained_variance_ratio: np.ndarray    # (n_components,) -- PCA only; None when method='nmf'
                                            # (not a defined concept for NMF)
    pca_components: np.ndarray              # (n_components, n_bins) -- PCA loadings or NMF basis
                                            # vectors, per `method`
    cluster_labels: np.ndarray              # (N,)
    centroids: np.ndarray                   # (n_clusters, n_components)
    method: str                             # 'pca' or 'nmf' -- which decomposition produced
                                            # X_pca/pca_components/explained_variance_ratio above
    Frames: int
    Rows: int
    Cols: int
    timings: dict = field(default_factory=dict)  # stage name -> seconds


def run_dynamic_pipeline(video_tensor: torch.Tensor, tile_size: int, stride: int,
                         expected_radius: float, n_components: int = 10, n_clusters: int = 5,
                         apply_clahe: bool = False, crop_black_margins: bool = True,
                         tolerance: float = 0.15, n_bins: int = 360, apply_hann: bool = True,
                         subtract_mean: bool = False, frame_chunk_size: int = 32,
                         random_state: int = 0, verbose: bool = True,
                         method: str = 'pca', nmf_max_iter: int = 200) -> DynamicPipelineResult:
    """End-to-end dynamic run: (optional) global per-frame CLAHE -> tile
    -> batched correlogram generation -> standardize -> GPU PCA (or NMF)
    -> GPU K-means. Meant to be fast enough to call again whenever the
    source video or CLAHE setting changes, unlike the notebook pipeline
    this mirrors (correlogram_pca_full.ipynb + tile_phase_map.ipynb /
    pixel_phase_map.ipynb), which computes this once and caches to disk.

    Parameters
    ----------
    video_tensor : torch.Tensor [Frames, H, W]
        Raw video, values in [0, 1], BEFORE tiling -- CLAHE (if
        apply_clahe=True) runs on this whole tensor first, once per
        frame (see video_preprocessing.apply_clahe_to_video), not per
        tile, precisely so this step doesn't reintroduce the
        per-tile-CPU-call bottleneck the batched pipeline exists to
        avoid.
    tile_size, stride, crop_black_margins
        Passed to TiledVideo.
    expected_radius, tolerance, n_bins, apply_hann, subtract_mean, dr
        Passed to BatchedCorrelogramPipeline.
    n_components, n_clusters, random_state
        Passed to the decomposition stage (PCA or NMF, see `method`) / K-means.
    frame_chunk_size
        Passed to BatchedCorrelogramPipeline (bounds VRAM use).
    method : 'pca' or 'nmf'
        Which decomposition to run on the correlogram matrix before
        K-means. 'pca' (default): cuML PCA on z-scored correlograms, same
        as before this parameter existed. 'nmf': scikit-learn NMF
        (compute_nmf) on the correlograms shifted to be non-negative
        (z-scoring would violate NMF's X >= 0 requirement, so this branch
        does NOT z-score first -- see compute_nmf's docstring for what
        that changes about the resulting components). Everything after
        this stage (z-scoring the scores, K-means, pixel-blend
        classification, PC-spectra plotting) is identical either way.
    nmf_max_iter
        Passed to compute_nmf when method='nmf'; ignored otherwise.

    Returns
    -------
    DynamicPipelineResult
    """
    timings = {}
    method = method.lower()
    if method not in ('pca', 'nmf'):
        raise ValueError(f"method must be 'pca' or 'nmf', got {method!r}")

    t0 = time.time()
    if apply_clahe:
        video_tensor = apply_clahe_to_video(video_tensor)
    timings['clahe'] = time.time() - t0

    t0 = time.time()
    tiled = TiledVideo(video_tensor, tile_size=tile_size, stride=stride,
                       crop_black_margins=crop_black_margins)
    timings['tiling'] = time.time() - t0

    t0 = time.time()
    pipeline = BatchedCorrelogramPipeline(tiled, expected_radius=expected_radius, tolerance=tolerance,
                                          n_bins=n_bins, apply_hann=apply_hann,
                                          subtract_mean=subtract_mean, frame_chunk_size=frame_chunk_size)
    correlograms, spatial_coords = pipeline.compute_correlograms(verbose=verbose)
    timings['correlograms'] = time.time() - t0

    t0 = time.time()
    # Standardization runs in torch (on whatever device the correlograms
    # live on), so it needs neither cupy nor a GPU. correction=0 matches
    # cupy/numpy's default population std (ddof=0) used previously.
    X = correlograms
    if method == 'nmf':
        # NMF requires X >= 0 -- shift by the global min instead of
        # z-scoring (z-scoring would reintroduce negative values). See
        # compute_nmf's docstring for what this means for the result.
        X_standardized = X - X.min()
    else:
        X_standardized = (X - X.mean(dim=0)) / (X.std(dim=0, correction=0) + 1e-8)
    timings['standardize'] = time.time() - t0

    t0 = time.time()
    if method == 'nmf':
        X_pca_np, pca_components = compute_nmf(X_standardized, n_components,
                                               random_state=random_state, max_iter=nmf_max_iter)
        explained_variance_ratio = None  # not a defined concept for NMF
    else:
        X_pca_np, explained_variance_ratio, pca_components = _fit_pca(X_standardized, n_components)
    timings['decompose'] = time.time() - t0

    t0 = time.time()
    X_pca_z = (X_pca_np - X_pca_np.mean(axis=0)) / (X_pca_np.std(axis=0) + 1e-8)
    cluster_labels, centroids = refit_kmeans(X_pca_z, n_clusters, random_state=random_state)
    timings['kmeans'] = time.time() - t0

    timings['total'] = sum(timings.values())
    if verbose:
        print("Stage timings (s):", {k: round(v, 2) for k, v in timings.items()})

    return DynamicPipelineResult(
        tiled=tiled,
        correlograms=correlograms,
        spatial_coords=spatial_coords,
        X_standardized=X_standardized.to(tiled.video.device),
        X_pca=torch.as_tensor(X_pca_np, device=tiled.video.device),
        explained_variance_ratio=explained_variance_ratio,
        pca_components=pca_components,
        cluster_labels=cluster_labels,
        centroids=centroids,
        method=method,
        Frames=pipeline.Frames,
        Rows=pipeline.Rows,
        Cols=pipeline.Cols,
        timings=timings,
    )
