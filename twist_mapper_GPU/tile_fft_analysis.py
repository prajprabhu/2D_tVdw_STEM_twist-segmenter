"""
tile_fft_analysis.py — Bragg-ring-radius -> angular-profile -> twist-angle
pipeline for a single tile.

TileFFTAnalysis takes one zero-copy tile (e.g. from TiledVideo.get_tile())
and walks the three-stage pipeline against it, computing the tile's FFT
magnitude exactly once (cached on self) instead of recomputing it on every
stage the way vectorized_code_trial.ipynb's free functions
(`estimate_bragg_ring_radius`, `compute_angular_intensity_profile`,
`estimate_twist_from_correlogram`, each calling `_fft_magnitude_for_tile`
independently) did. Each stage has a `compute_*` method that computes and
caches its result, and a matching `plot_*` method that only plots the
already-cached result — so plotting never triggers recomputation, and
recomputation never happens silently as a side effect of plotting.

Only tiny 1D profiles (length ~ H/2 for the radial profile, `n_bins` for
the angular profile) ever leave the GPU — the underlying algorithms are
unchanged from the notebook prototypes, just moved from free functions
with repeated (view_5d, frame_idx, row_idx, col_idx, ...) arguments into
methods with `self` state.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class TileFFTAnalysis:
    """Bragg-ring / angular-profile / twist-angle pipeline for one tile.

    Parameters
    ----------
    tile : torch.Tensor
        [tile_size, tile_size] tile, e.g. from TiledVideo.get_tile().
    frame_idx, row_idx, col_idx : int
        Indices identifying this tile — kept only for plot titles/labels.
    apply_hann : bool
        Whether to Hann-window the tile before the FFT (default True).
    subtract_mean : bool
        Whether to subtract the tile's mean before windowing/FFT (default
        False). Zeroes the FFT's DC bin (for real input, DC = sum of pixel
        values, so subtracting the mean first makes that sum exactly 0 —
        Hann-windowing afterwards reweights pixels non-uniformly, so DC
        ends up only close to 0, not exactly, but the huge DC spike that
        otherwise dominates the raw magnitude spectrum is removed either
        way). Does NOT sharpen Bragg peaks — DC removal only touches the
        single k=0 bin and has no effect on any other frequency's
        amplitude/phase, so peak width is unchanged. What it can improve
        is peak *prominence* in the radial profile used by
        `compute_ring_radius` — less DC-adjacent leakage bleeding into
        the low-radius bins — particularly useful when `expected_radius`
        is small (the ring sits close to DC).
    """

    def __init__(self, tile: torch.Tensor, frame_idx: int, row_idx: int, col_idx: int,
                apply_hann: bool = True, subtract_mean: bool = False):
        self.frame_idx = frame_idx
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.subtract_mean = subtract_mean

        self.fft_magnitude = self._compute_fft_magnitude(tile, apply_hann, subtract_mean)
        self.H, self.W = self.fft_magnitude.shape
        self.cy, self.cx = self.H // 2, self.W // 2

        # Cached stage results — populated by the compute_* methods
        self._radial_profile = None
        self.expected_radius = None
        self.ring_radius = None
        self._ring_search_window = None
        self.dr = None
        self.angles = None
        self.angular_intensity = None
        self.n_sectors = None
        self.twist_angle = None
        self.n_folds = None
        self._folded_profile = None
        self._sector_angles = None
        self._lags = None
        self._autocorr = None
        self._corr_threshold = None

    # ------------------------------------------------------------------
    # Stage 0 — FFT magnitude (computed once at construction)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fft_magnitude(tile: torch.Tensor, apply_hann: bool,
                               subtract_mean: bool = False) -> torch.Tensor:
        h, w = tile.shape
        device = tile.device

        if subtract_mean:
            # Subtract before windowing: this exactly zeroes the pre-window
            # DC term (sum of pixel values); windowing afterwards leaves a
            # small residual DC (windowing reweights pixels non-uniformly)
            # rather than an exact zero, but removes the dominant spike.
            tile = tile - tile.mean()

        if apply_hann:
            hann_h = torch.hann_window(h, periodic=True, device=device, dtype=tile.dtype)
            hann_w = torch.hann_window(w, periodic=True, device=device, dtype=tile.dtype)
            tile = tile * torch.outer(hann_h, hann_w)

        fft_complex = torch.fft.fft2(tile)
        fft_shifted = torch.fft.fftshift(fft_complex)
        return torch.abs(fft_shifted)

    @staticmethod
    def expected_ring_radius(tile_size: int, pixel_size_ang: float,
                             d_spacing_ang: float) -> float:
        """Physics-guided expected 1st-order Bragg ring radius (px).

            r = tile_size_px * pixel_size_ang / d_spacing_ang

        d_spacing_ang is material- and plane-dependent — look it up (e.g.
        from a populated ImageStack-equivalent) or set it explicitly for
        the (1,0) plane you expect.
        """
        if d_spacing_ang is None or d_spacing_ang <= 0:
            raise ValueError(f"d_spacing_ang must be positive, got {d_spacing_ang}")
        if pixel_size_ang is None or pixel_size_ang <= 0:
            raise ValueError(f"pixel_size_ang must be positive, got {pixel_size_ang}")
        return tile_size * pixel_size_ang / d_spacing_ang

    # ------------------------------------------------------------------
    # Stage 1 — radial profile / ring radius
    # ------------------------------------------------------------------

    @property
    def radial_profile(self) -> np.ndarray:
        """Azimuthally averaged radial profile of the FFT magnitude.

        Lazily computed once and cached — only this tiny 1D profile
        (length ~ H/2) ever reaches the CPU.
        """
        if self._radial_profile is None:
            device = self.fft_magnitude.device
            yy, xx = torch.meshgrid(
                torch.arange(self.H, device=device) - self.cy,
                torch.arange(self.W, device=device) - self.cx,
                indexing='ij'
            )
            radius_idx = torch.sqrt(yy.float() ** 2 + xx.float() ** 2).round().long().flatten()
            values = self.fft_magnitude.flatten()

            max_radius = int(radius_idx.max().item()) + 1
            radial_sum = torch.zeros(max_radius, device=device, dtype=values.dtype)
            radial_count = torch.zeros(max_radius, device=device, dtype=values.dtype)
            radial_sum.scatter_add_(0, radius_idx, values)
            radial_count.scatter_add_(0, radius_idx, torch.ones_like(values))

            self._radial_profile = (radial_sum / radial_count.clamp(min=1)).detach().cpu().numpy()
        return self._radial_profile

    def compute_ring_radius(self, expected_radius: float, tolerance: float = 0.15,
                            prominence_factor: float = 0.10) -> float:
        """Detects the 1st-order Bragg ring radius, guided by `expected_radius`.

        `expected_radius` anchors the search to a narrow window around the
        physically expected location (a blind global search would lock
        onto the DC/low-frequency falloff instead of the ring) — the
        peak closest to it within that window (or, failing that, the
        window's argmax) is reported as the ring. Caches the result for
        `plot_ring_radius()` / `compute_angular_profile()`.
        """
        profile = self.radial_profile
        self.expected_radius = expected_radius

        r_min = max(1, int(expected_radius * (1.0 - tolerance)))
        r_max = min(len(profile) - 1, int(expected_radius * (1.0 + tolerance)) + 1)
        if r_min >= r_max:
            raise ValueError(f"Search window empty around expected_radius={expected_radius:.1f}")
        self._ring_search_window = (r_min, r_max)

        window = profile[r_min:r_max]
        min_prom = prominence_factor * window.max()
        peak_idx, _ = find_peaks(window, prominence=min_prom, distance=2)

        if len(peak_idx) == 0:
            ring_radius = float(np.argmax(window) + r_min)
        else:
            peak_radii = peak_idx + r_min
            ring_radius = float(peak_radii[np.argmin(np.abs(peak_radii - expected_radius))])

        self.ring_radius = ring_radius
        return ring_radius

    def plot_ring_radius(self):
        """Plots the radial profile (with search window/peak) and the FFT
        magnitude with the detected ring overlaid. Requires
        compute_ring_radius() to have been called first."""
        if self.ring_radius is None:
            raise ValueError("Call compute_ring_radius() before plot_ring_radius().")

        profile = self.radial_profile
        r_min, r_max = self._ring_search_window

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.semilogy(profile, 'k-', lw=0.8)
        ax1.axvspan(r_min, r_max, alpha=0.15, color='#E24B4A', label=f'search window [{r_min},{r_max}]')
        ax1.axvline(self.ring_radius, color='#E24B4A', ls='--', lw=1.5,
                    label=f'r_1st={self.ring_radius:.1f}px')
        ax1.axvline(self.expected_radius, color='royalblue', ls=':', lw=1, alpha=0.7,
                    label=f'expected={self.expected_radius:.1f}px')
        ax1.set_xlabel('Radius (px from FFT centre)')
        ax1.set_ylabel('Mean |F| (log scale)')
        ax1.set_title(f'Radial Profile\nFrame {self.frame_idx} | Grid (R:{self.row_idx}, C:{self.col_idx})')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        fft_log = np.log1p(self.fft_magnitude.detach().cpu().numpy())
        ax2.imshow(fft_log, cmap='viridis')
        theta = np.linspace(0, 2 * np.pi, 360)
        ax2.plot(self.cx + self.ring_radius * np.cos(theta), self.cy + self.ring_radius * np.sin(theta),
                  'w--', lw=1.2, alpha=0.85, label=f'r_1st={self.ring_radius:.1f}px')
        ax2.set_title('2D FFT Magnitude (log scale)')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Stage 2 — angular intensity profile
    # ------------------------------------------------------------------

    def compute_angular_profile(self, dr: float = 2.0, n_bins: int = 360) -> tuple:
        """Samples the angular intensity profile around the Bragg ring
        (ring_radius +/- dr), binning by azimuth into n_bins bins.

        Requires compute_ring_radius() to have been called first. Caches
        the result for plot_angular_profile() / compute_twist().
        """
        if self.ring_radius is None:
            raise ValueError("Call compute_ring_radius() before compute_angular_profile().")

        device = self.fft_magnitude.device
        yy, xx = torch.meshgrid(
            torch.arange(self.H, device=device) - self.cy,
            torch.arange(self.W, device=device) - self.cx,
            indexing='ij'
        )
        yy, xx = yy.float(), xx.float()
        radius_map = torch.sqrt(yy ** 2 + xx ** 2)
        angle_map = torch.rad2deg(torch.atan2(yy, xx)) % 360.0

        annulus_mask = (radius_map >= (self.ring_radius - dr)) & (radius_map <= (self.ring_radius + dr))

        bin_width = 360.0 / n_bins
        bin_idx = (angle_map / bin_width).long().clamp(max=n_bins - 1)

        masked_bins = bin_idx[annulus_mask]
        masked_vals = self.fft_magnitude[annulus_mask]

        bin_sum = torch.zeros(n_bins, device=device, dtype=self.fft_magnitude.dtype)
        bin_count = torch.zeros(n_bins, device=device, dtype=self.fft_magnitude.dtype)
        bin_sum.scatter_add_(0, masked_bins, masked_vals)
        bin_count.scatter_add_(0, masked_bins, torch.ones_like(masked_vals))

        self.dr = dr
        self.angular_intensity = (bin_sum / bin_count.clamp(min=1)).detach().cpu().numpy()
        self.angles = np.arange(n_bins) * bin_width
        return self.angles, self.angular_intensity

    def plot_angular_profile(self, n_sectors: int = 6):
        """Plots the angular intensity profile (linear + polar, with the
        polar view divided into n_sectors equal sectors — 6 for hexagonal
        symmetry, 4 for square). Requires compute_angular_profile() first."""
        if self.angular_intensity is None:
            raise ValueError("Call compute_angular_profile() before plot_angular_profile().")

        self.n_sectors = n_sectors
        angles, intensity = self.angles, self.angular_intensity

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(angles, intensity, 'b-', lw=1.0)
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Mean |F| on ring')
        ax1.set_title(f'Angular Intensity Profile\nr={self.ring_radius:.1f}px +/- {self.dr}px | '
                      f'Frame {self.frame_idx} (R:{self.row_idx}, C:{self.col_idx})')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2, projection='polar')

        sector_span = 360.0 / n_sectors
        sector_edges_deg = np.arange(0, 360, sector_span)
        r_max = float(intensity.max()) * 1.05 if intensity.max() > 0 else 1.0
        for i, edge in enumerate(sector_edges_deg):
            theta_fill = np.deg2rad([edge, edge + sector_span])
            color = '#cccccc' if i % 2 == 0 else 'white'
            ax2.fill_between(theta_fill, 0, r_max, color=color, alpha=0.3, zorder=0)

        ax2.plot(np.deg2rad(angles), intensity, 'b-', lw=1.0, zorder=2)
        ax2.set_xticks(np.deg2rad(sector_edges_deg))
        ax2.set_xticklabels([f'{d:.0f}°' for d in sector_edges_deg])
        ax2.grid(True, color='gray', alpha=0.5)
        ax2.set_ylim(0, r_max)
        ax2.set_title(f'Polar View ({n_sectors} sectors)', pad=20)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Stage 3 — twist angle from correlogram
    # ------------------------------------------------------------------

    def compute_twist(self, n_folds: int = 6, smooth_sigma_deg: float = 1.0,
                      min_twist_deg: float = 0.5, n_mad: float = 6.0) -> float:
        """Estimates the twist angle via circular autocorrelation
        ("correlogram") of the folded angular intensity profile.

        Folds the 360-degree profile into n_folds symmetric sectors (6 for
        hexagonal, 4 for square) and averages them, boosting SNR by
        sqrt(n_folds). The circular autocorrelation of the folded profile
        peaks at a lag equal to the twist angle, even when the two layers'
        Bragg peaks are not individually resolved. Peaks are detected via
        a robust MAD-based adaptive threshold over the search region
        (excludes the near-zero-lag central peak, which is real structure,
        not noise). Requires compute_angular_profile() first.
        """
        if self.angular_intensity is None:
            raise ValueError("Call compute_angular_profile() before compute_twist().")

        intensity = self.angular_intensity
        self.n_folds = n_folds

        base_interval = 360.0 / n_folds
        n_bins = len(intensity)
        bps = n_bins // n_folds

        folded = intensity[:bps * n_folds].reshape(n_folds, bps).mean(axis=0)
        sec_angles = np.linspace(0, base_interval, bps, endpoint=False)

        bin_width = base_interval / bps
        sigma_bins = max(1.0, smooth_sigma_deg / bin_width)

        # Folded profile is periodic (ends are adjacent sectors on the
        # physical ring), so smoothing must wrap, not reflect at the edges.
        smoothed = gaussian_filter1d(folded.astype(float), sigma=sigma_bins, mode='wrap')
        smoothed = smoothed - smoothed.mean()

        F = np.fft.rfft(smoothed)
        autocorr = np.fft.irfft(F * np.conj(F)).real
        autocorr = autocorr[:bps // 2 + 1]
        if autocorr[0] > 1e-12:
            autocorr = autocorr / autocorr[0]

        lags = np.arange(len(autocorr)) * bin_width
        min_lag_bins = max(2, int(min_twist_deg / bin_width))

        search = autocorr[min_lag_bins:]
        med = np.median(search)
        mad = np.median(np.abs(search - med))
        threshold = max(0.0, med + n_mad * mad)
        denoised = np.where(search >= threshold, search, 0.0)

        peaks_idx, _ = find_peaks(denoised, distance=min_lag_bins)

        if len(peaks_idx) == 0:
            twist_angle = 0.0
        else:
            twist_angle = float(lags[peaks_idx[0] + min_lag_bins])

        self._folded_profile = folded
        self._sector_angles = sec_angles
        self._lags = lags
        self._autocorr = autocorr
        self._corr_threshold = threshold
        self.twist_angle = twist_angle
        return twist_angle

    def plot_correlogram(self):
        """Plots the folded angular profile and its correlogram (circular
        autocorrelation), with the detected twist-angle peak marked.
        Requires compute_twist() to have been called first."""
        if self.twist_angle is None:
            raise ValueError("Call compute_twist() before plot_correlogram().")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(self._sector_angles, self._folded_profile, 'g-', lw=1.2)
        ax1.set_xlabel(f'Angle within {360.0/self.n_folds:.0f} deg sector')
        ax1.set_ylabel('Mean |F|')
        ax1.set_title(f'Folded Angular Profile (n_folds={self.n_folds})')
        ax1.grid(True, alpha=0.3)

        ax2.plot(self._lags, self._autocorr, 'k-', lw=1.0)
        ax2.axhline(self._corr_threshold, color='gray', ls=':', lw=1,
                    label=f'threshold={self._corr_threshold:.3f}')
        if self.twist_angle > 0:
            ax2.axvline(self.twist_angle, color='#E24B4A', ls='--', lw=1.5,
                        label=f'twist={self.twist_angle:.2f} deg')
        ax2.set_xlabel('Lag (degrees)')
        ax2.set_ylabel('Normalized autocorrelation')
        ax2.set_title('Correlogram (Circular Autocorrelation)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return (f"TileFFTAnalysis(frame={self.frame_idx}, row={self.row_idx}, col={self.col_idx}, "
                f"ring_radius={self.ring_radius}, twist_angle={self.twist_angle})")
