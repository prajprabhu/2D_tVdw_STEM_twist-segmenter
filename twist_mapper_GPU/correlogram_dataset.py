"""
correlogram_dataset.py — Per-tile Bragg-ring correlogram Dataset, for PCA
over an entire TiledVideo's tile grid.

CorrelogramDataset is the class-based rebuild of Correlogram_PCA.ipynb's
`Correlogram5DDataset`: for every (frame, row, col) tile in a `TiledVideo`,
compute the 1D circular autocorrelation ("correlogram", Wiener-Khinchin
theorem) of the angular intensity profile around the tile's 1st-order
Bragg ring — a feature vector that isolates each tile's internal layer
twist and is insensitive to the lattice's absolute global orientation,
suitable as PCA input across many tiles.

Where `Correlogram5DDataset` recomputed the Hann window, 2D FFT, and
polar-coordinate ring isolation inline for every `__getitem__`, this
version composes `TiledVideo` (tile access) and `TileFFTAnalysis`
(per-tile FFT magnitude, ring-radius detection, angular profile) instead
— the tensor algebra is unchanged, only reorganized behind the two
existing classes. The one algorithmic difference from `Correlogram5DDataset`
is that the Bragg ring radius is *refined per tile* (via
`TileFFTAnalysis.compute_ring_radius`) around a single physics-guided
`expected_radius`, rather than using one fixed (r_min, r_max) window for
every tile in the video — more robust to local strain/twist variation
across the frame, and no more expensive since it reuses the tile's radial
profile that ring detection already has to build.

Performance note: `TileFFTAnalysis.compute_angular_profile()` deliberately
returns its result as a small CPU NumPy array (it was designed for
interactive single-tile inspection, where minimizing what leaves the GPU
matters most). Reusing it here means every `__getitem__` does a GPU->CPU
round trip for the angular profile before the correlogram step moves it
back to GPU — a real but small cost (the profile is only `n_bins` floats)
traded for not duplicating that logic. If this dataset becomes a
throughput bottleneck for very large tile counts, that's the first place
to optimize (e.g. a batched, GPU-resident angular-profile path).
"""

import torch
from torch.utils.data import Dataset

from .tiled_video import TiledVideo
from .tile_fft_analysis import TileFFTAnalysis


class CorrelogramDataset(Dataset):
    """Yields one Bragg-ring correlogram per tile in a TiledVideo's grid.

    Parameters
    ----------
    tiled_video : TiledVideo
        Already-built tile grid (owns tile_size/stride and the zero-copy
        view_5d) — this is where "given tile size and stride" comes from.
    expected_radius : float
        Physics-guided 1st-order Bragg ring radius in pixels, e.g. from
        `TileFFTAnalysis.expected_ring_radius(tiled_video.tile_size,
        pixel_size_ang, d_spacing_ang)`. Used as the per-tile ring-radius
        search center (see `TileFFTAnalysis.compute_ring_radius`).
    tolerance : float
        Fractional half-width of the ring-radius search window around
        expected_radius (passed through to `compute_ring_radius`).
    dr : float or None
        Half-width in pixels of the annulus sampled for the angular
        profile around the detected ring radius. Defaults to
        `expected_radius * tolerance` if not given.
    n_bins : int
        Number of angular bins for the profile (and hence the length of
        the returned correlogram).
    apply_hann : bool
        Whether to Hann-window each tile before its FFT.
    subtract_mean : bool
        Whether to subtract each tile's mean before windowing/FFT — see
        `TileFFTAnalysis.__init__` for what this does and doesn't change
        (reduces the FFT's DC bin; does not sharpen Bragg peaks).
    prominence_factor : float
        Passed through to `TileFFTAnalysis.compute_ring_radius`.

    Yields (via __getitem__)
    -------------------------
    correlogram : torch.Tensor (n_bins,)
        Zero-lag-centered circular autocorrelation of the tile's angular
        intensity profile, on `tiled_video.view_5d`'s device.
    coords : torch.Tensor (3,), dtype long
        [frame_idx, row_idx, col_idx] for this tile, same device.
    """

    def __init__(self, tiled_video: TiledVideo, expected_radius: float,
                tolerance: float = 0.15, dr: float = None, n_bins: int = 360,
                apply_hann: bool = True, subtract_mean: bool = False,
                prominence_factor: float = 0.10):
        self.tiled_video = tiled_video
        self.Frames, self.Rows, self.Cols = (
            tiled_video.view_5d.shape[0], tiled_video.n_rows, tiled_video.n_cols
        )
        self.total_tiles = self.Frames * self.Rows * self.Cols

        self.expected_radius = expected_radius
        self.tolerance = tolerance
        self.dr = dr if dr is not None else expected_radius * tolerance
        self.n_bins = n_bins
        self.apply_hann = apply_hann
        self.subtract_mean = subtract_mean
        self.prominence_factor = prominence_factor

    def __len__(self) -> int:
        return self.total_tiles

    def _unravel_index(self, idx: int) -> tuple:
        """Flat loader index -> (frame_idx, row_idx, col_idx)."""
        f_idx = idx // (self.Rows * self.Cols)
        rem = idx % (self.Rows * self.Cols)
        r_idx = rem // self.Cols
        c_idx = rem % self.Cols
        return f_idx, r_idx, c_idx

    def __getitem__(self, idx: int) -> tuple:
        f_idx, r_idx, c_idx = self._unravel_index(idx)

        # Stage 0-2 (TileFFTAnalysis): FFT magnitude (computed once, cached),
        # per-tile refined ring radius, angular intensity profile
        tile = self.tiled_video.get_tile(f_idx, r_idx, c_idx)
        analysis = TileFFTAnalysis(tile, f_idx, r_idx, c_idx, apply_hann=self.apply_hann,
                                   subtract_mean=self.subtract_mean)
        analysis.compute_ring_radius(
            self.expected_radius, tolerance=self.tolerance,
            prominence_factor=self.prominence_factor,
        )
        _, angular_intensity = analysis.compute_angular_profile(dr=self.dr, n_bins=self.n_bins)

        # Stage 3 (dataset-specific): Wiener-Khinchin circular autocorrelation
        # of the *unfolded* angular profile, isolating internal layer twist
        # from absolute global lattice orientation.
        device = tile.device
        profile = torch.as_tensor(angular_intensity, device=device, dtype=torch.float32)
        profile_dc = profile - profile.mean()

        fft_1d = torch.fft.fft(profile_dc)
        power_spectrum = torch.abs(fft_1d) ** 2
        correlogram = torch.fft.ifft(power_spectrum).real
        correlogram = torch.fft.fftshift(correlogram)  # center zero-displacement

        coords = torch.tensor([f_idx, r_idx, c_idx], device=device, dtype=torch.long)
        return correlogram, coords

    def __repr__(self):
        return (f"CorrelogramDataset(tiles={self.total_tiles}, "
                f"grid=({self.Frames}x{self.Rows}x{self.Cols}), "
                f"expected_radius={self.expected_radius:.2f}, n_bins={self.n_bins}, "
                f"subtract_mean={self.subtract_mean})")
