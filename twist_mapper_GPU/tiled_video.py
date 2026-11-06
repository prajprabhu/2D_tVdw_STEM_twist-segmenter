"""
tiled_video.py — Zero-copy sliding-window tiling of a GPU video tensor.

TiledVideo owns a grayscale video tensor and its tile-grid geometry
(tile_size, stride), and exposes the zero-copy `as_strided` 5D view
(`[Frames, Rows, Cols, H, W]`) plus per-tile access and plotting so callers
stop re-deriving/re-passing tile_size/stride and stop re-slicing view_5d by
hand. Logic is carried over unchanged from vectorized_code_trial.ipynb's
`extract_windows_nocopy` / `reconstruct_window_nocopy` /
`plot_window_on_original_frame` / `plot_window_and_fft(_with_hann)`
prototypes — this only groups that logic behind one object instead of five
free functions that each repeat (view_5d, frame_idx, row_idx, col_idx, ...).
"""

import torch
import torchvision.io as io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TiledVideo:
    """Owns a [Frames, H, W] GPU video tensor and its zero-copy tile grid.

    Parameters
    ----------
    video_tensor_3d : torch.Tensor
        Grayscale video, shape [Frames, Height, Width], on GPU or CPU.
    tile_size : int
        Square tile side length in pixels.
    stride : int
        Sliding-window stride in pixels (same on both axes).
    crop_black_margins : bool
        If True, detects and crops off any all-black letterbox/pillarbox
        border (rows/columns whose max intensity across the frame never
        exceeds `black_threshold`) before tiling — so tiles are never
        placed over dead padding and plots don't show black bars.
    black_threshold : float
        Intensity threshold (on the [0, 1] normalized frame) used to tell
        real content from black padding, when `crop_black_margins=True`.
    apply_clahe : bool
        If True, `plot_tile_fft` (and any other caller that checks this
        flag) runs each tile through `apply_clahe_to_tile` before Hann
        windowing/FFT — see that method's docstring. Off by default; a
        plain attribute, so it can also just be flipped directly
        (`tiled.apply_clahe = True`) after construction.
    """

    def __init__(self, video_tensor_3d: torch.Tensor, tile_size: int, stride: int,
                crop_black_margins: bool = False, black_threshold: float = 0.02,
                apply_clahe: bool = False):
        self.crop_bbox = None
        if crop_black_margins:
            video_tensor_3d, self.crop_bbox = self._crop_to_content(
                video_tensor_3d, black_threshold
            )

        self.video = video_tensor_3d
        self.tile_size = tile_size
        self.stride = stride
        self.apply_clahe = apply_clahe
        self.view_5d, self.n_rows, self.n_cols = self._build_view(
            video_tensor_3d, tile_size, stride
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_to_content(video_tensor_3d: torch.Tensor, black_threshold: float,
                         ref_frame_idx: int = 0) -> tuple:
        """Detects and crops off all-black letterbox/pillarbox borders.

        Uses one representative frame to find the bounding box of rows/
        columns whose max intensity exceeds `black_threshold`, then slices
        every frame to that box (a view — still zero-copy).

        Returns
        -------
        cropped : torch.Tensor [Frames, H', W']
        bbox : dict {'y0', 'y1', 'x0', 'x1'} — crop bounds in the
            *original* (uncropped) frame's pixel coordinates.
        """
        frame = video_tensor_3d[ref_frame_idx]
        col_max = frame.amax(dim=0)
        row_max = frame.amax(dim=1)

        content_cols = torch.nonzero(col_max > black_threshold).flatten()
        content_rows = torch.nonzero(row_max > black_threshold).flatten()
        if len(content_cols) == 0 or len(content_rows) == 0:
            raise ValueError(
                f"No content above black_threshold={black_threshold} found in "
                f"frame {ref_frame_idx} — lower black_threshold or check the video."
            )

        x0, x1 = int(content_cols.min()), int(content_cols.max()) + 1
        y0, y1 = int(content_rows.min()), int(content_rows.max()) + 1

        cropped = video_tensor_3d[:, y0:y1, x0:x1]
        bbox = {'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1}
        return cropped, bbox

    @staticmethod
    def _build_view(video_tensor_3d: torch.Tensor, tile_size: int, stride: int) -> tuple:
        """Zero-copy sliding-window view via torch.Tensor.as_strided().

        CONSUMES ZERO EXTRA VRAM OR CPU HEAP MEMORY.
        """
        n_frames, h, w = video_tensor_3d.shape

        n_rows = (h - tile_size) // stride + 1
        n_cols = (w - tile_size) // stride + 1

        s_f, s_r, s_c = video_tensor_3d.stride()

        target_shape = (n_frames, n_rows, n_cols, tile_size, tile_size)
        target_strides = (s_f, stride * s_r, stride * s_c, s_r, s_c)

        view_5d = video_tensor_3d.as_strided(size=target_shape, stride=target_strides)
        return view_5d, n_rows, n_cols

    @classmethod
    def from_path(cls, video_path: str, tile_size: int, stride: int,
                  device: str = 'cuda', crop_black_margins: bool = False,
                  black_threshold: float = 0.02, apply_clahe: bool = False) -> 'TiledVideo':
        """Loads a video file straight to a GPU grayscale tensor and tiles it.

        Drops the duplicate RGB channels (grayscale video stored 3x) by
        keeping only channel 0, matching vectorized_code_trial.ipynb's
        `video_tensor[:, 0, :, :]` step -- done on the CPU tensor, before
        the GPU/float32 conversion, not after: moving all 3 channels to
        GPU first would allocate ~3x more memory than the final result
        needs, and a separate (out-of-place) /255.0 division would briefly
        hold a second full-size copy on top of that.
        """
        video_tensor, _, _ = io.read_video(video_path, pts_unit='sec', output_format='TCHW')
        video_channel0 = video_tensor[:, 0, :, :]  # zero-copy CPU view
        video_gpu = video_channel0.to(device=device, dtype=torch.float32)
        video_gpu.div_(255.0)
        return cls(video_gpu, tile_size, stride, crop_black_margins=crop_black_margins,
                   black_threshold=black_threshold, apply_clahe=apply_clahe)

    # ------------------------------------------------------------------
    # Tile access
    # ------------------------------------------------------------------

    def apply_clahe_to_tile(self, tile: torch.Tensor, clip_limit: float = 2.0,
                            tile_grid_size: tuple = (8, 8)) -> torch.Tensor:
        """Contrast Limited Adaptive Histogram Equalization (via OpenCV),
        applied to one [tile_size, tile_size] tile -- a local-contrast
        preprocessing step meant to run before Hann windowing/FFT (see
        `apply_clahe`/`plot_tile_fft`), to bring out faint Bragg-ring
        structure that a uniform intensity scale can wash out, without
        blindly boosting noise the way a single global contrast stretch
        would (CLAHE equalizes within small local tiles, each contrast-
        limited to cap noise amplification).

        cv2's CLAHE operates on 8-bit single-channel images (CPU, uint8),
        so this round-trips the tile through CPU/uint8 and back --
        acceptable for one tile at a time (interactive use), not meant for
        batch preprocessing of every tile in a large dataset.
        """
        device, dtype = tile.device, tile.dtype
        tile_u8 = (tile.detach().cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_u8 = clahe.apply(tile_u8)
        enhanced = enhanced_u8.astype(np.float32) / 255.0
        return torch.from_numpy(enhanced).to(device=device, dtype=dtype)

    def get_tile(self, frame_idx: int, row_idx: int, col_idx: int) -> torch.Tensor:
        """Zero-copy [tile_size, tile_size] slice from the 5D view."""
        return self.view_5d[frame_idx, row_idx, col_idx]

    def tile_bbox(self, row_idx: int, col_idx: int) -> dict:
        """Top-left pixel coordinates of a tile in the original frame."""
        return {'y': row_idx * self.stride, 'x': col_idx * self.stride}

    # ------------------------------------------------------------------
    # Stage 1 plot — raw tile in isolation
    # ------------------------------------------------------------------

    def plot_tile(self, frame_idx: int, row_idx: int, col_idx: int):
        """Plots one tile pulled zero-copy from the 5D view."""
        tile = self.get_tile(frame_idx, row_idx, col_idx)
        bbox = self.tile_bbox(row_idx, col_idx)
        tile_cpu = tile.detach().cpu().numpy()

        plt.figure(figsize=(5, 5))
        plt.imshow(tile_cpu, cmap='gray', vmin=0.0, vmax=1.0)
        plt.title(
            f"Zero-Copy Window View\nFrame: {frame_idx} | Grid: (R:{row_idx}, C:{col_idx})\n"
            f"Pixels: Y:[{bbox['y']}:{bbox['y']+self.tile_size}], "
            f"X:[{bbox['x']}:{bbox['x']+self.tile_size}]",
            fontsize=10,
        )
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Stage 2 plot — tile located on the full original frame
    # ------------------------------------------------------------------

    def plot_tile_on_frame(self, frame_idx: int, row_idx: int, col_idx: int,
                           max_fig_size: float = 10.0):
        """Plots the full frame with the tile's bounding box overlaid.

        figsize is derived from the frame's own aspect ratio (H, W) and
        capped so its larger dimension is `max_fig_size` inches, and the
        axes fill the entire figure canvas (no blank border) — the title
        is drawn as an overlay on the image itself instead of reserving
        margin space above the axes.
        """
        bbox = self.tile_bbox(row_idx, col_idx)
        full_frame_cpu = self.video[frame_idx].detach().cpu().numpy()
        h, w = full_frame_cpu.shape

        # figsize matches the frame's own aspect ratio; larger side capped
        # at max_fig_size inches so oversized frames don't blow up the plot
        scale = max_fig_size / max(h, w)
        figsize = (w * scale, h * scale)

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # no blank margins
        ax.imshow(full_frame_cpu, cmap='gray', vmin=0.0, vmax=1.0)

        rect = patches.Rectangle(
            (bbox['x'], bbox['y']), self.tile_size, self.tile_size,
            linewidth=2, edgecolor='red', facecolor='none',
        )
        ax.add_patch(rect)

        center_x = bbox['x'] + self.tile_size / 2.0
        center_y = bbox['y'] + self.tile_size / 2.0
        ax.plot(center_x, center_y, 'ro', markersize=4)

        title = (
            f"Frame {frame_idx} | Grid (R:{row_idx}, C:{col_idx}) | "
            f"Y:[{bbox['y']}:{bbox['y']+self.tile_size}], "
            f"X:[{bbox['x']}:{bbox['x']+self.tile_size}]"
        )
        ax.text(0.5, 0.99, title, transform=ax.transAxes, ha='center', va='top',
                fontsize=9, color='white',
                bbox=dict(facecolor='black', alpha=0.6, pad=3, edgecolor='none'))

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis('off')
        plt.show()

    # ------------------------------------------------------------------
    # Stage 3 plot — tile's FFT (optionally Hann-windowed)
    # ------------------------------------------------------------------

    def plot_tile_fft(self, frame_idx: int, row_idx: int, col_idx: int, hann: bool = False,
                      subtract_mean: bool = False):
        """Plots the tile alongside (and, if hann=True, the Hann-windowed
        version of) its 2D FFT magnitude spectrum.

        subtract_mean : bool
            If True, subtracts the tile's mean before windowing/FFT —
            zeroes (or, combined with Hann windowing, nearly zeroes) the
            FFT's DC bin. Does not sharpen Bragg peaks (DC removal only
            touches the k=0 bin); it reduces DC-adjacent leakage into the
            radial profile's low-radius bins. See
            TileFFTAnalysis._compute_fft_magnitude for the full explanation.

        Also honors `self.apply_clahe` (see `apply_clahe_to_tile`) — when
        set, CLAHE runs first, before subtract_mean/windowing. Like
        subtract_mean, this only affects `fft_input` — the "Raw Spatial
        Domain" panel still shows the true, unprocessed tile.
        """
        spatial_tile = self.get_tile(frame_idx, row_idx, col_idx)
        device = spatial_tile.device

        fft_input = self.apply_clahe_to_tile(spatial_tile) if self.apply_clahe else spatial_tile
        fft_input = fft_input - fft_input.mean() if subtract_mean else fft_input

        if hann:
            h, w = spatial_tile.shape
            hann_h = torch.hann_window(h, periodic=True, device=device, dtype=spatial_tile.dtype)
            hann_w = torch.hann_window(w, periodic=True, device=device, dtype=spatial_tile.dtype)
            windowed_tile = fft_input * torch.outer(hann_h, hann_w)
        else:
            windowed_tile = fft_input

        fft_complex = torch.fft.fft2(windowed_tile)
        fft_shifted = torch.fft.fftshift(fft_complex)
        fft_magnitude = torch.abs(fft_shifted)
        fft_imag = fft_shifted.imag

        spatial_cpu = spatial_tile.detach().cpu().numpy()
        fft_cpu = fft_magnitude.detach().cpu().numpy()
        fft_log = np.log1p(fft_cpu)

        if hann:
            windowed_cpu = windowed_tile.detach().cpu().numpy()
            fft_imag_cpu = fft_imag.detach().cpu().numpy()
            imag_peak = float(np.abs(fft_imag_cpu).max()) or 1.0

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            axes[0, 0].imshow(spatial_cpu, cmap='gray', vmin=0.0, vmax=1.0)
            axes[0, 0].set_title(f"1. Raw Spatial Domain\n(R:{row_idx}, C:{col_idx})")
            axes[0, 0].axis('off')

            # Once mean-subtracted, pixel values leave [0, 1] — let imshow
            # auto-scale instead of clipping to the raw tile's fixed range.
            windowed_vminmax = {} if subtract_mean else dict(vmin=0.0, vmax=1.0)
            axes[0, 1].imshow(windowed_cpu, cmap='gray', **windowed_vminmax)
            title2 = "2. Hann Windowed Spatial"
            if self.apply_clahe:
                title2 += " (CLAHE)"
            if subtract_mean:
                title2 += " (mean-subtracted)"
            if not (self.apply_clahe or subtract_mean):
                title2 += "\n(Edges Tapered to Zero)"
            axes[0, 1].set_title(title2)
            axes[0, 1].axis('off')

            axes[1, 0].imshow(fft_log, cmap='viridis')
            axes[1, 0].set_title("3. Frequency Domain (2D FFT Magnitude)\nLog Scale")
            axes[1, 0].axis('off')

            # Signed, so a diverging colormap centered at zero (not the
            # log1p scaling used for magnitude, which is undefined for
            # negative values) -- symmetric vmin/vmax keep zero at the
            # colormap's neutral midpoint.
            im = axes[1, 1].imshow(fft_imag_cpu, cmap='RdBu_r', vmin=-imag_peak, vmax=imag_peak)
            axes[1, 1].set_title("4. Frequency Domain (2D FFT Imaginary Part)")
            axes[1, 1].axis('off')
            fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.03)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(spatial_cpu, cmap='gray', vmin=0.0, vmax=1.0)
            axes[0].set_title(f"Spatial Domain\nFrame {frame_idx} | Grid: (R:{row_idx}, C:{col_idx})")
            axes[0].axis('off')

            axes[1].imshow(fft_log, cmap='viridis')
            axes[1].set_title("Frequency Domain (2D FFT Magnitude)\nLog Scale (Low Frequencies Centered)")
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Handoff to the next pipeline stage
    # ------------------------------------------------------------------

    def analyze(self, frame_idx: int, row_idx: int, col_idx: int,
               apply_hann: bool = True, subtract_mean: bool = False) -> 'TileFFTAnalysis':
        """Builds a TileFFTAnalysis for one tile (Bragg ring / angular
        profile / twist estimation pipeline)."""
        from .tile_fft_analysis import TileFFTAnalysis
        tile = self.get_tile(frame_idx, row_idx, col_idx)
        return TileFFTAnalysis(tile, frame_idx, row_idx, col_idx, apply_hann=apply_hann,
                               subtract_mean=subtract_mean)

    def __repr__(self):
        crop = f", crop_bbox={self.crop_bbox}" if self.crop_bbox else ""
        return (f"TiledVideo(frames={self.video.shape[0]}, "
                f"frame_shape={tuple(self.video.shape[1:])}, "
                f"tile_size={self.tile_size}, stride={self.stride}, "
                f"grid=({self.n_rows}x{self.n_cols}){crop})")
