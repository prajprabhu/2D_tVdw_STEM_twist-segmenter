"""
video_preprocessing.py — Whole-video preprocessing steps applied BEFORE
tiling (contrast with TiledVideo.apply_clahe_to_tile, which is per-tile
and meant for interactive single-tile inspection in tile_inspector_app.py).

Motivation: applying CLAHE per tile over an entire dataset means one
CPU/OpenCV call per tile — for a full video's tile grid that's hundreds
of thousands of calls, which would dominate wall-clock time and defeat
the point of a batched, GPU-resident dynamic pipeline (see
dynamic_pipeline.py). Applying it once per FRAME, before tiling, means
one CPU/OpenCV call per frame — hundreds, not hundreds of thousands —
so this preprocessing step's cost scales with video length, not with
tile count, and stays small relative to the batched GPU work that
follows it.
"""

import numpy as np
import torch
import cv2


def apply_clahe_to_video(video_3d: torch.Tensor, clip_limit: float = 2.0,
                         tile_grid_size: tuple = (8, 8)) -> torch.Tensor:
    """Contrast Limited Adaptive Histogram Equalization, applied once per
    frame to a whole [Frames, H, W] video tensor, before tiling.

    Parameters
    ----------
    video_3d : torch.Tensor [Frames, H, W]
        Grayscale video, values in [0, 1] (same convention as
        TiledVideo.video / TiledVideo.from_path's output).
    clip_limit, tile_grid_size
        Passed straight to cv2.createCLAHE — same defaults as
        TiledVideo.apply_clahe_to_tile, so per-tile (interactive) and
        per-frame (bulk/dynamic) CLAHE behave comparably.

    Returns
    -------
    torch.Tensor [Frames, H, W], same device/dtype as `video_3d`.

    Notes
    -----
    cv2's CLAHE operates on 8-bit single-channel images (CPU, uint8), so
    this round-trips the whole video through CPU/uint8 and back — one
    round trip of `Frames` frames, not `Frames * Rows * Cols` tiles. A
    single cv2.CLAHE object is reused across all frames (state is just
    the clip/grid parameters, not per-call), avoiding the per-call
    allocation TiledVideo.apply_clahe_to_tile pays for a single tile.
    """
    device, dtype = video_3d.device, video_3d.dtype
    frames_u8 = (video_3d.detach().cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_u8 = np.empty_like(frames_u8)
    for i in range(frames_u8.shape[0]):
        enhanced_u8[i] = clahe.apply(frames_u8[i])

    enhanced = enhanced_u8.astype(np.float32) / 255.0
    return torch.from_numpy(enhanced).to(device=device, dtype=dtype)
