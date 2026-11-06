"""
twist_mapper_GPU — torch/GPU-native, class-based rebuild of the twist-angle
pipeline prototyped in vectorized_code_trial.ipynb / Correlogram_PCA.ipynb.

Replaces the earlier twist_mapper package (CuPy/NumPy per-tile pipeline)
on this branch. Three classes plus a small functional module:

    TiledVideo          — owns a video tensor + its zero-copy as_strided
                           tile grid (see tiled_video.py).
    TileFFTAnalysis      — Bragg-ring-radius -> angular-profile ->
                           twist-angle pipeline for one tile (see
                           tile_fft_analysis.py).
    CorrelogramDataset   — per-tile Bragg-ring correlogram Dataset over an
                           entire TiledVideo's grid, for PCA across tiles
                           (see correlogram_dataset.py).
    pixel_cluster_map    — Hann-weighted blending of per-tile PCA scores
                           onto pixel space + nearest-centroid classification,
                           for pixel-resolution cluster maps from a tile-
                           resolution K-means fit (see pixel_cluster_map.py).
    materials            — Material dataclass + registry of common 2D
                           materials' lattice constants, ported over from
                           the earlier twist_mapper package (see
                           materials.py).
    video_preprocessing   — Whole-video preprocessing applied before
                           tiling, e.g. per-frame CLAHE (see
                           video_preprocessing.py).
    source_loading        — Shared video/image/.npz source-loading
                           utilities for the interactive apps (see
                           source_loading.py).
    dynamic_pipeline      — Batched, GPU-vectorized replacement for
                           CorrelogramDataset's per-tile loop, plus an
                           end-to-end tiling->correlograms->PCA->K-means
                           orchestration function, for recomputing a full
                           clustering result on demand instead of only
                           reading a cache built once offline (see
                           dynamic_pipeline.py). Not imported here since
                           it depends on cuML/cuPy, unlike every other
                           module in this package — import it explicitly:
                           `from twist_mapper_GPU.dynamic_pipeline import
                           run_dynamic_pipeline`.

Usage
-----
    from twist_mapper_GPU import TiledVideo, TileFFTAnalysis, CorrelogramDataset

    tiled = TiledVideo.from_path('video.mp4', tile_size=128, stride=32,
                                  crop_black_margins=True)
    tiled.plot_tile_on_frame(frame_idx=0, row_idx=5, col_idx=10)

    analysis = tiled.analyze(frame_idx=0, row_idx=5, col_idx=10)
    expected_r = analysis.expected_ring_radius(tiled.tile_size, pixel_size_ang=0.25, d_spacing_ang=3.14)
    analysis.compute_ring_radius(expected_r)
    analysis.plot_ring_radius()

    analysis.compute_angular_profile()
    analysis.plot_angular_profile()

    analysis.compute_twist()
    analysis.plot_correlogram()

    # PCA over every tile's correlogram
    dataset = CorrelogramDataset(tiled, expected_radius=expected_r)
"""

__version__ = "0.1.0"

from .tiled_video import TiledVideo
from .tile_fft_analysis import TileFFTAnalysis
from .correlogram_dataset import CorrelogramDataset
from .pixel_cluster_map import (
    build_symmetric_hann2d,
    blend_pc_field,
    classify_nearest_centroid,
    blend_and_classify_frame,
)
from .materials import Material, MATERIALS, get_material
from .video_preprocessing import apply_clahe_to_video
from .source_loading import (
    load_video_tensor,
    extract_pixel_size_ang,
    default_d_spacing_ang,
    IMAGE_EXTENSIONS,
    NPZ_STACK_KEYS,
    NPZ_PIXEL_SIZE_KEYS,
)

__all__ = [
    "TiledVideo", "TileFFTAnalysis", "CorrelogramDataset",
    "build_symmetric_hann2d", "blend_pc_field", "classify_nearest_centroid",
    "blend_and_classify_frame",
    "Material", "MATERIALS", "get_material",
    "apply_clahe_to_video",
    "load_video_tensor", "extract_pixel_size_ang", "default_d_spacing_ang",
    "IMAGE_EXTENSIONS", "NPZ_STACK_KEYS", "NPZ_PIXEL_SIZE_KEYS",
]
