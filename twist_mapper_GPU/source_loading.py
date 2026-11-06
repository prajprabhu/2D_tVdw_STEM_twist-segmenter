"""
source_loading.py — Shared video/image/.npz source-loading utilities for
the interactive apps (tile_inspector_app.py, cluster_map_inspector_app.py).

Extracted from tile_inspector_app.py (where this was first built and
tested) so a second app doesn't have to duplicate/diverge from the same
"load a video, image, or .npz array-stack as a [Frames, H, W] tensor,
plus its pixel-size metadata if any" logic. No cuML/cuPy dependency here
(unlike dynamic_pipeline.py) -- safe to import from anywhere.
"""

import os

import numpy as np
import torch
import torchvision.io as tvio
from PIL import Image

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
NPZ_STACK_KEYS = ('im_array', 'video', 'images', 'data', 'array')  # tried in this order
NPZ_PIXEL_SIZE_KEYS = ('pixel_size', 'pixel_size_ang', 'pixel_size_angstrom', 'pixelsize')


def default_d_spacing_ang(material) -> float:
    """A sensible first-order Bragg-plane spacing to prefill a d-spacing
    box with when `material` is picked -- NOT a fixed physical property of
    the material (which plane is actually imaged is an experimental
    choice, per twist_mapper_GPU.materials' docstring), just a reasonable
    starting point the user can edit afterward."""
    if material.lattice_type == 'hexagonal':
        return material.lattice_constant_ang * (3 ** 0.5) / 2
    return material.lattice_constant_ang  # square lattice primary spacing


def _normalize_to_unit_range(arr: np.ndarray) -> np.ndarray:
    """float32 array rescaled by its own observed peak (not a fixed bit
    depth), so 8-bit, 16-bit, or already-[0,1]-normalized-float sources
    (e.g. an .npz array someone pre-normalized) all come out consistently
    scaled -- unlike video frames (always uint8 by decode convention), a
    single array/image's original bit depth isn't reliably knowable from
    dtype alone (e.g. a 16-bit TIFF still reads as plain int32 via PIL)."""
    arr = arr.astype(np.float32)
    peak = float(arr.max())
    return arr / peak if peak > 0 else arr


def _load_npz_stack(path: str) -> np.ndarray:
    """Loads an .npz archive's image stack as a [Frames, H, W] (or [H, W])
    array. Tries NPZ_STACK_KEYS first (this project's own datasets use
    'im_array' -- see e.g. WSSe_haadf.npz), then falls back to the sole
    array if the archive only contains one, else raises listing the keys
    found so the caller knows what to pick."""
    archive = np.load(path)
    key = next((k for k in NPZ_STACK_KEYS if k in archive.files), None)
    if key is None:
        if len(archive.files) == 1:
            key = archive.files[0]
        else:
            raise ValueError(
                f"can't tell which array is the image stack -- keys found: {archive.files}. "
                f"Expected one of {list(NPZ_STACK_KEYS)} or a single array."
            )
    arr = archive[key]
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"array '{key}' has shape {arr.shape} -- expected 2D (H,W) or 3D (Frames,H,W)")
    return arr


def extract_pixel_size_ang(path: str):
    """Best-effort pixel-size (Å/pixel) lookup from the source file's own
    metadata. Returns None (leave it to the user) if not found or not
    applicable.

    Currently only .npz archives carry this reliably -- this project's own
    datasets store it under 'pixel_size' (see WSSe_haadf.npz), tried along
    with a few other common spellings (NPZ_PIXEL_SIZE_KEYS). Values stored
    in SI meters (< 1e-6) are converted to Å. Videos/images
    have no dependable place to store physical calibration, so those
    always return None.
    """
    if os.path.splitext(path)[1].lower() != '.npz':
        return None
    try:
        archive = np.load(path)
    except Exception:
        return None
    key = next((k for k in NPZ_PIXEL_SIZE_KEYS if k in archive.files), None)
    if key is None:
        return None
    try:
        value = float(archive[key])
    except (TypeError, ValueError):
        return None
    # Some archives store the pixel size in SI meters rather than Å (e.g.
    # Gomb-Net's Moire.npz: 1.06e-11, i.e. 0.106 Å). No real STEM pixel is
    # anywhere near 1e-6 Å, so anything that small must be meters.
    if 0 < value < 1e-6:
        value *= 1e10
    return value


def load_video_tensor(path: str, device) -> torch.Tensor:
    """Loads `path` as a [Frames, H, W] grayscale tensor on `device`.

    A real video decodes to its normal multi-frame tensor (unchanged from
    TiledVideo.from_path's own loading, duplicated here since from_path is
    video-only). A still image, or an .npz array stack, is loaded as a
    **one-frame** (or N-frame, for an .npz stack) **video** -- same
    [Frames, H, W] shape TiledVideo already expects, so TiledVideo itself
    (and everything downstream) needs no changes to accept any of these.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        arr = _normalize_to_unit_range(_load_npz_stack(path))
        return torch.from_numpy(arr).to(device=device)
    elif ext in IMAGE_EXTENSIONS:
        img = Image.open(path)
        if img.mode != 'L' and 'I' not in img.mode:  # 'I'/'I;16'/... = integer grayscale modes
            img = img.convert('L')  # matches TiledVideo.from_path's channel-0 convention
        arr = _normalize_to_unit_range(np.array(img))
        return torch.from_numpy(arr).unsqueeze(0).to(device=device)  # [1, H, W]
    else:
        video_tensor, _, _ = tvio.read_video(path, pts_unit='sec', output_format='TCHW')
        if video_tensor.shape[0] == 0 or video_tensor.shape[-2] <= 1 or video_tensor.shape[-1] <= 1:
            raise ValueError(
                f"decoded an empty/degenerate video ({tuple(video_tensor.shape)}) from {path} -- "
                "likely an unsupported or non-video file format."
            )
        # Slice to a single channel (grayscale video stored 3x) BEFORE moving
        # to GPU/float32, not after -- moving the full 3-channel tensor first
        # allocates ~3x more GPU memory than the final result needs, and a
        # separate (out-of-place) /255.0 division would briefly hold a second
        # full-size copy on top of that. video_tensor[:, 0, :, :] is a
        # zero-copy CPU view, so only one GPU allocation (the final
        # single-channel float32 tensor) ever happens.
        video_channel0 = video_tensor[:, 0, :, :]
        video_gpu = video_channel0.to(device=device, dtype=torch.float32)
        video_gpu.div_(255.0)
        return video_gpu
