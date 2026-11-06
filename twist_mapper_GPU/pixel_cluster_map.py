"""
pixel_cluster_map.py — Hann-weighted per-pixel blending of per-tile PCA
scores, and nearest-centroid classification, for pixel-resolution cluster
maps derived from a tile-resolution K-means fit.

Motivation: a tile's correlogram/PCA score is computed with a Hann window
(TileFFTAnalysis._compute_fft_magnitude), so pixels near a tile's edge
barely influenced that tile's score while pixels near its center dominate
it. Any given pixel is covered by multiple overlapping tiles at once
(tile_size > stride), so assigning it a single phase means combining
those tiles' scores somehow — the natural choice is to weight each
covering tile's contribution by the *same* Hann value that pixel already
carried in that tile's own signature, rather than an unrelated kernel.

Because PCA is a linear (affine) map, Hann-weighted-averaging tiles'
already-computed PCA scores and then classifying the result is
mathematically identical to Hann-weighting the raw correlograms first and
then projecting — but far cheaper (10 numbers per tile instead of 360),
so that's what `blend_pc_field` does directly on cached PCA scores.

The K-means fit itself still happens once, at tile resolution (see
correlogram_pca_full.ipynb) — these functions only handle blending scores
onto pixel space and assigning each pixel to its nearest fitted centroid
(a predict step, not a fresh fit), so pixel-resolution and tile-resolution
maps share the exact same cluster identities.
"""

import torch


def build_symmetric_hann2d(tile_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """2D symmetric (periodic=False) Hann kernel, tile_size x tile_size.

    Deliberately NOT the periodic=True window TileFFTAnalysis uses for its
    own FFT windowing — periodic=True is intentionally asymmetric (correct
    for feeding an FFT), but here the window is being reused as a purely
    spatial confidence kernel, where exact symmetry about the tile center
    (zero at both edges) is what's wanted.
    """
    h = torch.hann_window(tile_size, periodic=False, device=device, dtype=dtype)
    return torch.outer(h, h)


def blend_pc_field(tile_pc_scores: torch.Tensor, stride: int, H: int, W: int,
                   hann2d: torch.Tensor) -> tuple:
    """Hann-weighted overlap-add blend of one frame's per-tile PC scores
    onto pixel space.

    Parameters
    ----------
    tile_pc_scores : torch.Tensor (n_rows, n_cols, n_dims)
        One frame's tile-grid worth of per-tile PCA scores.
    stride : int
        Tile stride in pixels (tile_size is inferred from hann2d.shape).
    H, W : int
        Frame height/width in pixels.
    hann2d : torch.Tensor (tile_size, tile_size)
        From build_symmetric_hann2d(tile_size, ...).

    Returns
    -------
    pc_field : torch.Tensor (H, W, n_dims) -- blended PC score per pixel
        (meaningless where `covered` is False -- see below).
    covered : torch.Tensor (H, W), bool -- True where the summed Hann
        weight is non-negligible. False near the frame border for two
        distinct reasons: (1) the tile grid may not physically reach every
        pixel (e.g. if (n_rows-1)*stride + tile_size < H); (2) even where
        it does, a pixel at the very top-left corner is reachable by only
        one tile (no adjacent tile exists past the boundary to pick up the
        slack), and that single tile's own Hann taper goes to exactly 0
        right at the corner -- the same edge effect windowed overlap-add
        reconstruction has at signal boundaries. Either way these pixels
        got no meaningful vote and should be treated as masked/transparent,
        not classified.
    """
    n_rows, n_cols, n_dims = tile_pc_scores.shape
    tile_size = hann2d.shape[0]
    device = tile_pc_scores.device
    dtype = tile_pc_scores.dtype

    pc_sum = torch.zeros(H, W, n_dims, device=device, dtype=dtype)
    weight_sum = torch.zeros(H, W, device=device, dtype=dtype)
    hann2d = hann2d.to(device=device, dtype=dtype)

    for r in range(n_rows):
        y0 = r * stride
        for c in range(n_cols):
            x0 = c * stride
            pc_vec = tile_pc_scores[r, c]  # (n_dims,)
            pc_sum[y0:y0 + tile_size, x0:x0 + tile_size, :] += hann2d.unsqueeze(-1) * pc_vec
            weight_sum[y0:y0 + tile_size, x0:x0 + tile_size] += hann2d

    covered = weight_sum > 1e-6
    pc_field = pc_sum / weight_sum.clamp(min=1e-6).unsqueeze(-1)
    return pc_field, covered


def classify_nearest_centroid(pc_field: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Assigns each pixel's blended PC vector to its nearest centroid
    (Euclidean, matching K-means' own assignment rule).

    Parameters
    ----------
    pc_field : torch.Tensor (H, W, n_dims)
    centroids : torch.Tensor (n_clusters, n_dims), same device/dtype family

    Returns
    -------
    torch.Tensor (H, W), dtype long -- cluster index per pixel.
    """
    H, W, n_dims = pc_field.shape
    flat = pc_field.reshape(-1, n_dims)
    dists = torch.cdist(flat, centroids.to(device=flat.device, dtype=flat.dtype))
    labels = dists.argmin(dim=1)
    return labels.reshape(H, W)


def blend_and_classify_frame(tile_pc_scores: torch.Tensor, centroids: torch.Tensor,
                             stride: int, H: int, W: int, hann2d: torch.Tensor) -> tuple:
    """Convenience wrapper: blend_pc_field() + classify_nearest_centroid()
    for one frame.

    Returns
    -------
    labels : torch.Tensor (H, W), dtype long -- cluster index per pixel
        (meaningless where `covered` is False).
    covered : torch.Tensor (H, W), bool
    """
    pc_field, covered = blend_pc_field(tile_pc_scores, stride, H, W, hann2d)
    labels = classify_nearest_centroid(pc_field, centroids)
    return labels, covered
