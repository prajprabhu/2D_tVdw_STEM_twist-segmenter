# 2D tVdW STEM twist segmenter

Interactive tools for mapping the **local twist angle** and **structural phase** of twisted van der Waals bilayers (WSSe, MoS₂, graphene, …) from HAADF-STEM images and video.

Each frame is cut into overlapping tiles, and each tile is analysed as follows:

1. Take the tile's 2D FFT.
2. Find the first-order Bragg ring.
3. Sample the angular intensity around that ring.
4. Take its circular autocorrelation, the "correlogram".

The correlogram gives that tile's twist angle. Across all tiles, the correlograms are the feature vectors for PCA/NMF and K-means, which segment the image into structural phases at pixel resolution.

There are two browser apps and three example notebooks:

| | What it does |
|---|---|
| **Tile Inspector** (`tile_inspector_app.py`) | Drag a box onto any tile of a frame and see each stage for that tile live: raw tile, Hann-windowed FFT, radial profile and detected Bragg ring, angular profile, and correlogram. Tile size, stride, material/d-spacing, pixel size, mean subtraction and CLAHE are all adjustable. |
| **Cluster Map Inspector** (`cluster_map_inspector_app.py`) | Runs the full pipeline on a whole image or video: tiling → batched correlograms → PCA or NMF → K-means. It then overlays a pixel-resolution cluster map on the frame. You can choose which PCs feed the clustering, change the number of clusters live, view the PC spectra, scrub or play through frames, and record the overlay to `.mp4`. |
| `examples/twist_mapper_GPU_demo.ipynb` | Step-by-step, single-tile walk-through of the pipeline. |
| `examples/correlogram_pca_demo.ipynb` | Correlograms for every tile, followed by PCA and loading plots. |
| `examples/pixel_phase_map.ipynb` | Pixel-resolution K-means phase maps, PC score maps, and area fraction vs. frame. |

Accepted inputs:
- **Videos:** e.g. `.mp4`
- **Images:** `.png`, `.jpg`, `.tif`, …
- **`.npz` archives:** an image or a stack of frames. Calibration in the archive is read automatically; see [examples/data/README.md](examples/data/README.md).

## Requirements

- Python 3.10+. The environment files use 3.12.
- **Operating system:** Windows, Linux, WSL2 or macOS.
- **NVIDIA GPU:** optional. PyTorch uses CUDA when it's available; otherwise everything runs on the CPU, more slowly on large videos.
- **RAPIDS (cuML + CuPy):** optional, Linux/WSL2 + NVIDIA only. The Cluster Map Inspector uses it for GPU PCA and K-means. Without it, the app falls back to scikit-learn on the CPU automatically and gives equivalent results.

## Installation

You need [Miniforge](https://conda-forge.org/download/) or Anaconda.

```bash
git clone https://github.com/prajprabhu/2D_tVdw_STEM_twist-segmenter.git
cd 2D_tVdw_STEM_twist-segmenter

conda env create -f environment.yml        # Windows / Linux / WSL / macOS
#   or, on Linux/WSL2 with an NVIDIA GPU, for GPU-accelerated clustering:
# conda env create -f environment-rapids.yml

conda activate twist-segmenter
pip install --no-deps -e .                 # makes the twist_mapper_GPU package importable
python examples/download_sample_data.py    # fetches the sample image (0.5 MB)
```

On Windows, run these commands in the **Miniforge Prompt** (or Anaconda Prompt).

<details>
<summary>Without conda (pip only)</summary>

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e ".[notebooks]"              # add ,rapids on Linux for GPU clustering
python examples/download_sample_data.py
```

pip installs the CPU build of PyTorch by default on Windows and Linux. For a CUDA build, first install PyTorch using the command from <https://pytorch.org/get-started/locally/>.
</details>

## Running the apps

From the repository folder, with the environment active:

| App | Linux / WSL / macOS | Windows | Opens at |
|---|---|---|---|
| Tile Inspector | `./run_tile_inspector_app.sh` | `run_tile_inspector_app.bat` | <http://localhost:5009/tile_inspector_app> |
| Cluster Map Inspector | `./run_cluster_map_inspector_app.sh` | `run_cluster_map_inspector_app.bat` | <http://localhost:5006/cluster_map_inspector_app> |

The launchers run `panel serve <app>.py --port <port> --show`, which opens your browser. Stop an app with **Ctrl+C** in the terminal. The two apps use different ports, so both can run at once.

Neither app loads anything at startup:

1. The **Source path** box is pre-filled with the sample `examples/data/moire.npz`. Click **Load** (Tile Inspector) or **Load & Recompute** (Cluster Map Inspector) to open it, or enter the path to your own file first.
2. Choose the **Material**. This sets the expected Bragg d-spacing, which you can override in the **d-spacing** box. Check the **Pixel size**: it is read from `.npz` metadata when present, and otherwise you enter it.
3. Set the **Tile size** and **Stride** for your magnification. The expected first-order ring radius in FFT pixels is `tile_size × pixel_size / d_spacing`. It should sit well away from the FFT center, ideally at least 10–15 px, so increase the tile size for high-magnification images.

The Cluster Map Inspector's status line shows whether clustering is running on **RAPIDS (GPU)** or **scikit-learn (CPU)**.

## Example notebooks

```bash
cd examples
jupyter lab
```

Each notebook runs on `data/moire.npz` by default. To analyse your own data, change `SOURCE_PATH` in the setup cell. The notebooks use the same RAPIDS-or-scikit-learn fallback as the apps.

## Using the package directly

```python
from twist_mapper_GPU import TiledVideo, TileFFTAnalysis, load_video_tensor, get_material, default_d_spacing_ang

video = load_video_tensor("examples/data/moire.npz", device="cuda")   # or "cpu"
tiled = TiledVideo(video, tile_size=128, stride=32)
analysis = tiled.analyze(frame_idx=0, row_idx=2, col_idx=4)

r0 = TileFFTAnalysis.expected_ring_radius(128, pixel_size_ang=0.106,
                                          d_spacing_ang=default_d_spacing_ang(get_material("WSSe")))
analysis.compute_ring_radius(r0)
analysis.compute_angular_profile()
print(analysis.compute_twist(n_folds=6))
```

For whole-video clustering, see `twist_mapper_GPU.dynamic_pipeline.run_dynamic_pipeline`.

## Troubleshooting

- **"file not found" for the sample:** run `python examples/download_sample_data.py`.
- **`ModuleNotFoundError: twist_mapper_GPU`:** run `pip install --no-deps -e .` from the repository folder, with the environment active.
- **Port already in use:** another copy of the app is still running. Stop it, or launch on another port, e.g. `panel serve tile_inspector_app.py --port 5010 --show`.
- **GPU not used:** check with `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, check your NVIDIA driver; conda-forge installs a CUDA build of PyTorch only when it detects one.
- **Cluster labels change between runs or backends:** K-means with a few hundred tiles, e.g. a single 512 × 512 image, is sensitive to initialization. Cluster numbering and borderline tiles can differ between seeds, or between RAPIDS and scikit-learn. Larger inputs, or fewer clusters, give more stable maps.

## Sample data and citation

The sample image comes from the Gomb-Net project and is downloaded from its repository, not redistributed here. If you use it, please cite:

> A. C. Houston, S. B. Harris, H. Wang, Y.-C. Lin, D. B. Geohegan, K. Xiao, G. Duscher, "Atom Identification in Bilayer Moiré Materials with Gomb-Net," *Nano Letters* **25**(23), 9277–9284 (2025). <https://doi.org/10.1021/acs.nanolett.5c01460>

See [examples/data/README.md](examples/data/README.md) for details.

## License

[MIT](LICENSE) © 2026 Prajakta Prabhune. The sample data is not covered by this license; it belongs to its authors.
