# Sample data

The apps and example notebooks use `moire.npz`, a 512 × 512 HAADF-STEM image of a bilayer moiré material.

This file is **not included in this repository**. It belongs to the authors of Gomb-Net and is hosted in their repository. Download it with:

```bash
python examples/download_sample_data.py
```

The script saves the file to `examples/data/moire.npz`, where both apps and the notebooks expect it.

## Source and citation

- **Repository:** <https://github.com/AustinHouston/Gomb-Net> (file `Moire.npz`)
- **Paper:** A. C. Houston, S. B. Harris, H. Wang, Y.-C. Lin, D. B. Geohegan, K. Xiao, G. Duscher, "Atom Identification in Bilayer Moiré Materials with Gomb-Net," *Nano Letters* **25**(23), 9277–9284 (2025). <https://doi.org/10.1021/acs.nanolett.5c01460>
- **Full experimental dataset:** the Gomb-Net README links the complete dataset on Google Drive: <https://drive.google.com/file/d/1DyKtrmJ8wNYQg3YEJ8_iXjz6lB_DQfwy/view?usp=sharing>

If you use this data, please cite the paper above.

## File format

| Key | Contents |
|---|---|
| `im_array` | `uint16` image, shape `(512, 512)` |
| `pixel_size` | pixel size in **meters** (`1.06e-11` m = 0.106 Å) |

The loader in `twist_mapper_GPU.source_loading` reads `im_array` and converts `pixel_size` from meters to Å, so the "Pixel size" box fills in automatically.

## Using your own data

Both apps accept any of these formats in the **Source path** box:

- **Videos:** e.g. `.mp4`
- **Images:** e.g. `.png`, `.jpg`, `.tif`
- **`.npz` archives:** the image or stack goes under `im_array`, `video`, `images`, `data` or `array`. An optional `pixel_size` gives the calibration, in Å or in meters.
