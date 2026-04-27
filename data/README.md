## Overview

The dataset used in this study is publicly available at Zenodo:

https://doi.org/10.5281/zenodo.19817363

## Dataset Contents

The dataset includes Sentinel-2 imagery and labeled samples for pixel-based classification of water and vegetation across nine major rivers.

The following files contain the raw Sentinel-2 GeoTIFF imagery:

- r1_amazon.zip 
- r2_ganges.zip 
- r3_karoun.zip 
- r4_mississippi.zip 
- r5_nile.zip 
- r6_rhine.zip 
- r7_seine.zip 
- r8_shatt-al-arab.zip 
- r9_yangtze.zip 

These files can be extracted and used directly in GIS software.

The following files contain the labeled datasets used for supervised learning:

- test_data.zip 
- train_validation_data.zip 

These include manually masked GeoTIFF files used for training and evaluation.

## Usage Instructions

To run the models in this repository:

1. Download the dataset from Zenodo.
2. Extract the files `test_data.zip` and `train_validation_data.zip`.
3. Place the extracted folders inside the `data/` directory.

The final directory structure should be:

data/
├── test_data/
│   ├── c1_r1.tif
│   ├── c2_r2.tif
│   └── ...
│
└── train_validation_data/
    ├── c1_r1.tif
    ├── c2_r2.tif
    └── ...

## Notes

- The dataset is provided in compressed format due to file size considerations.
- High compression is achieved because many raster regions contain masked (zero-valued) pixels.
- After extraction, the original full-resolution GeoTIFF files are restored.

## License

The dataset is distributed under the Creative Commons Attribution 4.0 (CC-BY 4.0) license via Zenodo.

## Citation

If you use this dataset, please cite:

https://doi.org/10.5281/zenodo.19817363

