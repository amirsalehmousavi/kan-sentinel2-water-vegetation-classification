# Kolmogorov-Arnold Networks for Water and Vegetation Mapping using Sentinel-2

---

![Study Areas](images/Fig1.png)

## Overview

This repository is associated with the paper:

**“Kolmogorov-Arnold Networks for High Accuracy Mapping of Water and Vegetation Using Sentinel-2 Imagery.”**

It provides all code required to reproduce the results.

---

## Reproducibility

1. Clone the repository:
```bash
git clone https://github.com/amirsalehmousavi/kan-sentinel2-water-vegetation-classification.git
cd kan-sentinel2-water-vegetation-classification
```

2. Install dependencies: 

```
pip install -r requirements.txt
```

3. Download the dataset from Zenodo and extract it.

Instructions for downloading and organizing the data are provided in:

```
data/README.md
```

4. Place the data into the following directories:

```
data/train_validation_data/
data/test_data/
data/original/
```

5. Run the experiment:

```
python scripts/run_experiment.py
```

Alternatively, use the `run_kan_mlp_river.ipynb` notebook for step-by-step execution.

