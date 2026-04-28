# Kolmogorov-Arnold Networks for Water and Vegetation Mapping using Sentinel-2

Kolmogorov-Arnold Networks (KAN) for pixel-based water and vegetation classification using Sentinel-2 imagery, with comparison to Multi-Layer Perceptron (MLP) models across major global rivers.

---

## Associated Publication

This repository accompanies the manuscript:

**"Kolmogorov-Arnold Networks for High Accuracy Mapping of Water and Vegetation: A Study of Global Main Rivers"**

The study investigates the performance of Kolmogorov-Arnold Networks (KANs) for pixel-level classification of water and vegetation using Sentinel-2 multispectral imagery across nine major global rivers.

---

## Study Areas

The dataset includes nine major global rivers:

- Amazon  
- Ganges  
- Karun  
- Mississippi  
- Nile  
- Rhine  
- Seine  
- Shatt-al-Arab  
- Yangtze  

---

## Study Overview

![Study Areas](images/Fig1.png)

*Figure 1: Research Flowchart*

## Repository Structure

kan-sentinel2-water-vegetation-classification/
├── data/
├── src/
├── notebooks/
├── scripts/
├── checkpoints/
├── results/
└── images/

---

## Dataset

The dataset is publicly available at Zenodo:

https://doi.org/10.5281/zenodo.19817363

Includes:

- Sentinel-2 GeoTIFF imagery  
- Labeled masks for supervised learning  
- Data from nine global river systems  

See `data/README.md` for details.

---

## Installation

Clone the repository and install dependencies;  run_kan_mlp_river.ipynb notebook has been composed for reproducing the results. 

```bash
git clone https://github.com/amirsalehmousavi/kan-water-vegetation-mapping.git
cd kan-water-vegetation-mapping
pip install -r requirements.txt

