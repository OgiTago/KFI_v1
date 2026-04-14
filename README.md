## Kalman-filter Force inference
===

## Description
This repository provides the implementation of Kalman-filter Force Inference (KFI), 
a Bayesian framework to estimate cellular forces (junctional tensions and cellular pressures) from time-lapse microscopy images of epithelial tissues. Unlike static methods, KFI explicitly accounts for tissue dynamics and temporal continuity of forces.
Details of the method and its validation are described in Ogita et al. (2026) [1].

The current version does not support topological changes, such as cell rearrangement or cell division.

## Usage
1. Prepare input files
<br>The inference script requires vertex positions and connectivity data for the entire time series.
<br>-Data Format: Prepare files containing vertex coordinates and connectivity data for each frame. The format must be identical to the output of [GetVertex](https://github.com/Sugimuralab/GetVertexPlugin). Refer to the sample_data/ directory in this repository for a template.
<br>-File Organization : Place all files for a single time-series sequence into one directory. Include a time index in the filenames (e.g., frame_001.csv, frame_002.csv). The script sorts files alphabetically to process them in chronological order.
2. Change the variable "file_directory" in KFI_v1.py to the input directory in step 1.
3. Run KFI_v1.py on IDE or IPython.


## Reference

1. Kalman-filter Force Inference: an estimation framework for cellular forces from temporal evolution of epithelial morphogenesis
Goshi Ogita, Takemasa Miyoshi, Tatsuo Shibata
bioRxiv 2025.12.19.695339; doi: https://doi.org/10.64898/2025.12.19.695339
