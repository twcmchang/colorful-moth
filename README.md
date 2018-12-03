# Moth Segmentation

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A

---

### Usage

```

# train a new model
python3 Sup_train.py --XX_DIR=/path/to/image/ --YY_DIR=/path/to/groundtruth/ --SAVEDIR=/path/to/save/ --num_class= number of output class --gpu='gpu_id'

# prediction
## for background removal
python3 Sup_predict_rmbg.py --XX_DIR=/path/to/image --model_dir=/path/to/checkpoint/ --gpu='gpu_id'
## for 5 comps
python3 Sup_predict_5comps.py --XX_DIR=/path/to/image --model_dir=/path/to/checkpoint/ --gpu='gpu_id'

```

Postprocess.ipynb postprcesses background-removal model results. (find_contour and condition random field)
5comps_output_process.ipynb processes 5-comps model resultsto generate final images.
visualize.ipynb visulizes some samples of different background-removal steps and 5-comps results.


### Acknowledgements 
This repository reuses code from pytorch-unsupervised-segmentation by kanezaki and a U-Net shaped 10-layer CNN by RaoulMa. Many thanks to all the contributions!

### Environment

The test environment is
	- Python 2.7.12
	- tensorflow-gpu 1.4.1	
	- torch 0.2.0

