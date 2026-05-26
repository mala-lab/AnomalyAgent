# AnomalyAgent

This repository is a reference implementation of AnomalyAgent. After the paper is accepted, a cleaner and more complete version will be officially released.

## Setup

Use Python 3.11 or 3.12 for the local PyTorch/Qwen runtime.

Install the package in editable mode:

```bash
pip install -e .
```

Local model and dataset paths can be configured by CLI flags or environment variables:

```bash
export MVTEC_DATA_PATH=/path/to/mvtec
export MVTEC_LOCO_DATA_PATH=/path/to/mvtec_loco
export HEADCT_DATA_PATH=/path/to/headct/images
export HEADCT_LABELS_CSV=/path/to/headct/labels.csv
export LAG_DATA_PATH=/path/to/lag/test
export KAPUTT_PARQUET_PATH=/path/to/kaputt/query-test.parquet
export KAPUTT_CROP_ROOT=/path/to/kaputt/query-crop/data/test/query-data/crop
export KAPUTT_QUERY_IMAGE_ROOT=/path/to/kaputt/query-image/data/test/query-data
export QWEN_VL_MODEL_PATH=/path/to/qwen3_vl_model
export QWEN_TEXT_MODEL_PATH=/path/to/qwen3_text_model
```

Optional super-resolution weights can be configured when `realesrgan` is installed:

```bash
export REALESRGAN_MODEL_PATH=/path/to/RealESRGAN_x4plus.pth
export REALESRNET_MODEL_PATH=/path/to/RealESRNet_x4plus.pth
```

## Run

```bash
bash run_all.sh
```

`run_all.sh` integrates the commands for MVTec, MVTec LOCO, HeadCT, LAG, and Kaputt. Configure paths through the environment variables above or edit the script directly.
