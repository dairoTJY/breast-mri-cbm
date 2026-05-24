# MICCAI 2026 Code Release

This repository contains the code used for the MICCAI 2026 paper submission.

The current release includes:
- `code/train_resnet.py`: 3D ResNet training for the imaging backbone
- `code/train_cbm.py`: concept bottleneck model training and evaluation
- `code/ispy2_dataset_finetune.py`: shared ISPY2 dataset loader

This repository does not include:
- pretrained model weights
- raw imaging data
- patient-level private files

## Project Structure

```text
miccai2026/
├─ code/
│  ├─ ispy2_dataset_finetune.py
│  ├─ train_resnet.py
│  └─ train_cbm.py
├─ DATASET.md
├─ LICENSE
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## Environment

Python `3.10` is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

The experiments use an ISPY2-based JSON file and a split JSON file.

- `ISPY2_json_minimal.json`: dataset metadata
- `split_seed20.json`: train/val/test split

Please see [DATASET.md](./DATASET.md) for the expected format and data access notes.

## Training the ResNet Backbone

The ResNet script reads the dataset and split paths from environment variables by default:

```bash
export ISPY2_JSON_PATH=./ISPY2_json_minimal.json
export ISPY2_SPLIT_JSON_PATH=./split_seed20.json
python code/train_resnet.py 20
```

Outputs are written under `./log_resnet_onefold` by default. You can override that with:

```bash
export RESNET_LOG_DIR=./log_resnet_onefold
```

## Training the CBM

The CBM script requires:
- the dataset JSON
- the split JSON
- a trained ResNet checkpoint
- concept embeddings in `.pt` format

Example:

```bash
python code/train_cbm.py \
  --mode train \
  --json_path ./ISPY2_json_minimal.json \
  --split_json ./split_seed20.json \
  --resnet_ckpt ./resnet_best.pt \
  --concept_pt ./concept_embeddings.pt \
  --resnet_depth 50 \
  --img_size_dce 160 160 160 \
  --batch_size 8 \
  --epochs 30 \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --out_root ./log_icb \
  --run_name release_run
```

## CBM Evaluation

Example:

```bash
python code/train_cbm.py \
  --mode eval \
  --json_path ./ISPY2_json_minimal.json \
  --split_json ./split_seed20.json \
  --resnet_ckpt ./resnet_best.pt \
  --concept_pt ./concept_embeddings.pt \
  --eval_ckpt ./log_icb/release_run/best.pt \
  --eval_split test \
  --out_dir ./eval_out
```

## Weights and Reproducibility

Pretrained weights are not released in this repository.

To reproduce results, users should:
- obtain access to the relevant dataset
- prepare the JSON metadata and split files
- train the ResNet backbone
- prepare or generate concept embeddings
- train and evaluate the CBM using the provided scripts

## License

This code is released under the MIT License. See [LICENSE](./LICENSE).

## Citation

If you use this repository, please cite the corresponding paper once bibliographic information is finalized.
