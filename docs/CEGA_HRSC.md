# CEGA HRSC Experiment

This document describes the CEGA HRSC2016 test and training experiments.

## 1. Environment

Use the PETDet environment described in the PETDet README. The released
checkpoint was produced with:

```text
python == 3.10
torch == 1.13.1
cuda == 11.7
mmcv == 1.7.1
mmdet == 2.28.2
mmrotate == 0.3.2
```

Set `HRSC_ROOT` to the HRSC2016 dataset root when running commands.

## 2. Dataset Preparation

Download: https://www.kaggle.com/datasets/guofeng/hrsc2016

The downloaded package contains images, XML annotations, and split files.

`HRSC_ROOT` is only an environment variable. It tells the scripts where your
HRSC2016 folder is.

After downloading:

1. Unzip the package.
2. Find the parent folder of `ImageSets/` and `FullDataSet/`.
   For example, if the unzipped dataset looks like this:

```text
/data/HRSC2016/
  ImageSets/
  FullDataSet/
```

then the parent folder is `/data/HRSC2016`.

3. Save that parent folder path in `HRSC_ROOT`.

```bash
export HRSC_ROOT=/data/HRSC2016
```

No image splitting is needed for HRSC2016 in this project.

## 3. Test Experiment

The test experiment loads `work_dirs/CEGA_HRSC.pth` with
`experiments/ablation/CEGA_HRSC_config.py`.

Download the checkpoint from
[Google Drive](https://drive.google.com/file/d/1s5pOkjIgufcuBbfDuQkIjXjBCfc8zwxg/view?usp=drive_link)
and place it at `work_dirs/CEGA_HRSC.pth`.

```bash
mkdir -p work_dirs
export HRSC_ROOT=/path/to/hrsc
export CHECKPOINT="$PWD/work_dirs/CEGA_HRSC.pth"
bash tools/CEGA_HRSC_test.sh
```

The checkpoint metadata records:

| Item | Value |
|---|---|
| Model | `ParallelBranchPETDet` |
| Fusion | `CEGAParallelBranchFusion` |
| RPN | `ACLRPNHead` |
| RoI head | `StripHead` |
| Classes | `ship` |
| Angle version | `le90` |

## 4. Training Experiment

The training experiment uses the original PETDet config file:

```text
PETDet/experiments/ablation/serial_rot_scale_aclrpn_striphead_hrsc.py
```

Main training settings:

| Setting | Value |
|---|---|
| Backbone branches | `ReResNet-50` and `ScaleReResNet-18` |
| Fusion | `CEGAParallelBranchFusion` |
| RPN | `ACLRPNHead` |
| Head | `StripHead` |
| Optimizer | `AdamW` |
| Learning rate | `0.0005` |
| Epochs | `72` |
| Seed | `3407` |

Run from the PETDet root:

```bash
python tools/train.py experiments/ablation/serial_rot_scale_aclrpn_striphead_hrsc.py --seed 3407
```

If your dataset path is different from the path in the config, update the config
or override the data path with `--cfg-options`.

The PETDet config writes training output to:

```text
work_dirs/cega_parallel_aclrpn_striphead_hrsc/
```

Do not upload HRSC2016 images or annotations to this repository.
