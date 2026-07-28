# CEGA DOTA Experiment

This document describes the CEGA DOTA v1.0 test and training experiments.

## 1. Environment

Use the PETDet environment described in the PETDet README. The released
checkpoint was produced with:

```text
python == 3.10.16
torch == 1.13.1
cuda == 11.7
mmcv == 1.7.1
mmdet == 2.28.2
mmrotate == 0.3.2+
```

## 2. Dataset Preparation

Download: https://captain-whu.github.io/DOTA/dataset.html

The downloaded train/val sets contain large images and `labelTxt` annotations.
The downloaded test set contains images only.

`DOTA_ROOT` is only an environment variable. It tells the scripts where your
split DOTA folder is.

After downloading:

1. Unzip train, val, and test files.
2. Put the original files under `data/DOTA/`.

```text
data/DOTA/
  train/images/
  train/labelTxt/
  val/images/
  val/labelTxt/
  test/images/
```

3. Split DOTA into 1024 x 1024 patches.

```bash
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ss_trainval.json
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ss_test.json
```

4. Save the generated split folder path in the `DOTA_ROOT` variable. Replace the
   example path below if your split folder is somewhere else.

```text
DOTA_ROOT/
  trainval/
    annfiles/
    images/
  test/
    images/
```

```bash
export DOTA_ROOT=data/split_ss_dota
```

## 3. Test Experiment

The DOTA test experiment loads:

```text
work_dirs/CEGA_DOTA/CEGA_DOTA.pth
```

Download the checkpoint from
[Google Drive](https://drive.google.com/file/d/1h6GoFyKSDvI_xxGmi31IarjgUn04mp3O/view?usp=drive_link)
and place it at `work_dirs/CEGA_DOTA/CEGA_DOTA.pth`.

The test config in this repository is:

```text
configs/strip_rcnn/CEGA_DOTA_config.py
```

Run validation mAP from the PETDet root:

```bash
mkdir -p work_dirs/CEGA_DOTA
export DOTA_ROOT=/path/to/split_1024_dota1_0
export CHECKPOINT="$PWD/work_dirs/CEGA_DOTA/CEGA_DOTA.pth"
bash tools/CEGA_DOTA_test.sh val
```

Generate DOTA Task1 submission files:

```bash
export DOTA_ROOT=/path/to/split_1024_dota1_0
export CHECKPOINT="$PWD/work_dirs/CEGA_DOTA/CEGA_DOTA.pth"
bash tools/CEGA_DOTA_test.sh submit
```

The output is written to:

```text
work_dirs/CEGA_DOTA_test/
```

The checkpoint metadata records:

| Item | Value |
|---|---|
| Model | `StripRCNN` |
| Backbone | `StripNet-S` |
| Neck | `FPN` |
| RPN | `OrientedRPNHead` |
| RoI head | `StripHead` |
| Classes | DOTA v1.0 15 classes |
| Angle version | `le90` |

## 4. Training Experiment

The training experiment uses the original PETDet config file:

```text
PETDet/experiments/ablation/serial_rot_scale_aclrpn_striphead_dota.py
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
| Epochs | `12` |
| Seed | `332845056` |

Run from the PETDet root:

```bash
python tools/train.py experiments/ablation/serial_rot_scale_aclrpn_striphead_dota.py --seed 332845056
```

If your dataset path is different from the path in the config, update the config
or override the data path with `--cfg-options`.

The PETDet config writes training output to:

```text
work_dirs/cega_parallel_aclrpn_striphead_dota/
```

Do not upload DOTA images or annotations to this repository.
