## 1 PETDET 

This warehouse provides testing and training instructions for the project. At present, the project provides models and testing steps. The complete training experiment file is not yet included, and the training file will be added later.

Test: Load the trained models, evaluate CEGA_HRSC.pth on the HRSC2016 dataset, and evaluate CEGA_DOTA.pth on the DOTA v1.0 dataset. The accuracy mAP on HRSC2016 is 90.80%, and the accuracy on DOTA v1.0 is 87.85%.

Training: Trained on the HRSC2016 dataset and the DOTA v1.0 dataset.

## 1.1 Configure The Environment

Use a PETDet environment with the same core dependencies:

```text
python == 3.10
torch == 1.13.1
cuda == 11.7
mmcv == 1.7.1
mmdet == 2.28.2
mmrotate == 0.3.2
```

## 1.2 Dataset Preparation

Download links:

| Dataset | Link |
|---|---|
| HRSC2016 | https://www.kaggle.com/datasets/guofeng/hrsc2016 | 
| DOTA v1.0 | https://captain-whu.github.io/DOTA/dataset.html |

HRSC_ROOT and DOTA_ROOT are only environment variables. They tell the scripts where your dataset folders are.

## 1.2.1

（1）After downloading HRSC2016:

 First,unzip the package.
 Second,find the parent folder of ImageSets/  and FullDataSet/.
   For example, if the unzipped dataset looks like this:

```text
/data/HRSC2016/ImageSets
/FullDataSet/
```

then the parent folder is /data/HRSC2016.

Final,save that parent folder path in HRSC_ROOT.

```bash
export HRSC_ROOT=/data/HRSC2016
```

（2）After downloading DOTA v1.0:

1. Unzip train, val, and test files.
2. Put the original files under data/DOTA/.

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

```bash
export DOTA_ROOT=data/split_ss_dota
```

## 1.3 Test Experiments

| Dataset | Checkpoint | Download | Test script | Config | mAP |
|---|---|---|---|---|---|
| HRSC2016 | `work_dirs/CEGA_HRSC.pth` | [Google Drive](https://drive.google.com/file/d/1s5pOkjIgufcuBbfDuQkIjXjBCfc8zwxg/view?usp=drive_link) | `tools/CEGA_HRSC_test.sh` | `experiments/ablation/CEGA_HRSC_config.py` | `90.80%` |
| DOTA v1.0 | `work_dirs/CEGA_DOTA/CEGA_DOTA.pth` | [Google Drive](https://drive.google.com/file/d/1h6GoFyKSDvI_xxGmi31IarjgUn04mp3O/view?usp=drive_link) | `tools/CEGA_DOTA_test.sh` | `configs/strip_rcnn/CEGA_DOTA_config.py` | `87.85%` |

Download the checkpoint and dataset, then put them in the corresponding paths.

### 1.3.1 HRSC2016 Test

Set `HRSC_ROOT` to the HRSC2016 dataset root, then run:

```bash
mkdir -p work_dirs
export HRSC_ROOT=/path/to/hrsc
export CHECKPOINT="$PWD/work_dirs/CEGA_HRSC.pth"
bash tools/CEGA_HRSC_test.sh
```

### 1.3.2 DOTA v1.0 Test

Use the split DOTA directory as `DOTA_ROOT`:

```text
DOTA_ROOT/
  trainval/
    annfiles/
    images/
  test/
    images/
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

## 1.4 Training Experiments

This project uses the following two datasets.

| Dataset | PETDet training config | Main setting |
|---|---|---|
| HRSC2016 | `PETDet/experiments/ablation/serial_rot_scale_aclrpn_striphead_hrsc.py` | CEGA parallel branch, ACL-RPN, StripHead, 72 epochs |
| DOTA v1.0 | `PETDet/experiments/ablation/serial_rot_scale_aclrpn_striphead_dota.py` | CEGA parallel branch, ACL-RPN, StripHead, 12 epochs |

HRSC2016: https://www.kaggle.com/datasets/guofeng/hrsc2016

DOTA v1.0: https://captain-whu.github.io/DOTA/dataset.html

### 1.4.1 HRSC2016 Training

Run from the PETDet root:

```bash
python tools/train.py experiments/ablation/serial_rot_scale_aclrpn_striphead_hrsc.py --seed 3407
```

If your dataset path is different from the path in the config, update the config
or override the data path with `--cfg-options`.

### 1.4.2 DOTA v1.0 Training

Run from the PETDet root:

```bash
python tools/train.py experiments/ablation/serial_rot_scale_aclrpn_striphead_dota.py --seed 332845056
```

If your dataset path is different from the path in the config, update the config
or override the data path with `--cfg-options`.
