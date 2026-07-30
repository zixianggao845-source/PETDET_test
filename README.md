#  PETDET 

This repository provides testing and training documentation for the project. Currently, it includes the model and test programs; the complete training experiment files have not yet been incorporated and will be uploaded together with the training code during the paper revision stage.

# 1 Test

Test: Load the trained models, evaluate CEGA_HRSC.pth on the HRSC2016 dataset, and evaluate CEGA_DOTA.pth on the DOTA v1.0 dataset. The accuracy mAP on HRSC2016 is 74.58%, and the accuracy on DOTA v1.0 is 87.85%.

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

## 1.2 Dataset and Code Preparation

To obtain the testing code, navigate to the repository homepage, click the Code button, and select`Download ZIP`. After downloading, extract the archive and install the required interpreter packages as described above. Then, test the model on the HRSC2016 and DOTA v1.0 datasets.HRSC_ROOT and DOTA_ROOT are only environment variables. 

Datasets download links:

| Dataset | Link |
|---|---|
| HRSC2016 | https://www.kaggle.com/datasets/guofeng/hrsc2016 | 
| DOTA v1.0 (Split) | https://pan.baidu.com/s/1RMn76fktv1AmRB7kLlBmew |
| DOTA v1.0 | https://captain-whu.github.io/DOTA/dataset.html |

### 1.2.1 Dataset Processing

(1) HRSC2016 Dataset

First, access the HRSC2016 dataset from Kaggle at https://www.kaggle.com/datasets/guofeng/hrsc2016 and download the archive.zip file. After downloading, extract the archive into the data directory and rename the extracted folder as HRSC2016.

Then, save the dataset path to HRSC_ROOT and run the following command in the terminal.

```bash
export HRSC_ROOT=/data/HRSC2016
```

(2) DOTA v1.0 Dataset

First, download the DOTA v1.0 (Split) dataset from the following Baidu Netdisk link: https://pan.baidu.com/s/1RMn76fktv1AmRB7kLlBmew. After downloading the .part_00–.part_11 files, place them in the same directory and merge them into the original archive using the following procedure.

```bash
cat CEGA_DOTA_split_1024_dota1_0.tar.zst.part_* > CEGA_DOTA_split_1024_dota1_0.tar.zst
```

Then, extract CEGA_DOTA_split_1024_dota1_0.tar.zst into the data directory and set DOTA_ROOT to the path of the extracted dataset.

```bash
export DOTA_ROOT=data/CEGA_DOTA_split_1024_dota1_0
```

### 1.2.2 Test Experiments


| Dataset | Model | Download | Test script | Config | mAP |
|---|---|---|---|---|---|
| HRSC2016 | `work_dirs/CEGA_HRSC.pth` | [Google Drive](https://drive.google.com/file/d/1s5pOkjIgufcuBbfDuQkIjXjBCfc8zwxg/view?usp=drive_link) | `tools/CEGA_HRSC_test.sh` | `experiments/ablation/CEGA_HRSC_config.py` | `74.58%` |
| DOTA v1.0 | `work_dirs/CEGA_DOTA/CEGA_DOTA.pth` | [Google Drive](https://drive.google.com/file/d/1h6GoFyKSDvI_xxGmi31IarjgUn04mp3O/view?usp=drive_link) | `tools/CEGA_DOTA_test.sh` | `configs/strip_rcnn/CEGA_DOTA_config.py` | `87.85%` |

(1) HRSC2016 Test

Set `HRSC_ROOT` to the HRSC2016 dataset root, then run:

```bash
mkdir -p work_dirs
export HRSC_ROOT=/path/to/hrsc
export CHECKPOINT="$PWD/work_dirs/CEGA_HRSC.pth"
bash tools/CEGA_HRSC_test.sh
```
Here are the HRSC2016 test results in the picture below:

<img width="407" height="15" alt="image" src="https://github.com/user-attachments/assets/2f8176fe-5dc3-4373-80a0-306efa8e864c" />


(2) DOTA v1.0 Test

Run validation mAP from the PETDet root:

```bash
mkdir -p work_dirs/CEGA_DOTA
export DOTA_ROOT=/path/to/CEGA_DOTA_split_1024_dota1_0
export CHECKPOINT="$PWD/work_dirs/CEGA_DOTA/CEGA_DOTA.pth"
bash tools/CEGA_DOTA_test.sh val
```

Generate DOTA Task1 submission files:

```bash
export DOTA_ROOT=/path/to/CEGA_DOTA_split_1024_dota1_0
export CHECKPOINT="$PWD/work_dirs/CEGA_DOTA/CEGA_DOTA.pth"
bash tools/CEGA_DOTA_test.sh submit
```

The output is written to:

```text
work_dirs/CEGA_DOTA_test/
```
The DOTA v1.0 test results are shown in the picture below:

<img width="301" height="19" alt="image" src="https://github.com/user-attachments/assets/f58b1513-fc6a-4db2-bde9-43a874c58878" />

## 2 Training Experiments
To obtain the code, click the Code button on the repository homepage and select Download ZIP. After downloading, extract the archive and install the required interpreter packages as described above. Then, train the model on the HRSC2016 and DOTA v1.0 datasets.
| Dataset | PETDet training config | Main setting |
|---|---|---|
| HRSC2016 | `PETDet/experiments/ablation/serial_rot_scale_aclrpn_striphead_hrsc.py` | CEGA parallel branch, ACL-RPN, StripHead, 72 epochs |
| DOTA v1.0 | `PETDet/experiments/ablation/serial_rot_scale_aclrpn_striphead_dota.py` | CEGA parallel branch, ACL-RPN, StripHead, 12 epochs |

### 2.2.1 Dataset Processing
The dataset preparation follows the same procedure as described in Section 1.2.1.

Note: If you prefer to split the DOTA v1.0 dataset into patches yourself, please download the original dataset from https://captain-whu.github.io/DOTA/dataset.html, then use the following code to perform the splitting, and set the DOTA_ROOT path variable accordingly.

```bash
python tools/data/dota/split/img_split.py \
  --base-json tools/data/dota/split/split_configs/ss_trainval.json

python tools/data/dota/split/img_split.py \
  --base-json tools/data/dota/split/split_configs/ss_test.json
```
### 2.2.2 Training The Model
(1) HRSC2016 Training

Run from the PETDet root:

```bash
python tools/train.py experiments/ablation/serial_rot_scale_aclrpn_striphead_hrsc.py --seed 3407
```

If your dataset path is different from the path in the config, update the config.

(2) DOTA v1.0 Training

Run from the PETDet root:

```bash
python tools/train.py experiments/ablation/serial_rot_scale_aclrpn_striphead_dota.py --seed 332845056
```

If your dataset path is different from the path in the config, update the config
or override the data path with --cfg-options.

### 2.2.3 Main Code File Location
(1) Rotation Equivariant Group: PETDet/mmrotate/models/backbones/re_resnet.py

(2) Scale Equivariant Group: PETDet/mmrotate/models/backbones/scale_re_resnet.py

(3) ACL-RPN: PETDet/mmrotate/models/dense_heads/acl_rpn_head.py
