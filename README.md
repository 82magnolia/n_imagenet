# N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras
Official PyTorch implementation of **N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras (ICCV 2021)** [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_N-ImageNet_Towards_Robust_Fine-Grained_Object_Recognition_With_Event_Cameras_ICCV_2021_paper.html) [[Video]](https://www.youtube.com/watch?v=7mWPYGRfk-I).

[<img src="sample_1.png" width="500"/>](sample_1.png)
[<img src="sample_2.png" width="500"/>](sample_2.png)


In this repository, we provide instructions for downloading N-ImageNet along with the implementation of the baseline models presented in the paper. 
If you have any questions regarding the dataset or the baseline implementations, please leave an issue or contact 82magnolia@snu.ac.kr.

## Downloading N-ImageNet
To download N-ImageNet, please use the following [DropBox link](https://www.dropbox.com/sh/47we7z2gff5barh/AADU4GyWnzLFzMzBDjLP00baa?dl=0).
**Note:** We are having a temporal problem with DropBox right now. We expect to fix the issue within five business days. Sorry for the inconvenience.
Once you go to the DropBox folder, there will be two .7z files: N_ImageNet.7z (original N-ImageNet train/val set) and N_ImageNet_cam.7z (N-ImageNet variants recorded under diverse camera motion and illumination).
As the file sizes are large, it may take up to 1 day to download the entire N-ImageNet dataset.
If the download is too slow, we also provide an option to share the DropBox folder once you fill out the following questionaire: [[Link]](https://docs.google.com/forms/d/e/1FAIpQLSehK79GEtkyqKLTcrzD_MAOpEo6oL-zHBT32CcOvz1KjzsfIA/viewform?usp=sf_link).
**Note:** While we try to review the questionaires on a regular basis, we often forget to do so. Therefore, if you want quick DropBox sharing, please drop an email to 82magnolia@snu.ac.kr notifying that you have filled out the form.

Note that for N_ImageNet_cam.7z, the file names are different from the names used in the ICCV 2021 paper.
Please use the following table to convert folder names between the N-ImageNet variants (saved in N_Imagenet_cam.7z) and the variant names presented in the paper.

| DropBox                | Paper                   |
|------------------------|-------------------------|
| val_mode_1             | Validation 1            |
| val_mode_3             | Validation 5            |
| val_mode_5             | Validation 2            |
| val_mode_6             | Validation 3            |
| val_mode_7             | Validation 4            |
| val_mode_brightness_4  | Validation 6            |
| val_mode_brightness_5  | Validation 7            |
| val_mode_brightness_6  | Validation 8            |
| val_mode_brightness_7  | Validation 9            |

If you have any additional questions regarding the dataset, drop an email to 82magnolia@snu.ac.kr.


## Training / Evaluating Baseline Models
### Installation
The codebase is tested on a Ubuntu 18.04 machine with CUDA 10.1. However, it may work with other configurations as well.
First, create and activate a conda environment with the following command.
```
conda env create -f environment.yml
conda activate e2t
```
In addition, you must install pytorch_scatter. Follow the instructions provided in the [pytorch_scatter github repo](https://github.com/rusty1s/pytorch_scatter). You need to install the version for torch 1.7.1 and CUDA 10.1.

### Dataset Setup
Before you move on to the next step, please download N-ImageNet. Once you download N-ImageNet, you will spot a structure as follows.
```
N_Imagenet
├── train_list.txt
├── val_list.txt
├── extracted_train (train split)
│   ├── nXXXXXXXX (label)
│   │   ├── XXXXX.npz (event data)
│   │   │
│   │   ⋮
│   │   │
│   │   └── YYYYY.npz (event data)
└── extracted_val (val split)
    └── nXXXXXXXX (label)
        ├── XXXXX.npz (event data)
        │
        ⋮
        │
        └── YYYYY.npz (event data)
```
The N-ImageNet variants file (which would be saved as `N_Imagenet_cam` once downloaded) will have a similar file structure, except that it only contains validation files.
The following instruction is based on N-ImageNet, but one can follow a similar step to test with N-ImageNet variants.

First, modify `train_list.txt` and `val_list.txt` such that it matches the directory structure of the downloaded data.
To illustrate, if you open `train_list.txt` you will see the following
```
/home/jhkim/Datasets/N_Imagenet/extracted_train/n01440764/n01440764_10026.npz
⋮
/home/jhkim/Datasets/N_Imagenet/extracted_train/n15075141/n15075141_999.npz
```
Modify each path within the .txt file so that it accords with the directory in which N-ImageNet is downloaded.
For example, if N-ImageNet is located in `/home/user/assets/Datasets/`, modify `train.txt` as follows.
```
/home/user/assets/Datasets/N_Imagenet/extracted_train/n01440764/n01440764_10026.npz
⋮
/home/user/assets/Datasets/N_Imagenet/extracted_train/n15075141/n15075141_999.npz
```
Once this is done, create a `Datasets/` directory within `real_cnn_model`, and create a symbolic link within `Datasets`.
To illustrate, using the directory structure of the previous example, deploy the following command.
```
cd PATH_TO_REPOSITORY/real_cnn_model
mkdir Datasets; cd Datasets
ln -sf /home/user/assets/Datasets/N_Imagenet/ ./
ln -sf /home/user/assets/Datasets/N_Imagenet_cam/ ./  (If you have also downloaded the variants)
```
Congratulations! Now you can start training/testing models on N-ImageNet.

### Training a Model
You can train a model based on the binary event image representation with the following command.
```
export PYTHONPATH=PATH_TO_REPOSITORY:$PYTHONPATH
cd PATH_TO_REPOSITORY/real_cnn_model
python main.py --config configs/imagenet/cnn_adam_acc_two_channel_big_kernel_random_idx.ini
```
For the examples below, we assume the `PYTHONPATH` environment variable is set as above.
Also, you can change minor details within the config before training by using the `--override` flag.
For example, if you want to change the batch size use the following command.
```
python main.py --config configs/imagenet/cnn_adam_acc_two_channel_big_kernel_random_idx.ini --override 'batch_size=8'
```

### Evaluating a Model
Suppose you have a pretrained model saved in `PATH_TO_REPOSITORY/real_cnn_model/experiments/best.tar`.
You evaluate the performance of this model on the N-ImageNet validation split by using the following command.
```
python main.py --config configs/imagenet/cnn_adam_acc_two_channel_big_kernel_random_idx.ini --override 'load_model=PATH_TO_REPOSITORY/real_cnn_model/experiments/best.tar'
```

### Naming Conventions
The naming of event representations used in the codebase is different from that of the original paper. Please use the following table to convert event representations used in the paper to event representations used in the codebase.

| Paper               | Codebase                   |
|---------------------|----------------------------|
| DiST                | reshape_then_acc_adj_sort  |
| Binary Event Image  | reshape_then_acc_flat_pol  |
| Event Image         | reshape_then_acc           |
| Timestamp Image     | reshape_then_acc_time_pol  |
| Event Histogram     | reshape_then_acc_count_pol |
| Sorted Time Surface | reshape_then_acc_sort      |

### Downloading Pretrained Models
One can download the pretrained models through the [following link](https://drive.google.com/drive/folders/1aK7oLyu0_TKWdDNB8j1X4PSceIPtWifA?usp=sharing).
Here we contain pretrained models and the configs used to train them.

## Citation
If you find the dataset or codebase useful, please cite

```bibtex
@InProceedings{Kim_2021_ICCV,
    author    = {Kim, Junho and Bae, Jaehyeok and Park, Gangin and Zhang, Dongsu and Kim, Young Min},
    title     = {N-ImageNet: Towards Robust, Fine-Grained Object Recognition With Event Cameras},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2146-2156}
}
```
