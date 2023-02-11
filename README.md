## Installation
Our installation is mainly based on [Detectron2's installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) with few modifications
* Clone this repo, suppose the source code is saved in `[ROOT_DIR]/Energy-Aware-CDA/`
* Install Python >= 3.7 (tested on Python 3.9)
* Install Pytorch >= 1.8 and torchvision that matches the Pytorch installation (tested on Pytorch 1.11, torchvision 0.12, CUDA 11.3)
* Install OpenCV `pip install opencv-python` (tested on opencv-python 4.5.5.64)
* Run `cd [ROOT_DIR]`
* Run `python -m pip install -e Energy-Aware-CDA`
* Run `cd [ROOT_DIR]/Energy-Aware-CDA`
* Install POT library (`pip install POT`)


## Dataset preparation
* Download the dataset **DGTA_SeaDronesSee_jpg** from [DeepGTAV](https://github.com/David0tt/DeepGTAV) (suppose save it to `[ROOT_DIR]/DGTA_SeaDronesSee_jpg`)
* Create folder `DGTA_SeaDronesSee_merged/` inside `[ROOT_DIR]/Energy-Aware-CDA/datasets/`
* Create folder `images/` inside `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/`
* Copy all images from `[ROOT_DIR]/DGTA_SeaDronesSee_jpg/train/images/` to `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/images/`
* Copy all images from `[ROOT_DIR]/DGTA_SeaDronesSee_jpg/val/images/` to `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/images/`
* Create folder `experiments/` inside `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/`
* Download all json files from [HERE](https://drive.google.com/drive/folders/1pYuIfSNG31ks6Q1_Bb292cdOa32R68PZ?usp=sharing) and put them in `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/experiments/`

Finally, we should obtain
* all images located in `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/images/` 
* and all json files located in `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/experiments/`

## DEMO
* Pretrain RetinaNet on source domain `python cda/pretrain_source.py --config-file configs/dgta.yaml`



