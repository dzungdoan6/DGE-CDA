## INSTALLATION
Our installation is mainly based on [Detectron2's installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) with few modifications
* Clone this repo
* Install python 3.9





## DATASET PREPARATION
Suppose the source code is saved in `[ROOT_DIR]/Energy-Aware-CDA/`

* Download the dataset **DGTA_SeaDronesSee_jpg** from [DeepGTAV](https://github.com/David0tt/DeepGTAV) (suppose save it to `[ROOT_DIR]/DGTA_SeaDronesSee_jpg`)
* Create folder `DGTA_SeaDronesSee_merged/` inside `[ROOT_DIR]/Energy-Aware-CDA/datasets/`
* Create folder `images/` inside `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/`
* Copy all images from `[ROOT_DIR]/DGTA_SeaDronesSee_jpg/train/images/` to `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/images/`
* Copy all images from `[ROOT_DIR]/DGTA_SeaDronesSee_jpg/val/images/` to `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/images/`
* Create folder `experiments/` inside `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/`
* Download all json files from HERE and put them in `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/experiments/`

Finally, we should obtain
* all images saved in `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/images/` 
* and all json files located in `[ROOT_DIR]/Energy-Aware-CDA/datasets/DGTA_SeaDronesSee_merged/experiments/`


