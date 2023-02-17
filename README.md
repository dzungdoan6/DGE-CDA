# Installation
Our installation is mainly based on [Detectron2's installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) with few modifications
* Clone this repo, suppose the source code is saved in `[ROOT_DIR]/DGE-CDA/`

* Install Python >= 3.7 (tested on Python 3.9)

* Install Pytorch >= 1.8 and torchvision that matches the Pytorch installation (tested on Pytorch 1.11, torchvision 0.12, CUDA 11.3)

* Install OpenCV 
    ```
    pip install opencv-python
    ``` 
    (tested on opencv-python 4.5.5.64)

* Install detectron2 
    ```
    cd [ROOT_DIR] 
    python -m pip install -e DGE-CDA 
    cd [ROOT_DIR]/DGE-CDA
    ```

* Install POT library 
    ```
    pip install POT
    ```

* Install pyJoules 
    ```
    pip install pyJoules[nvidia]
    ```

* Install tqdm 
    ```
    pip install tqdm
    ```


# Dataset preparation
* Download the dataset **DGTA_SeaDronesSee_jpg** from [DeepGTAV](https://github.com/David0tt/DeepGTAV) (suppose save it to `[ROOT_DIR]/DGTA_SeaDronesSee_jpg`)

* Create folder `DGTA_SeaDronesSee_merged/` inside `[ROOT_DIR]/DGE-CDA/datasets/`

* Create folder `images/` inside `[ROOT_DIR]/DGE-CDA/datasets/DGTA_SeaDronesSee_merged/`

* Copy all images from `[ROOT_DIR]/DGTA_SeaDronesSee_jpg/train/images/` to `[ROOT_DIR]/DGE-CDA/datasets/DGTA_SeaDronesSee_merged/images/`

* Copy all images from `[ROOT_DIR]/DGTA_SeaDronesSee_jpg/val/images/` to `[ROOT_DIR]/DGE-CDA/datasets/DGTA_SeaDronesSee_merged/images/`

* Create folder `experiments/` inside `[ROOT_DIR]/DGE-CDA/datasets/DGTA_SeaDronesSee_merged/`

* Download all json files from [HERE](https://drive.google.com/drive/folders/1pYuIfSNG31ks6Q1_Bb292cdOa32R68PZ?usp=sharing) and put them in `[ROOT_DIR]/DGE-CDA/datasets/DGTA_SeaDronesSee_merged/experiments/`

Finally, we should obtain
* all images located in `[ROOT_DIR]/DGE-CDA/datasets/DGTA_SeaDronesSee_merged/images/` 

* and all json files located in `[ROOT_DIR]/DGE-CDA/datasets/DGTA_SeaDronesSee_merged/experiments/`

# DEMO
* Pretrain RetinaNet on source domain 
    ```
    python cda/pretrain_source.py --config-file configs/dgta.yaml
    ```
    After the training finishes, we should find `model_final.pth` located in `work_dir/DeepGTAV/CLEAR_9-15/`

* Randomly generate random projections 
    ```
    python cda/generate_random_projections.py --save-dir work_dir/DeepGTAV/projections
    ```
    We should find 3 projections in the folder `work_dir/DeepGTAV/projections`

* To plot the histograms of AP discrepancy and domain gaps (Fig.5 in the paper), run 
    ```
    python cda/plot_correlation.py --config-file configs/dgta.yaml --projs-dir work_dir/DeepGTAV/projections
    ```

    This script will take about 4h in Intel-Core-i7 with 16GB RAM and NVIDIA GeForce GTX 1080Ti

* To reproduce the results in Table 5, 

    - For SDA with DGE
    ```
    python cda/cont_dgta.py --config-file configs/dgta_cont.yaml --projs-dir work_dir/DeepGTAV/projections --gap-thr 0.02 --dge
    ```
    The trained models can be found in `work_dir/DeepGTAV/adapt_with_dge/`. The energy consumption record can be found in its subfolders, e.g., `work_dir/DeepGTAV/adapt_with_dge/CLEAR_9-15_and_OVERCAST_0-1/energy.csv`

    - For SDA w/o DGE 
    ```
    python cda/cont_dgta.py --config-file configs/dgta_cont.yaml
    ```
    The trained models can be found in `work_dir/DeepGTAV/adapt_wo_dge/`. The energy consumption record can be found in its subfolders.
    
    - It should be noted that 
        - The energy unit is Joule
        - there is a common error `Permission denied: '/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj'`. To resolve it, simply grant the permission 
        ```
        sudo chmod -R 7777 /sys/class/powercap/intel-rapl/intel-rapl:0
        ```

    