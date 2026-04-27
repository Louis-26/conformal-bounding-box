## environment basic configuration
python: 3.9
cuda: 11.7
gcc: 9.3

## step 1: set up the environment
load appropriate cuda environment, meanwhile, should ignore json
```bash
cd $(git rev-parse --show-toplevel)
conda env create -f env_minimal_linux.yml -y
conda activate conf1m
```

### complementary setup
need to use cuda 11.7
```bash
conda install -n conf1m "mkl<2024" -c conda-forge
conda install -n conf1m -c "nvidia/label/cuda-11.7.1" cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvtx
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install shapely
pip install cityscapesscripts
pip install kaggle
```

## step 2: data preparation
Download COCO, Cityscapes and BDD100k datasets from their respective websites

### download COCO
```bash
cd $(git rev-parse --show-toplevel)/data
mkdir -p coco
cd coco
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && rm -rf val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm -rf annotations_trainval2017.zip

wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
unzip coco2017labels.zip && rm -rf coco2017labels.zip
mkdir -p labels/val2017

cd coco

mv labels/val2017 ../labels
cd .. && rm -rf coco
```

### download cityscapes
download cityscapes from [gtFine](https://www.cityscapes-dataset.com/file-handling/?packageID=1) and [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3), then scp the zip files

```bash
cd $(git rev-parse --show-toplevel)/data
mkdir -p cityscapes
cd cityscapes
# use your email and password
wget --keep-session-cookies --save-cookies=cookies.txt \
  --post-data 'username=$YOUR_EMAIL$&password=$YOUR_PASSWORD$&submit=Login' \
  https://www.cityscapes-dataset.com/login/ \
  -O /dev/null

# download gtFine
wget --load-cookies cookies.txt \
  --content-disposition \
  https://www.cityscapes-dataset.com/file-handling/?packageID=1


# download leftImg8bit
wget --load-cookies cookies.txt \
  --content-disposition \
  https://www.cityscapes-dataset.com/file-handling/?packageID=3

rm -rf cookies.txt

# unzip
unzip -q gtFine_trainvaltest.zip && rm -rf gtFine_trainvaltest.zip
unzip -q leftImg8bit_trainvaltest.zip && rm -rf leftImg8bit_trainvaltest.zip
rm -rf README license.txt

```


### download bdd100k
prepare kaggle
```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"YOUR_USERNAME","key":"YOUR_KEY"}
EOF

chmod 700 ~/.kaggle/kaggle.json
```

```bash
cd $(git rev-parse --show-toplevel)/data
mkdir -p bdd100k
cd bdd100k
kaggle datasets download marquis03/bdd100k
unzip -q bdd100k.zip && rm -rf bdd100k.zip

```

## step 3: Download necessary model checkpoints
```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints && cd checkpoints
conda activate conf1m
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1zzv8Q38GvwRKfDgSEgE_hrNGQ7fZCks5
mv ./conformalod_models/* . && rm -rf ./conformalod_models

cd "$(git rev-parse --show-toplevel)"
grep -qxF '/checkpoints/' .gitignore 2>/dev/null || echo '/checkpoints/' >> .gitignore
```

## step 4: test the main file
### ✅main script usage test
```bash
cd $(git rev-parse --show-toplevel)
conda activate conf1m
python main.py -h
```



### ✅main script for coco validation
add saving folder
```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p output
```


```bash
python main.py --config_file=cfg_std_rank --config_path=config/coco_val --run_collect_pred --save_file_pred --risk_control=std_conf --alpha=0.1 --label_set=class_threshold --label_alpha=0.01 --run_risk_control --save_file_control --save_label_set --run_eval --save_file_eval --file_name_suffix=_std_rank_class --device=cuda



```

### ⏳script of full suite of experiments 
```bash
python commands.py  # generate commands.txt
bash run.sh  # read and run all combos
```

### model baseline
```bash
python control/baseline_gaussian_yolo.py --config_file=cfg_base_gaussian_yolo --config_path=config/coco_val --run_collect_pred --save_file_pred --risk_control=gaussian_yolo --alpha=0.1 --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_base_gaussian_yolo --device=cuda
```

### output interpretation
```bash
python main.py --config_file=cfg_std_rank --config_path=config/coco_val --run_collect_pred --save_file_pred --risk_control=std_conf --alpha=0.1 --label_set=class_threshold --label_alpha=0.01 --run_risk_control --save_file_control --save_label_set --run_eval --save_file_eval --file_name_suffix=_std_rank_class --device=cuda
```