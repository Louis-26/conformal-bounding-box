change all .py files under control/ folder
```bash
cd "$(git rev-parse --show-toplevel)/control"

find . -name "*.py" -type f | while read -r file; do
    sed -i 's|^sys.path.insert(0, "/home/atimans/Desktop/project_1/conformalbb/detectron2")|# &\nsys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "detectron2"))|' "$file"
    
    echo "Processed: $file"
done
```

```python
# original
sys.path.insert(0, "/home/atimans/Desktop/project_1/conformalbb/detectron2")
# new
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "detectron2"))
```


## main.py
```python
import sys
# original
sys.path.insert(0, "/home/atimans/Desktop/project_1/conformalbb/detectron2")
# new
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "detectron2"))

# add
from pathlib import Path
```

change all yaml files under config, for directory of data and output


change bdd100k yaml files under config

```bash
cd "$(git rev-parse --show-toplevel)/config"

find . -type f \( -name "*.yaml" -o -name "*.yml" \) -print0 |
while IFS= read -r -d '' file; do

    if grep -q "/media/atimans/hdd/" "$file"; then
        sed -i -E '
s|^([[:space:]]*OUTPUT_DIR: \&outputdir ").*/media/atimans/hdd/(output[^"]*)"$|# &\n\1./\2"|;
s|^([[:space:]]*DIR: ").*/media/atimans/hdd/datasets"$|# &\n\1./data"|;
' "$file"

        echo "✅ Updated: $file"
    fi
done
```




for instance
## /config/coco_val/cfg_std_rank.yaml
```yaml
# line 10
# original
OUTPUT_DIR: &outputdir "../../../../media/atimans/hdd/output"
# new
OUTPUT_DIR: &outputdir "./output"


# line 17
# original
DIR: "../../../../media/atimans/hdd/datasets"
# new
DIR: "./data"
```


## data/data_loader.py
```python
# line 151-152
# original
    if mapper is None:
        mapper = DatasetMapper(cfg_model, is_train=train)
# new
    if mapper is None:
        # mapper = DatasetMapper(cfg_model, is_train=train)
        if train:
            mapper = DatasetMapper(cfg_model, is_train=True)
        else:
            # Custom test-mode mapper that keeps raw 'annotations' field
            # for conformal prediction (GT box matching, residual computation).
            from detectron2.data import transforms as T
            from detectron2.data import detection_utils as utils
            import copy
            import torch

            def mapper(dataset_dict):
                dataset_dict = copy.deepcopy(dataset_dict)
                image = utils.read_image(
                    dataset_dict["file_name"], format=cfg_model.INPUT.FORMAT
                )
                utils.check_image_size(dataset_dict, image)

                aug = T.ResizeShortestEdge(
                    short_edge_length=cfg_model.INPUT.MIN_SIZE_TEST,
                    max_size=cfg_model.INPUT.MAX_SIZE_TEST,
                    sample_style="choice",
                )
                transform = aug.get_transform(image)
                image = transform.apply_image(image)
                if "annotations" in dataset_dict:
                    for anno in dataset_dict["annotations"]:
                        anno.pop("segmentation", None)
                        anno.pop("keypoints", None)
                dataset_dict["image"] = torch.as_tensor(
                    image.transpose(2, 0, 1).astype("float32")
                )
                # 'annotations' field is kept untouched in original image coords
                return dataset_dict
```


### detectron2/detectron2/model_zoo/model_zoo.py
```python
# line 136-137
# original
def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: the real path to the config file.
    """
    # print("config_path: ", config_path)
    cfg_file = pkg_resources.resource_filename(
        "detectron2.model_zoo", os.path.join("configs", config_path)
    )
    
    # print("cfg_file: ", cfg_file)
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file
# new 
def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: the real path to the config file.
    """
    # print("config_path: ", config_path)
    detectron2_root = Path(__file__).resolve().parents[2]

    cfg_file = detectron2_root / "configs" / config_path
    
    # print("cfg_file: ", cfg_file)
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file
```

commands.py
```python
# line 35-41
# remove all conformalbb/
```

## model/qr_head.py
```python
# line 62 
# original
cfg_q = io_file.load_yaml("qr_cfg", "conformalbb/model", True)
# new
cfg_q = io_file.load_yaml("qr_cfg", "./model", True)
```

## model/qr_head.py
```python
# line 202 
# original
default="conformalbb/config",
# new
default="./config",
```