import json

def wash_bdd(file_path):
    print(f"🔄 Cleaning and mapping names for {file_path} ...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # CORRECT mapping: Convert Kaggle 2018 old names to the code's expected Scalabel new names
    name_map = {
        "person": "pedestrian",
        "bike": "bicycle",
        "motor": "motorcycle"
    }
    
    for item in data:
        if "labels" in item:
            valid_labels = []
            for label in item["labels"]:
                if "box2d" in label:
                    # Map to the new expected name if it exists in our dictionary
                    current_name = label.get("category", "")
                    if current_name in name_map:
                        label["category"] = name_map[current_name]
                    
                    valid_labels.append(label)
            item["labels"] = valid_labels
            
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print(f"✅ Correction complete: {file_path}\n")

# Execute cleaning and downgrading
wash_bdd('bdd100k/labels/det_train.json')
wash_bdd('bdd100k/labels/det_val.json')