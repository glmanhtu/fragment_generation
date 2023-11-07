# Papyrus fragment generator

### Step 1: remove background of the papyrus fragments
```bash
python3 clean_images.py --dataset-dir /path/to/michigan_base --output-dir /path/to/michigan_clean
```

### Step 2: Extract edges
```bash
python3 extract_edges.py --dataset-dir /path/to/michigan_clean --edge-json edges.json --output-dir /path/to/michigan_edges
```

### Step 3: Standardize the images
```bash
git clone https://github.com/glmanhtu/homer-competition.git
cd homer-competition
python3 standarize_images.py --pretrained_model_path /path/to/pretrained_models --ref_box_height 32 --image_size 800 --dataset /path/to/michigan_clean --prediction_path /path/to/michigan_clean_standardized --cuda --p1_arch resnet50 --p2_arch resnet50
```

### Step 4: Generate fragments
```bash
python3 fragment_generator.py --dataset-dir /path/to/michigan_clean_standardized --output-dir /path/to/fragments --edges-json /path/to/michigan_edges/edges.json
```