#!/bin/bash

# Check if city name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <city_name>"
    echo "Example: $0 Beijing"
    echo "Available cities: Beijing, Shanghai, NYC, Singapore"
    exit 1
fi

CITY=$1
echo "Starting SubUrban pipeline for city: $CITY"

# Change to the SubUrban directory
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# Step 1: Preprocess and Pretrain
echo "Step 1: Preprocessing and Pretraining..."
pushd baselines/CityFM > /dev/null
python CityFM_preprocess.py -c "$CITY"
CUDA_VISIBLE_DEVICES=5 python CityFM_pretrain.py -c "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: Preprocessing or Pretraining failed"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

# Step 2: Encode POIs by CityFM
echo "Step 2: Encoding POIs using CityFM..."
CUDA_VISIBLE_DEVICES=5 python baselines/CityFM/encode.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: CityFM encoding failed"
    exit 1
fi

# Step 3: Preprocess data for CityFM
echo "Step 3: Preprocessing data for CityFM..."
python preprocess/preprocess.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi

# Step 4: Average pooling for POI embeddings
echo "Step 4: Performing average pooling for POI embeddings..."
python baselines/CityFM/avg_pooling.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: Average pooling failed"
    exit 1
fi  

# Step 5: Predict with CityFM
echo "Step 5: Running CityFM prediction..."
CUDA_VISIBLE_DEVICES=5 python baselines/CityFM/predict_k-fold.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: CityFM prediction failed"
    exit 1
fi

