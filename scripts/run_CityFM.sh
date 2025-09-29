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

# Step 1: Encode POIs by CityFM
echo "Step 1: Encoding POIs using CityFM..."
CUDA_VISIBLE_DEVICES=5 python baselines/CityFM/encode.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: CityFM encoding failed"
    exit 1
fi

# Step 2: Preprocess data for CityFM
echo "Step 2: Preprocessing data for CityFM..."
python preprocess/preprocess.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi

# Step 3: Average pooling for POI embeddings
echo "Step 3: Performing average pooling for POI embeddings..."
python baselines/CityFM/avg_pooling.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: Average pooling failed"
    exit 1
fi  

# Step 4: Predict with CityFM
echo "Step 4: Running CityFM prediction..."
CUDA_VISIBLE_DEVICES=5 python baselines/CityFM/predict_k-fold.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: CityFM prediction failed"
    exit 1
fi

