#!/bin/bash

# Check if city name is provided
if [ $# -eq 0 ]; then
	echo "Usage: $0 <city_name>"
	echo "Example: $0 Beijing"
	echo "Available cities: Beijing, Shanghai, NYC, Singapore"
	exit 1
fi

CITY=$1
echo "Starting SubUrban HGI pipeline for city: $CITY"

# Change to the SubUrban directory
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# Step 1: Build HGI graph
echo "Step 1: Building HGI graph..."
python baselines/HGI/build_graph.py --city "$CITY"
if [ $? -ne 0 ]; then
	echo "Error: HGI graph building failed"
	exit 1
fi

# Step 2: BERT encode for HGI
echo "Step 2: BERT encoding for HGI..."
CUDA_VISIBLE_DEVICES=5 python baselines/BERT/BERT_encode.py --city "$CITY"
if [ $? -ne 0 ]; then
	echo "Error: BERT encoding failed"
	exit 1
fi

# Step 3: Preprocess data for HGI
echo "Step 3: Preprocessing data for HGI..."
python preprocess/preprocess.py --city "$CITY"
if [ $? -ne 0 ]; then
	echo "Error: Preprocessing failed"
	exit 1
fi

# Step 4: HGI dataloader
echo "Step 4: Running HGI dataloader..."
python baselines/HGI/dataloader.py --city "$CITY"
if [ $? -ne 0 ]; then
	echo "Error: HGI dataloader failed"
	exit 1
fi

# Step 5: Train HGI model
echo "Step 5: Training HGI model..."
CITY_LOWER=$(echo "$CITY" | awk '{print tolower($0)}')
CUDA_VISIBLE_DEVICES=5 python baselines/HGI/train.py --city "$CITY_LOWER"
if [ $? -ne 0 ]; then
	echo "Error: HGI training failed"
	exit 1
fi

# Step 6: Prepare HGI output
echo "Step 6: Preparing HGI output..."
python baselines/HGI/prepare.py --city "$CITY"
if [ $? -ne 0 ]; then
	echo "Error: HGI prepare failed"
	exit 1
fi

# Step 7: Predict with HGI
echo "Step 7: Running HGI prediction..."
python baselines/HGI/predict_k-fold.py --city "$CITY"
if [ $? -ne 0 ]; then
	echo "Error: HGI prediction failed"
	exit 1
fi