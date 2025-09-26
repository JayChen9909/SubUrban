#!/bin/bash

# Check if city name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <city_name>"
    echo "Example: $0 Beijing"
    echo "Available cities: Beijing, Shanghai"
    exit 1
fi

CITY=$1
echo "Starting SubUrban pipeline for city: $CITY"

# Change to the SubUrban directory
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# Step 1: Run preprocessing
echo "Step 1: Running preprocessing..."
python ./preprocess/preprocess.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi

# Step 2: Generate keywords using GPT
echo "Step 2: Generating keywords using GPT..."
python ./preprocess/GPT_get_keywords.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: GPT keyword generation failed"
    exit 1
fi

# Step 3: Run BM25 filtering with keywords_kmeans
echo "Step 3: Running BM25 filtering..."
python ./preprocess/BM25_filtering_keywordKmeans.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: BM25 filtering failed"
    exit 1
fi

# Step 4: Re-scan and Re-encode for filtered POIs
echo "Step 4: Re-scan and Re-encode for filtered POIs..."
python ./preprocess/preprocess.py --city "$CITY" --poi_mode filtered
python ./baselines/BERT/BERT_encode.py --city "$CITY" --mode filtered
if [ $? -ne 0 ]; then
    echo "Error: Re-scan and Re-encode failed"
    exit 1
fi

# Step 5: Run SubUrban model
echo "Step 5: Running SubUrban model..."
python ./model/SubUrban_model.py --city "$CITY"
if [ $? -ne 0 ]; then
    echo "Error: SubUrban model failed"
    exit 1
fi

echo "SubUrban pipeline completed successfully for $CITY"
