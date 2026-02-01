# SubUrban

- This is the official repository of the submitted paper: *Learning Autonomous Urban Region Representations through
LLM-Informed Reinforcement Optimization*

## Quick Start

- With this repository, you can
  - Process urban POI data with automatic keyword generation and filtering
  - Train and Evaluate SubUrban model on urban region prediction tasks (population density prediction, house price prediction, GDP density prediction)
  - Compare with baselines (SOTA baselines: CityFM, HGI)

### Environment Preparation

- Please use **Miniconda or Anaconda**
- We use Python 3.10+. Lower versions are not tested.

  ```bash
  conda create -n SubUrban python==3.10
  conda activate SubUrban
  ```

We require the following packages:

- PyTorch>=1.12.0
- torch-geometric (latest version), and its dependencies including torch-scatter and torch-sparse
- transformers>=4.20.0
- scikit-learn>=1.0.0
- numpy>=1.21.0
- pandas>=1.3.0
- shapely>=1.8.0
- tqdm
- openai (for GPT-based keyword generation)
- rank-bm25 (for BM25 filtering)

Install dependencies:
```bash
pip install -r requirements.txt
```

### API Key Configuration

**Important**: This project uses LLM APIs for keyword generation and model operations. API keys are read from environment variables.

**Setup Instructions**:
1. Export your API key in the shell before running any script:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

If you use DeepSeek for LLM guidance, also export:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

After exporting, all scripts will automatically read the key from the environment. You do **not** need to edit any `.sh` file.

## Project Structure

```
SubUrban/
├── baselines/          # Baseline methods for comparison
│   ├── BERT/
│   ├── CityFM/
│   └── HGI/
├── data/              # Data storage
│   ├── Gaode/         # For Beijing and Shanghai
│   └── OSM/           # For Singapore and NYC
├── embs/              # Generated embeddings storage (created at runtime)
├── model/             # Main SubUrban model implementation
│   └── SubUrban_model.py
├── preprocess/        # Data preprocessing pipeline
│   ├── preprocess.py
│   ├── GPT_get_keywords.py
│   └── BM25_filtering_keywordKmeans.py
├── scripts/           # Execute the piplines of SubUrban and SOTA baselines
│   ├── run_SubUrban.sh
│   ├── run_CityFM.sh
│   └── run_HGI.sh
└── tmp/               # Temporary files during processing (created at runtime)
```

### Singapore GDP Data (Optional, Large File)

The Singapore GDP raster (2019GDP.tif) is an estimated dataset derived from nighttime-light calibrated economic activity (not official statistics).  
It is large and not included in this repository.  
If you need the Singapore GDP task, download it from:

```
https://zenodo.org/records/16741980
```

Place the file at:

```
SubUrban/data/Gaode/raw/GDP/2019GDP.tif
```

### Data Processing Pipeline

The SubUrban pipeline consists of 4 main steps:

1. **Data Preprocessing**: Process raw urban data (POI, housing, GDP, population)
2. **GPT Keyword Generation**: Generate region-specific keywords using GPT-4
3. **BM25 Filtering**: Filter and rank POIs using BM25 + K-means clustering
4. **BERT Encoding**: Generate embeddings for filtered POIs
5. **SubUrban Model Training**: Train RL model with multi-task optimization

## Experiments

- We provide scripts to repeat the experiments and run the complete pipeline of SubUrban.
- Please run the main script in the scripts/ folder.

### Running the Complete Pipeline

```bash
cd scripts/
./run_SubUrban.sh Beijing
```

or

```bash
cd scripts/
./run_SubUrban.sh Shanghai
```

### Running SOTA Baselines (CityFM, HGI)

We provide runnable scripts for the two baseline methods:

```bash
cd scripts/
./run_CityFM.sh Beijing
```

```bash
cd scripts/
./run_HGI.sh Beijing
```

You can replace `Beijing` with `Shanghai`, `Singapore`, or `NYC` as needed.

## Thanks for reading!
