# SubUrban

- This is the official repository of the submitted paper: *AUTONOMOUS URBAN REGION REPRESENTATION
WITH LLM-INFORMED REINFORCEMENT LEARNING*

## Quick Start

- With this repository, you can
  - Process urban POI data with automatic keyword generation and filtering
  - Train and Evaluate SubUrban model on urban region prediction tasks (population density prediction, house price prediction, GDP density prediction)
  - Compare with baselines (update soon)

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

**Important**: This project uses LLM APIs for keyword generation and model operations. You need to configure your API key before running the pipeline.

The following scripts require API key configuration:
- `SubUrban_model.py` (main model with LLM integration)
- `GPT_get_keywords.py` (GPT-based keyword generation)

**Setup Instructions**:
1. Open the respective script files
2. Locate the variable named `api_key`
3. Replace the placeholder with your actual API key from OpenAI or DeepSeek platform

Example:
```python
# In SubUrban_model.py and GPT_get_keywords.py
api_key = 'your-actual-api-key-here'  # Replace with your OpenAI or DeepSeek API key
```

**Supported Platforms**:
- OpenAI
- DeepSeek

## Project Structure

```
SubUrban/
├── baselines/          # Baseline methods including BERT encoding
│   └── BERT/
├── data/              # Dataset storage (Beijing, Shanghai)
│   ├── Gaode/
│   │   └── projected/
├── embs/              # Generated embeddings storage
├── model/             # Main SubUrban model implementation
│   └── SubUrban_model.py
├── preprocess/        # Data preprocessing pipeline
│   ├── preprocess.py
│   ├── GPT_get_keywords.py
│   └── BM25_filtering_keywordKmeans.py
├── scripts/           # Execution scripts
│   └── run_SubUrban.sh
└── tmp/               # Temporary files during processing
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

## Thanks for reading!
