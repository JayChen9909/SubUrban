import torch
import numpy as np
import os

from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from torch.utils.data import DataLoader, TensorDataset

def get_suburban_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))  # Go from baselines/BERT to SubUrban root

BATCH_SIZE =  2048

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='Beijing', help='dataset city name from Beijing, Shanghai, Singapore, NYC')
parser.add_argument('--mode', type=str, choices=['original','filtered'],default='original')
parser.add_argument('--dataset', type=str, default='Gaode', choices=['Meituan','Gaode','OSM'], help='dataset name from Gaode, Meituan')
parser.add_argument('--drop', type=str, choices=['BM25','random'], default='BM25')
parser.add_argument('--version', type=str, choices=['keywords_kmeans','cat', 'v3','v4', 'v5','all'], default='keywords_kmeans')
parser.add_argument('--top_k', type=int, default=8000)
dataset = parser.parse_args().dataset
mode = parser.parse_args().mode
city = parser.parse_args().city
version = parser.parse_args().version
drop = parser.parse_args().drop
top_k = parser.parse_args().top_k

# if mode in ['enhanced','mini']:
#     file_name = f'poi_{mode}.txt'
# elif mode == 'filtered':
#     file_name = f'poi_{mode}_v2_mini.txt'
if mode == 'original':
    file_name = 'poi.txt'
    # file_name = f'poi_{drop}_{version}_{top_k}.txt'
elif mode == 'filtered':
    if city in ['Singapore','NYC']:
        file_name = f'poi_keywords_kmeans_filtered.txt'
    else:
        file_name = f'poi_{drop}_{version}_{top_k}.txt'
else:
    print('Invalid input mode')

poi_lines = None
poi_texts = []
suburban_dir = get_suburban_dir()
processed_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city)
poi_path = os.path.join(processed_path, file_name)

with open(poi_path, 'r', encoding='utf-8') as file:
    poi_lines = file.readlines()
    print(f'Loaded {len(poi_lines)} lines from poi.txt')
    for poi_line in poi_lines:
        poi_line = poi_line.strip()
        fields = poi_line.split('\t')
        # assert len(fields) == 3 or 4
        assert len(fields) in (3, 4)
        if dataset == 'Gaode' or dataset == 'Meituan':
            components = fields[0].split(',')
            # poi_texts.append(components[0] + components[1])
            poi_texts.append(components[0] + components[1] + components[2])
            # poi_texts.append(fields[0])
        else:
            # components = fields[0].split(',')
            # poi_texts.append(components[0] + components[1])
            poi_texts.append(fields[0])




# test_queries = []
# with open(f'data/projected/{dataset}/test.txt', 'r', encoding='utf-8') as file:
#     test_lines = file.readlines()
#     print(f'Loaded {len(test_lines)} lines from test.txt')
#     for test_line in test_lines:
#         test_line = test_line.strip()
#         fields = test_line.split('\t')
#         assert len(fields) == 4
#         test_queries.append(fields[0])

embedding_path = os.path.join(suburban_dir, 'embs', 'BERT', city)
if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)

# Compute the embeddings for the POI txt
if city == 'Singapore' or city == 'NYC':
    MODEL_NAME = "bert-base-uncased"
else:  
    MODEL_NAME = "bert-base-chinese"

bert = BertModel.from_pretrained(MODEL_NAME).to('cuda')
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# Function to encode texts using the tokenizer
def encode_texts(texts, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="Text tokenization"):
        encoded_dict = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',   
            truncation=True,       
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])          # [1, max_length]
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)                  # [N, max_length]
    attention_masks = torch.cat(attention_masks, dim=0)      # [N, max_length]
    return input_ids, attention_masks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to generate embeddings
def generate_embeddings(model, dataloader):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Take the embeddings from the last hidden state
            # You might want to experiment with pooling strategies (e.g., mean pooling)
            # Here, we simply take the embedding of the [CLS] token (first token)
            cls_embeddings = mean_pooling(outputs, attention_mask)
            cls_embeddings = cls_embeddings.cpu().numpy()
            embeddings.append(cls_embeddings)

    embeddings = np.vstack(embeddings)
    return embeddings

# Tokenize POI texts and test queries
poi_input_ids, poi_attention_masks = encode_texts(poi_texts, tokenizer)
# test_input_ids, test_attention_masks = encode_texts(test_queries, tokenizer)

# Create DataLoader for POIs and test queries
poi_dataset = TensorDataset(poi_input_ids, poi_attention_masks)
# test_dataset = TensorDataset(test_input_ids, test_attention_masks)

poi_dataloader = DataLoader(poi_dataset, batch_size=BATCH_SIZE)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Generate embeddings
poi_embeddings = generate_embeddings(bert, poi_dataloader)
np.save(os.path.join(embedding_path, f'poi_embeddings_{drop}_{version}.npy' if mode != 'original' else f'poi_embeddings.npy'), poi_embeddings)
# test_embeddings = generate_embeddings(bert, test_dataloader)
# np.save(os.path.join(embedding_path, 'test_embeddings.npy'), test_embeddings)


print("Embeddings generated and saved successfully.")