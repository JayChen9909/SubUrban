import jieba_fast
import numpy as np

from tqdm import tqdm

def ngram_split(text, n=3):
    ngrams = set()
    for k in range(1, n + 1):
        for i in range(len(text) - k + 1):
            ngrams.add(text[i:i + k])
    return ngrams

def hybrid_split(text):
    fields = text.split(',')
    tokens = ngram_split(fields[0], 2)
    tokens.update(ngram_split(fields[1], 2))
    for field in fields:
        tokens.update(jieba_fast.lcut_for_search(field))
    return tokens

def hybrid_split_with_count(text):
    fields = text.split(',')
    tokens = {}
    for k in range(1, 3):
        for i in range(len(text) - k + 1):
            token = text[i:i + k]
            if token not in tokens:
                tokens[token] = 0
            tokens[token] += 1
    tokens_jieba = {}
    for field in fields:
        for token in jieba_fast.lcut_for_search(field):
            if token not in tokens_jieba:
                tokens_jieba[token] = 0
            tokens_jieba[token] += 1
    for token, count in tokens_jieba.items():
        if token not in tokens:
            tokens[token] = count
        else:
            tokens[token] = max(tokens[token], count)
    return tokens

def query_split(text):
    tokens = ngram_split(text, 2)
    tokens.update(jieba_fast.lcut_for_search(text))
    return tokens

def load_query_txt(query_path):
    query_txt = []
    query_locations = []
    query_truth = []
    with open(query_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Loading query data'):
            line = line.strip().split('\t')
            query_txt.append(line[0])
            query_utm_lat = float(line[1])
            query_utm_lon = float(line[2])
            query_locations.append([query_utm_lat, query_utm_lon])
            query_truth_str = line[3]
            query_truth_split = query_truth_str.split(',')
            query_truth.append([int(x) for x in query_truth_split])
    query_locations = np.array(query_locations)
    return query_txt, query_locations, query_truth

def load_poi_txt(poi_path):
    poi_txt = []
    poi_locations = []
    min_x = 1e9
    max_x = -1e9
    min_y = 1e9
    max_y = -1e9
    with open(poi_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Loading POI data'):
            line = line.strip().split('\t')
            poi_txt.append(line[0])
            poi_utm_lat = float(line[1])
            poi_utm_lon = float(line[2])
            poi_locations.append([poi_utm_lat, poi_utm_lon])
            min_x = min(min_x, poi_utm_lat)
            max_x = max(max_x, poi_utm_lat)
            min_y = min(min_y, poi_utm_lon)
            max_y = max(max_y, poi_utm_lon)

    poi_locations = np.array(poi_locations)
    d_norm = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    return poi_txt, poi_locations, d_norm


if __name__ == '__main__':
    # we split the pois into words and save the hash values for poi words
    import os
    import argparse
    from util_hash import HASH2x4096
    from collections import OrderedDict
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--city', type=str, default='Shanghai', help='city name')

    args = argparser.parse_args()
    city = args.city

    poi_path = f'data/projected/{city}/poi.txt'

    # Load the POI data
    poi_txt, poi_locations, d_norm = load_poi_txt(poi_path)
    print(f'Loaded {len(poi_txt)} POIs from {poi_path}')
    word_hashes = OrderedDict()

    for poi in tqdm(poi_txt, desc='Tokenizing POIs'):
        poi_hash = np.zeros(8192, dtype=np.int32)
        for token in hybrid_split(poi):
            if token not in word_hashes:
                word_hashes[token] = HASH2x4096(token)


    # Save the POI hashes
    word_hash_path = f'model/cache/{city}/word_hashes.txt'
    # make sure the directory exists
    os.makedirs(os.path.dirname(word_hash_path), exist_ok=True)
    with open(word_hash_path, 'w') as f:
        for token, hash_values in word_hashes.items():
            f.write(f'{token}\t')
            # join the hash values with a comma
            f.write(','.join(str(x) for x in hash_values))
            f.write('\n')
