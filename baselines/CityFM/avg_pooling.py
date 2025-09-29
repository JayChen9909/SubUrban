import numpy as np
import pickle as pkl
import argparse
import torch
import os
from tqdm import tqdm

def get_suburban_dir(*paths):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *paths)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Aggregate poi embeddings from SpaBERT method into region embeddings via average-pooling.')
    parser.add_argument('--city', type=str, required=True, help='City name (e.g., Beijing, Shanghai, Singapore, NYC)')
    parser.add_argument('--integrate', type=str, choices=['average', 'or'], default='average', help='Integration method ("average" or "or")')
    return parser.parse_args()

def load_region_data(city):
    abbr = city_abbr(city)
    if abbr in ['BJ', 'SH']:
        dataset = 'Gaode'
    elif abbr in ['SG', 'NYC']:
        dataset = 'OSM'
    else:
        raise ValueError("Unsupported city. Please choose city among 'Beijing', 'Shanghai', 'Singapore', 'NYC'")
    filepath = get_suburban_dir('data', dataset, 'processed', 'Integral', f'{abbr}_data.pkl')
    with open(filepath, 'rb') as file:
        region_data = pkl.load(file)
    return region_data


def load_poi_emb(city):
    pt_file_path = get_suburban_dir('embs', 'CityFM', city, 'poi_node_embeddings.pt')
    node_embeddings = torch.load(pt_file_path)
    embeddings = node_embeddings.cpu().numpy()
    return embeddings

def city_abbr(city):
    if city == 'Beijing':
        return 'BJ'
    elif city == 'Shanghai':
        return 'SH'
    elif city == 'Singapore':
        return 'SG'
    elif city == 'NYC':
        return 'NYC'
    else:
        raise ValueError('Unsupported city')

def main():
    args = parse_arguments()
    city = args.city
    integrate = args.integrate

    region_data = load_region_data(city)
    embeddings = load_poi_emb(city)
    if city in ['Beijing', 'Shanghai']:
        tasks =  ['pop','house','gdp']
    elif city in ['Singapore', 'NYC']:
        tasks = ['pop']
    for task in tasks:
        save_data = {}
        if task == 'house':
            regions_with_poi_and_house = set()
            for region_id, region_info in region_data.items():
                if region_info['pois'] and region_info.get('house_price'):
                    regions_with_poi_and_house.add(region_id)
            for region_id in tqdm(regions_with_poi_and_house, desc=f"Processing regions with POI and House Price in {city}"):
                region_info = region_data[region_id]
                poi_indices = [poi['index'] for poi in region_info['pois']]
                if poi_indices:
                    poi_embeddings = embeddings[poi_indices]
                    if integrate == 'average':
                        region_embedding = np.mean(poi_embeddings, axis=0)
                    elif integrate == 'or':
                        region_embedding = np.sum(poi_embeddings, axis=0)
                    region_info['region_embedding'] = region_embedding
                else:
                    region_info['region_embedding'] = np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
                save_data[region_id] = {
                    'region_embedding': region_info['region_embedding'],
                    'house_price': region_info['house_price']
                }
        elif task == 'pop':
            regions_with_poi_and_pop = set()
            for region_id, region_info in region_data.items():
                if region_info['pois'] and region_info.get('population'):
                    regions_with_poi_and_pop.add(region_id)
            for region_id in tqdm(regions_with_poi_and_pop, desc=f"Processing regions with POI and Population in {city}"):
                region_info = region_data[region_id]
                poi_indices = [poi['index'] for poi in region_info['pois']]
                if poi_indices:
                    poi_embeddings = embeddings[poi_indices]
                    if integrate == 'average':
                        region_embedding = np.mean(poi_embeddings, axis=0)
                    elif integrate == 'or':
                        region_embedding = np.sum(poi_embeddings, axis=0)
                    region_info['region_embedding'] = region_embedding
                else:
                    region_info['region_embedding'] = np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
                save_data[region_id] = {
                    'region_embedding': region_info['region_embedding'],
                    'population': region_info['population']
                }
        elif task == 'gdp':
            regions_with_poi_and_gdp = set()
            for region_id, region_info in region_data.items():
                if region_info['pois'] and region_info.get('gdp'):
                    regions_with_poi_and_gdp.add(region_id)
            for region_id in tqdm(regions_with_poi_and_gdp, desc=f"Processing regions with POI and GDP in {city}"):
                region_info = region_data[region_id]
                poi_indices = [poi['index'] for poi in region_info['pois']]
                if poi_indices:
                    poi_embeddings = embeddings[poi_indices]
                    if integrate == 'average':
                        region_embedding = np.mean(poi_embeddings, axis=0)
                    elif integrate == 'or':
                        region_embedding = np.sum(poi_embeddings, axis=0)
                    region_info['region_embedding'] = region_embedding
                else:
                    region_info['region_embedding'] = np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
                save_data[region_id] = {
                    'region_embedding': region_info['region_embedding'],
                    'gdp': region_info['gdp']
                }
        out_dir = get_suburban_dir('embs', 'CityFM', city)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'{task}_{city_abbr(city)}_CityFM.pkl')
        with open(save_path, 'wb') as f:
            pkl.dump(save_data, f)
        print(f"Region embeddings for {city} {task} saved to {save_path}")

if __name__ == '__main__':
    main()