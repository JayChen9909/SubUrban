import os
import pickle as pkl
import torch
import argparse

def get_suburban_dir(*paths):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *paths)

def city_abbr(city):
    if city == 'Beijing':
        return 'BJ'
    elif city == 'Shanghai':
        return 'SH'
    elif city == 'Hangzhou':
        return 'HZ'
    elif city == 'Singapore':
        return 'SG'
    elif city == 'NYC':
        return 'NYC'
    else:
        raise ValueError("Unsupported city.")

def city_dataset(city):
    if city in ['Beijing', 'Shanghai']:
        return 'Gaode'
    elif city in ['Singapore', 'NYC']:
        return 'OSM'
    else:
        raise ValueError("Unsupported city.")

def load_region_data(city):
    abbr = city_abbr(city)
    dataset = city_dataset(city)
    filepath = get_suburban_dir('data', dataset, 'processed', 'Integral', f'{abbr}_data.pkl')
    region_mapping_filepath = get_suburban_dir('baselines', 'HGI', 'Data', f'{city.lower()}_data.pkl')
    with open(filepath, 'rb') as file:
        region_data = pkl.load(file)
    with open(region_mapping_filepath, 'rb') as file:
        region_mapping = pkl.load(file)['region_mapping']
    return region_data, region_mapping, abbr

def load_region_emb(city):
    emb_path = get_suburban_dir('baselines', 'HGI', 'Emb', f'{city.lower()}_emb')
    embeddings = torch.load(emb_path)
    embedding_dict = {idx: embeddings[idx].tolist() for idx in range(embeddings.shape[0])}
    return embedding_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True, help='City name (e.g., Beijing, Shanghai, Singapore, NYC)')
    args = parser.parse_args()
    city = args.city

    if city in ['Beijing', 'Shanghai']:
        tasks = ['pop', 'house', 'gdp']
    elif city in ['Singapore', 'NYC']:
        tasks = ['pop']
    else:
        raise ValueError("Unsupported city.")

    region_data, region_mapping, abbr = load_region_data(city)
    region_emb = load_region_emb(city)

    for task in tasks:
        save_data = {}
        if task == 'house':
            for idx, region_info in region_data.items():
                if region_info['pois'] and region_info.get('house_price'):
                    remapped_idx = region_mapping[idx]
                    save_data[idx] = {
                        'region_embedding': region_emb[remapped_idx],
                        'house_price': region_info['house_price']
                    }
        elif task == 'pop':
            for idx, region_info in region_data.items():
                if region_info['pois'] and region_info.get('population'):
                    remapped_idx = region_mapping[idx]
                    save_data[idx] = {
                        'region_embedding': region_emb[remapped_idx],
                        'population': region_info['population']
                    }
        elif task == 'gdp':
            for idx, region_info in region_data.items():
                if region_info['pois'] and region_info.get('gdp'):
                    remapped_idx = region_mapping[idx]
                    save_data[idx] = {
                        'region_embedding': region_emb[remapped_idx],
                        'gdp': region_info['gdp']
                    }
        out_dir = get_suburban_dir('embs', 'HGI', city)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'{task}_{abbr}_HGI.pkl')
        if save_data:
            with open(save_path, 'wb') as file:
                pkl.dump(save_data, file)
            print(f"HGI region embeddings for {city} {task} saved to {save_path}")
        else:
            print(f"No data to save for {city} - {task}")

if __name__ == '__main__':
    main()