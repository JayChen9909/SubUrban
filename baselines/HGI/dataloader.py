import pickle
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import argparse
import os

def get_suburban_dir(*paths):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *paths)

def load_file(integral_file_path, emb_file_path, graph_file_path):
    with open(integral_file_path, 'rb') as file:
        data = pickle.load(file)
    emb = np.load(emb_file_path)
    graph = np.load(graph_file_path, allow_pickle=True).item()
    return data, emb, graph

def create_poi_index_to_row_map(poi_index):
    return {poi_idx: row for row, poi_idx in enumerate(poi_index)}

def load_region(data):
    if 'region_shape' in data[list(data.keys())[0]]:
        region_area = []
        region_shapes = []
        region_mapping = {}
        current_region_idx = 0
        for idx, region_info in data.items():
            region_shape = region_info['region_shape']
            polygon = Polygon(region_shape)
            area = polygon.area
            region_area.append(area)
            region_shapes.append(polygon)
            region_mapping[idx] = current_region_idx
            current_region_idx += 1
        region_area = np.array(region_area)
        total_area = region_area.sum()
        region_ratios = region_area / total_area
        n_regions = len(region_shapes)
        adj_list = []
        for i in tqdm(range(n_regions), desc="Generating adjacency matrix"):
            for j in range(i + 1, n_regions):
                if region_shapes[i].touches(region_shapes[j]):
                    adj_list.append([i, j])
                    adj_list.append([j, i])
                elif region_shapes[i].intersects(region_shapes[j]):
                    adj_list.append([i, j])
                    adj_list.append([j, i])
                elif region_shapes[i].centroid.distance(region_shapes[j].centroid) < 50:
                    adj_list.append([i, j])
                    adj_list.append([j, i])
        adj_list = np.array(adj_list).T
        region_id = []
        for idx, region_info in data.items():
            pois = region_info['pois']
            mapped_region_id = region_mapping[idx]
            region_id.extend([mapped_region_id] * len(pois))
        region_id = np.array(region_id)
    else:
        region_ratios, adj_list, region_id, region_mapping = None, None, None, None
    return region_ratios, adj_list, region_id, region_mapping

def load_emb(data, emb):
    poi_embeddings = []
    poi_index = []
    for idx, region_info in data.items():
        poi_indices = [poi['index'] for poi in region_info['pois']]
        valid_poi_indices = [poi_idx for poi_idx in poi_indices if poi_idx < emb.shape[0]]
        if valid_poi_indices:
            poi_embedding = emb[valid_poi_indices]
            poi_embeddings.append(poi_embedding)
            poi_index.extend(valid_poi_indices)
    poi_index = np.array(poi_index)
    poi_embeddings = np.vstack(poi_embeddings)
    return poi_embeddings, poi_index

def remap_edge_index(edge_index, poi_index_to_row):
    remapped_edge_index = np.array([[poi_index_to_row.get(idx, -1) for idx in edge] for edge in edge_index.T]).T
    valid_mask = (remapped_edge_index >= 0).all(axis=0)
    remapped_edge_index = remapped_edge_index[:, valid_mask]
    return remapped_edge_index

def load_edge(graph, poi_index):
    edge_index = np.array(graph['edge_index']).T
    edge_weight = np.array(graph['edge_weight']) + 1e-8
    poi_index_to_row = create_poi_index_to_row_map(poi_index)
    remapped_edge_index = []
    remapped_edge_weight = []
    for i, edge in enumerate(edge_index.T):
        if edge[0] in poi_index_to_row and edge[1] in poi_index_to_row:
            remapped_edge_index.append([poi_index_to_row[edge[0]], poi_index_to_row[edge[1]]])
            remapped_edge_weight.append(edge_weight[i])
    remapped_edge_index = np.array(remapped_edge_index).T
    remapped_edge_weight = np.array(remapped_edge_weight)
    return remapped_edge_index, remapped_edge_weight

def generate_coarse_region_similarity(region_area):
    n_regions = region_area.shape[0]
    coarse_region_similarity = np.random.rand(n_regions, n_regions)
    return coarse_region_similarity

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True, help='City name (e.g., Beijing, Shanghai, Singapore, NYC)')
    args = parser.parse_args()
    city = args.city
    city_abbr_ = city_abbr(city)
    dataset = city_dataset(city)

    integral_file_path = get_suburban_dir('data', dataset, 'processed', 'Integral', f'{city_abbr_}_data.pkl')
    emb_file_path = get_suburban_dir('embs', 'BERT', city, 'poi_embeddings.npy')
    graph_file_path = get_suburban_dir('data', dataset, 'graph', f'{city}_graph_data.npy')

    data = {
        'city': city.lower(),
        'node_features': None,
        'edge_index': None,
        'edge_weight': None,
        'region_id': None,
        'region_area': None,
        'coarse_region_similarity': None,
        'region_adjacency': None,
        'region_mapping': None
    }

    dict_, emb, graph = load_file(integral_file_path, emb_file_path, graph_file_path)
    x, poi_index = load_emb(dict_, emb)
    edge_index, edge_weight = load_edge(graph, poi_index)
    region_area, region_adjacency, region_id, region_mapping = load_region(dict_)
    coarse_region_similarity = generate_coarse_region_similarity(region_area)

    data['node_features'] = x
    data['edge_index'] = edge_index
    data['edge_weight'] = edge_weight
    data['region_id'] = region_id
    data['region_area'] = region_area
    data['coarse_region_similarity'] = coarse_region_similarity
    data['region_adjacency'] = region_adjacency
    data['region_mapping'] = region_mapping

    out_dir = get_suburban_dir('baselines', 'HGI', 'Data')
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f'{city.lower()}_data.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {file_path}")

if __name__ == '__main__':
    main()