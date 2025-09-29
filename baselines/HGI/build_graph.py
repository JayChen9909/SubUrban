import numpy as np
from collections import OrderedDict, defaultdict
from scipy.spatial import Delaunay
from tqdm import tqdm
import random
import argparse
import os

def get_suburban_dir(*paths):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *paths)

def load_data(city):
    if city in ['Beijing', 'Shanghai']:
        dataset = 'Gaode'
    elif city in ['Singapore', 'NYC']:
        dataset = 'OSM'
    else:
        raise ValueError("Unsupported city.")
    input_path = get_suburban_dir('data', dataset, 'projected', city, 'poi.txt')
    poi_categories, coordinates, indices, has_category = label_generator(city, input_path)
    return poi_categories, coordinates, indices, has_category, dataset

def label_generator(city, input_path):
    poi_categories = []
    coordinates = []
    indices = []
    has_category = False
    category2idx = OrderedDict()
    with open(input_path, 'r', encoding='utf-8') as file:
        for idx, poi_line in enumerate(file):
            poi_line = poi_line.strip()
            fields = poi_line.split('\t')
            assert len(fields) == 3
            components = fields[0].split(',')
            if len(components) > 1:
                has_category = True
                if city == 'Hangzhou':
                    category = components[1]
                else:
                    category = components[-2]
                if category not in category2idx:
                    category2idx[category] = len(category2idx)
                poi_categories.append(category2idx[category])
            else:
                poi_categories.append(None)
            coordinates.append((float(fields[1]), float(fields[2])))
            indices.append(idx)
    return poi_categories, coordinates, indices, has_category

def delaunay_triangulation(coordinates):
    try:
        tri = Delaunay(coordinates)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add((simplex[i], simplex[j]))
        return list(edges)
    except Exception as e:
        print(f"Delaunay triangulation failed: {e}")
        return []

def create_adj_lists(num_nodes, edges):
    adj_lists = defaultdict(set)
    for edge in edges:
        node1, node2 = edge
        adj_lists[node1].add(node2)
        adj_lists[node2].add(node1)
    isolated_nodes = [node for node in range(num_nodes) if node not in adj_lists]
    return adj_lists, isolated_nodes

def adjust_coordinates(coordinates, isolated_nodes):
    for node in isolated_nodes:
        dx = random.uniform(1, 3)
        dy = random.uniform(1, 3)
        coordinates[node] = (coordinates[node][0] + dx, coordinates[node][1] + dy)
    return coordinates

def compute_edge_weight(coordinates, edges):
    edge_weights = []
    for edge in edges:
        coord1 = np.array(coordinates[edge[0]])
        coord2 = np.array(coordinates[edge[1]])
        distance = np.linalg.norm(coord1 - coord2)
        edge_weights.append(distance)
    return np.array(edge_weights)

def check_isolated_nodes(edge_index, num_nodes):
    degrees = np.zeros(num_nodes)
    for i in range(edge_index.shape[1]):
        degrees[edge_index[0, i]] += 1
        degrees[edge_index[1, i]] += 1
    isolated_nodes = np.where(degrees == 0)[0]
    if len(isolated_nodes) > 0:
        print(f"Warning: {len(isolated_nodes)} isolated nodes found: {isolated_nodes}")
    else:
        print("No isolated nodes found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True, help='City name (e.g., Beijing, Shanghai, Singapore, NYC)')
    args = parser.parse_args()
    city = args.city

    poi_categories, coordinates, indices, has_category, dataset = load_data(city)
    print(f'Loaded {len(poi_categories)} POIs from {city}')

    edges = delaunay_triangulation(coordinates)
    adj_lists, isolated_nodes = create_adj_lists(len(coordinates), edges)
    while isolated_nodes:
        coordinates = adjust_coordinates(coordinates, isolated_nodes)
        edges = delaunay_triangulation(coordinates)
        adj_lists, isolated_nodes = create_adj_lists(len(coordinates), edges)

    final_edges = []
    for node, neighbors in adj_lists.items():
        for neighbor in neighbors:
            final_edges.append((node, neighbor))

    edge_indices = []
    for node, neighbors in adj_lists.items():
        for neighbor in neighbors:
            edge_indices.append((indices[node], indices[neighbor]))

    num_nodes = len(coordinates)
    check_isolated_nodes(np.array(edge_indices).T, num_nodes)

    edge_weights = compute_edge_weight(coordinates, final_edges)

    graph_data = {
        'indices': indices,
        'edge': final_edges,
        'edge_index': edge_indices,
        'edge_weight': edge_weights
    }
    if has_category:
        graph_data['label'] = np.array(poi_categories).reshape(-1, 1)

    out_dir = get_suburban_dir('data', dataset, 'graph')
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'{city}_graph_data.npy')
    np.save(output_path, graph_data)
    print(f'Saved graph data for {city} to {output_path}')

if __name__ == "__main__":
    main()