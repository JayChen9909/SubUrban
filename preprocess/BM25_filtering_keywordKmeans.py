import jieba_fast
import os
import numpy as np
import argparse
import pickle as pkl

from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from scipy.spatial import cKDTree

def get_suburban_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

CITY_CONFIGS = {
    "Beijing": {
        "dataset": "Gaode",
        "use_chinese": True,
        "district_column": 2,
        "special_handling": {"亦庄开发区": "大兴区"}
    },
    "Shanghai": {
        "dataset": "Gaode", 
        "use_chinese": True,
        "district_column": 2,
        "special_handling": {}
    },
    "Singapore": {
        "dataset": "OSM",
        "use_chinese": False,
        "district_column": -1,
        "special_handling": {}
    },
    "NYC": {
        "dataset": "OSM",
        "use_chinese": False,
        "district_column": -1,
        "special_handling": {}
    }
}

def bm25_search_chinese(query_district, description_text, poi_txt, poi_lines, num_results, city_config):
    filtered_indices = []
    special_handling = city_config["special_handling"]
    
    for i, poi_line in enumerate(poi_lines):
        first_col = poi_line[0]
        comma_split = first_col.split(',')
        if len(comma_split) > 2:
            district_name = comma_split[2]
            if "亦庄开发区" in district_name:
                district_name = "大兴区"
            if query_district == "亦庄开发区" and "亦庄开发区" in comma_split[2]:
                filtered_indices.append(i)
            elif query_district in district_name:
                filtered_indices.append(i)
    
    if not filtered_indices:  
        return [], []
    
    filtered_poi_txt = [poi_txt[i] for i in filtered_indices]
    filtered_bm25 = BM25Okapi(filtered_poi_txt, k1=0.3, b=0.1)

    tokenized_query = jieba_fast.lcut_for_search(description_text)
    doc_scores = filtered_bm25.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:num_results]
    remaining_indices = np.argsort(doc_scores)[len(top_indices):]
    return [filtered_indices[i] for i in top_indices], [filtered_indices[i] for i in remaining_indices]

def bm25_search_english(query_area, description_text, poi_txt, poi_lines, num_results, city):
    filtered_indices = []
    
    for i, poi_line in enumerate(poi_lines):
        first_col = poi_line[0]
        comma_split = first_col.split(',')
        
        if city == 'Singapore':
            if len(comma_split) >= 3:
                poi_area = comma_split[-1].strip()
                if query_area.upper() == poi_area.upper():
                    filtered_indices.append(i)
        
        elif city == 'NYC':
            if len(comma_split) >= 4:
                poi_borough = comma_split[-1].strip()
                if query_area.upper() == poi_borough.upper():
                    filtered_indices.append(i)
        
        else:
            if len(comma_split) >= 3:
                poi_area = comma_split[-1].strip()
                if query_area.upper() == poi_area.upper():
                    filtered_indices.append(i)
    
    if not filtered_indices:
        return []
    
    filtered_poi_txt = [poi_txt[i] for i in filtered_indices]
    filtered_bm25 = BM25Okapi(filtered_poi_txt, k1=0.3, b=0.1)
    
    tokenized_query = jieba_fast.lcut_for_search(description_text)
    doc_scores = filtered_bm25.get_scores(tokenized_query)
    
    top_indices = np.argsort(doc_scores)[::-1][:num_results]
    
    return [filtered_indices[i] for i in top_indices]

parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='Beijing')
parser.add_argument('--version', type=str, default='keywords_kmeans')
parser.add_argument('--top_k', type=int, default=8000)
parser.add_argument('--drop', type=str, choices=['BM25','random'], default='BM25')

args = parser.parse_args()
city = args.city
num_results = args.top_k
drop = args.drop
version = args.version

if city not in CITY_CONFIGS:
    print(f"Error: Unsupported city '{city}'. Available cities: {list(CITY_CONFIGS.keys())}")
    exit(1)

city_config = CITY_CONFIGS[city]
suburban_dir = get_suburban_dir()
dataset = city_config["dataset"]

processed_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city)
processed_poi_file = os.path.join(processed_path, 'poi.txt')
processed_query_file = os.path.join(processed_path, f'district_desc_{version}.txt')

output_dir = os.path.join(suburban_dir, 'tmp', 'BM25')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{city}_BM25_top{num_results}.txt')
if city in ['Beijing', 'Shanghai']:
    filtered_poi_path = os.path.join(processed_path, f"poi_{drop}_{version}_{num_results}.txt")
else:
    filtered_poi_path = os.path.join(processed_path, f"poi_{version}_filtered.txt")

tmp_split_poi_txt = os.path.join(suburban_dir, 'tmp', f"poi_split_{city}.pkl")

if os.path.exists(tmp_split_poi_txt):
    with open(tmp_split_poi_txt, 'rb') as f:
        poi_txt = pkl.load(f)
        poi_lines = pkl.load(f) 
else:
    poi_txt = []
    poi_lines = []

    with open(processed_poi_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Splitting POI text'):
            parts = line.strip().split('\t')
            poi_lines.append(parts)
            poi_txt.append(jieba_fast.lcut_for_search(parts[0]))

    with open(tmp_split_poi_txt, 'wb') as f:
        pkl.dump(poi_txt, f)
        pkl.dump(poi_lines, f)

if drop == 'BM25' and version == 'keywords_kmeans':
    queries = []
    with open(processed_query_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Loading query text'):
            query_district, description_text = line.strip().split('\t')
            keywords = [keyword.strip("'") for keyword in description_text.split(',')]
            queries.append((query_district, keywords))

    if city_config["use_chinese"]:
        selected_poi_lines = set()

        with open(output_path, 'w') as f:
            for query_district, keywords in tqdm(queries, desc="Processing K-Means with keywords"):
                combined_filtered_indices = set()

                for keyword in keywords:
                    top_indices, _ = bm25_search_chinese(query_district, keyword, poi_txt, poi_lines, 1000, city_config)
                    combined_filtered_indices.update(top_indices)

                combined_filtered_indices = sorted(combined_filtered_indices)

                coordinates = []
                valid_indices = []
                for idx in combined_filtered_indices:
                    line_parts = poi_lines[idx]
                    try:
                        lon, lat = float(line_parts[1]), float(line_parts[2])
                        coordinates.append((lon, lat))
                        valid_indices.append(idx)
                    except ValueError:
                        continue

                if not coordinates:
                    print(f"No valid coordinates found for {query_district}. Skipping.")
                    continue

                coordinates = np.array(coordinates)

                max_clusters = 200
                k = min(max_clusters, max(1, len(coordinates) // 20))

                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                poi_clusters = kmeans.fit_predict(coordinates)
                centroids = kmeans.cluster_centers_

                representative_indices = []
                num_points_per_cluster = 80

                for cluster_id in range(k):
                    cluster_points = [i for i, cluster in enumerate(poi_clusters) if cluster == cluster_id]
                    cluster_coords = coordinates[cluster_points]
                    centroid = centroids[cluster_id]

                    distances = np.linalg.norm(cluster_coords - centroid, axis=1)
                    average_distance = np.mean(distances[:30])

                    within_average_indices = [i for i, dist in enumerate(distances) if dist <= average_distance]
                    outside_average_indices = [i for i, dist in enumerate(distances) if dist > average_distance]

                    within_average_points = cluster_coords[within_average_indices]
                    within_distances = distances[within_average_indices]

                    if len(within_average_points) >= 5:
                        farthest_within_indices = np.argsort(within_distances)[-5:]
                        within_selected = [cluster_points[within_average_indices[i]] for i in farthest_within_indices]
                    else:
                        within_selected = [cluster_points[within_average_indices[i]] for i in range(len(within_average_points))]

                    outside_average_points = cluster_coords[outside_average_indices]
                    outside_distances = distances[outside_average_indices]

                    needed_outside_points = num_points_per_cluster - len(within_selected)
                    if len(outside_average_points) >= needed_outside_points:
                        closest_outside_indices = np.argsort(np.abs(outside_distances - average_distance))[:needed_outside_points]
                        outside_selected = [cluster_points[outside_average_indices[i]] for i in closest_outside_indices]
                    else:
                        outside_selected = [cluster_points[outside_average_indices[i]] for i in range(len(outside_average_points))]

                    combined_selected = within_selected + outside_selected

                    if len(combined_selected) > num_points_per_cluster:
                        combined_selected = combined_selected[:num_points_per_cluster]

                    for idx in combined_selected:
                        representative_indices.append(valid_indices[idx])

                top_indices_str = ','.join(map(str, representative_indices))
                f.write(f"{query_district}\t{top_indices_str}\n")

                for idx in representative_indices:
                    poi_line = poi_lines[idx]
                    selected_poi_lines.add("\t".join(poi_line) + f"\t{idx}")

    else:
        selected_poi_with_indices = {}
        
        print(f"Processing {city} areas with K-Means clustering...")
        for area_name, keywords in tqdm(queries, desc=f"Processing {city} areas"):
            combined_filtered_indices = set()
            
            for keyword in keywords:
                top_indices = bm25_search_english(area_name, keyword, poi_txt, poi_lines, 100, city)
                combined_filtered_indices.update(top_indices)
            
            combined_filtered_indices = sorted(combined_filtered_indices)
            
            if not combined_filtered_indices:
                print(f"No POIs found for {area_name}")
                continue
            
            coordinates = []
            valid_indices = []
            for idx in combined_filtered_indices:
                line_parts = poi_lines[idx]
                if len(line_parts) >= 3:
                    try:
                        x, y = float(line_parts[1]), float(line_parts[2])
                        coordinates.append((x, y))
                        valid_indices.append(idx)
                    except (ValueError, IndexError):
                        continue
            
            if not coordinates:
                print(f"No valid coordinates found for {area_name}")
                continue
            
            coordinates = np.array(coordinates)
            
            max_clusters = 300
            k = min(max_clusters, max(1, len(coordinates) // 10))
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            poi_clusters = kmeans.fit_predict(coordinates)
            centroids = kmeans.cluster_centers_
            
            representative_indices = []
            points_per_cluster = 100
            
            for cluster_id in range(k):
                cluster_points = [i for i, cluster in enumerate(poi_clusters) if cluster == cluster_id]
                if not cluster_points:
                    continue
                    
                cluster_coords = coordinates[cluster_points]
                centroid = centroids[cluster_id]
                
                distances = np.linalg.norm(cluster_coords - centroid, axis=1)
                average_distance = np.mean(distances[:100])
                
                within_average_indices = [i for i, dist in enumerate(distances) if dist <= average_distance]
                outside_average_indices = [i for i, dist in enumerate(distances) if dist > average_distance]
                
                within_average_points = cluster_coords[within_average_indices]
                within_distances = distances[within_average_indices]
                
                if len(within_average_points) >= 5:
                    farthest_within_indices = np.argsort(within_distances)[-5:]
                    within_selected = [cluster_points[within_average_indices[i]] for i in farthest_within_indices]
                else:
                    within_selected = [cluster_points[within_average_indices[i]] for i in range(len(within_average_points))]
                
                outside_average_points = cluster_coords[outside_average_indices]
                outside_distances = distances[outside_average_indices]
                
                needed_outside_points = points_per_cluster - len(within_selected)
                if len(outside_average_points) >= needed_outside_points:
                    closest_outside_indices = np.argsort(np.abs(outside_distances - average_distance))[:needed_outside_points]
                    outside_selected = [cluster_points[outside_average_indices[i]] for i in closest_outside_indices]
                else:
                    outside_selected = [cluster_points[outside_average_indices[i]] for i in range(len(outside_average_points))]
                
                combined_selected = within_selected + outside_selected
                
                if len(combined_selected) > points_per_cluster:
                    combined_selected = combined_selected[:points_per_cluster]
                
                for idx in combined_selected:
                    representative_indices.append(valid_indices[idx])
            
            print(f"{area_name}: Selected {len(representative_indices)} representative POIs")
            
            for idx in representative_indices:
                poi_line = poi_lines[idx]
                selected_poi_with_indices[idx] = poi_line

        selected_poi_lines = set()
        for original_idx, poi_line in selected_poi_with_indices.items():
            line_with_index = "\t".join(poi_line) + "\t" + str(original_idx)
            selected_poi_lines.add(line_with_index)

elif drop == 'random':
    total_pois = len(poi_lines)
    target_sample_size = 16 * num_results

    if total_pois <= target_sample_size:
        print(f"Total POIs ({total_pois}) are less than or equal to target size ({target_sample_size}). Keeping all data.")
        sampled_indices = list(range(total_pois))
    else:
        np.random.seed(42)
        sampled_indices = np.random.choice(total_pois, target_sample_size, replace=False)

    selected_poi_lines = set()
    for index in sampled_indices:
        poi_line = poi_lines[index]
        selected_poi_lines.add("\t".join(poi_line) + f"\t{index}")

with open(filtered_poi_path, 'w', encoding="utf-8") as filtered_file:
    for line in selected_poi_lines:
        filtered_file.write(line + "\n")

print(f"Filtered POIs saved to {filtered_poi_path}")