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
    return os.path.dirname(script_dir)  # Go from preprocess folder to SubUrban root

def bm25_search(query_district, description_text, num_results):
    filtered_indices = []
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
        return []
    
    filtered_poi_txt = [poi_txt[i] for i in filtered_indices]
    filtered_bm25 = BM25Okapi(filtered_poi_txt, k1=0.3, b=0.1)

    tokenized_query = jieba_fast.lcut_for_search(description_text)
    doc_scores = filtered_bm25.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:num_results]
    remaining_indices = np.argsort(doc_scores)[len(top_indices):]
    return [filtered_indices[i] for i in top_indices], [filtered_indices[i] for i in remaining_indices]

parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='Beijing')
parser.add_argument('--version', type=str, default='keywords_kmeans')
parser.add_argument('--dataset', type=str, default='Gaode', choices=['Gaode', 'Meituan'])
parser.add_argument('--top_k', type=int, default=8000)
parser.add_argument('--drop', type=str, choices=['BM25','random'], default='BM25')

city = parser.parse_args().city
dataset = parser.parse_args().dataset
num_results = parser.parse_args().top_k
drop = parser.parse_args().drop
version = parser.parse_args().version

suburban_dir = get_suburban_dir()
processed_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city)
processed_poi_file = os.path.join(processed_path, 'poi.txt')
processed_query_file = os.path.join(processed_path, f'district_desc_{version}.txt')

output_dir = os.path.join(suburban_dir, 'tmp', 'BM25')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{city}_BM25_top{num_results}.txt')
filtered_poi_path = os.path.join(processed_path, f"poi_{drop}_{version}_{num_results}.txt")

tmp_split_poi_txt = os.path.join(suburban_dir, 'tmp', f"poi_split_{city}.pkl")

if os.path.exists(tmp_split_poi_txt):
    with open(tmp_split_poi_txt, 'rb') as f:
        poi_txt = pkl.load(f)
        poi_lines = pkl.load(f) 
else:
    poi_txt = []
    poi_lines = []

    with open(processed_poi_file, 'r') as f:
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
    with open(processed_query_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Loading query text'):
            query_district, description_text = line.strip().split('\t')
            keywords = [keyword.strip("'") for keyword in description_text.split(',')]
            queries.append((query_district, keywords))

    selected_poi_lines = set()

    with open(output_path, 'w') as f:
        for query_district, keywords in tqdm(queries, desc="Processing K-Means with keywords"):
            combined_filtered_indices = set()

            for keyword in keywords:
                # Step 1: Use BM25 to get top k results for each keyword
                top_indices, _ = bm25_search(query_district, keyword, 1000)
                # Add the filtered indices for this keyword to the combined set
                combined_filtered_indices.update(top_indices)

            # Convert the combined set to a sorted list
            combined_filtered_indices = sorted(combined_filtered_indices)

            # Step 2: Extract coordinates for clustering
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

            # Step 3: Apply K-Means clustering
            max_clusters = 200
            k = min(max_clusters, max(1, len(coordinates) // 20))  # Dynamically adjust clusters based on density

            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            poi_clusters = kmeans.fit_predict(coordinates)
            centroids = kmeans.cluster_centers_

            # Step 4: Select representative POIs from each cluster
            representative_indices = []
            num_points_per_cluster = 80  # Total points per cluster

            for cluster_id in range(k):
                cluster_points = [i for i, cluster in enumerate(poi_clusters) if cluster == cluster_id]
                cluster_coords = coordinates[cluster_points]
                centroid = centroids[cluster_id]

                # Calculate distances from points to centroid
                distances = np.linalg.norm(cluster_coords - centroid, axis=1)
                average_distance = np.mean(distances[:30])  # Calculate average distance of 30 closest points

                # Split points into two groups based on the average distance
                within_average_indices = [i for i, dist in enumerate(distances) if dist <= average_distance]
                outside_average_indices = [i for i, dist in enumerate(distances) if dist > average_distance]

                # Select points from within_average group
                within_average_points = cluster_coords[within_average_indices]
                within_distances = distances[within_average_indices]

                if len(within_average_points) >= 5:
                    # Retain the 5 farthest points from the centroid
                    farthest_within_indices = np.argsort(within_distances)[-5:]
                    within_selected = [cluster_points[within_average_indices[i]] for i in farthest_within_indices]
                else:
                    # Retain all points if fewer than 5 are available
                    within_selected = [cluster_points[within_average_indices[i]] for i in range(len(within_average_points))]

                # Select points from outside_average group
                outside_average_points = cluster_coords[outside_average_indices]
                outside_distances = distances[outside_average_indices]

                needed_outside_points = num_points_per_cluster - len(within_selected)
                if len(outside_average_points) >= needed_outside_points:
                    # Retain the points closest to the average distance
                    closest_outside_indices = np.argsort(np.abs(outside_distances - average_distance))[:needed_outside_points]
                    outside_selected = [cluster_points[outside_average_indices[i]] for i in closest_outside_indices]
                else:
                    # Retain all points if fewer than needed are available
                    outside_selected = [cluster_points[outside_average_indices[i]] for i in range(len(outside_average_points))]

                # Combine selected points
                combined_selected = within_selected + outside_selected

                # Ensure total points do not exceed the limit
                if len(combined_selected) > num_points_per_cluster:
                    combined_selected = combined_selected[:num_points_per_cluster]

                # Add to representative indices
                for idx in combined_selected:
                    representative_indices.append(valid_indices[idx])

            # Step 5: Write results
            top_indices_str = ','.join(map(str, representative_indices))
            f.write(f"{query_district}\t{top_indices_str}\n")

            for idx in representative_indices:
                poi_line = poi_lines[idx]
                selected_poi_lines.add("\t".join(poi_line) + f"\t{idx}")

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

# Save filtered POIs to file
with open(filtered_poi_path, 'w', encoding="utf-8") as filtered_file:
    for line in selected_poi_lines:
        filtered_file.write(line + "\n")

print(f"Filtered POIs saved to {filtered_poi_path}")