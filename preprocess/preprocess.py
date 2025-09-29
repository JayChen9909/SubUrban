from collections import OrderedDict
import os
import pickle as pkl
import numpy as np
import math
import geopandas as gpd
import pandas as pd
import argparse
import sys
import json
import pyproj
from functools import partial

from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import transform

def get_suburban_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

# Fix for shapely.io compatibility issue with old pickle files
try:
    import shapely.io
except ImportError:
    import shapely
    import types
    from shapely import wkb, wkt
    
    shapely.io = types.ModuleType('shapely.io')
    shapely.io.from_wkb = wkb.loads
    shapely.io.from_wkt = wkt.loads
    shapely.io.to_wkb = wkb.dumps
    shapely.io.to_wkt = wkt.dumps
    sys.modules['shapely.io'] = shapely.io

tqdm.pandas()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data for SparseRegion downstream tasks.')
    parser.add_argument('--city', type=str, choices=['Beijing', 'Shanghai', 'Singapore', 'NYC'], 
                       default='Beijing', help='City: "Beijing", "Shanghai", "Singapore", or "NYC"')
    parser.add_argument('--near_assign', action='store_true', help='Whether to assign those unassigned POIs to the nearest road region')
    parser.add_argument('--poi_mode', type=str, choices=['original', 'filtered'], default='original', help='Mode for loading poi.txt dataset')
    parser.add_argument('--version', type=str, default='keywords_kmeans', help='Version of the filtered poi data')
    # parser.add_argument('--dataset', type=str, default='Gaode', choices=['Gaode','OSM'], help='Dataset to use')
    parser.add_argument('--top_k', type=int, default=8000, help='Number of top k pois to use')
    parser.add_argument('--drop', type=str, choices=['BM25','random'], default='BM25', help='Drop method for poi filtering')
    return parser.parse_args()

def load_poi(city, poi_mode, version, dataset, top_k, drop):
    """Load POI data for all supported cities"""
    suburban_dir = get_suburban_dir()
    poi_loc = []
    poi_index = []
    poi_text = []
    
    if poi_mode == 'original':
        file_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city, 'poi.txt')
    elif poi_mode == 'filtered':
        if city in ['Singapore', 'NYC']:
            file_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city, f'poi_{version}_filtered.txt')
        else:  # Beijing, Shanghai
            file_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city, f'poi_{drop}_{version}_{top_k}.txt')

    with open(file_path, 'r') as f:
        lines = f.readlines()
        print(f"Number of POIs in {poi_mode} txt file: {len(lines)}")
        
        for line_idx, line in enumerate(tqdm(lines)):
            fields = line.strip().split('\t')
            
            if city in ['Singapore', 'NYC']:
                # Singapore/NYC format: poi_text, longitude, latitude, [original_index]
                poi_loc.append([float(fields[1]), float(fields[2])])
                poi_text.append(fields[0])
                if poi_mode == 'filtered':
                    if len(fields) > 3:
                        try:
                            poi_index.append(int(fields[3]))
                        except ValueError:
                            poi_index.append(line_idx)
                    else:
                        poi_index.append(line_idx)
                else:
                    poi_index.append(line_idx)
            else:
                # Beijing/Shanghai format: poi_text, latitude, longitude, [original_index]
                poi_loc.append([float(fields[2]), float(fields[1])])
                poi_text.append(fields[0])
                if poi_mode == 'filtered':
                    try:
                        poi_index.append(int(fields[-1]))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Index parsing failed: {e}, line: {line.strip()}")
                        poi_index.append(None)
                else:
                    poi_index.append(None)
    
    print(f"Loaded {len(poi_loc)} POIs from {file_path}")
    return poi_text, poi_loc, poi_index, file_path

def load_original_poi_for_alignment(city, dataset='OSM'):
    """Load original POI data for alignment when in filtered mode"""
    suburban_dir = get_suburban_dir()
    original_poi_loc = []
    original_poi_text = []
    original_poi_index = []
    
    if city in ['Singapore', 'NYC']:
        original_file_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city, 'poi.txt')
    else:
        original_file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'projected', city, 'poi.txt')
    
    with open(original_file_path, 'r') as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            fields = line.strip().split('\t')
            if city in ['Singapore', 'NYC']:
                original_poi_loc.append([float(fields[1]), float(fields[2])])
            else:
                original_poi_loc.append([float(fields[2]), float(fields[1])])
            original_poi_text.append(fields[0])
            original_poi_index.append(line_idx)
    
    print(f"Loaded {len(original_poi_loc)} original POIs for alignment")
    return original_poi_text, original_poi_loc, original_poi_index

def load_and_project_subzones(city, region_file):
    """Load and project subzones for Singapore/NYC"""
    if city == 'NYC':
        # Load NYC regions from pickle file
        with open(region_file, 'rb') as f:
            regions = pkl.load(f)
        
        subzones = {}
        valid_count = 0
        invalid_count = 0
        
        # Check if the data is already in UTM or needs projection
        sample_polygon = None
        if isinstance(regions, list) and len(regions) > 0:
            sample_polygon = regions[0]['shape']
        elif isinstance(regions, dict) and len(regions) > 0:
            sample_polygon = next(iter(regions.values()))['shape']
        
        # Check coordinate range to determine coordinate system
        needs_projection = False
        project = None
        if sample_polygon:
            bounds = sample_polygon.bounds
            print(f"Sample polygon bounds: {bounds}")
            
            # If coordinates are in typical lat/lon range (-180 to 180, -90 to 90), it's WGS84
            if abs(bounds[0]) <= 180 and abs(bounds[1]) <= 90 and abs(bounds[2]) <= 180 and abs(bounds[3]) <= 90:
                needs_projection = True
                print("Data appears to be in WGS84, will project to UTM")
                wgs84 = pyproj.CRS("EPSG:4326")
                utm_proj = pyproj.CRS("EPSG:32618")  # NYC UTM zone 18N
                project = partial(pyproj.Transformer.from_crs(wgs84, utm_proj, always_xy=True).transform)
            
            # Check if coordinates are in State Plane range (NYC area: X ~290k-310k, Y ~50k-70k)
            elif (290000 <= bounds[0] <= 310000 and 50000 <= bounds[1] <= 70000 and 
                  290000 <= bounds[2] <= 310000 and 50000 <= bounds[3] <= 70000):
                needs_projection = True
                print("Data appears to be in NAD83 UTM coordinates, will project to WGS84 UTM")
                nad83_utm = pyproj.CRS("EPSG:32118")
                wgs84_utm = pyproj.CRS("EPSG:32618")
                project = partial(pyproj.Transformer.from_crs(nad83_utm, wgs84_utm, always_xy=True).transform)
            
            # Check if coordinates are already in UTM range (NYC area: X ~580k-590k, Y ~4.5M-4.6M)
            elif (580000 <= bounds[0] <= 600000 and 4500000 <= bounds[1] <= 4600000):
                needs_projection = False
                print("Data appears to be already in UTM coordinates, will use as-is")
            
            else:
                print("Warning: Coordinate system not recognized, will use as-is")
                needs_projection = False
        
        if isinstance(regions, list):
            for idx, data in enumerate(regions):
                region_id = data.get('name', str(idx))
                try:
                    region_id = int(region_id)
                except ValueError:
                    region_id = idx
                
                polygon = data['shape']
                
                try:
                    if polygon.geom_type == 'Polygon' and polygon.is_valid:
                        if needs_projection:
                            projected_polygon = transform(project, polygon)
                        else:
                            projected_polygon = polygon
                        
                        if projected_polygon.is_valid and not projected_polygon.is_empty:
                            subzones[region_id] = {
                                'geometry': projected_polygon,
                                'population': data['population']
                            }
                            valid_count += 1
                        else:
                            invalid_count += 1
                    else:
                        invalid_count += 1
                except Exception as e:
                    print(f"Error processing region {region_id}: {e}")
                    invalid_count += 1
                    
        elif isinstance(regions, dict):
            for region_id, data in regions.items():
                try:
                    region_id = int(region_id)
                except ValueError:
                    pass
                
                polygon = data['shape']
                
                try:
                    if polygon.geom_type == 'Polygon' and polygon.is_valid:
                        if needs_projection:
                            projected_polygon = transform(project, polygon)
                        else:
                            projected_polygon = polygon
                        
                        if projected_polygon.is_valid and not projected_polygon.is_empty:
                            subzones[region_id] = {
                                'geometry': projected_polygon,
                                'population': data['population']
                            }
                            valid_count += 1
                        else:
                            print(f"Invalid final polygon for region {region_id}")
                            invalid_count += 1
                    else:
                        print(f"Invalid input polygon for region {region_id}: type={polygon.geom_type}, valid={polygon.is_valid}")
                        invalid_count += 1
                except Exception as e:
                    print(f"Error processing region {region_id}: {e}")
                    invalid_count += 1
        
        print(f"Loaded {valid_count} valid and {invalid_count} invalid NYC subzones from {region_file}")
        return subzones
    
    elif city == 'Singapore':
        # Load Singapore subzones from JSON file
        with open(region_file, 'r') as f:
            regions = json.load(f)

        # Define UTM projection for Singapore
        wgs84 = pyproj.CRS("EPSG:4326")
        utm_proj = pyproj.CRS("EPSG:32648")  # Singapore UTM zone 48N
        project = partial(pyproj.Transformer.from_crs(wgs84, utm_proj, always_xy=True).transform)

        subzones = {}
        for region_id, data in regions.items():
            # Convert polygon to UTM coordinates
            polygon = transform(project, Polygon(data['polygon']))
            subzones[int(region_id)] = {
                'geometry': polygon,
                'area': data['area'],
                'population': data['population']
            }
        print(f"Loaded and projected {len(subzones)} Singapore subzones from {region_file}")
        return subzones
    
    else:
        # Beijing/Shanghai - return None as they use different region loading
        return None

def load_region_geometry(city):
    """Load Region Geometry for Beijing/Shanghai"""
    suburban_dir = get_suburban_dir()
    with open(os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'region', f'{city.lower()}_geometry.pkl'), 'rb') as f:
        landuse_regions = pkl.load(f)

    with open(os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'region', f'{city.lower()}_road_geometry.pkl'), 'rb') as f:
        road_regions = pkl.load(f)

    # Compute area for each region using geopandas
    gdf = gpd.GeoDataFrame(road_regions, columns=['geometry'])
    gdf['area'] = gdf['geometry'].area

    # Create a dictionary to return area along with the regions
    area_dict = gdf['area'].to_dict()
    return road_regions, area_dict

def load_pop(city):
    """Load population data"""
    suburban_dir = get_suburban_dir()
    if city == 'Beijing':
        file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'population', 'beijing_pop.csv')
        data = pd.read_csv(file_path)
    elif city == 'Shanghai':
        file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'population', 'shanghai_pop.csv')
        data = pd.read_csv(file_path)
    else:
        print(f'No population data available for {city}')
        return None

    return data

def load_house(city):
    """Load house price data for Beijing/Shanghai"""
    suburban_dir = get_suburban_dir()
    file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'house', f'{city}_house_data.txt')
    if not os.path.exists(file_path):
        print(f"House data file not found: {file_path}")
        return [], []
    
    house_loc = []
    house_price = []
    with open(file_path, 'r') as f:
        next(f)  # Skip the header line
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Loading house data for {city}"):
            fields = line.strip().split('\t')
            house_loc.append([float(fields[3]), float(fields[2])]) # (lon, lat) -> (x, y)
            house_price.append(float(fields[4]))

    return house_loc, house_price

def load_gdp(city):
    """Load GDP data for Beijing/Shanghai"""
    suburban_dir = get_suburban_dir()
    if city == 'Beijing':
        file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'GDP', 'BJ_gdp.txt')
    elif city == 'Shanghai':
        file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'GDP', 'SH_gdp.txt')
    else:
        print(f"No GDP data available for {city}")
        return [], []
        
    if not os.path.exists(file_path):
        print(f"GDP data file not found: {file_path}")
        return [], []
    
    gdp_locations = []
    gdp_values = []
    
    with open(file_path, 'r') as f:
        next(f)  # Skip header line
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Loading GDP data for {city}"):
            fields = line.strip().split('\t')
            if len(fields) >= 3:
                x = float(fields[0])  # UTM_X
                y = float(fields[1])  # UTM_Y
                gdp_value = float(fields[2])
                gdp_locations.append([y, x])
                gdp_values.append(gdp_value)
    
    print(f"Loaded {len(gdp_locations)} GDP grid points")
    return gdp_locations, gdp_values

def process_pop(data):
    """Process population data for Beijing/Shanghai"""
    percentage_sum_by_fid = data.groupby('FID')['PERCENTAGE'].transform('sum')
    data['adjusted_percentage'] = data['PERCENTAGE'] / percentage_sum_by_fid
    data['gridcode_adj_percentage_product'] = data['gridcode'] * data['adjusted_percentage']
    sum_adj_gridcode_percentage_by_fid = data.groupby('FID')['gridcode_adj_percentage_product'].sum()
    return sum_adj_gridcode_percentage_by_fid

def scan_pois_in_subzones(poi_loc, poi_index, subzones, poi_mode='original'):
    """Scan POIs and assign them to subzones (for Singapore/NYC)"""
    poi_tree = cKDTree(poi_loc)
    assigned_indices = set()
    region_data = {}

    # Initialize region data
    for subzone_id in subzones.keys():
        region_data[subzone_id] = {
            'region_shape': subzones[subzone_id]['geometry'],
            'pois': [],
            'population': subzones[subzone_id]['population']
        }

    # Scan subzones for POIs
    for subzone_id, subzone in tqdm(subzones.items(), desc="Scanning POIs for subzones"):
        polygon = subzone['geometry']
        if not polygon.is_valid or polygon.is_empty:
            print(f"Skipping invalid polygon for region {subzone_id}")
            continue
        bounds = polygon.bounds
        dx = bounds[2] - bounds[0]
        dy = bounds[3] - bounds[1]
        if np.isnan(dx) or np.isnan(dy) or np.isinf(dx) or np.isinf(dy):
            print(f"Skipping region {subzone_id} due to invalid bounds: dx={dx}, dy={dy}")
            continue
        radius = np.sqrt(dx**2 + dy**2) / 2
        if np.isnan(radius) or np.isinf(radius):
            print(f"Skipping region {subzone_id} due to invalid radius: {radius}")
            continue
        radius = min(radius, 5000)  # 5km maximum radius
        subzone_centroid = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        poi_indices_within_radius = poi_tree.query_ball_point(subzone_centroid, radius)
        for poi_idx in poi_indices_within_radius:
            poi_point = Point(poi_loc[poi_idx])
            if polygon.contains(poi_point):
                # 强化index mapping逻辑：SG/NYC filtered模式下，index字段始终为原始index
                if poi_mode == 'filtered':
                    region_data[subzone_id]['pois'].append({
                        'index': poi_index[poi_idx],  # 原始index
                        'location': poi_loc[poi_idx]
                    })
                else:
                    region_data[subzone_id]['pois'].append({
                        'index': poi_idx,
                        'location': poi_loc[poi_idx]
                    })
                assigned_indices.add(poi_idx)

    unassigned_indices = set(range(len(poi_loc))) - assigned_indices
    print(f"Number of POIs not assigned to any subzone: {len(unassigned_indices)}")
    
    return region_data

def align_with_original_data(city, filtered_region_data, subzones, filtered_file_path, dataset='OSM'):
    """Align filtered data with original data for Singapore/NYC"""
    print("Starting alignment with original POI data...")
    
    regions_with_pois_before = len([r for r in filtered_region_data.values() if len(r['pois']) > 0])
    initial_poi_count = sum(len(r['pois']) for r in filtered_region_data.values())
    print(f"Before alignment: {regions_with_pois_before} regions have POIs")
    
    if city == 'Singapore':
        city_abbr = 'SG'
    elif city == 'NYC':
        city_abbr = 'NYC'
    
    original_pkl_path = f'/home/yuanlong001/BanditRegion/data/{dataset}/processed/Integral/{city_abbr}_data.pkl'
    
    try:
        with open(original_pkl_path, 'rb') as f:
            original_region_data = pkl.load(f)
        print(f"Loaded original region data from {original_pkl_path}")
    except FileNotFoundError:
        print(f"Original pkl file not found at {original_pkl_path}, falling back to scanning original POI data...")
        original_poi_text, original_poi_loc, original_poi_index = load_original_poi_for_alignment(city, dataset)
        original_region_data = scan_pois_in_subzones(original_poi_loc, original_poi_index, subzones)
    
    original_regions_with_pois = len([r for r in original_region_data.values() if len(r['pois']) > 0])
    
    current_poi_set = set()
    for subzone_data in filtered_region_data.values():
        for poi_info in subzone_data['pois']:
            current_poi_set.add(tuple(poi_info['location']))
    
    current_filtered_pois = []
    with open(filtered_file_path, 'r') as f:
        for line in f:
            current_filtered_pois.append(line.strip())
    
    original_poi_text, original_poi_loc, original_poi_index = load_original_poi_for_alignment(city, dataset)
    
    index_to_text_map = {}
    for i, poi_text in enumerate(original_poi_text):
        index_to_text_map[i] = poi_text
    
    added_regions = 0
    added_pois = 0
    new_pois_to_write = []
    
    for subzone_id, original_data in original_region_data.items():
        if subzone_id in filtered_region_data:
            if len(filtered_region_data[subzone_id]['pois']) == 0 and len(original_data['pois']) > 0:
                # 补齐region时，index字段也用原始index，保持和BJ/SH一致
                for poi_info in original_data['pois']:
                    poi_location = tuple(poi_info['location'])
                    if poi_location not in current_poi_set:
                        original_index = poi_info['index']
                        poi_text_value = index_to_text_map.get(original_index, f"POI_{original_index}")
                        filtered_region_data[subzone_id]['pois'].append({
                            'index': original_index,
                            'location': poi_info['location']
                        })
                        new_poi_line = f"{poi_text_value}\t{poi_info['location'][0]}\t{poi_info['location'][1]}\t{original_index}"
                        new_pois_to_write.append(new_poi_line)
                        current_poi_set.add(poi_location)
                        added_pois += 1
                if filtered_region_data[subzone_id]['pois']:
                    added_regions += 1
    
    if new_pois_to_write:
        with open(filtered_file_path, 'a') as f:
            for poi_line in new_pois_to_write:
                f.write(poi_line + '\n')
    
    regions_with_pois_after = len([r for r in filtered_region_data.values() if len(r['pois']) > 0])
    total_pois_after = sum(len(r['pois']) for r in filtered_region_data.values())
    
    alignment_stats = {
        'original_regions_with_pois': original_regions_with_pois,
        'regions_with_pois_before': regions_with_pois_before,
        'regions_with_pois_after': regions_with_pois_after,
        'added_regions': added_regions,
        'initial_poi_count': initial_poi_count,
        'total_pois_after': total_pois_after,
        'added_pois': added_pois
    }
    
    return filtered_region_data, alignment_stats

def abbreviation(city):
    if city == 'Beijing':
        return 'bj'
    elif city == 'Shanghai':
        return 'sh'
    elif city == 'Singapore':
        return 'SG'
    elif city == 'NYC':
        return 'NYC'
    else:
        raise ValueError(f"Unsupported city: {city}")

def main():
    args = parse_arguments()
    city = args.city 
    near_assign = args.near_assign
    poi_mode = args.poi_mode
    version = args.version
    # dataset = args.dataset
    top_k = args.top_k
    drop = args.drop
    city_abb = abbreviation(city)
    
    suburban_dir = get_suburban_dir()
    if city in ['Beijing', 'Shanghai']:
        dataset = 'Gaode'
    else:
        dataset = 'OSM'
    
    # Load POI locations
    poi_text, poi_loc, poi_ori_index, file_path = load_poi(city, poi_mode, version, dataset, top_k, drop)
    
    # Check POI coordinate ranges
    if poi_loc:
        poi_array = np.array(poi_loc)
        print(f"POI coordinate ranges:")
        print(f"  X (longitude/easting): {poi_array[:, 0].min():.2f} to {poi_array[:, 0].max():.2f}")
        print(f"  Y (latitude/northing): {poi_array[:, 1].min():.2f} to {poi_array[:, 1].max():.2f}")
    
    if city in ['Singapore', 'NYC']:
        # Singapore/NYC processing path
        if city == 'NYC':
            region_file = os.path.join(suburban_dir, 'data', dataset, 'raw', 'NYC_RegionDCL.pkl')
        else:  # Singapore
            region_file = os.path.join(suburban_dir, 'data', dataset, 'raw', f'{city}_pop_density.json')

        # Load and project subzones
        subzones = load_and_project_subzones(city, region_file)

        # Scan POIs and assign to subzones
        region_data = scan_pois_in_subzones(poi_loc, poi_ori_index, subzones, poi_mode)

        # If in filtered mode, align with original data
        alignment_stats = None
        if poi_mode == 'filtered':
            region_data, alignment_stats = align_with_original_data(city, region_data, subzones, file_path, dataset)
        
        # Save results for Singapore/NYC
        if poi_mode == 'filtered':
            save_path = os.path.join(suburban_dir, 'data', dataset, 'processed', 'Integral', f'{city_abb}_data_BM25_keywords_kmeans.pkl')
        else:
            save_path = os.path.join(suburban_dir, 'data', dataset, 'processed', 'Integral', f'{city_abb}_data.pkl')
            
        # Convert region_data to match expected format
        final_region_data = {}
        for region_id, data in region_data.items():
            if data['pois']:  # Only save regions with POIs
                # 保证SG/NYC filtered模式下index字段为原始index
                final_region_data[region_id] = {
                    'region_shape': data['region_shape'],
                    'pois': [{'index': poi_info['index'], 'location': poi_info['location']} for poi_info in data['pois']],
                    'population': data['population']
                }
        with open(save_path, 'wb') as f:
            pkl.dump(final_region_data, f)

        print(f"Processed subzone data saved to {save_path}")
        print(f"Final statistics:")
        
        if poi_mode == 'filtered' and alignment_stats:
            print(f"  - Original regions with POIs: {alignment_stats['original_regions_with_pois']}")
            print(f"  - Added regions through alignment: {alignment_stats['added_regions']}")
            print(f"  - Total regions with POIs: {alignment_stats['regions_with_pois_after']}")
            print(f"  - Total POIs: {alignment_stats['total_pois_after']}")
            print(f"  - Added POIs: {alignment_stats['added_pois']}")
        else:
            print(f"  - Total regions: {len(region_data)}")
            print(f"  - Regions with POIs: {len([r for r in region_data.values() if len(r['pois']) > 0])}")
            print(f"  - Total POIs: {sum(len(r['pois']) for r in region_data.values())}")

    else:
        # Beijing/Shanghai processing path
        # Load house locations (only for Beijing/Shanghai)
        house_loc, house_price = load_house(city)
        gdp_locations, gdp_values = load_gdp(city)
        
        # Load original regions and road segmented regions geometry
        road_regions, area_dict = load_region_geometry(city)
        # Load and process population data
        pop_data = load_pop(city)
        population_dict = process_pop(pop_data).to_dict()

        # Create a KDTree for POIs based on their locations
        poi_tree = cKDTree(poi_loc)
        print('Aggregating poi...')

        # Create KDTrees for additional data
        if house_loc:
            house_tree = cKDTree(house_loc)
        else:
            house_tree = None
        
        if gdp_locations:
            gdp_tree = cKDTree(gdp_locations)
            print('Aggregating GDP data...')
            gdp_cell_size = 972.96
            
            gdp_x_vals = [loc[0] for loc in gdp_locations]
            gdp_y_vals = [loc[1] for loc in gdp_locations]
            print(f"GDP X coordinate range: {min(gdp_x_vals)} to {max(gdp_x_vals)}")
            print(f"GDP Y coordinate range: {min(gdp_y_vals)} to {max(gdp_y_vals)}")
        else:
            gdp_tree = None
            print('No GDP data to aggregate')

        region_centers = []
        for region in road_regions:
            region_centroid = ((region.bounds[0] + region.bounds[2]) / 2, (region.bounds[1] + region.bounds[3]) / 2)
            region_centers.append(region_centroid)
        
        if gdp_locations:
            region_x_vals = [center[0] for center in region_centers]
            region_y_vals = [center[1] for center in region_centers]
            print(f"Region X coordinate range: {min(region_x_vals)} to {max(region_x_vals)}")
            print(f"Region Y coordinate range: {min(region_y_vals)} to {max(region_y_vals)}")

        # Handle MultiPolygons in road_regions
        new_regions = []
        for idx, region in enumerate(tqdm(road_regions)):
            if not region.is_valid:
                region = region.buffer(0)
            if isinstance(region, MultiPolygon):
                for polygon in region.geoms:
                    new_regions.append(polygon)
                print(f"Road Region {idx} was a MultiPolygon and has been split.")
            else:
                new_regions.append(region)

        road_regions = new_regions

        # Create a dict to save data for road_region
        road_region_data = {
            idx: {
                'pois': [],
                'gdp': 0.0,
                'gdp_density': 0.0
            } for idx in range(len(road_regions))
        }

        assigned_indices = set()
        # Aggregate POIs, house prices, and GDP to road_regions
        for idx, region in enumerate(tqdm(road_regions, desc="Aggregating POIs, house_price to road regions")):
            dx = region.bounds[2] - region.bounds[0]
            dy = region.bounds[3] - region.bounds[1]
            radius = math.sqrt(dx**2 + dy**2) / 2
            region_centroid = region_centers[idx] 
            
            # Process POIs
            poi_indices_within_radius = poi_tree.query_ball_point(region_centroid, radius)
            for poi_index in poi_indices_within_radius:
                poi_point = Point(poi_loc[poi_index])
                if region.contains(poi_point):
                    if poi_mode == 'filtered':
                        road_region_data[idx]['pois'].append({
                            'poi_index': poi_ori_index[poi_index],
                            'location': poi_loc[poi_index]
                        })
                    else:
                        road_region_data[idx]['pois'].append({
                            'poi_index': poi_index,
                            'location': poi_loc[poi_index]
                        })
                    assigned_indices.add(poi_index)
            
            # Process house prices
            if house_tree is not None:
                house_indices_within_radius = house_tree.query_ball_point(region_centroid, radius)
                house_prices = []
                for house_index in house_indices_within_radius:
                    house_point = Point(house_loc[house_index])
                    if region.contains(house_point):
                        house_prices.append(house_price[house_index])
                
                if house_prices:
                    average_house_price = np.mean(house_prices)
                else:
                    average_house_price = 0

                road_region_data[idx]['average_house_price'] = average_house_price
            else:
                road_region_data[idx]['average_house_price'] = 0
            
            # Process GDP data
            if gdp_tree is not None:
                gdp_indices_within_radius = gdp_tree.query_ball_point(region_centroid, radius + gdp_cell_size)
                region_gdp = 0.0
                
                for gdp_index in gdp_indices_within_radius:
                    gdp_x, gdp_y = gdp_locations[gdp_index]
                    gdp_value = gdp_values[gdp_index]
                    
                    half_size = gdp_cell_size / 2
                    gdp_cell = Polygon([
                        (gdp_x - half_size, gdp_y - half_size),
                        (gdp_x + half_size, gdp_y - half_size),
                        (gdp_x + half_size, gdp_y + half_size),
                        (gdp_x - half_size, gdp_y + half_size)
                    ])
                    
                    if region.intersects(gdp_cell):
                        if region.contains(gdp_cell):
                            region_gdp += gdp_value
                        else:
                            intersection = region.intersection(gdp_cell)
                            intersection_area = intersection.area
                            gdp_cell_area = gdp_cell.area
                            area_ratio = intersection_area / gdp_cell_area
                            region_gdp += gdp_value * area_ratio
                
                region_area = area_dict[idx]
                gdp_density = (region_gdp / region_area) * 1000000 if region_area > 0 else 0.0
                
                road_region_data[idx]['gdp'] = region_gdp
                road_region_data[idx]['gdp_density'] = gdp_density
            else:
                road_region_data[idx]['gdp'] = 0.0
                road_region_data[idx]['gdp_density'] = 0.0
        
        # Count regions with no POIs
        count_no_point = sum(len(data['pois']) == 0 for data in road_region_data.values())
        print(f"(Before) Number of road regions without any POI: {count_no_point}")

        unassigned_indices = set(range(len(poi_loc))) - assigned_indices
        print(f"(Before) Number of POIs not assigned to any region: {len(unassigned_indices)}")

        # Assign unassigned POIs to nearest road_region if near_assign is True
        if near_assign:
            region_with_poi_loc = []
            region_with_poi_2_all_region_id = OrderedDict()
            for idx, data in road_region_data.items():
                pois = data['pois']
                if pois:
                    region_with_poi_2_all_region_id[len(region_with_poi_2_all_region_id)] = idx
                    region_with_poi_loc.append(region_centers[idx])
                    
            region_center_tree = cKDTree(region_with_poi_loc)
            unassigned_indices = list(unassigned_indices)
            _, nearest_region_indices = region_center_tree.query(np.array(poi_loc)[unassigned_indices])
            
            for i in tqdm(range(len(unassigned_indices)), desc="Assigning unassigned POIs to nearest road region"):
                region_idx = nearest_region_indices[i]
                region_idx = region_with_poi_2_all_region_id[region_idx]
                poi_idx = unassigned_indices[i]
                if poi_mode == 'filtered':
                    road_region_data[region_idx]['pois'].append({
                        'poi_index': poi_ori_index[poi_idx],
                        'location': poi_loc[poi_idx]
                    })
                else:
                    road_region_data[region_idx]['pois'].append({
                        'poi_index': poi_idx,
                        'location': poi_loc[poi_idx]
                    })

            # DEBUG: check the number of pois in regions
            debug_poi_idxs = set()
            for data in road_region_data.values():
                for poi in data['pois']:
                    debug_poi_idxs.add(poi['poi_index'])
            assert len(debug_poi_idxs) == len(poi_loc), f"Total number of POIs in regions is not equal to the total number of POIs: {len(debug_poi_idxs)} != {len(poi_loc)}"

        # Final statistics
        count_no_point = sum(len(data['pois']) == 0 for data in road_region_data.values())
        print(f"(After) Number of road regions without any POI: {count_no_point}")

        if house_tree is not None:
            count_no_house = sum(data['average_house_price'] == 0 for data in road_region_data.values())
            print(f"Number of road regions without any house price information: {count_no_house}")
        
        if gdp_tree is not None:
            count_no_gdp = sum(data['gdp'] == 0 for data in road_region_data.values())
            count_no_gdp_density = sum(data['gdp_density'] == 0 for data in road_region_data.values())
            total_gdp = sum(data['gdp'] for data in road_region_data.values())
            
            print(f"Number of road regions without any GDP data: {count_no_gdp}")
            print(f"Number of road regions without any GDP density data: {count_no_gdp_density}")

            regions_with_gdp = len(road_regions) - count_no_gdp
            print(f"Number of regions with GDP > 0: {regions_with_gdp}")
            
            if regions_with_gdp > 0:
                print(f"Average GDP per road region (with GDP > 0): {total_gdp / regions_with_gdp:.4f}")

        # Save results for Beijing/Shanghai
        region_data = {}

        for idx, data in road_region_data.items():
            pois = data['pois']
            
            if pois:  
                region_info = {
                    'region_shape': road_regions[idx],
                    'pois': [{'index': poi_info.get('index', poi_info.get('poi_index')), 'location': poi_info['location']} for poi_info in pois],
                    'population': population_dict.get(idx, 0),
                }
                
                # Add house price data if available
                if house_tree is not None:
                    region_info['house_price'] = data['average_house_price']
                
                # Add GDP data if available
                if gdp_tree is not None:
                    region_info['gdp'] = data['gdp_density']
                
                region_data[idx] = region_info

        # Alignment for filtered mode
        if poi_mode == 'filtered':
            if dataset == 'Gaode':
                check_region_path = os.path.join(get_suburban_dir(), 'data', dataset, 'processed', 'Integral', f'{city_abb.upper()}_data.pkl')
            
            try:
                with open(check_region_path, 'rb') as f:
                    check_region_data = pkl.load(f)
                
                current_poi_set = set()
                for region_data_item in region_data.values():
                    for poi_info in region_data_item['pois']:
                        current_poi_set.add(tuple(poi_info['location']))
                
                original_poi_file = os.path.join(get_suburban_dir(), 'data', dataset, 'projected', city, 'poi.txt')
                index_to_text_map = {}
                with open(original_poi_file, 'r') as f:
                    for line_idx, line in enumerate(f):
                        fields = line.strip().split('\t')
                        index_to_text_map[line_idx] = fields[0]
                
                add_region_count = 0
                add_poi_count = 0
                new_pois_to_write = []
                
                for idx, original_data in check_region_data.items():
                    if idx in region_data:
                        if len(region_data[idx]['pois']) == 0 and len(original_data['pois']) > 0:
                            
                            for poi_info in original_data['pois']:
                                poi_location = tuple(poi_info['location'])
                                if poi_location not in current_poi_set:
                                    original_index = poi_info['index']
                                    poi_text_value = index_to_text_map.get(original_index, f"POI_{original_index}")
                                    
                                    region_data[idx]['pois'].append({
                                        'index': original_index,
                                        'location': poi_info['location']
                                    })
                                    
                                    new_poi_line = f"{poi_text_value}\t{poi_info['location'][0]}\t{poi_info['location'][1]}\t{original_index}"
                                    new_pois_to_write.append(new_poi_line)
                                    
                                    current_poi_set.add(poi_location)
                                    add_poi_count += 1
                            
                            if region_data[idx]['pois']:
                                add_region_count += 1
                    
                    elif idx not in region_data:
                        region_data[idx] = original_data
                        add_region_count += 1
                        for poi_info in original_data['pois']:
                            poi_location = tuple(poi_info['location'])
                            if poi_location not in current_poi_set:
                                original_index = poi_info['index']
                                poi_text_value = index_to_text_map.get(original_index, f"POI_{original_index}")
                                
                                new_poi_line = f"{poi_text_value}\t{poi_info['location'][0]}\t{poi_info['location'][1]}\t{original_index}"
                                new_pois_to_write.append(new_poi_line)
                                
                                current_poi_set.add(poi_location)
                            add_poi_count += 1
                
                if new_pois_to_write:
                    with open(file_path, 'a') as f:
                        for poi_line in new_pois_to_write:
                            f.write(poi_line + '\n')
                
                print(f"Number of regions in the final data: {len(region_data)}")
                print(f"Number of added region: {add_region_count}")
                print(f"Number of added pois: {add_poi_count}")
                
            except FileNotFoundError:
                print(f"Original region data not found at {check_region_path}, skipping alignment")

        # Save final results
        if poi_mode == 'filtered':
            save_path = os.path.join(suburban_dir, 'data', dataset, 'processed', 'Integral', f'{city_abb.upper()}_data_{drop}_{version}_{top_k}.pkl')
        else:
            save_path = os.path.join(suburban_dir, 'data', dataset, 'processed', 'Integral', f'{city_abb.upper()}_data.pkl')
        
        with open(save_path, 'wb') as f:
            pkl.dump(region_data, f)

        print(f"Processed region data saved to {save_path}")


if __name__ == "__main__":
    main()