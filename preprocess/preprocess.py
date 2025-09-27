from collections import OrderedDict
import os
import pickle as pkl
import numpy as np
import math
import geopandas as gpd
import pandas as pd
import argparse
import sys

from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon

def get_suburban_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)  # Go from preprocess folder to SubUrban root

# Fix for shapely.io compatibility issue with old pickle files
try:
    import shapely.io
except ImportError:
    # Create a dummy shapely.io module for backward compatibility
    import shapely
    import types
    from shapely import wkb, wkt
    
    shapely.io = types.ModuleType('shapely.io')
    # Add necessary functions that might be referenced in old pickle files
    shapely.io.from_wkb = wkb.loads
    shapely.io.from_wkt = wkt.loads
    shapely.io.to_wkb = wkb.dumps
    shapely.io.to_wkt = wkt.dumps
    sys.modules['shapely.io'] = shapely.io

tqdm.pandas()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data for SparseRegion downstream tasks.')
    parser.add_argument('--city', type=str, choices=['Beijing', 'Shanghai', 'Hangzhou'], default='Beijing', help='City: "Beijing", "Shanghai", "Hangzhou"')
    parser.add_argument('--near_assign', action='store_true', help='Whether to assign those unassigned POIs to the nearest road region')
    parser.add_argument('--poi_mode', type=str, choices=['original', 'filtered'], default='original', help='Mode for loading poi.txt dataset')
    parser.add_argument('--version' , type=str, default='keywords_kmeans', help='Version of the filtered poi data')
    parser.add_argument('--dataset', type= str, default='Gaode', choices=['Meituan','Gaode'], help='Dataset to use')
    parser.add_argument('--top_k', type=int, default=8000, help='Number of top k pois to use')
    parser.add_argument('--drop', type=str, choices=['BM25','random'], default='BM25', help='Drop method for poi filtering')
    return parser.parse_args()

def load_poi(city, poi_mode, version, dataset, top_k, drop):
    # Load poi.txt concludes text and location of POIs
    suburban_dir = get_suburban_dir()
    poi_loc = []
    poi_index = []
    if poi_mode == 'original':
        # Load the original full POI dataset
        file_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city, 'poi.txt')
    elif poi_mode == 'filtered':
        # Load the filtered POI dataset
        file_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city, f'poi_{drop}_{version}_{top_k}.txt')

    
    if city in ['Beijing', 'Shanghai']:
        with open(file_path, 'r') as f:
        # with open(f'/home_nfs/regionteam/BanditRegion/data/GPT4o_RAG/Shanghai/RAG_pois_with_UTM_coords.txt', 'r') as f:
            lines = f.readlines()
            # Count the number of lines in the file
            print(f"Number of pois in original txt file: {len(lines)}")
            for line in tqdm(lines):
                fields = line.strip().split('\t')
                poi_loc.append([float(fields[2]), float(fields[1])])
                poi_text = fields[0]
                if poi_mode == 'filtered':
                    # 在筛选后的文件中，索引应该在最后一列，无论原始行有多少列
                    try:
                        poi_index.append(int(fields[-1]))  # 使用最后一列作为索引
                    except (ValueError, IndexError) as e:
                        print(f"警告：无法解析索引，使用None代替。错误: {e}, 行: {line.strip()}")
                        poi_index.append(None)
                else:
                    poi_index.append(None)
    # elif (city == 'Hangzhou'):
    #     with open(f'/home_nfs/regionteam/BanditRegion/data/Meituan/projected/{city}/poi.txt', 'r') as f:
    #         lines = f.readlines()
    #         for line in tqdm(lines):
    #             fields = line.strip().split('\t')
    #             poi_loc.append([float(fields[2]), float(fields[1])])
    else:
        raise ValueError("Unsupported city. Please choose 'Beijing', 'Shanghai' or 'Hangzhou'")
    
    print(f"Loaded {len(poi_loc)} POIs")
    return poi_text, poi_loc, poi_index, file_path


def load_region_geometry(city):
    # Load Region Geometry and compute areas
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
    return landuse_regions, road_regions, area_dict

def load_region_data(city):
    # Load xx_data.txt concludes lon lat of centroid, categories of Level1 and Level2 of regions
    suburban_dir = get_suburban_dir()
    with open(os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'region', f'{city.lower()}_data.txt'), 'r') as f:
        lines = f.readlines()
        region_level1 = []
        for line in tqdm(lines):
            fields = line.strip().split('\t')
            try:
                # Try to convert level1 to integer, it is an inefficient data if it fails
                level1_value = int(fields[2])
                region_level1.append(level1_value)
            except ValueError:
                continue

    return region_level1

def load_pop(city):
    # Load population data from the corresponding city
    suburban_dir = get_suburban_dir()
    if (city == 'Beijing'):
        file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'population', 'beijing_pop.csv')
        data = pd.read_csv(file_path)
    elif (city == 'Shanghai'):
        file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'population', 'shanghai_pop.csv')
        data = pd.read_csv(file_path)
    else:
        print('Invalid city name')

    return data



def load_house(city):
    suburban_dir = get_suburban_dir()
    file_path = os.path.join(suburban_dir, 'data', 'Gaode', 'processed', 'house', f'{city}_house_data.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
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
        # Skip header line
        next(f)
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Loading GDP data for {city}"):
            fields = line.strip().split('\t')
            if len(fields) >= 3:
                # 需要交换坐标顺序 - 数据文件中是 UTM_X, UTM_Y, 但在程序中的坐标系统可能是反的
                x = float(fields[0])  # UTM_X
                y = float(fields[1])  # UTM_Y
                gdp_value = float(fields[2])
                
                # 重要：尝试两种不同的坐标顺序，以便找出哪种匹配
                # 第一种方式：不交换坐标
                gdp_locations.append([y, x])
                gdp_values.append(gdp_value)
    
    print(f"Loaded {len(gdp_locations)} GDP grid points")
    return gdp_locations, gdp_values

# def process_house_price(house_loc):
#     house_price = []
#     for house in house_loc:
#         house_price=
#     return house_price

def process_pop(data):
    # Calculate the percentages sum of each FID region
    percentage_sum_by_fid = data.groupby('FID')['PERCENTAGE'].transform('sum')

    # Create a new column to calculate the newly calculated percentage for each patch in region
    data['adjusted_percentage'] = data['PERCENTAGE'] / percentage_sum_by_fid

    # Calculate the weighted density of each patch in a region
    data['gridcode_adj_percentage_product'] = data['gridcode'] * data['adjusted_percentage']

    # Sum the weigthed densities of patehcs in each region
    sum_adj_gridcode_percentage_by_fid = data.groupby('FID')['gridcode_adj_percentage_product'].sum()

    return sum_adj_gridcode_percentage_by_fid

def abbreviation(city):
    if (city == 'Beijing'):
        return 'bj'
    elif (city == 'Shanghai'):
        return 'sh'
    elif (city == 'Hangzhou'):
        return 'hz'
    else:
        raise ValueError("Unsupported city. Please choose 'Beijing', 'Shanghai' or 'Hangzhou'")

# Function to extract latitude and longitude from filename
def extract_lat_lon_from_filename(filename):
    parts = filename.split('_')
    if len(parts) >= 4:
        lon = float(parts[1])
        lat = float(parts[2])
        heading = parts[3]
        return lat, lon, heading
    return None, None, None

def extract_index_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 0:
        try:
            return int(parts[0])
        except ValueError:
            return None
    return None

# def load_image_filenames_and_extract_coordinates(directory):
#     images_info = []
#     filenames = sorted([f for f in os.listdir(directory) if f.endswith('.png')], key=extract_index_from_filename)
    
#     # Create a dictionary to count the images for each sampling point
#     sampling_point_counters = {}

#     for filename in tqdm(filenames, desc="Loading images"):
#         lat, lon , heading = extract_lat_lon_from_filename(filename)
#         if lat is not None and lon is not None:
#             sampling_point_index = extract_index_from_filename(filename)
            
#             # Generate a unique index
#             if sampling_point_index not in sampling_point_counters:
#                 sampling_point_counters[sampling_point_index] = 0
#             else:
#                 sampling_point_counters[sampling_point_index] += 1
            
#             unique_index = f"{sampling_point_index}_{sampling_point_counters[sampling_point_index]}"
#             images_info.append({'filename': filename, 'latitude': lat, 'longitude': lon, 'heading': heading,'unique_index': unique_index})
    
#     print(f"Successfully loaded {len(images_info)} images")
#     return pd.DataFrame(images_info)

# # Function to transform coordinates
# def transform_coords(lon, lat, city):
#     if city in ['Beijing', 'Shanghai']:
#         utm_y, utm_x, _, _ = utm.from_latlon(lat, lon)
#         return utm_x, utm_y
#     elif city == 'Hangzhou':
#         transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2385", always_xy=True)
#         x, y = transformer.transform(lon, lat)
#         return x, y
#     else:
#         raise ValueError(f"Unsupported city: {city}")

# def buffer_assignment(unassigned_svi, svi_loc, road_region_data, road_regions, buffer_radius):
#     assigned_svi = []
#     road_region_df = gpd.GeoDataFrame(road_regions, geometry=[region.centroid for region in road_regions])
    
#     for svi_index in tqdm(unassigned_svi, desc="Assigning unassigned SVIs with buffer"):
#         svi_point = svi_loc[svi_loc['unique_index'] == svi_index].geometry.values[0]
#         buffer = svi_point.buffer(buffer_radius)
        
#         intersecting_regions = road_region_df[road_region_df.geometry.intersects(buffer)]
        
#         if not intersecting_regions.empty:
#             nearest_region = intersecting_regions.iloc[0]
#             road_region_data[nearest_region.name]['svis'].append({
#                 'unique_index': svi_index,
#                 'location': (svi_point.x, svi_point.y)
#             })
#             assigned_svi.append(svi_index)
    
#     return assigned_svi

def plot_and_summarize(city, road_regions, road_region_data, poi_loc, svi_loc=None):
    # Helper function to plot and summarize data
    def plot_data(data_key, data_name, loc_data, index_field, filename_suffix):
        # Create a list of color intensities based on the number of items in each road region
        color_intensities = [len(data[data_key]) for data in road_region_data.values()]

        # Normalize the color intensities to the range [0, 1]
        max_intensity = max(color_intensities, default=1)  # Prevent division by zero if all regions have no items
        min_intensity = min(color_intensities, default=0)
        color_intensities = [(intensity - min_intensity) / (max_intensity - min_intensity) if max_intensity > 0 else 0 for intensity in color_intensities]

        # Plot the regions with the number of items as the color intensity
        fig, ax = plt.subplots(figsize=(10, 10))  # Adjust size as needed

        # Decide the coordinate order based on the city
        if city == 'Hangzhou':
            coord_order = lambda x, y: (x, y)
        else:
            coord_order = lambda x, y: (y, x)

        for idx, (region, data) in enumerate(zip(road_regions, road_region_data.values())):
            color_intensity = color_intensities[idx]
            region_color = (color_intensity, color_intensity, color_intensity)  # Grayscale intensity

            # Draw each polygon in the region
            if isinstance(region, MultiPolygon):
                for polygon in region.geoms:
                    x, y = polygon.exterior.xy
                    ax.fill(*coord_order(x, y), color=region_color, alpha=0.5)
            else:
                x, y = region.exterior.xy
                ax.fill(*coord_order(x, y), color=region_color, alpha=0.5)

            # Highlight regions with no items in red
            if len(data[data_key]) == 0:
                ax.fill(*coord_order(x, y), color='red', alpha=0.5)

        # Set to store indices of items assigned to regions
        assigned_indices = set()

        # Iterate over road_region_data to collect assigned item indices
        for data in road_region_data.values():
            for item_info in data[data_key]:
                assigned_indices.add(item_info[index_field])

        # Check which item indices are not assigned to any region
        if data_key == 'pois':
            unassigned_indices = set(range(len(loc_data))) - assigned_indices
        elif data_key == 'svis':
            unassigned_indices = set(loc_data['unique_index']) - assigned_indices

        # Mark unassigned items in blue on the regions plot
        for item_index in unassigned_indices:
            if data_key == 'pois':
                item_x, item_y = loc_data[item_index]
            elif data_key == 'svis':
                item = loc_data[loc_data['unique_index'] == item_index].iloc[0]
                item_x, item_y = item['projected_x'], item['projected_y']
            ax.plot(*coord_order(item_x, item_y), marker='o', color='blue', markersize=1, linestyle='None')  # Adjust markersize for visibility

        # Print the number of unassigned item embeddings
        print(f"Number of {data_name} not assigned to any region: {len(unassigned_indices)}")

        # Finalize and save the plot
        suburban_dir = get_suburban_dir()
        visualize_dir = os.path.join(suburban_dir, 'tmp', 'visualizations')
        os.makedirs(visualize_dir, exist_ok=True)
        plt.savefig(os.path.join(visualize_dir, f"ZQ_road_region_{filename_suffix}_mapping_with_unassigned_{city}.png"))

    # Plot and summarize POIs
    plot_data('pois', 'POI', poi_loc, 'poi_index', 'poi')

    # Plot and summarize SVIs if provided
    # if svi_loc is not None:
    #     plot_data('svis', 'SVI', svi_loc, 'unique_index', 'svi')

def main():
    args = parse_arguments()
    city = args.city 
    near_assign = args.near_assign
    poi_mode = args.poi_mode
    version = args.version
    dataset = args.dataset
    top_k = args.top_k
    drop = args.drop
    city_abb = abbreviation(city)
    # Load poi locations
    poi_text, poi_loc, poi_ori_index, file_path = load_poi(city, poi_mode, version, dataset, top_k, drop)
    # Load house locations
    house_loc, house_price = load_house(city)
    # Load GDP data
    gdp_locations, gdp_values = load_gdp(city)
    # Load region level1 labels
    region_level1= load_region_data(city)
    # Load original regions and road segmented regions geometry
    landuse_regions, road_regions, area_dict = load_region_geometry(city)
    # Load and process population data
    pop_data = load_pop(city)
    population_dict = process_pop(pop_data).to_dict()

    # # Load and process svi data
    # if (city == 'Beijing') or (city == 'Shanghai'):
    #     directory = f'/home_nfs/regionteam/SparseRegion/data/raw/svi/{city}/svi_{city.lower()}'
    # elif (city == 'Hangzhou'):
    #     directory = f'/home_nfs/regionteam/SparseRegion/data/raw/svi/{city}'

    # images_df = load_image_filenames_and_extract_coordinates(directory)

    # # Transform the coordinates and add them to the DataFrame
    # images_df['projected_coords'] = images_df.progress_apply(lambda row: transform_coords(row['longitude'], row['latitude'], city), axis=1)
    # successful_conversions = images_df['projected_coords'].notnull().sum()
    # print(f"Successfully transformed coordinates for {successful_conversions} images")

    # images_df[['projected_x', 'projected_y']] = pd.DataFrame(images_df['projected_coords'].tolist(), index=images_df.index)
    # svi_loc = gpd.GeoDataFrame(images_df, geometry=gpd.points_from_xy(images_df.projected_x, images_df.projected_y))


    #######
    # Create a KDTree for POIs based on their locations
    poi_tree = cKDTree(poi_loc)
    print('Aggregating poi...')

    # Create a KDTree for houses based on their locations
    house_tree = cKDTree(house_loc)
    
    # Create a KDTree for GDP grid points if available
    if gdp_locations:
        gdp_tree = cKDTree(gdp_locations)
        print('Aggregating GDP data...')
        # Define the GDP grid cell size (972.96m x 972.96m as specified)
        gdp_cell_size = 972.96
        
        # Print coordinate ranges for debugging
        gdp_x_vals = [loc[0] for loc in gdp_locations]
        gdp_y_vals = [loc[1] for loc in gdp_locations]
        print(f"GDP X coordinate range: {min(gdp_x_vals)} to {max(gdp_x_vals)}")
        print(f"GDP Y coordinate range: {min(gdp_y_vals)} to {max(gdp_y_vals)}")
    else:
        gdp_tree = None
        print('No GDP data to aggregate')

    # 初始化区域中心点列表
    region_centers = []
    for region in road_regions:
        region_centroid = ((region.bounds[0] + region.bounds[2]) / 2, (region.bounds[1] + region.bounds[3]) / 2)
        region_centers.append(region_centroid)
    
    # 如果有GDP数据，打印区域坐标范围进行比较
    if gdp_locations:
        region_x_vals = [center[0] for center in region_centers]
        region_y_vals = [center[1] for center in region_centers]
        print(f"Region X coordinate range: {min(region_x_vals)} to {max(region_x_vals)}")
        print(f"Region Y coordinate range: {min(region_y_vals)} to {max(region_y_vals)}")

    # Create a KDTree for SVIs based on their locations
    # svi_tree = cKDTree(svi_loc[['projected_x', 'projected_y']])
    # print('Aggregating SVIs...')

    count_no_point = 0

    # Iterate over each region to find whether there are still MultiPolygons
    new_regions = []
    new_regions_level1 = []
    for idx, region in enumerate(tqdm(landuse_regions)):
        if not region.is_valid:
            region = region.buffer(0)  # Make the region valid
        if isinstance(region, MultiPolygon):
            for polygon in region.geoms:
                new_regions.append(polygon)
                new_regions_level1.append(region_level1[idx])
            print(f"Region {idx} was a MultiPolygon and has been split.")
        else:
            new_regions.append(region)  # Add the region as is if it's not a MultiPolygon
            new_regions_level1.append(region_level1[idx])

    assert len(new_regions) == len(new_regions_level1)

    landuse_regions = new_regions
    landuse_region_level1 = new_regions_level1

    # Iterate over each road_region to find whether there are still MultiPolygons
    new_regions = []
    for idx, region in enumerate(tqdm(road_regions)):
        if not region.is_valid:
            region = region.buffer(0)  # Make the region valid
        if isinstance(region, MultiPolygon):
            for polygon in region.geoms:
                new_regions.append(polygon)
            print(f"Road Region {idx} was a MultiPolygon and has been split.")
        else:
            new_regions.append(region)  # Add the region as is if it's not a MultiPolygon

    road_regions = new_regions

    # This should theoretically not happen if step 1 is correct
    for idx, region in enumerate(landuse_regions):
        if isinstance(region, MultiPolygon):
            print(f"Region {idx} is still a MultiPolygon.")  

    for idx, region in enumerate(road_regions):
        if isinstance(region, MultiPolygon):
            print(f"Road Region {idx} is still a MultiPolygon.")  


    ########### Aggregate landuse_regions into road_regions
    # Create a KDTree for landuse_regions based on their centroids
    landuse_region_centroids = [(region.centroid.x, region.centroid.y) for region in landuse_regions]
    land_tree = cKDTree(landuse_region_centroids)

    # Create a dict to save data for road_region，includes label distribution and POIs
    road_region_data = {
        idx: {
            'label_distribution': np.zeros(5, dtype=np.float32),
            'pois': [],
            'svis': [],
            'gdp': 0.0,  # Initialize GDP value for each road region
            'gdp_density': 0.0  # Initialize GDP density value (GDP/area) for each road region
        } for idx in range(len(road_regions))
    }

    # Aggregate landuse_regions into the higher level of road_regions
    for idx, road_region in enumerate(tqdm(road_regions, desc="Aggregating landuse to roads")):
        lu_dx = road_region.bounds[2] - road_region.bounds[0]
        lu_dy = road_region.bounds[3] - road_region.bounds[1]
        lu_radius = math.sqrt(lu_dx**2 + lu_dy**2) / 2
        lu_centroid = ((road_region.bounds[0] + road_region.bounds[2]) / 2
                       , (road_region.bounds[1] + road_region.bounds[3]) / 2)
        nearby_landuse_indices = land_tree.query_ball_point(lu_centroid, lu_radius)
        total_area = 0
        for landuse_idx in nearby_landuse_indices:
            landuse_region = landuse_regions[landuse_idx]
            if road_region.intersects(landuse_region):
                intersection_area = road_region.intersection(landuse_region).buffer(0).area
                total_area += landuse_region.area
                label_index = landuse_region_level1[landuse_idx]
                road_region_data[idx]['label_distribution'][label_index] += intersection_area

        # Normalize the label distribution of road_region
        if total_area > 0:
            road_region_data[idx]['label_distribution'] /= total_area

    # Normalize the label distribution of road_region
    for idx, data in road_region_data.items():
        total_labels = np.sum(data['label_distribution'])
        if total_labels > 0:
            data['label_distribution'] /= total_labels

    assigned_indices = set()
    # Aggregating all inside info to road_regions (ensure 'index' field is used consistently)
    for idx, region in enumerate(tqdm(road_regions, desc="Aggregating POIs, house_price to road regions")):
        dx = region.bounds[2] - region.bounds[0]
        dy = region.bounds[3] - region.bounds[1]
        radius = math.sqrt(dx**2 + dy**2) / 2
        region_centroid = region_centers[idx]  # 使用已经计算好的区域中心点
        
        poi_indices_within_radius = poi_tree.query_ball_point(region_centroid, radius)
        for poi_index in poi_indices_within_radius:
            poi_point = Point(poi_loc[poi_index])
            if region.contains(poi_point):
                if poi_mode == 'filtered':
                    road_region_data[idx]['pois'].append({
                        'poi_index': poi_ori_index[poi_index],  # Ensure 'index' field is included
                        # 'poi_index': poi_index,
                        'location': poi_loc[poi_index]
                    })
                elif poi_mode == 'original' or 'mini':
                    road_region_data[idx]['pois'].append({
                        'poi_index': poi_index,  # Ensure 'index' field is included
                        'location': poi_loc[poi_index]
                    })
                assigned_indices.add(poi_index)
        
        house_indices_within_radius = house_tree.query_ball_point(region_centroid, radius)
        house_prices=[]
        for house_index in house_indices_within_radius:
            house_point = Point(house_loc[house_index])
            if region.contains(house_point):
                house_prices.append(house_price[house_index])
        
        if house_prices:
            average_house_price = np.mean(house_prices)
        else:
            # print(f"No house found in region {idx}")
            average_house_price = 0

        road_region_data[idx]['average_house_price'] = average_house_price
        
        # Process GDP data if available
        if gdp_tree is not None:
            # Find GDP grid points within or intersecting with the region
            gdp_indices_within_radius = gdp_tree.query_ball_point(region_centroid, radius + gdp_cell_size)
            region_gdp = 0.0
            

            
            gdp_cells_intersecting = 0  # Count intersecting cells for debugging
            
            for gdp_index in gdp_indices_within_radius:
                # Get GDP grid cell center coordinates
                gdp_x, gdp_y = gdp_locations[gdp_index]
                gdp_value = gdp_values[gdp_index]
                
                # Create a GDP grid cell polygon (square centered at the grid point)
                half_size = gdp_cell_size / 2
                # 尝试创建一个更大的GDP单元格多边形，以增加匹配的机会
                gdp_cell = Polygon([
                    (gdp_x - half_size, gdp_y - half_size),
                    (gdp_x + half_size, gdp_y - half_size),
                    (gdp_x + half_size, gdp_y + half_size),
                    (gdp_x - half_size, gdp_y + half_size)
                ])
                
                # Check if the GDP cell intersects with the region
                if region.intersects(gdp_cell):
                    gdp_cells_intersecting += 1
                    if region.contains(gdp_cell):
                        # If the region completely contains the GDP cell, add the full GDP value
                        region_gdp += gdp_value
                    else:
                        # If there's partial overlap, calculate the intersection area ratio
                        intersection = region.intersection(gdp_cell)
                        intersection_area = intersection.area
                        gdp_cell_area = gdp_cell.area
                        area_ratio = intersection_area / gdp_cell_area
                        
                        # Add the proportional GDP value based on the intersection area
                        region_gdp += gdp_value * area_ratio
            

            
            # 计算区域面积
            region_area = area_dict[idx]
            
            # 计算GDP密度（GDP值除以区域面积），并乘以10^6调整单位到合适的数量级
            # 将单位从"百万元/平方米"调整为"百万元/平方千米"，乘以10^6
            gdp_density = (region_gdp / region_area) * 1000000 if region_area > 0 else 0.0
            
            # Save the calculated GDP value and GDP density for the region
            road_region_data[idx]['gdp'] = region_gdp
            road_region_data[idx]['gdp_density'] = gdp_density
            

        
        # svi_indices_within_radius = svi_tree.query_ball_point(region_centroid, radius)
        # for svi_index in svi_indices_within_radius:
        #     svi_point = svi_loc.iloc[svi_index].geometry
        #     if region.contains(svi_point):
        #         road_region_data[idx]['svis'].append({
        #             'unique_index': svi_loc.iloc[svi_index]['unique_index'],  # Use 'index' field for consistency
        #             'location': (svi_loc.iloc[svi_index].projected_x, svi_loc.iloc[svi_index].projected_y)
        #         })
    
    #####################################################################
    
    # Count the road_regions with no POIs
    count_no_point = sum(len(data['pois']) == 0 for data in road_region_data.values())
    print(f"(Before) Number of road regions without any POI: {count_no_point}")

    unassigned_indices = set(range(len(poi_loc))) - assigned_indices
    print(f"(Before) Number of POIs not assigned to any region: {len(unassigned_indices)}")

    # Assign unassigned POIs to the nearest road_region if near_assign is True
    if near_assign:
        region_with_poi_loc = []
        region_with_poi_2_all_region_id = OrderedDict()
        for idx, data in road_region_data.items():
            pois = data['pois']
            if pois:
                region_with_poi_2_all_region_id[len(region_with_poi_2_all_region_id)] = idx
                region_with_poi_loc.append(region_centers[idx])
                
        region_center_tree = cKDTree(region_with_poi_loc)
        # find the nearest region center for each unassigned poi
        unassigned_indices = list(unassigned_indices)
        _, nearest_region_indices = region_center_tree.query(np.array(poi_loc)[unassigned_indices])
        # add the pois to the corresponding road_region
        for i in tqdm(range(len(unassigned_indices)), desc="Assigning unassigned POIs to nearest road region"):
            region_idx = nearest_region_indices[i]
            region_idx = region_with_poi_2_all_region_id[region_idx]
            poi_idx = unassigned_indices[i]
            if poi_mode == 'filtered':
                road_region_data[region_idx]['pois'].append({
                    'poi_index': poi_ori_index[poi_idx],
                    'location': poi_loc[poi_idx]
                })
            elif poi_mode == 'original' or 'mini':
                road_region_data[region_idx]['pois'].append({
                    'poi_index': poi_idx,
                    'location': poi_loc[poi_idx]
                })

        # DEBUG: check the number of pois in regions are equal to the total number of pois
        debug_poi_idxs = set()
        for data in road_region_data.values():
            for poi in data['pois']:
                debug_poi_idxs.add(poi['poi_index'])
        assert len(debug_poi_idxs) == len(poi_loc), f"Total number of POIs in regions is not equal to the total number of POIs: {len(debug_poi_idxs)} != {len(poi_loc)}"

    # Count the road_regions with no POIs
    count_no_point = sum(len(data['pois']) == 0 for data in road_region_data.values())
    print(f"(After) Number of road regions without any POI: {count_no_point}")

    # Count the road_regions with no houses
    count_no_house = sum(data['average_house_price'] == 0 for data in road_region_data.values())
    print(f"Number of road regions without any house price information: {count_no_house}")
    
    # Count the road_regions with no GDP data
    if gdp_tree is not None:
        count_no_gdp = sum(data['gdp'] == 0 for data in road_region_data.values())
        count_no_gdp_density = sum(data['gdp_density'] == 0 for data in road_region_data.values())
        total_gdp = sum(data['gdp'] for data in road_region_data.values())
        total_gdp_density = sum(data['gdp_density'] for data in road_region_data.values())
        
        print(f"Number of road regions without any GDP data: {count_no_gdp}")
        print(f"Number of road regions without any GDP density data: {count_no_gdp_density}")

        
        # Add debug information about GDP data
        if len(gdp_locations) > 0:
            print(f"Example GDP data: First 3 GDP locations: {gdp_locations[:3]}")
        
        # Check if we have any regions with GDP > 0
        regions_with_gdp = len(road_regions) - count_no_gdp
        print(f"Number of regions with GDP > 0: {regions_with_gdp}")
        
        # Get statistics for non-zero GDP density
        gdp_density_values = [data['gdp_density'] for data in road_region_data.values() if data['gdp_density'] > 0]

        
        # Only calculate average if there are regions with GDP
        if regions_with_gdp > 0:
            print(f"Average GDP per road region (with GDP > 0): {total_gdp / regions_with_gdp:.4f}")
        else:
            print("No regions have GDP data > 0, can't calculate average.")

    # # Count the road_regions with no SVIs
    # count_no_svi = sum(len(data['svis']) == 0 for data in road_region_data.values())
    # print(f"Number of road regions without any SVI: {count_no_svi}")

    # Draw the pics and print the summary of these two data in regions
    # plot_and_summarize(city, road_regions, road_region_data, poi_loc, svi_loc=None)

    ##### Save results into file
    region_data = {}

    for idx, data in road_region_data.items():
        pois = data['pois']
        # svis = data['svis']
        
        if pois:  
            # Save all required info
            region_data[idx] = {
                'region_shape': road_regions[idx], # Save the shape of the region
                'pois': [{'index': poi_info.get('index', poi_info.get('poi_index')), 'location': poi_info['location']} for poi_info in pois], # Save the POIs with consistent index field
                # 'svis': [{'index': svi_info['unique_index'], 'location': svi_info['location']} for svi_info in svis], # Save the SVIs
                'landuse_level1_distribution': data['label_distribution'], # Save the label distribution
                'population': population_dict.get(idx, 0), # Save the population of each road region
                'house_price': data['average_house_price'], # Save the house information
                'gdp': data['gdp_density'], # Save the calculated GDP density value
                }
    # Search in check_region data and find those regions not in region_data, add those region into region_data
    if poi_mode == 'filtered':
        if dataset == 'Gaode':
            check_region_path = f'/home/yuanlong001/BanditRegion/data/{dataset}/processed/Integral/{city_abb.upper()}_data.pkl'
        # elif dataset == 'Meituan':
        #     check_region_path = f'/home_nfs/regionteam/BanditRegion/data/{dataset}/processed/Integral/{city_abb.upper()}_data.pkl'
        
        with open(check_region_path, 'rb') as f:
            check_region_data = pkl.load(f)
        
        # Create a set of current POI locations for quick lookup
        current_poi_set = set()
        for region_data_item in region_data.values():
            for poi_info in region_data_item['pois']:
                current_poi_set.add(tuple(poi_info['location']))
        
        # Create index mapping for new POIs (original index to text)
        # Load original POI text data for mapping
        original_poi_file = f'/home/yuanlong001/BanditRegion/data/{dataset}/projected/{city}/poi.txt'
        index_to_text_map = {}
        with open(original_poi_file, 'r') as f:
            for line_idx, line in enumerate(f):
                fields = line.strip().split('\t')
                index_to_text_map[line_idx] = fields[0]  # Map index to POI text
        
        # Track alignment statistics
        add_region_count = 0
        add_poi_count = 0
        new_pois_to_write = []
        
        # Only fill empty regions in filtered data with original data
        for idx, original_data in check_region_data.items():
            # Only process regions that already exist in filtered data
            if idx in region_data:
                # Only fill if the region exists in filtered data but has NO POIs
                if len(region_data[idx]['pois']) == 0 and len(original_data['pois']) > 0:
                    
                    # Add original POIs to this empty region
                    for poi_info in original_data['pois']:
                        poi_location = tuple(poi_info['location'])
                        if poi_location not in current_poi_set:
                            # Get original POI text and index
                            original_index = poi_info['index']  # This is the index in original poi.txt
                            poi_text_value = index_to_text_map.get(original_index, f"POI_{original_index}")
                            
                            # Add to filtered region data with original index to maintain consistency
                            region_data[idx]['pois'].append({
                                'index': original_index,  # Use original index to maintain consistency with file
                                'location': poi_info['location']
                            })
                            
                            # Prepare to write to file with original index in the last column
                            new_poi_line = f"{poi_text_value}\t{poi_info['location'][0]}\t{poi_info['location'][1]}\t{original_index}"
                            new_pois_to_write.append(new_poi_line)
                            
                            current_poi_set.add(poi_location)
                            add_poi_count += 1
                    
                    if region_data[idx]['pois']:
                        add_region_count += 1
            
            # Add regions that don't exist in filtered data
            elif idx not in region_data:
                region_data[idx] = original_data
                add_region_count += 1
                for poi_info in original_data['pois']:
                    poi_location = tuple(poi_info['location'])
                    if poi_location not in current_poi_set:
                        # Get original POI text and index
                        original_index = poi_info['index']  # This is the index in original poi.txt
                        poi_text_value = index_to_text_map.get(original_index, f"POI_{original_index}")
                        
                        # Prepare to write to file with original index in the last column
                        new_poi_line = f"{poi_text_value}\t{poi_info['location'][0]}\t{poi_info['location'][1]}\t{original_index}"
                        new_pois_to_write.append(new_poi_line)
                        
                        current_poi_set.add(poi_location)
                    add_poi_count += 1
        
        # Write new POIs to the filtered file
        if new_pois_to_write:
            with open(file_path, 'a') as f:
                for poi_line in new_pois_to_write:
                    f.write(poi_line + '\n')
        
        print(f"Number of regions in the final data: {len(region_data)}")
        print(f"Number of added region: {add_region_count}")
        print(f"Number of added pois: {add_poi_count}")
        suburban_dir = get_suburban_dir()
        save_path = os.path.join(suburban_dir, 'data', dataset, 'processed', 'Integral', f'{city_abb.upper()}_data_{drop}_{version}_{top_k}.pkl')
        with open(save_path, 'wb') as f:
            pkl.dump(region_data, f)

    if poi_mode == 'original':
        # For original mode, simply save the processed data without alignment
        print(f"Number of regions in the final data: {len(region_data)}")
        suburban_dir = get_suburban_dir()
        save_path = os.path.join(suburban_dir, 'data', dataset, 'processed', 'Integral', f'{city_abb.upper()}_data.pkl')
        with open(save_path, 'wb') as f:
            pkl.dump(region_data, f)

    print(f"Processed region data saved to {save_path}")


if __name__ == "__main__":
    main()