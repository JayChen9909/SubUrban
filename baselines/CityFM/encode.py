import os
import utm
import argparse
import torch

# Use dynamic path resolution for BanditRegion/model
def get_suburban_dir(*paths):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *paths)

from encoder import encode_nodes

proj_utm_beijing = lambda proj_lat, proj_lon: utm.to_latlon(proj_lat, proj_lon, 50, 'N')
proj_utm_shanghai = lambda proj_lat, proj_lon: utm.to_latlon(proj_lat, proj_lon, 51, 'N')
proj_utm_singapore = lambda proj_lat, proj_lon: utm.to_latlon(proj_lat, proj_lon, 58, 'N')
proj_utm_nyc = lambda proj_lat, proj_lon: utm.to_latlon(proj_lat, proj_lon, 26, 'N')

def city_dataset(city):
    if city in ['Beijing', 'Shanghai']:
        return 'Gaode'
    elif city in ['Singapore', 'NYC']:
        return 'OSM'
    else:
        raise ValueError(f"Unsupported city: {city}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True)
    parser.add_argument('--data_type', type=str, default='poi')
    args = parser.parse_args()

    city = args.city
    data_type = args.data_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_type == 'poi':
        dataset = city_dataset(city)
        input_file = get_suburban_dir('data', dataset, 'projected', city, 'poi.txt')

        my_nodes = []
        with open(input_file, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                try:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        text = ' '.join(parts[:-2])
                        x_str, y_str = parts[-2], parts[-1]
                        try:
                            x, y = float(x_str), float(y_str)
                        except ValueError:
                            continue

                        if city == "Beijing":
                            lat, lon = proj_utm_beijing(x, y)
                        elif city == "Shanghai":
                            lat, lon = proj_utm_shanghai(x, y)
                        elif city == "Singapore":
                            lat, lon = proj_utm_singapore(x, y)
                        elif city == "NYC":
                            lat, lon = proj_utm_nyc(x, y)
                        else:
                            continue

                        my_node = {
                            "type": "node",
                            "id": i,
                            "lat": lat,
                            "lon": lon,
                            "tags": {
                                "text": text
                            }
                        }
                        my_nodes.append(my_node)
                except Exception:
                    continue

    positional_embs, textual_embs = encode_nodes(city, my_nodes, device)
    node_embeddings = torch.cat((positional_embs, textual_embs), dim=1)

    save_dir = get_suburban_dir('embs', 'CityFM', city)
    save_path = os.path.join(save_dir, f'{data_type}_node_embeddings.pt')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(node_embeddings, save_path)
    print(f"Embeddings saved to {save_path}")

if __name__ == '__main__':
    main()