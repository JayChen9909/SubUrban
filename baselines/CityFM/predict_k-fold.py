import torch
import numpy as np
import pickle as pkl
import argparse
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import os

def get_suburban_dir(*paths):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *paths)

def parse_arguments():
    parser = argparse.ArgumentParser(description='CityFM region prediction (k-fold, single city, original mode only)')
    parser.add_argument('--city', type=str, required=True, help='City name (e.g., Beijing, Shanghai, Singapore, NYC)')
    parser.add_argument('--model_type', type=str, choices=['RF','LR'], default='RF', help='Model type: \"RF\" or \"LR\"')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for KFold cross-validation')
    parser.add_argument('--repeats', type=int, default=5, help='Number of times to repeat the cross-validation')
    return parser.parse_args()

def city_abbreviation(city):
    if city == 'Beijing':
        return 'BJ'
    elif city == 'Shanghai':
        return 'SH'
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
    abbr = city_abbreviation(city)
    dataset = city_dataset(city)
    filepath = get_suburban_dir('data', dataset, 'processed', 'Integral', f'{abbr}_data.pkl')
    with open(filepath, 'rb') as file:
        region_data = pkl.load(file)
    return region_data

def load_embeddings(city, task):
    abbr = city_abbreviation(city)
    pkl_path = get_suburban_dir('embs', 'CityFM', city, f'{task}_{abbr}_CityFM.pkl')
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    return data

def prepare_data(city, task):
    region_data = load_region_data(city)
    emb_data = load_embeddings(city, task)
    embeddings_list = []
    label_list = []
    for region_id, region_info in region_data.items():
        emb = np.array(emb_data.get(region_id, {}).get('region_embedding', []))
        has_embedding = emb.size > 0
        has_pois = bool(region_info.get('pois', []))
        if task == 'pop':
            label = region_info.get('population')
            valid_label = label is not None and label > 0
        elif task == 'house':
            label = region_info.get('house_price')
            valid_label = label is not None and label > 0
        elif task == 'gdp':
            label = region_info.get('gdp')
            valid_label = label is not None and label > 0
        else:
            continue
        if has_embedding and has_pois and valid_label:
            embeddings_list.append(emb)
            label_list.append(label)
    embeddings = np.array(embeddings_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)
    return embeddings, labels

def evaluate_rf_repeat(embeddings, labels, model_type, n_splits=5, repeats=5):
    val_r2_list, val_mae_list, val_rmse_list = [], [], []
    test_r2_list, test_mae_list, test_rmse_list = [], [], []
    for i in range(repeats):
        train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(
            embeddings, labels, test_size=0.2, random_state=i
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=i)
        fold_r2, fold_mae, fold_rmse = [], [], []
        for train_idx, val_idx in kf.split(train_val_embeddings):
            X_train, X_val = train_val_embeddings[train_idx], train_val_embeddings[val_idx]
            y_train, y_val = train_val_labels[train_idx], train_val_labels[val_idx]
            if model_type == 'RF':
                model = RandomForestRegressor(n_estimators=100, random_state=i)
            else:
                model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fold_r2.append(r2_score(y_val, y_pred))
            fold_mae.append(mean_absolute_error(y_val, y_pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        val_r2_list.append(np.mean(fold_r2))
        val_mae_list.append(np.mean(fold_mae))
        val_rmse_list.append(np.mean(fold_rmse))
        # test
        if model_type == 'RF':
            final_model = RandomForestRegressor(n_estimators=100, random_state=i)
        else:
            final_model = LinearRegression()
        final_model.fit(train_val_embeddings, train_val_labels)
        test_pred = final_model.predict(test_embeddings)
        test_r2_list.append(r2_score(test_labels, test_pred))
        test_mae_list.append(mean_absolute_error(test_labels, test_pred))
        test_rmse_list.append(np.sqrt(mean_squared_error(test_labels, test_pred)))
    return (
        np.mean(val_r2_list), np.std(val_r2_list),
        np.mean(val_mae_list), np.std(val_mae_list),
        np.mean(val_rmse_list), np.std(val_rmse_list),
        np.mean(test_r2_list), np.std(test_r2_list),
        np.mean(test_mae_list), np.std(test_mae_list),
        np.mean(test_rmse_list), np.std(test_rmse_list)
    )

def main():
    args = parse_arguments()
    city = args.city
    model_type = args.model_type
    n_splits = args.n_splits
    repeats = args.repeats
    if city in ['Beijing', 'Shanghai']:
        tasks =  ['pop','house','gdp']
    elif city in ['Singapore', 'NYC']:
        tasks = ['pop']
    for task in tasks:
        embeddings, labels = prepare_data(city, task)
        print(f"Loaded {len(embeddings)} samples for {city} {task}, embedding dim: {embeddings.shape[1]}")
        results = evaluate_rf_repeat(embeddings, labels, model_type, n_splits=n_splits, repeats=repeats)
        (val_r2, std_val_r2, val_mae, std_val_mae, val_rmse, std_val_rmse,
        test_r2, std_test_r2, test_mae, std_test_mae, test_rmse, std_test_rmse) = results
        print(f"Validation: MAE={val_mae:.4f}(±{std_val_mae:.4f}), RMSE={val_rmse:.4f}(±{std_val_rmse:.4f}), R2={val_r2:.4f}(±{std_val_r2:.4f})")
        print(f"Test: MAE={test_mae:.4f}(±{std_test_mae:.4f}), RMSE={test_rmse:.4f}(±{std_test_rmse:.4f}), R2={test_r2:.4f}(±{std_test_r2:.4f})")

if __name__ == '__main__':
    main()