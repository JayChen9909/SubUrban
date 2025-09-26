import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import os
from scipy.spatial import Delaunay, cKDTree
from torch_geometric.data import Data
import sys
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiPoint, MultiLineString
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
import math
import time
import json
import openai
import re

from torch_geometric.utils import subgraph
from torch_geometric.nn import GATConv, global_mean_pool
from shapely.prepared import prep

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from util_text import load_poi_txt

from collections import Counter
from scipy.stats import multivariate_normal
import re

try:
    import shapely.io
except ImportError:
    import shapely
    from shapely import wkt, wkb
    
    class ShapelyIO:
        @staticmethod
        def dumps(obj, **kwargs):
            if hasattr(wkt, 'dumps'):
                return wkt.dumps(obj, **kwargs)
            else:
                return obj.wkt
        
        @staticmethod
        def loads(s, **kwargs):
            if hasattr(wkt, 'loads'):
                return wkt.loads(s, **kwargs)
            else:
                return wkt.loads(s)
        
        @staticmethod
        def from_wkt(s):
            if hasattr(wkt, 'loads'):
                return wkt.loads(s)
            else:
                return wkt.loads(s)
        
        @staticmethod
        def to_wkt(obj):
            if hasattr(wkt, 'dumps'):
                return wkt.dumps(obj)
            else:
                return obj.wkt
        
        @staticmethod
        def from_wkb(s):
            if hasattr(wkb, 'loads'):
                return wkb.loads(s)
            else:
                return wkb.loads(s)
        
        @staticmethod
        def to_wkb(obj):
            if hasattr(wkb, 'dumps'):
                return wkb.dumps(obj)
            else:
                return obj.wkb
        
        # Add common geometry types
        Point = Point
        Polygon = Polygon
        LineString = LineString
        MultiPoint = MultiPoint
        MultiPolygon = MultiPolygon
        MultiLineString = MultiLineString
    
    # Inject simulated module
    shapely_io_module = ShapelyIO()
    shapely.io = shapely_io_module
    sys.modules['shapely.io'] = shapely_io_module
    
    # Ensure wkt and wkb submodules are also available
    if not hasattr(shapely, 'wkt'):
        shapely.wkt = wkt
    if not hasattr(shapely, 'wkb'):
        shapely.wkb = wkb

def get_suburban_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def city_abbr(city):
    return 'BJ' if city == 'Beijing' else 'SH' if city == 'Shanghai' else None

def load_poi_categories(poi_txt):
    categories = []
    for line in poi_txt:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            categories.append(parts[1].strip())
        else:
            categories.append("Unknown")
    return categories

def create_subgraph_from_region(region_info, large_graph, poi_locations, buffer_value=0, original_to_filtered_mapping=None):
    region_shape = region_info['region_shape']
    poi_indices = [poi['index'] for poi in region_info['pois']]
    mapping_stats = {'total_pois': len(poi_indices), 'failed_mappings': 0}
    
    # If mapping is provided, convert original indices to filtered indices
    if original_to_filtered_mapping is not None:
        original_poi_indices = poi_indices.copy()  # Save original indices for debugging
        
        # Check if each original index can find corresponding mapping
        unmapped_indices = []
        mapped_poi_indices = []
        
        for original_idx in poi_indices:
            if original_idx in original_to_filtered_mapping:
                filtered_idx = original_to_filtered_mapping[original_idx]
                mapped_poi_indices.append(filtered_idx)
            else:
                unmapped_indices.append(original_idx)
        
        mapping_stats['failed_mappings'] = len(unmapped_indices)
        mapping_stats['failed_indices'] = unmapped_indices  # Record specific failed indices
        poi_indices = mapped_poi_indices
    
    nodes_to_keep = set(poi_indices)
    if buffer_value > 0:
        buffered_region = region_shape.buffer(buffer_value)
        poi_locations_adjusted = [(loc[1], loc[0]) for loc in poi_locations]
        filtered_poi_indices = [i for i, loc in enumerate(poi_locations_adjusted) if buffered_region.contains(Point(loc))]
        nodes_to_keep.update(filtered_poi_indices)
    subset = list(nodes_to_keep)
    subset = [node for node in subset if node < large_graph.num_nodes]
    sub_edge_index, _ = subgraph(subset=subset, edge_index=large_graph.edge_index, relabel_nodes=True, num_nodes=large_graph.num_nodes)
    sub_x = large_graph.x[subset]
    valid_edges_mask = (sub_edge_index[0] < sub_x.size(0)) & (sub_edge_index[1] < sub_x.size(0))
    sub_edge_index = sub_edge_index[:, valid_edges_mask]
    deg = torch.zeros(len(subset), dtype=torch.long, device=sub_edge_index.device).scatter_add_(0, sub_edge_index[0], torch.ones_like(sub_edge_index[0]))
    isolated_nodes = torch.arange(sub_x.size(0), device=sub_x.device)[deg == 0]
    if len(isolated_nodes) > 0:
        self_loops = torch.stack([isolated_nodes, isolated_nodes], dim=0)
        sub_edge_index = torch.cat([sub_edge_index, self_loops], dim=1)
    subgraph_data = Data(x=sub_x, edge_index=sub_edge_index)
    subgraph_data.orig_indices = subset
    return subgraph_data, mapping_stats

def compute_region_representation_avg(x):
    return torch.mean(x, dim=0)

def ret_cell_coverage(sg, grid_resolution=0.01):
    global poi_locations
    indices = sg.orig_indices
    if len(indices) == 0:
        return 0.0
    points = poi_locations[indices]
    grid_coords = np.floor(points / grid_resolution)
    unique_grids = {tuple(coord) for coord in grid_coords}
    coverage_reward = len(unique_grids)
    return coverage_reward

def ret_saturation_reward_dynamic(sg, poi_categories, method='mean'):
    indices = sg.orig_indices
    if len(indices) == 0:
        return 0.0
    cat_counts = {}
    for idx in indices:
        cat = poi_categories[idx]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    counts = np.array(list(cat_counts.values()))
    if method == 'median':
        dynamic_threshold = np.median(counts)
    else:
        dynamic_threshold = np.mean(counts)
    reward = 0.0
    for cat, count in cat_counts.items():
        reward += min(count, dynamic_threshold)
    return reward

def evaluate_rf_5fold_single(embeddings, labels, random_state=42):
    labels = np.array(labels)
    
    # First split 8:2 for data
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)), 
        test_size=0.2, 
        random_state=random_state
    )
    
    # Get training validation set and test set
    X_train_val = embeddings[train_val_idx]
    y_train_val = labels[train_val_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]
    
    # Perform 5-fold cross-validation on 80% of data (mainly to maintain original interface)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = []
    
    for fold_train_idx, fold_val_idx in kf.split(X_train_val):
        X_fold_train = X_train_val[fold_train_idx]
        X_fold_val = X_train_val[fold_val_idx]
        y_fold_train = y_train_val[fold_train_idx]
        y_fold_val = y_train_val[fold_val_idx]
        
        # Train model
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_fold_train, y_fold_train)
        
        # Evaluate on validation set
        y_fold_pred = rf.predict(X_fold_val)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        cv_scores.append(fold_r2)
    
    # Train final model with all 80% data
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_final.fit(X_train_val, y_train_val)
    
    # Evaluate on 20% test set (return this result)
    y_test_pred = rf_final.predict(X_test)
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return test_r2, test_mae, test_rmse

def evaluate_rf_repeat_5fold(embeddings, labels, repeats=5):
    r2_list = []
    mae_list = []
    rmse_list = []
    
    for i in range(repeats):
        r2, mae, rmse = evaluate_rf_5fold_single(embeddings, labels, random_state=i)
        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)
    
    return {
        'r2_mean': np.mean(r2_list),
        'r2_std': np.std(r2_list),
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list)
    }


class RegionGAT_PReLU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super(RegionGAT_PReLU, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_channels * heads)
        self.prelu1 = nn.PReLU()
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.ln2 = nn.LayerNorm(out_channels)
        self.prelu2 = nn.PReLU()
        if in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = None

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x1 = self.gat1(x, edge_index)
        x1 = self.ln1(x1)
        x1 = self.prelu1(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.gat2(x1, edge_index)
        x2 = self.ln2(x2)

        if self.res_proj is not None:
            res = self.res_proj(x)
        else:
            res = x
        x_out = x2 + res
        x_out = self.prelu2(x_out)
        region_emb = global_mean_pool(x_out, batch)
        return region_emb


class BufferController(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super(BufferController, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, 1)
        self.fc_log_std = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        raw_delta = mean + std
        delta = F.softplus(raw_delta)
        const = torch.tensor(2 * math.pi, device=state.device, dtype=state.dtype)
        log_prob = -0.5 * (((raw_delta - mean) / (std + 1e-8)) ** 2 + 2 * log_std + torch.log(const))
        return delta, log_prob

# CandidateAttention module, receives candidate embeddings (projected to region_emb dimension by proj) and region_emb
class CandidateAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(CandidateAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, candidate_embs, region_emb, is_training=True):
        # candidate_embs: (L, out_dim)
        # region_emb: (out_dim,)
        query = region_emb.unsqueeze(0).unsqueeze(0)  # (1,1,out_dim)
        key = candidate_embs.unsqueeze(0)             # (1,L,out_dim)
        value = key
        attn_output, attn_weights = self.attention(query, key, value)
        scores = attn_weights.squeeze(0).squeeze(0)
        return scores


# Graph connectivity update function
def update_connectivity(sg, poi_locations):
    indices = sg.orig_indices
    coords = np.array([[poi_locations[i][1], poi_locations[i][0]] for i in indices])
    
    try:
        tri = Delaunay(coords)
        edges = []
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edges.append((simplex[i], simplex[j]))
                    edges.append((simplex[j], simplex[i]))
        new_edge_index = torch.tensor(edges, dtype=torch.long, device=sg.x.device)
    except Exception as e:
        num_points = len(coords)
        if num_points == 0:
            new_edge_index = torch.empty((2, 0), dtype=torch.long, device=sg.x.device)
        elif num_points == 1:
            new_edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=sg.x.device)
        else:
            tree = cKDTree(coords)
            edges = []
            distances, nn_indices = tree.query(coords, k=2)
            for i in range(num_points):
                nearest = nn_indices[i][1]
                edges.append((i, nearest))
                edges.append((nearest, i))
            new_edge_index = torch.tensor(edges, dtype=torch.long, device=sg.x.device)
    
    if new_edge_index.dim() == 2:
        if new_edge_index.size(0) != 2:
            new_edge_index = new_edge_index.t().contiguous()
    else:
        raise ValueError("new_edge_index is not a 2D tensor.")
    
    num_nodes = sg.x.size(0)
    if new_edge_index.numel() > 0 and new_edge_index.max().item() >= num_nodes:
        new_edge_index = new_edge_index.clamp(max=num_nodes - 1)
    
    sg.edge_index = new_edge_index
    return sg

def compute_norm_stats(subgraphs, poi_categories, train_ids):
    states = []
    for region_id in train_ids:
        sg, _, region_info = subgraphs[region_id]
        cov = ret_cell_coverage(sg, grid_resolution=0.01)
        sat = ret_saturation_reward_dynamic(sg, poi_categories, method='mean')
        buffer_val = region_info['buffer']
        state = np.array([cov, sat, buffer_val])
        states.append(state)
    states = np.stack(states, axis=0)
    norm_mean = np.mean(states, axis=0)
    norm_std = np.std(states, axis=0)
    norm_std[norm_std == 0] = 1.0
    return norm_mean, norm_std


def pretrain_buffer_controller(subgraphs, pretrain_ids, buffer_controller, poi_categories, device, num_epochs=20):
    threshold = 1.0
    def reset_module(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    while True:
        print("\n--- Pretraining Buffer Controller ---")
        norm_mean_all, norm_std_all = compute_norm_stats(subgraphs, poi_categories, pretrain_ids)
        norm_mean_12 = torch.tensor(norm_mean_all[:2], dtype=torch.float32).to(device)
        norm_std_12 = torch.tensor(norm_std_all[:2], dtype=torch.float32).to(device)
        
        optimizer_pretrain = torch.optim.Adam(buffer_controller.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        final_epoch_loss = None
        for epoch in range(num_epochs):
            total_loss = 0.0
            count = 0
            for region_id in pretrain_ids:
                sg, _, region_info = subgraphs[region_id]
                cov = ret_cell_coverage(sg, grid_resolution=0.01)
                sat = ret_saturation_reward_dynamic(sg, poi_categories, method='mean')
                current_buffer = region_info['buffer']
                raw_state = torch.tensor([[cov, sat, current_buffer]], dtype=torch.float32).to(device)
                norm_state = torch.cat([
                    (raw_state[:, :2] - norm_mean_12.unsqueeze(0)) / (norm_std_12.unsqueeze(0) + 1e-8),
                    raw_state[:, 2:]  # buffer passed as-is
                ], dim=1)
                
                # Pretraining target: expect expansion absolute value to be current_buffer * 0.2 (e.g., 500×0.2=100)
                target = torch.tensor([[current_buffer * 0.2]], dtype=torch.float32).to(device)
                
                delta_pred, _ = buffer_controller(norm_state)
                
                loss = loss_fn(delta_pred, target)
                optimizer_pretrain.zero_grad()
                loss.backward()
                optimizer_pretrain.step()
                
                total_loss += loss.item()
                count += 1
            epoch_loss = total_loss / count
            print(f"Pretrain Epoch {epoch+1}/{num_epochs}, Loss = {epoch_loss:.4f}")
            final_epoch_loss = epoch_loss
        if final_epoch_loss <= threshold:
            print("--- Pretraining Complete ---\n")
            break
        else:
            print(f"Final epoch loss {final_epoch_loss:.4f} > {threshold}, reinitializing buffer_controller and retraining...")
            buffer_controller.apply(reset_module)
    return buffer_controller

# New CEM optimizer class
class CEMOptimizer:
    def __init__(self, n_categories, category_names, elite_fraction=0.2, smoothing_factor=0.7, 
                 initial_mean=None, initial_std=None, min_weight=0.2, max_weight=2.0):
        # n_categories: Number of POI categories
        self.n_categories = n_categories
        self.elite_fraction = elite_fraction
        self.smoothing_factor = smoothing_factor
        self.min_weight = min_weight  # reduced from 0.5 to 0.2
        self.max_weight = max_weight  # increased from 2.0 to 3.0
        self.category_names = category_names
        
        # Initialize distribution parameters
        if initial_mean is None:
            self.mean = np.ones(n_categories)  # initial weights all 1.0
        else:
            self.mean = initial_mean
            
        if initial_std is None:
            self.std = np.ones(n_categories) * 0.5  # initial standard deviation
        else:
            self.std = initial_std
            
        self.best_weights = None
        self.best_reward = -float('inf')
        self.iteration_history = []
        
    def sample_weights(self, n_samples):
        # Sample weights from multivariate normal distribution
        samples = multivariate_normal.rvs(mean=self.mean, cov=np.diag(self.std**2), size=n_samples)
        # Limit weight range
        samples = np.clip(samples, self.min_weight, self.max_weight)
        return samples
    
    def update_distribution(self, weights, rewards, iteration):
        # Select elite samples
        n_elite = max(1, int(self.elite_fraction * len(weights)))
        elite_idx = np.argsort(rewards)[-n_elite:]
        elite_weights = weights[elite_idx]
        
        # Record current iteration results
        iter_result = {
            'iteration': iteration,
            'mean_reward': np.mean(rewards),
            'best_reward': np.max(rewards),
            'elite_mean_reward': np.mean(rewards[elite_idx]),
            'mean': self.mean.copy(),
            'std': self.std.copy(),
            'elite_weights': elite_weights.copy(),
            'rewards': rewards.copy(),
            'elite_idx': elite_idx.copy(),
        }
        self.iteration_history.append(iter_result)
        
        # Update distribution parameters
        new_mean = np.mean(elite_weights, axis=0)
        new_std = np.std(elite_weights, axis=0)
        
        # Smooth update
        self.mean = self.smoothing_factor * self.mean + (1 - self.smoothing_factor) * new_mean
        self.std = self.smoothing_factor * self.std + (1 - self.smoothing_factor) * new_std
        
        # Update best weights
        best_idx = np.argmax(rewards)
        if rewards[best_idx] > self.best_reward:
            self.best_reward = rewards[best_idx]
            self.best_weights = weights[best_idx]
            
        return self.best_weights, self.best_reward
    
    def summarize_training_info(self, start_iteration, end_iteration, save_path):
        summary = []
        summary.append(f"=== CEM Optimization Phase Summary (Rounds {start_iteration+1} to {end_iteration+1}) ===\n")
        
        # Record initial round information
        start_round_info = self.iteration_history[start_iteration]
        start_best_reward = start_round_info['best_reward']
        start_elite_rewards = start_round_info['rewards'][start_round_info['elite_idx']]
        start_elite_weights = start_round_info['elite_weights']
        
        # Record final round information
        end_round_info = self.iteration_history[end_iteration]
        end_best_reward = end_round_info['best_reward']
        end_elite_rewards = end_round_info['rewards'][end_round_info['elite_idx']]
        end_elite_weights = end_round_info['elite_weights']
        
        # 1. Overall performance change - updated for triple-task description
        summary.append(f"1. Performance Change:")
        summary.append(f"   Starting round {start_iteration+1} best mixed reward: {start_best_reward:.4f}")
        summary.append(f"   Ending round {end_iteration+1} best mixed reward: {end_best_reward:.4f}")
        summary.append(f"   Mixed reward change: {end_best_reward - start_best_reward:.4f}")
        summary.append(f"   Elite sample average mixed reward change: {np.mean(end_elite_rewards) - np.mean(start_elite_rewards):.4f}\n")
        
        # 2. Weight distribution changes
        summary.append(f"2. Weight Distribution Changes:")
        # Calculate average weight changes
        weight_changes = {}
        for cat_idx, cat_name in enumerate(self.category_names):
            start_weights = np.array([weights[cat_idx] for weights in start_round_info['elite_weights']])
            end_weights = np.array([weights[cat_idx] for weights in end_round_info['elite_weights']])
            weight_changes[cat_name] = {
                'start_mean': np.mean(start_weights),
                'end_mean': np.mean(end_weights),
                'change': np.mean(end_weights) - np.mean(start_weights),
                'start_std': np.std(start_weights),
                'end_std': np.std(end_weights),
            }
        
        # Sort and display categories with largest weight changes
        sorted_increases = sorted(weight_changes.items(), key=lambda x: -x[1]['change'])
        sorted_decreases = sorted(weight_changes.items(), key=lambda x: x[1]['change'])
        
        summary.append("   Top 5 categories with most weight increase:")
        for cat_name, info in sorted_increases[:5]:
            summary.append(f"   - {cat_name}: {info['start_mean']:.4f} to {info['end_mean']:.4f} (change: +{info['change']:.4f})")
        
        summary.append("\n   Top 5 categories with most weight decrease:")
        for cat_name, info in sorted_decreases[:5]:
            summary.append(f"   - {cat_name}: {info['start_mean']:.4f} to {info['end_mean']:.4f} (change: {info['change']:.4f})")
        
        # 3. Current weight rankings
        summary.append("\n3. Current Weight Rankings:")
        
        current_weights = {self.category_names[i]: end_round_info['mean'][i] for i in range(len(self.category_names))}
        sorted_high = sorted(current_weights.items(), key=lambda x: -x[1])
        sorted_low = sorted(current_weights.items(), key=lambda x: x[1])
        
        summary.append("   Top 10 categories with highest weights:")
        for cat_name, weight in sorted_high[:10]:
            summary.append(f"   - {cat_name}: {weight:.4f}")
        
        summary.append("\n   Top 10 categories with lowest weights:")
        for cat_name, weight in sorted_low[:10]:
            summary.append(f"   - {cat_name}: {weight:.4f}")
        
        # 4. Iteration process stability
        summary.append("\n4. Optimization Process Stability Analysis:")
        rewards = [info['best_reward'] for info in self.iteration_history[start_iteration:end_iteration+1]]
        reward_deltas = [rewards[i+1] - rewards[i] for i in range(len(rewards)-1)]
        
        summary.append(f"   Average reward change: {np.mean(reward_deltas):.4f}")
        summary.append(f"   Reward change standard deviation: {np.std(reward_deltas):.4f}")
        summary.append(f"   Is optimization converging: {'Yes' if np.std(reward_deltas) < 0.01 else 'No'}")
        
        # 5. Category correlation analysis
        summary.append("\n5. Category Weight and Performance Correlation Analysis:")
        
        # Collect all sample weights and corresponding rewards
        all_weights = np.vstack([info['elite_weights'] for info in self.iteration_history[start_iteration:end_iteration+1]])
        all_rewards = np.concatenate([info['rewards'][info['elite_idx']] for info in self.iteration_history[start_iteration:end_iteration+1]])
        
        # Calculate correlation between each category's weight and rewards
        correlations = {}
        for i, cat_name in enumerate(self.category_names):
            cat_weights = all_weights[:, i]
            if np.std(cat_weights) > 1e-8:  # Only calculate for categories with sufficient variance
                # Add smoothing factor to avoid division by zero
                x_std = np.std(cat_weights) + 1e-8
                y_std = np.std(all_rewards) + 1e-8
                corr = np.mean((cat_weights - np.mean(cat_weights)) * (all_rewards - np.mean(all_rewards))) / (x_std * y_std)
                correlations[cat_name] = corr
            else:
                # Standard deviation too small, consider no correlation
                correlations[cat_name] = 0.0

        # Sort and display categories with highest and lowest correlations
        sorted_corrs = sorted(correlations.items(), key=lambda x: -abs(x[1]))
        
        summary.append("   Top 5 categories with strongest weight-performance correlation:")
        for cat_name, corr in sorted_corrs[:5]:
            summary.append(f"   - {cat_name}: {corr:.4f} {'(positive correlation)' if corr > 0 else '(negative correlation)'}")
        
        # Save summary to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(summary))
        
        return summary


def track_category_growth(subgraphs, region_ids, poi_categories, round_num, phase="train"):
    WEIGHTED_CATEGORIES = [
        "Automotive Sales",
        "Motorcycle Services", 
        "Events and Activities",
        "Road Auxiliary Facilities",
        "Automotive Services",
        "Automotive Repair"
    ]

    total_poi_count = 0
    priority_poi_count = 0
    unique_poi_indices = set()
    
    for region_id in region_ids:
        sg, _, _ = subgraphs[region_id]
        indices = sg.orig_indices
        unique_poi_indices.update(sg.orig_indices)
        
        # Count POIs by category
        for idx in indices:
            total_poi_count += 1
            if idx < len(poi_categories) and poi_categories[idx] in WEIGHTED_CATEGORIES:
                priority_poi_count += 1
    
    total_unique_poi_count = len(unique_poi_indices)

    print(f"\n--- {phase.capitalize()} Round {round_num} Category Statistics ---")
    print(f"Total POIs (with duplications): {total_poi_count}")
    print(f"Total POIs (w/o duplications): {total_unique_poi_count}")
    print(f"Priority category POIs: {priority_poi_count} ({priority_poi_count/total_poi_count*100:.2f}%)")
    
    return total_poi_count, priority_poi_count

def compute_dynamic_threshold(scores, strategy='dynamic_mean', percentile=70, std_factor=0.5):
    if scores.numel() == 0:
        return 0.0
    
    if strategy == 'dynamic_mean':
        # Use score mean as threshold
        threshold = scores.mean().item()
    
    elif strategy == 'dynamic_adaptive':
        # Adaptive threshold: dynamically adjust based on score distribution characteristics
        mean_score = scores.mean()
        std_score = scores.std()
        
        # If score distribution is concentrated, use higher threshold
        if std_score < 0.1:
            threshold = (mean_score + 0.1).item()
        # If score distribution is moderately dispersed, use medium threshold
        elif std_score < 0.3:
            threshold = mean_score.item()
        # If score distribution is highly dispersed, use lower threshold
        else:
            threshold = (mean_score - 0.1 * std_score).item()
            
    else:  # fixed threshold
        threshold = 0.1  # Fixed threshold
    
    return threshold

def track_expansion_statistics(subgraphs, region_ids, round_num, phase="train"):
    zero_expansion_count = 0
    expansion_stats = []
    total_pois_added = 0
    
    for region_id in region_ids:
        sg, _, region_info = subgraphs[region_id]
        if 'last_expansion_count' in region_info:
            expansion_count = region_info['last_expansion_count']  # Changed to dictionary operation
        else:
            expansion_count = 0  # Default to 0 if no record exists
            
        expansion_stats.append(expansion_count)
        total_pois_added += expansion_count
        
        if expansion_count == 0:
            zero_expansion_count += 1
    
    total_regions = len(region_ids)
    zero_expansion_ratio = zero_expansion_count / total_regions * 100 if total_regions > 0 else 0
    avg_expansion = np.mean(expansion_stats) if expansion_stats else 0
    max_expansion = np.max(expansion_stats) if expansion_stats else 0
    
    print(f" {phase.capitalize()} Round {round_num} Expansion Details:")
    print(f"   Non-expanded regions: {zero_expansion_count}/{total_regions} ({zero_expansion_ratio:.1f}%)")
    print(f"   Successfully expanded regions: {total_regions - zero_expansion_count}/{total_regions} ({100-zero_expansion_ratio:.1f}%)")
    print(f"   Total new POIs added: {total_pois_added}")
    print(f"   Average expansion count: {avg_expansion:.2f}")
    print(f"   Maximum expansion count: {max_expansion}")
    
    return zero_expansion_count, expansion_stats

def rl_expand_region(sg, region_shape, candidate_buffer, topk, region_gnn, large_graph,
                    poi_locations, poi_tree, proj, candidate_attention,
                    is_training=True, poi_categories=None, category_weights=None):
    if sg.x.size(0) == 0:
        dummy = torch.zeros((proj.out_features,), device=sg.x.device, requires_grad=True)
        return sg, dummy, dummy.mean(dim=0), torch.tensor(0.0, device=sg.x.device, requires_grad=True), torch.tensor(0.0, device=sg.x.device, requires_grad=True)
    
    batch = torch.zeros(sg.x.size(0), dtype=torch.long).to(sg.x.device)
    region_emb = region_gnn(sg.x, sg.edge_index, batch).squeeze(0)
    
    if region_emb.numel() == 0:
        dummy = torch.zeros((proj.out_features,), device=sg.x.device, requires_grad=True)
        return sg, dummy, dummy.mean(dim=0), torch.tensor(0.0, device=sg.x.device, requires_grad=True), torch.tensor(0.0, device=sg.x.device, requires_grad=True)
    
    buffered_poly = region_shape.buffer(candidate_buffer)
    prepared_region = prep(buffered_poly)
    minx, miny, maxx, maxy = buffered_poly.bounds
    center = [(minx + maxx) / 2, (miny + maxy) / 2]
    radius = np.sqrt((maxx - minx)**2 + (maxy - miny)**2) / 2
    candidate_indices = poi_tree.query_ball_point(center, radius)
    
    # Construct the Candidate pool
    candidate_pool = []
    for idx in candidate_indices:
        if idx in sg.orig_indices:
            continue
        pt = Point(poi_locations[idx][1], poi_locations[idx][0])
        if prepared_region.contains(pt) and (not region_shape.contains(pt)):
            candidate_pool.append(idx)
    
    if len(candidate_pool) == 0:
        dummy = torch.zeros(1, device=sg.x.device, requires_grad=True)
        return sg, region_emb, compute_region_representation_avg(sg.x), dummy, dummy

    # Action for candidate selection
    raw_candidate_embs = large_graph.x[candidate_pool]
    projected_candidate_embs = proj(raw_candidate_embs)
    attn_scores = candidate_attention(projected_candidate_embs, region_emb, is_training=is_training)

    attn_weights = torch.ones_like(attn_scores)
    if category_weights is not None:
        for i, idx in enumerate(candidate_pool):
            if idx < len(poi_categories):
                category = poi_categories[idx]
                if category in category_weights:
                    attn_weights[i] = category_weights[category]
    
    # Apply weights to attention scores
    adjusted_attn_scores = attn_scores * attn_weights
    
    # first statge of candidate reduction based on attention scores
    num_attn = min(int(1.5 * topk), adjusted_attn_scores.size(0))
    attn_topk_indices = torch.topk(adjusted_attn_scores, num_attn).indices
    reduced_candidate_pool = [candidate_pool[i] for i in attn_topk_indices.cpu().numpy()]
    
    # Second stage scoring and selection
    projected_candidates = proj(large_graph.x[torch.tensor(reduced_candidate_pool, device=sg.x.device)])
    base_scores = torch.matmul(projected_candidates, region_emb)
    
    score_weights = torch.ones_like(base_scores)
    if category_weights is not None:
        for i, idx in enumerate(reduced_candidate_pool):
            if idx < len(poi_categories):
                category = poi_categories[idx]
                if category in category_weights:
                    score_weights[i] = category_weights[category]
    
    scores = base_scores * score_weights

    if is_training:
        num_final = min(topk, scores.size(0))
        topk_indices = torch.topk(scores, num_final).indices
        selected_log_probs = torch.log(F.softmax(base_scores, dim=0)[topk_indices]) 
    else:
        if scores.size(0) > 0:
            topk_indices = torch.topk(scores, min(1, scores.size(0))).indices
            selected_log_probs = torch.log(F.softmax(base_scores, dim=0)[topk_indices])
        else:
            topk_indices = torch.tensor([], dtype=torch.long, device=scores.device)
            selected_log_probs = torch.tensor(0.0, device=scores.device, requires_grad=True)

    selected_candidates = [reduced_candidate_pool[i] for i in topk_indices.cpu().numpy()]

    if len(selected_candidates) == 0:
        dummy = torch.zeros(1, device=sg.x.device, requires_grad=True)
        return sg, region_emb, compute_region_representation_avg(sg.x), dummy, dummy
    
    new_x = torch.cat([sg.x, large_graph.x[selected_candidates]], dim=0)
    new_orig_indices = sg.orig_indices + selected_candidates
    new_sg = Data(x=new_x, edge_index=sg.edge_index)
    new_sg.orig_indices = new_orig_indices
    updated_region_emb = compute_region_representation_avg(new_x)
    
    multihead_selection_loss = torch.tensor(0.0, device=sg.x.device, requires_grad=True)
    
    if len(reduced_candidate_pool) > 0 and is_training:
        if base_scores.size(0) > 0:
            target_probs = F.softmax(scores, dim=0)
            pred_probs = F.softmax(base_scores, dim=0) 
            multihead_selection_loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-8))
    
    return new_sg, region_emb, updated_region_emb, selected_log_probs, multihead_selection_loss

def rl_expand_region_with_dynamic_threshold(sg, region_shape, candidate_buffer, topk, region_gnn, large_graph,
                                            poi_locations, poi_tree, proj, candidate_attention,
                                            is_training=True, poi_categories=None, category_weights=None,
                                            threshold_strategy='dynamic_mean'):
    if sg.x.size(0) == 0:
        dummy = torch.zeros((proj.out_features,), device=sg.x.device, requires_grad=True)
        return sg, dummy, dummy.mean(dim=0), torch.tensor(0.0, device=sg.x.device, requires_grad=True), torch.tensor(0.0, device=sg.x.device, requires_grad=True)
    
    batch = torch.zeros(sg.x.size(0), dtype=torch.long).to(sg.x.device)
    region_emb = region_gnn(sg.x, sg.edge_index, batch).squeeze(0)
    
    if region_emb.numel() == 0:
        dummy = torch.zeros((proj.out_features,), device=sg.x.device, requires_grad=True)
        return sg, dummy, dummy.mean(dim=0), torch.tensor(0.0, device=sg.x.device, requires_grad=True), torch.tensor(0.0, device=sg.x.device, requires_grad=True)
    
    buffered_poly = region_shape.buffer(candidate_buffer)
    prepared_region = prep(buffered_poly)
    minx, miny, maxx, maxy = buffered_poly.bounds
    center = [(minx + maxx) / 2, (miny + maxy) / 2]
    radius = np.sqrt((maxx - minx)**2 + (maxy - miny)**2) / 2
    candidate_indices = poi_tree.query_ball_point(center, radius)
    
    candidate_pool = []
    for idx in candidate_indices:
        if idx in sg.orig_indices:
            continue
        pt = Point(poi_locations[idx][1], poi_locations[idx][0])
        if prepared_region.contains(pt) and (not region_shape.contains(pt)):
            candidate_pool.append(idx)
    
    if len(candidate_pool) == 0:
        dummy = torch.zeros(1, device=sg.x.device, requires_grad=True)
        return sg, region_emb, compute_region_representation_avg(sg.x), dummy, dummy

    raw_candidate_embs = large_graph.x[candidate_pool]
    projected_candidate_embs = proj(raw_candidate_embs)
    attn_scores = candidate_attention(projected_candidate_embs, region_emb, is_training=is_training)
    
    attn_weights = torch.ones_like(attn_scores)
    if category_weights is not None:
        for i, idx in enumerate(candidate_pool):
            if idx < len(poi_categories):
                category = poi_categories[idx]
                if category in category_weights:
                    attn_weights[i] = category_weights[category]
    
    adjusted_attn_scores = attn_scores * attn_weights

    attn_threshold = compute_dynamic_threshold(adjusted_attn_scores, threshold_strategy, 
                                             percentile=70, std_factor=0.0)
    attn_valid_mask = adjusted_attn_scores >= attn_threshold
    attn_valid_indices = torch.where(attn_valid_mask)[0]

    max_attn_candidates = min(int(1.5 * topk), len(attn_valid_indices))
    if len(attn_valid_indices) > max_attn_candidates:
        attn_scores_valid = adjusted_attn_scores[attn_valid_indices]
        top_attn_indices = torch.topk(attn_scores_valid, max_attn_candidates).indices
        attn_valid_indices = attn_valid_indices[top_attn_indices]
    
    if len(attn_valid_indices) == 0:
        dummy = torch.zeros(1, device=sg.x.device, requires_grad=True)
        return sg, region_emb, compute_region_representation_avg(sg.x), dummy, dummy
            
    reduced_candidate_pool = [candidate_pool[i] for i in attn_valid_indices.cpu().numpy()]
    
    projected_candidates = proj(large_graph.x[torch.tensor(reduced_candidate_pool, device=sg.x.device)])
    base_scores = torch.matmul(projected_candidates, region_emb)
    
    score_weights = torch.ones_like(base_scores)
    if category_weights is not None:
        for i, idx in enumerate(reduced_candidate_pool):
            if idx < len(poi_categories):
                category = poi_categories[idx]
                if category in category_weights:
                    score_weights[i] = category_weights[category]
    
    scores = base_scores * score_weights
    
    similarity_threshold = compute_dynamic_threshold(scores, threshold_strategy, 
                                                   percentile=60, std_factor=0.0)
    similarity_valid_mask = scores >= similarity_threshold
    similarity_valid_indices = torch.where(similarity_valid_mask)[0]

    if len(similarity_valid_indices) == 0:
        dummy = torch.zeros(1, device=sg.x.device, requires_grad=True)
        return sg, region_emb, compute_region_representation_avg(sg.x), dummy, dummy
    
    valid_scores = scores[similarity_valid_indices]
    sorted_indices = torch.argsort(valid_scores, descending=True)
    
    num_to_select = min(len(sorted_indices), topk)
    final_selected_indices = similarity_valid_indices[sorted_indices[:num_to_select]]
    
    selected_candidates = [reduced_candidate_pool[i] for i in final_selected_indices.cpu().numpy()]

    if is_training and len(selected_candidates) > 0:
        selected_log_probs = torch.log(F.softmax(base_scores[final_selected_indices], dim=0) + 1e-8).sum()
        multihead_selection_loss = -torch.mean(adjusted_attn_scores[attn_valid_indices])
    else:
        selected_log_probs = torch.tensor(0.0, device=sg.x.device, requires_grad=True)
        multihead_selection_loss = torch.tensor(0.0, device=sg.x.device, requires_grad=True)

    if len(selected_candidates) == 0:
        return sg, region_emb, compute_region_representation_avg(sg.x), selected_log_probs, multihead_selection_loss
    
    new_x = torch.cat([sg.x, large_graph.x[selected_candidates]], dim=0)
    new_orig_indices = sg.orig_indices + selected_candidates
    new_sg = Data(x=new_x, edge_index=sg.edge_index)
    new_sg.orig_indices = new_orig_indices
    updated_region_emb = compute_region_representation_avg(new_x)
    
    return new_sg, region_emb, updated_region_emb, selected_log_probs, multihead_selection_loss

def get_llm_analysis(current_summary_path, global_history_path, previous_analyses, llm_type='GPT'):
    if llm_type == 'GPT':
        api_key = "insert-your-gpt-api-key-here"
        base_url = None
        model_name = "gpt-4.1"
    elif llm_type == 'DeepSeek':
        base_url = "https://api.deepseek.com/v1" 
        api_key = "insert-your-deepseek-api-key-here"
        model_name = "deepseek-reasoner" 
    else:
        print("LLM choices are only GPT or DeepSeek.")
        return {
            'analysis': f"Unsupported LLM type: {llm_type}",
            'suggestions': json.dumps({}),
            'applied_changes': ''
        }
        
    with open(current_summary_path, 'r') as f:
        current_summary = f.read()
    
    with open(global_history_path, 'r') as f:
        full_history = f.read()
    

    sections = full_history.split("\n\n===")
    
    if len(sections) <= 3:
        limited_history = full_history
    else:
        limited_history = sections[0] + "\n\n===" + "\n\n===".join(sections[-2:])
    
    print(f"Original history length: {len(full_history)} characters")
    print(f"Processed history length: {len(limited_history)} characters")
    
    
    prompt = f"""
Analyze the CEM optimization process and provide improvement suggestions.

**Important Background**: The current system uses a triple-task mixed reward for optimization, where mixed reward = Population prediction task R² * weight + Housing price prediction task R² * weight + GDP prediction task R² * weight. All "rewards" and "performance" metrics refer to this mixed reward value.

Current 3-round optimization summary:
{current_summary}

Global optimization history summary:
{limited_history}

Please provide the following content:
1. Analysis of the current triple-task mixed reward optimization state, particularly focusing on whether local optimum problems exist
2. Identify which POI categories significantly affect triple-task mixed performance (positive or negative)
3. Specific suggestions on how to adjust CEM parameters:
   - For 3-5 categories with the greatest weight impact, suggest significant adjustments (±0.5 or more)
   - For 5-8 categories with moderate weight impact, suggest moderate adjustments (±0.2 to ±0.4)
   - Whether smoothing_factor needs adjustment, considering more aggressive exploration strategies
   - Whether elite_fraction needs adjustment
   - Provide larger standard deviation (0.2-0.5) for specific categories to increase exploration
4. If optimization stagnates, suggest restarting distribution parameters for at least 3 categories

Please provide specific parameter adjustment suggestions in JSON format as follows:
{{
  "category_adjustments": [
    {{"name": "category_name", "mean_adjustment": 0.5, "std_adjustment": 0.3}}
  ],
  "global_adjustments": {{
    "smoothing_factor": 0.1,
    "elite_fraction": 0.05
  }},
  "restart_categories": ["category1", "category2", "category3"]
}}
"""
    
    try:
        print(f"\n=== Starting {llm_type} API call ===")
        
        if base_url:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = openai.OpenAI(api_key=api_key)
        
        print(f"Using {llm_type} API...")
        # print(f"API Key first 10 characters: {api_key[:10]}...")
        if base_url:
            print(f"Base URL: {base_url}")
        print(f"Model: {model_name}")
        
        if llm_type == 'DeepSeek':
            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes CEM optimization data and provides suggestions in JSON format."},
                {"role": "user", "content": prompt}
            ]
            temperature = 0.0
        else:  # GPT
            messages = [{"role": "user", "content": prompt}]
            temperature = 0.0
        
        print(f"Sending request to {llm_type}...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=20000,
            stream=False  
        )
        
        llm_response = response.choices[0].message.content
        print(f"Received {llm_type} response, length: {len(llm_response)} characters")
        
        if llm_type == 'DeepSeek':
            debug_file = current_summary_path.replace('.txt', '_debug.txt')
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"=== DeepSeek Debug Information ===\n")
                f.write(f"API call time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Base URL: {base_url}\n")
                f.write(f"Input prompt length: {len(prompt)} characters\n")
                f.write(f"Response length: {len(llm_response)} characters\n")
                f.write(f"\n=== Input Prompt ===\n")
                f.write(prompt)
                f.write(f"\n\n=== DeepSeek Response ===\n")
                f.write(llm_response)
            
            print(f"Debug info saved to: {debug_file}")
        
        if not llm_response or len(llm_response.strip()) < 10:
            print(f"Warning: {llm_type} response is empty or too short")
            return {
                'analysis': f"{llm_type} response is empty or too short, API call may have failed",
                'suggestions': json.dumps({}),
                'applied_changes': f'{llm_type} API call may have failed, no valid response received'
            }
        
        print(f"{llm_type} response content preview: {llm_response[:300]}...")
        
        suggestions_json = {}
        try:
            json_start = llm_response.find('{')
            if json_start != -1:
                json_text = llm_response[json_start:]
                bracket_count = 0
                for i, char in enumerate(json_text):
                    if char == '{': 
                        bracket_count += 1
                    elif char == '}': 
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_text = json_text[:i+1]
                            break
                
            
                json_text = re.sub(r'//.*$', '', json_text, flags=re.MULTILINE)
    
                json_text = re.sub(r',\s*}', '}', json_text)
                json_text = re.sub(r',\s*]', ']', json_text)
                
                # print(f"JSON after cleaning comments: {json_text[:200]}...")
                suggestions_json = json.loads(json_text)
                print("JSON parsing successful")
            else:
                print("No JSON format suggestions found in LLM response")
        except json.JSONDecodeError as e:
            print(f"Unable to parse JSON suggestions: {e}")
            print(f"JSON text attempted to parse: {json_text if 'json_text' in locals() else 'No JSON text extracted'}")
    
    except Exception as e:
        print(f"Error calling {llm_type} API: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Detailed error information: {traceback.format_exc()}")
        
        llm_response = f"{llm_type} API call failed: {str(e)}"
        suggestions_json = {}
    
    return {
        'analysis': llm_response,
        'suggestions': json.dumps(suggestions_json, indent=2),
        'applied_changes': ''
    }

def apply_llm_suggestions(cem, llm_analysis):
    """
    Apply CEM optimizer parameter adjustments based on LLM suggestions - supports larger magnitude adjustments
    """
    applied_changes = []
    all_applied_successfully = True  
    total_suggestions = 0  
    successful_applications = 0 
    
    try:
        suggestions = json.loads(llm_analysis['suggestions'])
        if 'global_adjustments' in suggestions:
            globals_adj = suggestions['global_adjustments']
            total_suggestions += len(globals_adj)
            
            if 'smoothing_factor' in globals_adj:
                try:
                    old_val = cem.smoothing_factor
                    new_val = float(globals_adj['smoothing_factor'])
                    new_val = max(0.05, min(0.9, new_val))
                    cem.smoothing_factor = new_val
                    applied_changes.append(f"Adjusted smoothing_factor: {old_val} -> {cem.smoothing_factor}")
                    successful_applications += 1
                except Exception as e:
                    applied_changes.append(f"Failed to adjust smoothing_factor: {str(e)}")
                    all_applied_successfully = False
            
            if 'elite_fraction' in globals_adj:
                try:
                    old_val = cem.elite_fraction
                    new_val = float(globals_adj['elite_fraction'])
                    new_val = max(0.01, min(0.3, new_val))
                    cem.elite_fraction = new_val
                    applied_changes.append(f"Adjusted elite_fraction: {old_val} -> {cem.elite_fraction}")
                    successful_applications += 1
                except Exception as e:
                    applied_changes.append(f"Failed to adjust elite_fraction: {str(e)}")
                    all_applied_successfully = False
        
        if 'category_adjustments' in suggestions:
            category_adjustments = suggestions['category_adjustments']
            total_suggestions += len(category_adjustments)
            
            for adj in category_adjustments:
                try:
                    cat_name = adj['name']
                    if cat_name in cem.category_names:
                        idx = cem.category_names.index(cat_name)
                        category_success = True
                        
                        if 'mean_adjustment' in adj:
                            try:
                                old_mean = cem.mean[idx]
                                adjustment = float(adj['mean_adjustment'])
                                noise = np.random.normal(0, 0.05)
                                cem.mean[idx] += adjustment + noise
                                cem.mean[idx] = np.clip(cem.mean[idx], cem.min_weight, cem.max_weight)
                                applied_changes.append(f"Adjusted '{cat_name}' mean: {old_mean:.4f} -> {cem.mean[idx]:.4f} (with random noise)")
                            except Exception as e:
                                applied_changes.append(f"Failed to adjust '{cat_name}' mean: {str(e)}")
                                category_success = False
                        
                        if 'std_adjustment' in adj:
                            try:
                                old_std = cem.std[idx]
                                adjustment = float(adj['std_adjustment'])
                                cem.std[idx] += adjustment
                                cem.std[idx] = max(0.05, min(0.8, cem.std[idx]))
                                applied_changes.append(f"Adjusted '{cat_name}' standard deviation: {old_std:.4f} -> {cem.std[idx]:.4f}")
                            except Exception as e:
                                applied_changes.append(f"Failed to adjust '{cat_name}' standard deviation: {str(e)}")
                                category_success = False
                        
                        if category_success:
                            successful_applications += 1
                        else:
                            all_applied_successfully = False
                    else:
                        applied_changes.append(f"Category '{cat_name}' does not exist in category list")
                        all_applied_successfully = False
                        
                except Exception as e:
                    applied_changes.append(f"Error processing category adjustment: {str(e)}")
                    all_applied_successfully = False
        
        if 'restart_categories' in suggestions:
            restart_categories = suggestions['restart_categories']
            total_suggestions += len(restart_categories)
            
            for cat_name in restart_categories:
                try:
                    if cat_name in cem.category_names:
                        idx = cem.category_names.index(cat_name)
                        old_mean = cem.mean[idx]
                        old_std = cem.std[idx]
                        
                        cem.mean[idx] = np.random.uniform(0.8, 1.2)
                        cem.std[idx] = np.random.uniform(0.4, 0.6)
                        
                        applied_changes.append(f"Restarted '{cat_name}' distribution: mean {old_mean:.4f}->{cem.mean[idx]:.4f}, std {old_std:.4f}->{cem.std[idx]:.4f}")
                        successful_applications += 1
                    else:
                        applied_changes.append(f"Failed to restart category '{cat_name}': category does not exist")
                        all_applied_successfully = False
                except Exception as e:
                    applied_changes.append(f"Failed to restart '{cat_name}' distribution: {str(e)}")
                    all_applied_successfully = False
    
    except Exception as e:
        applied_changes.append(f"Error applying LLM suggestions: {str(e)}")
        all_applied_successfully = False
    

    llm_analysis['applied_changes'] = "\n".join(applied_changes)

    print(f"\n=== LLM Suggestion Application Results ===")
    print(f"Total suggestions: {total_suggestions}")
    print(f"Successfully applied: {successful_applications}")
    print(f"Application success rate: {successful_applications/total_suggestions*100:.1f}%" if total_suggestions > 0 else "Application success rate: 0.0%")
    
    if all_applied_successfully and total_suggestions > 0:
        print("All LLM suggestions applied successfully!")
    elif successful_applications > 0:
        print(f"Partial LLM suggestions applied successfully ({successful_applications}/{total_suggestions})")
    else:
        print("LLM suggestion application failed")
    
    return applied_changes

def optimize_category_weights_with_cem_triple_task(subgraphs, train_ids, region_gnn, projection_layer, 
                                   large_graph, poi_locations, poi_tree, poi_categories, 
                                   device, candidate_attention, expand_steps=5, fixed_buffer=200.0, 
                                   n_iterations=20, n_samples=50, cem_samples=10, 
                                   population_weight=0.33, housing_weight=0.33, gdp_weight=0.34, city=None, rl_topk=10,llm_type=None, llm_instruct=True):
    start_time = time.time()
    region_gnn.eval()
    projection_layer.eval()
    candidate_attention.eval()
    
    unique_categories = sorted(set(poi_categories))
    n_categories = len(unique_categories)
    category_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    
    print(f"\n--- Optimizing {n_categories} POI category weights using CEM (triple-task R² mixed signal) ---")
    print(f"Population task weight: {population_weight}, Housing task weight: {housing_weight}, GDP task weight: {gdp_weight}")
    
    total_weight = population_weight + housing_weight + gdp_weight
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: weight sum is {total_weight:.6f}, not equal to 1.0, will normalize")
        population_weight = population_weight / total_weight
        housing_weight = housing_weight / total_weight
        gdp_weight = gdp_weight / total_weight
        print(f"Weights normalized: population task={population_weight:.3f}, housing task={housing_weight:.3f}, GDP task={gdp_weight:.3f}")
    
    print(f"Fixed expansion steps: {expand_steps}, Fixed expansion distance: {fixed_buffer}m")
    print(f"CEM iterations: {n_iterations}, samples per round: {n_samples}")
    print(f"Early stopping: stop when mixed reward change < 0.001 for 4 consecutive rounds")
    
    cem = CEMOptimizer(
        n_categories=n_categories,
        category_names=unique_categories,
        elite_fraction=0.15,
        smoothing_factor=0.6,
        min_weight=1.0,
        max_weight=2.0
    )
    
    early_stop_threshold = 0.001 
    early_stop_patience = 3 
    reward_history = []         
    no_improvement_count = 0  
    early_stopped = False        
        
    if llm_instruct:
        if city is None:
            raise ValueError("City parameter must be specified when LLM guidance is enabled")
        if llm_type is None:
            raise ValueError("llm_type parameter must be specified when LLM guidance is enabled")
            
        suburban_dir = get_suburban_dir()
        llm_folder = os.path.join(suburban_dir, 'tmp', city, llm_type)
        os.makedirs(llm_folder, exist_ok=True)
        
        global_history_path = os.path.join(llm_folder, f'cem_global_history_triple_task_sample{cem_samples}_{n_iterations}_allTest.txt')
        with open(global_history_path, 'w') as f: 
            f.write("=== CEM Optimization Global History (Triple-Task) ===\n\n")
        
        llm_analyses = []
        print(f"LLM type: {llm_type}")
        print(f"LLM analysis files will be saved to: {llm_folder}")
    else:
        llm_folder = None
        global_history_path = None
        llm_analyses = None
        print("Will execute pure CEM optimization without LLM guidance")

    # Execute CEM iterations
    for iteration in range(n_iterations):
        print(f"\nCEM iteration {iteration+1}/{n_iterations}")        
    
        sampled_weights_arr = cem.sample_weights(n_samples)
        rewards = []
        
        for sample_idx, weights_arr in enumerate(sampled_weights_arr):
            category_weights = {unique_categories[i]: weights_arr[i] for i in range(n_categories)}
            
            temp_subgraphs = copy.deepcopy(subgraphs)
            for region_id in train_ids:
                sg, region_shape, region_info = temp_subgraphs[region_id]
                orig_buffer = region_info['buffer']
                region_info['buffer'] = fixed_buffer
                current_sg = sg
                for step in range(expand_steps):
                    new_sg, _, updated_region_emb, _, _ = rl_expand_region_with_dynamic_threshold(
                        current_sg, region_shape, fixed_buffer, rl_topk,
                        region_gnn, large_graph, poi_locations, poi_tree, projection_layer,
                        candidate_attention, is_training=False, 
                        poi_categories=poi_categories, category_weights=category_weights,
                        threshold_strategy='dynamic_mean')
                    if updated_region_emb is None:
                        break
                    
                    current_sg = new_sg
                current_sg = update_connectivity(current_sg, poi_locations)
                temp_subgraphs[region_id] = (current_sg, region_shape, region_info)
                
                region_info['buffer'] = orig_buffer

            triple_task_results = compute_triple_task_r2(temp_subgraphs, train_ids, use_weights=True,
                                                   category_weights=category_weights, poi_categories=poi_categories)
        
            population_r2 = triple_task_results['population_r2']
            housing_r2 = triple_task_results['housing_r2']
            gdp_r2 = triple_task_results['gdp_r2']
            mixed_reward = population_weight * population_r2 + housing_weight * housing_r2 + gdp_weight * gdp_r2
            
            rewards.append(mixed_reward)
            
            if (sample_idx + 1) % 10 == 0 or sample_idx == n_samples - 1:
                print(f"Sample {sample_idx+1}/{n_samples}: Population R²={population_r2:.4f}, Housing R²={housing_r2:.4f}, GDP R²={gdp_r2:.4f}, Mixed reward={mixed_reward:.4f}")
        

        best_weights, best_reward = cem.update_distribution(sampled_weights_arr, np.array(rewards), iteration)

        current_round_best_idx = np.argmax(rewards)
        current_round_best_reward = rewards[current_round_best_idx]
        print(f"Iteration {iteration+1} current round best reward: {current_round_best_reward:.4f}")
        print(f"Iteration {iteration+1} historical best weights current performance: {best_reward:.4f}")

        reward_history.append(best_reward)

        if len(reward_history) >= 4:
            reward_change = reward_history[-1] - reward_history[-2]
            
            if reward_change < early_stop_threshold:
                no_improvement_count += 1
                print(f"Round {iteration+1}: mixed reward change {reward_change:.6f} < {early_stop_threshold} (consecutive no improvement rounds: {no_improvement_count}/{early_stop_patience})")
            else:
                no_improvement_count = 0
                print(f"Round {iteration+1}: mixed reward change {reward_change:.6f} >= {early_stop_threshold} (reset consecutive count)")
            
            if no_improvement_count >= early_stop_patience:
                print(f"\nEarly stopping triggered! {early_stop_patience} consecutive rounds with mixed reward change < {early_stop_threshold}")
                print(f"Early termination of CEM optimization at round {iteration+1}")
                early_stopped = True

        best_category_weights = {unique_categories[i]: best_weights[i] for i in range(n_categories)}

        temp_subgraphs_best = copy.deepcopy(subgraphs)
        for region_id in train_ids:
            sg, region_shape, region_info = temp_subgraphs_best[region_id]
            orig_buffer = region_info['buffer']
            region_info['buffer'] = fixed_buffer
            
            current_sg = sg
            for step in range(expand_steps):
                new_sg, _, updated_region_emb, _, _ = rl_expand_region_with_dynamic_threshold(
                    current_sg, region_shape, fixed_buffer, rl_topk,
                    region_gnn, large_graph, poi_locations, poi_tree, projection_layer,
                    candidate_attention, is_training=False, 
                    poi_categories=poi_categories, category_weights=best_category_weights,
                    threshold_strategy='dynamic_mean')
                if updated_region_emb is None:
                    break
                current_sg = new_sg
            
            current_sg = update_connectivity(current_sg, poi_locations)
            temp_subgraphs_best[region_id] = (current_sg, region_shape, region_info)
            region_info['buffer'] = orig_buffer
        
        best_triple_results = compute_triple_task_r2(temp_subgraphs_best, train_ids, use_weights=True,
                                                category_weights=best_category_weights, poi_categories=poi_categories)
        
        print("Top 10 category weights from historical best sample:")
        sorted_cats = sorted(best_category_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        for cat, weight in sorted_cats:
            print(f"  {cat}: {weight:.4f}")

        if early_stopped:
            break
        if llm_instruct:
            if (iteration + 1) % 2 == 0 and iteration >= 3:
                print(f'\nLLM Analysis begins: ...')
                current_summary_path = os.path.join(llm_folder, f'cem_summary_triple_task_rounds_{iteration-1}_to_{iteration+1}_sample{cem_samples}_{n_iterations}_allTest.txt')
                summary = cem.summarize_training_info(iteration-2, iteration, current_summary_path)
                
                llm_analysis = get_llm_analysis(current_summary_path, global_history_path, llm_analyses, llm_type)
                llm_analyses.append(llm_analysis)
                
                apply_llm_suggestions(cem, llm_analysis)
                
                with open(global_history_path, 'a') as f:
                    f.write(f"\n\n=== LLM Analysis #{(iteration+1)//2} (Rounds {iteration-1} to {iteration+1}) ===\n")
                    f.write(f"Current best mixed reward: {cem.best_reward:.4f}\n\n")
                    f.write("LLM Analysis:\n")
                    f.write(llm_analysis['analysis'])
                    f.write("\n\nLLM Suggestions:\n")
                    f.write(llm_analysis['suggestions'])
                    f.write("\n\nApplied Changes:\n")
                    f.write(llm_analysis['applied_changes'])
        else:
            llm_folder = None
            global_history_path = None
            llm_analyses = None
    
    print("\n=== CEM Optimization Complete (Triple-task) ===")
    if early_stopped:
        print(f"Optimization ended at round {iteration+1} through early stopping mechanism")
        print(f"Best mixed reward: {cem.best_reward:.4f}")
        print(f"Reward history: {[f'{r:.4f}' for r in reward_history]}")
    else:
        print(f"Completed all {n_iterations} iterations")
        print(f"Best mixed reward: {cem.best_reward:.4f}")
    
    optimized_weights = {unique_categories[i]: cem.best_weights[i] for i in range(n_categories)}
    
    temp_subgraphs_final = copy.deepcopy(subgraphs)
    for region_id in train_ids:
        sg, region_shape, region_info = temp_subgraphs_final[region_id]
        orig_buffer = region_info['buffer']
        region_info['buffer'] = fixed_buffer
        
        current_sg = sg
        for step in range(expand_steps):
            new_sg, _, updated_region_emb, _, _ = rl_expand_region_with_dynamic_threshold(
                current_sg, region_shape, fixed_buffer, rl_topk,
                region_gnn, large_graph, poi_locations, poi_tree, projection_layer,
                candidate_attention, is_training=False, 
                poi_categories=poi_categories, category_weights=optimized_weights,
                threshold_strategy='dynamic_mean')
            
            if updated_region_emb is None:
                break
            current_sg = new_sg
        
        current_sg = update_connectivity(current_sg, poi_locations)
        temp_subgraphs_final[region_id] = (current_sg, region_shape, region_info)
        region_info['buffer'] = orig_buffer
    
    final_triple_results = compute_triple_task_r2(temp_subgraphs_final, train_ids, use_weights=True,
                                            category_weights=optimized_weights, poi_categories=poi_categories)
    
    print(f"Final best results:")
    print_triple_task_results(final_triple_results, "  ")
    print(f"  Mixed reward = {cem.best_reward:.4f}")
    
    cem_duration = time.time() - start_time
    print(f"\nCEM optimization total time: {cem_duration:.2f}s ({cem_duration/60:.2f}min)")
    
    suburban_dir = get_suburban_dir()
    if llm_instruct:
        save_path = os.path.join(suburban_dir, 'tmp', city, f'optimized_category_weights_triple_task_{llm_type}.pkl')
    else:
        save_path = os.path.join(suburban_dir, 'tmp', city, 'optimized_category_weights_triple_task_noLLM.pkl') 

    with open(save_path, 'wb') as f:
        pkl.dump({
            'weights': optimized_weights,
            'history': cem.iteration_history,
            'best_reward': cem.best_reward,
            'final_triple_results': final_triple_results,
            'population_weight': population_weight,
            'housing_weight': housing_weight,
            'early_stopped': early_stopped,
            'early_stop_iteration': iteration + 1 if early_stopped else None,
            'reward_history': reward_history,
            'early_stop_threshold': early_stop_threshold,
            'early_stop_patience': early_stop_patience
        }, f)
    print(f"Optimized category weights saved to: {save_path}")
    
    return optimized_weights


def train_rl_rounds_with_triple_task_weights(subgraphs, train_ids, region_gnn, projection_layer, buffer_controller,
                    large_graph, poi_locations, poi_tree, rl_topk, rl_rounds, poi_categories, device, 
                    candidate_attention, category_weights, population_weight=0.33, housing_weight=0.33, gdp_weight=0.34,
                    w1=1, w2=1, early_rounds=7, threshold_strategy='dynamic_adaptive'):
    total_weight = population_weight + housing_weight + gdp_weight
    if abs(total_weight - 1.0) > 1e-6:
        population_weight = population_weight / total_weight
        housing_weight = housing_weight / total_weight
        gdp_weight = gdp_weight / total_weight
    start_time = time.time()

    optimizer_buffer = torch.optim.Adam(buffer_controller.parameters(), lr=0.001)
    optimizer_multihead = torch.optim.Adam(candidate_attention.parameters(), lr=0.001)
    optimizer_gnn_proj = torch.optim.Adam(
        list(region_gnn.parameters()) + list(projection_layer.parameters()),
        lr=0.001
    )
    

    reward_history = {
        'delta_r2': [],     
        'delta_buffer': [], 
        'mixed_reward': []   
    }
    min_history_length = 3  
    norm_mean_all, norm_std_all = compute_norm_stats(subgraphs, poi_categories, train_ids)
    norm_mean_12 = torch.tensor(norm_mean_all[:2], dtype=torch.float32).to(device)
    norm_std_12 = torch.tensor(norm_std_all[:2], dtype=torch.float32).to(device)
    initial_results_no_weights = compute_triple_task_r2(subgraphs, train_ids, use_weights=False,
                                                      category_weights=None, poi_categories=poi_categories)
    initial_mixed_no_weights = population_weight * initial_results_no_weights['population_r2'] + housing_weight * initial_results_no_weights['housing_r2'] + gdp_weight * initial_results_no_weights['gdp_r2']
    
    initial_results_with_weights = compute_triple_task_r2(subgraphs, train_ids, use_weights=True,
                                                        category_weights=category_weights, poi_categories=poi_categories)
    initial_mixed_with_weights = population_weight * initial_results_with_weights['population_r2'] + housing_weight * initial_results_with_weights['housing_r2'] + gdp_weight * initial_results_with_weights['gdp_r2']

    best_subgraphs = copy.deepcopy(subgraphs)
    best_model_state = copy.deepcopy(region_gnn.state_dict())
    best_proj_state = copy.deepcopy(projection_layer.state_dict())
    
    baseline_reward_buffer = 0.0  # Buffer controller baseline 
    baseline_reward_multihead = 0.0  # MultiHead attention baseline reward
    baseline_reward_gnn = 0.0  # GNN+Projection baseline reward
    region0_old_buffer = None
    region0_cov_before = None
    region0_cov_after = None
    region0_sat_before = None
    region0_sat_after = None
    epsilon = 1e-8

    current_mixed_no_weights = initial_mixed_no_weights
    current_mixed_with_weights = initial_mixed_with_weights

    reward_history['mixed_reward'].append(initial_mixed_with_weights)

    initial_total_buffer = sum([subgraphs[rid][2]['buffer'] for rid in train_ids])

    for rnd in range(rl_rounds):

        
        old_buffer_controller = copy.deepcopy(buffer_controller)
        total_buffer_loss = []
        total_selection_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_sat_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_regions = 0
        total_penalty = 0.0

        round_start_total_buffer = sum([subgraphs[rid][2]['buffer'] for rid in train_ids])

        for region_id in train_ids:
            _, _, region_info = subgraphs[region_id]
            region_info['last_expansion_count'] = 0



        for region_id in train_ids:
            sg, region_shape, region_info = subgraphs[region_id]
            num_regions += 1
            original_poi_count = len(sg.orig_indices)
            if region_id == train_ids[0]:
                region0_old_buffer = region_info['buffer'] if rnd > 0 else 0
                region0_cov_before = ret_cell_coverage(sg, grid_resolution=0.01)
                region0_sat_before = ret_saturation_reward_dynamic(sg, poi_categories, method='mean')

            sat_before = ret_saturation_reward_dynamic(sg, poi_categories, method='mean')
            coverage_before = ret_cell_coverage(sg, grid_resolution=0.01)
            
            new_sg, region_emb, updated_region_emb, log_probs, selection_loss = rl_expand_region_with_dynamic_threshold(
                sg, region_shape, region_info['buffer'], rl_topk,
                region_gnn, large_graph, poi_locations, poi_tree, projection_layer,
                candidate_attention, is_training=True, poi_categories=poi_categories,
                category_weights=category_weights, threshold_strategy=threshold_strategy)
            
            
            new_poi_count = len(new_sg.orig_indices) if hasattr(new_sg, 'orig_indices') else original_poi_count
            expansion_count = max(0, new_poi_count - original_poi_count)
            region_info['last_expansion_count'] = expansion_count
            
        
            sat_after = ret_saturation_reward_dynamic(new_sg, poi_categories, method='mean')
            coverage_after = ret_cell_coverage(new_sg, grid_resolution=0.01)
            delta_sat = sat_after - sat_before
            delta_coverage = coverage_after - coverage_before
            
            
            try:
                total_selection_loss = total_selection_loss + selection_loss
                
                sigma_cov = norm_std_all[0] if norm_std_all[0] > 1e-8 else 1e-8  
                sigma_sat = norm_std_all[1] if norm_std_all[1] > 1e-8 else 1e-8  
                combined_delta = delta_sat / sigma_sat + delta_coverage / sigma_cov
                if isinstance(combined_delta, torch.Tensor):
                    total_sat_loss = total_sat_loss + combined_delta
                else:
                    total_sat_loss = total_sat_loss + torch.tensor(combined_delta, device=device, requires_grad=True)
            except RuntimeError as e:
                if "broadcast" in str(e) or "shape" in str(e):
                    if isinstance(selection_loss, torch.Tensor) and selection_loss.numel() > 1:
                        selection_loss_processed = selection_loss.sum()
                    elif isinstance(selection_loss, torch.Tensor):
                        selection_loss_processed = selection_loss
                    else:
                        selection_loss_processed = torch.tensor(selection_loss, device=device, requires_grad=True)
                    total_selection_loss = total_selection_loss + selection_loss_processed
                    
                    sigma_cov = norm_std_all[0] if norm_std_all[0] > 1e-8 else 1e-8
                    sigma_sat = norm_std_all[1] if norm_std_all[1] > 1e-8 else 1e-8
                    normalized_sat_loss = delta_sat / sigma_sat + delta_coverage / sigma_cov
                    if isinstance(normalized_sat_loss, torch.Tensor):
                        sat_loss_processed = normalized_sat_loss
                    else:
                        sat_loss_processed = torch.tensor(normalized_sat_loss, device=device, requires_grad=True)
                    total_sat_loss = total_sat_loss + sat_loss_processed
                else:
                    raise e

            current_sg = update_connectivity(new_sg, poi_locations)
            subgraphs[region_id] = (current_sg, region_shape, region_info)
            
            cov_after = ret_cell_coverage(current_sg, grid_resolution=0.01)
            sat_after = ret_saturation_reward_dynamic(current_sg, poi_categories, method='mean')
            if region_id == train_ids[0]:
                region0_cov_after = cov_after
                region0_sat_after = sat_after
            
            raw_state = torch.tensor([[cov_after, sat_after, region_info['buffer']]], dtype=torch.float32).to(device)
            norm_state = torch.cat([
                (raw_state[:, :2] - norm_mean_12.unsqueeze(0)) / (norm_std_12.unsqueeze(0) + epsilon),
                raw_state[:, 2:] 
            ], dim=1)
            
            old_delta, old_log_prob = old_buffer_controller(norm_state)
            new_delta, new_log_prob = buffer_controller(norm_state)
            ratio = torch.exp(new_log_prob - old_log_prob)
            
            delta_value = new_delta.item() if math.isfinite(new_delta.item()) else 0.0
            clipping_max = 0.3 * region_info['buffer']
            delta_clipped = max(min(delta_value, clipping_max), 0)
            new_buffer = region_info['buffer'] + delta_clipped

            
            desired_delta = 0.2 * region_info['buffer']
            lambda_penalty = 1
            region_penalty = lambda_penalty * max(delta_value - desired_delta, 0)
            total_penalty += region_penalty
            
            region_info['buffer'] = new_buffer
            total_buffer_loss.append((ratio, log_probs.mean(), region_id))
        
        zero_expansion_count, expansion_stats = track_expansion_statistics(subgraphs, train_ids, rnd+1, "train")

        after_results_no_weights = compute_triple_task_r2(subgraphs, train_ids, use_weights=False,
                                                        category_weights=None, poi_categories=poi_categories)
        after_mixed_no_weights = population_weight * after_results_no_weights['population_r2'] + housing_weight * after_results_no_weights['housing_r2'] + gdp_weight * after_results_no_weights['gdp_r2']
        
        after_results_with_weights = compute_triple_task_r2(subgraphs, train_ids, use_weights=True,
                                                          category_weights=category_weights, poi_categories=poi_categories)
        after_mixed_with_weights = population_weight * after_results_with_weights['population_r2'] + housing_weight * after_results_with_weights['housing_r2'] + gdp_weight * after_results_with_weights['gdp_r2']
        
        delta_mixed_no_weights = after_mixed_no_weights - current_mixed_no_weights
        delta_mixed_with_weights = after_mixed_with_weights - current_mixed_with_weights
        
        round_end_total_buffer = sum([subgraphs[rid][2]['buffer'] for rid in train_ids])
        delta_total_buffer = (round_end_total_buffer - round_start_total_buffer) / len(train_ids)
        
        print(f"Triple-task evaluation after round {rnd+1}:")
        print("Without weights:")
        print_triple_task_results(after_results_no_weights, "  ")
        print(f"  Mixed reward = {after_mixed_no_weights:.4f}")
        print("With weights:")
        print_triple_task_results(after_results_with_weights, "  ")
        print(f"  Mixed reward = {after_mixed_with_weights:.4f}")
        print(f"Round {rnd+1}: Mixed reward change (without weights) Δ = {delta_mixed_no_weights:.4f}")
        print(f"Round {rnd+1}: Mixed reward change (with weights) Δ = {delta_mixed_with_weights:.4f}")
        print(f"Round {rnd+1}: Total buffer change Δ = {delta_total_buffer:.4f}")
        print(f"With weights vs without weights improvement: {(after_mixed_with_weights - after_mixed_no_weights) / after_mixed_no_weights * 100:.2f}%")

        penalty_term = total_penalty / num_regions if num_regions > 0 else 0.0
        
        
        reward_history['delta_r2'].append(delta_mixed_with_weights)
        reward_history['delta_buffer'].append(delta_total_buffer)
        reward_history['mixed_reward'].append(after_mixed_with_weights) 
        
        if len(reward_history['delta_r2']) >= min_history_length:
            sigma_r2 = np.std(reward_history['delta_r2']) if np.std(reward_history['delta_r2']) > 1e-8 else 1e-8
            sigma_buffer = np.std(reward_history['delta_buffer']) if np.std(reward_history['delta_buffer']) > 1e-8 else 1e-8
            
            global_reward_buffer = (delta_mixed_with_weights / sigma_r2) + 0.1 * (delta_total_buffer / sigma_buffer) - penalty_term
            global_reward_multihead = delta_mixed_with_weights / sigma_r2
        else: 
            global_reward_buffer = w1 * delta_mixed_with_weights + 0.1 * delta_total_buffer - penalty_term
            global_reward_multihead = w1 * delta_mixed_with_weights

        avg_combined_change = total_sat_loss.item() / num_regions if num_regions > 0 else 0.0
        global_reward_gnn = avg_combined_change
        

        

        t = rnd + 1  
        epsilon = 1e-8
        
        if len(reward_history['mixed_reward']) >= 2:
            R_t_minus_1 = reward_history['mixed_reward'][-1]  
            R_t_minus_2 = reward_history['mixed_reward'][-2] 

            reward_change = abs(R_t_minus_1 - R_t_minus_2)
            reward_magnitude = abs(R_t_minus_1) + epsilon
            
            gradient_ratio = reward_change / reward_magnitude
            
            gamma = 1.0 / (1.0 + math.exp(-gradient_ratio))
            
            baseline_reward_buffer = gamma * baseline_reward_buffer + (1 - gamma) * global_reward_buffer
            baseline_reward_multihead = gamma * baseline_reward_multihead + (1 - gamma) * global_reward_multihead
            baseline_reward_gnn = gamma * baseline_reward_gnn + (1 - gamma) * global_reward_gnn
            
        else:
            baseline_reward_buffer = global_reward_buffer
            baseline_reward_multihead = global_reward_multihead
            baseline_reward_gnn = global_reward_gnn
        
        global_avg_buffer = np.mean([subgraphs[rid][2]['buffer'] for rid in train_ids])
        

        buffer_loss_list = []
        advantage_buffer = global_reward_buffer - baseline_reward_buffer
        for (ratio, region_adv, region_id) in total_buffer_loss:
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            ppo_loss = -torch.min(ratio * advantage_buffer, clipped_ratio * advantage_buffer)
            buffer_loss_list.append(ppo_loss)
        buffer_loss = sum(buffer_loss_list) / len(buffer_loss_list) if len(buffer_loss_list) > 0 else torch.tensor(0.0, device=device)
        
        advantage_multihead = global_reward_multihead - baseline_reward_multihead
        if num_regions > 0:
            multihead_loss = -(total_selection_loss / num_regions) * advantage_multihead
        else:
            multihead_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        advantage_gnn = global_reward_gnn - baseline_reward_gnn
        if num_regions > 0:
            gnn_proj_loss = -(total_sat_loss / num_regions) * advantage_gnn * w2
        else:
            gnn_proj_loss = torch.tensor(0.0, device=device, requires_grad=True)


        
        if rnd > early_rounds and delta_mixed_with_weights < -0.02:
            region_gnn.load_state_dict(best_model_state)
            projection_layer.load_state_dict(best_proj_state)
            break
        else:
            best_model_state = copy.deepcopy(region_gnn.state_dict())
            best_proj_state = copy.deepcopy(projection_layer.state_dict())
            best_subgraphs = copy.deepcopy(subgraphs)
        
        current_mixed_no_weights = after_mixed_no_weights
        current_mixed_with_weights = after_mixed_with_weights
        
        optimizer_buffer.zero_grad()
        buffer_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(buffer_controller.parameters(), max_norm=10)
        optimizer_buffer.step()
        
        optimizer_multihead.zero_grad()
        multihead_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(candidate_attention.parameters(), max_norm=10)
        optimizer_multihead.step()
    
        optimizer_gnn_proj.zero_grad()
        gnn_proj_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(region_gnn.parameters()) + list(projection_layer.parameters()), max_norm=10)
        optimizer_gnn_proj.step()
    
    train_duration = time.time() - start_time
    
    final_results_no_weights = compute_triple_task_r2(subgraphs, train_ids, use_weights=False,
                                                    category_weights=None, poi_categories=poi_categories)
    final_mixed_no_weights = population_weight * final_results_no_weights['population_r2'] + housing_weight * final_results_no_weights['housing_r2'] + gdp_weight * final_results_no_weights['gdp_r2']
    
    final_results_with_weights = compute_triple_task_r2(subgraphs, train_ids, use_weights=True,
                                                      category_weights=category_weights, poi_categories=poi_categories)
    final_mixed_with_weights = population_weight * final_results_with_weights['population_r2'] + housing_weight * final_results_with_weights['housing_r2'] + gdp_weight * final_results_with_weights['gdp_r2']
    
    print(f"\nFinal training set evaluation:")
    print("Without weights:")
    print_triple_task_results(final_results_no_weights, "  ")
    print(f"  Mixed reward = {final_mixed_no_weights:.4f}")
    print("With weights:")
    print_triple_task_results(final_results_with_weights, "  ")
    print(f"  Mixed reward = {final_mixed_with_weights:.4f}")
    print(f"Mixed reward improvement: {(final_mixed_with_weights - final_mixed_no_weights) / final_mixed_no_weights * 100:.2f}%")
    return subgraphs, norm_mean_12, norm_std_12

def compute_triple_task_r2(subgraphs, region_ids, use_weights=False, category_weights=None, poi_categories=None):
    if not use_weights:
        pop_embs = []
        pop_labels = []
        house_embs = []
        house_labels = []
        gdp_embs = []
        gdp_labels = []
        
        for region_id in region_ids:
            sg, _, region_info = subgraphs[region_id]
            if sg.x.size(0) == 0:
                continue
                
            emb = compute_region_representation_avg(sg.x)
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                continue
            
            emb_np = emb.cpu().numpy()
            
            if 'population' in region_info and region_info['population'] and region_info['population'] > 0:
                pop_embs.append(emb_np)
                pop_labels.append(region_info['population'])

            if 'house_price' in region_info and region_info['house_price'] and region_info['house_price'] > 0:
                house_embs.append(emb_np)
                house_labels.append(region_info['house_price'])
            
            if 'gdp' in region_info and region_info['gdp'] and region_info['gdp'] > 0:
                gdp_embs.append(emb_np)
                gdp_labels.append(region_info['gdp'])
                
    else:
        pop_embs = []
        pop_labels = []
        house_embs = []
        house_labels = []
        gdp_embs = []
        gdp_labels = []
        
        for region_id in region_ids:
            sg, _, region_info = subgraphs[region_id]
            
            if sg.x.size(0) == 0:
                continue

            weighted_sum = torch.zeros_like(sg.x[0])
            total_weight = 0.0
            
            for i, idx in enumerate(sg.orig_indices):
                if idx < len(poi_categories):
                    category = poi_categories[idx]
                    weight = category_weights.get(category, 1.0)
                    weighted_sum += sg.x[i] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_emb = weighted_sum / total_weight
            else:
                weighted_emb = compute_region_representation_avg(sg.x)
            
            if torch.isnan(weighted_emb).any() or torch.isinf(weighted_emb).any():
                continue
                
            emb_np = weighted_emb.cpu().numpy()
            if 'population' in region_info and region_info['population'] and region_info['population'] > 0:
                pop_embs.append(emb_np)
                pop_labels.append(region_info['population'])
            
            if 'house_price' in region_info and region_info['house_price'] and region_info['house_price'] > 0:
                house_embs.append(emb_np)
                house_labels.append(region_info['house_price'])
            
            if 'gdp' in region_info and region_info['gdp'] and region_info['gdp'] > 0:
                gdp_embs.append(emb_np)
                gdp_labels.append(region_info['gdp'])
    
    if len(pop_embs) > 0:
        pop_embs = np.stack(pop_embs, axis=0)
        # population_results = evaluate_rf_repeat_30(pop_embs, np.array(pop_labels), repeats=30)
        population_results = evaluate_rf_repeat_5fold(pop_embs, np.array(pop_labels), repeats=5)

    else:
        population_results = {
            'r2_mean': 0.0, 'r2_std': 0.0,
            'mae_mean': 0.0, 'mae_std': 0.0,
            'rmse_mean': 0.0, 'rmse_std': 0.0
        }

    if len(house_embs) > 0:
        house_embs = np.stack(house_embs, axis=0)
        # housing_results = evaluate_rf_repeat_30(house_embs, np.array(house_labels), repeats=30)
        housing_results = evaluate_rf_repeat_5fold(house_embs, np.array(house_labels), repeats=5)
    else:
        housing_results = {
            'r2_mean': 0.0, 'r2_std': 0.0,
            'mae_mean': 0.0, 'mae_std': 0.0,
            'rmse_mean': 0.0, 'rmse_std': 0.0
        }
    
    if len(gdp_embs) > 0:
        gdp_embs = np.stack(gdp_embs, axis=0)
        gdp_results = evaluate_rf_repeat_5fold(gdp_embs, np.array(gdp_labels), repeats=5)
    else:
        gdp_results = {
            'r2_mean': 0.0, 'r2_std': 0.0,
            'mae_mean': 0.0, 'mae_std': 0.0,
            'rmse_mean': 0.0, 'rmse_std': 0.0
        }
    
    return {
  
        'population_r2': population_results['r2_mean'],
        'population_r2_std': population_results['r2_std'],
        'population_mae': population_results['mae_mean'],
        'population_mae_std': population_results['mae_std'],
        'population_rmse': population_results['rmse_mean'],
        'population_rmse_std': population_results['rmse_std'],

        'housing_r2': housing_results['r2_mean'],
        'housing_r2_std': housing_results['r2_std'],
        'housing_mae': housing_results['mae_mean'],
        'housing_mae_std': housing_results['mae_std'],
        'housing_rmse': housing_results['rmse_mean'],
        'housing_rmse_std': housing_results['rmse_std'],
        
        'gdp_r2': gdp_results['r2_mean'],
        'gdp_r2_std': gdp_results['r2_std'],
        'gdp_mae': gdp_results['mae_mean'],
        'gdp_mae_std': gdp_results['mae_std'],
        'gdp_rmse': gdp_results['rmse_mean'],
        'gdp_rmse_std': gdp_results['rmse_std'],
        
        'pop_embeddings': pop_embs if len(pop_embs) > 0 else None,
        'house_embeddings': house_embs if len(house_embs) > 0 else None,
        'gdp_embeddings': gdp_embs if len(gdp_embs) > 0 else None
    }

def compute_triple_task_r2_final_test(subgraphs, region_ids, use_weights=False, category_weights=None, poi_categories=None):
    if not use_weights:
        pop_embs = []
        pop_labels = []
        house_embs = []
        house_labels = []
        gdp_embs = []
        gdp_labels = []
        
        for region_id in region_ids:
            sg, _, region_info = subgraphs[region_id]
            if sg.x.size(0) == 0:
                continue
                
            emb = compute_region_representation_avg(sg.x)

            if torch.isnan(emb).any() or torch.isinf(emb).any():
                continue
            
            emb_np = emb.cpu().numpy()

            if 'population' in region_info and region_info['population'] and region_info['population'] > 0:
                pop_embs.append(emb_np)
                pop_labels.append(region_info['population'])
 
            if 'house_price' in region_info and region_info['house_price'] and region_info['house_price'] > 0:
                house_embs.append(emb_np)
                house_labels.append(region_info['house_price'])
            
            if 'gdp' in region_info and region_info['gdp'] and region_info['gdp'] > 0:
                gdp_embs.append(emb_np)
                gdp_labels.append(region_info['gdp'])
                
    else:
        pop_embs = []
        pop_labels = []
        house_embs = []
        house_labels = []
        gdp_embs = []
        gdp_labels = []
        
        for region_id in region_ids:
            sg, _, region_info = subgraphs[region_id]

            if sg.x.size(0) == 0:
                continue

            weighted_sum = torch.zeros_like(sg.x[0])
            total_weight = 0.0
            
            for i, idx in enumerate(sg.orig_indices):
                if idx < len(poi_categories):
                    category = poi_categories[idx]
                    weight = category_weights.get(category, 1.0)
                    weighted_sum += sg.x[i] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_emb = weighted_sum / total_weight
            else:
                weighted_emb = compute_region_representation_avg(sg.x)

            if torch.isnan(weighted_emb).any() or torch.isinf(weighted_emb).any():
                continue
                
            emb_np = weighted_emb.cpu().numpy()
        
            if 'population' in region_info and region_info['population'] and region_info['population'] > 0:
                pop_embs.append(emb_np)
                pop_labels.append(region_info['population'])
            
            if 'house_price' in region_info and region_info['house_price'] and region_info['house_price'] > 0:
                house_embs.append(emb_np)
                house_labels.append(region_info['house_price'])
            
            if 'gdp' in region_info and region_info['gdp'] and region_info['gdp'] > 0:
                gdp_embs.append(emb_np)
                gdp_labels.append(region_info['gdp'])
    
    if len(pop_embs) > 0:
        pop_embs = np.stack(pop_embs, axis=0)
        population_results = evaluate_rf_repeat_5fold(pop_embs, np.array(pop_labels), repeats=5)
    else:
        population_results = {
            'r2_mean': 0.0, 'r2_std': 0.0,
            'mae_mean': 0.0, 'mae_std': 0.0,
            'rmse_mean': 0.0, 'rmse_std': 0.0
        }
    
    
    if len(house_embs) > 0:
        house_embs = np.stack(house_embs, axis=0)
        housing_results = evaluate_rf_repeat_5fold(house_embs, np.array(house_labels), repeats=5)
    else:
        housing_results = {
            'r2_mean': 0.0, 'r2_std': 0.0,
            'mae_mean': 0.0, 'mae_std': 0.0,
            'rmse_mean': 0.0, 'rmse_std': 0.0
        }
    
    
    if len(gdp_embs) > 0:
        gdp_embs = np.stack(gdp_embs, axis=0)
        gdp_results = evaluate_rf_repeat_5fold(gdp_embs, np.array(gdp_labels), repeats=5)
    else:
        gdp_results = {
            'r2_mean': 0.0, 'r2_std': 0.0,
            'mae_mean': 0.0, 'mae_std': 0.0,
            'rmse_mean': 0.0, 'rmse_std': 0.0
        }
    
    return {
        
        'population_r2': population_results['r2_mean'],
        'population_r2_std': population_results['r2_std'],
        'population_mae': population_results['mae_mean'],
        'population_mae_std': population_results['mae_std'],
        'population_rmse': population_results['rmse_mean'],
        'population_rmse_std': population_results['rmse_std'],
        
        'housing_r2': housing_results['r2_mean'],
        'housing_r2_std': housing_results['r2_std'],
        'housing_mae': housing_results['mae_mean'],
        'housing_mae_std': housing_results['mae_std'],
        'housing_rmse': housing_results['rmse_mean'],
        'housing_rmse_std': housing_results['rmse_std'],
    
        'gdp_r2': gdp_results['r2_mean'],
        'gdp_r2_std': gdp_results['r2_std'],
        'gdp_mae': gdp_results['mae_mean'],
        'gdp_mae_std': gdp_results['mae_std'],
        'gdp_rmse': gdp_results['rmse_mean'],
        'gdp_rmse_std': gdp_results['rmse_std'],
    
        'pop_embeddings': pop_embs if len(pop_embs) > 0 else None,
        'house_embeddings': house_embs if len(house_embs) > 0 else None
    }

def print_triple_task_results(triple_results, prefix=""):
    print(f"{prefix}Population prediction:")
    print(f"  R² = {triple_results['population_r2']:.4f}(±{triple_results['population_r2_std']:.4f})")
    print(f"  MAE = {triple_results['population_mae']:.4f}(±{triple_results['population_mae_std']:.4f})")
    print(f"  RMSE = {triple_results['population_rmse']:.4f}(±{triple_results['population_rmse_std']:.4f})")
    print(f"{prefix}Housing price prediction:")
    print(f"  R² = {triple_results['housing_r2']:.4f}(±{triple_results['housing_r2_std']:.4f})")
    print(f"  MAE = {triple_results['housing_mae']:.4f}(±{triple_results['housing_mae_std']:.4f})")
    print(f"  RMSE = {triple_results['housing_rmse']:.4f}(±{triple_results['housing_rmse_std']:.4f})")
    
    if 'gdp_r2' in triple_results:
        print(f"{prefix}GDP prediction:")
        print(f"  R² = {triple_results['gdp_r2']:.4f}(±{triple_results['gdp_r2_std']:.4f})")
        print(f"  MAE = {triple_results['gdp_mae']:.4f}(±{triple_results['gdp_mae_std']:.4f})")
        print(f"  RMSE = {triple_results['gdp_rmse']:.4f}(±{triple_results['gdp_rmse_std']:.4f})")

def testing_phase_with_triple_tasks_enhanced(subgraphs, test_ids, region_gnn, projection_layer, buffer_controller,
                                          large_graph, poi_locations, poi_tree, poi_categories, norm_mean_12, norm_std_12, device,
                                          category_weights, population_weight=0.33, housing_weight=0.33, gdp_weight=0.34,
                                          test_rounds=10, rl_topk=10, candidate_attention=None, city=None, mode=None):
    print("\n--- Test Phase (Population+Housing+GDP Triple-Task Evaluation - Enhanced) ---")
    print(f"Population task weight: {population_weight}, Housing task weight: {housing_weight}, GDP task weight: {gdp_weight}")
    

    total_weight = population_weight + housing_weight + gdp_weight
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: Weight sum is {total_weight:.6f}, not equal to 1.0, will normalize")
        population_weight = population_weight / total_weight
        housing_weight = housing_weight / total_weight
        gdp_weight = gdp_weight / total_weight
        print(f"Weights normalized: Population task={population_weight:.3f}, Housing task={housing_weight:.3f}, GDP task={gdp_weight:.3f}")
    
    start_time = time.time()
    region_gnn.eval()
    buffer_controller.eval()
    candidate_attention.eval()

    def compute_mixed_reward(triple_results):
        return population_weight * triple_results['population_r2'] + housing_weight * triple_results['housing_r2'] + gdp_weight * triple_results['gdp_r2']
    
    with torch.no_grad():
        initial_results_no_weights = compute_triple_task_r2(subgraphs, test_ids, use_weights=False)
        initial_results_with_weights = compute_triple_task_r2(subgraphs, test_ids, use_weights=True, 
                                                          category_weights=category_weights, poi_categories=poi_categories)
        
        initial_mixed_no_weights = compute_mixed_reward(initial_results_no_weights)
        initial_mixed_with_weights = compute_mixed_reward(initial_results_with_weights)
                
        print(f"\nInitial test set evaluation:")
        print("Without weights:")
        print_triple_task_results(initial_results_no_weights, "  ")
        print(f"  Mixed reward = {initial_mixed_no_weights:.4f}")
        print("With weights:")
        print_triple_task_results(initial_results_with_weights, "  ")
        print(f"  Mixed reward = {initial_mixed_with_weights:.4f}")

        population_improvement = (initial_results_with_weights['population_r2'] - initial_results_no_weights['population_r2']) / initial_results_no_weights['population_r2'] * 100
        housing_improvement = (initial_results_with_weights['housing_r2'] - initial_results_no_weights['housing_r2']) / initial_results_no_weights['housing_r2'] * 100
        mixed_improvement = (initial_mixed_with_weights - initial_mixed_no_weights) / initial_mixed_no_weights * 100
        
        print(f"Population prediction weight improvement: {population_improvement:.2f}%")
        print(f"Housing prediction weight improvement: {housing_improvement:.2f}%")
        print(f"Mixed reward improvement: {mixed_improvement:.2f}%")
        
        target_region_id = test_ids[0]
        target_initial_indices = copy.deepcopy(subgraphs[target_region_id][0].orig_indices)
        round_results = []
        
        for rnd in range(test_rounds):
            print(f"\nTest RL round {rnd+1}/{test_rounds}")
            track_category_growth(subgraphs, test_ids, poi_categories, rnd+1, phase="test")
            
            for region_id in test_ids:
                sg, region_shape, region_info = subgraphs[region_id]
                cov_before = ret_cell_coverage(sg, grid_resolution=0.01)
                steiner_before = ret_saturation_reward_dynamic(sg, poi_categories, method='mean')
                current_sg = sg
                
                new_sg, _, updated_region_emb, _, _ = rl_expand_region_with_dynamic_threshold(
                    sg, region_shape, region_info['buffer'], rl_topk, 
                    region_gnn, large_graph, poi_locations, poi_tree, projection_layer,
                    candidate_attention, is_training=False, poi_categories=poi_categories,
                    category_weights=category_weights, threshold_strategy='dynamic_mean')
                
                if updated_region_emb is None:
                    current_sg = sg
                    print(f"Region {region_id}: Expansion failed, keeping original state")
                else:
                    current_sg = new_sg
                    original_poi_count = len(sg.orig_indices)
                    new_poi_count = len(new_sg.orig_indices) if hasattr(new_sg, 'orig_indices') else original_poi_count
                    expansion_count = max(0, new_poi_count - original_poi_count)
                    if region_id == test_ids[0]: 
                        print(f"Region {region_id}: Successfully expanded {expansion_count} POIs ({original_poi_count} to {new_poi_count})")
                
                current_sg = update_connectivity(current_sg, poi_locations)
                subgraphs[region_id] = (current_sg, region_shape, region_info)
                
                cov_after = ret_cell_coverage(current_sg, grid_resolution=0.01)
                steiner_after = ret_saturation_reward_dynamic(current_sg, poi_categories, method='mean')
                
                raw_state = torch.tensor([[cov_after, steiner_after, region_info['buffer']]], dtype=torch.float32).to(device)
                norm_state = torch.cat([
                    (raw_state[:, :2] - norm_mean_12.unsqueeze(0)) / (norm_std_12.unsqueeze(0) + 1e-8),
                    raw_state[:, 2:]
                ], dim=1)
                
                delta, _ = buffer_controller(norm_state)
                delta_value = delta.item() if math.isfinite(delta.item()) else 0.0
                clipping_max = 0.3 * region_info['buffer']
                delta_clipped = max(min(delta_value, clipping_max), 0)
                
                if region_id == target_region_id:
                    print(f"Region{region_id}: delta_value = {delta_value:.4f}, delta_clipped = {delta_clipped:.4f}")
                
                new_buffer = region_info['buffer'] + delta_clipped
                region_info['buffer'] = new_buffer
                
                if region_id == target_region_id:
                    print(f"Region {region_id}: Coverage change = {cov_after - cov_before:.2f}, Steiner change = {steiner_after - steiner_before:.2f}")
            
            global_avg_buffer = np.mean([subgraphs[rid][2]['buffer'] for rid in test_ids])
            print(f"Test round {rnd+1}: Global average Buffer = {global_avg_buffer:.2f}")
            
            current_results_no_weights = compute_triple_task_r2(subgraphs, test_ids, use_weights=False)
            current_results_with_weights = compute_triple_task_r2(subgraphs, test_ids, use_weights=True,
                                                            category_weights=category_weights, poi_categories=poi_categories)
            
            current_mixed_no_weights = compute_mixed_reward(current_results_no_weights)
            current_mixed_with_weights = compute_mixed_reward(current_results_with_weights)

            pop_change_no_weights = current_results_no_weights['population_r2'] - initial_results_no_weights['population_r2']
            pop_change_with_weights = current_results_with_weights['population_r2'] - initial_results_with_weights['population_r2']
            housing_change_no_weights = current_results_no_weights['housing_r2'] - initial_results_no_weights['housing_r2']
            housing_change_with_weights = current_results_with_weights['housing_r2'] - initial_results_with_weights['housing_r2']
            gdp_change_no_weights = current_results_no_weights['gdp_r2'] - initial_results_no_weights['gdp_r2']
            gdp_change_with_weights = current_results_with_weights['gdp_r2'] - initial_results_with_weights['gdp_r2']
            mixed_change_no_weights = current_mixed_no_weights - initial_mixed_no_weights
            mixed_change_with_weights = current_mixed_with_weights - initial_mixed_with_weights
            
            print(f"Test round {rnd+1} triple-task evaluation:")
            print("  Without weights:")
            print_triple_task_results(current_results_no_weights, "    ")
            print(f"    Mixed reward = {current_mixed_no_weights:.4f}")
            print("  With weights:")
            print_triple_task_results(current_results_with_weights, "    ")
            print(f"    Mixed reward = {current_mixed_with_weights:.4f}")
            print(f"  Changes (without weights): Population ΔR²={pop_change_no_weights:.4f}, Housing ΔR²={housing_change_no_weights:.4f}, GDP ΔR²={gdp_change_no_weights:.4f}, Mixed Δ={mixed_change_no_weights:.4f}")
            print(f"  Changes (with weights): Population ΔR²={pop_change_with_weights:.4f}, Housing ΔR²={housing_change_with_weights:.4f}, GDP ΔR²={gdp_change_with_weights:.4f}, Mixed Δ={mixed_change_with_weights:.4f}")
            
            round_results.append({
                'round': rnd + 1,
                'results_no_weights': current_results_no_weights,
                'results_with_weights': current_results_with_weights,
                'mixed_no_weights': current_mixed_no_weights,
                'mixed_with_weights': current_mixed_with_weights,
                'changes_no_weights': {
                    'population': pop_change_no_weights,
                    'housing': housing_change_no_weights,
                    'gdp': gdp_change_no_weights,
                    'mixed': mixed_change_no_weights
                },
                'changes_with_weights': {
                    'population': pop_change_with_weights,
                    'housing': housing_change_with_weights,
                    'gdp': gdp_change_with_weights,
                    'mixed': mixed_change_with_weights
                }
            })
        
        print("\n=== Executing Final Evaluation (5x5-fold Cross Validation) ===")
        final_results_no_weights = compute_triple_task_r2_final_test(subgraphs, test_ids, use_weights=False)
        final_results_with_weights = compute_triple_task_r2_final_test(subgraphs, test_ids, use_weights=True,
                                                                category_weights=category_weights, poi_categories=poi_categories)
        print(f"\n=== Final Triple-Task Test Set Evaluation ===")
        print("Without weights:")
        print_triple_task_results(final_results_no_weights, "  ")
        print("With weights:")
        print_triple_task_results(final_results_with_weights, "  ")
        
        enhanced_triple_task_results = {
            'initial_results_no_weights': initial_results_no_weights,
            'initial_results_with_weights': initial_results_with_weights,
            'initial_mixed_no_weights': initial_mixed_no_weights,
            'initial_mixed_with_weights': initial_mixed_with_weights,
            'final_results_no_weights': final_results_no_weights,
            'final_results_with_weights': final_results_with_weights,
            'round_by_round_results': round_results,
            'category_weights': category_weights,
            'population_weight': population_weight,
            'housing_weight': housing_weight
        }
        
        suburban_dir = get_suburban_dir()
    save_path = os.path.join(suburban_dir, 'tmp', city, 'enhanced_triple_task_test_results.pkl')
    with open(save_path, 'wb') as f:
        pkl.dump(enhanced_triple_task_results, f)
    print(f"Enhanced triple-task test results saved to {save_path}")
    
    test_duration = time.time() - start_time
    print(f"\nTest completed. Test time: {test_duration:.2f} seconds")

    total_unique_pois = set()
    for region_id in test_ids:
        if region_id in subgraphs:
            sg, _, _ = subgraphs[region_id]
            for poi_idx in sg.orig_indices:
                total_unique_pois.add(poi_idx)
    
    print(f"Unique POI count after test phase: {len(total_unique_pois)}")
    
    return enhanced_triple_task_results

def test_openai_api(llm_type='GPT'):
    print(f"Testing {llm_type} API connection...")
    try:
        if llm_type == 'GPT':
            api_key = "your_openai_api_key_here"
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "Simple test: Please reply 'GPT-4.1 API working normally'"}],
                max_tokens=200
            )
            result = response.choices[0].message.content
            print(f"GPT API test successful! Response: {result}")
        elif llm_type == 'DeepSeek':
            api_key = "your_deepseek_api_key_here"
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages = [{"role": "system", "content": "You are DeepSeek-R1, I am trying to call you via API for simple question answering"},
                {"role": "user", "content": "Simple test: Please reply 'DeepSeek-R1 API working normally'"}],
                max_tokens=200,
                stream=False
            )
            result = response.choices[0].message.content
            print(f"DeepSeek API test successful! Response: {result}")
        return True
    except Exception as e:
        print(f"{llm_type} API test failed: {str(e)}")
        return False





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', choices=['Beijing', 'Shanghai','Singapore','NYC'])
    parser.add_argument('--dataset', type=str, default='Gaode', choices=['Meituan', 'Gaode','OSM'])
    parser.add_argument('--top_k', type=int, default=8000, help='number of top-k POIs to consider')
    parser.add_argument('--drop', type=str, choices=['BM25', 'random'], default='BM25')
    parser.add_argument('--version', type=str, default='keywords_kmeans')
    parser.add_argument('--candidate_buffer', type=float, default=500, help='Initial Candidate buffer distance (m)')
    parser.add_argument('--rl_topk', type=int, default=40, help='Number of candidate POIs to add in each expansion')
    parser.add_argument('--rl_rounds', type=int, default=10, help='Number of RL rounds')
    parser.add_argument('--toy', action='store_true', help='Run toy test with regions from a subset')
    parser.add_argument('--w1', type=float, default=100, help='Reward weight for ΔR^2')
    parser.add_argument('--w2', type=float, default=1.0, help='Reward weight for (Δsaturation + Δcoverage)')
    parser.add_argument('--cem_iterations', type=int, default=15, help='Number of CEM iterations')
    parser.add_argument('--cem_samples', type=int, default=30, help='Number of CEM samples per iteration')
    parser.add_argument('--expand_steps', type=int, default=1, help='Fixed number of expansion steps for CEM')
    parser.add_argument('--fixed_buffer', type=float, default=500.0, help='Fixed buffer distance for CEM')
    parser.add_argument('--population_weight', type=float, default=0.33, help='Weight for population task in mixed reward')
    parser.add_argument('--housing_weight', type=float, default=0.33, help='Weight for housing task in mixed reward')
    parser.add_argument('--gdp_weight', type=float, default=0.34, help='Weight for GDP task in mixed reward')
    parser.add_argument('--llm_type', type=str, choices=['GPT','DeepSeek'], default='DeepSeek')
    parser.add_argument('--mode', type=str, choices=['total', 'no_train', 'no_cem'], default='total', 
                       help='Execution mode: total=all stages, no_train=skip CEM and RL training')
    parser.add_argument('--disable_llm', action='store_true', default=False,
                    help='Disable LLM instruction during CEM optimization')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    city = args.city
    city_short = city_abbr(city)
    dataset = args.dataset
    top_k = args.top_k
    drop = args.drop
    version = args.version
    initial_candidate_buffer = args.candidate_buffer
    rl_topk = args.rl_topk
    rl_rounds = args.rl_rounds
    w1 = args.w1
    w2 = args.w2
    llm_type = args.llm_type
    mode = args.mode 
    
    # CEM-related parameters
    cem_iterations = args.cem_iterations
    cem_samples = args.cem_samples
    expand_steps = args.expand_steps
    fixed_buffer = args.fixed_buffer
    llm_instruct = not args.disable_llm

    # Triple Tasks parameters
    population_weight = args.population_weight
    housing_weight = args.housing_weight
    gdp_weight = args.gdp_weight
        
    total_start_time = time.time()

    total_weight = population_weight + housing_weight + gdp_weight
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Weight sum is {total_weight:.6f}, will normalize")
        population_weight = population_weight / total_weight
        housing_weight = housing_weight / total_weight
        gdp_weight = gdp_weight / total_weight
        print(f"Weights normalized: Population task={population_weight:.3f}, Housing task={housing_weight:.3f}, GDP task={gdp_weight:.3f}")
    else:
        print(f"Weight settings: Population task={population_weight:.3f}, Housing task={housing_weight:.3f}, GDP task={gdp_weight:.3f}")
    
    print(f"\n=== Execution Mode: {mode} ===")
    if mode == 'total':
        print("Will execute all phases: CEM optimization + RL training + Test")
    elif mode == 'no_train':
        print("Will skip CEM and RL training phases, only execute: Pre-train Buffer + Random expansion test")
    elif mode == 'no_cem':
        print("Will skip CEM phase, only execute: Pre-train Buffer + RL training + Expansion test")
    
    suburban_dir = get_suburban_dir()
    if city in ['Beijing','Shanghai']:
        poi_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city, f'poi_{drop}_{version}_{top_k}.txt')
    poi_embeddings_path = os.path.join(suburban_dir, 'embs', 'BERT', city, f'poi_embeddings_{drop}_{version}.npy')
    poi_embeddings = np.load(poi_embeddings_path)
    
    poi_txt, poi_locations_loaded, _ = load_poi_txt(poi_path)

    poi_embeddings = torch.tensor(poi_embeddings, dtype=torch.float32).to(device)
    print(f'Loaded {poi_embeddings.shape[0]} POIs, each with {poi_embeddings.shape[1]} dimensions')
    
    original_to_filtered_mapping = {}
    if city in ['Beijing','Shanghai']: 
        with open(poi_path, 'r') as f:
            for line_idx, line in enumerate(f):
                fields = line.strip().split('\t')
                if len(fields) >= 4:  
                    original_index = int(fields[-1])  
                    original_to_filtered_mapping[original_index] = line_idx 
        print(f'Created {len(original_to_filtered_mapping)} original index to filtered index mappings')
    else:
        print('Singapore mode, no index mapping needed')
    
    global poi_locations
    poi_locations = np.array(poi_locations_loaded)
    if poi_locations.shape[1] != 2:
        raise ValueError("POI locations must be 2D for Delaunay triangulation")
    
    poi_categories = load_poi_categories(poi_txt)
    
    
    category_counter = Counter(poi_categories)
    print(f"Total {len(category_counter)} different POI categories")
    print("POI category distribution (Top 10):")
    for cat, count in category_counter.most_common(10):
        print(f"  {cat}: {count} ({count/len(poi_categories)*100:.2f}%)")
    

    if mode == 'total' and llm_instruct:
        api_working = test_openai_api(llm_type)
        if not api_working:
            user_input = input("OpenAI API test failed. Continue executing program? (y/n): ")
            if user_input.lower() != 'y':
                print("Program terminated. Please check API credentials or network connection.")
                return
            print("Continuing execution, but LLM functionality may not be available...")
    else:
        print("LLM instruction banned.")

    tri = Delaunay(poi_locations)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add((simplex[i], simplex[j]))
                edges.add((simplex[j], simplex[i]))
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous().to(device)
    large_graph = Data(x=poi_embeddings, edge_index=edge_index)
    print(f'Created large graph with {large_graph.num_nodes} nodes and {large_graph.num_edges} edges')
    
    points_for_tree = np.column_stack((poi_locations[:, 1], poi_locations[:, 0]))
    poi_tree = cKDTree(points_for_tree)
    
    if city in ['Beijing','Shanghai']:
        suburban_dir = get_suburban_dir()
        region_data_path = os.path.join(suburban_dir, 'data', dataset, 'processed', 'Integral', f'{city_short}_data_{drop}_{version}_{top_k}.pkl')
    else:
        print("Please choose cities between Beijing and Shanghai")
    
    print(f"Loading region data from: {region_data_path}")
    with open(region_data_path, 'rb') as f:
        region_data = pkl.load(f)
    sorted_region_data = dict(sorted(region_data.items(), key=lambda item: item[0]))
    toy_keys = list(sorted_region_data.keys())
    toy_regions = {k: sorted_region_data[k] for k in toy_keys}
    
    
    subgraphs = {}
    total_mapping_stats = {'total_pois': 0, 'failed_mappings': 0, 'total_regions': 0}
    all_failed_indices = set()  
    
    for region_id in tqdm(toy_regions, desc="Creating subgraphs"):
        region_info = toy_regions[region_id]
        if region_info['pois']:
            mapping = original_to_filtered_mapping if city in ['Beijing','Shanghai'] else None
            sg, mapping_stats = create_subgraph_from_region(region_info, large_graph, poi_locations, buffer_value=0, original_to_filtered_mapping=mapping)
            region_info['buffer'] = initial_candidate_buffer
            subgraphs[region_id] = (sg, region_info['region_shape'], region_info)

        
            total_mapping_stats['total_pois'] += mapping_stats['total_pois']
            total_mapping_stats['failed_mappings'] += mapping_stats['failed_mappings']
            total_mapping_stats['total_regions'] += 1
            
            
            if 'failed_indices' in mapping_stats:
                all_failed_indices.update(mapping_stats['failed_indices'])
    
    
    # if city in ['Beijing','Shanghai'] and total_mapping_stats['failed_mappings'] > 0:
    #     print(f"\n=== POI Index Mapping Statistics ===")
    #     print(f"Total regions: {total_mapping_stats['total_regions']}")
    #     print(f"Total POIs: {total_mapping_stats['total_pois']}")
    #     print(f"Failed mapping POIs: {total_mapping_stats['failed_mappings']}")
    #     print(f"Mapping success rate: {((total_mapping_stats['total_pois'] - total_mapping_stats['failed_mappings']) / total_mapping_stats['total_pois'] * 100):.2f}%")
        
    #     print(f"\nFailed index analysis:")
    #     print(f"Unique failed indices: {len(all_failed_indices)}")
        
        
    #     sample_failed = list(all_failed_indices)[:10]
    #     print(f"Failed index samples: {sample_failed}")
        

    #     max_filtered_idx = len(original_to_filtered_mapping) - 1
    #     out_of_range_count = sum(1 for idx in all_failed_indices if idx > max_filtered_idx)
    #     print(f"Indices exceeding filtered file range: {out_of_range_count}")
    #     print(f"POI count in filtered file: {len(original_to_filtered_mapping)}")
        
    
    #     if all_failed_indices:
    #         min_failed = min(all_failed_indices)
    #         max_failed = max(all_failed_indices)
    #         print(f"Failed index range: {min_failed} - {max_failed}")
        
    #     print("========================\n")
    # elif city in ['Beijing','Shanghai']:
    #     print(f"All {total_mapping_stats['total_pois']} POI indices mapped successfully!")
    
    region_ids = list(subgraphs.keys())
    num_regions = len(region_ids)
    random.seed(42)
    random.shuffle(region_ids)

    pretrain_ids = region_ids[:num_regions // 10]
    rl_train_ids = region_ids[num_regions // 10 : (1 * num_regions) // 5]
    test_ids = region_ids
    
    print(f"Pre-training regions: {len(pretrain_ids)}")
    print(f"RL training regions: {len(rl_train_ids)}")
    print(f"Test regions: {len(test_ids)}")
    
    # Initialize models and components
    in_dim = poi_embeddings.shape[1]
    hidden_dim = 128
    out_dim = 128
    
    region_gnn = RegionGAT_PReLU(in_dim, hidden_dim, out_dim, heads=8, dropout=0.5).to(device)
    projection_layer = torch.nn.Linear(in_dim, out_dim).to(device)
    buffer_controller = BufferController(input_dim=3, hidden_dim=32).to(device)
    candidate_attention = CandidateAttention(embed_dim=out_dim, num_heads=8).to(device)
    
    # Pre-train BufferController
    pretrain_buffer_controller(subgraphs, pretrain_ids, buffer_controller, poi_categories, device, num_epochs=10)
    
    # Choose mode of execution
    if mode == 'total':
        # Complete execution path: CEM optimization + RL training + testing
        print("\n=== Phase 1: Optimize POI Category Weights using CEM (Triple-Task) ===")
        optimized_weights = optimize_category_weights_with_cem_triple_task(
            subgraphs, rl_train_ids, region_gnn, projection_layer, 
            large_graph, poi_locations, poi_tree, poi_categories, 
            device, candidate_attention,
            expand_steps=expand_steps, fixed_buffer=fixed_buffer, 
            n_iterations=cem_iterations, n_samples=cem_samples, cem_samples=cem_samples,
            population_weight=population_weight, housing_weight=housing_weight, gdp_weight=gdp_weight, city=city, rl_topk=rl_topk, llm_type=llm_type, llm_instruct=llm_instruct)
            
       
        print("\n=== Evaluate Optimized Weight Triple-Task Performance ===")
        initial_triple_no_weights = compute_triple_task_r2(subgraphs, rl_train_ids, use_weights=False)
        initial_triple_with_weights = compute_triple_task_r2(subgraphs, rl_train_ids, use_weights=True,
                                                       category_weights=optimized_weights, poi_categories=poi_categories)
            
        print(f"Training set triple-task evaluation:")
        print("Without weights:")
        print_triple_task_results(initial_triple_no_weights, "  ")
        print("With weights:")
        print_triple_task_results(initial_triple_with_weights, "  ")
        
        pop_improvement = (initial_triple_with_weights['population_r2'] - initial_triple_no_weights['population_r2']) / initial_triple_no_weights['population_r2'] * 100
        house_improvement = (initial_triple_with_weights['housing_r2'] - initial_triple_no_weights['housing_r2']) / initial_triple_no_weights['housing_r2'] * 100
        
        print(f"Population prediction weight improvement: {pop_improvement:.2f}%")
        print(f"Housing prediction weight improvement: {house_improvement:.2f}%")

        print("\n=== Phase 2: RL Training with Optimized Weights (Triple-Task) ===")
        subgraphs, norm_mean_tensor, norm_std_tensor = train_rl_rounds_with_triple_task_weights(
            subgraphs, rl_train_ids, region_gnn, projection_layer, buffer_controller,
            large_graph, poi_locations, poi_tree, rl_topk, rl_rounds, poi_categories, device,
            candidate_attention, optimized_weights, 
            population_weight=population_weight, housing_weight=housing_weight, gdp_weight=gdp_weight,
            w1=w1, w2=w2, threshold_strategy = 'dynamic_mean')
        
        avg_buffer = np.mean([subgraphs[rid][2]['buffer'] for rid in rl_train_ids])
        print("Training completed. Global average Buffer value =", avg_buffer)
        
        
        print("\n=== Stage 3: Testing Phase (Triple-task evaluation - training mode) ===")
        testing_phase_with_triple_tasks_enhanced(
            subgraphs, test_ids, region_gnn, projection_layer, buffer_controller,
            large_graph, poi_locations, poi_tree, poi_categories, norm_mean_tensor, norm_std_tensor, device,
            optimized_weights, population_weight=population_weight, housing_weight=housing_weight, gdp_weight=gdp_weight,
            test_rounds=rl_rounds, rl_topk=rl_topk, candidate_attention=candidate_attention, city=city, mode=mode)

    elif mode == 'no_train':
        
        print("\n=== Skipping CEM and RL training phases ===")
        
    
        unique_categories = sorted(set(poi_categories))
        default_weights = {cat: 1.0 for cat in unique_categories}
        print(f"Using default weights (all {len(unique_categories)} category weights set to 1.0)")
        
    
        norm_mean_all, norm_std_all = compute_norm_stats(subgraphs, poi_categories, rl_train_ids)
        norm_mean_tensor = torch.tensor(norm_mean_all[:2], dtype=torch.float32).to(device)
        norm_std_tensor = torch.tensor(norm_std_all[:2], dtype=torch.float32).to(device)
        print("Normalization statistics calculated from training data (coverage & Steiner): mean", norm_mean_all[:2], "std", norm_std_all[:2])
        
      
        print("\n=== Initial Triple-task Evaluation (Untrained Model) ===")
        initial_triple_baseline = compute_triple_task_r2(subgraphs, test_ids, use_weights=False)
        initial_triple_default_weights = compute_triple_task_r2(subgraphs, test_ids, use_weights=True,
                                                          category_weights=default_weights, poi_categories=poi_categories)
        
        print(f"Test set triple-task evaluation (initial state):")
        print("Without weights:")
        print_triple_task_results(initial_triple_baseline, "  ")
        print("Using default weights:")
        print_triple_task_results(initial_triple_default_weights, "  ")
        
       
        print("\n=== Testing Phase (Triple-task evaluation - random expansion mode) ===")
        print("Note: Using untrained model, expansion behavior is close to random")
        testing_phase_with_triple_tasks_enhanced(
            subgraphs, test_ids, region_gnn, projection_layer, buffer_controller,
            large_graph, poi_locations, poi_tree, poi_categories, norm_mean_tensor, norm_std_tensor, device,
            default_weights, population_weight=population_weight, housing_weight=housing_weight, gdp_weight=gdp_weight,
            test_rounds=rl_rounds, rl_topk=rl_topk, candidate_attention=candidate_attention, city=city, mode=mode)
    elif mode == 'no_cem':
              
        unique_categories = sorted(set(poi_categories))
        default_weights = {cat: 1.0 for cat in unique_categories}
        print(f"Skipping CEM, using default weights directly (all {len(unique_categories)} category weights set to 1.0)")
       
        print("\n=== Evaluating Optimized Weights for Triple-task Performance ===")
        initial_triple_no_weights = compute_triple_task_r2(subgraphs, rl_train_ids, use_weights=False)
        initial_triple_with_weights = compute_triple_task_r2(subgraphs, rl_train_ids, use_weights=True,
                                                       category_weights=default_weights, poi_categories=poi_categories)
            
        print(f"Training set triple-task evaluation:")
        print("Without weights:")
        print_triple_task_results(initial_triple_no_weights, "  ")
        print("Using weights:")
        print_triple_task_results(initial_triple_with_weights, "  ")
        
        pop_improvement = (initial_triple_with_weights['population_r2'] - initial_triple_no_weights['population_r2']) / initial_triple_no_weights['population_r2'] * 100
        house_improvement = (initial_triple_with_weights['housing_r2'] - initial_triple_no_weights['housing_r2']) / initial_triple_no_weights['housing_r2'] * 100
        
        print(f"Population prediction weight improvement: {pop_improvement:.2f}%")
        print(f"Housing price prediction weight improvement: {house_improvement:.2f}%")

        print("\n=== Stage 2: RL Training with Optimized Weights (Triple-task) ===")
        subgraphs, norm_mean_tensor, norm_std_tensor = train_rl_rounds_with_triple_task_weights(
            subgraphs, rl_train_ids, region_gnn, projection_layer, buffer_controller,
            large_graph, poi_locations, poi_tree, rl_topk, rl_rounds, poi_categories, device,
            candidate_attention, default_weights, 
            population_weight=population_weight, housing_weight=housing_weight, gdp_weight=gdp_weight,
            w1=w1, w2=w2, threshold_strategy = 'dynamic_mean')
        
        avg_buffer = np.mean([subgraphs[rid][2]['buffer'] for rid in rl_train_ids])
        print("Training completed. Global average Buffer value =", avg_buffer)
        
    
        print("\n=== Stage 3: Testing Phase (Triple-task evaluation - training mode) ===")
        testing_phase_with_triple_tasks_enhanced(
            subgraphs, test_ids, region_gnn, projection_layer, buffer_controller,
            large_graph, poi_locations, poi_tree, poi_categories, norm_mean_tensor, norm_std_tensor, device,
            default_weights, population_weight=population_weight, housing_weight=housing_weight, gdp_weight=gdp_weight,
            test_rounds=rl_rounds, rl_topk=rl_topk, candidate_attention=candidate_attention, city=city, mode=mode)

    total_duration = time.time() - total_start_time
    print(f"\n=== Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes) ===")
    print(f"=== Execution mode: {mode} completed ===")


if __name__ == '__main__':
    main()