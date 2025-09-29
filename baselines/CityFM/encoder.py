import pickle
import torch
import numpy as np
from utils.pre_processing import *
from utils.models import PE_new
from utils.training import Tokenizer, serialize_data
from tqdm import tqdm

def get_suburban_dir(*paths):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *paths)

def encode_nodes(city, nodes, device):
    with open(get_suburban_dir('baselines','CityFM',city, 'Model', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
        
    tok = Tokenizer(model.lm_name)
    model = model.to(device)
    model.eval()
    textual_embs = []
    loc = []
    for node in tqdm(nodes, desc=f"Encoding pos_emb for nodes in {city}", unit="node"):
        loc.append((node['lat'], node['lon']))

    pe = torch.tensor(loc, device=device)
    positional_embs = PE_new(pe, 256).cpu()

    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(nodes), batch_size), desc=f"Encoding text_emb for nodes in {city}", unit="node"):
            text_batch = []
            for node in nodes[i:i+batch_size]:
                text_batch.append("name:" + node['tags']['text'])
            pois = torch.tensor(tok.tokenize_batch(text_batch), device=device).t()
            poi_m = (pois != 0).float()
            encoded = model.lm(pois, attention_mask=poi_m)
            emb = model.projection_p(torch.mean(encoded[0][:, :, :].squeeze(), 0))
            textual_embs.append(emb.cpu())
    
    textual_embs = torch.cat(textual_embs, dim=0)
    assert positional_embs.shape[0] == textual_embs.shape[0]
    return positional_embs, textual_embs

