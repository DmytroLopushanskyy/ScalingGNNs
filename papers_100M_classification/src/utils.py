import json
import torch

MODEL_PARAMS_PATH = "./config/model_params.json"
LOADER_PARAMS_PATH = "./config/loader_params.json"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_test_split(num_nodes):
    # Generate custom train/test split at 70/30 ratio
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * 0.7)

    # Train/Test masks
    train = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)

    train[indices[:train_size]] = True
    test[indices[train_size:]] = True

    return train, test


def get_model_params():
    with open(MODEL_PARAMS_PATH, 'r') as file:
        params = json.load(file)
    return params


def get_loader_params():
    with open(LOADER_PARAMS_PATH, 'r') as file:
        params = json.load(file)
    return params


model_params = get_model_params()
loader_params = get_loader_params()
train_mask, test_mask = train_test_split(model_params["num_nodes"])
