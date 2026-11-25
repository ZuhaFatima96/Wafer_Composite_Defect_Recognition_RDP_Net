import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

SPLIT_SEED = 3529  # fixed, so every run and evaluate.py see exactly the same train/val/test maps


def make_dataloaders(npz_path, batch_size=128):
    data = np.load(npz_path)
    maps = torch.from_numpy(data["arr_0"].astype(np.float32))
    labels = torch.from_numpy(data["arr_1"].astype(np.float32))
    dataset = TensorDataset(maps, labels)

    # each map gets the id of its label combination (38 in total), so every split keeps the same class balance
    _, class_id = np.unique(data["arr_1"], axis=0, return_inverse=True)

    # 20% test first, then 10% of what is left for validation (about 72 / 8 / 20 overall)
    all_idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, stratify=class_id, random_state=SPLIT_SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=class_id[train_idx],
                                          random_state=SPLIT_SEED)
    print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)} maps")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)
    return train_loader, val_loader, test_loader
