import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import lightning as l
import numpy as np
import os


class WDBCDataModule(l.LightningDataModule):
    def __init__(
        self,
        batch_size = 64,
        val_frac = 0.1,
        seed = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed

        dir_name = os.path.dirname(__file__)
        X = pd.read_parquet(os.path.join(dir_name, "data/train/features.parquet")).values.astype(np.float32)
        y = pd.get_dummies(pd.read_parquet(os.path.join(dir_name, "data/train/targets.parquet")), drop_first=True).values.ravel().astype(np.float32)
        
        # save normalization stats for generating new data
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        
        X_norm = (X - self.feature_mean) / self.feature_std

        self.X = X_norm.astype('float32')
        self.y = y.astype('float32')
    
    def setup(self, stage=None):
        # split into train/val
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.X))
        val_size = int(len(self.X) * self.val_frac)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        self.train_X = torch.tensor(self.X[train_indices])
        self.train_y = torch.tensor(self.y[train_indices])
        self.val_X = torch.tensor(self.X[val_indices])
        self.val_y = torch.tensor(self.y[val_indices])
    
    def train_dataloader(self):
        train_y_np = self.train_y.numpy() 
        class_counts = np.bincount(train_y_np.astype(int), minlength=2)
        weights = np.where(train_y_np == 0, 1.0 / class_counts[0], 1.0 / class_counts[1])
        sampler = WeightedRandomSampler(weights, num_samples=len(self.train_y), replacement=True)
        return DataLoader(TensorDataset(self.train_X, self.train_y), batch_size=self.batch_size, sampler=sampler)
    
    def val_dataloader(self):
        return DataLoader(TensorDataset(self.val_X, self.val_y), batch_size=self.batch_size, shuffle=False)


# test datamodule
if __name__ == "__main__":
    dm = WDBCDataModule(batch_size=32)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    for batch in train_loader:
        x, y = batch
        print(x.shape, y.shape)
        break
