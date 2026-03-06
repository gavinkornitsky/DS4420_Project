import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import lightning as l


class WDBCDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WDBCDataModule(l.LightningDataModule):
    def __init__(self, mode, batch_size=32):
        super().__init__()

        if mode == 'train':
            self.X = pd.read_parquet('data/train/features.parquet')
            self.y = pd.read_parquet('data/train/targets.parquet')
        else:
            self.X = pd.read_parquet('data/test/features.parquet')
            self.y = pd.read_parquet('data/test/targets.parquet')

        self.feature_columns = list(self.X.columns)
        self.target_column = self.y.columns[0]

        self.y = pd.get_dummies(self.y.squeeze(), drop_first=True).values.astype('float32')

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X).astype('float32')

        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = WDBCDataset(self.X, self.y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
