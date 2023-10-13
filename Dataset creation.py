from torch.utils.data import DataLoader, Dataset

class CustomNLPDataset(Dataset):
    def __init__(self, df, training=True):
        self.data=df
        self.training=training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx].master
        if self.training:
            label = self.data.answer_le.iloc[idx]
            return text,label
        return text
