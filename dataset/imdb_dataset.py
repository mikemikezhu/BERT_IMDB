from torch.utils.data import Dataset
import numpy as np


class ImdbDataset(Dataset):

    def __init__(self, x, y):

        self._y = y
        self._x = x

    def classes(self):
        return self._y

    def __len__(self):
        return len(self._y)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self._y[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self._x[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
