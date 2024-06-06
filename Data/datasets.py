from torch import Tensor
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X: Tensor, y: Tensor):
        self.X = X.clone()
        self.y = y.clone()
        self.len = self.X.size(dim=0)
        print(f"Size of the feature input: {self.X.size()}")

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
