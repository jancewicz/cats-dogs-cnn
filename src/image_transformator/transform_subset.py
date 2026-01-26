from torch.utils.data import Dataset


class SubsetTransformator(Dataset):
    """
    Custom subset transformator class that enables to apply different types of transformations to dataset without data
    leakage issue.
    """

    def __init__(self, data_subset, transform=None):
        self.data_subset = data_subset
        self.transform = transform

    def __getitem__(self, item):
        X, y = self.data_subset[item]

        if self.transform:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.data_subset)
