import pathlib
import pickle
import torch
import torch.utils.data
import numpy as np
from .cifar10 import train_val_split, TransformTwice
import itertools

DATA_PATH = pathlib.Path('RDC_data')

class RakDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, idx):
        return [np.array(self.x[idx]), np.array(self.y[idx])]

    def __len__(self):
        return len(self.x)


def load_datasets():
    with open(DATA_PATH/'datasets.pkl', 'rb') as pf:
        lh, rh = pickle.load(pf)
    lx, ly = lh
    rx, ry = rh
    return RakDataset(lx, ly), RakDataset(rx, ry)


class RDC_labeled(object):

    def __init__(self, ds, indexs=None,
                 transform=None, target_transform=None, MAX_LEN=256):
        self.data = ds.x
        self.targets = ds.y
        self.transform=transform
        self.target_transform=target_transform
        self.MAX_LEN = MAX_LEN
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        # pad self.data
        self.data = list(np.column_stack(itertools.zip_longest(*self.data, fillvalue=0)))
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.targets)

class RDC_unlabeled(RDC_labeled):

    def __init__(self, ds, indexs,
                 transform=None, target_transform=None):
        super(RDC_unlabeled, self).__init__(ds, indexs,
                 transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for i in range(len(self.targets))])
        

class RandomReplace(object):
    """Transform char array by randomly replcing some char.
    """
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, x):
        x = np.array(x)
        indices = np.random.choice(np.arange(x.size), replace=False,
                           size=int(x.size * self.ratio))
        x[indices] = 0
        x = torch.from_numpy(x)
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = np.array(x)
        x = torch.from_numpy(x)
        return x

def load_encoders():
    with open(DATA_PATH/'encodings.pkl', 'rb') as pf:
        lh, rh = pickle.load(pf)
    ch_itos, ch_freq = lh
    cat_itos, cat_freq = rh
    print(len(ch_itos))
    print(len(cat_itos))

def get_rdc(n_labeled, transform_train=None, transform_val=None):

    train_ds, test_ds = load_datasets()  # test will be from val DS that was saved
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(train_ds.y, int(n_labeled/10))

    train_labeled_dataset = RDC_labeled(train_ds, train_labeled_idxs, transform=transform_train)
    train_unlabeled_dataset = RDC_unlabeled(train_ds, train_unlabeled_idxs, transform=TransformTwice(transform_train))
    val_dataset = RDC_labeled(train_ds, val_idxs, transform=transform_val)
    test_dataset = RDC_labeled(test_ds, transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    get_rdc(n_labeled=6000)
    load_encoders() 