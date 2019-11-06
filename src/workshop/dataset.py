from typing import Union, List, overload
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    dtype_split = pd.CategoricalDtype(categories=['train', 'test'])

    def __init__(self, root: Union[str, Path], train: bool, transforms=None):
        self.split = 'train' if train else 'test'
        self.transforms = transforms
        root = Path(root).expanduser().resolve()

        # Image id to image path
        df_paths = pd.read_csv(root / 'images.txt',
                               sep=' ', names=['id', 'path'], header=None, index_col=0)
        df_paths['path'] = df_paths['path'].apply(root.joinpath('images').joinpath)

        # Class label (zero-based) to class name
        df_classes = pd.read_csv(root / 'classes.txt',
                                 sep=' ', names=['label', 'name'], header=None, index_col=0)
        df_classes.index = df_classes.index - 1
        self.classes = df_classes['name']
        self.number_classes = len(self.classes)

        # Image id to image label (zero-based)
        df_labels = pd.read_csv(root / 'image_class_labels.txt',
                                sep=' ', names=['id', 'label'], header=None, index_col=0)
        df_labels['label'] = df_labels['label'] - 1

        # Image id to train/test split
        df_split = pd.read_csv(root / 'train_test_split.txt',
                               sep=' ', names=['id', 'split'], header=None, index_col=0)
        df_split['split'] = df_split['split'].map({1: 'train', 0: 'test'}).astype(CUBDataset.dtype_split)

        df = pd.merge(df_paths, df_labels, on='id')
        df = pd.merge(df, df_split, on='id')
        df = pd.merge(df, df_classes, on='label')
        self.df = df.query(f'split=="{self.split}"')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img = Image.open(sample.path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, sample.label

    @overload
    def label_to_class_name(self, label: int) -> str: ...

    @overload
    def label_to_class_name(self, label: Union[List, np.ndarray]) -> List[str]: ...

    def label_to_class_name(self, label):
        """Convert zero-based class label(s) to class name(s)"""
        result = self.classes.loc[label]
        if isinstance(result, pd.Series):
            return self.classes.loc[label].tolist()
        else:
            return result

    def __repr__(self):
        return f'{self.__class__.__name__}({self.split}, {len(self)} images)'
