import os
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

MAIN_DIRECTORY = '/work/forkert_lab/mitacs_dataset/affine_using_nifty_reg'

# Set seeds for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class DataGenerator(Dataset):
    """Generates data for PyTorch"""

    def __init__(self, list_IDs, dim, filename, column, transform=None):
        """
        Args:
            list_IDs (list): List of subject IDs.
            dim (tuple): Dimensions of the input images (D, H, W).
            filename (str): Path to the CSV file.
            column (str): Name of the label column.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dim = dim
        self.list_IDs = list_IDs
        self.filename = filename
        self.column = column
        self.transform = transform
        self.dataset = pd.read_csv(self.filename)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        ID = self.list_IDs[idx]
        subject_data = self.dataset[self.dataset['Subject'] == ID]
        subject_str = subject_data['Subject'].values[0]
        extension_str = subject_data['Extension'].values[0]
        path = os.path.join(MAIN_DIRECTORY, f"{subject_str}{extension_str}")

        itk_img = sitk.ReadImage(path)
        np_img = sitk.GetArrayFromImage(itk_img)
        slice_y96 = np_img[:, 96, :]
        
        # Нормалізація
        if slice_y96.max() > slice_y96.min():
            slice_y96 = (slice_y96 - slice_y96.min()) / (slice_y96.max() - slice_y96.min())
        
        np_img = np.float32(slice_y96[:, :, np.newaxis]) 
        X = torch.from_numpy(np_img).permute(2, 0, 1)

        y = subject_data[self.column].values[0]
        y = torch.tensor(y, dtype=torch.long)

        if self.transform:
            X = self.transform(X)

        return X, y


    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        ID = self.list_IDs[idx]
        subject_data = self.dataset[self.dataset['Subject'] == ID]
        subject_str = subject_data['Subject'].values[0]
        extension_str = subject_data['Extension'].values[0]
        path = os.path.join(MAIN_DIRECTORY, f"{subject_str}{extension_str}")

        itk_img = sitk.ReadImage(path)
        np_img = sitk.GetArrayFromImage(itk_img)
        slice_y96 = np_img[:, 96, :]
        
        # Перевертаємо зображення
        slice_y96 = np.flip(slice_y96, axis=0)  # flip по першій осі
        
        # Нормалізація
        if slice_y96.max() > slice_y96.min():
            slice_y96 = (slice_y96 - slice_y96.min()) / (slice_y96.max() - slice_y96.min())
        
        np_img = np.float32(slice_y96[:, :, np.newaxis]) 
        X = torch.from_numpy(np_img).permute(2, 0, 1)

        y = subject_data[self.column].values[0]
        y = torch.tensor(y, dtype=torch.long)

        if self.transform:
            X = self.transform(X)

        return X, y