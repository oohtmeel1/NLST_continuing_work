
""" Trying to load some files """
### Download and unzip the files and place them into a directory

from PIL import Image
import pydicom
import torch
import numpy as np
from PIL import Image
import torch
import os
import torchvision.transforms as T
device = torch.device('cuda') 
from torch.utils.data import DataLoader
img_transforms = T.Compose([T.Resize((512,512), interpolation=2),T.RandomHorizontalFlip(p=0.5)
                            , T.RandomRotation(12), T.RandomVerticalFlip(p=0.5),
                            T.ToTensor()])

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image
import pandas as pd
import numpy as np
import argparse
from typing import NamedTuple
from defining_directories import getting_all_labels



default_folder,train_labels,val_labels,test_labels = getting_all_labels()
class Loading_train(Dataset):

    def __init__(self):
        """
        All images are in one folder for this part. And this code matches files to their respective lists.
        """
        #Define dataset
        print('loading training')
        self.selected_dataset_dir = os.path.join(os.path.join(os.getcwd(),default_folder))
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(os.getcwd(),train_labels))
        self.all_filenames = [file for file in self.all_labels['id'] if file in self.all_filenames]
        self.label_meanings = self.all_labels.columns.values.tolist()
    def __len__(self):
        return len(self.all_filenames)
        
    def __getitem__(self, idx):

        #Loads the image files and reads their labels
        selected_filename = self.all_filenames[idx]
        imagepil = Image.open(os.path.join(self.selected_dataset_dir,selected_filename))
        image = img_transforms(imagepil)
        self.all_labels = self.all_labels
        label= self.all_labels['label'][idx]
    
        
        sample = {'data':image, 
                'label':label,
                'img_idx':idx, 'sample_name':selected_filename}
        return(sample) 

class Loading_val(Dataset):

    def __init__(self):
        """
        This should load everything from the directory that matches the test df
        """
        print('Loading validation')
        self.selected_dataset_dir = os.path.join(os.path.join(os.getcwd(),default_folder))
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(os.getcwd(),val_labels))
        self.all_filenames = [file for file in self.all_labels['id'] if file in self.all_filenames]
        self.label_meanings = self.all_labels.columns.values.tolist()
    def __len__(self):
        return len(self.all_filenames)
        
    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        imagepil = Image.open(os.path.join(self.selected_dataset_dir,selected_filename))
        image = img_transforms(imagepil)
        self.all_labels = self.all_labels
        label= self.all_labels['label'][idx]
        sample = {'data':image, 
                'label':label,
                'img_idx':idx, 'sample_name':selected_filename}
        return(sample)

class Loading_test(Dataset):
    def __init__(self):
        """
        This should load everything from the directory that matches the test df
        """
        print('Loading testing')
        self.selected_dataset_dir = os.path.join(os.path.join(os.getcwd(),default_folder))
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(os.getcwd(),test_labels))
        self.all_filenames = [file for file in self.all_labels['id'] if file in self.all_filenames]
        self.label_meanings = self.all_labels.columns.values.tolist()
    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        imagepil = Image.open(os.path.join(self.selected_dataset_dir,selected_filename))
        image = img_transforms(imagepil)
        self.all_labels = self.all_labels
        label= self.all_labels['label'][idx]
        sample = {'data':image, 
                'label':label,
                'img_idx':idx, 'sample_name':selected_filename}
        return(sample)


