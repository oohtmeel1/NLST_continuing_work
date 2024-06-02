from PIL import Image
import pydicom
import torch
import numpy as np
from PIL import Image
import torch
import os
import torchvision.transforms as T
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



def getting_all_labels():
    default_folder = './together'
    train_labels = 'df_train1.csv'
    val_labels = 'df_val1.csv'
    test_labels ='df_test1.csv'
    l_ = input(f'Please specify folder where images reside. If none default will be: {default_folder}', )
    if l_ == '':
        pass
    else:
        default_folder = l_
        
    l_ = input(f'Please specify file for training labels if no folder selected, folder will be : {train_labels}', )
    if l_ == '':
        print('folder selected : ', train_labels)

    else:
        train_labels = l_
        
    l_ = input(f'Please specify file for validation labels if no folder selected, folder will be : {val_labels}', )
    if l_ == '':
        print('folder selected : ', val_labels)

    else:
        val_labels = l_
        
    l_ = input(f'Please specify file for testing labels if no folder selected, folder will be : {test_labels}', )
    if l_ == '':
        print('folder selected : ', test_labels)

    else:
        test_labels = l_
    return default_folder,train_labels,val_labels,test_labels

