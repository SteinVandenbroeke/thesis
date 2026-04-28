# ------------------------------------------------------------------------------
# Reference: https://github.com/qjadud1994/DRS/blob/main/utils/LoadData.py
# ------------------------------------------------------------------------------

from .transforms import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

def train_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose(
        [
            transforms.Resize(input_size),  
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean_vals, std_vals),
        ]
    )

    train_list = os.path.join(args.root_dir, "ImageSets/Segmentation/train_cls.txt")
    
    img_train = VOCDataset(train_list, crop_size, root_dir=args.root_dir, num_classes=args.num_classes, transform=tsfm_train, mode='train')

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return train_loader


def test_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose(
        [
            transforms.Resize(crop_size),  
            transforms.ToTensor(),
            transforms.Normalize(mean_vals, std_vals),
        ]
    )

    test_list = os.path.join(args.root_dir, "ImageSets/Segmentation/val.txt")
    
    img_test = VOCDataset(test_list, crop_size, root_dir=args.root_dir, num_classes=args.num_classes, transform=tsfm_test, mode='test')

    test_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader



class VOCDataset(Dataset):
    def __init__(self, datalist_file, input_size, root_dir, num_classes=20, transform=None, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes

        classes_file = os.path.join(self.root_dir, 'classes.txt')
        with open(classes_file, 'r') as f:
            self.categories = [line.strip() for line in f.readlines()]

        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
            
        
    def __len__(self):
        return len(self.image_list)

    
    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        meta = {"img_name": img_name, "ori_size": image.size}
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, self.label_list[idx], meta

    
    def read_labeled_image_list(self, data_dir, data_list):
        img_dir = os.path.join(data_dir, "JPEGImages")
        
        # Read the text file (even if it only contains the image names without numbers)
        with open(data_list, 'r') as f:
            lines = f.readlines()
            
        img_name_list = []
        img_labels = []
        
        for line in lines:
            # Grab just the image name (ignores any broken/missing numbers)
            img_name_no_ext = line.strip().split()[0] 
            image_filename = img_name_no_ext + '.jpg'
            
            # Initialize empty label array (zeros)
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            
            # DYNAMIC LABEL EXTRACTION:
            # Loop through your 200 categories and see which one is in the filename
            for idx, category_name in enumerate(self.categories):
                if category_name in img_name_no_ext:
                    labels[idx] = 1.0
                    break # Stop searching once we found the matching bird
                
            img_name_list.append(os.path.join(img_dir, image_filename))
            img_labels.append(labels)
            
        return img_name_list, img_labels