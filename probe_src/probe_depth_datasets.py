import torch
from torch.utils.data import Dataset

from torchvision.io import read_image

import os
import pickle


class ProbeOSDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 inter_rep_dir, 
                 label_dir, 
                 transform=None, 
                 inter_rep_transform=None,
                 pre_load=False,
                 target_transform=None,
                 vae_inter_rep=False):
        self.img_dir = img_dir
        self.inter_rep_dir = inter_rep_dir
        self.label_dir = label_dir
        self.vae_inter_rep = vae_inter_rep
        
        self.images = os.listdir(self.img_dir)
        self.images = [image for image in self.images if image.endswith(".png")]
        self.images.sort()
        
        self.inter_reps = os.listdir(self.inter_rep_dir)
        self.inter_reps = [inter_rep for inter_rep in self.inter_reps if inter_rep.endswith(".pkl")]
        self.inter_reps.sort()
        
        self.labels = os.listdir(self.label_dir)
        self.labels = [label for label in self.labels if label.endswith(".png")]
        self.labels.sort()
        
        self.transform = transform
        self.inter_rep_transform = inter_rep_transform
        self.target_transform = target_transform
        
        self.pre_load = pre_load
        if self.pre_load:
            self._pre_load_inter_rep()
        
        self._consistency_check()

    def _consistency_check(self):
        assert len(self.images) == len(self.labels) == len(self.inter_reps), "Inconsistent number of labels and images"
        for i in range(len(self.images)):
            assert self.images[i][:-4] == self.labels[i][:-4], "Inconsistent mapping between label and image"
            
    def _pre_load_inter_rep(self):
        assert self.pre_load
        for i in range(len(self.inter_reps)):
            inter_rep_path = os.path.join(self.inter_rep_dir, self.inter_reps[i])
            with open(inter_rep_path, "rb") as infile:
                inter_rep = pickle.load(infile)[:, 1].to(torch.float)
                self.inter_reps[i] = inter_rep
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        
        label_path = os.path.join(self.label_dir, self.labels[idx])
        label = read_image(label_path)[0]
        
        if self.pre_load:
            inter_rep = self.inter_reps[idx]
        else:
            inter_rep_path = os.path.join(self.inter_rep_dir, self.inter_reps[idx])
            with open(inter_rep_path, "rb") as infile:
                if self.vae_inter_rep:
                    inter_rep = pickle.load(infile).to(torch.float)[0]
                else:
                    inter_rep = pickle.load(infile)[:, 1].to(torch.float)
            
        if "conv" in self.inter_rep_dir:
            inter_rep = inter_rep[0].permute([1, 2, 0])
        
        if self.transform:
            image = self.transform(image)
        if self.inter_rep_transform:
            inter_rep = self.inter_rep_transform(inter_rep)
            
        if self.target_transform:
            label = self.target_transform(label)
        return image, inter_rep, label

    
def threshold_target(target, threshold=0):
    return torch.Tensor(target > threshold).to(torch.long)


class ProbeDEDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 inter_rep_dir, 
                 label_dir, 
                 transform=None, 
                 inter_rep_transform=None,
                 target_transform=None,
                 pre_load=False,
                 vae_inter_rep=False,
                 scale_factor=None):
        self.scale_factor = scale_factor
        self.img_dir = img_dir
        self.inter_rep_dir = inter_rep_dir
        self.label_dir = label_dir
        self.vae_inter_rep = vae_inter_rep
        
        self.images = os.listdir(self.img_dir)
        self.images = [image for image in self.images if image.endswith(".png")]
        self.images.sort()
        
        self.inter_reps = os.listdir(self.inter_rep_dir)
        self.inter_reps = [inter_rep for inter_rep in self.inter_reps if inter_rep.endswith(".pkl")]
        self.inter_reps.sort()
        
        self.labels = os.listdir(self.label_dir)
        self.labels = [label for label in self.labels if label.endswith(".pkl")]
        self.labels.sort()
        
        self.transform = transform
        self.inter_rep_transform = inter_rep_transform
        self.target_transform = target_transform
        
        self._consistency_check()
        self.pre_load = pre_load
        if self.pre_load:
            self._pre_load_inter_rep()

    def _consistency_check(self):
        assert len(self.images) == len(self.labels) == len(self.inter_reps), "Inconsistent number of labels and images"
        for i in range(len(self.images)):
            assert self.images[i][:-4] == self.labels[i][:-4], "Inconsistent mapping between label and image"
            
    def _pre_load_inter_rep(self):
        assert self.pre_load
        for i in range(len(self.inter_reps)):
            inter_rep_path = os.path.join(self.inter_rep_dir, self.inter_reps[i])
            with open(inter_rep_path, "rb") as infile:
                inter_rep = pickle.load(infile)[:, 1].to(torch.float)
                self.inter_reps[i] = inter_rep
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        
        label_path = os.path.join(self.label_dir, self.labels[idx])
        # label = read_image(label_path).to(torch.float)[0]
        with open(label_path, "rb") as infile:
            label = torch.Tensor(pickle.load(infile)).to(torch.float)
        
        if self.pre_load:
            inter_rep = self.inter_reps[idx]
        else:
            inter_rep_path = os.path.join(self.inter_rep_dir, self.inter_reps[idx])
            with open(inter_rep_path, "rb") as infile:
                if self.vae_inter_rep:
                    inter_rep = pickle.load(infile).to(torch.float)[0]
                else:
                    inter_rep = pickle.load(infile)[:, 1].to(torch.float)
            
        if "conv" in self.inter_rep_dir:
            inter_rep = inter_rep[0].permute([1, 2, 0])
        
        if self.transform:
            image = self.transform(image)
        if self.inter_rep_transform:
            inter_rep = self.inter_rep_transform(inter_rep)
            
        if self.target_transform:
            if self.scale_factor:
                label = self.target_transform(label, scale_factor=self.scale_factor)
            else:
                label = self.target_transform(label)
        return image, inter_rep, label

    
def min_max_norm_image(image):
    if image.shape[0] > 3:
        image = image[:3].to(torch.float)
    image -= image.min()
    image /= image.max()
    
    return image

    
def min_max_norm_target(target):
    target -= target.min()
    target /= target.max()
    return target


def standard_norm_target(target):
    target -= target.mean()
    target /= target.std()
    return target


def upsample_image(image, scale_factor=None):
    shrink=False
    if len(image.shape) <= 2:
        image = image.unsqueeze(0)
        shrink=True
    image = image.unsqueeze(0)
    image = torch.nn.functional.interpolate(image, scale_factor=scale_factor, mode="bilinear")[0]
    if shrink:
        image = image[0]
    
    return image


def scale_and_norm(target, scale_factor=None, std=9.281932, mean=15.740597):
    target -= mean
    target /= std
    target = upsample_image(target, scale_factor=scale_factor)
    
    return target


def norm_target(target, std=9.281932, mean=15.740597):
    target -= mean
    target /= std
    
    return target


def norm_intervention_target(target, guide_scale=18):
    target = min_max_norm_target(target)
    target *= guide_scale
    target -= (guide_scale / 2)
    
    return target
