import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import sys
from PIL import Image
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (CenterCrop, Compose, HorizontalFlip,  # FDA,
                            Normalize, PadIfNeeded, RandomCrop, Resize,
                            VerticalFlip)
import albumentations as A
from augmentations import *

class EndoDataset(Dataset):
    def __init__(self, args, file_names, transform=None, mode='train'):
        self.args = args
        self.file_names = file_names
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(self.args, img_file_name)

        if self.mode == 'train' or self.mode == 'val' or self.mode == 'eva':
            # normal transforms
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]


            # special transforms to training set
            if self.mode == 'train':
                if self.args.augmix != 'None':
                    return Image.fromarray(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()

        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(args, path):
    # convert image path to mask path
    mask_path = str(path).replace(
        'images', 'annotations/binary')  # endo18 dataset and paste dataset
    mask_path = mask_path.replace('.jpg', '_seg.png')  # blended dataset

    mask = cv2.imread(mask_path, 0)
    # mask[mask==255] = 1
    mask[mask != 0] = 1
    return mask


def train_transform(p=1):
    return Compose([
        Resize(224, 224, always_apply=True, p=1),
        Normalize(p=1)
    ], p=p)


def val_transform(p=1):
    return Compose([
        Resize(224, 224, always_apply=True, p=1),
        Normalize(p=1)
    ], p=p)


def test_transform(p=1):
    return Compose([
        Resize(224, 224, always_apply=True, p=1),
        Normalize(p=1)
    ], p=p)

def train_transform_augmix(p=1):
    return Compose([
        Resize(224, 224, always_apply=True, p=1),
    ], p=p)

_IMG_MEAN, _IMG_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # same as train_transform()

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_IMG_MEAN, _IMG_STD)
    ])

def make_loader(args, file_names, shuffle=False, transform=None, mode='train'):
    dataset=EndoDataset(args, file_names, transform=transform, mode=mode)

    if args.augmix != 'None' and mode == 'train': 
        dataset = AugMixData(dataset, preprocess, aug_type = args.augmix, level=args.augmix_level)

    return DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        num_workers=args.workers,
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available()
    )

class AugMixData(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, aug_type, js_loss=False, n_js=3, level=3, alpha=1, mixture_width=3, mixture_depth=0):
        self.dataset = dataset
        self.preprocess = preprocess
        self.js_loss = js_loss
        self.n_js = n_js
        self.level = level
        self.alpha = alpha
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_type = aug_type

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.js_loss:
            xs = [self.preprocess(x), self.augmix(x)]
            while len(xs) < self.n_js:
                xs.append(self.augmix(x))
            return xs, y
        else:
            return self.augmix(x), y

    def __len__(self):
        return len(self.dataset)

    def augmix(self, img):
        # aug_list = augmentations if True else augmentations_all
        if self.aug_type == 'I':
            aug_list = aug_type_list[0]
        elif self.aug_type == 'II':
            aug_list = aug_type_list[1]
        elif self.aug_type == 'III':
            aug_list = aug_type_list[2]
        elif self.aug_type == 'IV':
            aug_list = aug_type_list[3]
        
        ws = np.float32(np.random.dirichlet([self.alpha] * self.mixture_width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        mixed_image = torch.zeros_like(self.preprocess(img))
        IMAGE_SIZE = mixed_image.size()[0]
        for i in range(self.mixture_width):
            aug_img = img.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for d in range(depth):
                op = np.random.choice(aug_list)
                aug_img = op(aug_img, self.level)
            mixed_image += ws[i] * self.preprocess(aug_img)
        return m * self.preprocess(img) + (1-m) * mixed_image


