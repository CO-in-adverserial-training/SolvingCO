
from __future__ import print_function
import torch
import os
import os.path
import cv2
from PIL import Image
import torchvision.transforms as transforms
from datasets.index_dataset import IndexDataset

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

def get_loaders(args, index_dataset: bool, device):
    if args.normalize_dataset:
        imagenet_mean = [0.5071, 0.4865, 0.4409] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        imagenet_std =  [0.2673, 0.2564, 0.2762] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        imagenet_mean = [0., 0., 0.]
        imagenet_std = [1., 1., 1.]
    
    mu = torch.tensor(imagenet_mean).view(3,1,1).to(device)
    std = torch.tensor(imagenet_std).view(3,1,1).to(device)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    trainset = Imagenet100(
        root=f'{args.root_path}/Datasets/{args.dataset}', train=True, transform=train_transform)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset
    testset = Imagenet100(
        root=f'{args.root_path}/Datasets/{args.dataset}', train=False, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return trainloader, testloader, upper_limit, lower_limit, mu, std, None, 100, len(trainset), len(testset)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s : s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def loadPILImage(path):
    trans_img = Image.open(path).convert('RGB')
    return trans_img

def loadCVImage(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(trans_img.astype('uint8'), 'RGB')

def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_fpath)):
                    if is_image_file(imgname):
                        path = os.path.join(cls_fpath, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
    else:
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_fpath)):
                    if is_image_file(imgname):
                        path = os.path.join(cls_fpath, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)

        '''
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s : s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            if is_image_file(imgname):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)
        '''
    return images

class Imagenet100(data.Dataset):

    base_folder = 'Imagenet-100'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=True, loader = 'opencv'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.loader = loader

        if download:
            self.download()

        _, class_to_idx = find_classes(os.path.join(self.root, self.base_folder, 'wnids.txt'))
        # self.classes = classes

        if self.train:
            dirname = 'train'
        else:
            dirname = 'val'

        self.data_info = make_dataset(self.root, self.base_folder, dirname, class_to_idx)

        if len(self.data_info) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img_path, target) where target is index of the target class.
        """

        img_path, target = self.data_info[index][0], self.data_info[index][1]

        if self.loader == 'pil':
            img = loadPILImage(img_path)
        else:
            img = loadCVImage(img_path)

        if self.transform is not None:
            result_img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return result_img, target

    def __len__(self):
        return len(self.data_info)
