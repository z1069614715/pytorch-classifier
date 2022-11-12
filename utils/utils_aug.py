import torch, tqdm
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from copy import deepcopy
import albumentations as A

def get_mean_and_std(dataset, opt):
    '''Compute the mean and std value of dataset.'''
    if opt.imagenet_meanstd:
        print('using ImageNet Mean and Std. Mean:[0.485, 0.456, 0.406] Std:[0.229, 0.224, 0.225].')
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        print('Calculate the mean and variance of the dataset...')
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for inputs, targets in tqdm.tqdm(dataset):
            inputs = transforms.ToTensor()(inputs)
            for i in range(3):
                mean[i] += inputs[i, :, :].mean()
                std[i] += inputs[i, :, :].std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        print('Calculate complete. Mean:[{:.3f}, {:.3f}, {:.3f}] Std:[{:.3f}, {:.3f}, {:.3f}].'.format(*list(mean.detach().numpy()), *list(std.detach().numpy())))
        return mean, std

def get_processing(dataset, opt):
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(*get_mean_and_std(dataset, opt))])

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, opt, alpha=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    if opt.mixup == 'mixup':
        mixed_x = lam * x + (1 - lam) * x[index, :]
    elif opt.mixup == 'cutmix':
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed_x = deepcopy(x)
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    else:
        raise 'Unsupported MixUp Methods.'
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def select_Augment(opt):
    if opt.Augment == 'RandAugment':
        return transforms.RandAugment()
    elif opt.Augment == 'AutoAugment':
        return transforms.AutoAugment()
    elif opt.Augment == 'TrivialAugmentWide':
        return transforms.TrivialAugmentWide()
    elif opt.Augment == 'AugMix':
        return transforms.AugMix()
    else:
        return None

def get_dataprocessing(dataset, opt, preprocess=None):
    if not preprocess:
        preprocess = get_processing(dataset, opt)
        torch.save(preprocess, r'{}/preprocess.transforms'.format(opt.save_path))
    
    if len(opt.custom_augment.transforms) == 0:
        augment = select_Augment(opt)
    else:
        augment = opt.custom_augment

    if augment is None:
        train_transform = transforms.Compose(
            [transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),
             transforms.RandomCrop((opt.image_size, opt.image_size)),
             preprocess
             ])
    else:
        train_transform = transforms.Compose(
            [transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),
             transforms.RandomCrop((opt.image_size, opt.image_size)),
             augment,
             preprocess
             ])

    if opt.test_tta:
        test_transform = transforms.Compose([
            transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),
            transforms.TenCrop((opt.image_size, opt.image_size)),
            transforms.Lambda(lambda crops: torch.stack([preprocess(crop) for crop in crops]))
        ])
    else:
        test_transform = transforms.Compose([
            transforms.Resize((opt.image_size)),
            transforms.CenterCrop((opt.image_size, opt.image_size)),
            preprocess
        ])
    
    return train_transform, test_transform

def get_dataprocessing_teststage(train_opt, opt, preprocess):
    if opt.test_tta:
        test_transform = transforms.Compose([
            transforms.Resize((int(train_opt.image_size + train_opt.image_size * 0.1))),
            transforms.TenCrop((train_opt.image_size, train_opt.image_size)),
            transforms.Lambda(lambda crops: torch.stack([preprocess(crop) for crop in crops]))
        ])
    else:
        test_transform = transforms.Compose([
            transforms.Resize((train_opt.image_size)),
            transforms.CenterCrop((train_opt.image_size, train_opt.image_size)),
            preprocess
        ])
    return test_transform

class CutOut(object):
    def __init__(self, n_holes=4, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape[:2]
        mask = np.ones_like(img, np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0
        return Image.fromarray(np.array(img * mask, dtype=np.uint8))
    
    def __str__(self):
        return 'CutOut'

class Create_Albumentations_From_Name(object):
    # https://albumentations.ai/docs/api_reference/augmentations/transforms/
    def __init__(self, name, **kwargs):
        self.name = name
        self.transform = eval('A.{}'.format(name))(**kwargs)

    def __call__(self, img):
        img = np.array(img)
        return Image.fromarray(np.array(self.transform(image=img)['image'], dtype=np.uint8))
    
    def __str__(self):
        return self.name