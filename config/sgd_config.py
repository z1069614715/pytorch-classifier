import torch
import torchvision.transforms as transforms
from argparse import Namespace
from utils.utils_aug import CutOut, Create_Albumentations_From_Name

class Config:
    lr_scheduler = torch.optim.lr_scheduler.StepLR
    lr_scheduler_params = {
        'gamma': 0.8,
        'step_size': 5
    }
    random_seed = 0
    plot_train_batch_count = 5
    custom_augment = transforms.Compose([
        # transforms.RandomChoice([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        # ]),
        # transforms.RandomRotation(45),
    ])

    def _get_opt(self):
        config_dict = {name:getattr(self, name) for name in dir(self) if name[0] != '_'}
        return Namespace(**config_dict)

if __name__ == '__main__':
    config = Config()
    print(config._get_opt())