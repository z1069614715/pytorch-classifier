import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, argparse, shutil, random, imp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torchvision, time, datetime, copy
from sklearn.utils.class_weight import compute_class_weight
from copy import deepcopy
from utils.utils_fit import fitting, fitting_distill
from utils.utils_model import select_model
from utils import utils_aug
from utils.utils import save_model, plot_train_batch, WarmUpLR, show_config, setting_optimizer, check_batch_size, \
    plot_log, update_opt, load_weights, get_channels, dict_to_PrettyTable
from utils.utils_distill import *
from utils.utils_loss import *

torch.backends.cudnn.deterministic = True
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet18', help='model name')
    parser.add_argument('--pretrained', action="store_true", help='using pretrain weight')
    parser.add_argument('--weight', type=str, default='', help='loading weight path')
    parser.add_argument('--config', type=str, default='config/config.py', help='config path')

    parser.add_argument('--train_path', type=str, default=r'dataset/train', help='train data path')
    parser.add_argument('--val_path', type=str, default=r'dataset/val', help='val data path')
    parser.add_argument('--test_path', type=str, default=r'dataset/test', help='test data path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label path')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--image_channel', type=int, default=3, help='image channel')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (-1 for autobatch)')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--resume', action="store_true", help='resume from save_path traning')

    # optimizer parameters
    parser.add_argument('--loss', type=str, choices=['PolyLoss', 'CrossEntropyLoss', 'FocalLoss'],
                        default='CrossEntropyLoss', help='loss function')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW', 'RMSProp'], default='AdamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--class_balance', action="store_true", help='using class balance in loss')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum in optimizer')
    parser.add_argument('--amp', action="store_true", help='using AMP(Automatic Mixed Precision)')
    parser.add_argument('--warmup', action="store_true", help='using WarmUp LR')
    parser.add_argument('--warmup_ratios', type=float, default=0.05,
                        help='warmup_epochs = int(warmup_ratios * epoch) if warmup=True')
    parser.add_argument('--warmup_minlr', type=float, default=1e-6,
                        help='minimum lr in warmup(also as minimum lr in training)')
    parser.add_argument('--metrice', type=str, choices=['loss', 'acc', 'mean_acc'], default='acc', help='best.pt save relu')
    parser.add_argument('--patience', type=int, default=30, help='EarlyStopping patience (--metrice without improvement)')

    # Data Processing parameters
    parser.add_argument('--imagenet_meanstd', action="store_true", help='using ImageNet Mean and Std')
    parser.add_argument('--mixup', type=str, choices=['mixup', 'cutmix', 'none'], default='none', help='MixUp Methods')
    parser.add_argument('--Augment', type=str,
                        choices=['RandAugment', 'AutoAugment', 'TrivialAugmentWide', 'AugMix', 'none'], default='none',
                        help='Data Augment')
    parser.add_argument('--test_tta', action="store_true", help='using TTA')

    # Knowledge Distillation parameters
    parser.add_argument('--kd', action="store_true", help='Knowledge Distillation')
    parser.add_argument('--kd_ratio', type=float, default=0.7, help='Knowledge Distillation Loss ratio')
    parser.add_argument('--kd_method', type=str, choices=['SoftTarget', 'MGD', 'SP', 'AT'], default='SoftTarget', help='Knowledge Distillation Method')
    parser.add_argument('--teacher_path', type=str, default='', help='teacher model path')

    opt = parser.parse_known_args()[0]
    if opt.resume:
        opt.resume = True
        if not os.path.exists(os.path.join(opt.save_path, 'last.pt')):
            raise Exception('last.pt not found. please check your --save_path folder and --resume parameters')
        ckpt = torch.load(os.path.join(opt.save_path, 'last.pt'))
        opt = ckpt['opt']
        opt.resume = True
        print('found checkpoint from {}, model type:{}\n{}'.format(opt.save_path, ckpt['model'].name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))
    else:
        if os.path.exists(opt.save_path):
            shutil.rmtree(opt.save_path)
        os.makedirs(opt.save_path)
        config = imp.load_source('config', opt.config).Config()
        shutil.copy(__file__, os.path.join(opt.save_path, 'main.py'))
        shutil.copy(opt.config, os.path.join(opt.save_path, 'config.py'))
        opt = update_opt(opt, config._get_opt())

    set_seed(opt.random_seed)
    show_config(deepcopy(opt))

    CLASS_NUM = len(os.listdir(opt.train_path))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, test_transform = utils_aug.get_dataprocessing(torchvision.datasets.ImageFolder(opt.train_path),
                                                                   opt)
    train_dataset = torchvision.datasets.ImageFolder(opt.train_path, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(opt.val_path, transform=test_transform)
    if opt.resume:
        model = ckpt['model'].to(DEVICE)
    else:
        model = select_model(opt.model_name, CLASS_NUM, (opt.image_size, opt.image_size), opt.image_channel,
                             opt.pretrained)
        model = load_weights(model, opt).to(DEVICE)
        plot_train_batch(copy.deepcopy(train_dataset), opt)

    batch_size = opt.batch_size if opt.batch_size != -1 else check_batch_size(model, opt.image_size, amp=opt.amp)

    if opt.class_balance:
        class_weight = np.sqrt(
            compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets))
    else:
        class_weight = np.ones_like(np.unique(train_dataset.targets))
    print('class weight: {}'.format(class_weight))
    # try:
    #     with open(opt.label_path) as f:
    #         label = list(map(lambda x: x.strip(), f.readlines()))
    # except Exception as e:
    #     with open(opt.label_path, encoding='gbk') as f:
    #         label = list(map(lambda x: x.strip(), f.readlines()))
    # print(dict_to_PrettyTable({label[i]:class_weight[i] for i in range(len(label))}, 'Class Weight'))

    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=opt.workers)
    test_dataset = torch.utils.data.DataLoader(test_dataset, max(batch_size // (10 if opt.test_tta else 1), 1),
                                               shuffle=False, num_workers=(0 if opt.test_tta else opt.workers))
    scaler = torch.cuda.amp.GradScaler(enabled=(opt.amp if torch.cuda.is_available() else False))
    if opt.resume:
        optimizer = ckpt['optimizer']
        lr_scheduler = ckpt['lr_scheduler']
        loss = ckpt['loss']
        scaler.load_state_dict(ckpt['scaler'])
    else:
        optimizer = setting_optimizer(opt, model)
        lr_scheduler = WarmUpLR(optimizer, opt)
        loss = eval(opt.loss)(label_smoothing=opt.label_smoothing,
                              weight=torch.from_numpy(class_weight).to(DEVICE).float())
    return opt, model, train_dataset, test_dataset, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, (
        ckpt['epoch'] if opt.resume else 0), (ckpt['best_metrice'] if opt.resume else None)


if __name__ == '__main__':
    opt, model, train_dataset, test_dataset, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, begin_epoch, best_metrice = parse_opt()

    if not opt.resume:
        save_epoch = 0
        with open(os.path.join(opt.save_path, 'train.log'), 'w+') as f:
            if opt.kd:
                f.write('epoch,lr,loss,kd_loss,acc,mean_acc,test_loss,test_acc,test_mean_acc')
            else:
                f.write('epoch,lr,loss,acc,mean_acc,test_loss,test_acc,test_mean_acc')
    else:
        save_epoch = torch.load(os.path.join(opt.save_path, 'last.pt'))['best_epoch']

    if opt.kd:
        if not os.path.exists(os.path.join(opt.teacher_path, 'best.pt')):
            raise Exception('teacher best.pt not found. please check your --teacher_path folder')
        teacher_ckpt = torch.load(os.path.join(opt.teacher_path, 'best.pt'))
        teacher_model = teacher_ckpt['model']
        print('found teacher checkpoint from {}, model type:{}\n{}'.format(opt.teacher_path, teacher_model.name, dict_to_PrettyTable(teacher_ckpt['best_metrice'], 'Best Metrice')))
        
        if opt.kd_method == 'SoftTarget':
            kd_loss = SoftTarget().to(DEVICE)
        elif opt.kd_method == 'MGD':
            kd_loss = MGD(get_channels(model, opt), get_channels(teacher_model, opt)).to(DEVICE)
            optimizer.add_param_group({'params': kd_loss.parameters(), 'weight_decay': opt.weight_decay})
        elif opt.kd_method == 'SP':
            kd_loss = SP().to(DEVICE)
        elif opt.kd_method == 'AT':
            kd_loss = AT().to(DEVICE)

    print('{} begin train on {}!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), DEVICE))
    for epoch in range(begin_epoch, opt.epoch):
        if epoch > (save_epoch + opt.patience):
            print('No Improve from {} to {}, EarlyStopping.'.format(save_epoch + 1, epoch))
            break

        begin = time.time()
        if opt.kd:
            model, metrice = fitting_distill(teacher_model, model, loss, kd_loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler, '{}/{}'.format(epoch + 1,opt.epoch), opt)
        else:
            model, metrice = fitting(model, loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler,'{}/{}'.format(epoch + 1, opt.epoch), opt)
            

        with open(os.path.join(opt.save_path, 'train.log'), 'a+') as f:
            f.write(
                '\n{},{:.10f},{}'.format(epoch + 1, optimizer.param_groups[2]['lr'], metrice[1]))

        lr_scheduler.step()

        if best_metrice is None:
            best_metrice = metrice[0]
        else:
            if eval('{} {} {}'.format(metrice[0]['test_{}'.format(opt.metrice)], '<' if opt.metrice == 'loss' else '>', best_metrice['test_{}'.format(opt.metrice)])):
                best_metrice = metrice[0]
                save_model(
                    os.path.join(opt.save_path, 'best.pt'),
                    **{
                    'model': model.to('cpu'),
                    'opt': opt,
                    'best_metrice': best_metrice,
                    }
                )
                save_epoch = epoch
        
        save_model(
            os.path.join(opt.save_path, 'last.pt'),
            **{
               'model': model.to('cpu'),
               'opt': opt,
               'epoch': epoch + 1,
               'optimizer' : optimizer,
               'lr_scheduler': lr_scheduler,
               'best_metrice': best_metrice,
               'loss': loss,
               'scaler': scaler.state_dict(),
               'best_epoch': save_epoch,
            }
        )

        print(dict_to_PrettyTable(metrice[0], '{} epoch:{}/{}, best_epoch:{}, time:{:.2f}s, lr:{:.8f}'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    epoch + 1, opt.epoch, save_epoch + 1, time.time() - begin, optimizer.param_groups[2]['lr'],
                )))
    
    plot_log(opt)