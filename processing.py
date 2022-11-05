import warnings
warnings.filterwarnings("ignore")
import os, shutil, argparse
import numpy as np

# set random seed
np.random.seed(0)

'''
    This file help us to split the dataset.
    It's going to be a training set, a validation set, a test set.
    We need to get all the image data into --data_path
    Example:
        dataset/train/dog/*.(jpg, png, bmp, ...)
        dataset/train/cat/*.(jpg, png, bmp, ...)
        dataset/train/person/*.(jpg, png, bmp, ...)
        and so on...
    
    program flow:
    1. generate label.txt.
    2. rename --data_path.
    3. split dataset.
'''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'dataset/train', help='all data path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label txt save path')
    parser.add_argument('--val_size', type=float, default=0.2, help='size of val set')
    parser.add_argument('--test_size', type=float, default=0.2, help='size of test set')
    opt = parser.parse_known_args()[0]
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    with open(opt.label_path, 'w+', encoding='utf-8') as f:
        f.write('\n'.join(os.listdir(opt.data_path)))

    str_len = len(str(len(os.listdir(opt.data_path))))

    for idx, i in enumerate(os.listdir(opt.data_path)):
        os.rename(r'{}/{}'.format(opt.data_path, i), r'{}/{}'.format(opt.data_path, str(idx).zfill(str_len)))

    os.chdir(opt.data_path)

    for i in range(len(os.listdir(os.getcwd()))):
        base_path = os.path.join(os.getcwd(), str(i).zfill(str_len))
        end_path = base_path.replace('train', 'test')
        if not os.path.exists(end_path):
            os.makedirs(end_path)
        len_arr = os.listdir(base_path)
        need_copy = np.random.choice(np.arange(len(len_arr)), int(len(len_arr) * opt.test_size), replace=False)
        for j in need_copy:
            a = os.path.join(base_path, len_arr[j])
            b = os.path.join(end_path, len_arr[j])
            shutil.copy(a, b)
        for j in need_copy:
            os.remove(os.path.join(base_path, len_arr[j]))

    for i in range(len(os.listdir(os.getcwd()))):
        base_path = os.path.join(os.getcwd(), str(i).zfill(str_len))
        end_path = base_path.replace('train', 'val')
        if not os.path.exists(end_path):
            os.makedirs(end_path)
        len_arr = os.listdir(base_path)
        need_copy = np.random.choice(np.arange(len(len_arr)), int(len(len_arr) * opt.val_size), replace=False)
        for j in need_copy:
            a = os.path.join(base_path, len_arr[j])
            b = os.path.join(end_path, len_arr[j])
            shutil.copy(a, b)
        for j in need_copy:
            os.remove(os.path.join(base_path, len_arr[j]))