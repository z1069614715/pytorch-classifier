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
    parser.add_argument('--val_size', type=float, default=0.1, help='size of val set')
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

    for i in os.listdir(os.getcwd()):
        base_path = os.path.join(os.getcwd(), i)
        base_arr = os.listdir(base_path)
        np.random.shuffle(base_arr)

        val_path = base_path.replace('train', 'val')
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        val_need_copy = base_arr[int(len(base_arr) * (1 - opt.val_size - opt.test_size)):int(len(base_arr) * (1 - opt.test_size))]
        for j in val_need_copy:
            shutil.copy(os.path.join(base_path, j), os.path.join(val_path, j))
        
        test_path = base_path.replace('train', 'test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        test_need_copy = base_arr[int(len(base_arr) * (1 - opt.test_size)):]
        for j in test_need_copy:
            shutil.move(os.path.join(base_path, j), os.path.join(test_path, j))
