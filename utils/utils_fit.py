import torch, tqdm
import numpy as np
from .utils_aug import mixup_data, mixup_criterion
from .utils import Train_Metrice

def fitting(model, loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler, show_thing, opt):
    model.to(DEVICE)
    model.train()
    metrice = Train_Metrice(CLASS_NUM)
    for x, y in tqdm.tqdm(train_dataset, desc='{} Train Stage'.format(show_thing)):
        x, y = x.to(DEVICE), y.to(DEVICE).long()

        with torch.cuda.amp.autocast(opt.amp):
            if opt.mixup != 'none' and np.random.rand() > 0.5:
                x_mixup, y_a, y_b, lam = mixup_data(x, y, opt)
                pred = model(x_mixup.float())
                l = mixup_criterion(loss, pred, y_a, y_b, lam)
                pred = model(x.float())
            else:
                pred = model(x.float())
                l = loss(pred, y)

        metrice.update_loss(float(l.data))
        metrice.update_y(y, pred)

        scaler.scale(l).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(test_dataset, desc='{} Test Stage'.format(show_thing)):
            x, y = x.to(DEVICE), y.to(DEVICE).long()

            with torch.cuda.amp.autocast(opt.amp):
                if opt.test_tta:
                    bs, ncrops, c, h, w = x.size()
                    pred = model(x.view(-1, c, h, w))
                    pred = pred.view(bs, ncrops, -1).mean(1)
                    l = loss(pred, y)
                else:
                    pred = model(x.float())
                    l = loss(pred, y)
                
            metrice.update_loss(float(l.data), isTest=True)
            metrice.update_y(y, pred, isTest=True)

    return model, metrice.get()


def fitting_distill(teacher_model, student_model, loss, kd_loss, optimizer, train_dataset, test_dataset, CLASS_NUM,
                    DEVICE, scaler, show_thing, opt):
    teacher_model.to(DEVICE)
    teacher_model.eval()
    student_model.to(DEVICE)
    student_model.train()
    metrice = Train_Metrice(CLASS_NUM)
    for x, y in tqdm.tqdm(train_dataset, desc='{} Train Stage'.format(show_thing)):
        x, y = x.to(DEVICE), y.to(DEVICE).long()

        with torch.cuda.amp.autocast(opt.amp):
            if opt.mixup != 'none' and np.random.rand() > 0.5:
                x_mixup, y_a, y_b, lam = mixup_data(x, y, opt)
                s_features, s_features_fc, s_pred = student_model(x_mixup.float(), need_fea=True)
                t_features, t_features_fc, t_pred = teacher_model(x_mixup.float(), need_fea=True)
                l = mixup_criterion(loss, s_pred, y_a, y_b, lam)
                if str(kd_loss) in ['SoftTarget']:
                    kd_l = kd_loss(s_pred, t_pred)
                pred = student_model(x.float())
            else:
                s_features, s_features_fc, s_pred = student_model(x.float(), need_fea=True)
                t_features, t_features_fc, t_pred = teacher_model(x.float(), need_fea=True)
                l = loss(s_pred, y)
                if str(kd_loss) in ['SoftTarget']:
                    kd_l = kd_loss(s_pred, t_pred)
                elif str(kd_loss) in ['MGD']:
                    kd_l = kd_loss(s_features[-1], t_features[-1])
                elif str(kd_loss) in ['SP']:
                    kd_l = kd_loss(s_features[2], t_features[2]) + kd_loss(s_features[3], t_features[3])
                elif str(kd_loss) in ['AT']:
                    kd_l = kd_loss(s_features[2], t_features[2]) + kd_loss(s_features[3], t_features[3])
                    
                if str(kd_loss) in ['SoftTarget', 'SP', 'MGD']:
                    kd_l *= (opt.kd_ratio / (1 - opt.kd_ratio)) if opt.kd_ratio < 1 else opt.kd_ratio
                elif str(kd_loss) in ['AT']:
                    kd_l *= opt.kd_ratio

        metrice.update_loss(float(l.data))
        metrice.update_loss(float(kd_l.data), isKd=True)
        metrice.update_y(y, s_pred)

        scaler.scale(l + kd_l).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    student_model.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(test_dataset, desc='{} Test Stage'.format(show_thing)):
            x, y = x.to(DEVICE), y.to(DEVICE).long()

            with torch.cuda.amp.autocast(opt.amp):
                if opt.test_tta:
                    bs, ncrops, c, h, w = x.size()
                    pred = student_model(x.view(-1, c, h, w))
                    pred = pred.view(bs, ncrops, -1).mean(1)
                    l = loss(pred, y)
                else:
                    pred = student_model(x.float())
                    l = loss(pred, y)

            metrice.update_loss(float(l.data), isTest=True)
            metrice.update_y(y, pred, isTest=True)

    return student_model, metrice.get()