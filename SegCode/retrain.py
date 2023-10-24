#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File   : train3d.py
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from visdom import Visdom
import numpy as np
# from vis import Visualizeer
from models.vnet import VNet
# from models.unet3d import UNet3d
# from models.resunet import ResUNet
# from models.resunet import Uceptionpro5
from models.RENet import RE_Net
from dataloader.VesselLoader import Data
# from MRABrainLoader import Data
# from NiiDataLoader import Data
from models.Re_mui_net import Re_mui_net
from models.Muti_net import Muti_net
from utils.train_metrics import metrics, metrics3d
# from utils.evaluation_metrics3D import Dice
from utils.losses import WeightedCrossEntropyLoss, dice_coeff_loss, MSELoss
# from MLutils.dice_loss import dice_coeff_loss
# from utils.misc import get_class_weights
from models.unet import UNet
from utils.visualize import init_visdom_line, update_lines
from models.Uception import Uception

args = {
    # 'root': '/home/imed/segmentation/MICCAI_code/',
    # 'data_path': '/media/imed/新加卷/segdata/Bullitt_Isotrope_light_mult32/split',
    # 'epochs': 4000,
    # 'lr': 0.0001,
    # 'snapshot': 100,
    # 'test_step': 1,
    # 'ckpt_path': '/media/imed/新加卷/segdata/cerebravascular/X-NetPatchEnhancedDice/',
    # 'batch_size': 2,
    ##data change
    'root': 'D:/zhangchaoran/seg_code/',
    'data_path': 'D:/zhangchaoran/miccai_achieve/data/',
    'epochs': 1000,
    'lr': 0.00005,
    'snapshot': 200,
    'test_step': 1,
    'ckpt_path': 'D:/zhangchaoran/NEW_DATA_TRAIN/will/new-datad/our/retrain/',
    'batch_size': 4,
}
viz = Visdom()
# # Visdom---------------------------------------------------------
#
# X, Y1,Y2= 0, 1.0,1.0  # for visdom
# x_acc, y_acc = 0, 0
# x_sen, y_sen = 0, 0
# x_spe, y_spe = 0, 0
# x_iou, y_iou = 0, 0
# x_dsc, y_dsc = 0, 0
# x_pre, y_pre = 0, 0
# x_auc, y_auc = 0, 0
#
# viz = Visdom()
# viz.line([Y],[X], win ='test loss',opts= dict(title ='test loss'))
# env0 = 'loss'
# x_testsen, y_testsen = 0.0, 0.0
# x_testdsc, y_testdsc = 0.0, 0.0
# x_testpre, y_testpre = 0.0, 0.0
# x_testspe, y_testspe = 0.0, 0.0
# env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="epochs", ylabel="loss", env=env0)
# env1, panel1 = init_visdom_line(x_acc, y_acc, title="ACC", xlabel="iters", ylabel="ACC", env=env0)
# env2, panel2 = init_visdom_line(x_sen, y_sen, title="SEN", xlabel="iters", ylabel="SEN", env=env0)
# env3, panel3 = init_visdom_line(x_spe, y_spe, title="SPE", xlabel="iters", ylabel="SPE", env=env0)
# env4, panel4 = init_visdom_line(x_iou, y_iou, title="IOU", xlabel="iters", ylabel="IOU", env=env0)
# env5, panel5 = init_visdom_line(x_dsc, y_dsc, title="DSC", xlabel="iters", ylabel="DSC", env=env0)
# env6, panel6 = init_visdom_line(x_dsc, y_pre, title="PRE", xlabel="iters", ylabel="PRE", env=env0)

# env7, panel7 = init_visdom_line(X,Y, title="Test Loss", xlabel="epochs", ylabel="Test Loss", env=env0)
# env8, panel8 = init_visdom_line(x_testsen, y_testsen, title="Test SEN", xlabel="iters", ylabel="Test SEN", env=env0)
# env9, panel9 = init_visdom_line(x_testdsc, y_testdsc, title="Test DSC", xlabel="iters", ylabel="Test DSC", env=env0)
# env10, panel10 = init_visdom_line(x_testpre, y_testpre, title="Test PRE", xlabel="iters", ylabel="Test PRE", env=env0)
# env11, panel11 = init_visdom_line(x_testspe, y_testspe, title="Test SPE", xlabel="iters", ylabel="Test SPE", env=env0)
# env10, panel10 = init_visdom_line(x_auc, y_auc, title="AUC", xlabel="iters", ylabel="AUC", env=env0)


# env_img = visdom.Visdom(env="images")
# env_heatmap = visdom.Visdom(env="heatmap")
#
#
# # ---------------------------------------------------------------

# Setup CUDA device(s)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, args['ckpt_path'] + 'vnet_Dice' + iter + '.pkl')
    print("{} Saved model to:{}".format("\u2714", args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    loss_window1 = viz.line([1], [0],
                            opts=dict(title='our public retrain final1', legend=['train loss'], xlabel='epochs',
                                      ylabel='loss_value'),env='final_ex')
    value_window1 = viz.line([[0, 0, 0, 0]], [0],
                             opts=dict(title='our public retrain value final1', legend=['sen', 'spe', 'dsc', 'pre'],
                                       xlabel='epochs', ylabel='evaluation_value'),env='final_ex')
    # loss_window = viz.line([[1, 1]], [0],
    #                        opts=dict(title='public train loss', legend=['train loss', 'test loss'], xlabel='epochs',
    #                                  ylabel='loss_value'), env='final_ex')
    # value_window = viz.line([[0, 0, 0, 0]], [0],
    #                         opts=dict(title='public train value', legend=['sen', 'spe', 'dsc', 'pre'], xlabel='epochs',
    #                                   ylabel='evaluation_value'), env='final_ex')
    # 再训练
    # loss_window1 = viz.line([1], [0],
    # opts=dict(title='our public retrain', legend=['train loss'], xlabel='epochs', ylabel='loss_value'),env='final_ex')
    # value_window1 = viz.line([[0, 0, 0, 0]], [0],
    # opts=dict(title='our public retrain value', legend=['sen', 'spe', 'dsc', 'pre'], xlabel='epochs', ylabel='evaluation_value'),env='final_ex')
    # net = AANet(classes=1, channels=1).cuda()
    # net = CSNet3D(classes=2, channels=1).cuda()
    # net = UNet3d(classes=2, channels=1).cuda()
    # net = ResUNet().cuda()
    # net = UNet().cuda()
    # net = RE_Net().cuda()
    # net = Uception().cuda()
    # criterion = models.CASCADE
    # net = Re_mui_net().cuda()
    net = torch.load('D:/zhangchaoran/NEW_DATA_TRAIN/will/new-datad/our/train/vnet_Dicebest_DSC.pkl')
    # net = CSNet3D(classes=2, channels=1).cuda()

    net = nn.DataParallel(net).cuda()
    # 迁移学习
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5, weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)

    # load train dataset

    # print('train_data[0].shape,train_data[1].shape:', train_data[0].shape, train_data[1].shape) ###

    # weights = torch.FloatTensor(get_class_weights(args['data_path'])).cuda()
    # critrion2 = WeightedCrossEntropyLoss(weight=None).cuda()
    # critrion = dice_coeff_loss()
    # critrion2 = WeightedCrossEntropyLoss().cuda()
    # Start training
    print("{}{}{}{}".format(" " * 8, "\u250f", "\u2501" * 61, "\u2513"))
    print("{}{}{}{}".format(" " * 8, "\u2503", " " * 22 + " Start Straining " + " " * 22, "\u2503"))
    print("{}{}{}{}".format(" " * 8, "\u2517", "\u2501" * 61, "\u251b"))

    iters = 1
    best_sen, best_dsc = 0., 0.

    ACC, SEN, SPE, IOU, DSC, PRE, AUC = [], [], [], [], [], [], []

    for epoch in range(args['epochs']):
        net.train()
        train_data = Data(args['data_path'], train=True)

        batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=1, shuffle=True)
        # batch_iter = iter(batchs_data)
        # image, label = next(batch_iter)
        for idx, batch in enumerate(batchs_data):
            image = batch[0].type(torch.FloatTensor).cuda()  # [1, 64, 64, 64],tensor(8.9748),tensor(-2.3123)
            label = batch[1].cuda()  # [1, 64, 64, 64],tensor(255.),tensor(0.)
            # label = label.float()
            optimizer.zero_grad()

            pred = net(image)
            # critrion3 = dice_coeff_loss().cuda()
            # viz.img(name='images', img_=image[0, :, :, :])
            # viz.img(name='labels', img_=label[0, :, :, :])
            # viz.img(name='prediction', img_=pred[0, :, :, :])

            loss = dice_coeff_loss(pred, label)
            # label = label.squeeze(1)  # for CE Loss series
            # loss_ce = critrion(pred, label)
            # loss_wce = critrion2(pred, label)

            # loss = (0.8 * loss_ce + loss_wce + loss_dice) / 3
            # loss_dice = critrion3(pred, label)
            # label = label.squeeze(1)  # for CE Loss series
            # loss1 = critrion(pred,label)
            # loss2 = dice_coeff_loss(pred, label)
            # # loss_wce = critrion2(pred, label)
            #
            # # loss = (0.8 * loss_ce + loss_wce + loss_dice) / 3
            # loss = (loss1 + 0.8 * loss2) / 2.0
            loss.backward()
            optimizer.step()



            acc, sen, spe, iou, dsc, pre = metrics3d(pred, label, pred.shape[0])
            print(
                '{0:d}:{1:d}] \u25001\u2501\u2501 loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}\tspe:{5:.4f}\tiou:{6:.4f}\tdsc:{7:.4f}\tpre:{8:.4f}'.format
                (epoch + 1, iters, loss.item(), acc / pred.shape[0], sen / pred.shape[0], spe / pred.shape[0],
                 iou / pred.shape[0], dsc / pred.shape[0], pre / pred.shape[0]))
            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------
            X, x_sen, x_spe, x_dsc, x_pre = iters, iters, iters, iters, iters
            Y, y_sen, y_spe, y_dsc, y_pre = loss.item(), sen / pred.shape[0], spe / \
                                            pred.shape[0], dsc / pred.shape[0], pre / pred.shape[0]
            # viz.line([loss.item()],[iters],win='train loss',update='append')

            #
            # update_lines(env, panel, X, Y,title='Train Loss')
            # update_lines(env1, panel1, x_acc, y_acc)
            # update_lines(env2, panel2, x_sen, y_sen)
            # update_lines(env3, panel3, x_spe, y_spe)
            # # update_lines(env4, panel4, x_iou, y_iou)
            # update_lines(env5, panel5, x_dsc, y_dsc)
            # update_lines(env6, panel6, x_pre, y_pre)
            # update_lines(env10,panel10, x_auc, y_auc)

            # # --------------------------------------------------------------------------------------------
            ACC.append(acc)
            SEN.append(sen)
            SPE.append(spe)
            IOU.append(iou)
            DSC.append(dsc)
            PRE.append(pre)


        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)

        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, str(epoch + 1))
        if np.mean(SEN)>= best_sen:
            save_ckpt(net,"best_SEN")
            best_sen= np.mean(SEN)
        if np.mean(DSC) >= best_dsc:
            save_ckpt(net, "best_DSC")
            best_sen = np.mean(DSC)

        viz.line([[np.mean(loss.item())]], [epoch], win=loss_window1, update='append')
        viz.line([[np.mean(SEN) / 2, np.mean(SPE) / 2, np.mean(DSC) / 2, np.mean(PRE) / 2]], [epoch], win=value_window1,
                 update='append')
        #
        # # model eval
        # if (epoch + 1) % args['test_step'] == 0:
        #     test_acc, test_sen, test_spe, test_iou, test_dsc, test_pre, loss2 = model_eval(net, iters)
        # if test_sen >= best_sen and (epoch + 1) >= 500:
        #     save_ckpt(net, "best_SEN")
        # best_sen = test_sen
        # if test_dsc > best_dsc:
        #     save_ckpt(net, "best_DSC")
        # best_dsc = test_dsc
        # print(
        #     "average ACC:{0:.4f},average SEN:{1:.4f}, average SPE:{2:.4f},average IOU:{3:.4f}, average DSC:{4:.4f},average PRE:{5:.4f},".format(
        #         test_acc, test_sen, test_spe, test_iou, test_dsc, test_pre))
        # # 两个画图
        # viz.line([[np.mean(loss.item()), np.mean(loss2.item())]], [epoch], win=loss_window, update='append',
        #          env='final_ex')
        # viz.line([[test_sen, test_spe, test_dsc, test_pre]], [epoch], win=value_window, update='append', env='final_ex')
        # 画训练的loss与value



def model_eval(net, iters):
    print("{}{}{}{}".format(" " * 8, "\u250f", "\u2501" * 61, "\u2513"))
    print("{}{}{}{}".format(" " * 8, "\u2503", " " * 23 + " Start Testing " + " " * 23, "\u2503"))
    print("{}{}{}{}".format(" " * 8, "\u2517", "\u2501" * 61, "\u251b"))
    test_data = Data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    ACC, SEN, SPE, IOU, DSC, PRE, AUC = [], [], [], [], [], [], []
    file_num = 0
    for idx, batch in enumerate(batchs_data):
        image = batch[0].float().cuda()
        label = batch[1].cuda()
        pred_val = net(image)
        # label = label.squeeze(1)  # for CE loss
        loss2 = dice_coeff_loss(pred_val, label)
        acc, sen, spe, iou, dsc, pre = metrics3d(pred_val, label, pred_val.shape[0])
        print(
            "---test ACC:{0:.4f} test SEN:{1:.4f} test SPE:{2:.4f} test IOU:{3:.4f} test DSC:{4:.4f} test PRE:{5:.4f}".format
            (acc, sen, spe, iou, dsc, pre))
        ACC.append(acc)
        SEN.append(sen)
        SPE.append(spe)
        IOU.append(iou)
        DSC.append(dsc)
        PRE.append(pre)
        # AUC.append(auc)
        file_num += 1
        # # start visdom images

        X, x_testsen, x_testdsc, x_testpre, x_testspe = iters, iters, iters, iters, iters
        Y, y_testsen, y_testdsc, y_testpre, y_testspe = loss2.item(), sen / pred_val.shape[0], dsc / pred_val.shape[
            0], pre / pred_val.shape[0], spe / pred_val.shape[0]
        # update_lines(env7, panel7, X, Y ,title='Test Loss')
        # update_lines(env8, panel8, x_testsen, y_testsen)
        # update_lines(env9, panel9, x_testdsc, y_testdsc)
        # update_lines(env10,panel10,x_testpre, y_testpre)
        # update_lines(env11,panel11,x_testspe, y_testpre)
        # viz.line([loss2.item()],[iters],win='test loss',update='append')

        return np.mean(ACC), np.mean(SEN), np.mean(SPE), np.mean(IOU), np.mean(DSC), np.mean(PRE), loss2


if __name__ == '__main__':
    train()
