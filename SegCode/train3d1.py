#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File   : train3d.py
import os
import torch
import torch.nn as nn
import time
import glob
from torch import optim
from torch.utils.data import DataLoader
import visdom
import numpy as np
import SimpleITK as sitk
# from vis import Visualizeer
# from models.dolz import VNet
# from models.unet3d import UNet3d
#from models.resunet import ResUNet
#from models.vnet import VNet
#from models.unet import UNet
from models.unetplus import UNet_Nested
#from models.resunet import Uceptionpro5
#from models.resunet import Uception
from dataloader.VesselLoader import Data
# from MRABrainLoader import Data
# from NiiDataLoader import Data


from utils.train_metrics import metrics, metrics3d
# from utils.evaluation_metrics3D import Dice
from utils.losses import WeightedCrossEntropyLoss, dice_coeff_loss,MSELoss
# from MLutils.dice_loss import dice_coeff_loss
# from utils.misc import get_class_weights
from utils.visualize import init_visdom_line, update_lines
# from torchsummary import summary
from tqdm import tqdm
from utils.misc import get_spacing
from torch.utils.data import DataLoader
from dataloader.dataset import MRADataset     

args = {
    # 'root': '/home/imed/segmentation/MICCAI_code/',
    # 'data_path': '/media/imed/新加卷/segdata/MRABrain',
    # 'test_path': '/media/imed/新加卷/segdata/MRABrain/test',
    # 'epochs': 3000,
    # 'lr': 0.0001,
    # 'snapshot': 100,
    # 'test_step': 1,
    # 'ckpt_path': '/media/imed/新加卷/segdata/MRABrain/unet-save',
    # 'batch_size': 2,
    
    'root': '/home/imed/segmentation/MICCAI_code/',
    'data_path': '/media/imed/新加卷/segdata/Bullitt_Isotrope_light_mult32/split',
    'epochs': 3000,
    'lr': 0.0001,
    'snapshot': 100,
    'test_step': 1,
    'ckpt_path': '/media/imed/新加卷/segdata/cerebravascular/u-NetoraldataDice/',
    'batch_size': 2,
}

# # Visdom---------------------------------------------------------
#
X, Y = 0, 1.0  # for visdom
x_acc, y_acc = 0, 0
x_sen, y_sen = 0, 0
x_spe, y_spe = 0, 0
x_iou, y_iou = 0, 0
x_dsc, y_dsc = 0, 0
x_pre, y_pre = 0, 0
x_auc, y_auc = 0, 0


env0 = 'u-net2oraldata'
x_testsen, y_testsen = 0.0, 0.0
x_testdsc, y_testdsc = 0.0, 0.0
env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="iters", ylabel="loss", env=env0)
env1, panel1 = init_visdom_line(x_acc, y_acc, title="ACC", xlabel="iters", ylabel="ACC", env=env0)
env2, panel2 = init_visdom_line(x_sen, y_sen, title="SEN", xlabel="iters", ylabel="SEN", env=env0)
env3, panel3 = init_visdom_line(x_spe, y_spe, title="SPE", xlabel="iters", ylabel="SPE", env=env0)
env4, panel4 = init_visdom_line(x_iou, y_iou, title="IOU", xlabel="iters", ylabel="IOU", env=env0)
env5, panel5 = init_visdom_line(x_dsc, y_dsc, title="DSC", xlabel="iters", ylabel="DSC", env=env0)
env6, panel6 = init_visdom_line(x_dsc, y_pre, title="PRE", xlabel="iters", ylabel="PRE", env=env0)

env7, panel7 = init_visdom_line(x_testsen, y_testsen, title="Test Loss", xlabel="iters", ylabel="Test Loss", env=env0)
env8, panel8 = init_visdom_line(x_testsen, y_testsen, title="Test SEN", xlabel="iters", ylabel="Test SEN", env=env0)
env9, panel9 = init_visdom_line(x_testdsc, y_testdsc, title="Test DSC", xlabel="iters", ylabel="Test DSC", env=env0)
env10, panel10 = init_visdom_line(x_auc, y_auc, title="AUC", xlabel="iters", ylabel="AUC", env=env0)


# env_img = visdom.Visdom(env="images")
# env_heatmap = visdom.Visdom(env="heatmap")
#
#
# # ---------------------------------------------------------------

# Setup CUDA device(s)



def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, args['ckpt_path'] + 'u-netoraldata_Dice' + iter + '.pkl')
    print("{} Saved model to:{}".format("\u2714", args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # torch.cuda.synchronize()
    # time_start = time.time()

    # net = AANet(classes=1, channels=1).cuda()
    # net = CSNet3D(classes=2, channels=1).cuda()
    # net = UNet3d(classes=2, channels=1).cuda()
    #net = ResUNet().cuda()
    #net = VNet().cuda()
    #net = UNet().cuda()
    net = UNet_Nested().cuda()
    #net = torch.load('/media/imed/新加卷/segdata/MRABrain/ckpt-saveV-net_Dice2300.pkl')
    #net = Uceptionpro5().cuda()
    input_shape = (1,64,64,64)
    summary(net, input_shape)
    
    #net = Uception().cuda()
    #net = torch.load('/home/imed/disk5TA/hhy/xyx/xyxX-netPatchEnhanced_Dicebest_DSC.pkl')
    # net = CSNet3D(classes=2, channels=1).cuda()
    
    net = nn.DataParallel(net).cuda()
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
    for epoch in range(args['epochs']):
        #test_auc,test_acc, test_sen, test_spe, test_iou, test_dsc, test_pre = model_eval(net, iters)
        net.train()
        train_data = Data(args['data_path'], train=True)
        print()
        batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=1, shuffle=True)
        for idx, batch in enumerate(batchs_data):
            image = batch[0].type(torch.FloatTensor).cuda()
            label = batch[1].cuda()
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

            auc, acc, sen, spe, iou, dsc, pre = metrics3d(pred, label, pred.shape[0])
            print(
                '{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}\tspe:{5:.4f}\tiou:{6:.4f}\tdsc:{7:.4f}\tpre:{8:.4f}'.format
                (epoch + 1, iters, loss.item(), acc / pred.shape[0], sen / pred.shape[0], spe / pred.shape[0],iou / pred.shape[0], dsc / pred.shape[0], pre / pred.shape[0]))
            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------
            X, x_acc, x_sen, x_spe, x_iou, x_dsc, x_pre,x_auc = iters, iters, iters, iters, iters, iters, iters,iters
            Y, y_acc, y_sen, y_spe, y_iou, y_dsc, y_pre,y_auc= loss.item(), acc / pred.shape[0], sen / pred.shape[0], spe / \
                                                          pred.shape[0], iou / \
                                                          pred.shape[0], dsc / pred.shape[0], pre / pred.shape[0],auc / pred.shape[0]

            update_lines(env, panel, X, Y)
            update_lines(env1, panel1, x_acc, y_acc)
            update_lines(env2, panel2, x_sen, y_sen)
            update_lines(env3, panel3, x_spe, y_spe)
            update_lines(env4, panel4, x_iou, y_iou)
            update_lines(env5, panel5, x_dsc, y_dsc)
            update_lines(env6, panel6, x_pre, y_pre)
            update_lines(env10,panel10, x_auc, y_auc)

            # # --------------------------------------------------------------------------------------------

        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)

        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, str(epoch + 1))

        # model eval
        if (epoch + 1) % args['test_step'] == 0:
            test_auc,test_acc, test_sen, test_spe, test_iou, test_dsc, test_pre = model_eval(net, iters)
        if test_sen >= best_sen and (epoch + 1) >= 500:
            save_ckpt(net, "best_SEN")
        best_sen = test_sen
        if test_dsc > best_dsc:
            save_ckpt(net, "best_DSC")
        best_dsc = test_dsc
        print(
            "Average SEN:{0:.4f}, average SPE:{1:.4f},  average IOU:{2:.4f},average DSC:{3:.4f},average PRE:{4:.4f}".format(
                test_sen, test_spe, test_iou, test_dsc, test_pre))



def model_eval(net, iters):
    print("{}{}{}{}".format(" " * 8, "\u250f", "\u2501" * 61, "\u2513"))
    print("{}{}{}{}".format(" " * 8, "\u2503", " " * 23 + " Start Testing " + " " * 23, "\u2503"))
    print("{}{}{}{}".format(" " * 8, "\u2517", "\u2501" * 61, "\u251b"))
    # test_data = Data(args['data_path'],train = False)
    # batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    AUC ,ACC, SEN, SPE, IOU, DSC, PRE = [], [], [], [], [], [],[]
    # file_num = 0
    # for idx, batch in enumerate(batchs_data):
    #     image = batch[0].float().cuda()
    #     label = batch[1].cuda()
    #     pred_val = net(image)
    #     # label = label.squeeze(1)  # for CE loss
    #     loss2 = dice_coeff_loss(pred_val, label)
    #     auc, acc, sen, spe, iou, dsc, pre = metrics3d(pred_val, label, pred_val.shape[0])
    #     print(
    #         "--- test ACC:{0:.4f} test SEN:{1:.4f} test SPE:{2:.4f} test IOU:{3:.4f} test DSC:{4:.4f} test PRE:{5:.4f} test AUC:{6:.4f}".format
    #             (acc, sen, spe, iou, dsc, pre, auc))
    #     ACC.append(acc)
    #     SEN.append(sen)
    #     SPE.append(spe)
    #     IOU.append(iou)
    #     DSC.append(dsc)
    #     PRE.append(pre)
    #     AUC.append(auc)
    #     file_num += 1
    
    
    # images, labels= load_3d()
    
    # for i in tqdm(range(len(images))):
    #     name_list = images[i].split('/')
    #     index = name_list[-1][:-4]

    #     spacing = get_spacing(images[i])

    #     image = sitk.ReadImage(images[i])
        
    #     image = sitk.GetArrayFromImage(image).astype(np.float32)
        
    #     Image = standardization_intensity_normalization(image, 'float32')
    #     image = torch.from_numpy(np.ascontiguousarray(Image)).unsqueeze(0).unsqueeze(0)
    #     image = image.float().cuda()
    #     label = sitk.ReadImage(labels[i])
    #     label = sitk.GetArrayFromImage(label).astype(np.float32)
    #     label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0).unsqueeze(0)
    #     label = label / 255
    #     label = label.float().cuda()

    #test_dataloader = DataLoader(MRADataset('/media/imed/新加卷/segdata/MRABrain', isTraining=False), ##for .mha data
    test_dataloader = DataLoader(MRADataset('/media/imed/新加卷/segdata/Bullitt_Isotrope_light_mult32/split', isTraining=False),
                             batch_size=1,
                             num_workers=4,
                             shuffle=True)
        
    #for idx, batch in enumerate(test_dataloader):

    for batch in test_dataloader:
        image = batch[0].type(torch.FloatTensor).cuda()
        label = batch[1].cuda()
        
        with torch.no_grad():
            pred_val = net(image)
            pred_val = pred_val.squeeze(0)  # for CE loss
            loss2 = dice_coeff_loss(pred_val, label)
            auc, acc, sen, spe, iou, dsc, pre = metrics3d(pred_val, label, pred_val.shape[0])
            
            ACC.append(acc)
            SEN.append(sen)
            SPE.append(spe)
            IOU.append(iou)
            DSC.append(dsc)
            PRE.append(pre)
            AUC.append(auc)
    # file_num += 1

        # # start visdom images

            X,x_testsen, x_testdsc = iters,iters, iters
            Y,y_testsen, y_testdsc = loss2.item(),sen / pred_val.shape[0], dsc / pred_val.shape[0]
            update_lines(env7, panel7, X, Y)
            update_lines(env8, panel8, x_testsen, y_testsen)
            update_lines(env9, panel9, x_testdsc, y_testdsc)
    print(
                "--- test ACC:{0:.4f} test SEN:{1:.4f} test SPE:{2:.4f} test IOU:{3:.4f} test DSC:{4:.4f} test PRE:{5:.4f} test AUC:{6:.4f}".format
                    (np.mean(ACC), np.mean(SEN), np.mean(SPE), np.mean(IOU), np.mean(DSC), np.mean(PRE), np.mean(AUC)))
    return np.mean(AUC),np.mean(ACC), np.mean(SEN), np.mean(SPE), np.mean(IOU), np.mean(DSC), np.mean(PRE)

        # torch.cuda.synchronize()
        # time_end = time.time()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    train()
