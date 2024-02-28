from option import args
from data import dataset_MALIE
from torch.utils.data import DataLoader
import VisualAttentionNetwork
from train import train_utils
from collections import OrderedDict
import numpy as np
from loss import ploss, tvloss
import visdom
import PIL.Image as Image
from torchvision import transforms
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
#残差块将SE机制改为GAM机制---调试注意力生成模块
# 设置损失#
L1_loss = nn.L1Loss().cuda()#L1损失
Perceptual_loss = ploss.PerceptualLoss().cuda()#感知损失
TvLoss = tvloss.TVLoss().cuda()#总变分损失

# 设置实时变化的visdom参数 #
visdom = visdom.Visdom(env="MALIE_VAN")
loss_data = {'X': [], 'Y': [], 'legend_U':['mse_loss', 'p_loss']}

def visdom_loss(visdom, loss_step, loss_dict):
    loss_data['X'].append(loss_step)
    loss_data['Y'].append([loss_dict[k] for k in loss_data['legend_U']])
    visdom.line(
        X=np.stack([np.array(loss_data['X'])] * len(loss_data['legend_U']), 1),
        Y=np.array(loss_data['Y']),
        win=1,
        opts=dict(xlabel='Step',
                  ylabel='Loss',
                  title='Training loss',
                  legend=loss_data['legend_U']),
        update='append'
    )

def visdom_image(img_dict, window):
    for idx, key in enumerate(img_dict):
        win = window + idx
        tensor_img = train_utils.tensor2im(img_dict[key].data)
        visdom.image(tensor_img.transpose([2, 0, 1]), opts=dict(title=key), win=win)

def test(args, loader_test, model_AttentionNet, epoch, root_dir) :
    model_AttentionNet.eval()
    for itr, data in enumerate(loader_test):
        testImg, fileName = data[0], data[1]
        if args.cuda:
            testImg = testImg.cuda()

        with torch.no_grad():
            test_result = model_AttentionNet(testImg)
            test_result_img = train_utils.tensor2im(test_result)
            result_save_dir = root_dir + fileName[0].split('.')[0]+('_epoch_{}_itr_{}.png'.format(epoch, itr))
            train_utils.save_images(test_result_img, result_save_dir)

def main(args) :
    # 设置重要参数 #
    args.cuda = True
    args.epochs = 151
    args.lr = 1e-4
    args.batch_size = 8

    # 设置重要路径
    train_data_root='/x/datas/train/'
    model_save_root_dir = '/rcheckpoints/VAN/'
    model_root = '/checkpoint/VAN/'

    # 设置重要的训练变量 #
    VISUALIZATION_STEP = 20
    SAVE_STEP = 20

    print("视觉数据加载")

    train_data = dataset_MALIE.MALIETrain(train_data_root, args)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    print("视觉模型构建-加载model下面的模块")
    VisualAttentionNet =  VisualAttentionNetwork.VisualAttentionNetwork()
    
    print("优化设置")
    optG = torch.optim.Adam(list(VisualAttentionNet.parameters()), lr=args.lr, betas=(0.5, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optG, gamma=0.99)

    print("设置GPU参数")
    if args.cuda:
        print("Use GPU")
        VisualAttentionNet = VisualAttentionNet.cuda()
    print("Training")

    loss_step = 0

    for epoch in range(1, args.epochs):

        VisualAttentionNet.train()

        for itr, data in enumerate(loader_train):
            low_light_img, ground_truth_img, gt_Attention_img, file_name = data[0], data[1], data[2], data[3]
            if args.cuda:
                low_light_img = low_light_img.cuda()
                ground_truth_img = ground_truth_img.cuda()
                gt_Attention_img = gt_Attention_img.cuda()
            #清空优化器的梯度
            optG.zero_grad()
            #获得注意力图和损失值
            attention_result = VisualAttentionNet(low_light_img)
            mse_loss = L1_loss(attention_result, gt_Attention_img)
            p_loss = Perceptual_loss(attention_result, gt_Attention_img) * 10
            #总损失值
            total_loss = p_loss + mse_loss
            total_loss.backward()
            #
            optG.step()

            if epoch > 88 and itr==0:
                scheduler.step()#调整学习率
                print(scheduler.get_last_lr())
                #scheduler是lr_scheduler模块中的ExponentialLR类的一个实例，
                #它使每个epoch后的学习率以指数速率衰减，即当前学习率=初始学习率*gamma^(当前epoch数)。

            if itr != 0 and itr % VISUALIZATION_STEP == 0:
                #输出训练过程中的损失
                print("Epoch[{}/{}]({}/{}): "
                      "mse_loss : {:.6f}, "
                      "p_loss : {:.6f}"\
                      .format(epoch, args.epochs, itr, len(loader_train), mse_loss, p_loss))

                # Visdom损失图 #
                loss_dict = {
                    'mse_loss': mse_loss.item(),
                    'p_loss': p_loss.item(),
                }

                visdom_loss(visdom, loss_step, loss_dict)

                # Visdom 可视化 # -> tensor to numpy => list ('title_name', img_tensor)
                with torch.no_grad():
                    val_image = Image.open('15.JPG')

                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])

                    val_image = transform((val_image)).unsqueeze(0)

                    val_image = val_image.cuda()

                    val_attention = VisualAttentionNet.eval()(val_image)

                img_list = OrderedDict(
                    [('input', low_light_img),
                     ('attention_output', attention_result),
                     ('gt_Attention_img', gt_Attention_img),
                     ('batch_sum', attention_result+low_light_img),
                     ('ground_truth', ground_truth_img),
                     ('val_attention', val_attention),
                     ('val_sum', val_image+val_attention)
                     ])

                visdom_image(img_dict=img_list, window=10)
                loss_step = loss_step + 1

        print("DTesting")
        if epoch % SAVE_STEP == 0:
            train_utils.save_checkpoint(VisualAttentionNet, epoch, model_save_root_dir)

if __name__ == "__main__":
    opt = args
    main(opt)