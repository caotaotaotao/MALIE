from option import args
from data import dataset_MALIE
from torch.utils.data import DataLoader
import VisualAttentionNetwork, EnhancementNet,NoiseNetwork
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
# Setting Loss #
L2_loss = nn.MSELoss().cuda()
Perceptual_loss = ploss.PerceptualLoss().cuda()
TvLoss = tvloss.TVLoss().cuda()

# Setting Visdom #
visdom = visdom.Visdom(env="MALIE_NOise")
loss_data = {'X': [], 'Y': [], 'legend_U':['mse_loss']}

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
    args.batch_size = 6

    # 设置重要路径
    train_data_root='/datas/train/'
    model_save_root_dir = '/checkpoints/NOISE/'
    model_root = '/checkpoints/VAN/'#读取上个视觉注意力网络的权重来引导训练噪声网络

    # 设置重要的训练变量 #
    VISUALIZATION_STEP = 20
    SAVE_STEP = 20

    print("数据加载")
    train_data = dataset_DALE.DALETrainForNoise(train_data_root, args)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    print("视觉注意力模型构建")
    VisualAttentionNet =  VisualAttentionNetwork.VisualAttentionNetwork()
    #加载预训练的视觉注意力模型
    state_dict = torch.load(model_root + 'tunning_low_part_epoch_140.pth')
    VisualAttentionNet.load_state_dict(state_dict)
   
    # 创建一个噪声网络模型对象Noise_Net
    NoiseNet=NoiseNetwork.NoiseNetwork()

    print("优化设置")
    optG = torch.optim.Adam(list(VisualAttentionNet.parameters()), lr=args.lr, betas=(0.6, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optG, gamma=0.99)
    print("Setting GPU")
    if args.cuda:
        print(" Use GPU")
        VisualAttentionNet = VisualAttentionNet.cuda()
        Noisenet=NoiseNet.cuda()
    print("Training")

    loss_step = 0

    for epoch in range(1, args.epochs):

        Noisenet.train()

        for itr, data in enumerate(loader_train):
            low_light_img, ground_truth_img, gt_Attention_img,get_Noise_img, file_name = data[0], data[1], data[2], data[3],data[4]
            if args.cuda:
                low_light_img = low_light_img.cuda()
                ground_truth_img = ground_truth_img.cuda()
                gt_Attention_img = gt_Attention_img.cuda()
                get_Noise_img=get_Noise_img.cuda()
                #print("自生成噪声图shape")
                #print(get_Noise_img.shape)
            #清空优化器的梯度
            optG.zero_grad()
            attention_result = VisualAttentionNet(low_light_img)

            
            noise_result=Noisenet(low_light_img, attention_result.detach())
            mse_loss = L2_loss(noise_result,get_Noise_img)
            total_loss = mse_loss

            total_loss.backward()
            optG.step()

            if epoch > 70 and itr==0:
                scheduler.step()#调整学习率
                print(scheduler.get_last_lr())
                #scheduler是lr_scheduler模块中的ExponentialLR类的一个实例，
                #它使每个epoch后的学习率以指数速率衰减，即当前学习率=初始学习率*gamma^(当前epoch数)。

            if itr != 0 and itr % VISUALIZATION_STEP == 0:
                    # 输出训练过程中的损失
                    print("Epoch[{}/{}]({}/{}): "
                          "mse_loss : {:.6f}, " \
                          .format(epoch, args.epochs, itr, len(loader_train), mse_loss))
                    # Visdom损失图 #

                    loss_dict = {
                        'mse_loss': mse_loss.item(),
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
                        val_noise = Noisenet.eval()(val_image, val_attention)

                    img_list = OrderedDict(
                        [('input', low_light_img),
                         ('output', noise_result),
                         ('attention_output', attention_result),
                         ('gt_Attention_img', gt_Attention_img),
                         ('batch_sum', attention_result + low_light_img),
                         ('ground_truth', ground_truth_img),
                         ('get_Noise_img', get_Noise_img),
                         ('val_Noise_result', val_noise)])

                    visdom_image(img_dict=img_list, window=10)

                    loss_step = loss_step + 1

        print("Testing")
        if epoch % SAVE_STEP == 0:
            train_utils.save_checkpoint(Noisenet, epoch, model_save_root_dir)

if __name__ == "__main__":
    opt = args
    main(opt)