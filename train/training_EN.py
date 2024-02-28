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
visdom = visdom.Visdom(env="MALIE_EN")
loss_data = {'X': [], 'Y': [], 'legend_U':['mse_loss','tv_loss', 'p_loss']}

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
    args.cuda = True
    args.epochs = 151
    #原始学习率
    #args.lr = 1e-5
    #调整后的学习率
    args.lr = 1e-4
    args.batch_size = 4

    # Setting Important Path /MALIE/TRAIN/#
    train_data_root = '/datas/train/'
    model_save_root_dir = '/checkpoints/EN/'
    model_root1 = '/rcheckpoints/VAN/'
    model_root2 = '/checkpoints/NOISE/'

    # 设置重要的训练变量 #
    VISUALIZATION_STEP = 20
    SAVE_STEP = 20

    #加载数据
    print("加载EN数据")
    train_data = dataset_DALE.DALETrain(train_data_root, args)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    #构建模型
    print("构建EN模型")
    VisualAttentionNet =  VisualAttentionNetwork.VisualAttentionNetwork()
    #加载预训练视觉注意力模型，并赋值给VisualAttentionNet
    state_dict = torch.load(model_root1+'tunning_low_part_epoch_140.pth')
    VisualAttentionNet.load_state_dict(state_dict)
    #加载噪声网络NOISE模型
    NoiseNet=NoiseNetwork.NoiseNetwork()
    state_dict2 = torch.load(model_root2 + 'tunning_low_part_epoch_140.pth')
    NoiseNet.load_state_dict(state_dict2)
    #创建一个增强网络模型对象EnhanceNet
    EnhanceNet = EnhancementNet.EnhancementNet()

    print("设置优化")
    #betas第一个原始为0.5第二个为0.999
    optG = torch.optim.Adam(list(EnhanceNet.parameters()), lr=args.lr, betas=(0.5, 0.999))#Adam优化器optG，更新需要梯度更新的参数
    scheduler = lr_scheduler.ExponentialLR(optG, gamma=0.99)#学习率Lr调度器，调整优化器的学习率
    
    model_EnhanceNet_parameters = filter(lambda p: p.requires_grad, EnhanceNet.parameters())#过滤EnhanceNet模型中更新的参数并统计
    params1 = sum([np.prod(p.size()) for p in model_EnhanceNet_parameters])#记录EnhanceNet中需要梯度更新的参数数量
    print("Parameters | ", params1)
    print("MALIE=> Setting GPU")
    if args.cuda:
        print("Use GPU")
        VisualAttentionNet = VisualAttentionNet.cuda()
        NoiseNet=NoiseNet.cuda()
        EnhanceNet = EnhanceNet.cuda()
    print("Training")

    loss_step = 0

    for epoch in range(1, args.epochs):

        EnhanceNet.train()

        for itr, data in enumerate(loader_train):
            low_light_img, ground_truth_img, gt_Attention_img, file_name = data[0], data[1], data[2], data[3]
            if args.cuda:
                low_light_img = low_light_img.cuda()
                ground_truth_img = ground_truth_img.cuda()
                gt_Attention_img = gt_Attention_img.cuda()
            #清空优化器的梯度
            optG.zero_grad()

            attention_result = VisualAttentionNet(low_light_img)
            noisemap_result = NoiseNet(low_light_img, attention_result.detach())


            enhance_result=EnhanceNet(low_light_img,attention_result.detach(),noisemap_result.detach())
            
            mse_loss = L2_loss(enhance_result, ground_truth_img)
            p_loss = Perceptual_loss(enhance_result, ground_truth_img) * 50
            tv_loss = TvLoss(enhance_result) * 20

            total_loss = p_loss + mse_loss + tv_loss

            total_loss.backward()
            optG.step()

            if epoch > 100 and itr==0:
                scheduler.step()
                print(scheduler.get_last_lr())
                # scheduler是lr_scheduler模块中的ExponentialLR类的一个实例，
                # 它使每个epoch后的学习率以指数速率衰减，即当前学习率=初始学习率*gamma^(当前epoch数)。

            if itr != 0 and itr % VISUALIZATION_STEP == 0:
                #输出训练过程中的损失
                print("Epoch[{}/{}]({}/{}): "
                      "mse_loss : {:.6f}, "
                      "tv_loss : {:.6f}, "
                      "p_loss : {:.6f}"\
                      .format(epoch, args.epochs, itr, len(loader_train), mse_loss, tv_loss, p_loss))
                # Visdom损失图 #

                loss_dict = {
                    'mse_loss': mse_loss.item(),
                    'tv_loss' : tv_loss.item(),
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
                    val_noise=NoiseNet.eval()(val_image, val_attention)
                    val_result = EnhanceNet.eval()(val_image, val_attention,val_noise)

                img_list = OrderedDict(
                    [('input', low_light_img),
                     ('output', enhance_result),
                     ('attention_output', attention_result),
                     ('noisemap', noisemap_result),
                     ('gt_Attention_img', gt_Attention_img),
                     ('batch_sum', attention_result+low_light_img),
                     ('ground_truth', ground_truth_img),
                     ('val_result', val_result)])

                visdom_image(img_dict=img_list, window=10)

                loss_step = loss_step + 1

        print("Testing")
        if epoch % SAVE_STEP == 0:
            train_utils.save_checkpoint(EnhanceNet, epoch, model_save_root_dir)

if __name__ == "__main__":
    opt = args
    main(opt)