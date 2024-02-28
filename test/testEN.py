import VisualAttentionNetwork, EnhancementNet,NoiseNetwork 
from data import dataset_MALIE
from train import train_utils
import torch
from torch.utils.data import DataLoader

def main():
    benchmark = ['datasets__DICM', 'datasets__LIME', 'datasets__MEF', 'datasets__NPE']
    #test_data_root = '../datas/test/'+benchmark[1]
    #print(test_data_root)
    test_data_root = '/datas/test/datasets__NPE/'
    model_root1 = '/checkpoints/VAN/'
    model_root2 = '/checkpoints/NOISE/'
    model_root3 = '/checkpoints/EN/'
    test_result_root_dir = '/checkpoints/Result/'
    #加载VAN模型
    VisualAttentionNet =  VisualAttentionNetwork.VisualAttentionNetwork()
    state_dict1 = torch.load(model_root1 + 'tunning_low_part_epoch_140.pth')
    VisualAttentionNet.load_state_dict(state_dict1)
    #加载噪声Noise模型
    NoiseNet=NoiseNetwork.NoiseNetwork()
    state_dict2 = torch.load(model_root2 + 'tunning_low_part_epoch_140.pth')
    NoiseNet.load_state_dict(state_dict2)
    #加载增强网络EN模型
    EnhanceNet = EnhancementNet.EnhancementNet()
    state_dict3 = torch.load(model_root3 + 'tunning_low_part_epoch_140.pth')
    EnhanceNet.load_state_dict(state_dict3)
    #加载数据集和进行数据预处理
    test_data = dataset_MALIE.MALIETest(test_data_root)
    loader_test = DataLoader(test_data, batch_size=1, shuffle=False)

    VisualAttentionNet.cuda()
    NoiseNet.cuda()
    EnhanceNet.cuda()

    test(loader_test, VisualAttentionNet, NoiseNet,EnhanceNet,test_result_root_dir)

def test(loader_test, VisualAttentionNet, NoiseNet,EnhanceNet,root_dir):
    VisualAttentionNet.eval()
    NoiseNet.eval()
    EnhanceNet.eval()

    for itr, data in enumerate(loader_test):
        testImg, img_name = data[0], data[1]
        testImg = testImg.cuda()
        #testImg=testImg.expand(4, -1, -1, -1)
        print("testImg的维度")
        print(testImg.shape)

        with torch.no_grad():
            visual_attention_map = VisualAttentionNet(testImg)
            noise_map=NoiseNet(testImg, visual_attention_map)
            enhance_result = EnhanceNet(testImg, visual_attention_map,noise_map)
            enhance_result_img = train_utils.tensor2im(enhance_result)#将张量转换为图像
            result_save_dir = root_dir + 'enhance'+ img_name[0].split('.')[0]+('.png')
            train_utils.save_images(enhance_result_img, result_save_dir)

if __name__ == "__main__":
    main()