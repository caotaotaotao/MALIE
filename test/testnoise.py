import VisualAttentionNetwork, EnhancementNet,NoiseNetwork
from data import dataset_MALIE
from train import train_utils
import torch
from torch.utils.data import DataLoader
def main():
    benchmark = ['datasets__DICM', 'datasets__LIME', 'datasets__MEF', 'datasets__NPE']
    #test_data_root = '../datas/test/'+benchmark[1]
    #print(test_data_root)
    test_data_root = '/datas/test/datasets__VV/'
    model_root1 = '/checkpoints/VAN/'
    model_root2 = '/checkpoints/NOISE/'
    test_result_root_dir = '/checkpoints/Result/'  
    #加载VAN模型
    VisualAttentionNet =  VisualAttentionNetwork.VisualAttentionNetwork()
    state_dict1 = torch.load(model_root1 + 'tunning_low_part_epoch_140.pth')
    VisualAttentionNet.load_state_dict(state_dict1)
    #加载噪声网络NOISE模型
    NoiseNet=NoiseNetwork.NoiseNetwork()
    state_dict2 = torch.load(model_root2 + 'tunning_low_part_epoch_140.pth')
    NoiseNet.load_state_dict(state_dict2)
    #加载数据
    test_data = dataset_MALIE.MALIETest(test_data_root)
    loader_test = DataLoader(test_data, batch_size=1, shuffle=False)
    #应用Cuda
    VisualAttentionNet.cuda()
    NoiseNet.cuda()
    #调用test
    test(loader_test, VisualAttentionNet, NoiseNet, test_result_root_dir)

def test(loader_test, VisualAttentionNet, NoiseNet, root_dir):
    VisualAttentionNet.eval()
    NoiseNet.eval()

    for itr, data in enumerate(loader_test):
        testImg, img_name = data[0], data[1]
        testImg = testImg.cuda()

        with torch.no_grad():
            visual_attention_map = VisualAttentionNet(testImg)
            noise_result = EnhanceNet(testImg, visual_attention_map)
            noise_result_img = train_utils.tensor2im(enhance_result)
            result_save_dir = root_dir + 'noisemaps'+ img_name[0].split('.')[0]+('.png')
            train_utils.save_images(noise_result_img, result_save_dir)

if __name__ == "__main__":
    main()