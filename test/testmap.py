import VisualAttentionNetwork, EnhancementNet
from data import dataset_MALIE
from train import train_utils
import torch
from torch.utils.data import DataLoader
def main():
    benchmark = ['datasets__DICM', 'datasets__LIME', 'datasets__MEF', 'datasets__NPE']
    #test_data_root = '../datas/test/'+benchmark[1]
    #print(test_data_root)
    test_data_root = '/datas/test/datasets__NPE/'
    model_root1 = '/checkpoints/Result/'  
    #加载VAN模型
    VisualAttentionNet =  VisualAttentionNetwork.VisualAttentionNetwork()
    state_dict1 = torch.load(model_root1 + 'tunning_low_part_epoch_140.pth')
    VisualAttentionNet.load_state_dict(state_dict1)
    #加载数据
    test_data = dataset_DALE.DALETest(test_data_root)
    loader_test = DataLoader(test_data, batch_size=1, shuffle=False)
    #应用Cuda
    VisualAttentionNet.cuda()
    #调用test
    test(loader_test, VisualAttentionNet, test_result_root_dir)

def test(loader_test, VisualAttentionNet, root_dir):
    VisualAttentionNet.eval()
    
    for itr, data in enumerate(loader_test):
        testImg, fileName = data[0], data[1]
        testImg = testImg.cuda()

        with torch.no_grad():
            test_attention_result= VisualAttentionNet(testImg)

            test_recon_result_img = train_utils.tensor2im(test_attention_result)
            norm_input_img = train_utils.tensor2im(testImg+test_attention_result)

            recon_save_dir = root_dir + 'visual_attention_map_'+fileName[0].split('.')[0]+('.png')
            recon_save_dir2 = root_dir + 'sum_'+fileName[0].split('.')[0]+('.png')

            train_utils.save_images(test_recon_result_img, recon_save_dir)
            train_utils.save_images(norm_input_img, recon_save_dir2)

if __name__ == "__main__":
    main()