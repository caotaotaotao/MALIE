import os
from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms
from data import dataset_utils as utils
import numpy as np
import torch
# Class for train_lr
class MALIETrain(Dataset):
    def __init__(self, root_dir, args):
        """
        Arguments:
            1) root directory -> \MALIE\TRAIN\
            2) arguments -> args
        """
        self.low_light_dir = root_dir + 'low'
        self.ground_truth_dir = root_dir + 'high'
        self.low_light_img_list = os.listdir(root_dir + 'low')
        self.ground_truth_img_list = os.listdir(root_dir + 'high')
        
        print(self.low_light_img_list.__len__())
        print(self.ground_truth_img_list.__len__())


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform2 = transforms.Compose([
            transforms.ToTensor()
        ])

        # patch_size : default == 128
        self.args = args
        self.patch_size = 240


    def __len__(self):
        return len(self.low_light_img_list)

    def __getitem__(self, idx):
        """
        获取一对随机的图像裁剪。
        它返回一个形状为[3，height，width]的浮点张量元组。
        它们表示像素值在[0，1]范围内的RGB图像
        return：微光图像，真实标准图像张量
        """
        low_light_name, ground_truth_name = self.low_light_img_list[idx], self.ground_truth_img_list[idx]

        low_light_image = Image.open(os.path.join(self.low_light_dir, low_light_name))
        ground_truth_image = Image.open(os.path.join(self.ground_truth_dir, ground_truth_name))

        # 获取裁剪图像
        low_light_patch, ground_truth_patch = utils.get_patch_low_light(low_light_image, ground_truth_image, self.patch_size)
        # 获取增强图像,augmentation_low_light()是旋转增强函数
        low_light_patch, ground_truth_patch = utils.augmentation_low_light(low_light_patch, ground_truth_patch, self.args)

        # 将图像变量转换为Numpy数组，并将元素类型转换成长整型
        buffer1 = np.asarray(low_light_patch).astype(np.long)
        buffer2 = np.asarray(ground_truth_patch).astype(np.long)
        # 从图像1中减去图像2
        attention_patch = np.clip(buffer2 - buffer1, 0, 255).astype(np.uint8)
        # 转换为张量
        low_light_tensor = self.transform2(low_light_patch)
        ground_truth_tensor = self.transform2(ground_truth_patch)
        attention_tensor = self.transform2(attention_patch)

        return low_light_tensor, ground_truth_tensor, attention_tensor, ground_truth_name

class MALIETrainGlobal(Dataset):
    def __init__(self, root_dir, args):
        """
        Arguments:
            1) root directory -> \MALIE\TRAIN\
            2) arguments -> args
        """
        self.low_light_dir = root_dir + 'low'
        self.ground_truth_dir = root_dir + 'normal'
        self.low_light_img_list = os.listdir(root_dir + 'low')
        self.ground_truth_img_list = os.listdir(root_dir + 'normal')

        self.transform2 = transforms.Compose([
            transforms.ToTensor()
        ])

        # patch_size : default == 128
        self.args = args
        self.patch_size = 240


    def __len__(self):
        return len(self.low_light_img_list)

    def __getitem__(self, idx):

        low_light_name, ground_truth_name = self.low_light_img_list[idx], self.ground_truth_img_list[idx]

        low_light_image = Image.open(os.path.join(self.low_light_dir, low_light_name))
        ground_truth_image = Image.open(os.path.join(self.ground_truth_dir, ground_truth_name))

        # 获取裁剪图像
        low_light_patch, ground_truth_patch = utils.get_patch_low_light_global(low_light_image, ground_truth_image, self.patch_size)

        # 获得增强图像
        low_light_patch, ground_truth_patch = utils.augmentation_low_light(low_light_patch, ground_truth_patch, self.args)
        # 获取图像缓冲区作为ndarray  Get the image buffer as ndarray

        buffer1 = np.asarray(low_light_patch)

        buffer2 = np.asarray(ground_truth_patch)

        # 从图像1中减去图像2

        attention_patch = buffer2 - buffer1

        # 转换为张量
        low_light_tensor = self.transform2(low_light_patch)
        ground_truth_tensor = self.transform2(ground_truth_patch)
        attention_tensor = self.transform2(attention_patch)

        return low_light_tensor, ground_truth_tensor, attention_tensor, ground_truth_name

# Class for test
class MALIETest(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            1) root directory -> \MALIE\TEST\
            2) arguments -> args
        """
        self.root_dir = root_dir
        self.test_img_list = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        """
        获取测试图像张量。
        它返回一个形状为[3，height，width]的浮点张量元组。
        它们表示像素值在[0，1]范围内的RGB图像
        return：测试图像张量，文件名
        """
        test_img_name = self.test_img_list[idx]
        # 打开图像
        test_image = Image.open(os.path.join(self.root_dir, test_img_name))
        test_image_tensor = self.transform(test_image)
        return test_image_tensor, test_img_name

class MALIETrainForNoise(Dataset):
    def __init__(self, root_dir, args):
        self.low_light_dir = root_dir + 'low'
        self.ground_truth_dir = root_dir + 'high'
        self.low_light_img_list = os.listdir(root_dir + 'low')
        self.ground_truth_img_list = os.listdir(root_dir + 'high')
        # 输出数据集数量
        print(self.low_light_img_list.__len__())
        print(self.ground_truth_img_list.__len__())
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2 = transforms.Compose([
            transforms.ToTensor()
        ])
        # patch_size : default == 128
        self.args = args
        self.patch_size = 240

    def __len__(self):
        return len(self.low_light_img_list)

    def __getitem__(self, idx):
        """
        获取一对随机的图像裁剪。
        它返回一个形状为[3，height，width]的浮点张量元组。
        它们表示像素值在[0，1]范围内的RGB图像
        return：微光图像，真实标准图像张量
        """
        low_light_name, ground_truth_name = self.low_light_img_list[idx], self.ground_truth_img_list[idx]

        low_light_image = Image.open(os.path.join(self.low_light_dir, low_light_name))
        ground_truth_image = Image.open(os.path.join(self.ground_truth_dir, ground_truth_name))

        # 获取裁剪图像
        low_light_patch, ground_truth_patch = utils.get_patch_low_light(low_light_image, ground_truth_image,
                                                                        self.patch_size)
        # 获取增强图像,augmentation_low_light()是旋转增强函数
        low_light_patch, ground_truth_patch = utils.augmentation_low_light(low_light_patch, ground_truth_patch,
                                                                           self.args)
        # 根据low_light_patch生成噪声图像
        noise_patch = self.generate_noise_from_low_light(low_light_patch)
        # 将图像变量转换为Numpy数组，并将元素类型转换成长整型
        buffer1 = np.asarray(low_light_patch).astype(np.long)
        buffer2 = np.asarray(ground_truth_patch).astype(np.long)
        # 从图像1中减去图像2
        attention_patch = np.clip(buffer2 - buffer1, 0, 255).astype(np.uint8)
        # 转换为张量
        low_light_tensor = self.transform2(low_light_patch)
        ground_truth_tensor = self.transform2(ground_truth_patch)
        attention_tensor = self.transform2(attention_patch)
        noise_tensor = self.transform2(noise_patch)

        return low_light_tensor, ground_truth_tensor, attention_tensor, noise_tensor, ground_truth_name

    def generate_noise_from_low_light(self, low_light_patch):
        # 在 low_light_patch 上生成噪声
        # 返回噪声图像的Numpy数组
        # 示例：生成高斯噪声
        noise = np.random.normal(0, 25, low_light_patch.size[:2]).astype(np.long)
        noisy_patch = np.clip(low_light_patch + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)

        return noisy_patch