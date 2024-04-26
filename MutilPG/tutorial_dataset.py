import json
import cv2
import numpy as np
import os
from annotator.util import resize_image
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch



class MyDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self.dataset_path = dataset_path
        self.prompt_path = os.path.join(dataset_path, 'test_new.jsonl')
        with open(self.prompt_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_paths = []
        item = self.data[idx]
        source1_value = item["source"]

        paths = source1_value.split('; ')

        file_paths.extend(paths)

        images = []
        H, W = 512, 512
        # 遍历 file_path 列表，并为每个路径读取图片
        for i, source_path in enumerate(file_paths, start=1):
            # 读取图片
            source_path1 = os.path.join(self.dataset_path, source_path)
            source = cv2.imread(source_path1)
            source = cv2.resize(source, (H, W), interpolation=cv2.INTER_LINEAR)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = source.astype(np.float16) / 255.0

            # 将图片存储在字典中，键名为 'image1', 'image2', ...
            images.append(source)

        target_filename = item['target']
        mask_filename = item['mask']
        prompt = item['prompt']
        
        new_images = self.radom_choise(images)
        hint = np.concatenate(new_images, axis=2)

        target_filename = os.path.join(self.dataset_path, target_filename)
        target = cv2.imread(target_filename)
        target = cv2.resize(target, (H, W), interpolation=cv2.INTER_LINEAR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        mask_filename = os.path.join(self.dataset_path, mask_filename)
        mask = cv2.imread(mask_filename)
        mask = cv2.resize(mask, (H, W), interpolation=cv2.INTER_LINEAR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.get_mask(mask)

        #单独的创建hint



        # Normalize target images to [-1, 1].
        target = (target.astype(np.float16) / 127.5) - 1.0
        sample = {
            "jpg":target,
            "txt":prompt,
            "hint":hint,
            "mask":mask
        }
        return sample
    
    def radom_choise(self, image_list):
        target_length = 10
        new_images = image_list
        num = target_length-len(image_list)
        if num > 0 :
            for i in range(num):
                if random.random() < 0.9:
                    # 随机从原始图像中选择一个添加
                    new_images.append(random.choice(image_list))
                else:
                    # 添加一个形状和原图相同的零图像
                    new_images.append(np.zeros_like(image_list[0]))
        else:
            new_images = new_images[:target_length]
        return new_images
    

    def get_mask(self, mask):
        mask_blur = 1 
        image_resolution = 512
        mask_kernel_size = image_resolution // 8
        mask_kernel_size = mask_kernel_size if mask_kernel_size % 2 == 1 else mask_kernel_size + 1 
        H, W = 512, 512
        detected_map = mask
        detected_map[0:5, 0:5] = np.zeros((5, 5, 3), dtype=np.uint8)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        input_mask = np.zeros_like(detected_map[:, :, :1])

        input_mask[detected_map.sum(2) >  0] = 255
        # cv2.imwrite('/storage/zhaoliuqing/code/ControlNet-test/input_mask0.jpg', input_mask[:, :, ::-1])
        # dilation
        kernel = np.ones((mask_kernel_size, mask_kernel_size), np.uint8)
        input_mask = cv2.dilate(input_mask, kernel)

        # cv2.imwrite('/storage/zhaoliuqing/code/ControlNet-test/input_mask1.jpg', input_mask_expanded[:, :, ::-1])
        # mask_pixel: 1. visible
        mask_pixel = cv2.resize(input_mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)
        mask = cv2.resize(mask_pixel, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=-1)
        return mask
# def main():
#     # 创建数据集实例
#     dataset_path = "/storage/zhaoliuqing/data/train_controlnet/"
#     dataset = MyDataset(dataset_path)

#     # 创建 DataLoader
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

#     # 遍历 DataLoader
#     for i, batch in enumerate(dataloader):
#         for k in batch:
#             if k == "hint":
#                 zero_tensor = torch.zeros(512, 512,3)
#                 control  = batch[k]
#                 for n in range(10):
#                     sub_tensor = control[:, :, :, i:i+3]
#                     # 累加到全零张量上
#                     zero_tensor += sub_tensor.squeeze(0)  # 由于维度为1，需要去掉这个额外的维度
#                 final_tensor = zero_tensor.unsqueeze(0)

#                 final_tensor = final_tensor.clamp(0, 1)  # 确保所有值在0和1之间
#                 image_np = (final_tensor.squeeze(0).numpy() * 255).astype(np.uint8)  # 转换为uint8
#                 image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # 转换为BGR

#                 image_path = '/storage/zhaoliuqing/code/MutilPG/saved_image.png'
#                 cv2.imwrite(image_path, image_np)
#                 print(f"Image saved to {image_path}")
#                 print("1")

# if __name__ == "__main__":
#     main()


