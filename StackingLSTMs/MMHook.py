import torch
import torch.nn as nn
from torch import Tensor
from matplotlib import cm
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageOps

import torchvision.transforms as transforms
from NaiveLSTM import NaiveLSTM
import numpy as np
import matplotlib.pyplot as pyplot
import cv2
import datetime
import random

# hook
class My_hook_lstm:
    def __init__(self):
        self.out_map = []     # 建立列表容器，用于盛放输出特征图
        self.inp_map = []     # 建立列表容器，用于盛放输出特征图
    def forward_hook(self, module, inp, outp):     # 定义hook
        self.out_map.append(outp)    # 把输出装入字典feature_map
        self.inp_map.append(inp)    # 把输出装入字典feature_map

class My_hook_linear:
    def __init__(self):
        self.weights_map = []     # 建立列表容器，用于盛放输出特征图
        self.inp_map = []     # 建立列表容器，用于盛放输出特征图
    def forward_hook(self, module, inp, outp):     # 定义hook
        weights = module.weight.data  # 获取类别对应的权重
        self.weights_map.append(weights)    # 把输出装入字典feature_map
        self.inp_map.append(inp)    # 把输出装入字典feature_map

class Hook_process:
    def overlay_mask(self, img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
                """Overlay a colormapped mask on a background image
                Args:
                    img: background image
                    mask: mask to be overlayed in grayscale
                    colormap: colormap to be applied on the mask
                    alpha: transparency of the background image
                Returns:
                    overlayed image
                """
                if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
                    raise TypeError('img and mask arguments need to be PIL.Image')
                if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
                    raise ValueError('alpha argument is expected to be of type float between 0 and 1')

                cmap = cm.get_cmap(colormap)    
                # Resize mask and apply colormap
                overlay = mask.resize(img.size, resample=Image.BICUBIC)
                overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
                # Overlay the image with the mask
                overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))
                return overlayed_img

    def convert_to_3D(self, image):
        height, width = image.shape[:2]
        # 创建一个三维矩阵来保存转换后的图像
        new_image = np.zeros((height, width, 3), dtype=np.uint8)
    
        # 将二维图像的每个像素值赋值给新的三维矩阵的每个通道
        for i in range(height):
            for j in range(width):
                new_image[i][j] = [image[i][j], image[i][j], image[i][j]]
    
        return new_image

    def min_max_normalize(self, arr):
        min_val = torch.min(arr)
        max_val = torch.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        return normalized_arr

    def process_hook(self, my_hook_lstm, my_hook_linear):
        # 测试数据长度298
        indexSameple = random.sample(range(1, 298), 5)
        for index in indexSameple:
            cam = []
            cam_in = []
            # 获取当前时间
            now = datetime.datetime.now()
            # 格式化时间为字符串
            # filename = now.strftime("%Y-%m-%d_%H-%M-%S.%f")
            save_path1 = 'D:\\pythoncode\\Paper\\TravelTime0307\\A{}.png'.format(index)    # 类激活图保存路径
            save_path2 = 'D:\\pythoncode\\Paper\\TravelTime0307\\B{}.png'.format(index)    # 类激活图保存路径
            
            weights = my_hook_linear.weights_map[-index]  # 获取类别对应的权重
            nBegin = len(my_hook_lstm.out_map)-4*index
            nEnd = len(my_hook_lstm.out_map)-4*(index-1)
            for (hlist,clist) in my_hook_lstm.out_map[nBegin:nEnd]:
                cam.append(hlist)
            cam = weights.view((7,16)) * torch.cat(cam,dim=1)

            nBegin = len(my_hook_lstm.inp_map)-4*index
            nEnd = len(my_hook_lstm.inp_map)-4*(index-1)
            for inp in my_hook_lstm.inp_map[nBegin:nEnd]:
                cam_in.append(inp[0][0,:,:])
            cam_in = torch.cat(cam_in,dim=1)

            def _normalize(cams: Tensor) -> Tensor:
                    """CAM normalization"""
                    cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
                    cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
                    # cams.sub_(cams.min(-1).values.unsqueeze(-1))
                    # cams.div_(cams.max(-1).values.unsqueeze(-1))

                    return cams

            cam_in = self.min_max_normalize(cam_in).cpu()
            data = (cam_in * 255.0).type(torch.uint8)  # 转换数据类型
            data_3D = self.convert_to_3D(data.detach().numpy())
            orign_img = Image.fromarray(data_3D)
            # orign_img.save(save_path1)
            # 显示新图片
            # pyplot.imshow(data)
            pyplot.imshow(data_3D)
            # pyplot.imshow(orign_img)
            pyplot.savefig(save_path1)
            # pyplot.show()

            feature = self.min_max_normalize(cam).cpu()
            cam = _normalize(F.relu(feature, inplace=True)).cpu()
            feature = feature.reshape((7,4,4)).sum(2)
            feature = (feature * 255.0).type(torch.uint8)  # 转换数据类型
            feature_3D = self.convert_to_3D(feature.detach().numpy())
            mask = Image.fromarray(feature_3D)
            # mask.save(save_path2)
            # pyplot.imshow(feature)
            pyplot.imshow(feature_3D)
            # pyplot.imshow(mask)
            pyplot.savefig(save_path2)
            # pyplot.show()

    def process_hook20240317(self, model):
        print(self.feature_map[-1][0].size())
        save_path = 'D:\\pythoncode\\0Aliyun-run\\develop-service\\CAM1.png'    # 类激活图保存路径
        weights = model.linear.weight.data  # 获取类别对应的权重
        cam, cam_in = [], []
        for (hlist,clist) in self.feature_map:
            cam.append(hlist)
        cam = weights.view((7,16)) * torch.cat(cam,dim=1)

        for inp in self.inp_map:
            cam_in.append(inp[0][0,:,:])
        cam_in = torch.cat(cam_in,dim=1)

        def _normalize(cams: Tensor) -> Tensor:
                """CAM normalization"""
                cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
                cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
                # cams.sub_(cams.min(-1).values.unsqueeze(-1))
                # cams.div_(cams.max(-1).values.unsqueeze(-1))

                return cams

        cam_in = self.min_max_normalize(cam_in).cpu()
        data = (cam_in * 255.0).type(torch.uint8)  # 转换数据类型
        data_3D = self.convert_to_3D(data.detach().numpy())
        orign_img = Image.fromarray(data_3D)
        orign_img1 = ImageOps.equalize(orign_img)
        # 显示新图片
        pyplot.imshow(data)
        pyplot.imshow(orign_img)
        pyplot.imshow(orign_img1)
        pyplot.show()

        import numpy as np

        preprocess = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        feature = self.min_max_normalize(cam).cpu()
        cam = _normalize(F.relu(feature, inplace=True)).cpu()
        feature = feature.reshape((7,4,4)).sum(2)
        feature = (feature * 255.0).type(torch.uint8)  # 转换数据类型
        feature_3D = self.convert_to_3D(feature.detach().numpy())
        mask = Image.fromarray(feature_3D)
        mask1 = ImageOps.equalize(mask)
        pyplot.imshow(feature_3D)
        pyplot.imshow(mask)
        pyplot.imshow(mask1)
        pyplot.show()
        mask = to_pil_image(cam.detach().numpy(), mode='F')
        result = self.overlay_mask(orign_img, mask) 
        result.show()
        result.save(save_path)