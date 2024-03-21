
import torch
import torch.nn as nn
from torch import Tensor
from matplotlib import cm
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from PIL import Image

import torchvision.transforms as transforms
from NaiveLSTM import NaiveLSTM
# hook
feature_map = []     # 建立列表容器，用于盛放输出特征图

def forward_hook(module, inp, outp):     # 定义hook
    feature_map.append(outp)    # 把输出装入字典feature_map

### test 
inputs = torch.ones(1, 1, 10)
h0 = torch.ones(1, 1, 20)
c0 = torch.ones(1, 1, 20)
print(h0.shape, h0)
print(c0.shape, c0)
print(inputs.shape, inputs)
# test naive_lstm with input_size=10, hidden_size=20
naive_lstm = NaiveLSTM()
layer1 = nn.Linear(20, 1)
naive_lstm.register_forward_hook(forward_hook) 
# reset_weigths(naive_lstm)
output1, hn1, cn1 = naive_lstm(inputs, (h0, c0))
res = layer1(output1)
print(hn1.shape, cn1.shape, output1.shape)
print(hn1)
print(cn1)
print(output1)
print(feature_map[0][0].size())

img_path = 'D:\\pythoncode\\0Aliyun-run\\develop-service\\1.JPEG'     # 输入图片的路径
save_path = 'D:\\pythoncode\\0Aliyun-run\\develop-service\\CAM1.png'    # 类激活图保存路径
# cls = torch.argmax(out).item()    # 获取预测类别编码
# weights = net._modules.get('fc').weight.data[cls,:]    # 获取类别对应的权重
weights = layer1.weight.data
cam = (weights * feature_map[0][0]).sum(0)

def _normalize(cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

        return cams

cam = _normalize(F.relu(cam, inplace=True)).cpu()
mask = to_pil_image(cam.detach().numpy(), mode='F')
import numpy as np
def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
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

preprocess = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
orign_img = Image.open(img_path).convert('RGB')    # 打开图片并转换为RGB模型
img = preprocess(orign_img)     # 图片预处理
img = torch.unsqueeze(img, 0)     # 增加batch维度 [1, 3, 224, 224]

result = overlay_mask(orign_img, mask) 
result.show()
result.save(save_path)