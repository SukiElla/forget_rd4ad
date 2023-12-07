import PIL
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from de_resnet import de_wide_resnet50_2
from resnet import wide_resnet50_2
from test import min_max_norm, cvt2heatmap


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        # fs_norm = F.normalize(fs, p=2)
        # ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    # if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    print(img.shape)
    print(anomaly_map.shape)
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def chan():
    size = 256
    isize = 256
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    encoder.eval()
    data_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = './mvtec/bottle/test'
    data = ImageFolder(root=path, transform=data_transform)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    l1g = torch.zeros([1, 256, 64, 64])
    l2g = torch.zeros([1, 512, 32, 32])
    l3g = torch.zeros([1, 1024, 16, 16])
    l1a = torch.zeros([1, 256, 64, 64])
    l2a = torch.zeros([1, 512, 32, 32])
    l3a = torch.zeros([1, 1024, 16, 16])
    for img, label in loader:
        inputs = encoder(img)
        if label in [0, 1, 2]:
            print('ab')
            l1a = l1a + inputs[0]
            l2a = l2a + inputs[1]
            l3a = l3a + inputs[2]
        else:
            print('good')
            l1g = l1g + inputs[0]
            l2g = l2g + inputs[1]
            l3g = l3g + inputs[2]
    dif_1, dif_2, dif_3 = torch.abs(l1g - l1a), torch.abs(l2g - l2a), torch.abs(l3g - l3a)
    dif_1 = torch.mean(dif_1, dim=(2, 3)).view(256)
    dif_2 = torch.mean(dif_2, dim=(2, 3)).view(512)
    dif_3 = torch.mean(dif_3, dim=(2, 3)).view(1024)
    _, idx1 = torch.sort(dif_1, descending=True)
    _, idx2 = torch.sort(dif_2, descending=True)
    _, idx3 = torch.sort(dif_3, descending=True)
    print('layer1: {}, layer2: {}, layer3: {}'.format(idx1[:2], idx2[:2], idx3[:2]))
    print('layer1: {} {}, layer2: {} {}, layer3: {} {}'.format(
        dif_1[idx1[:2][0]], dif_1[idx1[:2][1]], dif_2[idx2[:2][0]], dif_2[idx2[:2][1]],
        dif_3[idx3[:2][0]], dif_3[idx3[:2][1]], ))


def show_i(a):
    a = min_max_norm(a)
    a = cvt2heatmap(a * 255)
    plt.imshow(a)
    plt.axis('off')
    plt.show()


size = 256
isize = 256
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
data_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
gt_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = './teimg'
gt_path = './gt'
ckp_path = './checkpoints/wres50_cable.pth'
imgs = []
encoder, bn = wide_resnet50_2(pretrained=True)
encoder.eval()
decoder = de_wide_resnet50_2(pretrained=False)
ckp = torch.load(ckp_path)
decoder.load_state_dict(ckp['decoder'])
bn.load_state_dict(ckp['bn'])
decoder.eval()
bn.eval()
data = ImageFolder(root=path, transform=data_transform)
gt = ImageFolder(root=gt_path, transform=data_transform)
loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
gt_loader = torch.utils.data.DataLoader(gt, batch_size=1, shuffle=False)
# for img, label in gt_loader:
#     img[img > 0.5] = 1
#     img[img <= 0.5] = 0
#     img = img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
#     imgs.append(img)
# count = 0
for img, label in loader:
    # print('img:')
    # inputs = encoder(img)
    # mid = bn(inputs)
    # outputs = decoder(mid)
    # for i in range(3):
    #     fs = inputs[i]
    #     ft = outputs[i]
    #     a = 1 - F.cosine_similarity(fs, ft, dim=-4)
    #     a = torch.mean(a, dim=(1, 2))
    #     _, idx = torch.sort(a, descending=True)
    #     print(idx[:5])
    # l1 = outputs[0][0][100].detach().numpy()
    # l2 = outputs[1][0][265].detach().numpy()
    # l3 = outputs[2][0][626].detach().numpy()
    # show_i(l1)
    # show_i(l2)
    # show_i(l3)
    # anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
    # # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    # ano_map = min_max_norm(anomaly_map)
    # ano_map = cvt2heatmap(ano_map * 255)
    # img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
    # img = np.uint8(min_max_norm(img) * 255)
    # print(ano_map.shape)
    # print(img.shape)
    # # plt.imshow(img)
    # # plt.axis('off')
    # # plt.show()
    # ano_map = show_cam_on_image(imgs[count], ano_map)
    # count += 1
    # plt.imshow(ano_map)
    # plt.axis('off')
    # plt.show()
