# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import torch
from matplotlib import pyplot as plt

from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_fucntion(a, b):
    # mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        # print(a[item].shape)
        # print(b[item].shape)
        # loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        # loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss


def train(_class_):
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = './mvtec/' + _class_ + '/train'
    test_path = './mvtec/' + _class_
    ckp_path = './checkpoints/' + 'wres50_' + _class_ + '.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))  # bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px


#######################################################################################
def dif(__class__):
    image_size = 256
    test_path = './mvtec/' + __class__
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    device = 'cpu'
    l1g = torch.zeros([1, 256, 64, 64])
    l2g = torch.zeros([1, 512, 32, 32])
    l3g = torch.zeros([1, 1024, 16, 16])
    l1a = torch.zeros([1, 256, 64, 64])
    l2a = torch.zeros([1, 512, 32, 32])
    l3a = torch.zeros([1, 1024, 16, 16])
    countg = 0
    counta = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            inputs = encoder(img)
            if label == 0:
                countg += 1
                l1g += inputs[0]
                l2g += inputs[1]
                l3g += inputs[2]
            else:
                counta += 1
                l1a += inputs[0]
                l2a += inputs[1]
                l3a += inputs[2]
    print(countg)
    print(counta)
    l1g = l1g.view(l1g.shape[1], -1) / countg
    l2g = l2g.view(l2g.shape[1], -1) / countg
    l3g = l3g.view(l3g.shape[1], -1) / countg
    l1a = l1a.view(l1a.shape[1], -1) / counta
    l2a = l2a.view(l2a.shape[1], -1) / counta
    l3a = l3a.view(l3a.shape[1], -1) / counta
    l1 = 1 - F.cosine_similarity(l1g, l1a)
    l2 = 1 - F.cosine_similarity(l2g, l2a)
    l3 = 1 - F.cosine_similarity(l3g, l3a)
    _, idx1 = torch.sort(l1, descending=True)
    _, idx2 = torch.sort(l2, descending=True)
    _, idx3 = torch.sort(l3, descending=True)
    x = range(256)
    y1 = torch.mean(l1g, dim=1)
    y2 = torch.mean(l1a, dim=1)
    print(y1.shape)
    showplt(x, y1, y2, idx1, __class__)
    showtest(x, y1, y2, idx1, __class__)
    print(__class__)
    print('l1:', idx1[:10])
    print('l2:', idx2[:10])
    print('l3:', idx3[:10])


def showplt(x, y1, y2, idx, __class__):
    pair = []
    color = ['b' for _ in range(256)]
    for i in idx[:10]:
        color[i] = 'r'
        pair.append([y1[i], y2[i]])
    pair_list = np.array(pair)
    img_df = pd.DataFrame(pair_list, index=idx[:10].tolist())
    img_df.plot(kind="bar", rot=0, title=__class__)
    plt.show()
    plt.title(__class__+'_good')
    plt.bar(x, y1, color=color)
    plt.show()
    plt.title(__class__+'_ab')
    plt.bar(x, y2, color=color)
    plt.show()


def showtest(x, y1, y2, idx, __class__):
    pair = []
    for i in range(200, 256):
        pair.append([y1[i], y2[i]])
    pair_list = np.array(pair)
    img_df = pd.DataFrame(pair_list, index=[i for i in range(200, 256)])
    img_df.plot(kind="bar", rot=270, title=__class__)
    plt.show()
#######################################################################################


if __name__ == '__main__':
    # setup_seed(111)
    # item_list = ['bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
    #              'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    item_list = ['bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        train(i)
    # image_size = 256
    # test_path = './mvtec/' + 'pill'
    # data_transform, gt_transform = get_data_transforms(image_size, image_size)
    # test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    #
    # device = 'cpu'
    # ckp_path = './checkpoints/wres50_pill.pth'
    # encoder, bn = wide_resnet50_2(pretrained=True)
    # decoder = de_wide_resnet50_2(pretrained=False)
    # ckp = torch.load(ckp_path)
    # for k, v in list(ckp['bn'].items()):
    #     if 'memory' in k:
    #         ckp['bn'].pop(k)
    # decoder.load_state_dict(ckp['decoder'])
    # bn.load_state_dict(ckp['bn'])
    # l1 = torch.zeros(256)
    # l2 = torch.zeros(512)
    # l3 = torch.zeros(1024)
    # count = 0
    # with torch.no_grad():
    #     for img, gt, label, _ in test_dataloader:
    #         print(count)
    #         count += 1
    #         inputs = encoder(img)
    #         outputs = decoder(bn(inputs))
    #         for i in range(3):
    #             fs = inputs[i]
    #             ft = outputs[i]
    #             a = 1 - F.cosine_similarity(fs, ft, dim=-4)
    #             a = torch.mean(a, dim=(1, 2))
    #             if i == 0:
    #                 l1 += a
    #             elif i == 1:
    #                 l2 += a
    #             else:
    #                 l3 += a
    # _, idx1 = torch.sort(l1, descending=True)
    # _, idx2 = torch.sort(l2, descending=True)
    # _, idx3 = torch.sort(l3, descending=True)
    # print(idx1[:10])
    # print(idx2[:10])
    # print(idx3[:10])
    # auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
    # print('Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Pixel Aupro:{:.3}'.format(auroc_px, auroc_sp, aupro_px))
