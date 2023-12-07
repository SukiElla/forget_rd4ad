from __future__ import division

import heapq

import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image

import copy
import heapq

from PIL import Image
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
from torchvision import transforms as T


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def imagenet_cos_mvtec(_class_):
    image_size = 256
    topk = 10
    count = 0
    max_count = 100
    data_transform, _ = get_data_transforms(image_size, image_size)
    cos_loss = torch.nn.CosineSimilarity()

    mvtec_path = './mvtec/' + _class_ + '/test'
    mini_path = 'mini-imagenet'
    save_path = './result/' + _class_ + '/train/imagenet_anomaly'
    mvtec_data = ImageFolder(root=mvtec_path, transform=data_transform)
    mini_data = ImageFolder(root=mini_path, transform=data_transform)
    mini_data_origin = ImageFolder(root=mini_path)
    cos_all = []
    for mini_img, mini_lable in mini_data:
        count += 1
        print(count)
        cos_result = []
        for mvtec_img, mvtec_lable in mvtec_data:
            result = torch.mean(1 - cos_loss(mini_img, mvtec_img))
            cos_result.append(result.item())
        cos_all.append(np.mean(cos_result))
        if count == max_count:
            break
    print(_class_, cos_all)
    min_num_index = heapq.nsmallest(topk, range(len(cos_all)), cos_all.__getitem__)
    for i in min_num_index:
        mini_data_origin[i][0].save(save_path + '/{name}.jpg'.format(name=cos_all[i]))
        # save_image(mini_data[i][0], save_path+'/{name}.jpg'.format(name=cos_all[i]))


def unlearnerloss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss


def unlearning_train(_class_):
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
    KL_temperature = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = './result/' + _class_
    train_data = ImageFolder(root=train_path, transform=data_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    competent_teacher, bn_ct = wide_resnet50_2(pretrained=True)
    incompetent_teacher, bn_it = wide_resnet50_2(pretrained=False)
    student, bn_s = wide_resnet50_2(pretrained=True)
    competent_teacher = competent_teacher.to(device)
    bn_ct = bn_ct.to(device)
    incompetent_teacher = incompetent_teacher.to(device)
    bn_it = bn_it.to(device)
    student = student.to(device)
    bn_s = bn_s.to(device)

    competent_teacher.eval()
    bn_ct.eval()
    incompetent_teacher.eval()
    bn_it.eval()

    optimizer = torch.optim.Adam(list(student.parameters()+list(bn_s.parameters())), lr=learning_rate,
                                 betas=(0.5, 0.999))
    for epoch in range(epochs):
        student.train()
        bn_s.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            with torch.no_grad():
                competent_output = bn_ct(competent_teacher(img))
                incompetent_output = bn_it(incompetent_teacher(img))
            student_output = bn_s(student(img))
            loss = unlearnerloss(output=student_output, labels=label, full_teacher_logits=competent_output,
                                 unlearn_teacher_logits=incompetent_output, KL_temperature=KL_temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
    torch.save(student.state_dict(), "unlearn.pth")
    return student, bn_s


def stu_bn_train(_class_, student, bn_s):
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

    encoder, bn = student, bn_s
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


if __name__ == '__main__':
    imagenet_cos_mvtec('bottle')
    item_list = ['bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        imagenet_cos_mvtec(i)









