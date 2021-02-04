import os
import cv2
import time
import torch
import numpy as np
import torchvision
from torch import nn
from PIL import Image
import torch.utils.data as Data

import matplotlib.pylab as plt
from torchvision.datasets import ImageFolder


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])


class Residual(nn.Module):  # 用于ResNet, 最后已经ReLU了!
    def __init__(self, in_c, out_c, reshape=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        if reshape:
            self.conv3 = nn.Conv2d(in_c, out_c, 1, stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nn.functional.relu(X + Y)


def is_valid_file(path):
    img_ext = ('.jpg', '.jpeg', '.png')
    if os.path.splitext(path)[-1] not in img_ext:
        return False
    else:
        if 'rgb' in path:
            return True
        else:
            return False


def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().cpu().sum().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X.to(device), is_training=False).argmax(dim=1) == y.to(device)).float().sum().item()
                else:
                    acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def data_loader(batch_size, split_eta=0.8, resize=None, root='./dataset/train'):
    transform = []
    if resize:
        transform.append(torchvision.transforms.Resize(resize))
    transform.append(torchvision.transforms.ToTensor())
    transform.append(torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))  # 归一化
    transform = torchvision.transforms.Compose(transform)
    data = ImageFolder(root, transform=transform, is_valid_file=is_valid_file)
    train_size = int(split_eta * len(data))
    train_data, test_data = torch.utils.data.random_split(data, [train_size, len(data) - train_size])
    train_iter = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_iter, test_iter, data.class_to_idx


def move_detection(images):
    image_count = len(images)
    images = [image > 0 for image in images]
    image_trajectory = np.mean(np.stack(images), axis=0)
    return np.sum(image_trajectory * images[0]) / np.sum(images[0]) < (1 - 2 / image_count)


def move_direction(last_pos, cur_pos):
    x_size, y_size = 80, 80  # 整体移动10像素视为发生运动
    last_x, last_y = last_pos[2], last_pos[3]
    cur_x, cur_y = cur_pos[2], cur_pos[3]
    move_right, move_left = cur_y - last_y > y_size / 10, last_y - cur_y > y_size / 10
    move_down, move_up = cur_x - last_x > x_size / 10, last_x - cur_x > x_size / 10
    return np.array([move_up, move_down, move_left, move_right])


def edge_contacted(image):
    x_len, y_len = image.shape
    x_pos, y_pos = np.nonzero(image)
    x_min, x_max = np.min(x_pos), np.max(x_pos)
    y_min, y_max = np.min(y_pos), np.max(y_pos)
    center_x, center_y = np.mean(x_pos), np.mean(y_pos)
    # center_x, center_y = (x_max + x_min) / 2, (y_max + y_min) / 2
    x_size, y_size = x_max - x_min, y_max - y_min
    x_top, x_mid, x_bottom = center_x < x_len // 5, x_len // 5 < center_x < 4 * x_len // 5, center_x > 4 * x_len // 5
    y_left, y_mid, y_right = center_y < y_len // 5, y_len // 5 < center_y < 4 * y_len // 5, center_y > 4 * y_len // 5
    if x_min > 0.07 * x_len:
        top_left, top_mid, top_right = False, False, False
    else:
        top_left, top_mid, top_right = y_left, y_mid, y_right
    if x_max < 0.93 * x_len:
        bottom_left, bottom_mid, bottom_right = False, False, False
    else:
        bottom_left, bottom_mid, bottom_right = y_left, y_mid, y_right
    if y_min > 0.07 * y_len:
        left_top, left_mid, left_bottom = False, False, False
    else:
        left_top, left_mid, left_bottom = x_top, x_mid, x_bottom
    if y_max < 0.93 * y_len:
        right_top, right_mid, right_bottom = False, False, False
    else:
        right_top, right_mid, right_bottom = x_top, x_mid, x_bottom
    return np.array([top_left, top_mid, top_right, bottom_left, bottom_mid, bottom_right, left_top, left_mid,
                     left_bottom, right_top, right_mid, right_bottom]), [x_size, y_size, center_x, center_y]


def video_loader(folder_name, is_mask=True):
    sub_folder = '/mask/' if is_mask else '/rgb/'
    extension = '.png' if is_mask else '.jpg'
    mask_dir = folder_name + sub_folder
    mask_names = list(filter(lambda x: x.endswith(extension), os.listdir(mask_dir)))
    images = [np.array(Image.open(mask_dir + mask_name)) for mask_name in mask_names]
    is_moved = move_detection(images)
    return images, is_moved


def image_process(image):  # 提取最大的连通分量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=None)
    largest_label = np.argmax(stats[1:, 4]) + 1
    # nearby_labels = np.where(
    #     np.linalg.norm(centroids[largest_label] - centroids, axis=1) < np.sqrt(stats[largest_label][4]))
    # image = image * np.in1d(labels, nearby_labels).reshape(image.shape)
    image = image * (labels == largest_label)
    return image
    # largest_centroid = centroids(largest_label)
    # largest_area = stats[largest_label, 4] 考虑保留前几块区域


def collide_detection(images, is_moved):
    collided = np.array([False for _ in range(12)])
    if not is_moved:
        return collided.astype("uint8").tolist()
    else:
        images = [image_process(image) for image in images]
        offshore, last_pos = edge_contacted(images[0])
        offshore = np.bitwise_not(offshore)  # 是否离开各点
        for image in images[1:]:
            contact, cur_pos = edge_contacted(image)
            move_dic = np.repeat(move_direction(last_pos, cur_pos), 3)
            # 检测是否离开初始位置，向对应边缘移动视为离开初始位置
            offshore = np.bitwise_or(np.bitwise_or(offshore, np.bitwise_not(contact)), move_dic)
            # 离开初始位置后再次接触则且有相对运动视为发生碰撞
            collided = np.bitwise_and(np.bitwise_or(np.bitwise_and(contact, move_dic), collided), offshore)
            last_pos = cur_pos

        return collided.astype("uint8").tolist()


def collide_detection_blocks(images, is_moved):
    collided = collide_detection(images, is_moved)
    # blocks = [collided[:, 0] | collided[:, 6], collided[:, 1], collided[:, 2] | collided[:, 9], collided[:, 7],
    #           collided[:, 10], collided[:, 3] | collided[:, 8], collided[:, 4], collided[:, 5] | collided[:, 11]]
    # blocks = [collided[0] or collided[6], collided[1], collided[2] or collided[9], collided[7],
    #           collided[10], collided[3] or collided[8], collided[4], collided[5] or collided[11]]
    blocks = [collided[0] or collided[1] or collided[2], collided[3] or collided[4] or collided[5],
              collided[6] or collided[7] or collided[8], collided[9] or collided[10] or collided[11]]
    return blocks


def train(net, train_iter, test_iter, optimizer, lr_scheduler, num_epochs, device, save_file=False,
          save_name='alexnet.pt'):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device)
        lr_scheduler.step()
        if test_acc > best_test_acc and save_file:
            torch.save(net.state_dict(), save_name)
            print('save state_dict at test_acc %.6f' % test_acc)
            best_test_acc = test_acc
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
