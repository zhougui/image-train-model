# -*-coding:utf-8-*-
# date:2019-05-20
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import math
import numpy as np
from data_iter.heatmap_label import draw_gaussian
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def my_loss(landmarks, labels, w=1, alpha=2.1, theta=0.5, m=1.25, n=2):

    x = landmarks - labels
    absolute_x = torch.abs(x)
    # AWing loss
    # # print(alpha - labels)
    # A = w * (alpha - labels) * (theta ** (alpha - labels - 1)) / (1 + theta ** (alpha - labels))
    # C = A * theta - w * torch.log(1.0 + theta ** (alpha - labels))
    # # print(absolute_x)
    # losses = torch.where((theta > absolute_x),
    #                      w * torch.log(1.0 + (absolute_x) ** (alpha - labels)),
    #                      A * absolute_x - C)
    # MY LOSS
    A = w * (alpha - labels) * (theta ** (alpha-labels-1)) / ((m - labels / n) ** (alpha-labels) + theta ** (alpha - labels))
    C = A * theta**2 - w*torch.log(1.0+(theta/(m - labels/n))**(alpha-labels))
    # print(A)
    # print(C)
    # print(absolute_x)
    losses = torch.where((theta > absolute_x),
                         torch.where((labels > 0),
                                     w * torch.log(1.0 + (absolute_x / (m - labels / n)) ** (alpha - labels)),
                                     0 * torch.log(1.0 + (absolute_x / (m - labels / n)) ** (alpha - labels))),
                         A * absolute_x ** 2 - C)

    losses = torch.mean(losses, dim=1, keepdim=True)
    loss = torch.mean(losses)
    # print(loss)
    # print(losses.sum())
    # print(losses.shape)
    # loss = losses.sum() / len(losses)
    return loss


def wing_loss(landmarks, labels, w=0.06, epsilon=0.01):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, landmarks].  landmarks means x1,x2,x3,x4...y1,y2,y3,y4   1-D
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """

    x = landmarks - labels

    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)

    losses = torch.where((w > absolute_x), w * torch.log(1.0 + absolute_x / epsilon), absolute_x - c)

    # loss = tf.reduce_mean(tf.reduce_mean(losses, axis=[1]), axis=0)
    losses = torch.mean(losses, dim=1, keepdim=True)
    loss = torch.mean(losses)
    return loss


def got_total_wing_loss(output, crop_landmarks):
    loss = wing_loss(output, crop_landmarks)

    return loss


# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, w=2, epsilon=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, pred, target):
        delta_y = (target - pred).abs()
        temp = torch.ones(delta_y.shape)
        print(temp.shape)
        delta_y1 = delta_y[delta_y < self.w]
        print("delta_y1:", delta_y1.shape)
        delta_y2 = delta_y[delta_y >= self.w]
        print("delta_y2:", delta_y2.shape)
        loss1 = self.w * torch.log(1 + delta_y1 / self.epsilon)
        C = self.w - self.w * math.log(1 + self.w / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


# 画图
def L1(y_gt, y_pre):
    loss = np.sum(np.abs(y_gt - y_pre))
    print(loss)

    # 画图
    x = np.arange(-10.01, 10.01, 0.01)
    z = np.arange(0, 1.01, 0.01)
    # L1
    y1 = abs(x)
    dy1 = np.where(x > 0, 1, -1)
    # L2
    y2 = (1 / 2) * x ** 2
    dy2 = x
    # Smooth L1
    y3 = []
    for xx in x:
        if -1 < xx < 1:
            y3.append(0.5*xx**2)
        else:
            y3.append(abs(xx)-0.5)
    dy3 = []
    for xx in x:
        if -1 < xx < 1:
            dy3.append(xx)
        else:
            if xx < -1:
                dy3.append(-1)
            else:
                dy3.append(1)
    # wing loss
    y4 = []
    for xx in x:
        if 10 > xx > -10:
            y4.append(10*math.log(1+abs(xx)/0.5))
        else:
            y4.append(abs(xx)-(10 - 10*math.log(1+10/0.5)))
    dy4 = []
    for xx in x:
        if 2 > xx > -2:
            if xx < 0:
                dy4.append(-2*1/(1+abs(xx)/0.5)*1/0.5)
            else:
                dy4.append(2 * 1 / (1 + abs(xx) / 0.5) * 1 / 0.5)
        else:
            if xx < -2:
                dy4.append(-1)
            else:
                dy4.append(1)
    # Awing loss 1
    y5 = []
    A = 10*1.1*(0.5**0.1)/(1+0.5**1.1)
    for xx in x:
        if 0.5 > xx > -0.5:
            y5.append(10*math.log(1+abs(xx)**(2.1-1)/1))
        else:
            y5.append(A*abs(xx)-(0.5*A-10*math.log(1+0.5**1.1/1)))
    dy5 = []
    A = 1 * 1.1 * (0.5 ** 0.1) / (1 + 0.5 ** 1.1)
    for xx in x:
        if 0.5 > xx > -0.5:
            if xx < 0:
                dy5.append(-1 * 1 / (1 + abs(xx)**1.1) * 1.1 * 0.5**0.1)
            else:
                dy5.append(1 * 1 / (1 + abs(xx)**1.1) * 1.1 * 0.5**0.1)
        else:
            if xx < -0.5:
                dy5.append(-1*A)
            else:
                dy5.append(A)
    # Awing loss 0
    y6 = []
    A = 10 * 2.1 * 0.5 ** 1.1 / (1 + 0.5 ** 2.1)
    for xx in x:
        if 0.5 > xx > -0.5:
            y6.append(10 * math.log(1 + abs(xx) ** 2.1 / 1))
        else:
            y6.append(A*abs(xx) - (0.5*A-10 * math.log(1 + 0.5**2.1 / 1)))
    dy6 = []
    A = 1 * 2.1 * (0.5 ** 1.1) / (1 + 0.5 ** 2.1)
    for xx in x:
        if 0.5 > xx > -0.5:
            if xx < 0:
                dy6.append(-1 * 2.1 * (abs(xx) ** 1.1) / (1 + abs(xx) ** 2.1))
            else:
                dy6.append(1 * 2.1 * (abs(xx) ** 1.1) / (1 + abs(xx) ** 2.1))
        else:
            if xx < -0.5:
                dy6.append(-1 * A)
            else:
                dy6.append(A)

    # my loss 0
    y7 = []
    A = 10 * 2.1 * 0.5 ** 1.1 / (1.1 + 0.5 ** 2.1)
    for xx in x:
        if 0.5 > xx > -0.5:
            y7.append(10 * math.log(1 + abs(xx) ** 2.1 / 1.1))
        else:
            y7.append(A * xx ** 2 - (A * 0.5 ** 2 - 10 * math.log(1 + 0.5 ** 2.1 / 1.1)))
    dy7 = []
    A = 10 * 2.1 * (0.5 ** 1.1) / (1.1 + 0.5 ** 2.1)
    for xx in x:
        if 0.5 > xx > -0.5:
            if xx < 0:
                dy7.append(-10 * 2.1 * (abs(xx) ** 1.1) / (1.1 + abs(xx) ** 2.1))
            else:
                dy7.append(10 * 2.1 * (abs(xx) ** 1.1) / (1.1 + abs(xx) ** 2.1))
        else:
            dy7.append(2 * A * xx)

    # my loss 1
    y8 = []
    A = 10 * 1.1 * 0.5 ** 0.1 / (0.8 + 0.5 ** 1.1)
    for xx in x:
        if 0.5 > xx > -0.5:
            y8.append(10 * math.log(1 + abs(xx) ** 1.1 / 0.8))
        else:
            y8.append(A * xx ** 2 - (A * 0.5 ** 2 - 10 * math.log(1 + 0.5 ** 1.1 / 0.8)))
    dy8 = []
    A = 1 * 1.1 * (0.5 ** 0.1) / (0.8 + 0.5 ** 1.1)
    for xx in x:
        if 0.5 > xx > -0.5:
            if xx < 0:
                dy8.append(-1 * 1.1 * (abs(xx) ** 0.1) / (0.8 + abs(xx) ** 1.1))
            else:
                dy8.append(1 * 1.1 * (abs(xx) ** 0.1) / (0.8 + abs(xx) ** 1.1))
        else:
            dy8.append(2 * A * xx)

    # dy9 = []
    # x, z = np.meshgrid(x, z)
    # for i in range(len(x)):
    #     dy9.append([])
    #     for j in range(len(x[i])):
    #         A = 1 * (2.1 - z[i][j]) * (0.5 ** (1.1 - z[i][j])) / (0.8 + 0.5 ** (2.1 - z[i][j]))
    #         if 0.5 > x[i][j] > -0.5:
    #             if x[i][j] < 0:
    #                 dy9[i].append(-1 * (2.1-z[i][j]) * (abs(x[i][j]) ** (1.1-z[i][j])) / (0.8 + abs(x[i][j]) ** (2.1-z[i][j])))
    #             else:
    #                 dy9[i].append(1 * (2.1-z[i][j]) * (abs(x[i][j]) ** (1.1-z[i][j])) / (0.8 + abs(x[i][j]) ** (2.1-z[i][j])))
    #         else:
    #             dy9[i].append(2 * A * x[i][j])

    # Sigmoid
    sigmoid = []
    for xx in x:
        sigmoid.append(1/(1+math.e**(-xx)))

    # tanh
    tanh = []
    for xx in x:
        tanh.append((math.e**xx - math.e**(-xx))/(math.e**xx + math.e**(-xx)))

    # ReLu
    relu = []
    for xx in x:
        relu.append(max(0, xx))

    # Leaky_ReLu
    leaky_relu = []
    for xx in x:
        leaky_relu.append(max(0.1*xx, xx))


    # plt.plot(x, y1, label="l1")
    # plt.plot(x, y2, label="l2")
    # plt.plot(x, y3, label="smooth l1")
    # plt.plot(x, y4, label="wing loss")
    # plt.plot(x, y5, label="Awing loss y=1")
    # plt.plot(x, y6, label="Awing loss y=0")
    # plt.plot(x, y7, label="my loss y=0")
    # plt.plot(x, y8, label="my loss y=1")
    # plt.plot(x, dy1, label="l1'")
    # plt.plot(x, dy2, label="l2'")
    # plt.plot(x, dy3, label="Smooth l1'")
    # plt.plot(x, dy4, label="wing loss'")
    # plt.plot(x, dy5, label="Awing loss y=1'")
    # plt.plot(x, dy6, label="Awing loss y=0'")
    # plt.plot(x, dy7, label="my loss y=0'")
    # plt.plot(x, dy8, label="my loss y=1'")
    # plt.plot(x, sigmoid, label="Sigmoid")
    # plt.plot(x, tanh, label="Tanh")
    # plt.plot(x, relu, label="ReLu")
    plt.plot(x, leaky_relu, label="Leaky_ReLu")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # 3D梯度图
    # dy9 = np.array(dy9)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf = ax.plot_surface(z, x, dy9, rstride=1, cstride=1, cmap=cm.viridis)
    # ax.set_xlabel("y")
    # ax.set_ylabel("loss")
    # ax.set_zlabel("gradient")
    # plt.show()



def L2(y_gt, y_pre):
    loss = np.sum(np.square(y_gt - y_pre))
    print(loss)

    # 画图
    x = np.arange(-10, 11, 0.1)
    y = (1 / 2) * (x) ** 2
    dy = x
    plt.plot(x, y, label="l2")
    plt.plot(x, dy, label="l2'")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def smooth_l1(y_gt, y_pre):
    select = y_gt - y_pre
    loss = np.where(abs(select) < 1, 0.5*select**2, abs(select)-0.5)
    loss = np.sum(loss)

    # 画图
    x = np.arange(-10,11,0.1)
    y = []
    for xx in x:
        if xx > -1 and xx < 1:
            y.append(0.5*xx**2)
        else:
            y.append(abs(xx)-0.5)
    dy = []
    for xx in x:
        if xx > -1 and xx < 1:
            dy.append(xx)
        elif xx <= -1:
            dy.append(-1)
        else:
            dy.append(1)
    plt.plot(x, y, label="smooth l1")
    plt.plot(x, dy, label="smooth l1'")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # pts = [[103.7935268278302, 217.71975088443392], [120.50095754716979, 300.47031721698113], [184.28967275943396,
    #        391.83056692216985], [290.5383210495283, 316.20828714622644], [317.4345875589623, 217.3147137382076],
    #        [171.9472102004717, 219.6987081367925], [167.29825913915093, 277.9954333726415], [153.3024764150943,
    #        281.27914917452836], [172.89341214622644, 287.7021960495283], [199.76942865566036, 281.48627653301884],
    #        [124.45312234669812, 219.66688708726414], [135.6837612028302, 214.98715772405663], [160.26188826650946,
    #        221.08819457547176], [135.36744251179246, 224.28724616745276], [204.23525678066036, 219.26623584905656],
    #        [222.83161821933967, 212.62871462264158], [245.0150368514151, 217.31974174528298], [223.88247700471697,
    #        223.68511645047172], [143.83427741745285, 300.9560784198114], [173.76964652122643, 298.8621907429245],
    #        [239.53215566037736, 301.1298458136793], [180.6826574292453, 351.46261114386795], [173.88660406839625,
    #        303.1080389150943], [181.02590978773586, 340.7182827240566], [136.39680424528302, 221.3427629716981],
    #        [223.95092954009434, 219.62601562500004]]
    # heatmap = torch.ones((26, 64, 64))
    # gt_heatmap = torch.zeros((26, 64, 64))
    # for i in range(26):
    #     heatmap[i] = draw_gaussian(heatmap[i], pts[i], 1)
    #     x = pts[i][0]-1
    #     y = pts[i][1] -1
    #     gt_heatmap[i] = draw_gaussian(gt_heatmap[i], [x, y], 1)
    # # loss_func = WingLoss()
    # # heatmap = torch.ones(2, 68, 64, 64)
    # # gt_heatmap = torch.zeros(2, 68, 64, 64)
    # heatmap.requires_grad_(True)
    # # loss = loss_func(gt_heatmap, heatmap)
    # loss = my_loss(gt_heatmap, heatmap)
    # loss.backward()
    # print(loss)
    if __name__ == "__main__":
        y_gt = np.arange(1, 11)
        y_pre = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 13.13])
        L1(y_gt, y_pre)
        # L2(y_gt, y_pre)
        # smooth_l1(y_gt, y_pre)


