# 图片生成热图

import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.model_utils import get_max_preds
import torch
import math


# 生成热图
def HeatMap(img_width, img_height, c_x, c_y, sigma):
    # 从1到img_width均匀生成img_width个数据
    X1 = np.linspace(1, img_width, img_width)
    # 同上
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


def CenterLabelHeatMap(landmark, img_width, img_height):
    heatmap = []
    pts = []
    w = img_width // 2
    h = img_height // 2
    for i in range(len(landmark)):
        if i % 2 == 0:
            pts.append(landmark[i] * w)
        else:
            pts.append(landmark[i] * h)
    for i in range(len(landmark)//2):
        heatmap.append(HeatMap(w, h, int(pts[i*2]), int(pts[i*2+1]), 3))
    heatmap = np.stack(heatmap)
    heatmap = torch.tensor(heatmap)  # 报错
    return heatmap


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                    sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [np.floor(np.floor(point[0]) - 3 * sigma), np.floor(np.floor(point[1]) - 3 * sigma)]
    br = [np.floor(np.floor(point[0]) + 3 * sigma), np.floor(np.floor(point[1]) + 3 * sigma)]
    if ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1:
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    correct = False
    while not correct:
        try:
            image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] \
                                                                  + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
            correct = True
        except:
            print('img_x: {}, img_y: {}, g_x:{}, g_y:{}, point:{}, g_shape:{}, ul:{}, br:{}'.format(img_x, img_y, g_x,
                                                                                                    g_y, point, g.shape,
                                                                                                    ul, br))
            ul = [np.floor(np.floor(point[0]) - 3 * sigma), np.floor(np.floor(point[1]) - 3 * sigma)]
            br = [np.floor(np.floor(point[0]) + 3 * sigma), np.floor(np.floor(point[1]) + 3 * sigma)]
            g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
            g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
            img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
            img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
            pass
    image[image > 1] = 1
    return image


if __name__ == '__main__':
    landmarks = [103.44966420990566, 189.20597847877366, 103.50245695754717, 203.4644513561321, 103.7935268278302,
                 217.71975088443392, 104.49805277122643, 231.96033785377358, 105.63194663915097, 246.17338089622643,
                 107.04600530660377, 260.36143779481137, 109.24619221698114, 274.44348997641515, 113.65915860849056,
                 287.9712859669812, 120.50095754716979, 300.47031721698113, 128.05666450471696, 312.5621780660378,
                 135.0552470518868, 324.9822921580189, 141.36019280660378, 337.77037146226417, 147.5807741745283,
                 350.600319870283, 154.25147228773582, 363.200134139151, 161.95323290094342, 375.18965713443396,
                 171.59380630896226, 385.6464658018868, 184.28967275943396, 391.83056692216985, 201.37664504716986,
                 391.6737514740566, 217.80954392688682, 386.6080091391509, 233.25919015330192, 379.0449498820754,
                 247.46413531839622, 369.3403867924528, 260.2770044221698, 357.8561055424528, 271.6800627948113,
                 344.9673652712264, 281.7377644457547, 331.0007051886793, 290.5383210495283, 316.20828714622644,
                 298.1000005896227, 300.74551768867923, 304.40403626179244, 284.72857163915097, 309.44177034198117,
                 268.2691928066038, 313.2004245283019, 251.47135613207553, 315.77515949292456, 234.4504988207547,
                 317.4345875589623, 217.3147137382076, 318.4880200471698, 200.130118514151, 319.23995784198115,
                 182.9292452830189, 109.29479540094339, 198.06095430424529, 121.57159935141509, 191.01109669811322,
                 133.246172759434, 191.39595813679253, 144.74558048349056, 193.61411320754715, 155.38791863207544,
                 196.29130277122638, 155.8390170990566, 205.65745577830197, 144.72432753537734, 203.2083652712264,
                 132.98082665094338, 200.1172473466981, 121.3753213443396, 197.8460527712265, 196.74231898584904,
                 195.90125943396228, 213.10784257075474, 191.15967895047166, 229.73611998820758, 188.4984436910378,
                 245.17835819575473, 189.34751886792455, 259.8192514740566, 197.69479599056615, 245.33754834905668,
                 197.16677564858483, 229.78919929245285, 198.1696998820754, 213.2591391509434, 201.13077682783012,
                 196.69721816037733, 206.21195341981135, 171.9472102004717, 219.6987081367925, 172.03881279481132,
                 239.51134316037732, 172.37320577830187, 259.3109955778301, 167.29825913915093, 277.9954333726415,
                 153.3024764150943, 281.27914917452836, 162.7809914504717, 285.52481957547167, 172.89341214622644,
                 287.7021960495283, 186.60114711084907, 285.877783018868, 199.76942865566036, 281.48627653301884,
                 124.45312234669812, 219.66688708726414, 129.88282340801885, 216.86510112028296, 135.6837612028302,
                 214.98715772405663, 148.54299941037738, 215.5868835495283, 160.26188826650946, 221.08819457547176,
                 148.0113944575472, 224.56015418632077, 135.36744251179246, 224.28724616745276, 129.7667998231132,
                 222.3256441627358, 204.23525678066036, 219.26623584905656, 212.82468484669815, 213.90172110849062,
                 222.83161821933967, 212.62871462264158, 234.0612116745283, 214.28940300707555, 245.0150368514151,
                 217.31974174528298, 234.85924528301885, 221.91512529481128, 223.88247700471697, 223.68511645047172,
                 213.84265860849058, 222.48534905660378, 143.83427741745285, 300.9560784198114, 154.09018337264155,
                 298.7424153891509, 164.54976208726413, 298.2063767688679, 173.76964652122643, 298.8621907429245,
                 184.92514917452831, 298.57371845518867, 212.3051471108491, 297.71438826650945, 239.53215566037736,
                 301.1298458136793, 227.7658098466981, 326.80519280660377, 207.98239829009438, 346.58994958726413,
                 180.6826574292453, 351.46261114386795, 160.79333136792457, 341.6201771816038, 148.71779363207546,
                 322.8766397405661, 145.71137971698116, 302.58119398584904, 159.79154274764153, 303.4225427476415,
                 173.88660406839625, 303.1080389150943, 205.5132275943396, 301.5148478773584, 237.202500884434,
                 301.9296093750001, 215.65859699292454, 331.6678752948113, 181.02590978773586, 340.7182827240566,
                 158.02571727594338, 326.8792119693396, 136.39680424528302, 221.3427629716981, 223.95092954009434,
                 219.62601562500004]
    # d = []
    # pts = []
    # heatmap = []
    # for n in landmarks:
    #     n = n / 450 * 256
    #     pts.append(n)
    # # for n in pts:
    # #     print(n)
    # for i in range(len(landmarks)//2):
    #     d.append(CenterLabelHeatMap(landmarks, 256, 256))
    # heatmap.append(d)
    # heatmap = np.stack(heatmap)
    # heatmap = torch.tensor(heatmap)
    # # preds, maxvals = get_max_preds(heatmap)
    # for i in range(len(landmarks)//2):
    #     cv2.imshow('d', d[i])
    #     cv2.waitKey(0)
    pts = []
    for n in range(len(landmarks)//2):
        pt = [landmarks[n*2] / 450 * 512+1, landmarks[n*2+1] / 450 * 512+1]
        pts.append(pt)
    heatmap = np.zeros((98, 512, 512))
    for i in range(98):
        if pts[i][0] > 0:
            heatmap[i] = draw_gaussian(heatmap[i], pts[i], 10)
    for i in range(98):
        cv2.imshow('d', heatmap[i])
        cv2.waitKey(0)



