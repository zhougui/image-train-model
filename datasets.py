# 数据集处理
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import os
import copy
import torch
import cv2
import numpy as np
from utils.model_utils import get_final_preds
from utils.model_utils import get_preds_fromhm
import random
from data_iter.data_agu import *
from data_iter.heatmap_label import *

# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1 - c, b)
    return dst


class LoadLabels68(Dataset):
    def __init__(self, ops, img_size=(256, 256), model="train", flag_agu=False, fix_res=True, vis=False):
        # print('img_size (height,width) : ', img_size[0], img_size[1])
        if model == "train":
            r_ = open(ops.train_list, 'r')
        else:
            r_ = open(ops.test_list, 'r')
        # print(ops.train_list)
        lines = r_.readlines()
        idx = 0
        file_list = []
        landmarks_list = []
        for line in lines:
            # print(line)
            msg = line.strip().split(' ')
            idx += 1
            landmarks = msg[0:136]
            img_file = msg[136]
            pts = []
            for i in range(int(len(landmarks) / 2)):
                x = float(landmarks[i * 2 + 0])
                y = float(landmarks[i * 2 + 1])
                pts.append([x, y])
            landmarks_list.append(pts)
            file_list.append(ops.images_path + img_file)
        self.files = file_list
        self.landmarks = landmarks_list
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.fix_res = fix_res
        self.vis = vis
        self.model = model

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        if self.model == "train":
            if self.flag_agu is True:
                left_eye = np.average(pts[36:42], axis=0)
                right_eye = np.average(pts[42:48], axis=0)

                angle_random = random.randint(-36, 36)
                # 返回 crop 图 和 归一化 landmarks
                img_, landmarks_ = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
                                                      fix_res=self.fix_res, img_size=self.img_size, vis=False)
                if random.random() > 0.5:
                    c = float(random.randint(50, 150)) / 100.
                    b = random.randint(-20, 20)
                    img_ = contrast_img(img_, c, b)
                if random.random() > 0.7:
                    img_hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
                    hue_x = random.randint(-10, 10)
                    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_x)
                    img_hsv[:, :, 0] = np.maximum(img_hsv[:, :, 0], 0)
                    img_hsv[:, :, 0] = np.minimum(img_hsv[:, :, 0], 180)  # 范围 0 ~180
                    img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                if random.random() > 0.8:
                    img_ = img_agu_channel_same(img_)
            if self.vis is True:
                cv2.namedWindow('crop', 0)
                cv2.imshow('crop', img_)
                cv2.waitKey(0)
        else:
            img_, landmarks_ = landmarks_nom(img, pts)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(landmarks_)

        heatmap = np.zeros((68, 64, 64))
        for i in range(68):
            if landmarks_[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], landmarks_[i] / 4.0 + 1, 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap

    def function(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        if self.model == "train":
            if self.flag_agu is True:
                left_eye = np.average(pts[36:42], axis=0)
                right_eye = np.average(pts[42:48], axis=0)
                angle_random = random.randint(-36, 36)
                # 返回 crop 图 和 归一化 landmarks
                img_, landmarks_ = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
                                                      fix_res=self.fix_res, img_size=self.img_size, vis=False)
                if random.random() > 0.5:
                    c = float(random.randint(50, 150)) / 100.
                    b = random.randint(-20, 20)
                    img_ = contrast_img(img_, c, b)
                if random.random() > 0.7:
                    img_hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
                    hue_x = random.randint(-10, 10)
                    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_x)
                    img_hsv[:, :, 0] = np.maximum(img_hsv[:, :, 0], 0)
                    img_hsv[:, :, 0] = np.minimum(img_hsv[:, :, 0], 180)  # 范围 0 ~180
                    img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                if random.random() > 0.8:
                    img_ = img_agu_channel_same(img_)
            if self.vis is True:
                cv2.namedWindow('crop', 0)
                cv2.imshow('crop', img_)
                cv2.waitKey(0)
        else:
            img_, landmarks_ = landmarks_nom(img, pts)
        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(landmarks_)
        heatmap = np.zeros((68, 64, 64))
        for i in range(68):
            if landmarks_[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], landmarks_[i] / 4.0 + 1, 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap


class LoadLabels98(Dataset):
    def __init__(self, ops, img_size=(256, 256), model="train", flag_agu=False, fix_res=True, vis=False):
        if model == "train":
            r_ = open(ops.train_list, 'r')
        elif model == "test":
            r_ = open(ops.test_list, 'r')
        else:
            return
        lines = r_.readlines()
        idx = 0
        file_list = []
        landmarks_list = []
        for line in lines:
            # print(line)
            msg = line.strip().split(' ')
            idx += 1
            # print('idx-', idx, ' : ', len(msg))
            landmarks = msg[0:196]
            if model == "train":
                bbox = msg[196:200]
                attributes = msg[200:206]
                img_file = msg[206]
            else:
                img_file = msg[206]
            # print(img_file)
            pts = []
            for i in range(int(len(landmarks) / 2)):
                x = float(landmarks[i * 2 + 0])
                y = float(landmarks[i * 2 + 1])
                pts.append([x, y])

            landmarks_list.append(pts)
            file_list.append(ops.images_path + img_file)

        self.files = file_list
        self.landmarks = landmarks_list
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.fix_res = fix_res
        self.vis = vis
        self.model = model
        self.center_shift = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        if self.model == "train":
            if self.flag_agu:
                left_eye = np.average(pts[60:68], axis=0)
                right_eye = np.average(pts[68:76], axis=0)
                angle_random = random.randint(-36, 36)
                # 返回 crop 图 和 归一化 landmarks
                img_, landmarks_ = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
                                                      fix_res=self.fix_res, img_size=self.img_size, vis=False)
                if random.random() > 0.5:
                    c = float(random.randint(50, 150)) / 100.
                    b = random.randint(-20, 20)
                    img_ = contrast_img(img_, c, b)
                if random.random() > 0.7:
                    img_hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
                    hue_x = random.randint(-10, 10)
                    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_x)
                    img_hsv[:, :, 0] = np.maximum(img_hsv[:, :, 0], 0)
                    img_hsv[:, :, 0] = np.minimum(img_hsv[:, :, 0], 180)  # 范围 0 ~180
                    img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                if random.random() > 0.8:
                    img_ = img_agu_channel_same(img_)
        elif self.model == "test":
            img_, landmarks_ = landmarks_nom(img, pts)

        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(landmarks_)
        pts = []
        for n in range(len(landmarks_)):
            pts.append(landmarks_[n]/4+1)
        heatmap = np.zeros((98, 64, 64))
        for i in range(98):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], pts[i], 1)
        heatmap = torch.tensor(heatmap)
        landmarks_ = torch.tensor(landmarks_)
        return img_, landmarks_, heatmap

    def function(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        if self.model == "train":
            if self.flag_agu:
                left_eye = np.average(pts[60:68], axis=0)
                right_eye = np.average(pts[68:76], axis=0)
                angle_random = random.randint(-36, 36)
                # 返回 crop 图 和 归一化 landmarks
                img_, landmarks_ = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
                                                      fix_res=self.fix_res, img_size=self.img_size, vis=False)
                if random.random() > 0.5:
                    c = float(random.randint(50, 150)) / 100.
                    b = random.randint(-20, 20)
                    img_ = contrast_img(img_, c, b)
                if random.random() > 0.7:
                    img_hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
                    hue_x = random.randint(-10, 10)
                    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_x)
                    img_hsv[:, :, 0] = np.maximum(img_hsv[:, :, 0], 0)
                    img_hsv[:, :, 0] = np.minimum(img_hsv[:, :, 0], 180)  # 范围 0 ~180
                    img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                if random.random() > 0.8:
                    img_ = img_agu_channel_same(img_)
        elif self.model == "test":
            img_, landmarks_ = landmarks_nom(img, pts)

        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(landmarks_)
        pts = []
        for n in range(len(landmarks_)):
            pts.append(landmarks_[n] / 4 + 1)
        heatmap = np.zeros((98, 64, 64))
        for i in range(98):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], pts[i], 1)
        heatmap = torch.tensor(heatmap)
        landmarks_ = torch.tensor(landmarks_)
        return img_, landmarks_, heatmap


class LoadLabels32(Dataset):
    def __init__(self, ops, img_size=(256, 256), model="train", flag_agu=True, fix_res=True, vis=False):
        # print('img_size (height,width) : ', img_size[0], img_size[1])
        r_ = open(ops.train_list, 'r')
        lines = r_.readlines()
        idx = 0
        file_list = []
        landmarks_list = []
        for line in lines:
            msg = line.strip().split(' ')
            idx += 1
            landmarks = msg[0:64]
            img_file = msg[-1]
            # print(img_file)
            pts = []
            for i in range(int(len(landmarks) / 2)):
                x = float(landmarks[i * 2 + 0])
                y = float(landmarks[i * 2 + 1])
                pts.append([x, y])

            landmarks_list.append(pts)
            print(ops.images_path + img_file)
            file_list.append(ops.images_path + img_file)

        self.files = file_list
        self.landmarks = landmarks_list
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.fix_res = fix_res
        self.vis = vis

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        img, landmarks_ = landmarks_nom(img, pts)
        img_ = img.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        heatmap = np.zeros((32, 128, 128))
        landmarks = []
        for i in range(32):
            landmarks.append([landmarks_[i][0] / 2.0 + 1, landmarks_[i][1] / 2.0 + 1])
        for i in range(32):
            if landmarks[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], landmarks[i], 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap

    def function(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        img, landmarks_ = landmarks_nom(img, pts)
        img_ = img.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        heatmap = np.zeros((32, 128, 128))
        landmarks = []
        for i in range(32):
            landmarks.append([landmarks_[i][0] / 2.0 + 1 , landmarks_[i][1] / 2.0 + 1])
        for i in range(32):
            if landmarks[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], landmarks[i], 1)
        heatmap = torch.tensor(heatmap)
        # img = cv2.resize(img, (256, 256))
        return img, landmarks_, heatmap


class LoadLabels(Dataset):
    def __init__(self, ops, img_size=(256, 256), model="train"):
        if model == "train":
            r_ = open(ops.train_list, 'r')
        elif model == "test":
            r_ = open(ops.test_list, 'r')
        else:
            return
        lines = r_.readlines()
        idx = 0
        file_list = []
        landmarks_list = []
        for line in lines:
            # print(line)
            msg = line.strip().split(' ')
            idx += 1
            # print('idx-', idx, ' : ', len(msg))
            landmarks = msg[0:64]
            img_file = msg[-1]
            # print(img_file)
            pts = []
            for i in range(int(len(landmarks) / 2)):
                x = float(landmarks[i * 2 + 0])
                y = float(landmarks[i * 2 + 1])
                pts.append([x, y])

            landmarks_list.append(pts)
            file_list.append(ops.images_path + img_file)

        self.files = file_list
        self.landmarks = landmarks_list
        self.img_size = img_size
        self.model = model

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        img_ = img.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(pts)
        pts = []
        for n in range(len(landmarks_)):
            pts.append(landmarks_[n]/2+1)
        heatmap = np.zeros((32, 128, 128))
        for i in range(32):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], pts[i], 1)
        heatmap = torch.tensor(heatmap)
        landmarks_ = torch.tensor(landmarks_)
        return img_, landmarks_, heatmap

    def function(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        # img_ = img.astype(np.float32)
        # img_ = (img_ - 128.) / 256.
        img_ = img.transpose(2, 0, 1)
        landmarks_ = np.array(pts)
        pts = []
        for n in range(len(landmarks_)):
            pts.append(landmarks_[n] / 4 + 1)
        heatmap = np.zeros((32, 64, 64))
        for i in range(32):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], pts[i], 1)
        heatmap = torch.tensor(heatmap)
        landmarks_ = torch.tensor(landmarks_)
        return img_, landmarks_, heatmap

    def function128(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        img_ = img.astype(np.float32)
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(pts)
        pts = []
        for n in range(len(landmarks_)):
            pts.append(landmarks_[n] / 2 + 1)
        heatmap = np.zeros((32, 128, 128))
        for i in range(32):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], pts[i], 1)
        heatmap = torch.tensor(heatmap)
        landmarks_ = torch.tensor(landmarks_)
        return img_, landmarks_, heatmap


if __name__ == '__main__':
    # 300W
    # images_path = "../utils/300W/"
    # test_list = "../utils/300W/list_68pt_Fullset.txt"
    # # # test_list = "../utils/300W/list_68pt_Common Subset.txt"
    # # # test_list = "../utils/300W/list_68pt_Challenging Subset.txt"
    # train_list = "../utils/300W/list_68pt_train.txt"
    # 98
    # images_path = "../../WFLW_train/WFLW_images/"
    # # images_path = "../utils/WFLW_train/train_images/"
    # # train_list = "../../WFLW_train/landmarks/list_98pt_train.txt"
    # train_list = "../../WFLW_train/landmarks/list_98pt_train_all.txt"
    # # test_list = "../../WFLW_train/landmarks/list_98pt_test.txt"
    # test_list = "../../WFLW_train/landmarks/list_98pt_test.txt"
    # # 34
    # train_list = '../../mydataset34/labels/train34ce.txt'
    # images_path = '../../mydataset34/images/ce/'
    # 32
    train_list = '../../dataset32/labels/list32_train.txt'
    images_path = '../../dataset32/train images/'
    test_list = '../../dataset32/labels/list32_xiao.txt'
    parser = argparse.ArgumentParser(description=' Project facial landmarks Train')
    parser.add_argument('--seed', type=int, default=32, help='seed')  # 设置随机种子
    parser.add_argument('--model_exp', type=str, default='./model_exp', help='model_exp')  # 模型输出文件夹
    parser.add_argument('--model', type=str, default='HR-net', help='model : U-net')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=196, help='num_classes')  # landmarks 个数*2
    parser.add_argument('--GPUS', type=str, default='0', help='GPUS')  # GPU选择
    parser.add_argument('--images_path', type=str, default=images_path, help='images_path')  # 图片路径
    parser.add_argument('--train_list', type=str,
                        default=train_list,
                        help='annotations_train_list')  # 训练集标注信息
    parser.add_argument('--test_list', type=str, default=test_list, help='annotations_train_list')  # 训练集标注信息
    parser.add_argument('--pretrained', type=bool, default=True, help='imageNet_Pretrain')  # 预训练
    parser.add_argument('--fintune_model', type=str, default='./model_exp/2021-02-21_17-51-30/resnet_50-epoch-724.pth',
                        help='fintune_model')  # 预训练模型
    parser.add_argument('--loss_define', type=str, default='wing_loss', help='define_loss')  # 损失函数定义
    parser.add_argument('--init_lr', type=float, default=1e-3, help='init_learningRate')  # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learningRate_decay')  # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight_decay')  # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 优化器动量
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')  # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')  # dropout
    parser.add_argument('--epochs', type=int, default=1000, help='epochs')  # 训练周期
    parser.add_argument('--num_workers', type=int, default=2, help='num_workers')  # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=True, help='data_augmentation')  # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool, default=False, help='fix_resolution')  # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default=False, help='clear_model_exp')  # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default=False, help='log flag')  # 是否保存训练 log

    args = parser.parse_args()  # 解析添加参数

    dataset = LoadLabels(ops=args, img_size=args.img_size, model="test")
    # dataset = LoadLabels32(ops=args, img_size=args.img_size, model="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            pin_memory=False, drop_last=True)
    # ans = 0
    # for i, (images, landmarks_, heatmap) in enumerate(dataloader):
    #     for i in range(images.shape[0]):
    #         print(heatmap.shape)
    #         gt_heatmap = heatmap[i]
    #         print(gt_heatmap.shape)
    #         pred_landmarks, _ = get_preds_fromhm(gt_heatmap.unsqueeze(0))
    #         img = images[i].numpy()
    #         img = img.transpose(1, 2, 0)
    #         cv2.imshow("dd", img)
    #         cv2.waitKey(-1)
            # img = np.ascontiguousarray(img, dtype=np.uint8)
            # for j in range(98):
            #     img = cv2.circle(img, (int(float(pred_landmarks[0][j][0] * 4)), int(float(pred_landmarks[0][j][1] * 4))), 2,
            #                      (255, 0, 255), -1)
            # cv2.imshow("dd", img)
            # cv2.waitKey(-1)
    nme64 = 0
    nme128 = 0
    for i in range(len(dataset)):
        images, landmarks_, heatmap = dataset.function(i)
        _, landmarks_128, heatmap128 = dataset.function128(i)
        # print(i)
        images = images.transpose(1, 2, 0)
        heatmap = torch.unsqueeze(heatmap, 0)
        heatmap128 = torch.unsqueeze(heatmap128, 0)
        pre_, _ = get_preds_fromhm(heatmap)
        pre_128, _ = get_preds_fromhm(heatmap128)
        images = cv2.resize(images, (512, 512))
        for j in range(32):
            images = cv2.circle(images, (int(pre_[0][j][0] * 8), int(pre_[0][j][1] * 8)), 2, (0, 0, 255), -1)
            images = cv2.circle(images, (int(pre_128[0][j][0] * 4), int(pre_128[0][j][1] * 4)), 2, (0, 255, 0), -1)
            images = cv2.circle(images, (int(landmarks_[j][0]*2), int(landmarks_[j][1]*2)), 2, (255, 0, 0), -1)
        # norm_factor = np.linalg.norm(landmarks_[5] - landmarks_[6])
        # single_nme = (np.sum(np.linalg.norm(pre_[0]*4 - landmarks_, axis=1)) / pre_128.shape[1]) / norm_factor  # 64x64
        # single_nme1 = (np.sum(np.linalg.norm(pre_128[0]*2 - landmarks_, axis=1)) / pre_128.shape[1]) / norm_factor  # 128x128
        # nme64 += single_nme
        # nme128 += single_nme1
        cv2.imshow("aa", images)
        cv2.waitKey(-1)
        print(i)
    print(nme64/50)
    print(nme128/50)

