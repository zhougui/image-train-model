from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import os
import copy
import torch
import cv2
#from data_iter import transforms
import numpy as np

from utils.model_utils import get_preds_fromhm
from data_iter.data_agu import *
from data_iter.heatmap_label import *

class MyLoadLabels20(Dataset):
    def __init__(self, ops, img_size=(256, 256), model="train"):
        # print('img_size (height,width) : ', img_size[0], img_size[1])
        r_ = open(ops.train_list, 'r')
        lines = r_.readlines()
        idx = 0
        file_list = []
        landmarks_list = []
        for line in lines:
            # print(line)
            msg = line.strip().split(' ')
            idx += 1
            landmarks = msg[0:40]
            img_file = msg[40]
            # print(img_file)
            pts = []
            global_dict_landmarks = {}  # 全局坐标系坐标
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
        self.center_shift = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        center = [450 // 2, 450 // 2 + 0]
        if self.center_shift != 0:
            center[0] += int(np.random.uniform(-self.center_shift, self.center_shift))
            center[1] += int(np.random.uniform(-self.center_shift, self.center_shift))
        scale = 1.8
        if pts:
            new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450 // 2, 450 // 2 + 0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift, self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift, self.center_shift))

                new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450 // 2, 450 // 2 + 0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, 100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), "Landmarks out of boundary!"
            img_ = new_image
            landmarks_ = new_landmarks
        img_ = img_.transpose(2, 0, 1)
        heatmap = np.zeros((20, 64, 64))
        for i in range(20):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], pts[i], 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap

    def function(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        center = [450 // 2, 450 // 2 + 0]
        if self.center_shift != 0:
            center[0] += int(np.random.uniform(-self.center_shift, self.center_shift))
            center[1] += int(np.random.uniform(-self.center_shift, self.center_shift))
        scale = 1.8
        if pts:
            new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450 // 2, 450 // 2 + 0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift, self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift, self.center_shift))

                new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450 // 2, 450 // 2 + 0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, 100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), "Landmarks out of boundary!"
            img_ = new_image
            landmarks_ = new_landmarks
        img_ = img_.transpose(2, 0, 1)
        heatmap = np.zeros((20, 64, 64))
        for i in range(20):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], landmarks_[i] / 4.0 + 1, 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap


class MyLoadLabels34(Dataset):
    def __init__(self, ops, img_size=(256, 256), model="train"):
        # print('img_size (height,width) : ', img_size[0], img_size[1])
        r_ = open(ops.train_list, 'r')
        lines = r_.readlines()
        idx = 0
        file_list = []
        landmarks_list = []
        for line in lines:
            # print(line)
            msg = line.strip().split(' ')
            idx += 1
            landmarks = msg[0:68]
            img_file = msg[68]
            # print(img_file)
            pts = []
            global_dict_landmarks = {}  # 全局坐标系坐标
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
        self.center_shift = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        center = [450 // 2, 450 // 2 + 0]
        if self.center_shift != 0:
            center[0] += int(np.random.uniform(-self.center_shift, self.center_shift))
            center[1] += int(np.random.uniform(-self.center_shift, self.center_shift))
        scale = 1.8
        if pts:
            new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450 // 2, 450 // 2 + 0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift, self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift, self.center_shift))

                new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450 // 2, 450 // 2 + 0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, 100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), "Landmarks out of boundary!"
            img_ = new_image
            landmarks_ = new_landmarks
        img_ = img_.transpose(2, 0, 1)
        heatmap = np.zeros((34, 64, 64))
        for i in range(34):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], pts[i], 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap

    def function(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img = cv2.imread(img_path)  # BGR
        center = [450 // 2, 450 // 2 + 0]
        if self.center_shift != 0:
            center[0] += int(np.random.uniform(-self.center_shift, self.center_shift))
            center[1] += int(np.random.uniform(-self.center_shift, self.center_shift))
        scale = 1.8
        if pts:
            new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450 // 2, 450 // 2 + 0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift, self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift, self.center_shift))

                new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450 // 2, 450 // 2 + 0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(img, pts, center, scale, 256, 100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), "Landmarks out of boundary!"
            img_ = new_image
            landmarks_ = new_landmarks
        img_ = img_.transpose(2, 0, 1)
        heatmap = np.zeros((34, 64, 64))
        for i in range(34):
            if pts[i][0] > 0:
                heatmap[i] = draw_gaussian(heatmap[i], landmarks_[i] / 4.0 + 1, 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap


class MyLoadLabels68(Dataset):
    def __init__(self, ops, img_size=(256, 256), model="train"):
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
            print(len(msg))
            idx += 1
            landmarks = msg[0:136]
            img_file = msg[136]
            # print(img_file)
            pts = []
            global_dict_landmarks = {}  # 全局坐标系坐标
            for i in range(int(len(landmarks) / 2)):
                x = float(landmarks[i * 2 + 0])
                y = float(landmarks[i * 2 + 1])
                pts.append([x, y])

            landmarks_list.append(pts)
            file_list.append(ops.images_path + img_file)

        self.files = file_list
        self.landmarks = landmarks_list
        self.img_size = img_size
        self.center_shift = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        # print(img_path)
        img_ = cv2.imread(img_path)  # BGR
        center = [450 // 2, 450 // 2 + 0]
        if self.center_shift != 0:
            center[0] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
            center[1] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
        scale = 1.8
        # img_, landmarks_ = landmarks_nom(img, pts)
        if pts:
            new_image, new_landmarks = cv_crop(img_, pts, center,
                                               scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (
                    np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450 // 2, 450 // 2 + 0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift,
                                                   self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift,
                                                   self.center_shift))

                new_image, new_landmarks = cv_crop(img_, pts,
                                                   center, scale, 256,
                                                   self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450 // 2, 450 // 2 + 0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(img_, pts,
                                                   center, scale, 256,
                                                   100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), \
                "Landmarks out of boundary!"
            img_ = new_image
            landmarks_ = new_landmarks
        img_ = img_.transpose(2, 0, 1)
        heatmap = np.zeros((68, 64, 64))
        for i in range(68):
            if landmarks_[i][0] > 0:
                # print(pts[i])
                heatmap[i] = draw_gaussian(heatmap[i], landmarks_[i] / 4.0 + 1, 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap

    def function(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        print(img_path)
        img_ = cv2.imread(img_path)  # BGR
        center = [450 // 2, 450 // 2 + 0]
        if self.center_shift != 0:
            center[0] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
            center[1] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
        scale = 1.8
        # img_, landmarks_ = landmarks_nom(img, pts)
        if pts:
            new_image, new_landmarks = cv_crop(img_, pts, center,
                                               scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (
                    np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450 // 2, 450 // 2 + 0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift,
                                                   self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift,
                                                   self.center_shift))

                new_image, new_landmarks = cv_crop(img_, pts,
                                                   center, scale, 256,
                                                   self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450 // 2, 450 // 2 + 0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(img_, pts,
                                                   center, scale, 256,
                                                   100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), \
                "Landmarks out of boundary!"
            img_ = new_image
            landmarks_ = new_landmarks
        img_ = img_.transpose(2, 0, 1)
        heatmap = np.zeros((68, 64, 64))
        for i in range(68):
            if landmarks_[i][0] > 0:
                # print(pts[i])
                heatmap[i] = draw_gaussian(heatmap[i], landmarks_[i] / 4.0 + 1, 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap


class MyLoadLabels98(Dataset):
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
            landmarks = msg[0:196]
            bbox = msg[196:200]
            attributes = msg[200:206]
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
        self.model = model
        self.center_shift = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img_ = cv2.imread(img_path)  # BGR
        center = [450 // 2, 450 // 2 + 0]
        if self.center_shift != 0:
            center[0] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
            center[1] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
        scale = 1.8
        # img_, landmarks_ = landmarks_nom(img, pts)
        if pts:
            new_image, new_landmarks = cv_crop(img_, pts, center,
                                               scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (
                    np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450 // 2, 450 // 2 + 0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift,
                                                   self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift,
                                                   self.center_shift))

                new_image, new_landmarks = cv_crop(img_, pts,
                                                   center, scale, 256,
                                                   self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450 // 2, 450 // 2 + 0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(img_, pts,
                                                   center, scale, 256,
                                                   100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), \
                "Landmarks out of boundary!"
            img_ = new_image
            landmarks_ = new_landmarks
        img_ = img_.transpose(2, 0, 1)

        heatmap = np.zeros((98, 64, 64))
        for i in range(98):
            if landmarks_[i][0] > 0:
                # print(pts[i])
                heatmap[i] = draw_gaussian(heatmap[i], landmarks_[i] / 4.0 + 1, 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap

    def function(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        img_ = cv2.imread(img_path)  # BGR
        center = [450 // 2, 450 // 2 + 0]
        if self.center_shift != 0:
            center[0] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
            center[1] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
        scale = 1.8
        # img_, landmarks_ = landmarks_nom(img, pts)
        if pts:
            new_image, new_landmarks = cv_crop(img_, pts, center,
                                               scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (
                    np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450 // 2, 450 // 2 + 0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift,
                                                   self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift,
                                                   self.center_shift))

                new_image, new_landmarks = cv_crop(img_, pts,
                                                   center, scale, 256,
                                                   self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450 // 2, 450 // 2 + 0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(img_, pts,
                                                   center, scale, 256,
                                                   100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), \
                "Landmarks out of boundary!"
            img_ = new_image
            landmarks_ = new_landmarks
        img_ = img_.transpose(2, 0, 1)

        heatmap = np.zeros((98, 64, 64))
        for i in range(98):
            if landmarks_[i][0] > 0:
                # print(pts[i])
                heatmap[i] = draw_gaussian(heatmap[i], landmarks_[i] / 4.0 + 1, 1)
        heatmap = torch.tensor(heatmap)
        return img_, landmarks_, heatmap


if __name__ == '__main__':
    # 300W
    # images_path = "../utils/300W/images/Fullset_test_images/"
    # train_list = "../utils/300W/landmarks/list_68pt_train.txt"
    test_list = "../utils/300W/landmarks/list_68pt_Fullset.txt"
    # 98
    # images_path = "../utils/WFLW_train/train_images/"
    # train_list = "../../WFLW_train/landmarks/list_98pt_train.txt"
    # # test_list = "../../WFLW_train/landmarks/list_98pt_test.txt"
    # test_list = "../utils/WFLW_train/landmarks/list_98pt_train.txt"
    # # 34
    train_list = '../utils/mydataset34/landmarks/train34_2.txt'
    images_path = '../utils/mydataset34/train_images/a/'
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

    dataset = MyLoadLabels34(ops=args, img_size=args.img_size, model="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            pin_memory=False, drop_last=True)
    ans = 0
    for i in range(len(dataset)):
        # LoadLabels
        images, landmarks_, heatmap = dataset.function(i)
        images = images.transpose(1, 2, 0)
        # for j in range(len(landmarks_)//2):
        #     images = cv2.circle(images, (int(landmarks_[j*2]), int(landmarks_[j*2+1])), 2, (255, 0, 255), -1)
        # cv2.imshow("dd", images)
        # cv2.waitKey(-1)

        # for j in range(len(landmarks_)):
        #     images = cv2.circle(images, (int(landmarks_[j][0]*256), int(landmarks_[j][1]*256)), 2, (255, 0, 255), -1)
        # cv2.imshow("dd", images)
        # cv2.waitKey(-1)
        #
        heatmap = torch.unsqueeze(heatmap, 0)
        #
        # # heatmap = np.array(heatmap)
        # pre_, _ = get_final_preds(heatmap)
        pre_, _ =get_preds_fromhm(heatmap)
        # images = cv2.circle(images, (int(pre_[0][32][0] * 4), int(pre_[0][32][1] * 4)), 2, (0, 0, 255), -1)
        images = cv2.resize(images, (512, 512))
        for i in range(34):
            images = cv2.circle(images, (int(pre_[0][i][0]*8), int(pre_[0][i][1]*8)), 2, (0, 0, 255), -1)
            images = cv2.putText(images, str(i), (int(pre_[0][i][0]*8), int(pre_[0][i][1]*8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.imshow("dd", images)
        cv2.waitKey(-1)
    #     res = 0
    #     for j in range(len(landmarks_)//2):
    #         x1 = landmarks_[j*2]
    #         y1 = landmarks_[j*2+1]
    #         x2 = int(pre_[0][j][0] * 2)
    #         y2 = int(pre_[0][j][1] * 2)
    #         res += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    #     res /= 34
    #     ans += res
    #     if i % 100 == 0:
    #         print("已完成：", i)
    #     if i == 400:
    #         break
    # print(ans / 400)


