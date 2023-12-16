import matplotlib
import math
import torch
import argparse
import numpy as np
import time
import os
#from model.MHRB98net import HighResolutionNet
from tensorboardX import SummaryWriter
from data_iter.datasets import *
from data_iter.mydatasets import *
# from model.UHRnetv1 import MHRBnet
from model.MHRnet import MultipleHighResolutionBlockNet
import cv2
matplotlib.use('Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fan_NME(pred_heatmaps, gt_landmarks, num_landmarks=26):
    nme = 0
    pred_landmarks, _ = get_preds_fromhm(pred_heatmaps)
    gt_landmarks = gt_landmarks.numpy()
    for i in range(pred_landmarks.shape[0]):
        pred_landmark = pred_landmarks[i] * 4.0
        gt_landmark = gt_landmarks[i]
        if num_landmarks == 68:
            left_eye = np.average(gt_landmark[36:42], axis=0)
            right_eye = np.average(gt_landmark[42:48], axis=0)
            norm_factor = np.linalg.norm(left_eye - right_eye)
        elif num_landmarks == 98:
            norm_factor = np.linalg.norm(gt_landmark[60] - gt_landmark[72])
        elif num_landmarks == 26:
            norm_factor = np.linalg.norm(gt_landmark[24] - gt_landmark[25])
        nme += (np.sum(np.linalg.norm(pred_landmark - gt_landmark, axis=1)) / pred_landmark.shape[0]) / norm_factor
    return nme


def eval_model(model, dataloaders, dataset_sizes, use_gpu=True, epoches=1, save_path='./', num_landmarks=98):
    global_nme = 0
    model.eval()
    for epoch in range(epoches):
        running_loss = 0
        step = 0
        total_nme = 0
        total_count = 0
        fail_count = 0
        nmes = []
        index = 0
        # running_corrects = 0

        # Iterate over data.
        with torch.no_grad():
            for i, (images, landmarks_, heatmap) in enumerate(dataloaders):
                total_runtime = 0
                run_count = 0
                step_start = time.time()
                step += 1
                # wrap them in Variable
                if use_gpu:
                    images = images.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    heatmap = heatmap.cuda()
                    # landmarks_ = landmarks_.cuda()
                single_start = time.time()
                outputs = model(images.float())
                single_end = time.time()
                total_runtime += time.time() - single_start
                run_count += 1
                step_end = time.time()
                for i in range(images.shape[0]):
                    # pred_heatmap = outputs[-1][i].detach().cpu()[:-1, :, :]
                    pred_heatmap = outputs[i].detach().cpu()
                    pred_landmarks, _ = get_final_preds(pred_heatmap.unsqueeze(0))
                    pred_landmarks = pred_landmarks.squeeze().numpy()
                    # print("data['landmarks'][i]:",type(data['landmarks'][i]))
                    # print("landmarks'][i].shape", data['landmarks'][i].shape)
                    gt_landmarks = landmarks_

                    if num_landmarks == 68:
                        left_eye = np.average(gt_landmarks[36:42], axis=0)
                        right_eye = np.average(gt_landmarks[42:48], axis=0)
                        norm_factor = np.linalg.norm(left_eye - right_eye)
                        # norm_factor = np.linalg.norm(gt_landmarks[36]- gt_landmarks[45])

                    elif num_landmarks == 98:
                        norm_factor = np.linalg.norm(gt_landmarks[60] - gt_landmarks[72])
                    elif num_landmarks == 19:
                        left, top = gt_landmarks[-2, :]
                        right, bottom = gt_landmarks[-1, :]
                        norm_factor = math.sqrt(abs(right - left)*abs(top-bottom))
                        gt_landmarks = gt_landmarks[:-2, :]
                    elif num_landmarks == 29:
                        # norm_factor = np.linalg.norm(gt_landmarks[8]- gt_landmarks[9])
                        norm_factor = np.linalg.norm(gt_landmarks[16]- gt_landmarks[17])
                    single_nme = (np.sum(np.linalg.norm(pred_landmarks*4 - gt_landmarks, axis=1)) / pred_landmarks.shape[0]) / norm_factor

                    nmes.append(single_nme)
                    total_count += 1
                    if single_nme > 0.1:
                        fail_count += 1
                if step % 10 == 0:
                    print('Step {} Time: {:.6f} Input Mean: {:.6f} Output Mean: {:.6f}'.format(
                        step, step_end - step_start,
                        torch.mean(images),
                        torch.mean(outputs[0])))
                # gt_landmarks = landmarks.numpy()
                # pred_heatmap = outputs[-1].to('cpu').numpy()
                gt_landmarks = landmarks_
                index += 1
                batch_nme = fan_NME(outputs[-1][:, :-1, :, :].detach().cpu(), gt_landmarks, num_landmarks)
                # batch_nme = 0
                total_nme += batch_nme
        epoch_nme = total_nme / dataset_sizes['val']
        print("dataset_sizes['val']:", dataset_sizes['val'])
        global_nme += epoch_nme
        nme_save_path = os.path.join(save_path, 'nme_log.npy')
        np.save(nme_save_path, np.array(nmes))
        print('NME: {:.6f} Failure Rate: {:.6f} Total Count: {:.6f} Fail Count: {:.6f}'.format(epoch_nme, fail_count/total_count, total_count, fail_count))
    print('Evaluation done! Average NME: {:.6f}'.format(global_nme/epoches))
    print('Everage runtime for a single batch: {:.6f}'.format(total_runtime/run_count))
    return model


def CumulativeErrorDistribution(nmes):
    nmes.sort()
    j = 0
    proportion = []
    for i in range(0, 25):
        while j < len(nmes):
            if nmes[j] < i/100:
                break
            else:
                j += 1
        proportion[i] = j/len(nmes)


def heatmap_getmne(model, dataloader, dataset_sizes=2500, use_gpu=True, epoches=1, num_landmarks=98):
    global_nme = 0
    model.eval()
    for epoch in range(epoches):
        running_loss = 0
        step = 0
        total_nme = 0
        total_count = 0
        fail_count = 0
        nmes = []
        index = 0
        # running_corrects = 0
        res = 0
        # Iterate over data.
        with torch.no_grad():
            for i, (images, landmarks_, heatmap) in enumerate(dataloader):
                image = images
                total_runtime = 0
                run_count = 0
                step_start = time.time()
                step += 1
                # get the inputs
                # wrap them in Variable
                if use_gpu:
                    images = images.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    # heatmap = heatmap.cuda()
                    # landmarks_ = landmarks_.cuda()
                single_start = time.time()
                outputs = model(images.float())
                single_end = time.time()
                total_runtime += time.time() - single_start
                run_count += 1
                step_end = time.time()
                for i in range(images.shape[0]):
                    pred_heatmap = outputs[i].detach().cpu()
                    pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
                    pred_landmarks = pred_landmarks.squeeze()
                    gt_landmarks = landmarks_[i]
                    # gt_landmarks = gt_landmarks.detach().numpy()
                    # output = np.squeeze(output)

                    if num_landmarks == 68:
                        left_eye = np.average(gt_landmarks[36:42], axis=0)
                        right_eye = np.average(gt_landmarks[42:48], axis=0)
                        norm_factor = np.linalg.norm(left_eye - right_eye)
                    elif num_landmarks == 98:
                        norm_factor = np.linalg.norm(gt_landmarks[60] - gt_landmarks[72])
                    elif num_landmarks == 19:
                        left, top = gt_landmarks[-2, :]
                        right, bottom = gt_landmarks[-1, :]
                        norm_factor = math.sqrt(abs(right - left)*abs(top-bottom))
                        gt_landmarks = gt_landmarks[:-2, :]
                    elif num_landmarks == 29:
                        norm_factor = np.linalg.norm(gt_landmarks[16] - gt_landmarks[17])
                    pred_landmarks = torch.Tensor(pred_landmarks)

                    # img = image[i].numpy()
                    # img = img.transpose(1, 2, 0)
                    # cv2.imshow("dd", img)
                    # cv2.waitKey(-1)
                    # for j in range(98):
                    #     img = cv2.circle(img, (int(float(pred_landmarks[j][0]*4)), int(float(pred_landmarks[j][1] * 4))), 2,
                    #                      (255, 0, 255), -1)
                    # cv2.imshow("dd", img)
                    # cv2.waitKey(-1)
                    single_nme = (np.sum(np.linalg.norm(pred_landmarks*4 - gt_landmarks, axis=1)) / pred_landmarks.shape[0]) / norm_factor
                    res += single_nme
                    nmes.append(single_nme)
                    total_count += 1
                    if single_nme > 0.1:
                        fail_count += 1
                if step % 10 == 0:
                    print('Step {} Time: {:.6f} Input Mean: {:.6f} Output Mean: {:.6f}'.format(
                        step, step_end - step_start,
                        torch.mean(heatmap[0]),
                        torch.mean(outputs[0])))
                # gt_landmarks = landmarks.numpy()
                # pred_heatmap = outputs[-1].to('cpu').numpy()
                gt_landmarks = landmarks_
                index += 1
                batch_nme = fan_NME(outputs.detach().cpu(), gt_landmarks, num_landmarks)
                print("batch_nme", batch_nme)
                # batch_nme = 0
                total_nme += batch_nme
        epoch_nme = total_nme / dataset_sizes
        global_nme += epoch_nme
        # nme_save_path = os.path.join(save_path, 'nme_log.npy')
        # np.save(nme_save_path, np.array(nmes))
        print('NME: {:.6f} Failure Rate: {:.6f} Total Count: {:.6f} Fail Count: {:.6f}'.format(res / dataset_sizes, fail_count/total_count, total_count, fail_count))
    # print('Evaluation done! Average NME: {:.6f}'.format(global_nme/epoches))
    print('Evaluation done! Average NME: {:.6f}'.format(global_nme / epoches))
    print('Everage runtime for a single batch: {:.6f}'.format(total_runtime/run_count))
    return model


def regression_getmne(model, dataloader, dataset_sizes=2500, use_gpu=True, epoches=1, num_landmarks=98):
    model.eval()
    for epoch in range(epoches):
        step = 0
        total_count = 0
        fail_count = 0
        nmes = []
        index = 0
        res = 0
        batch_nme = 0
        # Iterate over data.
        with torch.no_grad():
            for i, (images, landmarks_, heatmap) in enumerate(dataloader):
                total_runtime = 0
                run_count = 0
                step_start = time.time()
                step += 1
                # get the inputs
                # wrap them in Variable
                if use_gpu:
                    images = images.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    # heatmap = heatmap.cuda()
                    # landmarks_ = landmarks_.cuda()
                single_start = time.time()
                outputs = model(images.float())
                single_end = time.time()
                total_runtime += time.time() - single_start
                run_count += 1
                step_end = time.time()
                for i in range(images.shape[0]):
                    pred_landmarks = outputs[i].detach().cpu()
                    pts = []
                    for j in range(len(pred_landmarks)//2):
                        pts.append([pred_landmarks[j*2]*256, pred_landmarks[j*2+1]*256])
                    gt_landmarks = landmarks_[i]
                    pts = torch.tensor(pts)
                    # print(pts.shape)
                    # print(gt_landmarks.shape)

                    if num_landmarks == 68:
                        left_eye = np.average(gt_landmarks[36:42], axis=0)
                        right_eye = np.average(gt_landmarks[42:48], axis=0)
                        norm_factor = np.linalg.norm(left_eye - right_eye)
                        # norm_factor = np.linalg.norm(gt_landmarks[36]- gt_landmarks[45])

                    elif num_landmarks == 98:
                        norm_factor = np.linalg.norm(gt_landmarks[60] - gt_landmarks[72])
                    elif num_landmarks == 19:
                        left, top = gt_landmarks[-2, :]
                        right, bottom = gt_landmarks[-1, :]
                        norm_factor = math.sqrt(abs(right - left)*abs(top-bottom))
                        gt_landmarks = gt_landmarks[:-2, :]
                    elif num_landmarks == 29:
                        # norm_factor = np.linalg.norm(gt_landmarks[8]- gt_landmarks[9])
                        norm_factor = np.linalg.norm(gt_landmarks[16] - gt_landmarks[17])

                    pred_landmarks = torch.Tensor(pred_landmarks)
                    # print("1:", pred_landmarks*4)
                    # print("2:", gt_landmarks)
                    # return
                    single_nme = (np.sum(np.linalg.norm(pts - gt_landmarks, axis=1)) / pred_landmarks.shape[0]) / norm_factor
                    batch_nme += single_nme
                    res += single_nme
                    nmes.append(single_nme)
                    total_count += 1
                    if single_nme > 0.1:
                        fail_count += 1
                if step % 10 == 0:
                    print('Step {} Time: {:.6f} Input Mean: {:.6f} Output Mean: {:.6f}'.format(
                        step, step_end - step_start,
                        torch.mean(heatmap[0]),
                        torch.mean(outputs[0])))
                index += 1
                print("batch_nme", batch_nme)
                batch_nme = 0
        print('NME: {:.6f} Failure Rate: {:.6f} Total Count: {:.6f} Fail Count: {:.6f}'.format(res / dataset_sizes, fail_count/total_count, total_count, fail_count))
    print('Everage runtime for a single batch: {:.6f}'.format(total_runtime/run_count))
    return model


if __name__ == '__main__':

    # 98
    # images_path = "../utils/WFLW_train/train_images/"
    images_path = "D:/PostgraduateInformation/KeyPointCode/utils/WFLW_train/train_images/"
    train_list = "D:/PostgraduateInformation/KeyPointCode/utils/WFLW_train/landmarks/list_98pt_train.txt"
    test_list = "D:/PostgraduateInformation/KeyPointCode/utils/WFLW_train/landmarks/list_98pt_test.txt"
    # 68
    # images_path = "../utils/300W/images/Fullset_test_images/"
    # train_list = "../../300W/landmarks/list_68pt_train.txt"
    # test_list = "../utils/300W/landmarks/list_68pt_Fullset.txt"

    # Parse arguments
    parser = argparse.ArgumentParser()
    # Dataset paths
    parser.add_argument('--images_path', type=str, default=images_path, help='images_path')  # 图片路径
    parser.add_argument('--test_list', type=str, default=test_list, help='annotations_train_list')  # 训练集标注信息
    parser.add_argument('--train_list', type=str, default=train_list, help='annotations_train_list')  # 训练集标注信息
    parser.add_argument('--num_landmarks', type=int, default=98,
                        help='Number of landmarks')

    # Checkpoint and pretrained weights
    parser.add_argument('--ckpt_save_path', type=str, default="./save/ckpt_save/",
                        help='a directory to save checkpoint file')
    parser.add_argument('--pretrained_weights', type=str, default="../MHRB-net-epoch-193.pth",
                        help='a directory to save pretrained_weights')
    # parser.add_argument('--pretrained_weights', type=str, default="../HR-net98-epoch-300.pth",
    #                     help='a directory to save pretrained_weights')
    # Eval options
    parser.add_argument('--batch_size', type=int, default=2,
                        help='learning rate decay after each epoch')

    # Network parameters
    parser.add_argument('--hg_blocks', type=int, default=4,
                        help='Number of HG blocks to stack')
    parser.add_argument('--gray_scale', type=str, default="False",
                        help='Whether to convert RGB image into gray scale during training')
    parser.add_argument('--end_relu', type=str, default="False",
                        help='Whether to add relu at the end of each HG module')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=True, help='data_augmentation')  # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool, default=False, help='fix_resolution')  # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')  # 训练数据生成器线程数

    args = parser.parse_args()

    PRETRAINED_WEIGHTS = args.pretrained_weights

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = LoadLabels98(ops=args, img_size=args.img_size, model="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                            pin_memory=False, drop_last=True)
    use_gpu = torch.cuda.is_available()
    # model_ft = HighResolutionNet(num_joints=98)
    # model_ft = HighResolutionNetv3(num_joints=98)
    # model_ft = mobilenet_v3_large(num_classes=136)
    model_ft = MultipleHighResolutionBlockNet(base_channel=64, num_joints=98)
    # model_ft = PHighResolutionNet(base_channel=128, num_joints=98)
    # model_ft = resnet50(num_classes=136, img_size=256)
    if PRETRAINED_WEIGHTS != "None":
        checkpoint = torch.load(PRETRAINED_WEIGHTS)
        if 'state_dict' not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights, strict=False)

    model_ft = model_ft.to(device)
    model_ft = heatmap_getmne(model_ft, dataloader, len(dataset), num_landmarks=98)

    # model_ft = regression_getmne(model_ft, dataloader, len(dataset), num_landmarks=68)


# if __name__ == '__main__':
#     getnms()
