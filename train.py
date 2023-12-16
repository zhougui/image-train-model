# 模型训练

import argparse
import sys
from torch.utils.data import DataLoader
from utils.model_utils import *
from utils.common_utils import *
from data_iter.datasets import *
from data_iter.mydatasets import *
from model.MHRB98net import MultipleHighResolutionBlockNet98
from model.MHRBnet import MultipleHighResolutionBlockNet
from loss.loss import *
import time
import random


def trainer(ops, f_log):
    # 使用指定的GPU及GPU显存
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    # 设置随机种子
    set_seed(ops.seed)
    print('use model : %s' % ops.model)
    # model_ = UHRnet(num_joints=98)
    # model_ = UNet()
    # model_ = mobilenet_v3_large(num_classes=196)
    # model_ = resnet50(pretrained=ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],
    #                   dropout_factor=ops.dropout)
    #model_ = HighResolutionNet(num_joints=98)
    model_ = MultipleHighResolutionBlockNet(base_channel=64, num_joints=32)
    #model_ = MultipleHighResolutionBlockNet98(num_joints=32)
    # cuda 是否可用
    use_cuda = torch.cuda.is_available()
    # GPU/CPU选择
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # 将模型放入GPU/CPU
    model = model_.to(device)
    # print(model_)# 打印模型结构
    dataset = LoadLabels32(ops=ops, img_size=ops.img_size, model="train", flag_agu=ops.flag_agu, fix_res=ops.fix_res, vis=False)
    dataloader = DataLoader(dataset, batch_size=ops.batch_size, num_workers=ops.num_workers, shuffle=True,
                            pin_memory=False, drop_last=True)
    print("wflw done")
    # 优化器设计
    # optimizer_SGD = optim.SGD(model_.parameters(), lr=ops.init_lr, momentum=ops.momentum,
    #                           weight_decay=ops.weight_decay)  # 优化器初始化
    optimizer_Adam = optim.AdamW(model_.parameters(), lr=ops.init_lr, weight_decay=ops.weight_decay, eps=1E-08,
                                 amsgrad=False)
    optimizer = optimizer_Adam
    # 加载 finetune 模型
    if os.access(ops.fintune_model, os.F_OK):  # checkpoint
        chkpt = torch.load(ops.fintune_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load fintune model : {}'.format(ops.fintune_model))
    print('/**********************************************/')
    # 损失函数
    criterion = nn.MSELoss(reduce=True, reduction='mean')
    # criterion = nn.BCELoss()
    step = 0
    # 变量初始化
    best_loss = np.inf
    loss_mean = 0.  # 损失均值
    loss_idx = 0.  # 损失计算计数器
    flag_change_lr_cnt = 0  # 学习率更新计数器
    init_lr = ops.init_lr  # 学习率
    acc = 0

    for epoch in range(0, ops.epochs):
        # 是否保存训练
        if ops.log_flag:
            sys.stdout = f_log
        print('\n epoch %d -------------------------------------------------------------------------------->>>' % epoch)
        model.train()
        loss_mean = 0.  # 损失均值
        loss_idx = 0.  # 损失计算计数器

        # LoadImagesAndLabels
        # for i, (imgs_, info) in enumerate(dataloader):
        #     # print('imgs_, pts_',imgs_.size(), pts_.size())
        #     if info is None:
        #         continue
        #     heat_map = info["heatmap"]
        #     # landmark = info["landmark"]

        # LoadLabels
        for i, (images, landmarks_, heatmap) in enumerate(dataloader):
            if use_cuda:
                images = images.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                heatmap = heatmap.cuda()
                # landmarks_ = landmarks_.cuda()
            output = model_(images.float())
            # if ops.loss_define == 'wing_loss':
            #     # 可以针对人脸部分的效果调损失权重，注意 关键点 id 映射关系 *2 ，因为涉及坐标（x，y）
            #     loss = got_total_wing_loss(output, landmarks_.float())
            # else:
            # loss = criterion(output, heatmap.float())
            loss = my_loss(output, heatmap.float())
            loss_mean += loss.item()
            loss_idx += 1.

            if i % 10 == 0:
                loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('  %s - %s - epoch [%s/%s] (%s/%s):' % (
                    loc_time, ops.model, epoch, ops.epochs, i, int(dataset.__len__() / ops.batch_size)),
                      'Mean Loss : %.6f - Loss: %.6f' % (loss_mean / loss_idx, loss.item()),
                      ' lr : %.7f' % init_lr,
                      ' bs :', ops.batch_size, ' img_size: %s x %s' % (ops.img_size[0], ops.img_size[1]),
                      ' best_loss: %.6f' % best_loss)
            # 计算梯度
            loss.backward()
            # 优化器对模型参数更新
            optimizer.step()
            # 优化器梯度清零
            optimizer.zero_grad()
            step += 1
        # 学习率更新策略
        if loss_mean != 0.:
            if best_loss > (loss_mean / loss_idx):
                flag_change_lr_cnt = 0
                best_loss = (loss_mean / loss_idx)
                if epoch > 300:
                    torch.save(model_.state_dict(), ops.model_exp + '{}-epoch-{}.pth'.format(ops.model, epoch))
            else:
                flag_change_lr_cnt += 1
                if flag_change_lr_cnt > 20:
                    init_lr = init_lr * ops.lr_decay
                    set_learning_rate(optimizer, init_lr)
                    flag_change_lr_cnt = 0
        w = open('save_loss.txt', 'a+')
        w.write(str(loss_mean / loss_idx) + " 学习率：" + str(init_lr) + "\n")
        w.close()
        if epoch % 10 == 0 and epoch > 0:
                torch.save(model_.state_dict(), ops.model_exp + '{}-epoch-{}-{}.pth'.format(ops.model, epoch, acc))
        set_seed(random.randint(0, 65535))


if __name__ == "__main__":
    # 68
    # images_path = "../utils/300W/"
    # train_list = "../utils/300W/list_68pt_train.txt"
    # 32
    images_path = "D:\PostgraduateInformation\KeyPointCode/utils/mydataset34/train_images/b/"
    train_list = "D:\PostgraduateInformation\KeyPointCode/utils/mydataset34/landmarks/train34_1.txt"
    # 98
    # images_path = "D:/PostgraduateInformation/KeyPointCode/utils/WFLW_train/train_images/"
    # train_list = "D:/PostgraduateInformation/KeyPointCode/utils/WFLW_train/landmarks/list_98pt_train.txt"
    # test_list = "D:/PostgraduateInformation/KeyPointCode/utils/WFLW_train/landmarks/list_98pt_test.txt"
    parser = argparse.ArgumentParser(description=' Project facial landmarks Train')
    parser.add_argument('--seed', type=int, default=32, help='seed')  # 设置随机种子
    parser.add_argument('--model_exp', type=str, default='./model_exp', help='model_exp')  # 模型输出文件夹
    parser.add_argument('--model', type=str, default='MHRB-net', help='model : Res-net')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=64, help='num_classes')  # landmarks 个数*2
    parser.add_argument('--GPUS', type=str, default='0', help='GPUS')  # GPU选择
    parser.add_argument('--images_path', type=str, default=images_path, help='images_path')  # 图片路径
    parser.add_argument('--train_list', type=str, default=train_list, help='annotations_train_list')  # 训练集标注信息
    #parser.add_argument('--test_list', type=str, default=test_list, help='annotations_train_list')  # 训练集标注信息
    parser.add_argument('--pretrained', type=bool, default=True, help='imageNet_Pretrain')  # 预训练
    parser.add_argument('--fintune_model', type=str, default='./model_exp/2021-02-21_17-51-30/resnet_50-epoch-724.pth',
                        help='fintune_model')  # 预训练模型
    parser.add_argument('--loss_define', type=str, default='wing_loss', help='define_loss')  # 损失函数定义
    parser.add_argument('--init_lr', type=float, default=1e-3, help='init_learningRate')  # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learningRate_decay')  # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight_decay')  # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 优化器动量
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')  # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')  # dropout
    parser.add_argument('--epochs', type=int, default=100, help='epochs')  # 训练周期
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')  # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=True, help='data_augmentation')  # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool, default=False, help='fix_resolution')  # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default=False, help='clear_model_exp')  # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default=False, help='log flag')  # 是否保存训练 log

    args = parser.parse_args()  # 解析添加参数

    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time) + '/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    f_log = None
    if args.log_flag:
        f_log = open(args.model_exp + '/train_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)), 'a+')
        sys.stdout = f_log

    print('---------------------------------- log : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", loc_time)))
    print('\n/******************* {} ******************/\n'.format(parser.description))
    unparsed = vars(args)  # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))
    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)
    trainer(ops=args, f_log=f_log)  # 模型训练
    if args.log_flag:
        sys.stdout = f_log
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

